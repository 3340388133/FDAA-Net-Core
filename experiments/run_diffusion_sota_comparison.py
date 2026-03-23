#!/usr/bin/env python3
"""
扩散模型检测方法 SOTA 对比实验

专门对比针对扩散模型生成图像的检测方法:
- DIRE (ICCV 2023) - 扩散重建误差
- LaRE² (CVPR 2024) - 潜在空间重建误差
- DRCT (ICML 2024) - 扩散重建对比训练
- C2P-CLIP (AAAI 2025) - 类别共同提示CLIP
- Ours (FDAA + MGFP) - 本文方法

同时包含经典baseline:
- CNNDetection (CVPR 2020)
- F3-Net (ECCV 2020)
- UnivFD (CVPR 2023)
- FreqNet (AAAI 2024)

Usage:
    python experiments/run_diffusion_sota_comparison.py --epochs 10
"""

import os
import sys
import json
import argparse
import time
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import numpy as np
from tqdm import tqdm
from PIL import Image

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.sota_methods import create_sota_model, list_sota_methods, list_diffusion_detection_methods
from models.detector import AIGCDetectorLite


# ============================================================================
# 数据集
# ============================================================================

class GenImageDataset(Dataset):
    """GenImage 数据集加载器"""

    def __init__(self, data_root, split='train', transform=None,
                 max_samples=None, apply_jpeg=False, jpeg_quality=75):
        self.data_root = Path(data_root)
        self.transform = transform
        self.apply_jpeg = apply_jpeg
        self.jpeg_quality = jpeg_quality
        self.samples = []

        split_dir = self.data_root / split

        # 查找 AI 和真实图像目录
        ai_dirs = [split_dir / 'ai', split_dir / 'fake', split_dir / '1']
        nature_dirs = [split_dir / 'nature', split_dir / 'real', split_dir / '0']

        ai_dir = next((d for d in ai_dirs if d.exists()), None)
        nature_dir = next((d for d in nature_dirs if d.exists()), None)

        if ai_dir is None or nature_dir is None:
            raise ValueError(f"数据目录不完整: {data_root}/{split}")

        # 收集图像
        exts = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        ai_images = []
        nature_images = []

        for ext in exts:
            ai_images.extend(list(ai_dir.glob(ext)))
            nature_images.extend(list(nature_dir.glob(ext)))

        random.shuffle(ai_images)
        random.shuffle(nature_images)

        # 平衡采样
        if max_samples:
            n_per_class = max_samples // 2
            ai_images = ai_images[:n_per_class]
            nature_images = nature_images[:n_per_class]

        for img_path in ai_images:
            self.samples.append((img_path, 1))
        for img_path in nature_images:
            self.samples.append((img_path, 0))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.apply_jpeg:
            import io
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=self.jpeg_quality)
            buffer.seek(0)
            image = Image.open(buffer).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': label}


def get_transforms(img_size=224):
    """获取数据变换"""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


# ============================================================================
# 训练和评估
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, desc="Training"):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc=desc, leave=False)
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(images)

        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs

        loss = criterion(logits, labels)
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.5
    return avg_loss, auc


def evaluate(model, dataloader, device, desc="Evaluating"):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    try:
        auc = roc_auc_score(all_labels, all_preds)
        ap = average_precision_score(all_labels, all_preds)
    except ValueError:
        auc = 0.5
        ap = 0.5
    pred_labels = (np.array(all_preds) > 0.5).astype(int)
    acc = accuracy_score(all_labels, pred_labels)

    return {'auc': auc, 'ap': ap, 'accuracy': acc}


def create_model(method_name, device):
    """创建模型"""
    if method_name == 'ours':
        model = AIGCDetectorLite(
            num_classes=2,
            img_size=224,
            embed_dim=768,
            num_prototypes=4,
            dropout=0.1
        )
    else:
        model = create_sota_model(method_name)
    return model.to(device)


# ============================================================================
# 主实验
# ============================================================================

def run_experiment(method_name, train_loader, test_loaders, args, device):
    """运行单个方法的实验"""
    print(f"\n{'='*70}")
    print(f"方法: {method_name.upper()}")
    print(f"{'='*70}")

    # 创建模型
    try:
        model = create_model(method_name, device)
    except Exception as e:
        print(f"创建模型失败: {e}")
        return None

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: {total_params/1e6:.2f}M (可训练: {trainable_params/1e6:.2f}M)")

    # 训练
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_time = time.time()
    best_val_auc = 0

    for epoch in range(args.epochs):
        train_loss, train_auc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            desc=f"{method_name} Epoch {epoch+1}/{args.epochs}"
        )
        scheduler.step()

        # 在训练集对应的验证集上评估
        if 'train_dataset_val' in test_loaders:
            val_metrics = evaluate(model, test_loaders['train_dataset_val'], device)
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
            print(f"  Epoch {epoch+1}: Loss={train_loss:.4f}, Train AUC={train_auc:.4f}, Val AUC={val_metrics['auc']:.4f}")
        else:
            print(f"  Epoch {epoch+1}: Loss={train_loss:.4f}, Train AUC={train_auc:.4f}")

    train_time = (time.time() - start_time) / 60
    print(f"训练耗时: {train_time:.1f} min")

    # 在所有测试集上评估
    results = {
        'method': method_name,
        'params_M': total_params / 1e6,
        'train_time_min': train_time,
        'test_results': {}
    }

    for test_name, test_loader in test_loaders.items():
        if test_name == 'train_dataset_val':
            continue
        metrics = evaluate(model, test_loader, device, desc=f"测试 {test_name}")
        results['test_results'][test_name] = metrics
        print(f"  {test_name}: AUC={metrics['auc']:.4f}, AP={metrics['ap']:.4f}, Acc={metrics['accuracy']:.4f}")

    # 清理
    del model
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description='扩散模型检测SOTA对比实验')

    # 数据集配置
    parser.add_argument('--train_dataset', type=str,
                        default='datasets/genimage_partial/imagenet_ai_0419_biggan',
                        help='训练数据集路径')
    parser.add_argument('--test_datasets', type=str, nargs='+',
                        default=[
                            'datasets/genimage_partial/imagenet_ai_0419_biggan',
                            'datasets/genimage_partial/imagenet_ai_0508_adm',
                        ],
                        help='测试数据集路径列表')
    parser.add_argument('--test_names', type=str, nargs='+',
                        default=['BigGAN', 'ADM'],
                        help='测试数据集名称')

    # 训练配置
    parser.add_argument('--train_samples', type=int, default=20000)
    parser.add_argument('--val_samples', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)

    # 方法选择
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['ours', 'dire', 'lare2', 'drct', 'c2pclip',
                                'cnndetection', 'f3net', 'univfd', 'freqnet'],
                        help='要测试的方法')
    parser.add_argument('--diffusion_only', action='store_true',
                        help='只测试扩散模型检测方法')

    # 输出配置
    parser.add_argument('--output_dir', type=str, default='outputs/diffusion_sota_comparison')

    args = parser.parse_args()

    # 设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 如果只测试扩散方法
    if args.diffusion_only:
        args.methods = ['ours', 'dire', 'lare2', 'drct', 'c2pclip']

    print(f"\n{'#'*70}")
    print("# 扩散模型生成图像检测 - SOTA方法对比实验")
    print(f"{'#'*70}")
    print(f"\n方法列表 ({len(args.methods)} 个):")
    all_methods = list_sota_methods()
    for m in args.methods:
        if m == 'ours':
            print(f"  - ours: 本文方法 (FDAA + MGFP)")
        elif m in all_methods:
            print(f"  - {m}: {all_methods[m]}")
        else:
            print(f"  - {m}: (未知方法)")

    # 准备数据
    train_transform, val_transform = get_transforms()

    # 训练数据
    print(f"\n加载训练数据: {args.train_dataset}")
    train_path = Path(args.train_dataset)
    if not train_path.exists():
        print(f"错误: 训练数据集不存在: {args.train_dataset}")
        sys.exit(1)

    train_dataset = GenImageDataset(
        data_root=args.train_dataset,
        split='train',
        transform=train_transform,
        max_samples=args.train_samples
    )
    print(f"训练样本: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )

    # 测试数据
    test_loaders = {}

    # 训练集的验证集
    train_val_dataset = GenImageDataset(
        data_root=args.train_dataset,
        split='val',
        transform=val_transform,
        max_samples=args.val_samples
    )
    test_loaders['train_dataset_val'] = DataLoader(
        train_val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )

    # 其他测试集
    for test_path, test_name in zip(args.test_datasets, args.test_names):
        test_path = Path(test_path)
        if not test_path.exists():
            print(f"警告: 测试数据集不存在: {test_path}")
            continue

        try:
            # 原始测试
            test_dataset = GenImageDataset(
                data_root=str(test_path),
                split='val',
                transform=val_transform,
                max_samples=args.val_samples
            )
            test_loaders[f'{test_name}'] = DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers
            )
            print(f"测试集 {test_name}: {len(test_dataset)} 样本")

            # JPEG压缩测试
            test_dataset_jpeg = GenImageDataset(
                data_root=str(test_path),
                split='val',
                transform=val_transform,
                max_samples=args.val_samples,
                apply_jpeg=True,
                jpeg_quality=75
            )
            test_loaders[f'{test_name}_JPEG75'] = DataLoader(
                test_dataset_jpeg, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers
            )
        except Exception as e:
            print(f"加载测试集 {test_name} 失败: {e}")

    # 运行实验
    all_results = []

    for method in args.methods:
        try:
            result = run_experiment(method, train_loader, test_loaders, args, device)
            if result:
                all_results.append(result)

                # 保存中间结果
                with open(f"{args.output_dir}/result_{method}.json", 'w') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"方法 {method} 失败: {e}")
            import traceback
            traceback.print_exc()

    # ========================================================================
    # 生成结果报告
    # ========================================================================
    print("\n" + "="*90)
    print("实验结果汇总")
    print("="*90)

    # 按测试集整理结果
    test_set_names = [name for name in test_loaders.keys() if name != 'train_dataset_val']

    # 表1: 各方法在不同数据集上的AUC
    print("\n【表1】各方法AUC对比")
    print("-"*90)

    header = f"{'方法':<12}"
    for name in test_set_names:
        header += f" {name:>12}"
    header += f" {'平均':>10}"
    print(header)
    print("-"*90)

    method_avg_aucs = []
    for result in all_results:
        method = result['method']
        line = f"{method:<12}"
        aucs = []
        for name in test_set_names:
            if name in result['test_results']:
                auc = result['test_results'][name]['auc']
                aucs.append(auc)
                line += f" {auc:>12.4f}"
            else:
                line += f" {'N/A':>12}"
        avg_auc = np.mean(aucs) if aucs else 0
        method_avg_aucs.append((method, avg_auc))
        line += f" {avg_auc:>10.4f}"
        print(line)

    # 排名
    print("\n【排名】按平均AUC排序")
    print("-"*50)
    method_avg_aucs.sort(key=lambda x: x[1], reverse=True)
    for rank, (method, avg_auc) in enumerate(method_avg_aucs, 1):
        marker = " ★" if method == 'ours' else ""
        print(f"{rank}. {method:<12} {avg_auc:.4f}{marker}")

    # 保存完整结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_output = {
        'config': vars(args),
        'timestamp': timestamp,
        'results': all_results,
        'ranking': [{'rank': i+1, 'method': m, 'avg_auc': a}
                   for i, (m, a) in enumerate(method_avg_aucs)]
    }

    output_file = f"{args.output_dir}/final_comparison_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"\n完整结果已保存到: {output_file}")

    # 生成LaTeX表格
    generate_latex_table(all_results, test_set_names, args.output_dir, timestamp)


def generate_latex_table(results, test_set_names, output_dir, timestamp):
    """生成LaTeX表格"""
    latex_file = f"{output_dir}/comparison_table_{timestamp}.tex"

    with open(latex_file, 'w') as f:
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Comparison with State-of-the-Art Methods for Diffusion-Generated Image Detection}\n")
        f.write("\\label{tab:sota_comparison}\n")
        f.write("\\resizebox{\\textwidth}{!}{\n")

        # 计算列数
        n_cols = len(test_set_names) + 2  # 方法 + 测试集 + 平均
        col_spec = "l" + "c" * (n_cols - 1)
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
        f.write("\\toprule\n")

        # 表头
        header = "Method"
        for name in test_set_names:
            header += f" & {name}"
        header += " & Avg. \\\\\n"
        f.write(header)
        f.write("\\midrule\n")

        # 方法分组
        classic_methods = ['cnndetection', 'f3net', 'spec', 'gramnet']
        recent_methods = ['univfd', 'npr', 'freqnet', 'dire']
        diffusion_methods = ['lare2', 'drct', 'c2pclip']

        # 按类别输出
        method_display_names = {
            'cnndetection': 'CNNDetection~\\cite{wang2020cnn}',
            'f3net': 'F3-Net~\\cite{qian2020f3net}',
            'spec': 'Spec~\\cite{zhang2019spec}',
            'gramnet': 'GramNet~\\cite{liu2020gramnet}',
            'univfd': 'UnivFD~\\cite{ojha2023univfd}',
            'npr': 'NPR~\\cite{tan2024npr}',
            'freqnet': 'FreqNet~\\cite{tan2024freqnet}',
            'dire': 'DIRE~\\cite{wang2023dire}',
            'lare2': 'LaRE$^2$~\\cite{luo2024lare}',
            'drct': 'DRCT~\\cite{zhong2024drct}',
            'c2pclip': 'C2P-CLIP~\\cite{ye2025c2pclip}',
            'ours': '\\textbf{Ours (FDAA+MGFP)}'
        }

        def write_method_row(result, is_ours=False):
            method = result['method']
            display_name = method_display_names.get(method, method)

            row = display_name
            aucs = []
            for name in test_set_names:
                if name in result['test_results']:
                    auc = result['test_results'][name]['auc']
                    aucs.append(auc)
                    if is_ours:
                        row += f" & \\textbf{{{auc:.2f}}}"
                    else:
                        row += f" & {auc:.2f}"
                else:
                    row += " & -"

            avg_auc = np.mean(aucs) if aucs else 0
            if is_ours:
                row += f" & \\textbf{{{avg_auc:.2f}}}"
            else:
                row += f" & {avg_auc:.2f}"
            row += " \\\\\n"
            return row

        # 经典方法
        f.write("\\multicolumn{" + str(n_cols) + "}{l}{\\textit{Classic Methods (2019-2022)}} \\\\\n")
        for result in results:
            if result['method'] in classic_methods:
                f.write(write_method_row(result))

        f.write("\\midrule\n")

        # 近期方法
        f.write("\\multicolumn{" + str(n_cols) + "}{l}{\\textit{Recent Methods (2023-2024)}} \\\\\n")
        for result in results:
            if result['method'] in recent_methods:
                f.write(write_method_row(result))

        f.write("\\midrule\n")

        # 扩散模型专用方法
        f.write("\\multicolumn{" + str(n_cols) + "}{l}{\\textit{Diffusion-Specific Methods (2024-2025)}} \\\\\n")
        for result in results:
            if result['method'] in diffusion_methods:
                f.write(write_method_row(result))

        f.write("\\midrule\n")

        # 本文方法
        for result in results:
            if result['method'] == 'ours':
                f.write(write_method_row(result, is_ours=True))

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("}\n")
        f.write("\\end{table*}\n")

    print(f"LaTeX表格已保存到: {latex_file}")


if __name__ == '__main__':
    main()
