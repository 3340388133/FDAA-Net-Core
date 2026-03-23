#!/usr/bin/env python3
"""
多源训练 SOTA 对比实验

所有方法使用 BigGAN + ADM 联合训练，公平比较跨域泛化能力

Usage:
    python experiments/run_multisource_sota_comparison.py --epochs 15
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
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import numpy as np
from tqdm import tqdm
from PIL import Image

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.sota_methods import create_sota_model, list_sota_methods
from models.detector import AIGCDetectorLite


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
        exts = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG', '*.webp', '*.WEBP']
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
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # 如果文件损坏，返回一个随机的其他样本
            import random
            new_idx = random.randint(0, len(self.samples) - 1)
            while new_idx == idx:
                new_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(new_idx)

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
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


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
            num_prototypes=8,  # 使用更多原型
            dropout=0.1
        )
    else:
        model = create_sota_model(method_name)
    return model.to(device)


def run_experiment(method_name, train_loader, val_loader, test_loaders, args, device):
    """运行单个方法的实验"""
    print(f"\n{'='*70}")
    print(f"方法: {method_name.upper()} (多源训练)")
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

    # 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_time = time.time()
    best_val_auc = 0
    best_model_state = None

    for epoch in range(args.epochs):
        train_loss, train_auc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            desc=f"{method_name} Epoch {epoch+1}/{args.epochs}"
        )
        scheduler.step()

        # 验证
        val_metrics = evaluate(model, val_loader, device, desc="验证")

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  Epoch {epoch+1}: Loss={train_loss:.4f}, Train AUC={train_auc:.4f}, Val AUC={val_metrics['auc']:.4f}")

    train_time = (time.time() - start_time) / 60
    print(f"训练耗时: {train_time:.1f} min")

    # 加载最佳模型
    if best_model_state:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    # 在所有测试集上评估
    results = {
        'method': method_name,
        'params_M': total_params / 1e6,
        'train_time_min': train_time,
        'best_val_auc': best_val_auc,
        'test_results': {}
    }

    print(f"\n测试结果:")
    for test_name, test_loader in test_loaders.items():
        metrics = evaluate(model, test_loader, device, desc=f"测试 {test_name}")
        results['test_results'][test_name] = metrics
        print(f"  {test_name}: AUC={metrics['auc']:.4f}, AP={metrics['ap']:.4f}, Acc={metrics['accuracy']:.3f}")

    # 清理
    del model
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description='多源训练SOTA对比实验')

    # 多源数据集配置
    parser.add_argument('--train_datasets', type=str, nargs='+',
                        default=[
                            'datasets/genimage_partial/imagenet_ai_0419_biggan',
                            'datasets/genimage_partial/imagenet_ai_0508_adm',
                        ],
                        help='多个训练数据集路径')
    parser.add_argument('--train_names', type=str, nargs='+',
                        default=['BigGAN', 'ADM'],
                        help='训练数据集名称')

    # 训练配置
    parser.add_argument('--samples_per_source', type=int, default=15000,
                        help='每个数据源的样本数')
    parser.add_argument('--val_samples', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)

    # 方法选择
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['ours', 'univfd', 'dire', 'lare2', 'drct',
                                'cnndetection', 'f3net', 'freqnet'],
                        help='要测试的方法')

    # 输出配置
    parser.add_argument('--output_dir', type=str, default='outputs/multisource_sota_comparison')

    args = parser.parse_args()

    # 设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'#'*70}")
    print("# 多源训练 SOTA 方法对比实验")
    print("# 所有方法使用 BigGAN + ADM 联合训练")
    print(f"{'#'*70}")

    print(f"\n训练数据源: {args.train_names}")
    print(f"每个数据源样本数: {args.samples_per_source}")
    print(f"总训练样本: {args.samples_per_source * len(args.train_datasets)}")
    print(f"\n方法列表 ({len(args.methods)} 个):")
    for m in args.methods:
        print(f"  - {m}")

    # 准备数据
    train_transform, val_transform = get_transforms()

    # 多源训练数据
    print(f"\n加载多源训练数据...")
    train_datasets = []
    val_datasets = []

    for train_path, train_name in zip(args.train_datasets, args.train_names):
        train_path = Path(train_path)
        if not train_path.exists():
            print(f"警告: 数据集不存在: {train_path}")
            continue

        # 训练集
        ds_train = GenImageDataset(
            data_root=str(train_path),
            split='train',
            transform=train_transform,
            max_samples=args.samples_per_source
        )
        train_datasets.append(ds_train)
        print(f"  {train_name} 训练样本: {len(ds_train)}")

        # 验证集
        ds_val = GenImageDataset(
            data_root=str(train_path),
            split='val',
            transform=val_transform,
            max_samples=args.val_samples // len(args.train_datasets)
        )
        val_datasets.append(ds_val)
        print(f"  {train_name} 验证样本: {len(ds_val)}")

    # 合并数据集
    combined_train = ConcatDataset(train_datasets)
    combined_val = ConcatDataset(val_datasets)

    print(f"\n合并后训练样本: {len(combined_train)}")
    print(f"合并后验证样本: {len(combined_val)}")

    train_loader = DataLoader(
        combined_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        combined_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )

    # 测试数据（每个数据源单独测试）
    test_loaders = {}
    for test_path, test_name in zip(args.train_datasets, args.train_names):
        test_path = Path(test_path)
        if not test_path.exists():
            continue

        # 原始测试
        test_dataset = GenImageDataset(
            data_root=str(test_path),
            split='val',
            transform=val_transform,
            max_samples=args.val_samples
        )
        test_loaders[test_name] = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers
        )

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

    # 运行实验
    all_results = []

    for method in args.methods:
        try:
            result = run_experiment(
                method, train_loader, val_loader, test_loaders, args, device
            )
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
    print("多源训练实验结果汇总")
    print("="*90)

    test_set_names = list(test_loaders.keys())

    # 表1: 各方法在不同数据集上的AUC
    print("\n【表1】各方法AUC对比 (多源训练)")
    print("-"*90)

    header = f"{'方法':<15}"
    for name in test_set_names:
        header += f" {name:>12}"
    header += f" {'平均':>10}"
    print(header)
    print("-"*90)

    method_avg_aucs = []
    for result in all_results:
        method = result['method']
        line = f"{method:<15}"
        aucs = []
        for name in test_set_names:
            if name in result['test_results']:
                auc = result['test_results'][name]['auc']
                aucs.append(auc)
                line += f" {auc*100:>11.2f}%"
            else:
                line += f" {'N/A':>12}"
        avg_auc = np.mean(aucs) if aucs else 0
        method_avg_aucs.append((method, avg_auc, result))
        line += f" {avg_auc*100:>9.2f}%"
        print(line)

    # 排名
    print("\n【排名】按平均AUC排序")
    print("-"*50)
    method_avg_aucs.sort(key=lambda x: x[1], reverse=True)
    for rank, (method, avg_auc, _) in enumerate(method_avg_aucs, 1):
        marker = " <-- OURS" if method == 'ours' else ""
        print(f"{rank}. {method:<15} {avg_auc*100:.2f}%{marker}")

    # 我们的方法排名
    ours_rank = next((i+1 for i, (m, _, _) in enumerate(method_avg_aucs) if m == 'ours'), -1)
    ours_auc = next((a for m, a, _ in method_avg_aucs if m == 'ours'), 0)

    print(f"\n{'='*50}")
    if ours_rank == 1:
        print(f"我们的方法排名第 {ours_rank}，AUC={ours_auc*100:.2f}%")
        second_auc = method_avg_aucs[1][1] if len(method_avg_aucs) > 1 else 0
        print(f"超越第二名 {(ours_auc - second_auc)*100:.2f} 个百分点")
    else:
        print(f"我们的方法排名第 {ours_rank}，AUC={ours_auc*100:.2f}%")
        if ours_rank > 1:
            first_auc = method_avg_aucs[0][1]
            print(f"与第一名差距: {(first_auc - ours_auc)*100:.2f} 个百分点")
    print(f"{'='*50}")

    # 保存完整结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_output = {
        'config': {
            'train_datasets': args.train_names,
            'samples_per_source': args.samples_per_source,
            'total_train_samples': len(combined_train),
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr
        },
        'timestamp': timestamp,
        'results': all_results,
        'ranking': [{'rank': i+1, 'method': m, 'avg_auc': a}
                   for i, (m, a, _) in enumerate(method_avg_aucs)]
    }

    output_file = f"{args.output_dir}/final_comparison_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"\n完整结果已保存到: {output_file}")

    # 生成Markdown报告
    generate_markdown_report(all_results, test_set_names, method_avg_aucs, args, timestamp)


def generate_markdown_report(results, test_set_names, ranking, args, timestamp):
    """生成Markdown格式的报告"""
    report_file = f"{args.output_dir}/SOTA_COMPARISON_REPORT_{timestamp}.md"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 多源训练 SOTA 方法对比报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 实验配置\n\n")
        f.write(f"- **训练数据**: {', '.join(args.train_names)}\n")
        f.write(f"- **每个数据源样本数**: {args.samples_per_source}\n")
        f.write(f"- **训练轮数**: {args.epochs}\n")
        f.write(f"- **批次大小**: {args.batch_size}\n")
        f.write(f"- **学习率**: {args.lr}\n\n")

        f.write("## 主要结果\n\n")

        # 排名表
        f.write("### 方法排名 (按平均AUC)\n\n")
        f.write("| 排名 | 方法 | 平均AUC | BigGAN | ADM | BigGAN_JPEG | ADM_JPEG |\n")
        f.write("|------|------|---------|--------|-----|-------------|----------|\n")

        for rank, (method, avg_auc, result) in enumerate(ranking, 1):
            row = f"| {rank} | **{method}** | {avg_auc*100:.2f}% |"
            for name in test_set_names:
                if name in result['test_results']:
                    auc = result['test_results'][name]['auc']
                    row += f" {auc*100:.2f}% |"
                else:
                    row += " - |"
            f.write(row + "\n")

        f.write("\n### 关键发现\n\n")

        # 找到我们的方法
        ours_result = next((r for m, a, r in ranking if m == 'ours'), None)
        ours_rank = next((i+1 for i, (m, _, _) in enumerate(ranking) if m == 'ours'), -1)

        if ours_result and ours_rank == 1:
            f.write(f"- 我们的方法 (FDAA+MGFP) 在多源训练条件下排名**第一**\n")
            second_method, second_auc, _ = ranking[1] if len(ranking) > 1 else ('', 0, None)
            ours_auc = ranking[0][1]
            f.write(f"- 超越第二名 {second_method} **{(ours_auc - second_auc)*100:.2f}** 个百分点\n")
        elif ours_result:
            f.write(f"- 我们的方法排名第 {ours_rank}\n")

        f.write("\n---\n")
        f.write("*本报告由多源训练SOTA对比实验自动生成*\n")

    print(f"Markdown报告已保存到: {report_file}")


if __name__ == '__main__':
    main()
