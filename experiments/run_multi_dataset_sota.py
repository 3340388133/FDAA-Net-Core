#!/usr/bin/env python3
"""
多数据集 SOTA 对比实验 - 跨数据集泛化能力测试

核心实验设计:
1. 在 Dataset A 上训练，在 Dataset A (域内) + Dataset B (跨域) 上测试
2. 在 Dataset B 上训练，在 Dataset B (域内) + Dataset A (跨域) 上测试
3. 同时测试鲁棒性 (JPEG压缩)
4. 对比 5 种 SOTA 方法

Usage:
    python experiments/run_multi_dataset_sota.py
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
from PIL import Image, ImageFilter

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.sota_methods import create_sota_model
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
            raise ValueError(f"数据目录不完整: {data_root}/{split}, "
                           f"检查到: ai={ai_dir}, nature={nature_dir}")

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

        print(f"  数据集 {data_root} [{split}]: "
              f"AI={len(ai_images)}, Real={len(nature_images)}, "
              f"Total={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert('RGB')

        # 可选 JPEG 压缩
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
# 主实验流程
# ============================================================================

def run_single_experiment(method_name, train_dataset_name, train_loader,
                          test_datasets, args, device):
    """
    在一个数据集上训练，在多个数据集上测试

    Args:
        method_name: 方法名称
        train_dataset_name: 训练数据集名称
        train_loader: 训练数据加载器
        test_datasets: dict {dataset_name: data_root}
        args: 参数
        device: 设备

    Returns:
        results: 实验结果
    """
    print(f"\n{'='*70}")
    print(f"方法: {method_name.upper()} | 训练集: {train_dataset_name}")
    print(f"{'='*70}")

    # 创建模型
    model = create_model(method_name, device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: {total_params/1e6:.2f}M (可训练: {trainable_params/1e6:.2f}M)")

    # 训练
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_time = time.time()
    for epoch in range(args.epochs):
        train_loss, train_auc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            desc=f"{method_name} Epoch {epoch+1}/{args.epochs}"
        )
        scheduler.step()
        print(f"  Epoch {epoch+1}/{args.epochs}: Loss={train_loss:.4f}, AUC={train_auc:.4f}")

    train_time = (time.time() - start_time) / 60
    print(f"训练耗时: {train_time:.1f} min")

    # 在多个数据集上测试
    _, val_transform = get_transforms()
    results = {
        'method': method_name,
        'train_dataset': train_dataset_name,
        'train_time_min': train_time,
        'params_M': total_params / 1e6,
        'test_results': {}
    }

    for ds_name, ds_root in test_datasets.items():
        is_cross_domain = (ds_name != train_dataset_name)
        tag = "跨域" if is_cross_domain else "域内"
        print(f"\n  测试: {ds_name} ({tag})")

        ds_results = {}

        # 原始数据测试
        try:
            val_dataset = GenImageDataset(
                data_root=ds_root, split='val',
                transform=val_transform, max_samples=args.val_samples
            )
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                    shuffle=False, num_workers=4)
            metrics = evaluate(model, val_loader, device,
                             desc=f"测试 {ds_name} (原始)")
            ds_results['original'] = metrics
            print(f"    原始: AUC={metrics['auc']:.4f}, "
                  f"AP={metrics['ap']:.4f}, Acc={metrics['accuracy']:.4f}")
        except Exception as e:
            print(f"    原始测试失败: {e}")
            ds_results['original'] = {'auc': 0, 'ap': 0, 'accuracy': 0}

        # JPEG压缩测试
        try:
            val_dataset_jpeg = GenImageDataset(
                data_root=ds_root, split='val',
                transform=val_transform, max_samples=args.val_samples,
                apply_jpeg=True, jpeg_quality=75
            )
            val_loader_jpeg = DataLoader(val_dataset_jpeg, batch_size=args.batch_size,
                                         shuffle=False, num_workers=4)
            metrics_jpeg = evaluate(model, val_loader_jpeg, device,
                                   desc=f"测试 {ds_name} (JPEG)")
            ds_results['jpeg_q75'] = metrics_jpeg
            print(f"    JPEG(Q75): AUC={metrics_jpeg['auc']:.4f}, "
                  f"AP={metrics_jpeg['ap']:.4f}, Acc={metrics_jpeg['accuracy']:.4f}")
        except Exception as e:
            print(f"    JPEG测试失败: {e}")
            ds_results['jpeg_q75'] = {'auc': 0, 'ap': 0, 'accuracy': 0}

        results['test_results'][ds_name] = ds_results

    return results


def main():
    parser = argparse.ArgumentParser(description='多数据集SOTA对比实验')
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['BigGAN', 'ADM'],
                       help='数据集名称列表')
    parser.add_argument('--data_roots', type=str, nargs='+',
                       default=[
                           'datasets/genimage_partial/imagenet_ai_0419_biggan',
                           'datasets/genimage_partial/imagenet_ai_0508_adm'
                       ],
                       help='数据集路径列表')
    parser.add_argument('--train_samples', type=int, default=20000)
    parser.add_argument('--val_samples', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default='outputs/multi_dataset_sota')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['ours', 'cnndetection', 'spec', 'f3net', 'freqnet'])
    args = parser.parse_args()

    # 设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"数据集: {args.datasets}")
    print(f"方法: {args.methods}")
    print(f"训练样本: {args.train_samples}, 测试样本: {args.val_samples}")
    print(f"训练轮数: {args.epochs}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 验证数据集存在
    dataset_map = {}
    for name, root in zip(args.datasets, args.data_roots):
        root_path = Path(root)
        if not root_path.exists():
            print(f"警告: 数据集 {name} 路径不存在: {root}")
            continue
        # 验证数据目录结构
        train_dir = root_path / 'train'
        val_dir = root_path / 'val'
        if not train_dir.exists() or not val_dir.exists():
            print(f"警告: 数据集 {name} 缺少 train/val 目录")
            continue
        dataset_map[name] = root
        print(f"数据集 {name}: {root} ✓")

    if len(dataset_map) < 2:
        print(f"\n错误: 需要至少2个数据集进行跨域测试，当前只有 {len(dataset_map)} 个")
        print("可用数据集:", list(dataset_map.keys()))
        sys.exit(1)

    # 准备训练数据加载器
    train_transform, _ = get_transforms()

    # 运行所有实验
    all_results = []

    for train_ds_name in dataset_map:
        print(f"\n{'#'*70}")
        print(f"# 训练集: {train_ds_name}")
        print(f"{'#'*70}")

        # 创建训练数据
        train_dataset = GenImageDataset(
            data_root=dataset_map[train_ds_name],
            split='train', transform=train_transform,
            max_samples=args.train_samples
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)

        for method in args.methods:
            try:
                results = run_single_experiment(
                    method_name=method,
                    train_dataset_name=train_ds_name,
                    train_loader=train_loader,
                    test_datasets=dataset_map,
                    args=args,
                    device=device
                )
                all_results.append(results)

                # 保存中间结果
                with open(f"{args.output_dir}/results_{train_ds_name}_{method}.json", 'w') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

            except Exception as e:
                print(f"方法 {method} (训练集={train_ds_name}) 失败: {e}")
                import traceback
                traceback.print_exc()

    # ========================================================================
    # 生成综合对比表格
    # ========================================================================
    print("\n" + "="*90)
    print("多数据集 SOTA 对比结果汇总")
    print("="*90)

    dataset_names = list(dataset_map.keys())

    # 表1: 域内性能对比 (在每个数据集上训练和测试)
    print("\n【表1】域内性能 (AUC / Accuracy)")
    print("-"*70)
    header = f"{'方法':<15}"
    for ds in dataset_names:
        header += f" {ds:^22}"
    print(header)
    print("-"*70)

    for method in args.methods:
        line = f"{method:<15}"
        for ds in dataset_names:
            # 找到对应的结果
            r = next((r for r in all_results
                      if r['method'] == method and r['train_dataset'] == ds), None)
            if r and ds in r['test_results']:
                orig = r['test_results'][ds].get('original', {})
                auc = orig.get('auc', 0)
                acc = orig.get('accuracy', 0)
                line += f" {auc:.4f} / {acc:.4f}   "
            else:
                line += f" {'N/A':^22}"
        print(line)

    # 表2: 跨域泛化性能 (训练在A测试在B)
    print(f"\n\n【表2】跨域泛化性能 (AUC / Accuracy)")
    print("-"*90)
    header = f"{'方法':<15}"
    for train_ds in dataset_names:
        for test_ds in dataset_names:
            if train_ds != test_ds:
                header += f" {train_ds}→{test_ds:^18}"
    header += f" {'平均跨域':^12}"
    print(header)
    print("-"*90)

    for method in args.methods:
        line = f"{method:<15}"
        cross_aucs = []
        for train_ds in dataset_names:
            for test_ds in dataset_names:
                if train_ds != test_ds:
                    r = next((r for r in all_results
                              if r['method'] == method and r['train_dataset'] == train_ds), None)
                    if r and test_ds in r['test_results']:
                        orig = r['test_results'][test_ds].get('original', {})
                        auc = orig.get('auc', 0)
                        acc = orig.get('accuracy', 0)
                        cross_aucs.append(auc)
                        line += f" {auc:.4f} / {acc:.4f}   "
                    else:
                        line += f" {'N/A':^18}"
        avg_cross = np.mean(cross_aucs) if cross_aucs else 0
        line += f"  {avg_cross:.4f}"
        print(line)

    # 表3: JPEG鲁棒性 (跨域 + JPEG压缩)
    print(f"\n\n【表3】跨域 + JPEG压缩 鲁棒性 (AUC / Accuracy)")
    print("-"*90)
    header = f"{'方法':<15}"
    for train_ds in dataset_names:
        for test_ds in dataset_names:
            if train_ds != test_ds:
                header += f" {train_ds}→{test_ds}(JPEG)   "
    header += f" {'平均':^12}"
    print(header)
    print("-"*90)

    for method in args.methods:
        line = f"{method:<15}"
        jpeg_aucs = []
        for train_ds in dataset_names:
            for test_ds in dataset_names:
                if train_ds != test_ds:
                    r = next((r for r in all_results
                              if r['method'] == method and r['train_dataset'] == train_ds), None)
                    if r and test_ds in r['test_results']:
                        jpeg = r['test_results'][test_ds].get('jpeg_q75', {})
                        auc = jpeg.get('auc', 0)
                        acc = jpeg.get('accuracy', 0)
                        jpeg_aucs.append(auc)
                        line += f" {auc:.4f} / {acc:.4f}   "
                    else:
                        line += f" {'N/A':^22}"
        avg_jpeg = np.mean(jpeg_aucs) if jpeg_aucs else 0
        line += f"  {avg_jpeg:.4f}"
        print(line)

    # 表4: 综合排名
    print(f"\n\n【表4】综合排名")
    print("-"*70)
    print(f"{'方法':<15} {'域内AUC':^12} {'跨域AUC':^12} {'跨域JPEG':^12} {'综合得分':^12}")
    print("-"*70)

    method_scores = []
    for method in args.methods:
        in_domain_aucs = []
        cross_aucs = []
        jpeg_aucs = []

        for r in all_results:
            if r['method'] != method:
                continue
            train_ds = r['train_dataset']
            for test_ds, test_r in r['test_results'].items():
                orig = test_r.get('original', {})
                jpeg = test_r.get('jpeg_q75', {})
                if test_ds == train_ds:
                    in_domain_aucs.append(orig.get('auc', 0))
                else:
                    cross_aucs.append(orig.get('auc', 0))
                    jpeg_aucs.append(jpeg.get('auc', 0))

        avg_in = np.mean(in_domain_aucs) if in_domain_aucs else 0
        avg_cross = np.mean(cross_aucs) if cross_aucs else 0
        avg_jpeg = np.mean(jpeg_aucs) if jpeg_aucs else 0

        # 综合得分: 域内(30%) + 跨域(40%) + JPEG鲁棒(30%)
        composite = 0.3 * avg_in + 0.4 * avg_cross + 0.3 * avg_jpeg
        method_scores.append((method, avg_in, avg_cross, avg_jpeg, composite))

    # 按综合得分排序
    method_scores.sort(key=lambda x: x[4], reverse=True)
    for rank, (method, avg_in, avg_cross, avg_jpeg, composite) in enumerate(method_scores, 1):
        marker = " ★" if method == 'ours' else ""
        print(f"{rank}. {method:<13} {avg_in:.4f}       {avg_cross:.4f}       "
              f"{avg_jpeg:.4f}       {composite:.4f}{marker}")

    # 保存完整结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_output = {
        'config': {
            'datasets': args.datasets,
            'data_roots': args.data_roots,
            'train_samples': args.train_samples,
            'val_samples': args.val_samples,
            'epochs': args.epochs,
            'methods': args.methods,
        },
        'results': all_results,
        'ranking': [
            {'rank': i+1, 'method': m, 'in_domain_auc': a,
             'cross_domain_auc': b, 'jpeg_auc': c, 'composite': d}
            for i, (m, a, b, c, d) in enumerate(method_scores)
        ]
    }

    output_file = f"{args.output_dir}/final_results_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"\n完整结果已保存到: {output_file}")


if __name__ == '__main__':
    main()
