#!/usr/bin/env python3
"""
SOTA方法对比实验脚本

对比方法:
1. CNNDetection (CVPR 2020)
2. Spec (2019)
3. GramNet (CVPR 2020)
4. F3-Net (ECCV 2020)
5. UnivFD (CVPR 2023)
6. NPR (2023)
7. FreqNet (AAAI 2024)
8. DIRE (ICCV 2023)
9. Ours (FDAA + MGFP)

Usage:
    python experiments/run_sota_comparison.py --train_samples 50000 --epochs 10
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.sota_methods import create_sota_model, list_sota_methods
from models.detector import AIGCDetector, AIGCDetectorLite
from data.genimage_dataset import GenImageDataset


def get_transforms(img_size=224):
    """获取数据变换"""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
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
        optimizer.step()

        total_loss += loss.item()

        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(all_labels, all_preds)

    return avg_loss, auc


def evaluate(model, dataloader, criterion, device, desc="Evaluating"):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=desc, leave=False)
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images)

            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs

            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(all_labels, all_preds)
    ap = average_precision_score(all_labels, all_preds)

    # 计算准确率
    pred_labels = (np.array(all_preds) > 0.5).astype(int)
    acc = accuracy_score(all_labels, pred_labels)

    return {
        'loss': avg_loss,
        'auc': auc,
        'ap': ap,
        'accuracy': acc
    }


def train_model(model, train_loader, val_loader, args, device, method_name):
    """训练单个模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auc = 0
    best_metrics = None

    print(f"\n{'='*60}")
    print(f"训练 {method_name}")
    print(f"{'='*60}")

    for epoch in range(args.epochs):
        # 训练
        train_loss, train_auc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            desc=f"{method_name} Epoch {epoch+1}/{args.epochs}"
        )

        # 验证
        val_metrics = evaluate(model, val_loader, criterion, device, desc="Validating")

        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss={train_loss:.4f}, Train AUC={train_auc:.4f}, "
              f"Val AUC={val_metrics['auc']:.4f}, Val Acc={val_metrics['accuracy']:.4f}")

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_metrics = val_metrics.copy()

    return best_metrics


def create_our_model(args, device):
    """创建我们的模型 (FDAA + MGFP)"""
    model = AIGCDetectorLite(
        num_classes=2,
        img_size=args.img_size,
        embed_dim=args.embed_dim,
        num_prototypes=4,
        dropout=0.1
    )
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description='SOTA方法对比实验')
    parser.add_argument('--data_root', type=str,
                        default='datasets/genimage_partial/imagenet_ai_0419_biggan',
                        help='数据集根目录')
    parser.add_argument('--train_samples', type=int, default=50000,
                        help='训练样本数')
    parser.add_argument('--val_samples', type=int, default=5000,
                        help='验证样本数')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='outputs/sota_comparison')
    parser.add_argument('--methods', nargs='+', default=['all'],
                        help='要测试的方法，使用 "all" 测试全部')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 数据变换
    train_transform, val_transform = get_transforms(args.img_size)

    # 创建数据集
    print(f"\n加载数据集: {args.data_root}")

    train_dataset = GenImageDataset(
        root_dir=args.data_root,
        split='train',
        transform=train_transform
    )

    val_dataset = GenImageDataset(
        root_dir=args.data_root,
        split='val',
        transform=val_transform
    )

    # 采样子集
    if args.train_samples < len(train_dataset):
        train_indices = np.random.choice(len(train_dataset), args.train_samples, replace=False)
        train_dataset = Subset(train_dataset, train_indices)

    if args.val_samples < len(val_dataset):
        val_indices = np.random.choice(len(val_dataset), args.val_samples, replace=False)
        val_dataset = Subset(val_dataset, val_indices)

    print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # 确定要测试的方法
    all_methods = list(list_sota_methods().keys())
    if 'all' in args.methods:
        methods_to_test = all_methods
    else:
        methods_to_test = [m for m in args.methods if m in all_methods]

    # 添加我们的方法
    methods_to_test = ['ours'] + methods_to_test

    print(f"\n将测试以下方法: {methods_to_test}")

    # 存储结果
    results = {
        'args': vars(args),
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'methods': {}
    }

    # 训练和评估每个方法
    for method_name in methods_to_test:
        print(f"\n{'#'*60}")
        print(f"# 方法: {method_name.upper()}")
        print(f"{'#'*60}")

        try:
            # 创建模型
            if method_name == 'ours':
                model = create_our_model(args, device)
            else:
                model = create_sota_model(method_name, num_classes=2)
                model = model.to(device)

            # 计算参数量
            num_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(f"参数量: {num_params/1e6:.2f}M (可训练: {trainable_params/1e6:.2f}M)")

            # 训练
            start_time = time.time()
            metrics = train_model(model, train_loader, val_loader, args, device, method_name)
            train_time = time.time() - start_time

            # 保存结果
            results['methods'][method_name] = {
                'auc': metrics['auc'],
                'ap': metrics['ap'],
                'accuracy': metrics['accuracy'],
                'params_m': num_params / 1e6,
                'train_time_s': train_time
            }

            print(f"\n{method_name} 最终结果:")
            print(f"  AUC: {metrics['auc']:.4f}")
            print(f"  AP: {metrics['ap']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  训练时间: {train_time/60:.1f} 分钟")

            # 清理GPU内存
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"错误: {method_name} 训练失败: {e}")
            results['methods'][method_name] = {'error': str(e)}

    # 保存结果
    result_file = output_dir / f"sota_comparison_{results['timestamp']}.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存到: {result_file}")

    # 打印汇总表格
    print("\n" + "="*80)
    print("SOTA方法对比汇总")
    print("="*80)
    print(f"{'方法':<15} {'AUC':>10} {'AP':>10} {'Accuracy':>10} {'参数量(M)':>12}")
    print("-"*80)

    # 按AUC排序
    sorted_methods = sorted(
        [(k, v) for k, v in results['methods'].items() if 'auc' in v],
        key=lambda x: x[1]['auc'],
        reverse=True
    )

    for method_name, metrics in sorted_methods:
        print(f"{method_name:<15} {metrics['auc']:>10.4f} {metrics['ap']:>10.4f} "
              f"{metrics['accuracy']:>10.4f} {metrics['params_m']:>12.2f}")

    print("="*80)

    # 生成LaTeX表格
    latex_file = output_dir / "sota_comparison_table.tex"
    with open(latex_file, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Comparison with State-of-the-Art Methods}\n")
        f.write("\\label{tab:sota}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Method & AUC & AP & Accuracy \\\\\n")
        f.write("\\midrule\n")

        for method_name, metrics in sorted_methods:
            is_ours = method_name == 'ours'
            if is_ours:
                f.write(f"\\textbf{{Ours (FDAA+MGFP)}} & \\textbf{{{metrics['auc']:.4f}}} & "
                       f"\\textbf{{{metrics['ap']:.4f}}} & \\textbf{{{metrics['accuracy']:.4f}}} \\\\\n")
            else:
                method_display = {
                    'cnndetection': 'CNNDetection',
                    'spec': 'Spec',
                    'gramnet': 'GramNet',
                    'f3net': 'F3-Net',
                    'univfd': 'UnivFD',
                    'npr': 'NPR',
                    'freqnet': 'FreqNet',
                    'dire': 'DIRE'
                }.get(method_name, method_name)
                f.write(f"{method_display} & {metrics['auc']:.4f} & {metrics['ap']:.4f} & "
                       f"{metrics['accuracy']:.4f} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"LaTeX表格已保存到: {latex_file}")


if __name__ == '__main__':
    main()
