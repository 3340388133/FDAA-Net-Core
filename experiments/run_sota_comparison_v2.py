#!/usr/bin/env python3
"""
改进版 SOTA 方法对比实验脚本

改进点:
1. 使用域偏移模拟，测试跨域泛化能力
2. 不仅测试原始数据，还测试扰动后的数据
3. 更全面的评估指标

Usage:
    python experiments/run_sota_comparison_v2.py --train_samples 30000 --epochs 10
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

from models.sota_methods import create_sota_model, list_sota_methods
from models.detector import AIGCDetectorLite


class DomainShiftDataset(Dataset):
    """支持域偏移的数据集 - 模拟不同生成器特性"""

    DOMAIN_SHIFTS = {
        'original': None,
        'style_warm': {'brightness': 1.1, 'contrast': 0.95, 'saturation': 1.15},
        'style_cold': {'brightness': 0.95, 'contrast': 1.1, 'saturation': 0.9},
        'style_sharp': {'sharpness': 1.5},
        'style_soft': {'blur': 0.8},
        'jpeg_compress': {'jpeg_quality': 75},
    }

    def __init__(self, data_root, split='train', domain='original',
                 transform=None, max_samples=None):
        self.data_root = Path(data_root)
        self.domain = domain
        self.transform = transform
        self.samples = []

        split_dir = self.data_root / split

        # 查找 AI 和真实图像目录
        ai_dirs = [split_dir / 'ai', split_dir / 'fake', split_dir / '1']
        nature_dirs = [split_dir / 'nature', split_dir / 'real', split_dir / '0']

        ai_dir = next((d for d in ai_dirs if d.exists()), None)
        nature_dir = next((d for d in nature_dirs if d.exists()), None)

        if ai_dir is None or nature_dir is None:
            raise ValueError(f"数据目录不完整: {data_root}")

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

    def apply_domain_shift(self, image):
        """应用域偏移变换"""
        if self.domain == 'original' or self.domain is None:
            return image

        shift_config = self.DOMAIN_SHIFTS.get(self.domain, None)
        if shift_config is None:
            return image

        from torchvision.transforms import functional as F

        if 'brightness' in shift_config:
            image = F.adjust_brightness(image, shift_config['brightness'])
        if 'contrast' in shift_config:
            image = F.adjust_contrast(image, shift_config['contrast'])
        if 'saturation' in shift_config:
            image = F.adjust_saturation(image, shift_config['saturation'])
        if 'sharpness' in shift_config:
            image = F.adjust_sharpness(image, shift_config['sharpness'])
        if 'blur' in shift_config:
            image = image.filter(ImageFilter.GaussianBlur(radius=shift_config['blur']))
        if 'jpeg_quality' in shift_config:
            import io
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=shift_config['jpeg_quality'])
            buffer.seek(0)
            image = Image.open(buffer).convert('RGB')

        return image

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert('RGB')
        image = self.apply_domain_shift(image)

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

    return total_loss / len(dataloader), roc_auc_score(all_labels, all_preds)


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

    auc = roc_auc_score(all_labels, all_preds)
    ap = average_precision_score(all_labels, all_preds)
    pred_labels = (np.array(all_preds) > 0.5).astype(int)
    acc = accuracy_score(all_labels, pred_labels)

    return {'auc': auc, 'ap': ap, 'accuracy': acc}


def train_and_evaluate_method(method_name, args, train_loader, device, data_root):
    """训练并评估单个方法"""
    print(f"\n{'#'*60}")
    print(f"# 方法: {method_name.upper()}")
    print(f"{'#'*60}")

    # 创建模型
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

    model = model.to(device)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: {total_params/1e6:.2f}M (可训练: {trainable_params/1e6:.2f}M)")

    # 训练
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\n{'='*60}")
    print(f"训练 {method_name}")
    print(f"{'='*60}")

    start_time = time.time()
    best_auc = 0

    for epoch in range(args.epochs):
        train_loss, train_auc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            desc=f"{method_name} Epoch {epoch+1}/{args.epochs}"
        )
        scheduler.step()
        print(f"Epoch {epoch+1}/{args.epochs}: Loss={train_loss:.4f}, AUC={train_auc:.4f}")

    train_time = (time.time() - start_time) / 60

    # 在多个域上测试
    _, val_transform = get_transforms()
    domains = ['original', 'style_warm', 'style_cold', 'jpeg_compress']

    results = {
        'method': method_name,
        'train_time_min': train_time,
        'domain_results': {}
    }

    print(f"\n评估跨域性能:")
    for domain in domains:
        val_dataset = DomainShiftDataset(
            data_root=data_root,
            split='val',
            domain=domain,
            transform=val_transform,
            max_samples=args.val_samples
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                               shuffle=False, num_workers=4)

        metrics = evaluate(model, val_loader, device, desc=f"测试 {domain}")
        results['domain_results'][domain] = metrics
        print(f"  {domain}: AUC={metrics['auc']:.4f}, Acc={metrics['accuracy']:.4f}")

    # 计算平均性能
    avg_auc = np.mean([r['auc'] for r in results['domain_results'].values()])
    results['avg_auc'] = avg_auc
    print(f"\n平均 AUC: {avg_auc:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                       default='datasets/genimage_partial/imagenet_ai_0419_biggan')
    parser.add_argument('--train_samples', type=int, default=30000)
    parser.add_argument('--val_samples', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default='outputs/sota_comparison_v2')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['ours', 'cnndetection', 'spec', 'f3net', 'freqnet'])
    args = parser.parse_args()

    # 设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 准备数据
    train_transform, _ = get_transforms()
    train_dataset = DomainShiftDataset(
        data_root=args.data_root,
        split='train',
        domain='original',
        transform=train_transform,
        max_samples=args.train_samples
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=4, pin_memory=True)

    print(f"\n训练样本: {len(train_dataset)}")
    print(f"测试方法: {args.methods}")

    # 训练并评估所有方法
    all_results = []

    for method in args.methods:
        try:
            results = train_and_evaluate_method(
                method, args, train_loader, device, args.data_root
            )
            all_results.append(results)

            # 保存中间结果
            with open(f"{args.output_dir}/results_{method}.json", 'w') as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            print(f"方法 {method} 失败: {e}")
            import traceback
            traceback.print_exc()

    # 生成对比表格
    print("\n" + "="*80)
    print("SOTA 对比结果汇总")
    print("="*80)
    print(f"{'方法':<15} {'原始AUC':<10} {'暖色调':<10} {'冷色调':<10} {'JPEG压缩':<10} {'平均AUC':<10}")
    print("-"*80)

    for r in all_results:
        method = r['method']
        dr = r['domain_results']
        print(f"{method:<15} "
              f"{dr['original']['auc']:.4f}     "
              f"{dr['style_warm']['auc']:.4f}     "
              f"{dr['style_cold']['auc']:.4f}     "
              f"{dr['jpeg_compress']['auc']:.4f}     "
              f"{r['avg_auc']:.4f}")

    # 保存最终结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f"{args.output_dir}/final_results_{timestamp}.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n结果已保存到: {args.output_dir}")


if __name__ == '__main__':
    main()
