#!/usr/bin/env python
"""
测试时增强(TTA)评估脚本
在不重新训练的情况下，通过测试时数据增强提升跨域泛化性能
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from tqdm import tqdm

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detector_generalized import GeneralizedAIGCDetector


class GenImageDataset(Dataset):
    """GenImage数据集加载器"""
    def __init__(self, root_dir, transform=None, max_samples=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        # 加载真实图像
        real_dir = self.root_dir / 'nature'
        if real_dir.exists():
            for img_path in real_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((img_path, 0))

        # 加载AI生成图像
        ai_dir = self.root_dir / 'ai'
        if ai_dir.exists():
            for img_path in ai_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((img_path, 1))

        # 限制样本数
        if max_samples and len(self.samples) > max_samples:
            np.random.seed(42)
            indices = np.random.choice(len(self.samples), max_samples, replace=False)
            self.samples = [self.samples[i] for i in indices]

        print(f"Loaded {len(self.samples)} samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return {'image': image, 'label': label, 'path': str(img_path)}
        except Exception as e:
            # 返回一个随机图像以防错误
            image = torch.randn(3, 224, 224)
            return {'image': image, 'label': label, 'path': str(img_path)}


class TTAAugmentor:
    """测试时增强器"""
    def __init__(self, img_size=224, tta_mode='full'):
        self.img_size = img_size
        self.tta_mode = tta_mode

        # 基础变换
        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # TTA变换列表
        if tta_mode == 'flip':
            # 只做水平翻转
            self.tta_transforms = [
                self.base_transform,
                transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
            ]
        elif tta_mode == 'scales':
            # 多尺度
            scales = [0.9, 1.0, 1.1]
            self.tta_transforms = []
            for scale in scales:
                size = int(img_size * scale)
                self.tta_transforms.append(transforms.Compose([
                    transforms.Resize((size, size)),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ]))
        elif tta_mode == 'full':
            # 完整TTA：翻转 + 多尺度 + 轻微旋转
            self.tta_transforms = [
                # 原始
                self.base_transform,
                # 水平翻转
                transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ]),
                # 缩小
                transforms.Compose([
                    transforms.Resize((int(img_size * 0.9), int(img_size * 0.9))),
                    transforms.Pad((img_size - int(img_size * 0.9)) // 2 + 1),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ]),
                # 放大
                transforms.Compose([
                    transforms.Resize((int(img_size * 1.1), int(img_size * 1.1))),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ]),
                # 翻转 + 缩小
                transforms.Compose([
                    transforms.Resize((int(img_size * 0.9), int(img_size * 0.9))),
                    transforms.RandomHorizontalFlip(p=1.0),
                    transforms.Pad((img_size - int(img_size * 0.9)) // 2 + 1),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ]),
            ]
        else:
            self.tta_transforms = [self.base_transform]

    def __call__(self, image):
        """返回所有TTA变换后的图像列表"""
        return [t(image) for t in self.tta_transforms]

    @property
    def num_augmentations(self):
        return len(self.tta_transforms)


def load_model(checkpoint_path, device):
    """加载模型检查点"""
    model = GeneralizedAIGCDetector(
        backbone_name="ViT-L/14",
        num_classes=2,
        img_size=224,
        patch_size=14,
        embed_dim=1024,
        num_adapter_layers=4,
        use_multi_level_da=True,
        num_prototypes=8,
        use_hierarchical=True,
        use_pcr=True,
        num_domains=2,
        use_domain_adversarial=False,
        dropout=0.1,
        freeze_backbone=True
    ).to(device)

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: checkpoint {checkpoint_path} not found, using random weights")

    return model


def evaluate_with_tta(model, data_root, device, tta_mode='full', max_samples=1000, batch_size=16):
    """使用TTA进行评估"""
    model.eval()

    tta_augmentor = TTAAugmentor(img_size=224, tta_mode=tta_mode)
    base_transform = tta_augmentor.base_transform

    # 加载数据集
    dataset = GenImageDataset(data_root, transform=None, max_samples=max_samples)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=f"Evaluating with TTA ({tta_mode})"):
            sample = dataset.samples[i]
            img_path, label = sample

            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                continue

            # 获取所有TTA变换
            tta_images = tta_augmentor(image)
            tta_batch = torch.stack(tta_images).to(device)

            # 前向传播
            outputs = model(tta_batch)

            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('output'))
            else:
                logits = outputs

            # 平均所有TTA预测
            probs = F.softmax(logits, dim=1)[:, 1]  # AI类别概率
            avg_prob = probs.mean().cpu().item()

            all_probs.append(avg_prob)
            all_labels.append(label)

    # 计算指标
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    auc = roc_auc_score(all_labels, all_probs)
    ap = average_precision_score(all_labels, all_probs)
    preds = (all_probs > 0.5).astype(int)
    acc = accuracy_score(all_labels, preds)

    return {'auc': auc, 'ap': ap, 'accuracy': acc}


def evaluate_without_tta(model, data_root, device, max_samples=1000, batch_size=16):
    """不使用TTA进行评估（基线）"""
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    dataset = GenImageDataset(data_root, transform=transform, max_samples=max_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating without TTA"):
            images = batch['image'].to(device)
            labels = batch['label'].numpy()

            outputs = model(images)

            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('output'))
            else:
                logits = outputs

            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    auc = roc_auc_score(all_labels, all_probs)
    ap = average_precision_score(all_labels, all_probs)
    preds = (all_probs > 0.5).astype(int)
    acc = accuracy_score(all_labels, preds)

    return {'auc': auc, 'ap': ap, 'accuracy': acc}


def main():
    parser = argparse.ArgumentParser(description='TTA评估')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--data_roots', type=str, nargs='+', required=True, help='测试数据集路径')
    parser.add_argument('--datasets', type=str, nargs='+', required=True, help='数据集名称')
    parser.add_argument('--tta_modes', type=str, nargs='+', default=['none', 'flip', 'scales', 'full'],
                       help='TTA模式列表')
    parser.add_argument('--max_samples', type=int, default=1000, help='每个数据集最大样本数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--output_dir', type=str, default='outputs/tta_evaluation', help='输出目录')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    model = load_model(args.checkpoint, device)

    # 评估每个数据集
    results = {}

    for data_root, dataset_name in zip(args.data_roots, args.datasets):
        print(f"\n{'='*60}")
        print(f"Evaluating on {dataset_name}: {data_root}")
        print('='*60)

        results[dataset_name] = {}

        for tta_mode in args.tta_modes:
            print(f"\nTTA Mode: {tta_mode}")

            if tta_mode == 'none':
                metrics = evaluate_without_tta(
                    model, data_root, device,
                    max_samples=args.max_samples,
                    batch_size=args.batch_size
                )
            else:
                metrics = evaluate_with_tta(
                    model, data_root, device,
                    tta_mode=tta_mode,
                    max_samples=args.max_samples,
                    batch_size=args.batch_size
                )

            results[dataset_name][tta_mode] = metrics
            print(f"  AUC: {metrics['auc']:.4f}, AP: {metrics['ap']:.4f}, Acc: {metrics['accuracy']:.4f}")

    # 计算汇总指标
    summary = {}
    for tta_mode in args.tta_modes:
        aucs = [results[ds][tta_mode]['auc'] for ds in args.datasets]
        summary[tta_mode] = {
            'avg_auc': np.mean(aucs),
            'min_auc': np.min(aucs),
            'aucs': {ds: results[ds][tta_mode]['auc'] for ds in args.datasets}
        }

    # 打印汇总
    print(f"\n{'='*60}")
    print("Summary")
    print('='*60)
    for tta_mode in args.tta_modes:
        print(f"\n{tta_mode}:")
        print(f"  Average AUC: {summary[tta_mode]['avg_auc']:.4f}")
        print(f"  Min AUC (Cross-domain): {summary[tta_mode]['min_auc']:.4f}")
        for ds, auc in summary[tta_mode]['aucs'].items():
            print(f"    {ds}: {auc:.4f}")

    # 保存结果
    output_file = output_dir / 'tta_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'summary': summary,
            'checkpoint': args.checkpoint,
            'tta_modes': args.tta_modes
        }, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # 找出最佳TTA模式
    best_mode = max(summary.keys(), key=lambda m: summary[m]['min_auc'])
    print(f"\nBest TTA mode for cross-domain: {best_mode}")
    print(f"  Cross-domain AUC: {summary[best_mode]['min_auc']:.4f}")


if __name__ == '__main__':
    main()
