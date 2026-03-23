#!/usr/bin/env python3
"""
泛化能力自动优化实验

系统性尝试多种泛化改进策略，自动评估并迭代优化，直到找到最佳配置。

策略包括：
1. 多源域联合训练 (Multi-Source Domain Training)
2. 强数据增强 (Strong Augmentation)
3. 域对抗训练 (Domain Adversarial Training)
4. 频率域正则化 (Frequency Regularization)
5. 对比学习 (Contrastive Learning)
6. 特征归一化 (Feature Normalization)
7. MixStyle风格混合
8. 梯度反转层优化

Usage:
    python experiments/run_generalization_optimization.py
"""

import os
import sys
import json
import time
import random
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# 数据集定义
# ============================================================================

class GenImageDataset(Dataset):
    """GenImage数据集"""

    def __init__(self, data_root, split='train', transform=None,
                 max_samples=None, domain_id=0):
        self.data_root = Path(data_root)
        self.transform = transform
        self.domain_id = domain_id
        self.samples = []

        split_dir = self.data_root / split

        ai_dirs = [split_dir / 'ai', split_dir / 'fake', split_dir / '1']
        nature_dirs = [split_dir / 'nature', split_dir / 'real', split_dir / '0']

        ai_dir = next((d for d in ai_dirs if d.exists()), None)
        nature_dir = next((d for d in nature_dirs if d.exists()), None)

        if ai_dir is None or nature_dir is None:
            raise ValueError(f"数据目录不完整: {data_root}/{split}")

        exts = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        ai_images = []
        nature_images = []

        for ext in exts:
            ai_images.extend(list(ai_dir.glob(ext)))
            nature_images.extend(list(nature_dir.glob(ext)))

        random.shuffle(ai_images)
        random.shuffle(nature_images)

        if max_samples:
            n_per_class = max_samples // 2
            ai_images = ai_images[:n_per_class]
            nature_images = nature_images[:n_per_class]

        for img_path in ai_images:
            self.samples.append((img_path, 1, self.domain_id))
        for img_path in nature_images:
            self.samples.append((img_path, 0, self.domain_id))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, domain = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': label, 'domain': domain}


# ============================================================================
# 数据增强策略
# ============================================================================

class RandAugment:
    """随机增强"""
    def __init__(self, n=2, m=10):
        self.n = n
        self.m = m
        self.augment_list = [
            ('brightness', 0.1, 1.9),
            ('contrast', 0.1, 1.9),
            ('saturation', 0.1, 1.9),
            ('sharpness', 0.1, 1.9),
            ('rotate', -30, 30),
        ]

    def __call__(self, img):
        from torchvision.transforms import functional as TF

        ops = random.choices(self.augment_list, k=self.n)
        for op_name, min_val, max_val in ops:
            magnitude = (self.m / 10) * (max_val - min_val) + min_val

            if op_name == 'brightness':
                img = TF.adjust_brightness(img, magnitude)
            elif op_name == 'contrast':
                img = TF.adjust_contrast(img, magnitude)
            elif op_name == 'saturation':
                img = TF.adjust_saturation(img, magnitude)
            elif op_name == 'sharpness':
                img = TF.adjust_sharpness(img, magnitude)
            elif op_name == 'rotate':
                angle = random.uniform(min_val, max_val)
                img = TF.rotate(img, angle)

        return img


class JPEGCompression:
    """JPEG压缩增强"""
    def __init__(self, quality_range=(30, 100)):
        self.quality_range = quality_range

    def __call__(self, img):
        import io
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')


class GaussianBlur:
    """高斯模糊"""
    def __init__(self, radius_range=(0.1, 2.0)):
        self.radius_range = radius_range

    def __call__(self, img):
        from PIL import ImageFilter
        radius = random.uniform(*self.radius_range)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


def get_strong_augmentation(img_size=224):
    """强数据增强"""
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomApply([JPEGCompression((50, 95))], p=0.5),
        transforms.RandomApply([GaussianBlur((0.5, 1.5))], p=0.3),
        RandAugment(n=2, m=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
    ])


def get_standard_augmentation(img_size=224):
    """标准数据增强"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transform(img_size=224):
    """验证集变换"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# ============================================================================
# 泛化增强模块
# ============================================================================

class MixStyle(nn.Module):
    """
    MixStyle: 风格混合用于域泛化
    Reference: Zhou et al., "Domain Generalization with MixStyle", ICLR 2021
    """
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x):
        if not self.training or random.random() > self.p:
            return x

        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        x_normed = (x - mu) / sig

        # 随机打乱batch获取混合风格
        perm = torch.randperm(B)
        mu2, sig2 = mu[perm], sig[perm]

        # 混合风格
        lmda = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1, 1)).to(x.device)
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)

        return x_normed * sig_mix + mu_mix


class FrequencyNormalization(nn.Module):
    """
    频率域归一化
    减少不同生成器产生的频率差异
    """
    def __init__(self, num_channels=3):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_channels, affine=True)

    def forward(self, x):
        # DCT变换（简化版本）
        x_freq = torch.fft.fft2(x, norm='ortho')
        x_freq_real = x_freq.real
        x_freq_imag = x_freq.imag

        # 归一化幅度谱
        magnitude = torch.sqrt(x_freq_real**2 + x_freq_imag**2 + 1e-8)
        phase = torch.atan2(x_freq_imag, x_freq_real)

        # 归一化
        magnitude_norm = self.norm(magnitude)

        # 重建
        x_freq_norm = torch.complex(
            magnitude_norm * torch.cos(phase),
            magnitude_norm * torch.sin(phase)
        )
        x_norm = torch.fft.ifft2(x_freq_norm, norm='ortho').real

        return x_norm


class GradientReversalLayer(torch.autograd.Function):
    """梯度反转层"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def gradient_reversal(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)


# ============================================================================
# 改进的检测器
# ============================================================================

class ImprovedAIGCDetector(nn.Module):
    """
    改进的AIGC检测器 - 专注于泛化能力
    """
    def __init__(
        self,
        embed_dim=768,
        num_classes=2,
        num_domains=2,
        use_mixstyle=True,
        use_freq_norm=True,
        use_domain_adversarial=True,
        use_contrastive=True,
        dropout=0.3
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_mixstyle = use_mixstyle
        self.use_freq_norm = use_freq_norm
        self.use_domain_adversarial = use_domain_adversarial
        self.use_contrastive = use_contrastive

        # Backbone: 使用预训练ResNet
        import torchvision.models as models
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        backbone_dim = 2048

        # 冻结早期层
        for name, param in self.backbone.named_parameters():
            if 'layer4' not in name and 'layer3' not in name:
                param.requires_grad = False

        # MixStyle
        if use_mixstyle:
            self.mixstyle = MixStyle(p=0.5, alpha=0.3)

        # 频率归一化
        if use_freq_norm:
            self.freq_norm = FrequencyNormalization(3)

        # 特征投影
        self.feature_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

        # 域判别器
        if use_domain_adversarial:
            self.domain_classifier = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(embed_dim // 4, num_domains)
            )

        # 对比学习投影头
        if use_contrastive:
            self.contrast_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, 128)
            )

        # 域对抗权重调度
        self.domain_lambda = 0.0

    def set_domain_lambda(self, epoch, total_epochs):
        """设置域对抗权重（渐进式增加）"""
        p = epoch / total_epochs
        self.domain_lambda = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

    def forward(self, x, return_features=False):
        outputs = {}

        # 频率归一化
        if self.use_freq_norm and hasattr(self, 'freq_norm'):
            x = self.freq_norm(x)

        # Backbone特征
        feat = self.backbone(x)

        # MixStyle（仅训练时）
        if self.use_mixstyle and self.training and hasattr(self, 'mixstyle'):
            feat = self.mixstyle(feat)

        # 特征投影
        feat = self.feature_proj(feat)
        outputs['features'] = feat

        # 分类
        logits = self.classifier(feat)
        outputs['logits'] = logits

        # 域对抗
        if self.use_domain_adversarial and hasattr(self, 'domain_classifier'):
            feat_rev = gradient_reversal(feat, self.domain_lambda)
            domain_logits = self.domain_classifier(feat_rev)
            outputs['domain_logits'] = domain_logits

        # 对比学习特征
        if self.use_contrastive and hasattr(self, 'contrast_proj'):
            contrast_feat = self.contrast_proj(feat)
            contrast_feat = F.normalize(contrast_feat, dim=-1)
            outputs['contrast_features'] = contrast_feat

        if return_features:
            return outputs
        return outputs


# ============================================================================
# 损失函数
# ============================================================================

class SupConLoss(nn.Module):
    """监督对比损失"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_feature = features
        anchor_feature = features

        # 计算相似度
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )

        # 数值稳定性
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 排除自身
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # 计算损失
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = -mean_log_prob_pos.mean()

        return loss


class CombinedLoss(nn.Module):
    """综合损失函数"""
    def __init__(
        self,
        use_focal=True,
        use_domain=True,
        use_contrastive=True,
        domain_weight=0.1,
        contrastive_weight=0.1,
        focal_gamma=2.0
    ):
        super().__init__()
        self.use_focal = use_focal
        self.use_domain = use_domain
        self.use_contrastive = use_contrastive
        self.domain_weight = domain_weight
        self.contrastive_weight = contrastive_weight
        self.focal_gamma = focal_gamma

        self.ce_loss = nn.CrossEntropyLoss()
        self.domain_loss = nn.CrossEntropyLoss()
        self.contrastive_loss = SupConLoss(temperature=0.1)

    def focal_loss(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.focal_gamma) * ce
        return focal.mean()

    def forward(self, outputs, labels, domain_labels=None):
        loss_dict = {}

        # 分类损失
        if self.use_focal:
            cls_loss = self.focal_loss(outputs['logits'], labels)
        else:
            cls_loss = self.ce_loss(outputs['logits'], labels)
        loss_dict['cls_loss'] = cls_loss
        total_loss = cls_loss

        # 域对抗损失
        if self.use_domain and 'domain_logits' in outputs and domain_labels is not None:
            domain_loss = self.domain_loss(outputs['domain_logits'], domain_labels)
            loss_dict['domain_loss'] = domain_loss
            total_loss = total_loss + self.domain_weight * domain_loss

        # 对比损失
        if self.use_contrastive and 'contrast_features' in outputs:
            contrast_loss = self.contrastive_loss(outputs['contrast_features'], labels)
            loss_dict['contrast_loss'] = contrast_loss
            total_loss = total_loss + self.contrastive_weight * contrast_loss

        loss_dict['total_loss'] = total_loss
        return loss_dict


# ============================================================================
# 训练和评估
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    model.set_domain_lambda(epoch, total_epochs)

    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False)
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        domain_labels = batch['domain'].to(device)

        optimizer.zero_grad()
        outputs = model(images, return_features=True)

        loss_dict = criterion(outputs, labels, domain_labels)
        loss = loss_dict['total_loss']

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        probs = torch.softmax(outputs['logits'], dim=1)[:, 1].detach().cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.5

    return avg_loss, auc


def evaluate(model, dataloader, device, apply_jpeg=False, jpeg_quality=75):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            images = batch['image'].to(device)
            labels = batch['label']

            outputs = model(images)
            probs = torch.softmax(outputs['logits'], dim=1)[:, 1].cpu().numpy()

            all_preds.extend(probs)
            all_labels.extend(labels.numpy())

    try:
        auc = roc_auc_score(all_labels, all_preds)
        ap = average_precision_score(all_labels, all_preds)
    except:
        auc, ap = 0.5, 0.5

    pred_labels = (np.array(all_preds) > 0.5).astype(int)
    acc = accuracy_score(all_labels, pred_labels)

    return {'auc': auc, 'ap': ap, 'accuracy': acc}


# ============================================================================
# 实验配置
# ============================================================================

STRATEGIES = {
    'baseline': {
        'use_mixstyle': False,
        'use_freq_norm': False,
        'use_domain_adversarial': False,
        'use_contrastive': False,
        'use_strong_aug': False,
        'use_multi_source': False,
    },
    'strong_aug': {
        'use_mixstyle': False,
        'use_freq_norm': False,
        'use_domain_adversarial': False,
        'use_contrastive': False,
        'use_strong_aug': True,
        'use_multi_source': False,
    },
    'multi_source': {
        'use_mixstyle': False,
        'use_freq_norm': False,
        'use_domain_adversarial': False,
        'use_contrastive': False,
        'use_strong_aug': False,
        'use_multi_source': True,
    },
    'domain_adv': {
        'use_mixstyle': False,
        'use_freq_norm': False,
        'use_domain_adversarial': True,
        'use_contrastive': False,
        'use_strong_aug': False,
        'use_multi_source': True,
    },
    'mixstyle': {
        'use_mixstyle': True,
        'use_freq_norm': False,
        'use_domain_adversarial': False,
        'use_contrastive': False,
        'use_strong_aug': False,
        'use_multi_source': True,
    },
    'contrastive': {
        'use_mixstyle': False,
        'use_freq_norm': False,
        'use_domain_adversarial': False,
        'use_contrastive': True,
        'use_strong_aug': False,
        'use_multi_source': True,
    },
    'freq_norm': {
        'use_mixstyle': False,
        'use_freq_norm': True,
        'use_domain_adversarial': False,
        'use_contrastive': False,
        'use_strong_aug': False,
        'use_multi_source': True,
    },
    'combined_v1': {
        'use_mixstyle': True,
        'use_freq_norm': False,
        'use_domain_adversarial': True,
        'use_contrastive': True,
        'use_strong_aug': True,
        'use_multi_source': True,
    },
    'combined_v2': {
        'use_mixstyle': True,
        'use_freq_norm': True,
        'use_domain_adversarial': True,
        'use_contrastive': True,
        'use_strong_aug': True,
        'use_multi_source': True,
    },
    # 精细化组合：只组合最佳两个策略
    'multi_source_domadv': {
        'use_mixstyle': False,
        'use_freq_norm': False,
        'use_domain_adversarial': True,
        'use_contrastive': False,
        'use_strong_aug': False,
        'use_multi_source': True,
    },
    'multi_source_mixstyle': {
        'use_mixstyle': True,
        'use_freq_norm': False,
        'use_domain_adversarial': False,
        'use_contrastive': False,
        'use_strong_aug': False,
        'use_multi_source': True,
    },
    'multi_source_freqnorm': {
        'use_mixstyle': False,
        'use_freq_norm': True,
        'use_domain_adversarial': False,
        'use_contrastive': False,
        'use_strong_aug': False,
        'use_multi_source': True,
    },
    'ms_domadv_mixstyle': {
        'use_mixstyle': True,
        'use_freq_norm': False,
        'use_domain_adversarial': True,
        'use_contrastive': False,
        'use_strong_aug': False,
        'use_multi_source': True,
    },
    'ms_domadv_freqnorm': {
        'use_mixstyle': False,
        'use_freq_norm': True,
        'use_domain_adversarial': True,
        'use_contrastive': False,
        'use_strong_aug': False,
        'use_multi_source': True,
    },
}


def run_strategy(strategy_name, config, args, device, dataset_map):
    """运行单个策略"""
    print(f"\n{'='*70}")
    print(f"策略: {strategy_name}")
    print(f"配置: {config}")
    print(f"{'='*70}")

    # 数据准备
    if config['use_strong_aug']:
        train_transform = get_strong_augmentation()
    else:
        train_transform = get_standard_augmentation()
    val_transform = get_val_transform()

    # 训练数据
    if config['use_multi_source']:
        # 多源域联合训练
        train_datasets = []
        for domain_id, (ds_name, ds_root) in enumerate(dataset_map.items()):
            ds = GenImageDataset(
                ds_root, split='train', transform=train_transform,
                max_samples=args.train_samples // len(dataset_map),
                domain_id=domain_id
            )
            train_datasets.append(ds)
        train_dataset = ConcatDataset(train_datasets)
        num_domains = len(dataset_map)
    else:
        # 单源域训练（使用第一个数据集）
        ds_name = list(dataset_map.keys())[0]
        train_dataset = GenImageDataset(
            dataset_map[ds_name], split='train', transform=train_transform,
            max_samples=args.train_samples, domain_id=0
        )
        num_domains = 1

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )

    # 创建模型
    model = ImprovedAIGCDetector(
        embed_dim=768,
        num_classes=2,
        num_domains=max(num_domains, 2),
        use_mixstyle=config['use_mixstyle'],
        use_freq_norm=config['use_freq_norm'],
        use_domain_adversarial=config['use_domain_adversarial'],
        use_contrastive=config['use_contrastive'],
        dropout=0.3
    ).to(device)

    # 损失函数
    criterion = CombinedLoss(
        use_focal=True,
        use_domain=config['use_domain_adversarial'],
        use_contrastive=config['use_contrastive'],
        domain_weight=0.2,
        contrastive_weight=0.15
    )

    # 优化器
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练
    best_cross_auc = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        train_loss, train_auc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        scheduler.step()
        print(f"  Epoch {epoch+1}: Loss={train_loss:.4f}, Train AUC={train_auc:.4f}")

    train_time = (time.time() - start_time) / 60

    # 评估
    results = {
        'strategy': strategy_name,
        'config': config,
        'train_time_min': train_time,
        'test_results': {}
    }

    for ds_name, ds_root in dataset_map.items():
        print(f"\n  测试: {ds_name}")

        val_dataset = GenImageDataset(
            ds_root, split='val', transform=val_transform,
            max_samples=args.val_samples, domain_id=0
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        metrics = evaluate(model, val_loader, device)
        results['test_results'][ds_name] = metrics
        print(f"    AUC={metrics['auc']:.4f}, AP={metrics['ap']:.4f}, Acc={metrics['accuracy']:.4f}")

    # 计算综合指标
    aucs = [r['auc'] for r in results['test_results'].values()]
    results['avg_auc'] = np.mean(aucs)
    results['min_auc'] = min(aucs)

    # 跨域性能（假设第一个是训练域，其他是测试域）
    if len(aucs) > 1:
        results['cross_domain_auc'] = np.mean(aucs[1:])
    else:
        results['cross_domain_auc'] = aucs[0]

    return results, model


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='泛化能力自动优化')
    parser.add_argument('--data_roots', type=str, nargs='+', default=[
        'datasets/genimage_partial/imagenet_ai_0419_biggan',
        'datasets/genimage_partial/imagenet_ai_0508_adm'
    ])
    parser.add_argument('--datasets', type=str, nargs='+', default=['BigGAN', 'ADM'])
    parser.add_argument('--train_samples', type=int, default=10000)
    parser.add_argument('--val_samples', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default='outputs/generalization_optimization')
    parser.add_argument('--strategies', type=str, nargs='+', default=None,
                       help='指定要运行的策略，默认运行所有')
    args = parser.parse_args()

    # 设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 验证数据集
    dataset_map = {}
    for name, root in zip(args.datasets, args.data_roots):
        if Path(root).exists():
            dataset_map[name] = root
            print(f"数据集 {name}: {root} ✓")

    if len(dataset_map) < 2:
        print("错误: 需要至少2个数据集")
        return

    # 选择策略
    if args.strategies:
        strategies_to_run = {k: v for k, v in STRATEGIES.items() if k in args.strategies}
    else:
        strategies_to_run = STRATEGIES

    # 运行所有策略
    all_results = []
    best_strategy = None
    best_cross_auc = 0

    for strategy_name, config in strategies_to_run.items():
        try:
            results, model = run_strategy(strategy_name, config, args, device, dataset_map)
            all_results.append(results)

            # 更新最佳策略
            if results['cross_domain_auc'] > best_cross_auc:
                best_cross_auc = results['cross_domain_auc']
                best_strategy = strategy_name
                # 保存最佳模型
                torch.save(model.state_dict(), f"{args.output_dir}/best_model.pth")

            # 保存中间结果
            with open(f"{args.output_dir}/results_{strategy_name}.json", 'w') as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            print(f"策略 {strategy_name} 失败: {e}")
            import traceback
            traceback.print_exc()

    # 结果汇总
    print("\n" + "="*90)
    print("泛化优化实验结果汇总")
    print("="*90)

    # 排序
    all_results.sort(key=lambda x: x['cross_domain_auc'], reverse=True)

    print(f"\n{'策略':<20} {'平均AUC':>10} {'跨域AUC':>10} {'最小AUC':>10} {'训练时间':>10}")
    print("-"*70)

    for r in all_results:
        marker = " ★" if r['strategy'] == best_strategy else ""
        print(f"{r['strategy']:<20} {r['avg_auc']:>10.4f} {r['cross_domain_auc']:>10.4f} "
              f"{r['min_auc']:>10.4f} {r['train_time_min']:>9.1f}m{marker}")

    print(f"\n最佳策略: {best_strategy} (跨域AUC: {best_cross_auc:.4f})")

    # 保存完整结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_output = {
        'timestamp': timestamp,
        'config': vars(args),
        'results': all_results,
        'best_strategy': best_strategy,
        'best_cross_domain_auc': best_cross_auc
    }

    with open(f"{args.output_dir}/final_results_{timestamp}.json", 'w') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {args.output_dir}/")

    return best_strategy, best_cross_auc


if __name__ == '__main__':
    main()
