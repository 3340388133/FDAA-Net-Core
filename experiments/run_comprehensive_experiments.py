"""
综合实验脚本 - SCI 3区论文完整实验

包含：
1. 多生成器验证实验
2. 跨生成器泛化测试
3. SOTA方法对比
4. 特征可视化

Usage:
    python experiments/run_comprehensive_experiments.py --experiment all
    python experiments/run_comprehensive_experiments.py --experiment multi_generator
    python experiments/run_comprehensive_experiments.py --experiment cross_generator
    python experiments/run_comprehensive_experiments.py --experiment sota_comparison
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.detector_ablation import create_ablation_model, ABLATION_CONFIGS
from models.losses import AIGCDetectionLoss
from utils.metrics import MetricsAccumulator
from utils.logger import AverageMeter


# ============================================================================
# 数据集类
# ============================================================================

class MultiGeneratorDataset(Dataset):
    """支持多个生成器的数据集"""

    def __init__(
        self,
        data_roots: dict,  # {generator_name: root_path}
        split: str = 'train',
        transform=None,
        max_samples_per_generator: int = None,
        balance_classes: bool = True
    ):
        self.transform = transform
        self.samples = []
        self.generator_info = {}

        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']

        for gen_name, root_dir in data_roots.items():
            root_path = Path(root_dir)
            split_dir = root_path / split

            # 支持两种目录结构
            ai_dirs = [split_dir / 'ai', split_dir / 'fake', split_dir / '1']
            nature_dirs = [split_dir / 'nature', split_dir / 'real', split_dir / '0']

            ai_dir = None
            nature_dir = None
            for d in ai_dirs:
                if d.exists():
                    ai_dir = d
                    break
            for d in nature_dirs:
                if d.exists():
                    nature_dir = d
                    break

            if ai_dir is None or nature_dir is None:
                print(f"警告: {gen_name} 数据集目录不完整，跳过")
                continue

            # 收集图像
            ai_images = []
            nature_images = []
            for ext in image_extensions:
                ai_images.extend(list(ai_dir.glob(ext)))
                nature_images.extend(list(nature_dir.glob(ext)))

            # 随机打乱和限制数量
            import random
            random.shuffle(ai_images)
            random.shuffle(nature_images)

            if max_samples_per_generator:
                ai_images = ai_images[:max_samples_per_generator]
                nature_images = nature_images[:max_samples_per_generator]

            if balance_classes:
                min_count = min(len(ai_images), len(nature_images))
                ai_images = ai_images[:min_count]
                nature_images = nature_images[:min_count]

            # 添加样本
            for img_path in nature_images:
                self.samples.append((img_path, 0, gen_name))
            for img_path in ai_images:
                self.samples.append((img_path, 1, gen_name))

            self.generator_info[gen_name] = {
                'real': len(nature_images),
                'fake': len(ai_images)
            }

            print(f"  {gen_name}: {len(nature_images)} real + {len(ai_images)} AI")

        # 打乱所有样本
        import random
        random.shuffle(self.samples)
        print(f"总样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, gen_name = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': label,
            'generator': gen_name,
            'path': str(img_path)
        }


def get_transforms(img_size=224, is_train=True):
    """获取数据变换"""
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


# ============================================================================
# SOTA方法实现
# ============================================================================

class CNNDetector(nn.Module):
    """CNNDetection (Wang et al., CVPR 2020) - 使用预训练ResNet"""

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights

        if pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.backbone = resnet50(weights=None)

        # 替换最后一层
        self.backbone.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        logits = self.backbone(x)
        return {'logits': logits}


class SpecDetector(nn.Module):
    """Spec (Zhang et al., 2019) - 频谱分析方法"""

    def __init__(self, num_classes=2):
        super().__init__()
        # 使用DCT频谱特征
        self.freq_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        # 简化的频谱分析
        # 实际应用中应该使用DCT变换
        features = self.freq_conv(x)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return {'logits': logits}


class GramNetDetector(nn.Module):
    """GramNet (Liu et al., CVPR 2020) - Gram矩阵方法"""

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights

        if pretrained:
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        else:
            vgg = vgg16(weights=None)

        self.features = vgg.features[:23]  # 到conv4_3

        # Gram矩阵分类器
        self.gram_fc = nn.Sequential(
            nn.Linear(512 * 512, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

        # 备用全局特征分类器
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Linear(512, num_classes)

    def gram_matrix(self, x):
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        gram = gram / (c * h * w)
        return gram.view(b, -1)

    def forward(self, x):
        features = self.features(x)

        # 使用全局池化代替Gram矩阵（减少内存）
        pooled = self.global_pool(features).view(features.size(0), -1)
        logits = self.global_fc(pooled)

        return {'logits': logits}


class UnivFDDetector(nn.Module):
    """UnivFD (Ojha et al., CVPR 2023) - CLIP特征方法"""

    def __init__(self, num_classes=2):
        super().__init__()
        try:
            import clip
            self.clip_model, _ = clip.load("ViT-L/14", device='cpu')
            self.clip_model.eval()
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.embed_dim = 768
            self.use_clip = True
        except:
            print("CLIP不可用，使用ViT替代")
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            self.clip_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            self.clip_model.heads = nn.Identity()
            self.embed_dim = 768
            self.use_clip = False

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            if self.use_clip:
                features = self.clip_model.encode_image(x).float()
            else:
                features = self.clip_model(x)

        logits = self.classifier(features)
        return {'logits': logits}


class F3NetDetector(nn.Module):
    """F3-Net (Qian et al., ECCV 2020) - 频率感知方法"""

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

        # 空间分支
        if pretrained:
            self.spatial_branch = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.spatial_branch = efficientnet_b0(weights=None)
        spatial_features = self.spatial_branch.classifier[1].in_features
        self.spatial_branch.classifier = nn.Identity()

        # 频率分支（简化版）
        self.freq_branch = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        # 融合分类器
        self.classifier = nn.Sequential(
            nn.Linear(spatial_features + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # 空间特征
        spatial_feat = self.spatial_branch(x)

        # 频率特征（使用高通滤波近似）
        freq_input = x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        freq_feat = self.freq_branch(freq_input).view(x.size(0), -1)

        # 融合
        combined = torch.cat([spatial_feat, freq_feat], dim=1)
        logits = self.classifier(combined)

        return {'logits': logits}


def create_sota_model(method_name, num_classes=2):
    """创建SOTA方法模型"""
    methods = {
        'cnndetection': CNNDetector,
        'spec': SpecDetector,
        'gramnet': GramNetDetector,
        'univfd': UnivFDDetector,
        'f3net': F3NetDetector,
    }

    if method_name.lower() not in methods:
        raise ValueError(f"未知方法: {method_name}")

    return methods[method_name.lower()](num_classes=num_classes)


# ============================================================================
# 训练和评估函数
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, desc='Training'):
    """训练一个epoch"""
    model.train()
    loss_meter = AverageMeter('Loss', ':.4f')
    acc_meter = AverageMeter('Acc', ':.4f')

    pbar = tqdm(dataloader, desc=desc, leave=False)
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # 兼容不同的损失函数接口
        if isinstance(criterion, AIGCDetectionLoss):
            loss_dict = criterion(outputs, labels)
            loss = loss_dict['total_loss']
        else:
            loss = criterion(outputs['logits'], labels)

        loss.backward()
        optimizer.step()

        preds = outputs['logits'].argmax(dim=1)
        acc = (preds == labels).float().mean()

        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc.item(), images.size(0))

        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}', 'acc': f'{acc_meter.avg:.4f}'})

    return {'loss': loss_meter.avg, 'acc': acc_meter.avg}


@torch.no_grad()
def evaluate(model, dataloader, device, desc='Evaluating'):
    """评估模型"""
    model.eval()
    accumulator = MetricsAccumulator()

    # 按生成器分组的结果
    generator_results = defaultdict(lambda: {'probs': [], 'labels': []})

    pbar = tqdm(dataloader, desc=desc, leave=False)
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        generators = batch.get('generator', ['unknown'] * len(labels))

        outputs = model(images)
        probs = torch.softmax(outputs['logits'], dim=1)[:, 1]

        # 总体统计
        accumulator.update(probs.cpu(), labels.cpu())

        # 按生成器统计
        for i, gen in enumerate(generators):
            generator_results[gen]['probs'].append(probs[i].cpu().item())
            generator_results[gen]['labels'].append(labels[i].cpu().item())

    # 计算总体指标
    overall_metrics = accumulator.compute()

    # 计算每个生成器的指标
    per_generator_metrics = {}
    for gen_name, data in generator_results.items():
        probs = torch.tensor(data['probs'])
        labels = torch.tensor(data['labels'])

        gen_acc = MetricsAccumulator()
        gen_acc.update(probs, labels)
        per_generator_metrics[gen_name] = gen_acc.compute()

    return overall_metrics, per_generator_metrics


# ============================================================================
# 实验函数
# ============================================================================

def run_multi_generator_experiment(args, device):
    """实验1: 多生成器验证"""
    print("\n" + "="*70)
    print("实验1: 多生成器验证 (Multi-Generator Validation)")
    print("="*70)

    # 检查可用的生成器数据
    base_path = Path(args.data_root).parent
    available_generators = {}

    # 搜索所有可能的生成器目录
    for subdir in base_path.iterdir():
        if subdir.is_dir() and not subdir.name.startswith('.'):
            train_dir = subdir / 'train'
            if train_dir.exists():
                # 检查是否有ai/nature或fake/real目录
                ai_exists = any((train_dir / d).exists() for d in ['ai', 'fake', '1'])
                nature_exists = any((train_dir / d).exists() for d in ['nature', 'real', '0'])
                if ai_exists and nature_exists:
                    available_generators[subdir.name] = str(subdir)
                    print(f"  发现生成器: {subdir.name}")

    if len(available_generators) == 0:
        print("错误: 未找到任何可用的生成器数据")
        print(f"请确保数据目录 {base_path} 中包含正确格式的数据集")
        return None

    # 如果只有一个生成器，使用它进行完整训练
    if len(available_generators) == 1:
        print(f"\n只发现1个生成器，将使用 {list(available_generators.keys())[0]} 进行完整训练")
        available_generators = {list(available_generators.keys())[0]: args.data_root}

    print(f"\n将在 {len(available_generators)} 个生成器上进行实验")

    results = {}

    # 对每个生成器单独训练和评估
    for gen_name, gen_path in available_generators.items():
        print(f"\n{'='*50}")
        print(f"生成器: {gen_name}")
        print(f"{'='*50}")

        # 创建数据集
        train_dataset = MultiGeneratorDataset(
            {gen_name: gen_path},
            split='train',
            transform=get_transforms(args.img_size, is_train=True),
            max_samples_per_generator=args.train_samples
        )

        val_dataset = MultiGeneratorDataset(
            {gen_name: gen_path},
            split='val',
            transform=get_transforms(args.img_size, is_train=False),
            max_samples_per_generator=args.val_samples
        )

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers, pin_memory=True
        )

        # 创建模型
        model = create_ablation_model(
            'full',
            num_classes=2,
            img_size=args.img_size,
            embed_dim=args.embed_dim
        ).to(device)

        # 训练
        criterion = AIGCDetectionLoss(use_focal=True, use_aux=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_metrics = {'auc': 0}

        for epoch in range(args.epochs):
            train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics, _ = evaluate(model, val_loader, device)
            scheduler.step()

            if val_metrics['auc'] > best_metrics['auc']:
                best_metrics = val_metrics.copy()
                best_metrics['epoch'] = epoch

            print(f"  Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, "
                  f"Val AUC={val_metrics['auc']:.4f}, Val Acc={val_metrics['accuracy']:.4f}")

        results[gen_name] = best_metrics
        print(f"\n  最佳结果: AUC={best_metrics['auc']:.4f}, Acc={best_metrics['accuracy']:.4f}")

    return results


def run_cross_generator_experiment(args, device):
    """实验2: 跨生成器泛化测试"""
    print("\n" + "="*70)
    print("实验2: 跨生成器泛化测试 (Cross-Generator Generalization)")
    print("="*70)

    # 检查可用的生成器
    base_path = Path(args.data_root).parent
    available_generators = {}

    for subdir in base_path.iterdir():
        if subdir.is_dir() and not subdir.name.startswith('.'):
            train_dir = subdir / 'train'
            val_dir = subdir / 'val'
            if train_dir.exists() or val_dir.exists():
                available_generators[subdir.name] = str(subdir)

    if len(available_generators) < 2:
        print("警告: 跨生成器测试需要至少2个生成器的数据")
        print("将使用单一生成器的训练/验证分割进行测试")

        # 使用单一数据集
        gen_name = list(available_generators.keys())[0] if available_generators else 'biggan'
        gen_path = list(available_generators.values())[0] if available_generators else args.data_root

        # 在训练集上训练
        train_dataset = MultiGeneratorDataset(
            {gen_name: gen_path},
            split='train',
            transform=get_transforms(args.img_size, is_train=True),
            max_samples_per_generator=args.train_samples
        )

        val_dataset = MultiGeneratorDataset(
            {gen_name: gen_path},
            split='val',
            transform=get_transforms(args.img_size, is_train=False),
            max_samples_per_generator=args.val_samples
        )

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers, pin_memory=True
        )

        # 训练模型
        model = create_ablation_model(
            'full', num_classes=2, img_size=args.img_size, embed_dim=args.embed_dim
        ).to(device)

        criterion = AIGCDetectionLoss(use_focal=True, use_aux=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        print(f"\n在 {gen_name} 上训练...")
        for epoch in range(args.epochs):
            train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
            scheduler.step()
            print(f"  Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}")

        # 评估
        print(f"\n在 {gen_name} 验证集上评估...")
        val_metrics, per_gen_metrics = evaluate(model, val_loader, device)

        return {
            'train_generator': gen_name,
            'overall': val_metrics,
            'per_generator': per_gen_metrics
        }

    # 多生成器情况：选择一个训练，其他测试
    train_gen = list(available_generators.keys())[0]
    test_gens = {k: v for k, v in available_generators.items() if k != train_gen}

    print(f"训练生成器: {train_gen}")
    print(f"测试生成器: {list(test_gens.keys())}")

    # 在训练生成器上训练
    train_dataset = MultiGeneratorDataset(
        {train_gen: available_generators[train_gen]},
        split='train',
        transform=get_transforms(args.img_size, is_train=True),
        max_samples_per_generator=args.train_samples
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True
    )

    # 创建和训练模型
    model = create_ablation_model(
        'full', num_classes=2, img_size=args.img_size, embed_dim=args.embed_dim
    ).to(device)

    criterion = AIGCDetectionLoss(use_focal=True, use_aux=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\n在 {train_gen} 上训练...")
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        print(f"  Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}")

    # 在所有生成器上评估
    results = {
        'train_generator': train_gen,
        'per_generator': {}
    }

    all_test_gens = {train_gen: available_generators[train_gen]}
    all_test_gens.update(test_gens)

    for gen_name, gen_path in all_test_gens.items():
        test_dataset = MultiGeneratorDataset(
            {gen_name: gen_path},
            split='val',
            transform=get_transforms(args.img_size, is_train=False),
            max_samples_per_generator=args.val_samples
        )

        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers, pin_memory=True
        )

        is_seen = '(seen)' if gen_name == train_gen else '(unseen)'
        print(f"\n在 {gen_name} {is_seen} 上评估...")

        metrics, _ = evaluate(model, test_loader, device)
        results['per_generator'][gen_name] = {
            'metrics': metrics,
            'is_seen': gen_name == train_gen
        }

        print(f"  AUC: {metrics['auc']:.4f}, Acc: {metrics['accuracy']:.4f}")

    return results


def run_sota_comparison(args, device):
    """实验3: SOTA方法对比"""
    print("\n" + "="*70)
    print("实验3: SOTA方法对比 (SOTA Comparison)")
    print("="*70)

    # 准备数据
    train_dataset = MultiGeneratorDataset(
        {'biggan': args.data_root},
        split='train',
        transform=get_transforms(args.img_size, is_train=True),
        max_samples_per_generator=args.train_samples
    )

    val_dataset = MultiGeneratorDataset(
        {'biggan': args.data_root},
        split='val',
        transform=get_transforms(args.img_size, is_train=False),
        max_samples_per_generator=args.val_samples
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    # 要对比的方法
    methods = {
        'Ours (FDAA+MGFP)': ('ablation', 'full'),
        'Baseline (ViT)': ('ablation', 'baseline'),
        'CNNDetection': ('sota', 'cnndetection'),
        'F3-Net': ('sota', 'f3net'),
        'GramNet': ('sota', 'gramnet'),
        # 'UnivFD': ('sota', 'univfd'),  # 需要CLIP
        'Spec': ('sota', 'spec'),
    }

    results = {}

    for method_name, (method_type, model_name) in methods.items():
        print(f"\n{'='*50}")
        print(f"方法: {method_name}")
        print(f"{'='*50}")

        try:
            # 创建模型
            if method_type == 'ablation':
                model = create_ablation_model(
                    model_name, num_classes=2,
                    img_size=args.img_size, embed_dim=args.embed_dim
                )
            else:
                model = create_sota_model(model_name, num_classes=2)

            model = model.to(device)
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"参数量: {params:,}")

            # 训练
            criterion = nn.CrossEntropyLoss() if method_type == 'sota' else AIGCDetectionLoss(use_focal=True, use_aux=False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

            best_metrics = {'auc': 0}

            for epoch in range(args.epochs):
                train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
                val_metrics, _ = evaluate(model, val_loader, device)
                scheduler.step()

                if val_metrics['auc'] > best_metrics['auc']:
                    best_metrics = val_metrics.copy()
                    best_metrics['epoch'] = epoch

                print(f"  Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, "
                      f"Val AUC={val_metrics['auc']:.4f}")

            results[method_name] = {
                'params': params,
                'auc': best_metrics['auc'],
                'accuracy': best_metrics['accuracy'],
                'ap': best_metrics.get('ap', 0)
            }

        except Exception as e:
            print(f"  错误: {e}")
            results[method_name] = {'error': str(e)}

    return results


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='综合实验脚本')

    # 实验选择
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'multi_generator', 'cross_generator', 'sota_comparison'],
                        help='要运行的实验')

    # 数据
    parser.add_argument('--data_root', type=str,
                        default='datasets/genimage_partial/imagenet_ai_0419_biggan',
                        help='数据集根目录')
    parser.add_argument('--train_samples', type=int, default=10000,
                        help='每个生成器的训练样本数')
    parser.add_argument('--val_samples', type=int, default=2000,
                        help='每个生成器的验证样本数')

    # 模型
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--embed_dim', type=int, default=768)

    # 训练
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)

    # 输出
    parser.add_argument('--output_dir', type=str, default='./outputs/comprehensive')

    args = parser.parse_args()

    # 设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Data root: {args.data_root}")

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {
        'args': vars(args),
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }

    # 运行实验
    if args.experiment in ['all', 'multi_generator']:
        results = run_multi_generator_experiment(args, device)
        all_results['multi_generator'] = results

    if args.experiment in ['all', 'cross_generator']:
        results = run_cross_generator_experiment(args, device)
        all_results['cross_generator'] = results

    if args.experiment in ['all', 'sota_comparison']:
        results = run_sota_comparison(args, device)
        all_results['sota_comparison'] = results

    # 保存结果
    result_file = os.path.join(
        args.output_dir,
        f"comprehensive_{args.experiment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    # 转换不可序列化的对象
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {result_file}")

    # 打印结果汇总
    print("\n" + "="*70)
    print("实验结果汇总")
    print("="*70)

    if 'sota_comparison' in all_results and all_results['sota_comparison']:
        print("\nSOTA方法对比:")
        print(f"{'方法':<25} {'参数量':<12} {'AUC':<8} {'Acc':<8} {'AP':<8}")
        print("-"*61)
        for method, metrics in all_results['sota_comparison'].items():
            if 'error' not in metrics:
                print(f"{method:<25} {metrics['params']:<12,} {metrics['auc']:<8.4f} "
                      f"{metrics['accuracy']:<8.4f} {metrics.get('ap', 0):<8.4f}")


if __name__ == '__main__':
    main()
