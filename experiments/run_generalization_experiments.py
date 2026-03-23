"""
泛化增强实验脚本
测试四种泛化改进技术的效果:
1. 域对抗训练 (Domain Adversarial Training)
2. 增强数据增强
3. 域不变特征正则化 (MMD/CORAL)
4. 频率域归一化

实验设计:
- 基线: 原始ours方法
- 消融实验: 每种改进单独测试
- 完整方法: 所有改进组合
- 跨数据集泛化测试
"""

import os
import sys
import json
import time
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import numpy as np
from tqdm import tqdm
from PIL import Image
import io

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.detector_generalization import (
    AIGCDetectorGeneralizedLite,
    create_generalized_model
)
from models.losses.losses import FocalLoss
from models.modules.domain_adaptation import DomainAlignmentLoss
from data.augmentations import get_robust_transforms, MixUp, CutMix
from utils.metrics import MetricsAccumulator


class MultiDomainDataset(Dataset):
    """
    多域数据集
    支持多个数据源并提供域标签
    """
    def __init__(
        self,
        data_roots: Dict[str, str],
        split: str = 'train',
        transform=None,
        max_samples_per_domain: int = None,
        balance_classes: bool = True
    ):
        self.transform = transform
        self.split = split  # 保存split用于_load_domain
        self.samples = []

        # 为每个域分配ID
        self.domain_names = list(data_roots.keys())
        self.domain_to_id = {name: i for i, name in enumerate(self.domain_names)}

        for domain_name, root in data_roots.items():
            domain_id = self.domain_to_id[domain_name]
            domain_samples = self._load_domain(root, domain_id, domain_name, split)

            # 平衡类别 (始终平衡以确保正负样本均衡)
            if balance_classes:
                domain_samples = self._balance_samples(domain_samples)

            # 限制样本数 - 先打乱再截断，避免只截到一个类别
            if max_samples_per_domain and len(domain_samples) > max_samples_per_domain:
                random.shuffle(domain_samples)
                domain_samples = domain_samples[:max_samples_per_domain]

            self.samples.extend(domain_samples)
            real_count = sum(1 for s in domain_samples if s['label'] == 0)
            fake_count = sum(1 for s in domain_samples if s['label'] == 1)
            print(f"  [{domain_name}] Loaded {len(domain_samples)} samples (real={real_count}, fake={fake_count})")

        # 打乱数据
        random.shuffle(self.samples)
        print(f"Total samples: {len(self.samples)}")

    def _load_domain(self, root: str, domain_id: int, domain_name: str, split: str = 'train') -> List[Dict]:
        """加载单个域的数据"""
        samples = []
        root_path = Path(root)

        # 检测目录结构 - 支持 train/val 子目录结构
        ai_dirs = ['ai', 'fake', '1_fake', 'ai_0508_adm', 'ai_0419_biggan']
        nature_dirs = ['nature', 'real', '0_real']

        # 根据split确定子目录优先级
        # train split: 优先train, 其次val(用于train数据不足时), 最后根目录
        # val/test split: 优先val, 其次test, 最后train作为fallback
        if split == 'train':
            sub_dirs = ['train', 'val', '']
        else:  # val 或 test
            sub_dirs = ['val', 'test', 'train', '']  # val优先，没有val时使用train

        ai_path = None
        nature_path = None

        # 查找 AI/fake 目录
        for sub in sub_dirs:
            if ai_path:
                break
            for d in ai_dirs:
                if sub:
                    check_path = root_path / sub / d
                else:
                    check_path = root_path / d
                if check_path.exists():
                    ai_path = check_path
                    break

        # 查找 nature/real 目录
        for sub in sub_dirs:
            if nature_path:
                break
            for d in nature_dirs:
                if sub:
                    check_path = root_path / sub / d
                else:
                    check_path = root_path / d
                if check_path.exists():
                    nature_path = check_path
                    break

        # 收集样本
        if ai_path:
            for img_path in ai_path.glob('**/*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.JPEG']:
                    samples.append({
                        'path': str(img_path),
                        'label': 1,  # fake
                        'domain_id': domain_id,
                        'domain_name': domain_name
                    })
            print(f"    Found ai_path: {ai_path}")

        if nature_path:
            for img_path in nature_path.glob('**/*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.JPEG']:
                    samples.append({
                        'path': str(img_path),
                        'label': 0,  # real
                        'domain_id': domain_id,
                        'domain_name': domain_name
                    })
            print(f"    Found nature_path: {nature_path}")

        if not ai_path and not nature_path:
            print(f"    WARNING: No data directories found in {root_path}")
            print(f"    Checked subdirs: {sub_dirs}")
            print(f"    Checked ai patterns: {ai_dirs}")
            print(f"    Checked nature patterns: {nature_dirs}")

        return samples

    def _balance_samples(self, samples: List[Dict]) -> List[Dict]:
        """平衡正负样本"""
        real_samples = [s for s in samples if s['label'] == 0]
        fake_samples = [s for s in samples if s['label'] == 1]

        min_count = min(len(real_samples), len(fake_samples))

        if len(real_samples) > min_count:
            real_samples = random.sample(real_samples, min_count)
        if len(fake_samples) > min_count:
            fake_samples = random.sample(fake_samples, min_count)

        balanced = real_samples + fake_samples
        random.shuffle(balanced)
        return balanced

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            image = Image.open(sample['path']).convert('RGB')
        except Exception as e:
            print(f"Error loading {sample['path']}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': sample['label'],
            'domain_id': sample['domain_id'],
            'domain_name': sample['domain_name']
        }


def train_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    epoch,
    total_epochs,
    use_mixup: bool = False,
    use_cutmix: bool = False,
    alignment_loss_fn=None,
    alignment_weight: float = 0.1
):
    """训练一个epoch"""
    model.train()
    model.set_epoch(epoch, total_epochs)

    total_loss = 0.0
    total_cls_loss = 0.0
    total_domain_loss = 0.0
    total_align_loss = 0.0
    correct = 0
    total = 0

    mixup = MixUp(alpha=0.2) if use_mixup else None
    cutmix = CutMix(alpha=1.0) if use_cutmix else None

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")

    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        domain_ids = batch['domain_id'].to(device)

        # MixUp或CutMix
        mixed = False
        if mixup and random.random() < 0.3:
            images, labels_a, labels_b, lam = mixup(images, labels)
            mixed = True
        elif cutmix and random.random() < 0.3:
            images, labels_a, labels_b, lam = cutmix(images, labels)
            mixed = True

        optimizer.zero_grad()

        # 前向传播
        outputs = model(images, domain_labels=domain_ids)

        # 分类损失
        if mixed:
            cls_loss = lam * criterion(outputs['logits'], labels_a) + (1 - lam) * criterion(outputs['logits'], labels_b)
        else:
            cls_loss = criterion(outputs['logits'], labels)

        total_cls_loss += cls_loss.item()

        # 域对抗损失
        domain_loss = outputs.get('domain_loss', torch.tensor(0.0, device=device))
        total_domain_loss += domain_loss.item()

        # 特征对齐损失（如果有多个域）
        align_loss = torch.tensor(0.0, device=device)
        if alignment_loss_fn is not None:
            features = outputs['features']
            unique_domains = domain_ids.unique()
            if len(unique_domains) >= 2:
                # 计算不同域之间的对齐损失
                for i in range(len(unique_domains) - 1):
                    mask_i = domain_ids == unique_domains[i]
                    mask_j = domain_ids == unique_domains[i + 1]
                    if mask_i.sum() > 0 and mask_j.sum() > 0:
                        align_dict = alignment_loss_fn(
                            features[mask_i],
                            features[mask_j]
                        )
                        align_loss = align_loss + align_dict['alignment_loss']

        total_align_loss += align_loss.item()

        # 总损失
        loss = cls_loss + domain_loss + alignment_weight * align_loss
        total_loss += loss.item()

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计准确率
        _, predicted = outputs['logits'].max(1)
        if mixed:
            # MixUp/CutMix时使用原始标签计算准确率
            total += labels.size(0)
            correct += (lam * (predicted == labels_a).float() + (1 - lam) * (predicted == labels_b).float()).sum().item()
        else:
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cls': f'{cls_loss.item():.4f}',
            'dom': f'{domain_loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return {
        'loss': total_loss / len(train_loader),
        'cls_loss': total_cls_loss / len(train_loader),
        'domain_loss': total_domain_loss / len(train_loader),
        'align_loss': total_align_loss / len(train_loader),
        'accuracy': 100. * correct / total
    }


@torch.no_grad()
def evaluate(
    model,
    test_loader,
    device,
    apply_jpeg: bool = False,
    jpeg_quality: int = 75
):
    """评估模型"""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    for batch in tqdm(test_loader, desc="Evaluating"):
        images = batch['image'].to(device)
        labels = batch['label']

        # JPEG压缩测试
        if apply_jpeg:
            # 在CPU上处理JPEG压缩
            images_np = images.cpu().numpy()
            compressed = []
            for img in images_np:
                img = (img.transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
                img = img.clip(0, 255).astype(np.uint8)
                pil_img = Image.fromarray(img)
                buffer = io.BytesIO()
                pil_img.save(buffer, format='JPEG', quality=jpeg_quality)
                buffer.seek(0)
                compressed_img = Image.open(buffer).convert('RGB')
                compressed_np = np.array(compressed_img).astype(np.float32) / 255.0
                compressed_np = (compressed_np - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                compressed.append(compressed_np.transpose(2, 0, 1))
            images = torch.tensor(np.stack(compressed)).float().to(device)

        outputs = model(images, compute_domain_loss=False)
        probs = torch.softmax(outputs['logits'], dim=1)[:, 1]

        all_probs.extend(probs.cpu().numpy())
        _, predicted = outputs['logits'].max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

    # 计算指标
    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # 检查类别分布
    unique_labels = np.unique(all_labels)
    print(f"    Labels distribution: {dict(zip(*np.unique(all_labels, return_counts=True)))}")

    # 如果只有一个类别，AUC 无法计算
    if len(unique_labels) < 2:
        print(f"    WARNING: Only {len(unique_labels)} class(es) present, cannot compute AUC")
        auc = 0.5  # 返回随机猜测的AUC
        ap = 0.5
    else:
        auc = roc_auc_score(all_labels, all_probs)
        ap = average_precision_score(all_labels, all_probs)

    acc = accuracy_score(all_labels, all_preds)

    return {
        'auc': auc,
        'ap': ap,
        'accuracy': acc
    }


def run_ablation_experiment(
    config: Dict,
    data_roots: Dict[str, str],
    output_dir: str,
    device: str = 'cuda'
):
    """
    运行消融实验

    Args:
        config: 模型配置
        data_roots: {域名: 数据路径}
        output_dir: 输出目录
        device: 设备
    """
    print(f"\n{'='*60}")
    print(f"实验配置: {config['name']}")
    print(f"{'='*60}")

    # 创建模型
    model = AIGCDetectorGeneralizedLite(
        num_classes=2,
        img_size=224,
        embed_dim=768,
        num_prototypes=4,
        use_domain_adversarial=config.get('use_domain_adversarial', False),
        num_domains=len(data_roots),
        use_freq_norm=config.get('use_freq_norm', False),
        use_style_norm=config.get('use_style_norm', False),
        domain_adv_weight=config.get('domain_adv_weight', 0.1)
    ).to(device)

    # 打印参数量
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量: {params/1e6:.2f}M")

    # 数据增强
    aug_level = config.get('augmentation_level', 'weak')
    train_transform = get_robust_transforms(224, 'train', aug_level)
    test_transform = get_robust_transforms(224, 'test')

    # 创建数据集
    train_dataset = MultiDomainDataset(
        data_roots=data_roots,
        split='train',
        transform=train_transform,
        max_samples_per_domain=config.get('train_samples', 10000),
        balance_classes=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    # 损失函数
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    # 特征对齐损失
    alignment_loss_fn = None
    if config.get('use_feature_alignment', False):
        alignment_loss_fn = DomainAlignmentLoss(
            use_mmd=True,
            use_coral=True,
            mmd_weight=0.05,
            coral_weight=0.05
        )

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 1e-4)
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.get('epochs', 8),
        eta_min=1e-6
    )

    # 训练
    epochs = config.get('epochs', 8)
    best_metrics = None
    train_history = []

    for epoch in range(epochs):
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            use_mixup=config.get('use_mixup', False),
            use_cutmix=config.get('use_cutmix', False),
            alignment_loss_fn=alignment_loss_fn,
            alignment_weight=config.get('alignment_weight', 0.1)
        )

        train_history.append(train_metrics)
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs}: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.2f}%")

    # 评估
    print("\n评估跨域泛化能力...")
    results = {'config': config, 'train_history': train_history, 'test_results': {}}

    for domain_name, root in data_roots.items():
        print(f"\n测试域: {domain_name}")

        # 创建测试数据集
        test_dataset = MultiDomainDataset(
            data_roots={domain_name: root},
            split='test',
            transform=test_transform,
            max_samples_per_domain=config.get('test_samples', 2000),
            balance_classes=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=4
        )

        # 原始测试
        metrics = evaluate(model, test_loader, device, apply_jpeg=False)
        print(f"  原始: AUC={metrics['auc']:.4f}, AP={metrics['ap']:.4f}, Acc={metrics['accuracy']:.4f}")

        # JPEG压缩测试
        metrics_jpeg = evaluate(model, test_loader, device, apply_jpeg=True, jpeg_quality=75)
        print(f"  JPEG Q75: AUC={metrics_jpeg['auc']:.4f}, AP={metrics_jpeg['ap']:.4f}")

        results['test_results'][domain_name] = {
            'original': metrics,
            'jpeg_q75': metrics_jpeg
        }

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, f"results_{config['name']}.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存到: {result_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description='泛化增强实验')
    parser.add_argument('--data_roots', nargs='+', required=True,
                        help='数据集路径列表')
    parser.add_argument('--domain_names', nargs='+', required=True,
                        help='域名列表')
    parser.add_argument('--output_dir', type=str, default='outputs/generalization',
                        help='输出目录')
    parser.add_argument('--train_samples', type=int, default=10000,
                        help='每个域的训练样本数')
    parser.add_argument('--test_samples', type=int, default=2000,
                        help='每个域的测试样本数')
    parser.add_argument('--epochs', type=int, default=8,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批大小')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')

    args = parser.parse_args()

    # 构建数据根目录字典
    data_roots = dict(zip(args.domain_names, args.data_roots))
    print(f"数据集: {data_roots}")

    # 实验配置列表
    configs = [
        # 1. 基线（无任何改进）
        {
            'name': 'baseline',
            'use_domain_adversarial': False,
            'use_freq_norm': False,
            'use_style_norm': False,
            'use_feature_alignment': False,
            'augmentation_level': 'weak',
            'use_mixup': False,
            'use_cutmix': False,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'train_samples': args.train_samples,
            'test_samples': args.test_samples,
        },

        # 2. 仅域对抗训练
        {
            'name': 'domain_adversarial',
            'use_domain_adversarial': True,
            'use_freq_norm': False,
            'use_style_norm': False,
            'use_feature_alignment': False,
            'augmentation_level': 'weak',
            'use_mixup': False,
            'use_cutmix': False,
            'domain_adv_weight': 0.1,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'train_samples': args.train_samples,
            'test_samples': args.test_samples,
        },

        # 3. 仅增强数据增强
        {
            'name': 'strong_augmentation',
            'use_domain_adversarial': False,
            'use_freq_norm': False,
            'use_style_norm': False,
            'use_feature_alignment': False,
            'augmentation_level': 'strong',
            'use_mixup': True,
            'use_cutmix': True,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'train_samples': args.train_samples,
            'test_samples': args.test_samples,
        },

        # 4. 仅频率归一化
        {
            'name': 'freq_norm',
            'use_domain_adversarial': False,
            'use_freq_norm': True,
            'use_style_norm': False,
            'use_feature_alignment': False,
            'augmentation_level': 'weak',
            'use_mixup': False,
            'use_cutmix': False,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'train_samples': args.train_samples,
            'test_samples': args.test_samples,
        },

        # 5. 仅风格归一化
        {
            'name': 'style_norm',
            'use_domain_adversarial': False,
            'use_freq_norm': False,
            'use_style_norm': True,
            'use_feature_alignment': False,
            'augmentation_level': 'weak',
            'use_mixup': False,
            'use_cutmix': False,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'train_samples': args.train_samples,
            'test_samples': args.test_samples,
        },

        # 6. 仅特征对齐
        {
            'name': 'feature_alignment',
            'use_domain_adversarial': False,
            'use_freq_norm': False,
            'use_style_norm': False,
            'use_feature_alignment': True,
            'augmentation_level': 'weak',
            'use_mixup': False,
            'use_cutmix': False,
            'alignment_weight': 0.1,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'train_samples': args.train_samples,
            'test_samples': args.test_samples,
        },

        # 7. 完整方法（所有改进）
        {
            'name': 'full_generalization',
            'use_domain_adversarial': True,
            'use_freq_norm': True,
            'use_style_norm': True,
            'use_feature_alignment': True,
            'augmentation_level': 'strong',
            'use_mixup': True,
            'use_cutmix': True,
            'domain_adv_weight': 0.1,
            'alignment_weight': 0.1,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'train_samples': args.train_samples,
            'test_samples': args.test_samples,
        },
    ]

    # 运行所有实验
    all_results = {}
    for config in configs:
        try:
            results = run_ablation_experiment(
                config=config,
                data_roots=data_roots,
                output_dir=args.output_dir,
                device=args.device
            )
            all_results[config['name']] = results
        except Exception as e:
            print(f"实验 {config['name']} 失败: {e}")
            import traceback
            traceback.print_exc()

    # 汇总结果
    print("\n" + "="*80)
    print("实验结果汇总")
    print("="*80)

    summary = []
    for name, results in all_results.items():
        test_results = results.get('test_results', {})

        # 计算跨域平均
        cross_domain_aucs = []
        in_domain_aucs = []

        for domain, metrics in test_results.items():
            auc = metrics['original']['auc']
            # 假设第一个域是训练域
            if domain == args.domain_names[0]:
                in_domain_aucs.append(auc)
            else:
                cross_domain_aucs.append(auc)

        avg_cross = np.mean(cross_domain_aucs) if cross_domain_aucs else 0
        avg_in = np.mean(in_domain_aucs) if in_domain_aucs else 0

        summary.append({
            'name': name,
            'in_domain_auc': avg_in,
            'cross_domain_auc': avg_cross,
            'test_results': test_results
        })

        print(f"\n{name}:")
        print(f"  同域 AUC: {avg_in:.4f}")
        print(f"  跨域 AUC: {avg_cross:.4f}")

    # 保存汇总
    summary_file = os.path.join(args.output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n汇总结果已保存到: {summary_file}")


if __name__ == '__main__':
    main()
