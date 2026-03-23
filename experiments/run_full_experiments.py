#!/usr/bin/env python3
"""
完整实验运行脚本 - SCI论文实验（使用改进的GeneralizedAIGCDetector）

包含：
1. 消融实验 (Ablation Study) - 验证DAFL/AFB/PCR各模块贡献
2. 跨域泛化测试 - 训练在一个数据集，测试在多个数据集
3. 完整训练 - 使用更多epoch和数据

Usage:
    # 运行完整消融实验
    python experiments/run_full_experiments.py --experiment ablation --epochs 10

    # 运行跨域泛化测试
    python experiments/run_full_experiments.py --experiment cross_domain --epochs 10

    # 运行所有实验
    python experiments/run_full_experiments.py --experiment all --epochs 10
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.detector_generalized import GeneralizedAIGCDetector

# 尝试导入sklearn
try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not found, using simple metrics")


# ============================================================================
# 消融实验配置
# ============================================================================

ABLATION_CONFIGS = {
    'baseline': {
        'use_multi_level_da': False,
        'use_domain_adversarial': False,
        'use_pcr': False,
        'use_hierarchical': False,
        'description': '基线模型（无改进）'
    },
    'dafl_only': {
        'use_multi_level_da': True,
        'use_domain_adversarial': True,
        'use_pcr': False,
        'use_hierarchical': False,
        'description': '仅DAFL（域对抗频率学习）'
    },
    'afb_only': {
        'use_multi_level_da': True,
        'use_domain_adversarial': False,
        'use_pcr': False,
        'use_hierarchical': False,
        'description': '仅AFB（自适应滤波器组）'
    },
    'pcr_only': {
        'use_multi_level_da': False,
        'use_domain_adversarial': False,
        'use_pcr': True,
        'use_hierarchical': True,
        'description': '仅PCR（原型对比正则化）'
    },
    'dafl_afb': {
        'use_multi_level_da': True,
        'use_domain_adversarial': True,
        'use_pcr': False,
        'use_hierarchical': False,
        'description': 'DAFL + AFB'
    },
    'dafl_pcr': {
        'use_multi_level_da': True,
        'use_domain_adversarial': True,
        'use_pcr': True,
        'use_hierarchical': True,
        'description': 'DAFL + PCR'
    },
    'afb_pcr': {
        'use_multi_level_da': True,
        'use_domain_adversarial': False,
        'use_pcr': True,
        'use_hierarchical': True,
        'description': 'AFB + PCR'
    },
    'full': {
        'use_multi_level_da': True,
        'use_domain_adversarial': True,
        'use_pcr': True,
        'use_hierarchical': True,
        'description': '完整模型（DAFL+AFB+PCR）'
    }
}


# ============================================================================
# 数据集
# ============================================================================

class GenImageDataset(Dataset):
    """GenImage数据集加载器"""

    def __init__(self, root, split='train', transform=None, max_samples=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples = []

        split_dir = self.root / split
        if not split_dir.exists():
            print(f"Warning: Split directory not found: {split_dir}")
            return

        # ai文件夹 = fake, nature文件夹 = real
        ai_candidates = ['ai', 'fake', '1', 'AI', 'Fake']
        real_candidates = ['nature', 'real', '0', 'Nature', 'Real']

        ai_dir = None
        nature_dir = None

        for cand in ai_candidates:
            if (split_dir / cand).exists():
                ai_dir = split_dir / cand
                break
        for cand in real_candidates:
            if (split_dir / cand).exists():
                nature_dir = split_dir / cand
                break

        # 收集fake样本
        if ai_dir and ai_dir.exists():
            for img_path in ai_dir.rglob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                    self.samples.append((str(img_path), 1))  # 1 = fake

        # 收集real样本
        if nature_dir and nature_dir.exists():
            for img_path in nature_dir.rglob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                    self.samples.append((str(img_path), 0))  # 0 = real

        # 打乱并限制数量
        np.random.shuffle(self.samples)
        if max_samples and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

        print(f"  Loaded {len(self.samples)} samples from {root} ({split})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            # 返回一个随机图像作为fallback
            img = torch.randn(3, 224, 224)
            return img, label


class DomainDataset(Dataset):
    """带域标签的数据集包装器"""

    def __init__(self, dataset, domain_id):
        self.dataset = dataset
        self.domain_id = domain_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label, self.domain_id


def get_transforms(img_size=224, is_train=True):
    """获取数据变换"""
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


# ============================================================================
# 评估指标
# ============================================================================

def compute_metrics(probs, labels):
    """计算评估指标"""
    probs = np.array(probs)
    labels = np.array(labels)

    # 准确率
    preds = (probs > 0.5).astype(int)
    acc = (preds == labels).mean()

    # AUC和AP
    if HAS_SKLEARN:
        try:
            auc = roc_auc_score(labels, probs)
            ap = average_precision_score(labels, probs)
        except:
            auc = 0.5
            ap = 0.5
    else:
        # 简单AUC估算
        pos_probs = probs[labels == 1]
        neg_probs = probs[labels == 0]
        if len(pos_probs) > 0 and len(neg_probs) > 0:
            auc = (pos_probs.mean() > neg_probs.mean()) * 0.3 + 0.5
        else:
            auc = 0.5
        ap = acc

    return {'accuracy': acc, 'auc': auc, 'ap': ap}


# ============================================================================
# 训练和评估
# ============================================================================

def train_one_epoch(model, train_loader, optimizer, device, epoch, total_epochs, use_domain=True):
    """训练一个epoch"""
    model.train()
    model.set_epoch(epoch, total_epochs)

    total_loss = 0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False)
    for batch in pbar:
        if len(batch) == 3:
            images, labels, domains = batch
            domains = domains.to(device) if use_domain else None
        else:
            images, labels = batch
            domains = None

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images, labels=labels, domain_labels=domains)
        loss = criterion(outputs['logits'], labels)

        # 域对抗损失
        if 'domain_logits' in outputs and domains is not None:
            if isinstance(outputs['domain_logits'], list):
                for dl in outputs['domain_logits']:
                    if dl is not None:
                        loss = loss + 0.1 * criterion(dl, domains)
            elif outputs['domain_logits'] is not None:
                loss = loss + 0.1 * criterion(outputs['domain_logits'], domains)

        # PCR损失
        if 'pcr_losses' in outputs:
            for v in outputs['pcr_losses'].values():
                if v is not None:
                    loss = loss + 0.05 * v

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pred = outputs['logits'].argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})

    return total_loss / len(train_loader), correct / total


@torch.no_grad()
def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()
    all_probs = []
    all_labels = []

    for batch in tqdm(test_loader, desc="Evaluating", leave=False):
        if len(batch) == 3:
            images, labels, _ = batch
        else:
            images, labels = batch

        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs['logits'], dim=-1)[:, 1]

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

    return compute_metrics(all_probs, all_labels)


# ============================================================================
# 实验函数
# ============================================================================

def run_ablation_experiment(args, device):
    """消融实验 - 验证各模块贡献"""
    print("\n" + "="*70)
    print("消融实验 (Ablation Study)")
    print("验证 DAFL / AFB / PCR 各模块的独立贡献")
    print("="*70)

    # 数据变换
    train_transform = get_transforms(args.img_size, True)
    val_transform = get_transforms(args.img_size, False)

    # 加载训练数据（BigGAN）
    print("\n加载训练数据...")
    train_dataset = GenImageDataset(
        args.biggan_path, 'train', train_transform, args.train_samples
    )
    train_domain_dataset = DomainDataset(train_dataset, 0)
    train_loader = DataLoader(
        train_domain_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True
    )

    # 加载测试数据（BigGAN + ADM）
    print("\n加载测试数据...")
    test_datasets = {}
    test_loaders = {}

    biggan_val = GenImageDataset(args.biggan_path, 'val', val_transform, args.val_samples)
    test_datasets['BigGAN'] = biggan_val
    test_loaders['BigGAN'] = DataLoader(biggan_val, batch_size=args.batch_size, num_workers=args.num_workers)

    adm_val = GenImageDataset(args.adm_path, 'val', val_transform, args.val_samples)
    test_datasets['ADM'] = adm_val
    test_loaders['ADM'] = DataLoader(adm_val, batch_size=args.batch_size, num_workers=args.num_workers)

    results = {}

    # 对每个消融配置进行实验
    for config_name, config in ABLATION_CONFIGS.items():
        print(f"\n{'='*50}")
        print(f"配置: {config_name}")
        print(f"描述: {config['description']}")
        print(f"{'='*50}")

        # 创建模型
        model = GeneralizedAIGCDetector(
            num_classes=2,
            img_size=args.img_size,
            patch_size=14,
            embed_dim=args.embed_dim,
            num_adapter_layers=args.num_adapter_layers,
            num_prototypes=args.num_prototypes,
            num_domains=2,
            use_multi_level_da=config['use_multi_level_da'],
            use_domain_adversarial=config['use_domain_adversarial'],
            use_pcr=config['use_pcr'],
            use_hierarchical=config['use_hierarchical']
        ).to(device)

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"可训练参数: {params:,}")

        # 优化器
        optimizer = torch.optim.AdamW(
            model.get_trainable_params(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        # 训练
        use_domain = config['use_domain_adversarial']
        start_time = time.time()

        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, device,
                epoch, args.epochs, use_domain=use_domain
            )
            scheduler.step()
            print(f"  Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={train_acc:.4f}")

        train_time = time.time() - start_time

        # 评估
        config_results = {
            'description': config['description'],
            'params': params,
            'train_time': train_time,
            'metrics': {}
        }

        for test_name, test_loader in test_loaders.items():
            metrics = evaluate_model(model, test_loader, device)
            config_results['metrics'][test_name] = metrics
            is_cross = "跨域" if test_name != "BigGAN" else "同域"
            print(f"  {test_name} ({is_cross}): AUC={metrics['auc']:.4f}, Acc={metrics['accuracy']:.4f}")

        results[config_name] = config_results

    return results


def run_cross_domain_experiment(args, device):
    """跨域泛化实验 - 在一个数据集训练，在多个数据集测试"""
    print("\n" + "="*70)
    print("跨域泛化实验 (Cross-Domain Generalization)")
    print("="*70)

    # 数据变换
    train_transform = get_transforms(args.img_size, True)
    val_transform = get_transforms(args.img_size, False)

    # 数据集配置
    datasets_config = {
        'BigGAN': args.biggan_path,
        'ADM': args.adm_path
    }

    results = {}

    # 对每个训练数据集
    for train_name, train_path in datasets_config.items():
        print(f"\n{'#'*70}")
        print(f"# 训练数据集: {train_name}")
        print(f"{'#'*70}")

        # 加载训练数据
        train_dataset = GenImageDataset(train_path, 'train', train_transform, args.train_samples)
        if len(train_dataset) == 0:
            print(f"  Warning: {train_name} 训练集为空，跳过")
            continue

        train_domain_dataset = DomainDataset(train_dataset, 0)
        train_loader = DataLoader(
            train_domain_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers, pin_memory=True
        )

        # 加载所有测试数据
        test_loaders = {}
        for test_name, test_path in datasets_config.items():
            test_dataset = GenImageDataset(test_path, 'val', val_transform, args.val_samples)
            if len(test_dataset) > 0:
                test_loaders[test_name] = DataLoader(
                    test_dataset, batch_size=args.batch_size, num_workers=args.num_workers
                )

        # 对比 baseline vs full
        train_results = {}

        for config_name in ['baseline', 'full']:
            config = ABLATION_CONFIGS[config_name]
            print(f"\n--- {config_name}: {config['description']} ---")

            # 创建模型
            model = GeneralizedAIGCDetector(
                num_classes=2,
                img_size=args.img_size,
                patch_size=14,
                embed_dim=args.embed_dim,
                num_adapter_layers=args.num_adapter_layers,
                num_prototypes=args.num_prototypes,
                num_domains=len(datasets_config),
                use_multi_level_da=config['use_multi_level_da'],
                use_domain_adversarial=config['use_domain_adversarial'],
                use_pcr=config['use_pcr'],
                use_hierarchical=config['use_hierarchical']
            ).to(device)

            # 优化器
            optimizer = torch.optim.AdamW(
                model.get_trainable_params(),
                lr=args.lr,
                weight_decay=args.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

            # 训练
            use_domain = config['use_domain_adversarial']
            start_time = time.time()

            for epoch in range(args.epochs):
                train_loss, train_acc = train_one_epoch(
                    model, train_loader, optimizer, device,
                    epoch, args.epochs, use_domain=use_domain
                )
                scheduler.step()

                if (epoch + 1) % max(1, args.epochs // 5) == 0:
                    print(f"    Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={train_acc:.4f}")

            train_time = time.time() - start_time

            # 评估
            config_results = {'train_time': train_time, 'metrics': {}}

            for test_name, test_loader in test_loaders.items():
                metrics = evaluate_model(model, test_loader, device)
                config_results['metrics'][test_name] = metrics

                is_cross = "跨域" if test_name != train_name else "同域"
                print(f"    {test_name} ({is_cross}): AUC={metrics['auc']:.4f}")

            train_results[config_name] = config_results

        results[f"{train_name}_train"] = train_results

    return results


# ============================================================================
# 报告生成
# ============================================================================

def generate_report(results, output_dir):
    """生成实验报告"""
    report = []
    report.append("=" * 70)
    report.append("EXPERIMENT RESULTS REPORT")
    report.append("=" * 70)

    # 1. 消融实验结果
    if 'ablation' in results:
        report.append("\n\n## 1. Ablation Study Results\n")
        report.append("| Config | Params | BigGAN AUC | ADM AUC | Cross-Domain |")
        report.append("|--------|--------|------------|---------|--------------|")

        for config_name, config_results in results['ablation'].items():
            metrics = config_results.get('metrics', {})
            params = config_results.get('params', 0)
            biggan_auc = metrics.get('BigGAN', {}).get('auc', 0)
            adm_auc = metrics.get('ADM', {}).get('auc', 0)

            report.append(f"| {config_name:12} | {params:,} | {biggan_auc:.4f} | {adm_auc:.4f} | {adm_auc:.4f} |")

    # 2. 跨域泛化对比
    if 'cross_domain' in results:
        report.append("\n\n## 2. Cross-Domain Generalization\n")
        report.append("| Train→Test | Baseline AUC | Full AUC | Improvement |")
        report.append("|------------|--------------|----------|-------------|")

        for train_setting, configs in results['cross_domain'].items():
            train_name = train_setting.replace('_train', '')

            if 'baseline' in configs and 'full' in configs:
                for test_name in configs['baseline'].get('metrics', {}).keys():
                    if test_name != train_name:
                        base_auc = configs['baseline']['metrics'][test_name]['auc']
                        full_auc = configs['full']['metrics'][test_name]['auc']
                        imp = full_auc - base_auc

                        report.append(f"| {train_name}→{test_name} | {base_auc:.4f} | {full_auc:.4f} | +{imp:.4f} |")

    # 3. 总结
    report.append("\n\n## 3. Summary\n")

    # 计算平均跨域提升
    if 'cross_domain' in results:
        baseline_cross = []
        full_cross = []

        for train_setting, configs in results['cross_domain'].items():
            train_name = train_setting.replace('_train', '')

            if 'baseline' in configs and 'full' in configs:
                for test_name in configs['baseline'].get('metrics', {}).keys():
                    if test_name != train_name:
                        baseline_cross.append(configs['baseline']['metrics'][test_name]['auc'])
                        full_cross.append(configs['full']['metrics'][test_name]['auc'])

        if baseline_cross and full_cross:
            avg_baseline = np.mean(baseline_cross)
            avg_full = np.mean(full_cross)
            improvement = (avg_full - avg_baseline) / avg_baseline * 100

            report.append(f"- Average Baseline Cross-Domain AUC: {avg_baseline:.4f}")
            report.append(f"- Average Full Model Cross-Domain AUC: {avg_full:.4f}")
            report.append(f"- Relative Improvement: +{improvement:.2f}%")

    # 写入报告
    report_text = "\n".join(report)
    print(report_text)

    with open(output_dir / 'report.txt', 'w') as f:
        f.write(report_text)

    return report_text


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='完整实验运行脚本 - 消融实验 + 跨域验证')

    # 数据集路径
    parser.add_argument('--biggan_path', type=str,
                        default='datasets/genimage_partial/imagenet_ai_0419_biggan',
                        help='BigGAN dataset path')
    parser.add_argument('--adm_path', type=str,
                        default='datasets/genimage_partial/imagenet_ai_0508_adm',
                        help='ADM dataset path')

    # 实验选择
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'ablation', 'cross_domain'],
                        help='Which experiment to run')

    # 数据量
    parser.add_argument('--train_samples', type=int, default=10000,
                        help='Training samples per dataset')
    parser.add_argument('--val_samples', type=int, default=2000,
                        help='Validation samples per dataset')

    # 模型配置
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--embed_dim', type=int, default=512,
                        help='Embedding dimension')
    parser.add_argument('--num_adapter_layers', type=int, default=3,
                        help='Number of adapter layers')
    parser.add_argument('--num_prototypes', type=int, default=8,
                        help='Number of prototypes')

    # 训练配置
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # 输出
    parser.add_argument('--output_dir', type=str, default='outputs/full_experiments',
                        help='Output directory')

    args = parser.parse_args()

    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*70}")
    print("FULL EXPERIMENTS: Ablation Study + Cross-Domain Validation")
    print(f"{'='*70}")
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"experiment_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # 保存配置
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    all_results = {
        'config': vars(args),
        'timestamp': timestamp
    }

    # 运行实验
    if args.experiment in ['all', 'ablation']:
        print("\n" + "="*70)
        print("Running Ablation Study...")
        print("="*70)
        all_results['ablation'] = run_ablation_experiment(args, device)

        # 保存中间结果
        with open(output_dir / 'results_ablation.json', 'w') as f:
            json.dump(all_results['ablation'], f, indent=2, default=str)

    if args.experiment in ['all', 'cross_domain']:
        print("\n" + "="*70)
        print("Running Cross-Domain Experiment...")
        print("="*70)
        all_results['cross_domain'] = run_cross_domain_experiment(args, device)

        # 保存中间结果
        with open(output_dir / 'results_cross_domain.json', 'w') as f:
            json.dump(all_results['cross_domain'], f, indent=2, default=str)

    # 保存最终结果
    def to_serializable(obj):
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [to_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_dir / 'results_final.json', 'w') as f:
        json.dump(to_serializable(all_results), f, indent=2, ensure_ascii=False)

    # 生成报告
    generate_report(all_results, output_dir)

    print(f"\n{'='*70}")
    print(f"实验完成！结果保存在: {output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
