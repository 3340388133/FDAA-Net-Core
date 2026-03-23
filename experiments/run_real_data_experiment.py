#!/usr/bin/env python
"""
真实数据集跨域泛化实验

实验设计：
1. 在BigGAN上训练，测试BigGAN和ADM（跨域）
2. 在ADM上训练，测试ADM和BigGAN（跨域）
3. 对比Baseline vs Generalized模型

数据集位置：
- BigGAN: datasets/genimage_partial/imagenet_ai_0419_biggan/
- ADM: datasets/genimage_partial/imagenet_ai_0508_adm/
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detector_generalized import GeneralizedAIGCDetector


class GenImageDataset(Dataset):
    """GenImage数据集加载器"""

    def __init__(self, root, split='train', transform=None, max_samples=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        # 查找图片
        self.samples = []

        split_dir = self.root / split
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")

        # ai文件夹是fake，nature文件夹是real
        ai_dir = split_dir / 'ai'
        nature_dir = split_dir / 'nature'

        # 收集fake图片
        if ai_dir.exists():
            for img_path in ai_dir.rglob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                    self.samples.append((str(img_path), 1))  # 1 = fake

        # 收集real图片
        if nature_dir.exists():
            for img_path in nature_dir.rglob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                    self.samples.append((str(img_path), 0))  # 0 = real

        # 限制样本数
        if max_samples and len(self.samples) > max_samples:
            np.random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        print(f"Loaded {len(self.samples)} samples from {root} ({split})")

        # 统计类别分布
        fake_count = sum(1 for _, l in self.samples if l == 1)
        real_count = len(self.samples) - fake_count
        print(f"  Real: {real_count}, Fake: {fake_count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # 返回一个随机噪声图像
            image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

        if self.transform:
            image = self.transform(image)

        return image, label


class MultiDomainDataset(Dataset):
    """多域数据集"""

    def __init__(self, datasets, domain_ids):
        self.datasets = datasets
        self.domain_ids = domain_ids

        # 计算累积大小
        self.cumulative_sizes = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative_sizes.append(total)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        # 找到对应的数据集
        dataset_idx = 0
        for i, size in enumerate(self.cumulative_sizes):
            if idx < size:
                dataset_idx = i
                break

        # 计算本地索引
        if dataset_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        image, label = self.datasets[dataset_idx][local_idx]
        domain = self.domain_ids[dataset_idx]

        return image, label, domain


def get_transforms(img_size=224, is_train=True):
    """获取数据变换"""
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, total_epochs):
    """训练一个epoch"""
    model.train()
    model.set_epoch(epoch, total_epochs)

    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{total_epochs}')

    for batch_idx, batch in enumerate(pbar):
        if len(batch) == 3:
            images, labels, domains = batch
            domains = domains.to(device)
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
            domain_criterion = nn.CrossEntropyLoss()
            if isinstance(outputs['domain_logits'], list):
                for dl in outputs['domain_logits']:
                    loss = loss + 0.1 * domain_criterion(dl, domains)
            else:
                loss = loss + 0.1 * domain_criterion(outputs['domain_logits'], domains)

        # PCR损失
        if 'pcr_losses' in outputs:
            for v in outputs['pcr_losses'].values():
                loss = loss + 0.05 * v

        # AFB正则化
        if 'afb_reg_losses' in outputs:
            for k, v in outputs['afb_reg_losses'].items():
                if 'total' in k:
                    loss = loss + v

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pred = outputs['logits'].argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            'loss': f'{total_loss/(batch_idx+1):.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })

    return total_loss / len(train_loader), correct / total


@torch.no_grad()
def evaluate(model, test_loader, device, desc='Evaluating'):
    """评估模型"""
    model.eval()

    all_probs = []
    all_labels = []

    for batch in tqdm(test_loader, desc=desc):
        if len(batch) == 3:
            images, labels, _ = batch
        else:
            images, labels = batch

        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs['logits'], dim=-1)[:, 1]

        all_probs.append(probs.cpu())
        all_labels.append(labels)

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    # 计算指标
    from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score

    preds = (probs > 0.5).astype(int)
    accuracy = accuracy_score(labels, preds)

    try:
        auc = roc_auc_score(labels, probs)
        ap = average_precision_score(labels, probs)
    except:
        auc = 0.5
        ap = 0.5

    return {
        'accuracy': float(accuracy),
        'auc': float(auc),
        'ap': float(ap)
    }


def run_experiment(args):
    """运行实验"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 数据路径
    data_root = Path(args.data_root)
    biggan_path = data_root / 'imagenet_ai_0419_biggan'
    adm_path = data_root / 'imagenet_ai_0508_adm'

    # 数据变换
    train_transform = get_transforms(args.img_size, is_train=True)
    test_transform = get_transforms(args.img_size, is_train=False)

    results = {}

    # =============================================
    # 实验配置
    # =============================================
    experiments = [
        {
            'name': 'BigGAN_train',
            'train_path': biggan_path,
            'test_paths': {
                'BigGAN': biggan_path,
                'ADM': adm_path
            }
        },
        {
            'name': 'ADM_train',
            'train_path': adm_path,
            'test_paths': {
                'ADM': adm_path,
                'BigGAN': biggan_path
            }
        }
    ]

    # =============================================
    # 对比两种模型
    # =============================================
    model_configs = {
        'baseline': {
            'use_multi_level_da': False,
            'use_domain_adversarial': False,
            'use_pcr': False,
            'use_hierarchical': False
        },
        'generalized': {
            'use_multi_level_da': True,
            'use_domain_adversarial': True,
            'use_pcr': True,
            'use_hierarchical': True
        }
    }

    for exp in experiments:
        exp_name = exp['name']
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {exp_name}")
        print(f"{'='*60}")

        results[exp_name] = {}

        for model_name, model_config in model_configs.items():
            print(f"\n--- Model: {model_name} ---")

            # 创建模型
            model = GeneralizedAIGCDetector(
                num_classes=2,
                img_size=args.img_size,
                patch_size=14,
                embed_dim=args.embed_dim,
                num_adapter_layers=args.num_layers,
                num_prototypes=4,
                num_domains=2,
                dropout=0.1,
                **model_config
            ).to(device)

            params = model.count_parameters()
            print(f"  Trainable parameters: {params['trainable']:,}")

            # 创建训练数据
            print(f"\n  Loading training data from {exp['train_path']}...")
            train_dataset = GenImageDataset(
                exp['train_path'],
                split='train',
                transform=train_transform,
                max_samples=args.train_samples
            )

            # 如果需要多域训练
            if model_config.get('use_domain_adversarial', False):
                # 同时加载另一个域的少量数据
                other_path = adm_path if 'BigGAN' in exp_name else biggan_path
                other_dataset = GenImageDataset(
                    other_path,
                    split='train',
                    transform=train_transform,
                    max_samples=args.train_samples // 4  # 25%的其他域数据
                )
                train_dataset = MultiDomainDataset(
                    [train_dataset, other_dataset],
                    [0, 1]
                )

            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )

            # 优化器
            optimizer = torch.optim.AdamW(
                model.get_trainable_params(),
                lr=args.lr,
                weight_decay=args.weight_decay
            )

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs
            )

            criterion = nn.CrossEntropyLoss()

            # 训练
            print(f"\n  Training for {args.epochs} epochs...")
            start_time = time.time()

            for epoch in range(args.epochs):
                train_loss, train_acc = train_epoch(
                    model, train_loader, optimizer, criterion,
                    device, epoch, args.epochs
                )
                scheduler.step()
                print(f"  Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={train_acc:.4f}")

            train_time = time.time() - start_time
            print(f"  Training time: {train_time/60:.1f} minutes")

            # 评估
            print(f"\n  Evaluating...")
            model_results = {'train_time': train_time}

            for test_name, test_path in exp['test_paths'].items():
                test_dataset = GenImageDataset(
                    test_path,
                    split='val',
                    transform=test_transform,
                    max_samples=args.test_samples
                )

                test_loader = DataLoader(
                    test_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=4
                )

                metrics = evaluate(model, test_loader, device, desc=f'  Testing {test_name}')
                model_results[test_name] = metrics
                print(f"    {test_name}: AUC={metrics['auc']:.4f}, Acc={metrics['accuracy']:.4f}")

            results[exp_name][model_name] = model_results

            # 释放内存
            del model
            torch.cuda.empty_cache()

    return results


def print_summary(results):
    """打印结果摘要"""
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)

    # 表格格式输出
    print("\n| Train Data | Model | In-Domain AUC | Cross-Domain AUC | Improvement |")
    print("|------------|-------|---------------|------------------|-------------|")

    for exp_name, exp_results in results.items():
        train_domain = exp_name.replace('_train', '')
        cross_domain = 'ADM' if train_domain == 'BigGAN' else 'BigGAN'

        baseline = exp_results.get('baseline', {})
        generalized = exp_results.get('generalized', {})

        base_in = baseline.get(train_domain, {}).get('auc', 0)
        base_cross = baseline.get(cross_domain, {}).get('auc', 0)

        gen_in = generalized.get(train_domain, {}).get('auc', 0)
        gen_cross = generalized.get(cross_domain, {}).get('auc', 0)

        print(f"| {train_domain:10} | baseline    | {base_in:.4f}        | {base_cross:.4f}           | -           |")
        imp = gen_cross - base_cross
        print(f"| {train_domain:10} | generalized | {gen_in:.4f}        | {gen_cross:.4f}           | {imp:+.4f}      |")

    # 计算平均改进
    print("\n" + "-"*70)

    all_base_cross = []
    all_gen_cross = []

    for exp_name, exp_results in results.items():
        train_domain = exp_name.replace('_train', '')
        cross_domain = 'ADM' if train_domain == 'BigGAN' else 'BigGAN'

        base_cross = exp_results.get('baseline', {}).get(cross_domain, {}).get('auc', 0)
        gen_cross = exp_results.get('generalized', {}).get(cross_domain, {}).get('auc', 0)

        all_base_cross.append(base_cross)
        all_gen_cross.append(gen_cross)

    avg_base = np.mean(all_base_cross)
    avg_gen = np.mean(all_gen_cross)

    print(f"\nAverage Cross-Domain AUC:")
    print(f"  Baseline:    {avg_base:.4f}")
    print(f"  Generalized: {avg_gen:.4f}")
    print(f"  Improvement: {avg_gen - avg_base:+.4f} ({(avg_gen - avg_base) / avg_base * 100:+.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='真实数据跨域泛化实验')

    parser.add_argument('--data_root', type=str,
                        default='datasets/genimage_partial',
                        help='数据集根目录')
    parser.add_argument('--train_samples', type=int, default=20000,
                        help='训练样本数')
    parser.add_argument('--test_samples', type=int, default=4000,
                        help='测试样本数')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--embed_dim', type=int, default=768,
                        help='特征维度(256用于快速测试，768/1024用于完整实验)')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default='outputs/real_data_experiment')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Cross-Domain Generalization Experiment")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Data root: {args.data_root}")
    print(f"  Train samples: {args.train_samples}")
    print(f"  Test samples: {args.test_samples}")
    print(f"  Embed dim: {args.embed_dim}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")

    # 运行实验
    results = run_experiment(args)

    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'results_{timestamp}.json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # 打印摘要
    print_summary(results)


if __name__ == '__main__':
    main()
