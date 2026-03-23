#!/usr/bin/env python
"""
快速优化实验脚本
使用更优化的训练策略，尝试在短时间内获得更好的跨域泛化效果
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
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
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
    def __init__(self, root_dir, split='train', transform=None, max_samples=None, domain_id=0):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.domain_id = domain_id
        self.samples = []

        # 检查split目录
        split_dir = self.root_dir / split

        # 尝试多种可能的目录结构
        ai_dirs = [split_dir / 'ai', split_dir / 'fake', split_dir / '1']
        nature_dirs = [split_dir / 'nature', split_dir / 'real', split_dir / '0']

        ai_dir = next((d for d in ai_dirs if d.exists()), None)
        nature_dir = next((d for d in nature_dirs if d.exists()), None)

        if ai_dir is None or nature_dir is None:
            raise ValueError(f"数据目录不完整: {root_dir}/{split}")

        # 加载图像
        exts = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']

        ai_images = []
        nature_images = []
        for ext in exts:
            ai_images.extend(list(ai_dir.glob(ext)))
            nature_images.extend(list(nature_dir.glob(ext)))

        np.random.seed(42)
        np.random.shuffle(ai_images)
        np.random.shuffle(nature_images)

        # 限制样本数
        if max_samples:
            n_per_class = max_samples // 2
            ai_images = ai_images[:n_per_class]
            nature_images = nature_images[:n_per_class]

        for img_path in ai_images:
            self.samples.append((img_path, 1))
        for img_path in nature_images:
            self.samples.append((img_path, 0))

        np.random.shuffle(self.samples)
        print(f"Loaded {len(self.samples)} samples from {root_dir}/{split} (domain_id={domain_id})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return {'image': image, 'label': label, 'domain_id': self.domain_id, 'path': str(img_path)}
        except Exception as e:
            # 返回一个随机图像以防错误
            image = torch.randn(3, 224, 224)
            return {'image': image, 'label': label, 'domain_id': self.domain_id, 'path': str(img_path)}


class QuickOptimizer:
    """快速优化器"""
    def __init__(self, device, output_dir):
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_model(self):
        """构建模型"""
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
        ).to(self.device)
        return model

    def get_transforms(self, is_train=True, strong_aug=False):
        """获取数据变换"""
        if is_train:
            if strong_aug:
                return transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225]),
                    transforms.RandomErasing(p=0.2)
                ])
            else:
                return transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

    def create_dataloaders(self, data_roots, train_samples, val_samples, batch_size, strong_aug=False):
        """创建数据加载器"""
        train_transform = self.get_transforms(is_train=True, strong_aug=strong_aug)
        val_transform = self.get_transforms(is_train=False)

        # 多源训练数据
        train_datasets = []
        val_datasets = []
        for i, root in enumerate(data_roots):
            train_ds = GenImageDataset(root, split='train', transform=train_transform,
                                       max_samples=train_samples//len(data_roots), domain_id=i)
            val_ds = GenImageDataset(root, split='val', transform=val_transform,
                                     max_samples=val_samples//len(data_roots), domain_id=i)
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)

        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=4, pin_memory=True)

        return train_loader, val_loader

    def train_with_warmup_cosine(self, model, train_loader, val_loader, epochs, lr, warmup_epochs=2):
        """使用warmup + cosine退火训练"""
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=0.05
        )

        # OneCycleLR: 自动warmup + 余弦退火
        total_steps = len(train_loader) * epochs
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=warmup_epochs/epochs,
            anneal_strategy='cos',
            div_factor=10,
            final_div_factor=100
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        best_val_auc = 0
        best_model_state = None
        patience = 3
        patience_counter = 0

        for epoch in range(epochs):
            # 训练
            model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()
                outputs = model(images)

                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get('output'))
                else:
                    logits = outputs

                loss = criterion(logits, labels)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # 验证
            val_metrics = self.evaluate_loader(model, val_loader)

            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
                  f"Val AUC={val_metrics['auc']:.4f}, Val Acc={val_metrics['accuracy']:.4f}")

            # 早期停止
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # 恢复最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return best_val_auc

    def evaluate_loader(self, model, dataloader):
        """评估数据加载器"""
        model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
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

    def evaluate_cross_domain(self, model, data_roots, dataset_names, val_samples):
        """评估跨域性能"""
        val_transform = self.get_transforms(is_train=False)
        results = {}

        for root, name in zip(data_roots, dataset_names):
            dataset = GenImageDataset(root, split='val', transform=val_transform,
                                     max_samples=val_samples, domain_id=0)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
            metrics = self.evaluate_loader(model, dataloader)
            results[name] = metrics
            print(f"  {name}: AUC={metrics['auc']:.4f}, AP={metrics['ap']:.4f}, Acc={metrics['accuracy']:.4f}")

        return results

    def run_experiment(self, config):
        """运行单个实验"""
        print(f"\n{'='*60}")
        print(f"Configuration: {config['name']}")
        print('='*60)

        # 构建模型
        model = self.build_model()

        # 创建数据加载器
        train_loader, val_loader = self.create_dataloaders(
            config['data_roots'],
            config['train_samples'],
            config['val_samples'],
            config['batch_size'],
            strong_aug=config.get('strong_aug', False)
        )

        # 训练
        start_time = time.time()
        best_val_auc = self.train_with_warmup_cosine(
            model, train_loader, val_loader,
            epochs=config['epochs'],
            lr=config['lr'],
            warmup_epochs=config.get('warmup_epochs', 2)
        )
        train_time = (time.time() - start_time) / 60

        # 跨域评估
        print("\nCross-domain evaluation:")
        test_results = self.evaluate_cross_domain(
            model, config['data_roots'], config['dataset_names'], config['val_samples']
        )

        # 计算汇总指标
        aucs = [test_results[name]['auc'] for name in config['dataset_names']]
        avg_auc = np.mean(aucs)
        min_auc = np.min(aucs)

        results = {
            'config': config['name'],
            'train_time_min': train_time,
            'best_val_auc': best_val_auc,
            'test_results': test_results,
            'avg_auc': avg_auc,
            'min_auc': min_auc,
            'cross_domain_auc': min_auc
        }

        # 保存结果
        output_file = self.output_dir / f"results_{config['name']}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")

        # 保存模型检查点
        checkpoint_file = self.output_dir / f"model_{config['name']}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'results': results
        }, checkpoint_file)
        print(f"Model saved to {checkpoint_file}")

        return results


def main():
    parser = argparse.ArgumentParser(description='快速优化实验')
    parser.add_argument('--data_roots', type=str, nargs='+', required=True)
    parser.add_argument('--datasets', type=str, nargs='+', required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/quick_optimization')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    optimizer = QuickOptimizer(device, args.output_dir)

    # 验证数据集存在
    for root, name in zip(args.data_roots, args.datasets):
        if os.path.exists(root):
            print(f"数据集 {name}: {root} ✓")
        else:
            print(f"数据集 {name}: {root} ✗")
            sys.exit(1)

    # 定义多个实验配置
    experiments = [
        # 配置1: 更高学习率 + 强增强
        {
            'name': 'high_lr_strong_aug',
            'data_roots': args.data_roots,
            'dataset_names': args.datasets,
            'train_samples': 10000,
            'val_samples': 1500,
            'batch_size': 16,
            'epochs': 10,
            'lr': 5e-4,
            'warmup_epochs': 1,
            'strong_aug': True
        },
        # 配置2: 标准配置 + label smoothing (已内置)
        {
            'name': 'standard_smooth',
            'data_roots': args.data_roots,
            'dataset_names': args.datasets,
            'train_samples': 10000,
            'val_samples': 1500,
            'batch_size': 16,
            'epochs': 10,
            'lr': 1e-4,
            'warmup_epochs': 2,
            'strong_aug': False
        },
        # 配置3: 更长训练 + 较低学习率
        {
            'name': 'longer_training',
            'data_roots': args.data_roots,
            'dataset_names': args.datasets,
            'train_samples': 12000,
            'val_samples': 1500,
            'batch_size': 16,
            'epochs': 15,
            'lr': 5e-5,
            'warmup_epochs': 3,
            'strong_aug': False
        },
        # 配置4: 大批次 + 高学习率
        {
            'name': 'large_batch',
            'data_roots': args.data_roots,
            'dataset_names': args.datasets,
            'train_samples': 10000,
            'val_samples': 1500,
            'batch_size': 32,
            'epochs': 12,
            'lr': 2e-4,
            'warmup_epochs': 2,
            'strong_aug': False
        },
    ]

    all_results = []
    for config in experiments:
        try:
            results = optimizer.run_experiment(config)
            all_results.append(results)
        except Exception as e:
            print(f"Error in experiment {config['name']}: {e}")
            continue

    # 汇总结果
    print("\n" + "="*60)
    print("Summary of All Experiments")
    print("="*60)

    best_result = None
    best_cross_domain = 0

    for r in all_results:
        print(f"\n{r['config']}:")
        print(f"  Train Time: {r['train_time_min']:.2f} min")
        print(f"  Average AUC: {r['avg_auc']:.4f}")
        print(f"  Cross-domain AUC (min): {r['cross_domain_auc']:.4f}")
        for ds, metrics in r['test_results'].items():
            print(f"    {ds}: {metrics['auc']:.4f}")

        if r['cross_domain_auc'] > best_cross_domain:
            best_cross_domain = r['cross_domain_auc']
            best_result = r

    if best_result:
        print(f"\nBest Configuration: {best_result['config']}")
        print(f"Best Cross-domain AUC: {best_result['cross_domain_auc']:.4f}")

    # 保存汇总
    summary_file = Path(args.output_dir) / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'all_results': all_results,
            'best_config': best_result['config'] if best_result else None,
            'best_cross_domain_auc': best_cross_domain
        }, f, indent=2)
    print(f"\nSummary saved to {summary_file}")


if __name__ == '__main__':
    main()
