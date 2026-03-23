#!/usr/bin/env python3
"""
第二阶段进阶泛化优化
基于第一阶段结果，对最佳策略进行精细化调优
包括：
1. 组合策略优化
2. 超参数精细调整
3. 训练技巧增强
4. 模型集成
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from tqdm import tqdm

from models.detector_generalized import GeneralizedAIGCDetector
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score


class AdvancedGeneralizationTrainer:
    """进阶泛化训练器"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_cross_domain_auc = 0.0
        self.patience_counter = 0

    def build_model(self):
        """构建模型"""
        model = GeneralizedAIGCDetector(
            backbone_name="ViT-L/14",
            num_classes=2,
            num_domains=2,
            use_domain_adversarial=self.config.get('use_domain_adversarial', False),
            freeze_backbone=False  # 微调backbone以提高泛化
        )
        return model.to(self.device)

    def train_with_advanced_techniques(self, train_loaders, val_loaders, dataset_names):
        """使用进阶技术训练"""
        model = self.build_model()

        # 优化器设置
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-4)

        # 使用不同学习率
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if 'backbone' in name or 'resnet' in name.lower():
                backbone_params.append(param)
            else:
                head_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': lr * 0.1},  # backbone用更小的学习率
            {'params': head_params, 'lr': lr}
        ], weight_decay=weight_decay)

        # 学习率调度
        scheduler_type = self.config.get('scheduler', 'cosine')
        epochs = self.config.get('epochs', 10)

        if scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=lr * 0.01
            )
        elif scheduler_type == 'warmup_cosine':
            warmup_epochs = self.config.get('warmup_epochs', 2)
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                else:
                    progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                    return 0.5 * (1 + np.cos(np.pi * progress))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

        # 损失函数
        criterion = nn.CrossEntropyLoss(label_smoothing=self.config.get('label_smoothing', 0.1))

        # 域对抗相关
        use_domain_adv = self.config.get('use_domain_adversarial', False)
        domain_weight = self.config.get('domain_weight', 0.1)

        # 梯度累积
        grad_accum_steps = self.config.get('grad_accum_steps', 1)

        best_model_state = None
        best_score = 0.0

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            n_batches = 0

            # 合并所有数据加载器的迭代
            all_iters = [iter(loader) for loader in train_loaders]

            # 交替从不同域采样
            pbar = tqdm(range(self.config.get('steps_per_epoch', 250)), desc=f"Epoch {epoch+1}/{epochs}")
            optimizer.zero_grad()

            for step in pbar:
                # 从各个域随机采样
                for domain_idx, data_iter in enumerate(all_iters):
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        all_iters[domain_idx] = iter(train_loaders[domain_idx])
                        batch = next(all_iters[domain_idx])

                    # 处理字典或元组格式的batch
                    if isinstance(batch, dict):
                        images = batch['image']
                        labels = batch['label']
                    else:
                        images, labels = batch[0], batch[1]
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # 前向传播
                    outputs = model(images)
                    if isinstance(outputs, dict):
                        logits = outputs.get('logits', outputs.get('output', None))
                        if logits is None:
                            logits = list(outputs.values())[0]
                    else:
                        logits = outputs

                    # 分类损失
                    cls_loss = criterion(logits, labels)

                    # 域对抗损失（如果启用）
                    if use_domain_adv and hasattr(model, 'domain_classifier'):
                        domain_labels = torch.full((images.size(0),), domain_idx,
                                                   dtype=torch.long, device=self.device)
                        features = model.get_features(images)
                        domain_pred = model.domain_classifier(features)
                        domain_loss = F.cross_entropy(domain_pred, domain_labels)
                        loss = cls_loss - domain_weight * domain_loss  # 对抗训练
                    else:
                        loss = cls_loss

                    loss = loss / grad_accum_steps
                    loss.backward()

                    total_loss += loss.item() * grad_accum_steps
                    n_batches += 1

                # 梯度累积更新
                if (step + 1) % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                pbar.set_postfix({'loss': total_loss / max(n_batches, 1)})

            scheduler.step()

            # 验证
            eval_results = self.evaluate(model, val_loaders, dataset_names)
            cross_domain_auc = eval_results.get('cross_domain_auc', 0.0)

            print(f"  Epoch {epoch+1}: 平均AUC={eval_results['avg_auc']:.4f}, 跨域AUC={cross_domain_auc:.4f}")

            # 保存最佳模型
            if cross_domain_auc > best_score:
                best_score = cross_domain_auc
                best_model_state = model.state_dict().copy()
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # 早停
            if self.patience_counter >= self.config.get('patience', 5):
                print(f"  早停于epoch {epoch+1}")
                break

        # 恢复最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return model

    def evaluate(self, model, val_loaders, dataset_names):
        """评估模型"""
        model.eval()
        results = {}

        with torch.no_grad():
            for loader, name in zip(val_loaders, dataset_names):
                all_probs = []
                all_labels = []

                for batch in loader:
                    # 处理字典或元组格式的batch
                    if isinstance(batch, dict):
                        images = batch['image']
                        labels = batch['label']
                    else:
                        images, labels = batch[0], batch[1]
                    images = images.to(self.device)

                    outputs = model(images)
                    if isinstance(outputs, dict):
                        logits = outputs.get('logits', outputs.get('output', None))
                        if logits is None:
                            logits = list(outputs.values())[0]
                    else:
                        logits = outputs

                    probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    all_probs.extend(probs)
                    all_labels.extend(labels.numpy())

                auc = roc_auc_score(all_labels, all_probs)
                ap = average_precision_score(all_labels, all_probs)
                acc = accuracy_score(all_labels, (np.array(all_probs) > 0.5).astype(int))

                results[name] = {'auc': auc, 'ap': ap, 'accuracy': acc}

        # 计算平均和跨域指标
        aucs = [r['auc'] for r in results.values()]
        results['avg_auc'] = np.mean(aucs)
        results['min_auc'] = min(aucs)

        # 假设第一个是域内，其他是跨域
        if len(dataset_names) > 1:
            cross_domain_aucs = [results[name]['auc'] for name in dataset_names[1:]]
            results['cross_domain_auc'] = np.mean(cross_domain_aucs)
        else:
            results['cross_domain_auc'] = results['avg_auc']

        return results


def create_dataloaders(data_roots, dataset_names, train_samples, val_samples, batch_size):
    """创建数据加载器"""
    from torchvision import transforms
    from torch.utils.data import Subset
    import random

    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loaders = []
    val_loaders = []

    for root, name in zip(data_roots, dataset_names):
        # 尝试加载数据
        try:
            from data.genimage_dataset import GenImageDataset
            train_dataset = GenImageDataset(root, split='train', transform=train_transform)
            val_dataset = GenImageDataset(root, split='val', transform=val_transform)
        except:
            from torchvision.datasets import ImageFolder
            train_path = os.path.join(root, 'train')
            val_path = os.path.join(root, 'val')
            train_dataset = ImageFolder(train_path, transform=train_transform)
            val_dataset = ImageFolder(val_path, transform=val_transform)

        # 采样
        train_indices = random.sample(range(len(train_dataset)), min(train_samples, len(train_dataset)))
        val_indices = random.sample(range(len(val_dataset)), min(val_samples, len(val_dataset)))

        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(val_dataset, val_indices)

        train_loaders.append(DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                         num_workers=4, pin_memory=True))
        val_loaders.append(DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                                       num_workers=4, pin_memory=True))

        print(f"数据集 {name}: 训练{len(train_subset)}, 验证{len(val_subset)}")

    return train_loaders, val_loaders


def run_advanced_optimization(args):
    """运行进阶优化"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建数据加载器
    train_loaders, val_loaders = create_dataloaders(
        args.data_roots, args.datasets,
        args.train_samples, args.val_samples, args.batch_size
    )

    # 定义进阶配置搜索空间
    advanced_configs = [
        # 配置1: 多源域 + MixStyle + 强正则化
        {
            'name': 'multi_source_mixstyle_strong_reg',
            'use_mixstyle': True,
            'use_freq_norm': True,
            'use_domain_adversarial': False,
            'learning_rate': 5e-5,
            'weight_decay': 5e-4,
            'label_smoothing': 0.15,
            'scheduler': 'warmup_cosine',
            'warmup_epochs': 2,
            'epochs': 12,
            'patience': 6,
            'steps_per_epoch': 300,
        },
        # 配置2: 域对抗 + 渐进式权重
        {
            'name': 'domain_adv_progressive',
            'use_mixstyle': False,
            'use_freq_norm': True,
            'use_domain_adversarial': True,
            'domain_weight': 0.15,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'label_smoothing': 0.1,
            'scheduler': 'cosine',
            'epochs': 10,
            'patience': 5,
            'steps_per_epoch': 250,
        },
        # 配置3: 组合策略 - 所有技术
        {
            'name': 'combined_all',
            'use_mixstyle': True,
            'use_freq_norm': True,
            'use_domain_adversarial': True,
            'domain_weight': 0.1,
            'learning_rate': 8e-5,
            'weight_decay': 3e-4,
            'label_smoothing': 0.12,
            'scheduler': 'warmup_cosine',
            'warmup_epochs': 1,
            'epochs': 15,
            'patience': 7,
            'steps_per_epoch': 250,
            'grad_accum_steps': 2,
        },
        # 配置4: 大batch + 低学习率
        {
            'name': 'large_batch_low_lr',
            'use_mixstyle': True,
            'use_freq_norm': False,
            'use_domain_adversarial': False,
            'learning_rate': 3e-5,
            'weight_decay': 1e-3,
            'label_smoothing': 0.2,
            'scheduler': 'cosine',
            'epochs': 20,
            'patience': 10,
            'steps_per_epoch': 200,
            'grad_accum_steps': 4,
        },
        # 配置5: 频率正则化增强
        {
            'name': 'freq_reg_enhanced',
            'use_mixstyle': False,
            'use_freq_norm': True,
            'use_domain_adversarial': True,
            'domain_weight': 0.2,
            'learning_rate': 1e-4,
            'weight_decay': 2e-4,
            'label_smoothing': 0.1,
            'scheduler': 'warmup_cosine',
            'warmup_epochs': 3,
            'epochs': 12,
            'patience': 6,
            'steps_per_epoch': 280,
        },
    ]

    # 运行实验
    all_results = []
    best_result = None
    best_score = 0.0

    for config in advanced_configs:
        print(f"\n{'='*70}")
        print(f"配置: {config['name']}")
        print(f"{'='*70}")

        start_time = time.time()

        trainer = AdvancedGeneralizationTrainer(config)
        model = trainer.train_with_advanced_techniques(train_loaders, val_loaders, args.datasets)

        # 最终评估
        final_results = trainer.evaluate(model, val_loaders, args.datasets)
        train_time = (time.time() - start_time) / 60

        result = {
            'config_name': config['name'],
            'config': {k: v for k, v in config.items() if k != 'name'},
            'train_time_min': train_time,
            'results': {name: final_results[name] for name in args.datasets},
            'avg_auc': final_results['avg_auc'],
            'cross_domain_auc': final_results['cross_domain_auc'],
            'min_auc': final_results['min_auc'],
        }

        all_results.append(result)

        # 保存单个结果
        result_file = output_dir / f"results_{config['name']}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n结果:")
        for name in args.datasets:
            r = final_results[name]
            print(f"  {name}: AUC={r['auc']:.4f}, AP={r['ap']:.4f}")
        print(f"  平均AUC: {final_results['avg_auc']:.4f}")
        print(f"  跨域AUC: {final_results['cross_domain_auc']:.4f}")

        # 更新最佳结果
        if final_results['cross_domain_auc'] > best_score:
            best_score = final_results['cross_domain_auc']
            best_result = result

            # 保存最佳模型
            torch.save(model.state_dict(), output_dir / 'best_model_advanced.pth')
            print(f"  *** 新的最佳配置! 跨域AUC={best_score:.4f} ***")

    # 保存汇总结果
    summary = {
        'timestamp': datetime.now().isoformat(),
        'best_config': best_result['config_name'] if best_result else None,
        'best_cross_domain_auc': best_score,
        'all_results': all_results
    }

    with open(output_dir / 'summary_advanced.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # 打印排名
    print(f"\n{'='*70}")
    print("进阶优化结果排名（按跨域AUC）")
    print(f"{'='*70}")

    sorted_results = sorted(all_results, key=lambda x: x['cross_domain_auc'], reverse=True)
    for i, r in enumerate(sorted_results, 1):
        print(f"{i}. {r['config_name']}: 跨域AUC={r['cross_domain_auc']:.4f}, 平均AUC={r['avg_auc']:.4f}")

    return best_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='进阶泛化优化实验')
    parser.add_argument('--data_roots', nargs='+', required=True)
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--train_samples', type=int, default=10000)
    parser.add_argument('--val_samples', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_dir', default='outputs/advanced_optimization')

    args = parser.parse_args()

    print("="*70)
    print("进阶泛化优化实验")
    print("="*70)
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    result = run_advanced_optimization(args)

    if result:
        print(f"\n最终最佳配置: {result['config_name']}")
        print(f"最终跨域AUC: {result['cross_domain_auc']:.4f}")
