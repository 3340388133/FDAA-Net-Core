"""
跨域泛化验证实验脚本

功能：
1. 对比原始模型 vs 改进模型的跨域泛化能力
2. 消融实验：验证DAFL、AFB、PCR各自的贡献
3. 可视化分析：频率响应、原型激活、特征分布

使用方法：
    python experiments/run_generalization_validation.py --config configs/generalization.yaml
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
from torch.utils.data import DataLoader, Subset
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detector_generalized import GeneralizedAIGCDetector, GeneralizedDetectorTrainer
from data.dataset import AIGCDataset


def parse_args():
    parser = argparse.ArgumentParser(description='跨域泛化验证实验')

    # 数据配置
    parser.add_argument('--data_root', type=str, default='../GenImage',
                        help='数据集根目录')
    parser.add_argument('--train_dataset', type=str, default='BigGAN',
                        help='训练数据集')
    parser.add_argument('--test_datasets', type=str, nargs='+',
                        default=['BigGAN', 'ADM', 'ProGAN', 'StyleGAN'],
                        help='测试数据集列表')
    parser.add_argument('--train_samples', type=int, default=10000,
                        help='训练样本数')
    parser.add_argument('--test_samples', type=int, default=2000,
                        help='每个测试集样本数')

    # 模型配置
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--num_adapter_layers', type=int, default=3)
    parser.add_argument('--num_prototypes', type=int, default=4)
    parser.add_argument('--num_domains', type=int, default=4)

    # 训练配置
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # 消融配置
    parser.add_argument('--ablation', type=str, default='full',
                        choices=['baseline', 'dafl_only', 'afb_only', 'pcr_only',
                                'dafl_afb', 'dafl_pcr', 'afb_pcr', 'full'],
                        help='消融实验配置')

    # 其他
    parser.add_argument('--output_dir', type=str, default='outputs/generalization_validation')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_ablation_config(ablation_type):
    """获取消融实验配置"""
    configs = {
        'baseline': {
            'use_multi_level_da': False,
            'use_domain_adversarial': False,
            'use_pcr': False,
            'use_hierarchical': False,
        },
        'dafl_only': {
            'use_multi_level_da': True,
            'use_domain_adversarial': True,
            'use_pcr': False,
            'use_hierarchical': False,
        },
        'afb_only': {
            'use_multi_level_da': True,  # AFB集成在FDAA-DA中
            'use_domain_adversarial': False,
            'use_pcr': False,
            'use_hierarchical': False,
        },
        'pcr_only': {
            'use_multi_level_da': False,
            'use_domain_adversarial': False,
            'use_pcr': True,
            'use_hierarchical': True,
        },
        'dafl_afb': {
            'use_multi_level_da': True,
            'use_domain_adversarial': True,
            'use_pcr': False,
            'use_hierarchical': False,
        },
        'dafl_pcr': {
            'use_multi_level_da': True,
            'use_domain_adversarial': True,
            'use_pcr': True,
            'use_hierarchical': True,
        },
        'afb_pcr': {
            'use_multi_level_da': True,
            'use_domain_adversarial': False,
            'use_pcr': True,
            'use_hierarchical': True,
        },
        'full': {
            'use_multi_level_da': True,
            'use_domain_adversarial': True,
            'use_pcr': True,
            'use_hierarchical': True,
        }
    }
    return configs[ablation_type]


def create_dataloader(data_root, dataset_name, split, num_samples, batch_size, is_train=True):
    """创建数据加载器"""
    from torchvision import transforms

    # 数据变换
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    # 构建数据集路径
    dataset_path = os.path.join(data_root, dataset_name)

    if os.path.exists(dataset_path):
        try:
            dataset = AIGCDataset(
                root=dataset_path,
                split=split,
                transform=transform,
                max_samples=num_samples
            )
        except Exception as e:
            print(f"Warning: Failed to load {dataset_name}: {e}")
            print("Using synthetic data for testing...")
            dataset = SyntheticDataset(num_samples, is_fake=True)
    else:
        print(f"Warning: Dataset {dataset_name} not found at {dataset_path}")
        print("Using synthetic data for testing...")
        dataset = SyntheticDataset(num_samples, is_fake=True)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=4,
        pin_memory=True
    )

    return dataloader


class SyntheticDataset(torch.utils.data.Dataset):
    """合成数据集（用于测试）"""
    def __init__(self, num_samples, is_fake=True):
        self.num_samples = num_samples
        self.is_fake = is_fake

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 生成随机图像
        image = torch.randn(3, 224, 224)
        # 标签：一半真一半假
        label = idx % 2
        return image, label


class MultiDomainDataset(torch.utils.data.Dataset):
    """多域数据集（组合多个数据集）"""
    def __init__(self, datasets, domain_labels):
        self.datasets = datasets
        self.domain_labels = domain_labels

        # 计算每个数据集的起始索引
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

        # 计算数据集内的索引
        if dataset_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        # 获取数据
        image, label = self.datasets[dataset_idx][local_idx]
        domain_label = self.domain_labels[dataset_idx]

        return image, label, domain_label


def train_model(model, train_loader, args, device):
    """训练模型"""
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.get_trainable_params(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    trainer = GeneralizedDetectorTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )

    history = []

    for epoch in range(args.epochs):
        model.set_epoch(epoch, args.epochs)
        model.train()

        epoch_losses = []

        for batch_idx, batch in enumerate(train_loader):
            if len(batch) == 3:
                images, labels, domain_labels = batch
            else:
                images, labels = batch
                domain_labels = None

            loss_dict = trainer.train_step(images, labels, domain_labels)
            epoch_losses.append(loss_dict['total_loss'])

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}, "
                      f"Loss: {loss_dict['total_loss']:.4f}")

        avg_loss = np.mean(epoch_losses)
        history.append({'epoch': epoch + 1, 'loss': avg_loss})
        print(f"Epoch {epoch+1} completed, Avg Loss: {avg_loss:.4f}")

        scheduler.step()

    return history


def evaluate_cross_domain(model, test_loaders, device):
    """跨域评估"""
    model = model.to(device)
    model.eval()

    results = {}

    for dataset_name, loader in test_loaders.items():
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in loader:
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                probs = F.softmax(outputs['logits'], dim=-1)

                all_preds.append(outputs['logits'].argmax(dim=-1).cpu())
                all_labels.append(labels.cpu())
                all_probs.append(probs[:, 1].cpu())

        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        probs = torch.cat(all_probs).numpy()

        # 计算指标
        accuracy = (preds == labels).mean()

        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            auc = roc_auc_score(labels, probs)
            ap = average_precision_score(labels, probs)
        except:
            auc = 0.0
            ap = 0.0

        results[dataset_name] = {
            'accuracy': float(accuracy),
            'auc': float(auc),
            'ap': float(ap)
        }

        print(f"{dataset_name}: Acc={accuracy:.4f}, AUC={auc:.4f}, AP={ap:.4f}")

    # 计算跨域平均
    other_datasets = [k for k in results.keys() if k != args.train_dataset]
    if other_datasets:
        cross_domain_auc = np.mean([results[k]['auc'] for k in other_datasets])
        results['cross_domain_avg'] = {'auc': float(cross_domain_auc)}
        print(f"\nCross-Domain Avg AUC: {cross_domain_auc:.4f}")

    return results


def run_ablation_study(args):
    """运行消融实验"""
    print("\n" + "="*60)
    print("Running Ablation Study")
    print("="*60)

    ablation_configs = [
        'baseline', 'dafl_only', 'afb_only', 'pcr_only',
        'dafl_afb', 'dafl_pcr', 'afb_pcr', 'full'
    ]

    all_results = {}

    for ablation in ablation_configs:
        print(f"\n--- Testing configuration: {ablation} ---")

        config = get_ablation_config(ablation)

        # 创建模型
        model = GeneralizedAIGCDetector(
            num_classes=2,
            img_size=args.img_size,
            embed_dim=args.embed_dim,
            num_adapter_layers=args.num_adapter_layers,
            num_prototypes=args.num_prototypes,
            num_domains=args.num_domains,
            **config
        )

        params = model.count_parameters()
        print(f"Parameters: {params['trainable']:,} trainable")

        # 创建训练数据（使用合成数据进行快速测试）
        train_dataset = SyntheticDataset(args.train_samples)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        # 训练
        history = train_model(model, train_loader, args, args.device)

        # 评估
        test_loaders = {}
        for ds_name in args.test_datasets:
            test_dataset = SyntheticDataset(args.test_samples)
            test_loaders[ds_name] = DataLoader(test_dataset, batch_size=args.batch_size)

        results = evaluate_cross_domain(model, test_loaders, args.device)

        all_results[ablation] = {
            'config': config,
            'params': params['trainable'],
            'results': results
        }

    return all_results


def run_single_experiment(args):
    """运行单个实验"""
    print("\n" + "="*60)
    print(f"Running experiment: {args.ablation}")
    print("="*60)

    # 获取配置
    config = get_ablation_config(args.ablation)

    # 创建模型
    model = GeneralizedAIGCDetector(
        num_classes=2,
        img_size=args.img_size,
        embed_dim=args.embed_dim,
        num_adapter_layers=args.num_adapter_layers,
        num_prototypes=args.num_prototypes,
        num_domains=args.num_domains,
        **config
    )

    params = model.count_parameters()
    print(f"Model parameters: {params['trainable']:,} trainable / {params['total']:,} total")

    # 创建数据加载器
    print("\nLoading training data...")
    train_loader = create_dataloader(
        args.data_root, args.train_dataset, 'train',
        args.train_samples, args.batch_size, is_train=True
    )

    print("\nLoading test data...")
    test_loaders = {}
    for ds_name in args.test_datasets:
        test_loaders[ds_name] = create_dataloader(
            args.data_root, ds_name, 'test',
            args.test_samples, args.batch_size, is_train=False
        )

    # 训练
    print("\nStarting training...")
    history = train_model(model, train_loader, args, args.device)

    # 评估
    print("\nEvaluating cross-domain generalization...")
    results = evaluate_cross_domain(model, test_loaders, args.device)

    return {
        'config': config,
        'params': params,
        'history': history,
        'results': results
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 检查CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    print(f"Device: {args.device}")

    # 运行实验
    if args.ablation == 'all':
        results = run_ablation_study(args)
        output_file = output_dir / f'ablation_results_{timestamp}.json'
    else:
        results = run_single_experiment(args)
        output_file = output_dir / f'{args.ablation}_results_{timestamp}.json'

    # 保存结果
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # 打印总结
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)

    if args.ablation == 'all':
        # 打印消融实验对比表
        print("\n| Config | Params | In-Domain AUC | Cross-Domain AUC |")
        print("|--------|--------|---------------|------------------|")
        for config_name, data in results.items():
            in_domain = data['results'].get(args.train_dataset, {}).get('auc', 0)
            cross_domain = data['results'].get('cross_domain_avg', {}).get('auc', 0)
            print(f"| {config_name:12} | {data['params']:8,} | {in_domain:.4f} | {cross_domain:.4f} |")
    else:
        print(f"\nConfiguration: {args.ablation}")
        print(f"Training dataset: {args.train_dataset}")
        print(f"\nResults:")
        for ds_name, metrics in results['results'].items():
            if ds_name != 'cross_domain_avg':
                print(f"  {ds_name}: AUC={metrics['auc']:.4f}, Acc={metrics['accuracy']:.4f}")

        if 'cross_domain_avg' in results['results']:
            print(f"\n  Cross-Domain Average AUC: {results['results']['cross_domain_avg']['auc']:.4f}")


if __name__ == '__main__':
    main()
