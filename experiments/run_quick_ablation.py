#!/usr/bin/env python3
"""
快速消融实验脚本 - 用于生成SCI论文所需的消融研究结果

运行5个关键配置的消融实验，并进行跨域测试
"""

import os
import sys
import json
import time
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.detector_generalized import GeneralizedAIGCDetector

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# 关键消融配置（5个）
ABLATION_CONFIGS = {
    'baseline': {
        'use_multi_level_da': False,
        'use_domain_adversarial': False,
        'use_pcr': False,
        'use_hierarchical': False,
        'description': '基线模型'
    },
    'dafl_only': {
        'use_multi_level_da': True,
        'use_domain_adversarial': True,
        'use_pcr': False,
        'use_hierarchical': False,
        'description': '+DAFL'
    },
    'afb_only': {
        'use_multi_level_da': True,
        'use_domain_adversarial': False,
        'use_pcr': False,
        'use_hierarchical': False,
        'description': '+AFB'
    },
    'pcr_only': {
        'use_multi_level_da': False,
        'use_domain_adversarial': False,
        'use_pcr': True,
        'use_hierarchical': True,
        'description': '+PCR'
    },
    'full': {
        'use_multi_level_da': True,
        'use_domain_adversarial': True,
        'use_pcr': True,
        'use_hierarchical': True,
        'description': 'Full (Ours)'
    }
}


class GenImageDataset(Dataset):
    """GenImage数据集加载器"""

    def __init__(self, root_dir: str, split: str = 'train', max_samples: int = None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.samples = []

        transform_list = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        if split == 'train':
            transform_list.insert(1, transforms.RandomHorizontalFlip())
        self.transform = transforms.Compose(transform_list)

        # 加载真实图像
        real_dir = self.root_dir / split / 'nature'
        if not real_dir.exists():
            real_dir = self.root_dir / split / '0_real'
        if real_dir.exists():
            for img_path in real_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    self.samples.append((str(img_path), 0))

        # 加载AI生成图像
        ai_dir = self.root_dir / split / 'ai'
        if not ai_dir.exists():
            ai_dir = self.root_dir / split / '1_fake'
        if ai_dir.exists():
            for img_path in ai_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    self.samples.append((str(img_path), 1))

        # 限制样本数
        if max_samples and len(self.samples) > max_samples:
            np.random.seed(42)
            indices = np.random.choice(len(self.samples), max_samples, replace=False)
            self.samples = [self.samples[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img, label
        except:
            return torch.zeros(3, 224, 224), label


def evaluate_model(model, dataloader, device):
    """评估模型性能"""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = (all_preds == all_labels).mean()

    if HAS_SKLEARN and len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
        ap = average_precision_score(all_labels, all_probs)
    else:
        auc = accuracy
        ap = accuracy

    return {'accuracy': float(accuracy), 'auc': float(auc), 'ap': float(ap)}


def train_model(model, train_loader, val_loader, device, epochs=3, lr=1e-4):
    """训练模型"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # 主分类损失
            loss = criterion(outputs['logits'], labels)

            # 添加辅助损失
            if 'domain_loss' in outputs and outputs['domain_loss'] is not None:
                loss = loss + 0.1 * outputs['domain_loss']
            if 'pcr_loss' in outputs and outputs['pcr_loss'] is not None:
                loss = loss + 0.1 * outputs['pcr_loss']

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs['logits'].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{correct/total:.4f}')

        scheduler.step()

        # 验证
        val_metrics = evaluate_model(model, val_loader, device)
        print(f'  Val: Acc={val_metrics["accuracy"]:.4f}, AUC={val_metrics["auc"]:.4f}')

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 恢复最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model


def run_ablation_experiment(output_dir: str, epochs: int = 3, train_samples: int = 5000,
                           val_samples: int = 1000, batch_size: int = 32):
    """运行消融实验"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    # 数据集路径
    datasets = {
        'BigGAN': 'datasets/genimage_partial/imagenet_ai_0419_biggan',
        'ADM': 'datasets/genimage_partial/imagenet_ai_0508_adm'
    }

    results = {}

    # 对每个训练数据集
    for train_name, train_path in datasets.items():
        print(f"\n{'='*60}")
        print(f"训练数据集: {train_name}")
        print(f"{'='*60}")

        results[f'{train_name}_train'] = {}

        # 加载训练数据
        print(f"\n加载训练数据: {train_path}")
        train_dataset = GenImageDataset(train_path, 'train', train_samples)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=4, pin_memory=True)
        print(f"  训练样本: {len(train_dataset)}")

        # 加载所有测试数据
        test_loaders = {}
        for test_name, test_path in datasets.items():
            test_dataset = GenImageDataset(test_path, 'val', val_samples)
            test_loaders[test_name] = DataLoader(test_dataset, batch_size=batch_size,
                                                 shuffle=False, num_workers=4, pin_memory=True)
            print(f"  {test_name}测试样本: {len(test_dataset)}")

        # 对每个消融配置
        for config_name, config in ABLATION_CONFIGS.items():
            print(f"\n{'-'*40}")
            print(f"配置: {config_name} - {config['description']}")
            print(f"{'-'*40}")

            # 创建模型
            model = GeneralizedAIGCDetector(
                num_classes=2,
                use_multi_level_da=config['use_multi_level_da'],
                use_domain_adversarial=config['use_domain_adversarial'],
                use_pcr=config['use_pcr'],
                use_hierarchical=config['use_hierarchical']
            ).to(device)

            # 计算参数量
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"可训练参数: {params:,}")

            # 训练
            start_time = time.time()
            model = train_model(model, train_loader, test_loaders[train_name],
                              device, epochs=epochs)
            train_time = time.time() - start_time

            # 测试所有数据集
            config_results = {'train_time': train_time}
            for test_name, test_loader in test_loaders.items():
                metrics = evaluate_model(model, test_loader, device)
                config_results[test_name] = metrics
                is_cross = "跨域" if test_name != train_name else "域内"
                print(f"  {test_name} ({is_cross}): Acc={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}")

            results[f'{train_name}_train'][config_name] = config_results

            # 清理GPU内存
            del model
            torch.cuda.empty_cache()

    return results


def generate_report(results: dict, output_dir: str):
    """生成实验报告"""

    report_lines = [
        "=" * 70,
        "消融实验结果报告 (Ablation Study Results)",
        "=" * 70,
        ""
    ]

    # 表1: 域内性能对比
    report_lines.extend([
        "表1: 域内检测性能 (In-Domain Detection Performance)",
        "-" * 70,
        f"{'配置':<15} {'BigGAN Acc':<12} {'BigGAN AUC':<12} {'ADM Acc':<12} {'ADM AUC':<12}",
        "-" * 70
    ])

    for config_name in ABLATION_CONFIGS.keys():
        biggan_acc = results.get('BigGAN_train', {}).get(config_name, {}).get('BigGAN', {}).get('accuracy', 0)
        biggan_auc = results.get('BigGAN_train', {}).get(config_name, {}).get('BigGAN', {}).get('auc', 0)
        adm_acc = results.get('ADM_train', {}).get(config_name, {}).get('ADM', {}).get('accuracy', 0)
        adm_auc = results.get('ADM_train', {}).get(config_name, {}).get('ADM', {}).get('auc', 0)

        desc = ABLATION_CONFIGS[config_name]['description']
        report_lines.append(f"{desc:<15} {biggan_acc:.4f}       {biggan_auc:.4f}       {adm_acc:.4f}       {adm_auc:.4f}")

    report_lines.extend(["", ""])

    # 表2: 跨域性能对比 (关键!)
    report_lines.extend([
        "表2: 跨域泛化性能 (Cross-Domain Generalization) - 核心创新点验证",
        "-" * 70,
        f"{'配置':<15} {'BigGAN→ADM':<12} {'ADM→BigGAN':<12} {'平均':<12}",
        "-" * 70
    ])

    for config_name in ABLATION_CONFIGS.keys():
        # BigGAN训练，ADM测试
        biggan_to_adm = results.get('BigGAN_train', {}).get(config_name, {}).get('ADM', {}).get('auc', 0)
        # ADM训练，BigGAN测试
        adm_to_biggan = results.get('ADM_train', {}).get(config_name, {}).get('BigGAN', {}).get('auc', 0)
        avg = (biggan_to_adm + adm_to_biggan) / 2

        desc = ABLATION_CONFIGS[config_name]['description']
        report_lines.append(f"{desc:<15} {biggan_to_adm:.4f}       {adm_to_biggan:.4f}       {avg:.4f}")

    report_lines.extend(["", ""])

    # 计算改进幅度
    baseline_biggan_adm = results.get('BigGAN_train', {}).get('baseline', {}).get('ADM', {}).get('auc', 0)
    full_biggan_adm = results.get('BigGAN_train', {}).get('full', {}).get('ADM', {}).get('auc', 0)
    baseline_adm_biggan = results.get('ADM_train', {}).get('baseline', {}).get('BigGAN', {}).get('auc', 0)
    full_adm_biggan = results.get('ADM_train', {}).get('full', {}).get('BigGAN', {}).get('auc', 0)

    improve_1 = (full_biggan_adm - baseline_biggan_adm) * 100
    improve_2 = (full_adm_biggan - baseline_adm_biggan) * 100

    report_lines.extend([
        "关键发现:",
        f"  - BigGAN→ADM 跨域AUC提升: {baseline_biggan_adm:.4f} → {full_biggan_adm:.4f} (+{improve_1:.2f}%)",
        f"  - ADM→BigGAN 跨域AUC提升: {baseline_adm_biggan:.4f} → {full_adm_biggan:.4f} (+{improve_2:.2f}%)",
        f"  - 平均跨域提升: +{(improve_1 + improve_2)/2:.2f}%",
        "",
        "=" * 70
    ])

    report = '\n'.join(report_lines)

    # 保存报告
    report_path = Path(output_dir) / 'ablation_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n{report}")

    return report


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--train_samples', type=int, default=5000)
    parser.add_argument('--val_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='outputs/ablation_results')
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("=" * 70)
    print("消融实验 (Ablation Study)")
    print("=" * 70)
    print(f"Epochs: {args.epochs}")
    print(f"训练样本: {args.train_samples}")
    print(f"验证样本: {args.val_samples}")
    print(f"输出目录: {output_dir}")

    # 运行实验
    results = run_ablation_experiment(
        str(output_dir),
        epochs=args.epochs,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        batch_size=args.batch_size
    )

    # 保存原始结果
    results_path = output_dir / f'results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {results_path}")

    # 生成报告
    generate_report(results, str(output_dir))

    print("\n实验完成!")


if __name__ == '__main__':
    main()
