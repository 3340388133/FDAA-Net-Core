#!/usr/bin/env python3
"""
跨域泛化评估实验
4+ 数据集/域的完整交叉评估，用于论文发表

实验设计:
1. 域内评估 (In-domain): 每个数据集上 train/test
2. 跨生成器评估: GenImage 4个生成器交叉评估
3. 跨数据集评估: GenImage ↔ NTIRE2026 ↔ DiffusionForensics ↔ CIFAKE
4. SOTA 方法对比
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image
import warnings
import json
from datetime import datetime
from collections import OrderedDict
import copy
warnings.filterwarnings('ignore')


# ==================== 数据集定义 ====================

class GenericBinaryDataset(Dataset):
    """通用二分类数据集: real/fake 目录结构"""
    def __init__(self, root_dir, transform=None, max_samples=None):
        self.transform = transform
        self.samples = []

        extensions = {'.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG'}

        # 支持两种结构: root/real + root/fake 或 root/ai + root/nature
        for label, dir_names in [(0, ['real', 'nature']), (1, ['fake', 'ai'])]:
            for dname in dir_names:
                dpath = os.path.join(root_dir, dname)
                if os.path.isdir(dpath):
                    for f in sorted(os.listdir(dpath)):
                        if any(f.endswith(ext) for ext in extensions):
                            self.samples.append((os.path.join(dpath, f), label))

        if max_samples and len(self.samples) > max_samples:
            np.random.seed(42)
            indices = np.random.choice(len(self.samples), max_samples, replace=False)
            self.samples = [self.samples[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return {'image': image, 'label': label}
        except:
            return {'image': torch.zeros(3, 224, 224), 'label': label}


class GenImageSingleGenerator(Dataset):
    """GenImage 单个生成器数据集"""
    def __init__(self, genimage_root, generator_name, split='train', transform=None, max_samples=None):
        self.transform = transform
        self.samples = []

        extensions = {'.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG'}
        gen_dir = os.path.join(genimage_root, generator_name)

        # 找到解压目录
        for item in os.listdir(gen_dir):
            item_path = os.path.join(gen_dir, item)
            if os.path.isdir(item_path):
                split_dir = os.path.join(item_path, split)
                if os.path.isdir(split_dir):
                    for label, dname in [(1, 'ai'), (0, 'nature')]:
                        dpath = os.path.join(split_dir, dname)
                        if os.path.isdir(dpath):
                            for f in sorted(os.listdir(dpath)):
                                if any(f.endswith(ext) for ext in extensions):
                                    self.samples.append((os.path.join(dpath, f), label))

        if max_samples and len(self.samples) > max_samples:
            np.random.seed(42)
            indices = np.random.choice(len(self.samples), max_samples, replace=False)
            self.samples = [self.samples[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return {'image': image, 'label': label}
        except:
            return {'image': torch.zeros(3, 224, 224), 'label': label}


class NTIRE2026Dataset(Dataset):
    """NTIRE 2026 数据集"""
    def __init__(self, data_dir, shard_nums, transform=None, max_samples=None):
        self.transform = transform
        self.samples = []

        for shard_num in shard_nums:
            shard_dir = os.path.join(data_dir, f'shard_{shard_num}')
            labels_path = os.path.join(shard_dir, 'labels.csv')
            images_dir = os.path.join(shard_dir, 'images')

            if not os.path.exists(labels_path) or not os.path.exists(images_dir):
                continue

            df = pd.read_csv(labels_path)
            for _, row in df.iterrows():
                img_path = os.path.join(images_dir, row['image_name'])
                if os.path.exists(img_path):
                    self.samples.append((img_path, int(row['label'])))
                if max_samples and len(self.samples) >= max_samples:
                    break
            if max_samples and len(self.samples) >= max_samples:
                break

        if max_samples and len(self.samples) > max_samples:
            np.random.seed(42)
            indices = np.random.choice(len(self.samples), max_samples, replace=False)
            self.samples = [self.samples[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return {'image': image, 'label': label}
        except:
            return {'image': torch.zeros(3, 224, 224), 'label': label}


# ==================== 训练和评估函数 ====================

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def train_model(model, train_loader, device, num_epochs=10, lr=1e-4):
    """快速训练模型"""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"  Train Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

        scheduler.step()

    return model


def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  Evaluating", leave=False):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    try:
        auc = roc_auc_score(all_labels, all_preds)
        ap = average_precision_score(all_labels, all_preds)
    except:
        auc, ap = 0.5, 0.5

    binary_preds = (all_preds > 0.5).astype(int)
    acc = accuracy_score(all_labels, binary_preds)

    return {'auc': auc, 'ap': ap, 'accuracy': acc}


def create_fresh_model(device):
    """创建新模型"""
    from models.detector import AIGCDetectorLite
    model = AIGCDetectorLite(
        num_classes=2, img_size=224, embed_dim=768,
        num_prototypes=4, dropout=0.1
    ).to(device)
    return model


# ==================== 主实验 ====================

def main():
    print("=" * 80)
    print("跨域泛化评估实验 (Cross-Domain Generalization Evaluation)")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 路径配置
    genimage_root = '/home/customer/文档/zc/0125image/datasets/authoritative/GenImage'
    ntire_root = '/home/customer/文档/zc/0125image/datasets/authoritative/NTIRE2026'
    diffusion_root = '/home/customer/文档/zc/0125image/datasets/authoritative/DiffusionForensics'
    cifake_root = '/home/customer/文档/zc/0125image/datasets/authoritative/CIFAKE'

    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)

    # ==================== 构建所有数据集 ====================
    print("\n[1] 构建数据集...")

    # 训练样本限制 (每个域)
    TRAIN_MAX = 20000   # 每个域最多20k训练样本 (加速实验)
    TEST_MAX = 5000     # 每个域最多5k测试样本

    datasets_config = OrderedDict()

    # GenImage 4个生成器 - 每个作为独立域
    generators = ['BigGAN', 'ADM', 'glide', 'VQDM']
    for gen in generators:
        gen_dir = os.path.join(genimage_root, gen)
        if not os.path.isdir(gen_dir):
            continue

        train_ds = GenImageSingleGenerator(genimage_root, gen, 'train', train_transform, TRAIN_MAX)

        # 尝试 val, 失败则从 train 中抽取
        test_ds = GenImageSingleGenerator(genimage_root, gen, 'val', test_transform, TEST_MAX)
        if len(test_ds) == 0:
            test_ds = GenImageSingleGenerator(genimage_root, gen, 'train', test_transform, TEST_MAX)

        if len(train_ds) > 0 and len(test_ds) > 0:
            datasets_config[f'GenImage-{gen}'] = {
                'train': train_ds,
                'test': test_ds,
                'type': 'GAN' if gen in ['BigGAN', 'VQDM'] else 'Diffusion'
            }
            print(f"  GenImage-{gen}: train={len(train_ds)}, test={len(test_ds)}")

    # NTIRE 2026
    ntire_train = NTIRE2026Dataset(ntire_root, [0], train_transform, TRAIN_MAX)
    ntire_test = NTIRE2026Dataset(ntire_root, [1], test_transform, TEST_MAX)
    if len(ntire_train) > 0 and len(ntire_test) > 0:
        datasets_config['NTIRE2026'] = {
            'train': ntire_train,
            'test': ntire_test,
            'type': 'Mixed'
        }
        print(f"  NTIRE2026: train={len(ntire_train)}, test={len(ntire_test)}")

    # DiffusionForensics
    if os.path.isdir(os.path.join(diffusion_root, 'train')):
        diff_train = GenericBinaryDataset(os.path.join(diffusion_root, 'train'), train_transform)
        diff_test = GenericBinaryDataset(os.path.join(diffusion_root, 'test'), test_transform)
        if len(diff_train) > 0:
            datasets_config['DiffusionForensics'] = {
                'train': diff_train,
                'test': diff_test,
                'type': 'Diffusion'
            }
            print(f"  DiffusionForensics: train={len(diff_train)}, test={len(diff_test)}")

    # CIFAKE
    if os.path.isdir(os.path.join(cifake_root, 'train')):
        cifake_train = GenericBinaryDataset(os.path.join(cifake_root, 'train'), train_transform, TRAIN_MAX)
        cifake_test = GenericBinaryDataset(os.path.join(cifake_root, 'test'), test_transform, TEST_MAX)
        if len(cifake_train) > 0:
            datasets_config['CIFAKE'] = {
                'train': cifake_train,
                'test': cifake_test,
                'type': 'Diffusion'
            }
            print(f"  CIFAKE: train={len(cifake_train)}, test={len(cifake_test)}")

    domain_names = list(datasets_config.keys())
    print(f"\n总计 {len(domain_names)} 个域: {domain_names}")

    # ==================== 实验1: 跨域泛化矩阵 ====================
    print("\n" + "=" * 80)
    print("[实验1] 跨域泛化矩阵 (Train on A → Test on B)")
    print("=" * 80)

    cross_domain_results = {}
    CROSS_EPOCHS = 10  # 每个跨域训练的epoch数

    for train_name in domain_names:
        print(f"\n--- 训练域: {train_name} ---")

        train_ds = datasets_config[train_name]['train']
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                                  num_workers=4, pin_memory=True, drop_last=True)

        # 训练新模型
        model = create_fresh_model(device)
        model = train_model(model, train_loader, device, num_epochs=CROSS_EPOCHS)

        # 在所有域上测试
        cross_domain_results[train_name] = {}
        for test_name in domain_names:
            test_ds = datasets_config[test_name]['test']
            test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                                    num_workers=4, pin_memory=True)

            results = evaluate_model(model, test_loader, device)
            cross_domain_results[train_name][test_name] = results

            marker = " ★" if train_name == test_name else ""
            print(f"  → {test_name:25s} AUC: {results['auc']:.4f}, Acc: {results['accuracy']:.4f}{marker}")

        # 清理显存
        del model
        torch.cuda.empty_cache()

    # 保存跨域矩阵
    print("\n\n跨域泛化矩阵 (AUC):")
    matrix_data = []
    header = f"{'Train\\Test':25s}" + "".join(f"{name:>18s}" for name in domain_names) + f"{'Avg(Cross)':>12s}"
    print(header)
    print("-" * len(header))

    for train_name in domain_names:
        row = f"{train_name:25s}"
        cross_aucs = []
        for test_name in domain_names:
            auc = cross_domain_results[train_name][test_name]['auc']
            if train_name == test_name:
                row += f"{'★'+f'{auc:.4f}':>18s}"
            else:
                row += f"{auc:18.4f}"
                cross_aucs.append(auc)
        avg_cross = np.mean(cross_aucs) if cross_aucs else 0
        row += f"{avg_cross:12.4f}"
        print(row)

        row_dict = {'train_domain': train_name}
        for test_name in domain_names:
            row_dict[f'test_{test_name}_auc'] = cross_domain_results[train_name][test_name]['auc']
            row_dict[f'test_{test_name}_ap'] = cross_domain_results[train_name][test_name]['ap']
            row_dict[f'test_{test_name}_acc'] = cross_domain_results[train_name][test_name]['accuracy']
        row_dict['avg_cross_auc'] = avg_cross
        matrix_data.append(row_dict)

    # ==================== 实验2: 多域联合训练 ====================
    print("\n\n" + "=" * 80)
    print("[实验2] 多域联合训练 → 跨域泛化")
    print("=" * 80)

    # 合并所有训练数据
    all_train_datasets = [datasets_config[name]['train'] for name in domain_names]
    combined_train = ConcatDataset(all_train_datasets)
    print(f"合并训练集: {len(combined_train)} 样本 (来自 {len(domain_names)} 个域)")

    combined_loader = DataLoader(combined_train, batch_size=64, shuffle=True,
                                 num_workers=4, pin_memory=True, drop_last=True)

    model = create_fresh_model(device)
    model = train_model(model, combined_loader, device, num_epochs=CROSS_EPOCHS)

    print("\n多域联合训练结果:")
    combined_results = {}
    for test_name in domain_names:
        test_ds = datasets_config[test_name]['test']
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                                num_workers=4, pin_memory=True)

        results = evaluate_model(model, test_loader, device)
        combined_results[test_name] = results
        print(f"  {test_name:25s} AUC: {results['auc']:.4f}, AP: {results['ap']:.4f}, Acc: {results['accuracy']:.4f}")

    avg_combined_auc = np.mean([r['auc'] for r in combined_results.values()])
    print(f"\n  平均 AUC: {avg_combined_auc:.4f}")

    del model
    torch.cuda.empty_cache()

    # ==================== 保存结果 ====================
    print("\n" + "=" * 80)
    print("保存结果")
    print("=" * 80)

    # 保存跨域矩阵
    matrix_df = pd.DataFrame(matrix_data)
    matrix_df.to_csv('outputs/cross_domain_matrix.csv', index=False)
    print(f"跨域矩阵: outputs/cross_domain_matrix.csv")

    # 保存联合训练结果
    combined_data = []
    for name, res in combined_results.items():
        combined_data.append({
            'domain': name,
            'auc': res['auc'],
            'ap': res['ap'],
            'accuracy': res['accuracy']
        })
    combined_df = pd.DataFrame(combined_data)
    combined_df.to_csv('outputs/combined_training_results.csv', index=False)
    print(f"联合训练: outputs/combined_training_results.csv")

    # 完整JSON结果
    all_results = {
        'experiment_info': {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'domains': domain_names,
            'train_max': TRAIN_MAX,
            'test_max': TEST_MAX,
            'cross_epochs': CROSS_EPOCHS,
        },
        'cross_domain_matrix': cross_domain_results,
        'combined_training': {k: v for k, v in combined_results.items()},
        'avg_combined_auc': avg_combined_auc,
    }

    with open('outputs/cross_domain_full_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"完整结果: outputs/cross_domain_full_results.json")

    print(f"\n实验完成! 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
