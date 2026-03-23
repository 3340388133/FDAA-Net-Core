#!/usr/bin/env python3
"""
CIFAKE 补充实验 + SOTA 对比
1. 在 CIFAKE 上训练，测试所有域
2. 用已有模型测试 CIFAKE
3. SOTA 方法对比（CNNDetection, FreqNet, Spec）
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image
import warnings
import json
from datetime import datetime
import torchvision.models as tv_models
warnings.filterwarnings('ignore')


class GenericBinaryDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_samples=None):
        self.transform = transform
        self.samples = []
        extensions = {'.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG'}
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
    def __init__(self, genimage_root, generator_name, split='train', transform=None, max_samples=None):
        self.transform = transform
        self.samples = []
        extensions = {'.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG'}
        gen_dir = os.path.join(genimage_root, generator_name)
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


# ==================== SOTA 方法实现 ====================

class CNNDetectionModel(nn.Module):
    """CNNDetection (Wang et al., CVPR 2020) - ResNet50"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.backbone(x)


class FreqNetModel(nn.Module):
    """FreqNet (AAAI 2024) - Frequency-aware ResNet"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
        # Add frequency branch
        self.freq_conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 256),
        )
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(2048 + 256, num_classes)

    def forward(self, x):
        # Spatial features
        spatial_feat = self.backbone(x)
        # Frequency features (DCT-like)
        freq_input = torch.fft.fft2(x).abs()
        freq_feat = self.freq_conv(freq_input)
        # Combine
        combined = torch.cat([spatial_feat, freq_feat], dim=1)
        return self.classifier(combined)


class SpecModel(nn.Module):
    """Spec (2019) - DCT Spectrum Analysis"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
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
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # Use frequency spectrum as input
        x_freq = torch.fft.fft2(x).abs()
        return self.net(x_freq)


# ==================== 工具函数 ====================

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
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{num_epochs}", leave=False)
        correct, total = 0, 0
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
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix(acc=f"{correct/total:.4f}")
        scheduler.step()
    return model


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  Eval", leave=False):
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
    acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
    return {'auc': auc, 'ap': ap, 'accuracy': acc}


def main():
    print("=" * 80)
    print("CIFAKE 补充实验 + SOTA 方法对比")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 路径
    genimage_root = '/home/customer/文档/zc/0125image/datasets/authoritative/GenImage'
    ntire_root = '/home/customer/文档/zc/0125image/datasets/authoritative/NTIRE2026'
    diffusion_root = '/home/customer/文档/zc/0125image/datasets/authoritative/DiffusionForensics'
    cifake_root = '/home/customer/文档/zc/0125image/datasets/authoritative/CIFAKE'

    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)

    TRAIN_MAX = 20000
    TEST_MAX = 5000

    # 构建所有测试集
    test_datasets = {}
    for gen in ['BigGAN', 'ADM', 'glide', 'VQDM']:
        ds = GenImageSingleGenerator(genimage_root, gen, 'val', test_transform, TEST_MAX)
        if len(ds) == 0:
            ds = GenImageSingleGenerator(genimage_root, gen, 'train', test_transform, TEST_MAX)
        if len(ds) > 0:
            test_datasets[f'GenImage-{gen}'] = ds

    ntire_test = NTIRE2026Dataset(ntire_root, [1], test_transform, TEST_MAX)
    if len(ntire_test) > 0:
        test_datasets['NTIRE2026'] = ntire_test

    diff_test = GenericBinaryDataset(os.path.join(diffusion_root, 'test'), test_transform)
    if len(diff_test) > 0:
        test_datasets['DiffForensics'] = diff_test

    cifake_test = GenericBinaryDataset(os.path.join(cifake_root, 'test'), test_transform, TEST_MAX)
    if len(cifake_test) > 0:
        test_datasets['CIFAKE'] = cifake_test

    domain_names = list(test_datasets.keys())
    print(f"测试域: {domain_names}")

    # ==================== Part 1: CIFAKE 跨域实验 ====================
    print("\n" + "=" * 80)
    print("[Part 1] CIFAKE 训练 → 跨域测试")
    print("=" * 80)

    cifake_train = GenericBinaryDataset(os.path.join(cifake_root, 'train'), train_transform, TRAIN_MAX)
    print(f"CIFAKE 训练集: {len(cifake_train)}")
    cifake_loader = DataLoader(cifake_train, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    from models.detector import AIGCDetectorLite
    model = AIGCDetectorLite(num_classes=2, img_size=224, embed_dim=768, num_prototypes=4, dropout=0.1).to(device)
    model = train_model(model, cifake_loader, device, num_epochs=10)

    cifake_cross = {}
    for test_name, test_ds in test_datasets.items():
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        results = evaluate_model(model, test_loader, device)
        cifake_cross[test_name] = results
        marker = " ★" if test_name == 'CIFAKE' else ""
        print(f"  CIFAKE → {test_name:20s} AUC: {results['auc']:.4f}, Acc: {results['accuracy']:.4f}{marker}")

    del model
    torch.cuda.empty_cache()

    # ==================== Part 2: 所有模型在 CIFAKE 上测试 ====================
    print("\n" + "=" * 80)
    print("[Part 2] 各域模型 → CIFAKE 测试")
    print("=" * 80)

    train_sources = {
        'GenImage-BigGAN': GenImageSingleGenerator(genimage_root, 'BigGAN', 'train', train_transform, TRAIN_MAX),
        'GenImage-ADM': GenImageSingleGenerator(genimage_root, 'ADM', 'train', train_transform, TRAIN_MAX),
        'GenImage-glide': GenImageSingleGenerator(genimage_root, 'glide', 'train', train_transform, TRAIN_MAX),
        'GenImage-VQDM': GenImageSingleGenerator(genimage_root, 'VQDM', 'train', train_transform, TRAIN_MAX),
        'NTIRE2026': NTIRE2026Dataset(ntire_root, [0], train_transform, TRAIN_MAX),
    }

    cifake_test_loader = DataLoader(test_datasets['CIFAKE'], batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    to_cifake_results = {}
    for src_name, src_ds in train_sources.items():
        print(f"\n  训练: {src_name} ({len(src_ds)} samples)")
        src_loader = DataLoader(src_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

        model = AIGCDetectorLite(num_classes=2, img_size=224, embed_dim=768, num_prototypes=4, dropout=0.1).to(device)
        model = train_model(model, src_loader, device, num_epochs=10)

        results = evaluate_model(model, cifake_test_loader, device)
        to_cifake_results[src_name] = results
        print(f"  {src_name} → CIFAKE: AUC: {results['auc']:.4f}, Acc: {results['accuracy']:.4f}")

        del model
        torch.cuda.empty_cache()

    # ==================== Part 3: SOTA 方法对比 ====================
    print("\n" + "=" * 80)
    print("[Part 3] SOTA 方法对比 (多域联合训练)")
    print("=" * 80)

    # 合并训练数据 (所有域)
    all_train = []
    for gen in ['BigGAN', 'ADM', 'glide', 'VQDM']:
        ds = GenImageSingleGenerator(genimage_root, gen, 'train', train_transform, 10000)
        if len(ds) > 0:
            all_train.append(ds)
    ntire_train = NTIRE2026Dataset(ntire_root, [0], train_transform, 10000)
    if len(ntire_train) > 0:
        all_train.append(ntire_train)
    cifake_tr = GenericBinaryDataset(os.path.join(cifake_root, 'train'), train_transform, 10000)
    if len(cifake_tr) > 0:
        all_train.append(cifake_tr)

    combined_train = ConcatDataset(all_train)
    print(f"SOTA对比训练集: {len(combined_train)} 样本")
    combined_loader = DataLoader(combined_train, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    # 所有方法
    methods = {
        'Ours (AIGCDetectorLite)': lambda: AIGCDetectorLite(num_classes=2, img_size=224, embed_dim=768, num_prototypes=4, dropout=0.1),
        'CNNDetection (ResNet50)': lambda: CNNDetectionModel(num_classes=2),
        'FreqNet': lambda: FreqNetModel(num_classes=2),
        'Spec': lambda: SpecModel(num_classes=2),
    }

    sota_results = {}
    for method_name, model_fn in methods.items():
        print(f"\n--- {method_name} ---")
        model = model_fn().to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"  参数量: {params:,}")

        model = train_model(model, combined_loader, device, num_epochs=10)

        sota_results[method_name] = {}
        method_aucs = []
        for test_name, test_ds in test_datasets.items():
            test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
            results = evaluate_model(model, test_loader, device)
            sota_results[method_name][test_name] = results
            method_aucs.append(results['auc'])
            print(f"  → {test_name:20s} AUC: {results['auc']:.4f}, Acc: {results['accuracy']:.4f}")

        avg_auc = np.mean(method_aucs)
        sota_results[method_name]['avg_auc'] = avg_auc
        print(f"  平均 AUC: {avg_auc:.4f}")

        del model
        torch.cuda.empty_cache()

    # ==================== 保存结果 ====================
    print("\n" + "=" * 80)
    print("保存结果")
    print("=" * 80)

    all_results = {
        'cifake_cross_domain': {k: v for k, v in cifake_cross.items()},
        'to_cifake_results': {k: v for k, v in to_cifake_results.items()},
        'sota_comparison': sota_results,
    }

    with open('outputs/cifake_sota_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    # SOTA对比表格
    print("\n\nSOTA 方法对比 (多域联合训练, AUC):")
    header = f"{'Method':30s}" + "".join(f"{name:>15s}" for name in domain_names) + f"{'Avg':>10s}"
    print(header)
    print("-" * len(header))
    for method_name in methods.keys():
        row = f"{method_name:30s}"
        for test_name in domain_names:
            auc = sota_results[method_name][test_name]['auc']
            row += f"{auc:15.4f}"
        row += f"{sota_results[method_name]['avg_auc']:10.4f}"
        print(row)

    # Save SOTA as CSV
    sota_rows = []
    for method_name in methods.keys():
        row = {'method': method_name}
        for test_name in domain_names:
            row[f'{test_name}_auc'] = sota_results[method_name][test_name]['auc']
            row[f'{test_name}_acc'] = sota_results[method_name][test_name]['accuracy']
        row['avg_auc'] = sota_results[method_name]['avg_auc']
        sota_rows.append(row)
    pd.DataFrame(sota_rows).to_csv('outputs/sota_comparison.csv', index=False)

    print(f"\n完成! {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
