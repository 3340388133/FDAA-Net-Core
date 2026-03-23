#!/usr/bin/env python3
"""
消融实验（从头训练 backbone）：验证FDAA和MGFP的贡献
4组实验：
  1. Baseline: 6层ViT (从头训练) + 分类头
  2. +FDAA: 6层ViT + FDAA + 分类头
  3. +MGFP: 6层ViT + MGFP + 分类头
  4. Full: 6层ViT + FDAA + MGFP + 分类头
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
warnings.filterwarnings('ignore')

from models.modules.fdaa import FDAA
from models.modules.mgfp import MGFP


# ==================== 消融模型（从头训练） ====================

class AblationModelScratch(nn.Module):
    """从头训练的消融模型，6层ViT backbone"""
    def __init__(self, num_classes=2, img_size=224, embed_dim=768,
                 num_prototypes=4, dropout=0.1,
                 use_fdaa=True, use_mgfp=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_fdaa = use_fdaa
        self.use_mgfp = use_mgfp
        self.patch_size = 16
        self.img_size = img_size
        self.num_patches = (img_size // self.patch_size) ** 2

        # 从头训练的ViT backbone (6层, 8头)
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=embed_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # FDAA
        if use_fdaa:
            self.fdaa = FDAA(
                dim=embed_dim, img_size=img_size,
                patch_size=self.patch_size, reduction=4,
                num_heads=8, dropout=dropout
            )

        # MGFP
        if use_mgfp:
            self.mgfp = MGFP(
                dim=embed_dim, num_patches=self.num_patches,
                num_prototypes=num_prototypes,
                use_hierarchical=False, dropout=dropout
            )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, image, return_attention=False):
        B = image.shape[0]

        x = self.patch_embed(image)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.encoder(x)

        cls_token = x[:, 0]
        patch_tokens = x[:, 1:]

        if self.use_fdaa:
            patch_tokens = self.fdaa(patch_tokens, image)

        if self.use_mgfp:
            output, _ = self.mgfp(cls_token, patch_tokens, return_attention=True)
        else:
            output = cls_token

        logits = self.classifier(output)
        return {'logits': logits}


# ==================== 数据集 ====================

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

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform: image = self.transform(image)
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

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform: image = self.transform(image)
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

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform: image = self.transform(image)
            return {'image': image, 'label': label}
        except:
            return {'image': torch.zeros(3, 224, 224), 'label': label}


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
                loss = criterion(outputs['logits'], labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            _, predicted = outputs['logits'].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix(acc=f"{correct/total:.4f}")
        scheduler.step()
        print(f"    Epoch {epoch+1}: train_acc={correct/total:.4f}")
    return model


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  Eval", leave=False):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            with torch.cuda.amp.autocast():
                outputs = model(images)
            probs = torch.softmax(outputs['logits'], dim=1)[:, 1].cpu().numpy()
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
    print("消融实验 (从头训练 backbone): FDAA 和 MGFP 贡献分析")
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
    TEST_MAX = 5000

    # 测试集
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

    # 训练集
    all_train = []
    per_domain = 10000
    for gen in ['BigGAN', 'ADM', 'glide', 'VQDM']:
        ds = GenImageSingleGenerator(genimage_root, gen, 'train', train_transform, per_domain)
        if len(ds) > 0: all_train.append(ds)
    ntire_train = NTIRE2026Dataset(ntire_root, [0], train_transform, per_domain)
    if len(ntire_train) > 0: all_train.append(ntire_train)
    cifake_tr = GenericBinaryDataset(os.path.join(cifake_root, 'train'), train_transform, per_domain)
    if len(cifake_tr) > 0: all_train.append(cifake_tr)
    combined_train = ConcatDataset(all_train)
    print(f"训练集: {len(combined_train)} 样本")
    combined_loader = DataLoader(combined_train, batch_size=64, shuffle=True,
                                 num_workers=4, pin_memory=True, drop_last=True)

    NUM_EPOCHS = 10
    LR = 1e-4

    ablation_configs = [
        ('Baseline (ViT only)',    False, False),
        ('+ FDAA',                 True,  False),
        ('+ MGFP',                 False, True),
        ('Full (FDAA + MGFP)',     True,  True),
    ]

    all_results = {}

    for variant_name, use_fdaa, use_mgfp in ablation_configs:
        print(f"\n{'='*60}")
        print(f"--- {variant_name} ---")
        fdaa_str = '✓' if use_fdaa else '✗'
        mgfp_str = '✓' if use_mgfp else '✗'
        print(f"  FDAA: {fdaa_str}  MGFP: {mgfp_str}")
        print(f"{'='*60}")

        model = AblationModelScratch(
            num_classes=2, img_size=224, embed_dim=768,
            num_prototypes=4, dropout=0.1,
            use_fdaa=use_fdaa, use_mgfp=use_mgfp
        ).to(device)

        total_p = sum(p.numel() for p in model.parameters())
        print(f"  参数: {total_p:,}")

        model = train_model(model, combined_loader, device, num_epochs=NUM_EPOCHS, lr=LR)

        all_results[variant_name] = {'params': total_p, 'use_fdaa': use_fdaa, 'use_mgfp': use_mgfp}
        variant_aucs = []
        for test_name, test_ds in test_datasets.items():
            test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                                    num_workers=4, pin_memory=True)
            results = evaluate_model(model, test_loader, device)
            all_results[variant_name][test_name] = results
            variant_aucs.append(results['auc'])
            print(f"  → {test_name:20s} AUC: {results['auc']:.4f}, AP: {results['ap']:.4f}, Acc: {results['accuracy']:.4f}")

        avg_auc = np.mean(variant_aucs)
        all_results[variant_name]['avg_auc'] = avg_auc
        print(f"  ★ 平均 AUC: {avg_auc:.4f}")

        del model
        torch.cuda.empty_cache()

    # 汇总
    print("\n" + "=" * 80)
    print("消融实验结果汇总 (从头训练)")
    print("=" * 80)

    header = f"{'Variant':25s} {'FDAA':>5s} {'MGFP':>5s} {'Params':>12s}"
    for name in domain_names:
        header += f" {name:>12s}"
    header += f" {'Avg AUC':>10s}"
    print(header)
    print("-" * len(header))

    for variant_name, use_fdaa, use_mgfp in ablation_configs:
        r = all_results[variant_name]
        fdaa_s = '✓' if use_fdaa else '✗'
        mgfp_s = '✓' if use_mgfp else '✗'
        row = f"{variant_name:25s} {fdaa_s:>5s} {mgfp_s:>5s} {r['params']:>12,d}"
        for test_name in domain_names:
            row += f" {r[test_name]['auc']:12.4f}"
        row += f" {r['avg_auc']:10.4f}"
        print(row)

    # 保存
    with open('outputs/ablation_scratch_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    rows = []
    for variant_name, use_fdaa, use_mgfp in ablation_configs:
        r = all_results[variant_name]
        row = {'variant': variant_name, 'use_fdaa': use_fdaa, 'use_mgfp': use_mgfp, 'params': r['params']}
        for test_name in domain_names:
            row[f'{test_name}_auc'] = r[test_name]['auc']
            row[f'{test_name}_ap'] = r[test_name]['ap']
            row[f'{test_name}_acc'] = r[test_name]['accuracy']
        row['avg_auc'] = r['avg_auc']
        rows.append(row)
    pd.DataFrame(rows).to_csv('outputs/ablation_scratch_results.csv', index=False)

    print(f"\n结果已保存: outputs/ablation_scratch_results.json / .csv")
    print(f"完成! {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
