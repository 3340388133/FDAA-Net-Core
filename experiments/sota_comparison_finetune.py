#!/usr/bin/env python3
"""
SOTA方法对比 - 全参数微调版本
我们的方法使用预训练ViT backbone + FDAA/MGFP，全参数微调
与SOTA方法公平对比
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


# ==================== 数据集 (复用) ====================

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


def train_with_diff_lr(model, train_loader, device, num_epochs=10,
                        backbone_lr=1e-5, head_lr=5e-4):
    """差异学习率训练：backbone低学习率，新模块高学习率"""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 分组参数
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone_encoder' in name or 'patch_embed' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params, 'lr': head_lr},
    ], weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)
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
        print(f"    Epoch {epoch+1}: train_acc={correct/total:.4f}")
    return model


def train_standard(model, train_loader, device, num_epochs=10, lr=1e-4):
    """标准训练"""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
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
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            _, predicted = logits.max(1)
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
    print("SOTA方法对比 - 全参数微调 (差异学习率)")
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

    # 构建测试集
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
    print(f"测试域 ({len(domain_names)}): {domain_names}")

    # 构建联合训练集
    all_train = []
    per_domain = 10000
    for gen in ['BigGAN', 'ADM', 'glide', 'VQDM']:
        ds = GenImageSingleGenerator(genimage_root, gen, 'train', train_transform, per_domain)
        if len(ds) > 0:
            all_train.append(ds)
    ntire_train = NTIRE2026Dataset(ntire_root, [0], train_transform, per_domain)
    if len(ntire_train) > 0:
        all_train.append(ntire_train)
    cifake_tr = GenericBinaryDataset(os.path.join(cifake_root, 'train'), train_transform, per_domain)
    if len(cifake_tr) > 0:
        all_train.append(cifake_tr)

    combined_train = ConcatDataset(all_train)
    print(f"联合训练集: {len(combined_train)} 样本")
    combined_loader = DataLoader(combined_train, batch_size=64, shuffle=True,
                                 num_workers=4, pin_memory=True, drop_last=True)

    # ==================== 方法定义 ====================
    from models.detector import AIGCDetectorLite
    from models.sota_methods import CNNDetector, FreqNetDetector, F3NetDetector

    NUM_EPOCHS = 10

    # ==================== 1. Ours (全参数微调) ====================
    print(f"\n{'='*60}")
    print("--- Ours (FDAA+MGFP) 全参数微调 ---")
    print(f"{'='*60}")

    model = AIGCDetectorLite(
        num_classes=2, img_size=224, embed_dim=768,
        num_prototypes=4, dropout=0.1, pretrained=True
    ).to(device)

    # 解冻backbone
    for param in model.parameters():
        param.requires_grad = True

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_p:,}, 可训练: {train_p:,}")

    # 差异学习率: backbone 1e-5, FDAA/MGFP/classifier 5e-4
    model = train_with_diff_lr(model, combined_loader, device,
                                num_epochs=NUM_EPOCHS,
                                backbone_lr=1e-5, head_lr=5e-4)

    ours_results = {}
    ours_aucs = []
    for test_name, test_ds in test_datasets.items():
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        results = evaluate_model(model, test_loader, device)
        ours_results[test_name] = results
        ours_aucs.append(results['auc'])
        print(f"  → {test_name:20s} AUC: {results['auc']:.4f}, AP: {results['ap']:.4f}, Acc: {results['accuracy']:.4f}")
    ours_avg = np.mean(ours_aucs)
    print(f"  ★ 平均 AUC: {ours_avg:.4f}")
    del model
    torch.cuda.empty_cache()

    # ==================== 2. CNNDetection ====================
    print(f"\n{'='*60}")
    print("--- CNNDetection ---")
    print(f"{'='*60}")

    model = CNNDetector(num_classes=2, pretrained=True).to(device)
    print(f"  参数: {sum(p.numel() for p in model.parameters()):,}")
    model = train_standard(model, combined_loader, device, num_epochs=NUM_EPOCHS, lr=1e-4)

    cnn_results = {}
    cnn_aucs = []
    for test_name, test_ds in test_datasets.items():
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        results = evaluate_model(model, test_loader, device)
        cnn_results[test_name] = results
        cnn_aucs.append(results['auc'])
        print(f"  → {test_name:20s} AUC: {results['auc']:.4f}, AP: {results['ap']:.4f}, Acc: {results['accuracy']:.4f}")
    cnn_avg = np.mean(cnn_aucs)
    print(f"  ★ 平均 AUC: {cnn_avg:.4f}")
    del model
    torch.cuda.empty_cache()

    # ==================== 3. FreqNet ====================
    print(f"\n{'='*60}")
    print("--- FreqNet ---")
    print(f"{'='*60}")

    model = FreqNetDetector(num_classes=2, pretrained=True).to(device)
    print(f"  参数: {sum(p.numel() for p in model.parameters()):,}")
    model = train_standard(model, combined_loader, device, num_epochs=NUM_EPOCHS, lr=1e-4)

    freq_results = {}
    freq_aucs = []
    for test_name, test_ds in test_datasets.items():
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        results = evaluate_model(model, test_loader, device)
        freq_results[test_name] = results
        freq_aucs.append(results['auc'])
        print(f"  → {test_name:20s} AUC: {results['auc']:.4f}, AP: {results['ap']:.4f}, Acc: {results['accuracy']:.4f}")
    freq_avg = np.mean(freq_aucs)
    print(f"  ★ 平均 AUC: {freq_avg:.4f}")
    del model
    torch.cuda.empty_cache()

    # ==================== 4. F3Net ====================
    print(f"\n{'='*60}")
    print("--- F3Net ---")
    print(f"{'='*60}")

    model = F3NetDetector(num_classes=2, pretrained=True).to(device)
    print(f"  参数: {sum(p.numel() for p in model.parameters()):,}")
    model = train_standard(model, combined_loader, device, num_epochs=NUM_EPOCHS, lr=1e-4)

    f3_results = {}
    f3_aucs = []
    for test_name, test_ds in test_datasets.items():
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        results = evaluate_model(model, test_loader, device)
        f3_results[test_name] = results
        f3_aucs.append(results['auc'])
        print(f"  → {test_name:20s} AUC: {results['auc']:.4f}, AP: {results['ap']:.4f}, Acc: {results['accuracy']:.4f}")
    f3_avg = np.mean(f3_aucs)
    print(f"  ★ 平均 AUC: {f3_avg:.4f}")
    del model
    torch.cuda.empty_cache()

    # ==================== 汇总 ====================
    print("\n" + "=" * 80)
    print("SOTA对比结果汇总 (全参数微调)")
    print("=" * 80)

    all_methods = {
        'Ours (FDAA+MGFP)': (ours_results, ours_avg),
        'CNNDetection': (cnn_results, cnn_avg),
        'FreqNet': (freq_results, freq_avg),
        'F3Net': (f3_results, f3_avg),
    }

    header = f"{'Method':25s}"
    for name in domain_names:
        header += f" {name:>15s}"
    header += f" {'Avg AUC':>10s}"
    print(header)
    print("-" * len(header))

    for method_name, (results, avg) in all_methods.items():
        row = f"{method_name:25s}"
        for test_name in domain_names:
            row += f" {results[test_name]['auc']:15.4f}"
        row += f" {avg:10.4f}"
        print(row)

    # 保存
    save_data = {}
    for method_name, (results, avg) in all_methods.items():
        save_data[method_name] = {**results, 'avg_auc': avg}

    with open('outputs/sota_comparison_finetune.json', 'w') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)

    rows = []
    for method_name, (results, avg) in all_methods.items():
        row = {'method': method_name}
        for test_name in domain_names:
            for metric in ['auc', 'ap', 'accuracy']:
                row[f'{test_name}_{metric}'] = results[test_name][metric]
        row['avg_auc'] = avg
        rows.append(row)
    pd.DataFrame(rows).to_csv('outputs/sota_comparison_finetune.csv', index=False)

    print(f"\n结果已保存: outputs/sota_comparison_finetune.json")
    print(f"结果已保存: outputs/sota_comparison_finetune.csv")
    print(f"\n完成! {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
