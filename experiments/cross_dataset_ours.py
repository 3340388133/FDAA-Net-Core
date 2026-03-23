#!/usr/bin/env python3
"""
跨数据集泛化实验 - 仅我们的方法
训练: BigGAN + ADM
测试: glide, VQDM, CIFAKE, DiffusionForensics, NTIRE2026 (均为未见过的数据集)
"""

import os
import sys
import json
import random
import time
import csv
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import numpy as np
from tqdm import tqdm
from PIL import Image
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.detector import AIGCDetectorLite

# ======== 数据集定义 ========

class GenImageDataset(Dataset):
    """GenImage 格式: root/split/ai + root/split/nature"""
    def __init__(self, data_root, split='train', transform=None, max_samples=None):
        self.transform = transform
        self.samples = []
        split_dir = Path(data_root) / split

        ai_dir = next((split_dir / d for d in ['ai', 'fake', '1'] if (split_dir / d).exists()), None)
        nat_dir = next((split_dir / d for d in ['nature', 'real', '0'] if (split_dir / d).exists()), None)

        if ai_dir is None or nat_dir is None:
            raise ValueError(f"目录不完整: {data_root}/{split}, ai={ai_dir}, nat={nat_dir}")

        exts = ['*.png', '*.jpg', '*.jpeg', '*.JPEG', '*.JPG', '*.PNG', '*.webp']
        ai_imgs = []
        nat_imgs = []
        for ext in exts:
            ai_imgs.extend(list(ai_dir.glob(ext)))
            nat_imgs.extend(list(nat_dir.glob(ext)))

        random.shuffle(ai_imgs)
        random.shuffle(nat_imgs)

        if max_samples:
            n = max_samples // 2
            ai_imgs = ai_imgs[:n]
            nat_imgs = nat_imgs[:n]

        self.samples = [(p, 1) for p in ai_imgs] + [(p, 0) for p in nat_imgs]
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except:
            return self.__getitem__(random.randint(0, len(self.samples) - 1))
        if self.transform:
            img = self.transform(img)
        return {'image': img, 'label': label}


class CIFAKEDataset(Dataset):
    """CIFAKE: root/split/real + root/split/fake"""
    def __init__(self, data_root, split='test', transform=None, max_samples=None):
        self.transform = transform
        self.samples = []
        split_dir = Path(data_root) / split

        real_dir = split_dir / 'real'
        fake_dir = split_dir / 'fake'

        if not real_dir.exists() or not fake_dir.exists():
            # 尝试 REAL/FAKE
            real_dir = split_dir / 'REAL'
            fake_dir = split_dir / 'FAKE'

        exts = ['*.png', '*.jpg', '*.jpeg', '*.JPEG']
        real_imgs, fake_imgs = [], []
        for ext in exts:
            real_imgs.extend(list(real_dir.glob(ext)))
            fake_imgs.extend(list(fake_dir.glob(ext)))

        random.shuffle(real_imgs)
        random.shuffle(fake_imgs)

        if max_samples:
            n = max_samples // 2
            real_imgs = real_imgs[:n]
            fake_imgs = fake_imgs[:n]

        self.samples = [(p, 0) for p in real_imgs] + [(p, 1) for p in fake_imgs]
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except:
            return self.__getitem__(random.randint(0, len(self.samples) - 1))
        if self.transform:
            img = self.transform(img)
        return {'image': img, 'label': label}


class NTIRE2026Dataset(Dataset):
    """NTIRE2026: shard_X/images/ + shard_X/labels.csv"""
    def __init__(self, data_root, shards=['shard_0', 'shard_1'], transform=None, max_samples=None):
        self.transform = transform
        self.samples = []

        for shard in shards:
            shard_dir = Path(data_root) / shard
            label_file = shard_dir / 'labels.csv'
            img_dir = shard_dir / 'images'

            if not label_file.exists():
                continue

            df = pd.read_csv(label_file)
            for _, row in df.iterrows():
                img_path = img_dir / row['image_name']
                if img_path.exists():
                    self.samples.append((img_path, int(row['label'])))

        random.shuffle(self.samples)
        if max_samples and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except:
            return self.__getitem__(random.randint(0, len(self.samples) - 1))
        if self.transform:
            img = self.transform(img)
        return {'image': img, 'label': label}


# ======== 训练/评估函数 ========

def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Train {epoch}/{total_epochs}", leave=False)
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        with autocast():
            out = model(images)
            logits = out['logits'] if isinstance(out, dict) else out
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})

    return total_loss / len(loader), correct / total


def evaluate(model, loader, device, desc="Test"):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            with autocast():
                out = model(images)
                logits = out['logits'] if isinstance(out, dict) else out

            probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    try:
        auc = roc_auc_score(all_labels, all_preds)
        ap = average_precision_score(all_labels, all_preds)
    except ValueError:
        auc, ap = 0.5, 0.5

    acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
    return {'auc': auc, 'ap': ap, 'accuracy': acc}


# ======== 主程序 ========

def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    BASE = "/home/customer/文档/zc/0125image/datasets/authoritative"

    # ---- 数据集路径 ----
    TRAIN_DATASETS = {
        'BigGAN': f"{BASE}/GenImage/BigGAN/imagenet_ai_0419_biggan",
        'ADM': f"{BASE}/GenImage/ADM/imagenet_ai_0508_adm",
    }

    TEST_DATASETS = {
        'glide': ('genimage', f"{BASE}/GenImage/glide/imagenet_glide"),
        'VQDM': ('genimage', f"{BASE}/GenImage/VQDM/imagenet_ai_0419_vqdm"),
        'CIFAKE': ('cifake', f"{BASE}/CIFAKE"),
        'DiffusionForensics': ('genimage', f"{BASE}/DiffusionForensics"),
        'NTIRE2026': ('ntire', f"{BASE}/NTIRE2026"),
    }

    # ---- 超参数 ----
    TRAIN_SAMPLES_PER_SOURCE = 15000
    TEST_SAMPLES = 5000       # 每个测试集最多5000样本
    BATCH_SIZE = 32
    EPOCHS = 15
    BACKBONE_LR = 1e-5
    HEAD_LR = 5e-4

    # ---- 数据变换 ----
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ---- 加载训练数据 ----
    print("\n=== 加载训练数据 ===")
    train_ds_list = []
    for name, path in TRAIN_DATASETS.items():
        ds = GenImageDataset(path, split='train', transform=train_tf,
                             max_samples=TRAIN_SAMPLES_PER_SOURCE)
        train_ds_list.append(ds)
        print(f"  {name}: {len(ds)} samples")

    # 验证集从训练源取
    val_ds_list = []
    for name, path in TRAIN_DATASETS.items():
        try:
            ds = GenImageDataset(path, split='val', transform=test_tf, max_samples=2000)
            val_ds_list.append(ds)
            print(f"  {name} val: {len(ds)} samples")
        except:
            # val 不完整，从 train 取一小部分
            ds = GenImageDataset(path, split='train', transform=test_tf, max_samples=2000)
            val_ds_list.append(ds)
            print(f"  {name} val(from train): {len(ds)} samples")

    combined_train = ConcatDataset(train_ds_list)
    combined_val = ConcatDataset(val_ds_list)
    print(f"\n  总训练: {len(combined_train)}, 总验证: {len(combined_val)}")

    train_loader = DataLoader(combined_train, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(combined_val, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    # ---- 加载测试数据 ----
    print("\n=== 加载跨数据集测试数据 ===")
    test_loaders = {}
    for name, (dtype, path) in TEST_DATASETS.items():
        try:
            if dtype == 'genimage':
                # 优先用 val, 否则用 train 的一部分
                try:
                    ds = GenImageDataset(path, split='val', transform=test_tf,
                                         max_samples=TEST_SAMPLES)
                except:
                    ds = GenImageDataset(path, split='test', transform=test_tf,
                                         max_samples=TEST_SAMPLES)
            elif dtype == 'cifake':
                ds = CIFAKEDataset(path, split='test', transform=test_tf,
                                    max_samples=TEST_SAMPLES)
            elif dtype == 'ntire':
                ds = NTIRE2026Dataset(path, shards=['shard_0', 'shard_1'],
                                       transform=test_tf, max_samples=TEST_SAMPLES)
            else:
                continue

            test_loaders[name] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                                            num_workers=4, pin_memory=True)
            print(f"  {name}: {len(ds)} samples")
        except Exception as e:
            print(f"  {name}: 加载失败 - {e}")

    # 也加入训练域测试 (域内基准)
    for name, path in TRAIN_DATASETS.items():
        try:
            ds = GenImageDataset(path, split='val', transform=test_tf, max_samples=TEST_SAMPLES)
            test_loaders[f"{name}(in-domain)"] = DataLoader(
                ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
            print(f"  {name}(in-domain): {len(ds)} samples")
        except:
            ds = GenImageDataset(path, split='train', transform=test_tf, max_samples=TEST_SAMPLES)
            test_loaders[f"{name}(in-domain)"] = DataLoader(
                ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
            print(f"  {name}(in-domain, from train): {len(ds)} samples")

    # ---- 创建模型 ----
    print("\n=== 创建模型 (AIGCDetectorLite + pretrained ViT-B/16) ===")
    model = AIGCDetectorLite(
        num_classes=2, img_size=224, embed_dim=768,
        num_prototypes=8, dropout=0.1, pretrained=True
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total: {total_params/1e6:.1f}M, Trainable: {trainable_params/1e6:.1f}M")

    # 差分学习率
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name or 'vit' in name or 'patch_embed' in name or 'blocks' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': BACKBONE_LR},
        {'params': head_params, 'lr': HEAD_LR},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    # ---- 训练 ----
    print(f"\n=== 开始训练 ({EPOCHS} epochs) ===")
    best_val_auc = 0
    best_state = None
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, EPOCHS
        )
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device, desc=f"Val {epoch}")
        print(f"  Epoch {epoch}: loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_auc={val_metrics['auc']:.4f}, val_acc={val_metrics['accuracy']:.4f}")

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.time() - start_time
    print(f"\n训练完成, 耗时 {train_time/60:.1f} min, 最佳 val AUC: {best_val_auc:.4f}")

    # 加载最佳模型
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # ---- 跨数据集测试 ----
    print(f"\n{'='*70}")
    print("跨数据集测试结果")
    print(f"{'='*70}")
    print(f"训练数据: {list(TRAIN_DATASETS.keys())}")
    print(f"{'='*70}")

    results = {}
    cross_aucs = []

    for test_name, loader in test_loaders.items():
        metrics = evaluate(model, loader, device, desc=test_name)
        results[test_name] = metrics
        is_cross = '(in-domain)' not in test_name
        marker = " [CROSS-DATASET]" if is_cross else " [IN-DOMAIN]"
        print(f"  {test_name:25s}: AUC={metrics['auc']:.4f}, AP={metrics['ap']:.4f}, "
              f"Acc={metrics['accuracy']:.4f}{marker}")
        if is_cross:
            cross_aucs.append(metrics['auc'])

    avg_cross_auc = np.mean(cross_aucs) if cross_aucs else 0
    print(f"\n  跨数据集平均 AUC: {avg_cross_auc:.4f} ({len(cross_aucs)} 个未见数据集)")

    # ---- 保存结果 ----
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)

    output = {
        'experiment': 'cross_dataset_generalization',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'train_datasets': list(TRAIN_DATASETS.keys()),
            'train_samples_per_source': TRAIN_SAMPLES_PER_SOURCE,
            'test_samples': TEST_SAMPLES,
            'epochs': EPOCHS,
            'backbone_lr': BACKBONE_LR,
            'head_lr': HEAD_LR,
            'batch_size': BATCH_SIZE,
            'model': 'AIGCDetectorLite (pretrained ViT-B/16)',
            'total_params_M': total_params / 1e6,
            'trainable_params_M': trainable_params / 1e6,
            'train_time_min': train_time / 60,
        },
        'results': results,
        'avg_cross_dataset_auc': avg_cross_auc,
    }

    json_path = output_dir / 'cross_dataset_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # CSV
    csv_path = output_dir / 'cross_dataset_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['test_dataset', 'type', 'auc', 'ap', 'accuracy']
        writer.writerow(header)
        for name, m in results.items():
            dtype = 'in-domain' if '(in-domain)' in name else 'cross-dataset'
            writer.writerow([name, dtype, m['auc'], m['ap'], m['accuracy']])
        writer.writerow(['AVG_CROSS', 'cross-dataset', avg_cross_auc, '', ''])

    print(f"\n结果已保存到:")
    print(f"  {json_path}")
    print(f"  {csv_path}")


if __name__ == '__main__':
    main()
