#!/usr/bin/env python3
"""
跨数据集泛化实验 v2
训练: BigGAN + ADM + CIFAKE (覆盖ImageNet和CIFAR两种域)
测试: glide, VQDM, DiffusionForensics, NTIRE2026 (4个完全未见数据集)
+ JPEG75 鲁棒性测试
"""

import os, sys, json, random, time, csv, io
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

BASE = "/home/customer/文档/zc/0125image/datasets/authoritative"


class GenImageDataset(Dataset):
    def __init__(self, data_root, split='train', transform=None, max_samples=None,
                 apply_jpeg=False, jpeg_quality=75):
        self.transform = transform
        self.apply_jpeg = apply_jpeg
        self.jpeg_quality = jpeg_quality
        self.samples = []
        split_dir = Path(data_root) / split
        ai_dir = next((split_dir / d for d in ['ai','fake','1'] if (split_dir/d).exists()), None)
        nat_dir = next((split_dir / d for d in ['nature','real','0'] if (split_dir/d).exists()), None)
        if ai_dir is None or nat_dir is None:
            raise ValueError(f"目录不完整: {data_root}/{split}")
        exts = ['*.png','*.jpg','*.jpeg','*.JPEG','*.JPG','*.PNG','*.webp']
        ai_imgs, nat_imgs = [], []
        for ext in exts:
            ai_imgs.extend(list(ai_dir.glob(ext)))
            nat_imgs.extend(list(nat_dir.glob(ext)))
        random.shuffle(ai_imgs); random.shuffle(nat_imgs)
        if max_samples:
            n = max_samples // 2
            ai_imgs = ai_imgs[:n]; nat_imgs = nat_imgs[:n]
        self.samples = [(p,1) for p in ai_imgs] + [(p,0) for p in nat_imgs]
        random.shuffle(self.samples)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try: img = Image.open(path).convert('RGB')
        except: return self.__getitem__(random.randint(0, len(self.samples)-1))
        if self.apply_jpeg:
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=self.jpeg_quality)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
        if self.transform: img = self.transform(img)
        return {'image': img, 'label': label}


class CIFAKEDataset(Dataset):
    def __init__(self, data_root, split='test', transform=None, max_samples=None,
                 apply_jpeg=False, jpeg_quality=75):
        self.transform = transform
        self.apply_jpeg = apply_jpeg
        self.jpeg_quality = jpeg_quality
        self.samples = []
        split_dir = Path(data_root) / split
        real_dir = split_dir / 'real' if (split_dir/'real').exists() else split_dir / 'REAL'
        fake_dir = split_dir / 'fake' if (split_dir/'fake').exists() else split_dir / 'FAKE'
        exts = ['*.png','*.jpg','*.jpeg','*.JPEG']
        real_imgs, fake_imgs = [], []
        for ext in exts:
            real_imgs.extend(list(real_dir.glob(ext)))
            fake_imgs.extend(list(fake_dir.glob(ext)))
        random.shuffle(real_imgs); random.shuffle(fake_imgs)
        if max_samples:
            n = max_samples // 2
            real_imgs = real_imgs[:n]; fake_imgs = fake_imgs[:n]
        self.samples = [(p,0) for p in real_imgs] + [(p,1) for p in fake_imgs]
        random.shuffle(self.samples)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try: img = Image.open(path).convert('RGB')
        except: return self.__getitem__(random.randint(0, len(self.samples)-1))
        if self.apply_jpeg:
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=self.jpeg_quality)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
        if self.transform: img = self.transform(img)
        return {'image': img, 'label': label}


class NTIRE2026Dataset(Dataset):
    def __init__(self, data_root, shards=None, transform=None, max_samples=None,
                 apply_jpeg=False, jpeg_quality=75):
        self.transform = transform
        self.apply_jpeg = apply_jpeg
        self.jpeg_quality = jpeg_quality
        self.samples = []
        if shards is None:
            shards = ['shard_0', 'shard_1']
        for shard in shards:
            label_file = Path(data_root) / shard / 'labels.csv'
            img_dir = Path(data_root) / shard / 'images'
            if not label_file.exists(): continue
            df = pd.read_csv(label_file)
            for _, row in df.iterrows():
                p = img_dir / row['image_name']
                if p.exists():
                    self.samples.append((p, int(row['label'])))
        random.shuffle(self.samples)
        if max_samples and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try: img = Image.open(path).convert('RGB')
        except: return self.__getitem__(random.randint(0, len(self.samples)-1))
        if self.apply_jpeg:
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=self.jpeg_quality)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
        if self.transform: img = self.transform(img)
        return {'image': img, 'label': label}


def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch, total):
    model.train()
    total_loss, correct, n = 0, 0, 0
    pbar = tqdm(loader, desc=f"Train {epoch}/{total}", leave=False)
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
        correct += (logits.argmax(1) == labels).sum().item()
        n += labels.size(0)
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/n:.4f}'})
    return total_loss / len(loader), correct / n


def evaluate(model, loader, device, desc="Test"):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            imgs = batch['image'].to(device)
            labs = batch['label'].to(device)
            with autocast():
                out = model(imgs)
                logits = out['logits'] if isinstance(out, dict) else out
            probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            preds.extend(probs)
            labels.extend(labs.cpu().numpy())
    preds, labels = np.array(preds), np.array(labels)
    try:
        auc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
    except: auc, ap = 0.5, 0.5
    acc = accuracy_score(labels, (preds > 0.5).astype(int))
    return {'auc': auc, 'ap': ap, 'accuracy': acc}


def load_test_dataset(name, path, dtype, transform, max_samples, jpeg=False, jq=75):
    if dtype == 'genimage':
        for split in ['val', 'test', 'train']:
            try:
                return GenImageDataset(path, split=split, transform=transform,
                                       max_samples=max_samples, apply_jpeg=jpeg, jpeg_quality=jq)
            except: continue
    elif dtype == 'cifake':
        return CIFAKEDataset(path, split='test', transform=transform,
                              max_samples=max_samples, apply_jpeg=jpeg, jpeg_quality=jq)
    elif dtype == 'ntire':
        return NTIRE2026Dataset(path, shards=['shard_0','shard_1'],
                                 transform=transform, max_samples=max_samples,
                                 apply_jpeg=jpeg, jpeg_quality=jq)
    raise ValueError(f"无法加载 {name}")


def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device('cuda')
    print(f"Device: {device}")

    # ===== 配置 =====
    TRAIN_SAMPLES = 20000   # 每个源
    TEST_SAMPLES = 5000
    BATCH_SIZE = 32
    EPOCHS = 20
    BACKBONE_LR = 1e-5
    HEAD_LR = 5e-4

    # 训练源: BigGAN + ADM + CIFAKE (覆盖ImageNet和CIFAR两种视觉域)
    TRAIN_SOURCES = {
        'BigGAN': ('genimage', f"{BASE}/GenImage/BigGAN/imagenet_ai_0419_biggan"),
        'ADM': ('genimage', f"{BASE}/GenImage/ADM/imagenet_ai_0508_adm"),
        'CIFAKE': ('cifake', f"{BASE}/CIFAKE"),
    }

    # 测试: 4个完全未见过的数据集
    CROSS_TEST = {
        'glide': ('genimage', f"{BASE}/GenImage/glide/imagenet_glide"),
        'VQDM': ('genimage', f"{BASE}/GenImage/VQDM/imagenet_ai_0419_vqdm"),
        'DiffForensics': ('genimage', f"{BASE}/DiffusionForensics"),
        'NTIRE2026': ('ntire', f"{BASE}/NTIRE2026"),
    }

    train_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    test_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # ===== 加载训练数据 =====
    print("\n=== 加载训练数据 ===")
    print(f"训练源: {list(TRAIN_SOURCES.keys())}")
    train_list, val_list = [], []
    for name, (dtype, path) in TRAIN_SOURCES.items():
        if dtype == 'genimage':
            ds = GenImageDataset(path, 'train', train_tf, TRAIN_SAMPLES)
            try: vds = GenImageDataset(path, 'val', test_tf, 2000)
            except: vds = GenImageDataset(path, 'train', test_tf, 2000)
        elif dtype == 'cifake':
            ds = CIFAKEDataset(path, 'train', train_tf, TRAIN_SAMPLES)
            vds = CIFAKEDataset(path, 'test', test_tf, 2000)
        train_list.append(ds)
        val_list.append(vds)
        print(f"  {name}: train={len(ds)}, val={len(vds)}")

    train_loader = DataLoader(ConcatDataset(train_list), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(ConcatDataset(val_list), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)
    print(f"  总训练: {sum(len(d) for d in train_list)}, 总验证: {sum(len(d) for d in val_list)}")

    # ===== 加载测试数据 =====
    print("\n=== 加载跨数据集测试数据 (4个未见数据集) ===")
    test_loaders = {}

    # 跨数据集 (原始)
    for name, (dtype, path) in CROSS_TEST.items():
        try:
            ds = load_test_dataset(name, path, dtype, test_tf, TEST_SAMPLES)
            test_loaders[name] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                                            num_workers=4, pin_memory=True)
            print(f"  {name}: {len(ds)} samples")
        except Exception as e:
            print(f"  {name}: 失败 - {e}")

    # 跨数据集 (JPEG75)
    for name, (dtype, path) in CROSS_TEST.items():
        try:
            ds = load_test_dataset(name, path, dtype, test_tf, TEST_SAMPLES, jpeg=True, jq=75)
            test_loaders[f"{name}_JPEG75"] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                                                         num_workers=4, pin_memory=True)
            print(f"  {name}_JPEG75: {len(ds)} samples")
        except Exception as e:
            print(f"  {name}_JPEG75: 失败 - {e}")

    # 域内基准
    for name, (dtype, path) in TRAIN_SOURCES.items():
        try:
            if dtype == 'genimage':
                ds = load_test_dataset(name, path, dtype, test_tf, TEST_SAMPLES)
            elif dtype == 'cifake':
                ds = CIFAKEDataset(path, 'test', test_tf, TEST_SAMPLES)
            test_loaders[f"{name}(in-domain)"] = DataLoader(ds, batch_size=BATCH_SIZE,
                                                             shuffle=False, num_workers=4, pin_memory=True)
            print(f"  {name}(in-domain): {len(ds)} samples")
        except Exception as e:
            print(f"  {name}(in-domain): 失败 - {e}")

    # ===== 模型 =====
    print("\n=== 创建模型 ===")
    model = AIGCDetectorLite(num_classes=2, img_size=224, embed_dim=768,
                              num_prototypes=8, dropout=0.1, pretrained=True).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total: {total_p/1e6:.1f}M, Trainable: {train_p/1e6:.1f}M")

    bb, hd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if any(k in n for k in ['backbone','vit','patch_embed','blocks']): bb.append(p)
        else: hd.append(p)

    optimizer = optim.AdamW([{'params': bb, 'lr': BACKBONE_LR},
                              {'params': hd, 'lr': HEAD_LR}], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    # ===== 训练 =====
    print(f"\n=== 训练 ({EPOCHS} epochs) ===")
    best_val_auc, best_state = 0, None
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        loss, acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device, ep, EPOCHS)
        scheduler.step()
        vm = evaluate(model, val_loader, device, f"Val {ep}")
        print(f"  Epoch {ep}: loss={loss:.4f}, acc={acc:.4f}, val_auc={vm['auc']:.4f}, val_acc={vm['accuracy']:.4f}")
        if vm['auc'] > best_val_auc:
            best_val_auc = vm['auc']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_min = (time.time() - t0) / 60
    print(f"\n训练完成 ({train_min:.1f} min), 最佳 val AUC: {best_val_auc:.4f}")

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # ===== 测试 =====
    print(f"\n{'='*70}")
    print(f"跨数据集泛化测试结果")
    print(f"训练: {list(TRAIN_SOURCES.keys())} | 测试: {list(CROSS_TEST.keys())} + JPEG75")
    print(f"{'='*70}")

    results = {}
    cross_aucs = []

    for tname, loader in test_loaders.items():
        m = evaluate(model, loader, device, tname)
        results[tname] = m
        is_cross = '(in-domain)' not in tname
        tag = "CROSS" if is_cross else "IN-DOMAIN"
        print(f"  {tname:30s}: AUC={m['auc']:.4f}  AP={m['ap']:.4f}  Acc={m['accuracy']:.4f}  [{tag}]")
        if is_cross and 'JPEG' not in tname:
            cross_aucs.append(m['auc'])

    avg_cross = np.mean(cross_aucs) if cross_aucs else 0
    print(f"\n  跨数据集平均AUC (原始): {avg_cross:.4f} ({len(cross_aucs)}个未见数据集)")

    # ===== 保存 =====
    out_dir = Path(__file__).parent.parent / 'outputs'
    out_dir.mkdir(exist_ok=True)

    output = {
        'experiment': 'cross_dataset_v2',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'train_sources': list(TRAIN_SOURCES.keys()),
            'cross_test_datasets': list(CROSS_TEST.keys()),
            'train_samples_per_source': TRAIN_SAMPLES,
            'test_samples': TEST_SAMPLES,
            'epochs': EPOCHS,
            'model': 'AIGCDetectorLite (pretrained ViT-B/16 + FDAA + MGFP)',
            'total_params_M': total_p / 1e6,
            'trainable_params_M': train_p / 1e6,
            'train_time_min': train_min,
        },
        'results': results,
        'avg_cross_dataset_auc': avg_cross,
    }

    with open(out_dir / 'cross_dataset_v2_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    with open(out_dir / 'cross_dataset_v2_results.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'type', 'auc', 'ap', 'accuracy'])
        for name, m in results.items():
            dtype = 'in-domain' if '(in-domain)' in name else ('cross-jpeg' if 'JPEG' in name else 'cross-dataset')
            w.writerow([name, dtype, f"{m['auc']:.6f}", f"{m['ap']:.6f}", f"{m['accuracy']:.4f}"])
        w.writerow(['AVG_CROSS', 'cross-dataset', f"{avg_cross:.6f}", '', ''])

    print(f"\n结果已保存: outputs/cross_dataset_v2_results.json / .csv")


if __name__ == '__main__':
    main()
