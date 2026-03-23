#!/usr/bin/env python3
"""
论文消融实验 (Table 3)
使用预训练ViT-B/16 backbone，消融FDAA和MGFP的贡献

配置:
  1. Baseline: ViT-B/16 + Linear head (无FDAA/MGFP)
  2. + FDAA: ViT-B/16 + FDAA + Linear head
  3. + MGFP: ViT-B/16 + MGFP + Linear head
  4. Full: ViT-B/16 + FDAA + MGFP (完整模型)

训练数据: 与最终确定的配置一致 (BigGAN+ADM+NTIRE2026 shard_4,5)
测试: 域内 + 跨数据集
"""

import os, sys, json, random, time, csv, io, copy
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
from PIL import Image, ImageFilter
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.detector_ablation import AIGCDetectorAblation

BASE = "/home/customer/文档/zc/0125image/datasets/authoritative"

# ==================== 数据增强 ====================

class RandomJPEGCompression:
    def __init__(self, quality_range=(50, 95), prob=0.3):
        self.quality_range = quality_range
        self.prob = prob
    def __call__(self, img):
        if random.random() < self.prob:
            q = random.randint(*self.quality_range)
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=q)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
        return img

class RandomGaussianBlur:
    def __init__(self, radius_range=(0.5, 1.5), prob=0.2):
        self.radius_range = radius_range
        self.prob = prob
    def __call__(self, img):
        if random.random() < self.prob:
            r = random.uniform(*self.radius_range)
            img = img.filter(ImageFilter.GaussianBlur(radius=r))
        return img

# ==================== 数据集 ====================

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
            buf = io.BytesIO(); img.save(buf, format='JPEG', quality=self.jpeg_quality)
            buf.seek(0); img = Image.open(buf).convert('RGB')
        if self.transform: img = self.transform(img)
        return {'image': img, 'label': label}


class NTIRE2026Dataset(Dataset):
    def __init__(self, data_root, shards=None, transform=None, max_samples=None,
                 apply_jpeg=False, jpeg_quality=75, balanced=False):
        self.transform = transform
        self.apply_jpeg = apply_jpeg
        self.jpeg_quality = jpeg_quality
        self.samples = []
        if shards is None: shards = ['shard_0', 'shard_1']
        real_samples, fake_samples = [], []
        for shard in shards:
            label_file = Path(data_root) / shard / 'labels.csv'
            img_dir = Path(data_root) / shard / 'images'
            if not label_file.exists(): continue
            df = pd.read_csv(label_file)
            for _, row in df.iterrows():
                p = img_dir / row['image_name']
                if p.exists():
                    lbl = int(row['label'])
                    if lbl == 0: real_samples.append((p, 0))
                    else: fake_samples.append((p, 1))
        random.shuffle(real_samples); random.shuffle(fake_samples)
        if balanced and max_samples:
            n = max_samples // 2
            real_samples = real_samples[:n]; fake_samples = fake_samples[:n]
            self.samples = real_samples + fake_samples
        elif max_samples:
            all_s = real_samples + fake_samples; random.shuffle(all_s)
            self.samples = all_s[:max_samples]
        else:
            self.samples = real_samples + fake_samples
        random.shuffle(self.samples)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try: img = Image.open(path).convert('RGB')
        except: return self.__getitem__(random.randint(0, len(self.samples)-1))
        if self.apply_jpeg:
            buf = io.BytesIO(); img.save(buf, format='JPEG', quality=self.jpeg_quality)
            buf.seek(0); img = Image.open(buf).convert('RGB')
        if self.transform: img = self.transform(img)
        return {'image': img, 'label': label}


# ==================== 训练/评估 ====================

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
    elif dtype == 'ntire':
        return NTIRE2026Dataset(path, shards=['shard_0','shard_1'],
                                 transform=transform, max_samples=max_samples,
                                 apply_jpeg=jpeg, jpeg_quality=jq)
    raise ValueError(f"无法加载 {name}")


# ==================== 消融配置 ====================

ABLATION_CONFIGS = [
    {
        'name': 'Baseline (ViT-B/16 + Linear)',
        'use_fdaa_spatial': False, 'use_fdaa_freq': False,
        'use_fdaa_cross': False, 'use_mgfp': False,
    },
    {
        'name': '+ FDAA',
        'use_fdaa_spatial': True, 'use_fdaa_freq': True,
        'use_fdaa_cross': True, 'use_mgfp': False,
    },
    {
        'name': '+ MGFP',
        'use_fdaa_spatial': False, 'use_fdaa_freq': False,
        'use_fdaa_cross': False, 'use_mgfp': True,
    },
    {
        'name': 'Full (FDAA + MGFP)',
        'use_fdaa_spatial': True, 'use_fdaa_freq': True,
        'use_fdaa_cross': True, 'use_mgfp': True,
    },
]


def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device('cuda')

    # ===== 配置 =====
    TRAIN_SAMPLES = 20000
    TEST_SAMPLES = 5000
    BATCH_SIZE = 32
    EPOCHS = 20
    BACKBONE_LR = 1e-5
    HEAD_LR = 5e-4

    print("\n" + "="*70)
    print("论文消融实验 (预训练ViT-B/16 backbone)")
    print("训练: BigGAN + ADM + NTIRE2026(shard_4,5)")
    print("测试: 域内(BigGAN,ADM,NTIRE) + 跨域(glide,VQDM,DiffForensics)")
    print("="*70)

    # ===== 数据增强 =====
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        RandomJPEGCompression(quality_range=(50, 95), prob=0.3),
        RandomGaussianBlur(radius_range=(0.5, 1.5), prob=0.2),
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
    train_list, val_list = [], []

    print("  BigGAN...")
    ds_bg = GenImageDataset(f"{BASE}/GenImage/BigGAN/imagenet_ai_0419_biggan", 'train', train_tf, TRAIN_SAMPLES)
    try: vds_bg = GenImageDataset(f"{BASE}/GenImage/BigGAN/imagenet_ai_0419_biggan", 'val', test_tf, 2000)
    except: vds_bg = GenImageDataset(f"{BASE}/GenImage/BigGAN/imagenet_ai_0419_biggan", 'train', test_tf, 2000)
    train_list.append(ds_bg); val_list.append(vds_bg)
    print(f"    BigGAN: train={len(ds_bg)}, val={len(vds_bg)}")

    print("  ADM...")
    ds_adm = GenImageDataset(f"{BASE}/GenImage/ADM/imagenet_ai_0508_adm", 'train', train_tf, TRAIN_SAMPLES)
    try: vds_adm = GenImageDataset(f"{BASE}/GenImage/ADM/imagenet_ai_0508_adm", 'val', test_tf, 2000)
    except: vds_adm = GenImageDataset(f"{BASE}/GenImage/ADM/imagenet_ai_0508_adm", 'train', test_tf, 2000)
    train_list.append(ds_adm); val_list.append(vds_adm)
    print(f"    ADM: train={len(ds_adm)}, val={len(vds_adm)}")

    print("  NTIRE2026 (shard_4,5)...")
    ds_ntire = NTIRE2026Dataset(f"{BASE}/NTIRE2026", shards=['shard_4','shard_5'],
                                  transform=train_tf, max_samples=TRAIN_SAMPLES, balanced=True)
    vds_ntire = NTIRE2026Dataset(f"{BASE}/NTIRE2026", shards=['shard_0'],
                                   transform=test_tf, max_samples=2000, balanced=True)
    train_list.append(ds_ntire); val_list.append(vds_ntire)

    total_train = sum(len(d) for d in train_list)
    print(f"\n  总训练: {total_train}, 总验证: {sum(len(d) for d in val_list)}")

    train_ds = ConcatDataset(train_list)
    val_ds = ConcatDataset(val_list)

    # ===== 加载测试数据 =====
    print("\n=== 加载测试数据 ===")
    CROSS_TEST = {
        'glide': ('genimage', f"{BASE}/GenImage/glide/imagenet_glide"),
        'VQDM': ('genimage', f"{BASE}/GenImage/VQDM/imagenet_ai_0419_vqdm"),
        'DiffForensics': ('genimage', f"{BASE}/DiffusionForensics"),
    }
    IN_DOMAIN = {
        'BigGAN': ('genimage', f"{BASE}/GenImage/BigGAN/imagenet_ai_0419_biggan"),
        'ADM': ('genimage', f"{BASE}/GenImage/ADM/imagenet_ai_0508_adm"),
    }

    test_datasets = {}
    for name, (dtype, path) in {**CROSS_TEST, **IN_DOMAIN}.items():
        try:
            ds = load_test_dataset(name, path, dtype, test_tf, TEST_SAMPLES)
            test_datasets[name] = ds
            tag = "CROSS" if name in CROSS_TEST else "IN-DOMAIN"
            print(f"  {name}: {len(ds)} samples [{tag}]")
        except Exception as e:
            print(f"  {name}: 失败 - {e}")

    # NTIRE域内
    try:
        ds_ntire_test = NTIRE2026Dataset(f"{BASE}/NTIRE2026", shards=['shard_1'],
                                           transform=test_tf, max_samples=TEST_SAMPLES, balanced=True)
        test_datasets['NTIRE2026'] = ds_ntire_test
        print(f"  NTIRE2026: {len(ds_ntire_test)} samples [IN-DOMAIN]")
    except Exception as e:
        print(f"  NTIRE2026: 失败 - {e}")

    # JPEG75版本
    for name, (dtype, path) in CROSS_TEST.items():
        try:
            ds = load_test_dataset(name, path, dtype, test_tf, TEST_SAMPLES, jpeg=True, jq=75)
            test_datasets[f"{name}_JPEG75"] = ds
        except: pass

    # ===== 消融实验主循环 =====
    all_results = {}

    for cfg_idx, config in enumerate(ABLATION_CONFIGS):
        cfg_name = config['name']
        print(f"\n{'='*70}")
        print(f"[{cfg_idx+1}/{len(ABLATION_CONFIGS)}] {cfg_name}")
        print(f"{'='*70}")

        # 重置随机种子保证公平
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        # 创建模型 (预训练backbone)
        model = AIGCDetectorAblation(
            num_classes=2, img_size=224, embed_dim=768,
            num_prototypes=8, dropout=0.1, pretrained=True,
            use_fdaa_spatial=config['use_fdaa_spatial'],
            use_fdaa_freq=config['use_fdaa_freq'],
            use_fdaa_cross=config['use_fdaa_cross'],
            use_mgfp=config['use_mgfp'],
        ).to(device)

        total_p = sum(p.numel() for p in model.parameters())
        train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  参数: Total={total_p/1e6:.1f}M, Trainable={train_p/1e6:.1f}M")

        # 分离backbone和head学习率
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

        # DataLoader (每次重新创建保证shuffle一致)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=4, pin_memory=True)

        # 训练
        best_val_auc, best_state = 0, None
        t0 = time.time()

        for ep in range(1, EPOCHS + 1):
            loss, acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device, ep, EPOCHS)
            scheduler.step()
            vm = evaluate(model, val_loader, device, f"Val {ep}")
            print(f"  Epoch {ep}: loss={loss:.4f}, acc={acc:.4f}, val_auc={vm['auc']:.4f}")
            if vm['auc'] > best_val_auc:
                best_val_auc = vm['auc']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        train_min = (time.time() - t0) / 60
        print(f"  训练完成 ({train_min:.1f} min), best val AUC: {best_val_auc:.4f}")

        if best_state:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        # 测试
        cfg_results = {'config': cfg_name, 'total_params_M': total_p/1e6,
                       'trainable_params_M': train_p/1e6, 'train_min': train_min,
                       'best_val_auc': best_val_auc}

        cross_aucs = []
        for tname, tds in test_datasets.items():
            loader = DataLoader(tds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
            m = evaluate(model, loader, device, tname)
            cfg_results[tname] = m
            is_cross = tname in CROSS_TEST or '_JPEG75' in tname
            tag = "CROSS" if is_cross else "IN-DOMAIN"
            print(f"  {tname:25s}: AUC={m['auc']:.4f}  AP={m['ap']:.4f}  [{tag}]")
            if tname in CROSS_TEST:
                cross_aucs.append(m['auc'])

        cfg_results['avg_cross_auc'] = np.mean(cross_aucs) if cross_aucs else 0
        in_domain_aucs = [cfg_results[k]['auc'] for k in ['BigGAN','ADM','NTIRE2026'] if k in cfg_results]
        cfg_results['avg_indomain_auc'] = np.mean(in_domain_aucs) if in_domain_aucs else 0

        print(f"\n  平均跨域AUC: {cfg_results['avg_cross_auc']:.4f}")
        print(f"  平均域内AUC: {cfg_results['avg_indomain_auc']:.4f}")

        all_results[cfg_name] = cfg_results

        # 释放显存
        del model, optimizer, scheduler, scaler, best_state
        torch.cuda.empty_cache()

    # ===== 保存结果 =====
    out_dir = Path(__file__).parent.parent / 'outputs'
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / 'paper_ablation_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # CSV格式
    with open(out_dir / 'paper_ablation_results.csv', 'w', newline='') as f:
        w = csv.writer(f)
        header = ['Config', 'Params(M)', 'Trainable(M)',
                  'BigGAN', 'ADM', 'NTIRE2026', 'Avg_InDomain',
                  'glide', 'VQDM', 'DiffForensics', 'Avg_Cross']
        w.writerow(header)
        for cfg_name, r in all_results.items():
            row = [cfg_name, f"{r['total_params_M']:.1f}", f"{r['trainable_params_M']:.1f}"]
            for ds in ['BigGAN', 'ADM', 'NTIRE2026']:
                row.append(f"{r.get(ds, {}).get('auc', 0):.4f}")
            row.append(f"{r['avg_indomain_auc']:.4f}")
            for ds in ['glide', 'VQDM', 'DiffForensics']:
                row.append(f"{r.get(ds, {}).get('auc', 0):.4f}")
            row.append(f"{r['avg_cross_auc']:.4f}")
            w.writerow(row)

    # 打印总结表
    print(f"\n{'='*70}")
    print("消融实验总结")
    print(f"{'='*70}")
    print(f"{'Config':<35s} {'Params':>8s} {'InDomain':>10s} {'Cross':>10s}")
    print("-" * 65)
    for cfg_name, r in all_results.items():
        print(f"{cfg_name:<35s} {r['trainable_params_M']:>7.1f}M "
              f"{r['avg_indomain_auc']:>9.4f} {r['avg_cross_auc']:>9.4f}")

    print(f"\n结果已保存: outputs/paper_ablation_results.json / .csv")


if __name__ == '__main__':
    main()
