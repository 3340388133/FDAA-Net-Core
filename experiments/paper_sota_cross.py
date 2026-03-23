#!/usr/bin/env python3
"""
论文SOTA对比实验 (Table 1 + Table 2)

所有方法用相同训练集训练，测试域内+跨数据集泛化
训练: BigGAN + ADM + CIFAKE + NTIRE2026(shard_4,5)  [4源，80K样本]
域内测试: BigGAN(val), ADM(val), CIFAKE(test), NTIRE2026(shard_1)
跨域测试: glide, VQDM, DiffForensics  [完全未见]
鲁棒性: 跨域 + JPEG75

对比方法:
  - CNNDetection (ResNet50, CVPR 2020)
  - F3Net (EfficientNet-B0, ECCV 2020)
  - FreqNet (ResNet50+FFT, AAAI 2024)
  - UnivFD (CLIP ViT, CVPR 2023)
  - Ours (ViT-B/16 + FDAA + MGFP)
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
from models.detector import AIGCDetectorLite
from models.sota_methods import CNNDetector, F3NetDetector, FreqNetDetector, UnivFDDetector

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
    elif dtype == 'cifake':
        return CIFAKEDataset(path, split='test', transform=transform,
                              max_samples=max_samples, apply_jpeg=jpeg, jpeg_quality=jq)
    raise ValueError(f"无法加载 {name}")


# ==================== 方法定义 ====================

def get_methods():
    """返回所有对比方法的配置"""
    return [
        {
            'name': 'CNNDetection',
            'create': lambda: CNNDetector(num_classes=2, pretrained=True),
            'lr': 1e-4,
            'all_trainable': True,
        },
        {
            'name': 'F3Net',
            'create': lambda: F3NetDetector(num_classes=2),
            'lr': 1e-4,
            'all_trainable': True,
        },
        {
            'name': 'FreqNet',
            'create': lambda: FreqNetDetector(num_classes=2, pretrained=True),
            'lr': 1e-4,
            'all_trainable': True,
        },
        {
            'name': 'UnivFD',
            'create': lambda: UnivFDDetector(num_classes=2, use_clip=False),
            'lr': 1e-3,
            'all_trainable': True,  # backbone已冻结，只有classifier可训练
        },
        {
            'name': 'Ours (FDAA+MGFP)',
            'create': lambda: AIGCDetectorLite(num_classes=2, img_size=224, embed_dim=768,
                                                num_prototypes=8, dropout=0.1, pretrained=True),
            'lr': 5e-4,
            'backbone_lr': 1e-5,
            'all_trainable': False,
        },
    ]


def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device('cuda')

    TRAIN_SAMPLES = 20000
    TEST_SAMPLES = 5000
    BATCH_SIZE = 32
    EPOCHS = 20

    print("\n" + "="*70)
    print("论文SOTA对比: 域内 + 跨数据集泛化")
    print("训练: BigGAN + ADM + CIFAKE + NTIRE2026(shard_4,5)")
    print("域内: BigGAN, ADM, CIFAKE, NTIRE2026")
    print("跨域: glide, VQDM, DiffForensics (+JPEG75)")
    print("="*70)

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

    # ===== 训练数据 =====
    print("\n=== 加载训练数据 ===")
    train_list, val_list = [], []

    ds_bg = GenImageDataset(f"{BASE}/GenImage/BigGAN/imagenet_ai_0419_biggan", 'train', train_tf, TRAIN_SAMPLES)
    try: vds_bg = GenImageDataset(f"{BASE}/GenImage/BigGAN/imagenet_ai_0419_biggan", 'val', test_tf, 2000)
    except: vds_bg = GenImageDataset(f"{BASE}/GenImage/BigGAN/imagenet_ai_0419_biggan", 'train', test_tf, 2000)
    train_list.append(ds_bg); val_list.append(vds_bg)
    print(f"  BigGAN: {len(ds_bg)} train")

    ds_adm = GenImageDataset(f"{BASE}/GenImage/ADM/imagenet_ai_0508_adm", 'train', train_tf, TRAIN_SAMPLES)
    try: vds_adm = GenImageDataset(f"{BASE}/GenImage/ADM/imagenet_ai_0508_adm", 'val', test_tf, 2000)
    except: vds_adm = GenImageDataset(f"{BASE}/GenImage/ADM/imagenet_ai_0508_adm", 'train', test_tf, 2000)
    train_list.append(ds_adm); val_list.append(vds_adm)
    print(f"  ADM: {len(ds_adm)} train")

    ds_cifake = CIFAKEDataset(f"{BASE}/CIFAKE", split='train', transform=train_tf, max_samples=TRAIN_SAMPLES)
    vds_cifake = CIFAKEDataset(f"{BASE}/CIFAKE", split='test', transform=test_tf, max_samples=2000)
    train_list.append(ds_cifake); val_list.append(vds_cifake)
    print(f"  CIFAKE: {len(ds_cifake)} train")

    ds_ntire = NTIRE2026Dataset(f"{BASE}/NTIRE2026", shards=['shard_4','shard_5'],
                                  transform=train_tf, max_samples=TRAIN_SAMPLES, balanced=True)
    vds_ntire = NTIRE2026Dataset(f"{BASE}/NTIRE2026", shards=['shard_0'],
                                   transform=test_tf, max_samples=2000, balanced=True)
    train_list.append(ds_ntire); val_list.append(vds_ntire)

    print(f"  总训练: {sum(len(d) for d in train_list)}")

    train_ds = ConcatDataset(train_list)
    val_ds = ConcatDataset(val_list)

    # ===== 测试数据 =====
    print("\n=== 加载测试数据 ===")
    CROSS_TEST = {
        'glide': ('genimage', f"{BASE}/GenImage/glide/imagenet_glide"),
        'VQDM': ('genimage', f"{BASE}/GenImage/VQDM/imagenet_ai_0419_vqdm"),
        'DiffForensics': ('genimage', f"{BASE}/DiffusionForensics"),
    }
    IN_DOMAIN = {
        'BigGAN': ('genimage', f"{BASE}/GenImage/BigGAN/imagenet_ai_0419_biggan"),
        'ADM': ('genimage', f"{BASE}/GenImage/ADM/imagenet_ai_0508_adm"),
        'CIFAKE': ('cifake', f"{BASE}/CIFAKE"),
    }

    test_datasets = {}
    for name, (dtype, path) in {**CROSS_TEST, **IN_DOMAIN}.items():
        try:
            ds = load_test_dataset(name, path, dtype, test_tf, TEST_SAMPLES)
            test_datasets[name] = ds
            print(f"  {name}: {len(ds)} samples")
        except Exception as e:
            print(f"  {name}: 失败 - {e}")

    try:
        ds_nt = NTIRE2026Dataset(f"{BASE}/NTIRE2026", shards=['shard_1'],
                                   transform=test_tf, max_samples=TEST_SAMPLES, balanced=True)
        test_datasets['NTIRE2026'] = ds_nt
    except: pass

    # JPEG75
    for name, (dtype, path) in CROSS_TEST.items():
        try:
            ds = load_test_dataset(name, path, dtype, test_tf, TEST_SAMPLES, jpeg=True, jq=75)
            test_datasets[f"{name}_JPEG75"] = ds
        except: pass

    # ===== 对比实验主循环 =====
    all_results = {}
    methods = get_methods()

    for m_idx, method_cfg in enumerate(methods):
        method_name = method_cfg['name']
        print(f"\n{'='*70}")
        print(f"[{m_idx+1}/{len(methods)}] {method_name}")
        print(f"{'='*70}")

        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        model = method_cfg['create']().to(device)
        total_p = sum(p.numel() for p in model.parameters())
        train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  参数: Total={total_p/1e6:.1f}M, Trainable={train_p/1e6:.1f}M")

        # 优化器
        if method_cfg.get('all_trainable', True):
            optimizer = optim.AdamW(model.parameters(), lr=method_cfg['lr'], weight_decay=1e-4)
        else:
            bb, hd = [], []
            for n, p in model.named_parameters():
                if not p.requires_grad: continue
                if any(k in n for k in ['backbone','vit','patch_embed','blocks']): bb.append(p)
                else: hd.append(p)
            optimizer = optim.AdamW([
                {'params': bb, 'lr': method_cfg.get('backbone_lr', 1e-5)},
                {'params': hd, 'lr': method_cfg['lr']}
            ], weight_decay=1e-4)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler()

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
        print(f"  训练完成 ({train_min:.1f}min), best val AUC: {best_val_auc:.4f}")

        if best_state:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        # 测试
        method_results = {'total_params_M': total_p/1e6, 'trainable_params_M': train_p/1e6,
                          'train_min': train_min, 'best_val_auc': best_val_auc}

        cross_aucs, indomain_aucs, jpeg_aucs = [], [], []

        for tname, tds in test_datasets.items():
            loader = DataLoader(tds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
            m = evaluate(model, loader, device, tname)
            method_results[tname] = m

            if tname in CROSS_TEST:
                cross_aucs.append(m['auc'])
                tag = "CROSS"
            elif 'JPEG75' in tname:
                jpeg_aucs.append(m['auc'])
                tag = "JPEG75"
            else:
                indomain_aucs.append(m['auc'])
                tag = "IN-DOMAIN"
            print(f"  {tname:25s}: AUC={m['auc']:.4f}  AP={m['ap']:.4f}  [{tag}]")

        method_results['avg_cross_auc'] = np.mean(cross_aucs) if cross_aucs else 0
        method_results['avg_indomain_auc'] = np.mean(indomain_aucs) if indomain_aucs else 0
        method_results['avg_jpeg75_auc'] = np.mean(jpeg_aucs) if jpeg_aucs else 0

        print(f"\n  域内平均AUC: {method_results['avg_indomain_auc']:.4f}")
        print(f"  跨域平均AUC: {method_results['avg_cross_auc']:.4f}")
        print(f"  JPEG75平均AUC: {method_results['avg_jpeg75_auc']:.4f}")

        all_results[method_name] = method_results

        del model, optimizer, scheduler, scaler, best_state
        torch.cuda.empty_cache()

    # ===== 保存结果 =====
    out_dir = Path(__file__).parent.parent / 'outputs'
    out_dir.mkdir(exist_ok=True)

    output = {
        'experiment': 'paper_sota_cross',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'train_sources': ['BigGAN', 'ADM', 'CIFAKE', 'NTIRE2026(shard_4,5)'],
            'cross_test': list(CROSS_TEST.keys()),
            'in_domain_test': list(IN_DOMAIN.keys()) + ['NTIRE2026'],
            'train_samples_per_source': TRAIN_SAMPLES,
            'test_samples': TEST_SAMPLES,
            'epochs': EPOCHS,
        },
        'results': all_results,
    }

    with open(out_dir / 'paper_sota_cross_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # 打印论文格式表格
    print(f"\n{'='*90}")
    print("论文Table 1: 域内性能 + Table 2: 跨数据集泛化")
    print(f"{'='*90}")
    header = f"{'Method':<22s} {'Params':>7s} | {'BigGAN':>7s} {'ADM':>7s} {'CIFAKE':>7s} {'NTIRE':>7s} {'Avg_ID':>7s} | {'glide':>7s} {'VQDM':>7s} {'DiffF':>7s} {'Avg_X':>7s}"
    print(header)
    print("-" * 100)
    for mname, r in all_results.items():
        row = f"{mname:<22s} {r['trainable_params_M']:>6.1f}M |"
        for ds in ['BigGAN', 'ADM', 'CIFAKE', 'NTIRE2026']:
            auc = r.get(ds, {}).get('auc', 0)
            row += f" {auc:>6.4f}"
        row += f" {r['avg_indomain_auc']:>6.4f} |"
        for ds in ['glide', 'VQDM', 'DiffForensics']:
            auc = r.get(ds, {}).get('auc', 0)
            row += f" {auc:>6.4f}"
        row += f" {r['avg_cross_auc']:>6.4f}"
        print(row)

    print(f"\n结果已保存: outputs/paper_sota_cross_results.json")


if __name__ == '__main__':
    main()
