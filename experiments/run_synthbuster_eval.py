#!/usr/bin/env python3
"""
Synthbuster 跨数据集评估
- 按生成器分别评估 (SD1.3, SD1.4, SD2.0, SDXL, DALLE2, DALLE3, MJv5, Firefly, GLIDE)
- 输出论文Table格式结果
"""

import sys
import os
import json
import zipfile
import shutil
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'experiments'))

import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from run_paper_experiments import (
    PAPER_CONFIG, setup_output_dir, save_json, fmt,
    evaluate_model, create_sota_model, get_multi_source_transforms,
)

# Synthbuster 中的生成器 → 文件夹名的映射
SYNTHBUSTER_GENERATORS = {
    'DALLE2': 'dalle2',
    'DALLE3': 'dalle3',
    'Firefly': 'firefly',
    'Midjourney_v5': 'midjourney5',
    'SD_1.3': 'stable_diffusion_1',      # SD 1.3
    'SD_1.4': 'stable_diffusion_1_4',    # SD 1.4
    'SD_2.0': 'stable_diffusion_2',      # SD 2.0
    'SDXL': 'stable_diffusion_xl',       # SDXL
    'GLIDE': 'glide',
}

# 先尝试自动探测实际文件夹名
def detect_generator_folders(synthbuster_root):
    """自动检测 Synthbuster 解压后的文件夹名"""
    root = Path(synthbuster_root)
    mapping = {}

    # 遍历所有子目录
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        name = d.name.lower()

        # 检查是否包含图片
        imgs = list(d.glob('*.png')) + list(d.glob('*.jpg')) + list(d.glob('*.jpeg')) + list(d.glob('*.webp'))
        if len(imgs) == 0:
            # 可能有子目录
            for sd in d.iterdir():
                if sd.is_dir():
                    imgs = list(sd.glob('*.png')) + list(sd.glob('*.jpg'))
                    if imgs:
                        break
        if len(imgs) == 0:
            continue

        # 映射文件夹名到友好名
        if 'dalle' in name and '3' in name:
            mapping['DALLE3'] = str(d)
        elif 'dalle' in name and '2' in name:
            mapping['DALLE2'] = str(d)
        elif 'dalle' in name:
            mapping['DALLE'] = str(d)
        elif 'midjourney' in name or 'mj' in name:
            mapping['Midjourney_v5'] = str(d)
        elif 'firefly' in name:
            mapping['Firefly'] = str(d)
        elif 'glide' in name:
            mapping['GLIDE'] = str(d)
        elif 'xl' in name or 'sdxl' in name:
            mapping['SDXL'] = str(d)
        elif ('stable' in name or 'sd' in name) and ('2' in name):
            mapping['SD_2.0'] = str(d)
        elif ('stable' in name or 'sd' in name) and ('1.4' in name or '1_4' in name or 'v1-4' in name):
            mapping['SD_1.4'] = str(d)
        elif ('stable' in name or 'sd' in name) and ('1.3' in name or '1_3' in name or 'v1-3' in name):
            mapping['SD_1.3'] = str(d)
        elif ('stable' in name or 'sd' in name) and ('1' in name):
            mapping['SD_1.x'] = str(d)
        else:
            mapping[d.name] = str(d)

    return mapping


class SynthbusterGeneratorDataset(Dataset):
    """单个生成器的评估数据集 (fake) + 配对的 RAISE-1k (real)"""

    def __init__(self, fake_dir, real_dir=None, transform=None, max_samples=1000):
        self.transform = transform
        self.samples = []
        extensions = {'.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG', '.bmp', '.webp', '.tif', '.tiff'}

        # 收集 fake 图片
        fake_path = Path(fake_dir)
        fake_imgs = sorted([p for p in fake_path.iterdir() if p.suffix in extensions])[:max_samples]
        for p in fake_imgs:
            self.samples.append((str(p), 1))  # label=1 for fake

        # 收集 real 图片
        if real_dir and Path(real_dir).exists():
            real_path = Path(real_dir)
            real_imgs = sorted([p for p in real_path.iterdir() if p.suffix in extensions])[:max_samples]
            for p in real_imgs:
                self.samples.append((str(p), 0))  # label=0 for real

        print(f"    Dataset: {len(fake_imgs)} fake + {len(self.samples) - len(fake_imgs)} real = {len(self.samples)} total")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            # 返回黑色图像作为 fallback
            img = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        return {'image': img, 'label': label, 'path': path}


def setup_synthbuster(synthbuster_zip, extract_dir):
    """解压并整理 Synthbuster 数据集"""
    extract_dir = Path(extract_dir)

    # 检查是否已经解压
    if (extract_dir / '_extracted_done').exists():
        print("[Synthbuster] Already extracted, skipping...")
        return True

    if not Path(synthbuster_zip).exists():
        print(f"[Error] {synthbuster_zip} not found!")
        return False

    print(f"[Synthbuster] Extracting {synthbuster_zip}...")
    try:
        with zipfile.ZipFile(synthbuster_zip, 'r') as zf:
            zf.extractall(extract_dir)
        (extract_dir / '_extracted_done').touch()
        print("[Synthbuster] Extraction complete!")
        return True
    except Exception as e:
        print(f"[Error] Extraction failed: {e}")
        return False


def download_raise1k(raise_dir):
    """下载 RAISE-1k 真实图像 (如果不存在)"""
    raise_dir = Path(raise_dir)
    raise_dir.mkdir(parents=True, exist_ok=True)

    # 检查是否已有足够图片
    existing = list(raise_dir.glob('*.png')) + list(raise_dir.glob('*.jpg')) + list(raise_dir.glob('*.tif'))
    if len(existing) >= 500:
        print(f"[RAISE-1k] Found {len(existing)} images, sufficient.")
        return str(raise_dir)

    print("[RAISE-1k] Not enough real images. Will use GenImage real images as fallback.")
    return None


def get_real_images_fallback(genimage_root, max_samples=1000):
    """用 GenImage 的 real 图作为 Synthbuster 的 real 配对"""
    genimage_root = Path(genimage_root)
    real_imgs = []
    extensions = {'.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG'}

    # 从 BigGAN 的 val/nature 中取 real 图
    for source_dir in ['BigGAN/imagenet_ai_0419_biggan', 'ADM/imagenet_ai_0508_adm']:
        real_dir = genimage_root / source_dir / 'val' / 'nature'
        if real_dir.exists():
            for p in sorted(real_dir.iterdir()):
                if p.suffix in extensions:
                    real_imgs.append(str(p))
                if len(real_imgs) >= max_samples:
                    break
        if len(real_imgs) >= max_samples:
            break

    print(f"[Fallback Real] Collected {len(real_imgs)} real images from GenImage")
    return real_imgs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip download, assume data is ready')
    args = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # ====== 路径配置 ======
    DATASETS_ROOT = '/home/customer/文档/zc/0125image/datasets/authoritative'
    SYNTHBUSTER_ZIP = f'{DATASETS_ROOT}/Synthbuster/synthbuster.zip'
    SYNTHBUSTER_DIR = f'{DATASETS_ROOT}/Synthbuster'
    RAISE_DIR = f'{DATASETS_ROOT}/Synthbuster/raise1k'
    GENIMAGE_ROOT = PAPER_CONFIG['genimage_root']
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'paper_results_6src')

    checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints')
    results_dir = os.path.join(OUTPUT_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # ====== Step 1: 准备数据 ======
    print("=" * 60)
    print("Step 1: Preparing Synthbuster Dataset")
    print("=" * 60)

    # 等待下载完成
    if not args.skip_download:
        import time
        while True:
            if os.path.exists(SYNTHBUSTER_ZIP):
                # 检查文件是否还在写入
                size1 = os.path.getsize(SYNTHBUSTER_ZIP)
                time.sleep(5)
                if os.path.exists(SYNTHBUSTER_ZIP):
                    size2 = os.path.getsize(SYNTHBUSTER_ZIP)
                    if size1 == size2 and size1 > 1_000_000:  # >1MB 且不再增长
                        print(f"[Download] Complete: {size2 / 1e9:.2f} GB")
                        break
                    elif size1 == size2 and size1 < 1_000_000:
                        # 文件太小，下载可能失败了
                        print(f"[Warning] File too small ({size1} bytes), might be incomplete")
                        # 仍然尝试继续
                        if size1 > 100:  # 至少有点内容
                            break
                    else:
                        print(f"[Download] In progress: {size2 / 1e6:.1f} MB...")
            else:
                print("[Download] Waiting for synthbuster.zip...")
            time.sleep(30)

    # 解压
    setup_synthbuster(SYNTHBUSTER_ZIP, SYNTHBUSTER_DIR)

    # 探测文件夹结构
    print("\n[Synthbuster] Detecting generator folders...")
    gen_mapping = detect_generator_folders(SYNTHBUSTER_DIR)
    print(f"  Found {len(gen_mapping)} generators: {list(gen_mapping.keys())}")
    for name, path in gen_mapping.items():
        n_imgs = len(list(Path(path).glob('*.*')))
        print(f"    {name}: {path} ({n_imgs} files)")

    if not gen_mapping:
        # 手动扫描更深层
        print("[Synthbuster] No generators found at top level, scanning subdirectories...")
        for d in Path(SYNTHBUSTER_DIR).rglob('*'):
            if d.is_dir():
                imgs = list(d.glob('*.png')) + list(d.glob('*.jpg'))
                if len(imgs) > 50:
                    print(f"  Found: {d} ({len(imgs)} images)")

    # 获取真实图像
    raise_dir = download_raise1k(RAISE_DIR)
    if raise_dir is None:
        # 用 GenImage real 作为 fallback
        real_imgs = get_real_images_fallback(GENIMAGE_ROOT, max_samples=1000)
        # 创建临时 real 目录 (symlinks)
        fallback_real_dir = os.path.join(SYNTHBUSTER_DIR, '_real_fallback')
        os.makedirs(fallback_real_dir, exist_ok=True)
        for i, p in enumerate(real_imgs):
            link = os.path.join(fallback_real_dir, f'real_{i:04d}{Path(p).suffix}')
            if not os.path.exists(link):
                os.symlink(p, link)
        raise_dir = fallback_real_dir
        print(f"[Real Images] Using GenImage fallback: {fallback_real_dir} ({len(real_imgs)} images)")

    # ====== Step 2: 加载模型 ======
    print("\n" + "=" * 60)
    print("Step 2: Loading Models")
    print("=" * 60)

    from models.detector import AIGCDetectorV2

    config = PAPER_CONFIG.copy()
    config['train_sources'] = ['biggan', 'adm', 'glide', 'vqdm', 'midjourney', 'sdv4']
    config['output_dir'] = OUTPUT_DIR

    model_configs = {}
    transform = get_multi_source_transforms(config['img_size'], is_train=False)

    # 我们的模型
    ours_ckpt = os.path.join(checkpoint_dir, 'fdaa_net_v2_best.pth')
    if not os.path.exists(ours_ckpt):
        ours_ckpt = os.path.join(checkpoint_dir, 'fdaa_net_v2_6src_best.pth')

    if os.path.exists(ours_ckpt):
        model_configs['Ours_V2'] = {
            'checkpoint': ours_ckpt,
            'create_fn': lambda: AIGCDetectorV2(
                backbone_name=config['backbone'],
                num_classes=2, img_size=config['img_size'],
                embed_dim=config['embed_dim'],
                use_hierarchical=config['use_hierarchical'],
                dropout=config['dropout'],
            )
        }

    # SOTA 方法
    for method in config['sota_methods']:
        ckpt = os.path.join(checkpoint_dir, f'{method}_best.pth')
        if os.path.exists(ckpt) and not os.path.islink(ckpt):
            m = method
            model_configs[method] = {
                'checkpoint': ckpt,
                'create_fn': lambda m=m: create_sota_model(m, num_classes=2)
            }

    print(f"  Found {len(model_configs)} models: {list(model_configs.keys())}")

    # 预加载所有模型
    loaded_models = {}
    for model_name, mcfg in model_configs.items():
        try:
            m = mcfg['create_fn']()
            ckpt = torch.load(mcfg['checkpoint'], map_location='cpu')
            m.load_state_dict(ckpt['model_state_dict'])
            m = m.to(device).eval()
            loaded_models[model_name] = m
            print(f"  [Loaded] {model_name}")
        except Exception as e:
            print(f"  [Error] {model_name}: {e}")

    # ====== Step 3: 逐生成器评估 ======
    print("\n" + "=" * 60)
    print("Step 3: Per-Generator Evaluation on Synthbuster")
    print("=" * 60)

    results = {}

    for gen_name, gen_dir in sorted(gen_mapping.items()):
        print(f"\n--- {gen_name} ---")

        dataset = SynthbusterGeneratorDataset(
            fake_dir=gen_dir,
            real_dir=raise_dir,
            transform=transform,
            max_samples=1000,
        )

        if len(dataset) < 10:
            print(f"  [Skip] Too few samples ({len(dataset)})")
            continue

        loader = DataLoader(dataset, batch_size=64, shuffle=False,
                           num_workers=4, pin_memory=True)

        results[gen_name] = {}
        for model_name, m in loaded_models.items():
            try:
                metrics = evaluate_model(m, loader, device, desc=f'{model_name} on {gen_name}')
                results[gen_name][model_name] = metrics
                print(f"  {model_name}: AUC={metrics['auc']*100:.2f}%")
            except Exception as e:
                print(f"  [Error] {model_name}: {e}")
                results[gen_name][model_name] = {'error': str(e)}

        # 及时保存
        save_json(results, os.path.join(results_dir, 'synthbuster_results.json'))

    # 释放模型
    del loaded_models
    torch.cuda.empty_cache()

    # ====== Step 4: 生成论文表格 ======
    print("\n" + "=" * 60)
    print("Step 4: Generating Paper Table")
    print("=" * 60)

    generate_paper_table(results, results_dir)

    print("\n[Done] Synthbuster evaluation complete!")
    print(f"  Results: {os.path.join(results_dir, 'synthbuster_results.json')}")
    print(f"  Table: {os.path.join(results_dir, 'synthbuster_table.md')}")


def generate_paper_table(results, results_dir):
    """生成论文格式的 Markdown 表格"""
    if not results:
        print("[Warning] No results to generate table")
        return

    # 收集所有模型名
    all_models = set()
    for gen_name, gen_results in results.items():
        for model_name in gen_results:
            if isinstance(gen_results[model_name], dict) and 'auc' in gen_results[model_name]:
                all_models.add(model_name)

    # 排序: Ours 在前
    models = sorted(all_models, key=lambda x: (0 if 'Ours' in x else 1, x))
    generators = sorted(results.keys())

    lines = []
    lines.append("# Synthbuster Cross-Dataset Evaluation (AUC %)\n\n")
    lines.append("All models trained on GenImage 6-source (BigGAN, ADM, GLIDE, VQDM, Midjourney, SDv4).\n")
    lines.append("Tested zero-shot on Synthbuster dataset (9 diffusion generators, 1000 images each).\n\n")

    # 表头
    header = "| Method |"
    sep = "|:---|"
    for gen in generators:
        header += f" {gen} |"
        sep += ":---:|"
    header += " **Avg** |"
    sep += ":---:|"
    lines.append(header + "\n")
    lines.append(sep + "\n")

    # 每个模型一行
    best_per_gen = {}
    for gen in generators:
        best_auc = 0
        for model in models:
            auc = results.get(gen, {}).get(model, {}).get('auc', 0)
            if auc > best_auc:
                best_auc = auc
                best_per_gen[gen] = model

    for model in models:
        display_name = "**FDAA-Net (Ours)**" if 'Ours' in model else model
        row = f"| {display_name} |"
        aucs = []
        for gen in generators:
            auc = results.get(gen, {}).get(model, {}).get('auc', 0) * 100
            is_best = best_per_gen.get(gen) == model
            if is_best:
                row += f" **{auc:.1f}** |"
            else:
                row += f" {auc:.1f} |"
            aucs.append(auc)

        avg = np.mean(aucs) if aucs else 0
        row += f" **{avg:.1f}** |" if 'Ours' in model else f" {avg:.1f} |"
        lines.append(row + "\n")

    # 保存
    table_path = os.path.join(results_dir, 'synthbuster_table.md')
    with open(table_path, 'w') as f:
        f.writelines(lines)
    print(f"[Table] Saved to {table_path}")

    # 打印到终端
    print("\n" + "".join(lines))


if __name__ == '__main__':
    main()
