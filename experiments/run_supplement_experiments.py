#!/usr/bin/env python3
"""
补充实验: 训练新SOTA方法 (RINE, SSP) + 鲁棒性消融 + 全部跨数据集评估(含DRCT)
一键完成所有补充实验
"""

import sys, os, json, glob
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'experiments'))

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import pyarrow.ipc as ipc

from run_paper_experiments import (
    PAPER_CONFIG, save_json, fmt,
    evaluate_model, DegradedDataset,
    get_multi_source_transforms, train_model,
    _create_ablation_model,
)
from data.multi_source_dataset import (
    create_multi_source_dataloader, MultiSourceGenImageDataset,
)
from models.sota_methods import create_sota_model
from models.detector import AIGCDetectorV2


def phase1_train_new_sota(config, device, num_gpus):
    """Phase 1: 训练 RINE 和 SSP"""
    output_dir = config['output_dir']
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    new_methods = ['rine', 'ssp']

    train_loader = create_multi_source_dataloader(
        genimage_root=config['genimage_root'],
        sources=config['train_sources'],
        split='train',
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        max_samples_per_source=config['max_samples_per_source'],
        num_workers=config['num_workers'],
        strong_aug=True,
    )

    val_loader = create_multi_source_dataloader(
        genimage_root=config['genimage_root'],
        sources=config['train_sources'],
        split='val',
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        max_samples_per_source=5000,
        num_workers=config['num_workers'],
        strong_aug=False,
    )

    for method in new_methods:
        ckpt_path = os.path.join(checkpoint_dir, f'{method}_best.pth')
        if os.path.exists(ckpt_path):
            print(f"\n  [Skip] {method}: checkpoint exists")
            continue

        print(f"\n{'='*60}")
        print(f"Training {method.upper()}")
        print(f"{'='*60}")

        try:
            model = create_sota_model(method, num_classes=2)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Trainable params: {trainable/1e6:.2f}M")

            train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                model_name=method,
                output_dir=config['output_dir'],
                device=device,
                num_gpus=num_gpus,
                is_our_model=False,
            )
        except Exception as e:
            print(f"  [Error] {method}: {e}")
            import traceback; traceback.print_exc()

    torch.cuda.empty_cache()


def phase2_robustness_ablation(config, device):
    """Phase 2: 鲁棒性消融 (baseline / +FDAA / +MGFP / full)"""
    output_dir = config['output_dir']
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    results_dir = os.path.join(output_dir, 'results')

    result_file = os.path.join(results_dir, 'robustness_ablation_results.json')
    if os.path.exists(result_file):
        print("\n  [Skip] robustness_ablation_results.json exists")
        return

    print(f"\n{'='*60}")
    print("Robustness Ablation")
    print(f"{'='*60}")

    variants = {
        'Baseline': ('baseline', 'abl_baseline_best.pth'),
        '+FDAA': ('baseline+fdaa', 'abl_baseline_fdaa_best.pth'),
        '+MGFP': ('baseline+mgfp', 'abl_baseline_mgfp_best.pth'),
        'Full (Ours)': ('full', None),
    }

    # Find full checkpoint
    for name in ['fdaa_net_v2_6src_best.pth', 'fdaa_net_v2_best.pth']:
        if os.path.exists(os.path.join(checkpoint_dir, name)):
            variants['Full (Ours)'] = ('full', name)
            break

    loaded_models = {}
    for display_name, (variant, ckpt_name) in variants.items():
        if ckpt_name is None:
            print(f"  [Skip] {display_name}: no checkpoint")
            continue
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"  [Skip] {display_name}: {ckpt_name} not found")
            continue
        try:
            m = _create_ablation_model(config, variant)
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            m.load_state_dict(ckpt['model_state_dict'])
            m = m.to(device).eval()
            loaded_models[display_name] = m
            params = sum(p.numel() for p in m.parameters() if p.requires_grad)
            print(f"  [Loaded] {display_name} ({params/1e6:.2f}M)")
        except Exception as e:
            print(f"  [Error] {display_name}: {e}")

    if not loaded_models:
        print("  [Error] No models loaded")
        return

    base_dataset = MultiSourceGenImageDataset(
        genimage_root=config['genimage_root'],
        sources=config['train_sources'],
        split='val',
        transform=None,
        max_samples_per_source=2000,
        balance_classes=True,
    )

    results = {}
    for test_name, test_cfg in config['robustness_tests'].items():
        deg_type = test_cfg['type']
        deg_param = test_cfg.get('quality') or test_cfg.get('radius') or test_cfg.get('std')
        print(f"\n--- {test_name} ---")

        degraded = DegradedDataset(base_dataset, deg_type, deg_param, config['img_size'])
        loader = DataLoader(degraded, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

        results[test_name] = {}
        for model_name, m in loaded_models.items():
            try:
                metrics = evaluate_model(m, loader, device, desc=f'{model_name} on {test_name}')
                results[test_name][model_name] = metrics
                print(f"  {model_name}: AUC={metrics['auc']*100:.2f}%")
            except Exception as e:
                results[test_name][model_name] = {'error': str(e)}

        save_json(results, result_file)

    del loaded_models; torch.cuda.empty_cache()


def phase3_crossdataset_new_methods(config, device):
    """Phase 3: 对新方法+DRCT跑跨数据集评估 (Ojha + Synthbuster + ForenSynths)"""
    output_dir = config['output_dir']
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    results_dir = os.path.join(output_dir, 'results')

    # 需要补充评估的方法
    methods_to_eval = ['drct', 'rine', 'ssp']

    print(f"\n{'='*60}")
    print("Cross-dataset Evaluation for New Methods")
    print(f"{'='*60}")

    transform = get_multi_source_transforms(config['img_size'], is_train=False)

    loaded_models = {}
    for method in methods_to_eval:
        ckpt_path = os.path.join(checkpoint_dir, f'{method}_best.pth')
        if not os.path.exists(ckpt_path):
            print(f"  [Skip] {method}: no checkpoint")
            continue
        try:
            m = create_sota_model(method, num_classes=2)
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            m.load_state_dict(ckpt['model_state_dict'])
            loaded_models[method] = m.to(device).eval()
            print(f"  [Loaded] {method}")
        except Exception as e:
            print(f"  [Error] {method}: {e}")

    if not loaded_models:
        print("  No new models to evaluate")
        return

    # === Ojha ===
    ojha_pattern = os.path.expanduser("~/.cache/huggingface/hub/datasets--nebula--DF-arrow/snapshots/*/Ojha/*.arrow")
    ojha_files = sorted(glob.glob(ojha_pattern))

    if ojha_files:
        print(f"\n--- Ojha Evaluation ({len(ojha_files)} files) ---")

        # Load existing results
        ojha_results_file = os.path.join(results_dir, 'crossdataset_ojha_results.json')
        if os.path.exists(ojha_results_file):
            with open(ojha_results_file) as f:
                ojha_results = json.load(f)
        else:
            ojha_results = {}

        generators = {
            'dalle': 'DALL-E', 'glide_100_27': 'GLIDE',
            'guided': 'Guided Diffusion', 'ldm_200': 'LDM-200',
            'ldm_200_cfg': 'LDM-200-CFG',
        }

        from run_ojha_eval_only import ArrowImageDataset

        for gen_key, gen_name in generators.items():
            if gen_name not in ojha_results:
                ojha_results[gen_name] = {}

            # Skip if all methods already evaluated
            missing = [m for m in loaded_models if m not in ojha_results[gen_name]]
            if not missing:
                continue

            print(f"\n  {gen_name}:")
            dataset = ArrowImageDataset(ojha_files, generator_filter=gen_key, transform=transform)
            if len(dataset) < 10:
                continue
            loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

            for method, m in loaded_models.items():
                if method in ojha_results[gen_name]:
                    continue
                try:
                    metrics = evaluate_model(m, loader, device, desc=f'{method} on {gen_name}')
                    ojha_results[gen_name][method] = metrics
                    print(f"    {method}: AUC={metrics['auc']*100:.2f}%")
                except Exception as e:
                    ojha_results[gen_name][method] = {'error': str(e)}

            save_json(ojha_results, ojha_results_file)
            torch.cuda.empty_cache()

    # === Synthbuster ===
    synth_pattern = os.path.expanduser("~/.cache/huggingface/hub/datasets--nebula--DF-arrow/snapshots/*/synthbuster/*.arrow")
    synth_files = sorted(glob.glob(synth_pattern))

    if synth_files:
        print(f"\n--- Synthbuster Evaluation ({len(synth_files)} files) ---")

        synth_results_file = os.path.join(results_dir, 'crossdataset_synthbuster_results.json')
        if os.path.exists(synth_results_file):
            with open(synth_results_file) as f:
                synth_results = json.load(f)
        else:
            synth_results = {}

        from run_synthbuster_hf_eval import SynthbusterArrowDataset

        gen_display = {
            'dalle2': 'DALL-E 2', 'dalle3': 'DALL-E 3', 'firefly': 'Firefly',
            'glide': 'GLIDE', 'midjourney-v5': 'midjourney-v5',
            'stable-diffusion-1-3': 'stable-diffusion-1-3',
            'stable-diffusion-1-4': 'stable-diffusion-1-4',
            'stable-diffusion-2': 'stable-diffusion-2',
            'stable-diffusion-xl': 'stable-diffusion-xl',
        }

        for gen_key, display in gen_display.items():
            if display not in synth_results:
                continue  # Only add to existing generators

            missing = [m for m in loaded_models if m not in synth_results[display]]
            if not missing:
                continue

            print(f"\n  {display}:")
            dataset = SynthbusterArrowDataset(synth_files, generator_filter=gen_key, transform=transform)
            if len(dataset) < 10:
                continue
            loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

            for method, m in loaded_models.items():
                if method in synth_results[display]:
                    continue
                try:
                    metrics = evaluate_model(m, loader, device, desc=f'{method} on {display}')
                    synth_results[display][method] = metrics
                    print(f"    {method}: AUC={metrics['auc']*100:.2f}%")
                except Exception as e:
                    synth_results[display][method] = {'error': str(e)}

            save_json(synth_results, synth_results_file)
            torch.cuda.empty_cache()

    # === ForenSynths ===
    foren_pattern = os.path.expanduser("~/.cache/huggingface/hub/datasets--nebula--DF-arrow/snapshots/*/ForenSynths/*.arrow")
    foren_files = sorted(glob.glob(foren_pattern))

    if foren_files:
        print(f"\n--- ForenSynths Evaluation ({len(foren_files)} files) ---")

        foren_results_file = os.path.join(results_dir, 'crossdataset_forensynths_results.json')
        if os.path.exists(foren_results_file):
            with open(foren_results_file) as f:
                foren_results = json.load(f)
        else:
            foren_results = {}

        from run_forensynths_eval import ForenSynthsDataset

        for gen_display in list(foren_results.keys()):
            gen_key = gen_display.lower().replace(' ', '')
            # Map display names back to folder names
            gen_map = {
                'BigGAN': 'biggan', 'CRN': 'crn', 'CycleGAN': 'cyclegan',
                'Deepfake': 'deepfake', 'GauGAN': 'gaugan', 'IMLE': 'imle',
                'ProGAN': 'progan', 'SAN': 'san', 'StarGAN': 'stargan',
                'StyleGAN': 'stylegan', 'StyleGAN2': 'stylegan2',
            }
            gen_key = gen_map.get(gen_display, gen_display.lower())

            missing = [m for m in loaded_models if m not in foren_results[gen_display]]
            if not missing:
                continue

            print(f"\n  {gen_display}:")
            dataset = ForenSynthsDataset(foren_files, generator_filter=gen_key, transform=transform)
            if len(dataset) < 10:
                continue
            loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

            for method, m in loaded_models.items():
                if method in foren_results[gen_display]:
                    continue
                try:
                    metrics = evaluate_model(m, loader, device, desc=f'{method} on {gen_display}')
                    foren_results[gen_display][method] = metrics
                    print(f"    {method}: AUC={metrics['auc']*100:.2f}%")
                except Exception as e:
                    foren_results[gen_display][method] = {'error': str(e)}

            save_json(foren_results, foren_results_file)
            torch.cuda.empty_cache()

    del loaded_models; torch.cuda.empty_cache()


def phase4_robustness_new_methods(config, device):
    """Phase 4: 新方法的鲁棒性评估"""
    output_dir = config['output_dir']
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    results_dir = os.path.join(output_dir, 'results')

    rob_file = os.path.join(results_dir, 'robustness_results.json')
    if os.path.exists(rob_file):
        with open(rob_file) as f:
            rob_results = json.load(f)
    else:
        rob_results = {}

    methods_to_add = ['rine', 'ssp']  # drct should already be there

    loaded_models = {}
    for method in methods_to_add:
        ckpt_path = os.path.join(checkpoint_dir, f'{method}_best.pth')
        if not os.path.exists(ckpt_path):
            continue
        try:
            m = create_sota_model(method, num_classes=2)
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            m.load_state_dict(ckpt['model_state_dict'])
            loaded_models[method] = m.to(device).eval()
            print(f"  [Loaded] {method}")
        except Exception as e:
            print(f"  [Error] {method}: {e}")

    if not loaded_models:
        return

    base_dataset = MultiSourceGenImageDataset(
        genimage_root=config['genimage_root'],
        sources=config['train_sources'],
        split='val', transform=None,
        max_samples_per_source=2000, balance_classes=True,
    )

    for test_name, test_cfg in config['robustness_tests'].items():
        deg_type = test_cfg['type']
        deg_param = test_cfg.get('quality') or test_cfg.get('radius') or test_cfg.get('std')

        if test_name not in rob_results:
            rob_results[test_name] = {}

        missing = [m for m in loaded_models if m not in rob_results[test_name]]
        if not missing:
            continue

        print(f"\n--- {test_name} ---")
        degraded = DegradedDataset(base_dataset, deg_type, deg_param, config['img_size'])
        loader = DataLoader(degraded, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

        for method, m in loaded_models.items():
            if method in rob_results[test_name]:
                continue
            try:
                metrics = evaluate_model(m, loader, device, desc=f'{method} on {test_name}')
                rob_results[test_name][method] = metrics
                print(f"  {method}: AUC={metrics['auc']*100:.2f}%")
            except Exception as e:
                rob_results[test_name][method] = {'error': str(e)}

        save_json(rob_results, rob_file)
        torch.cuda.empty_cache()

    del loaded_models; torch.cuda.empty_cache()


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    num_gpus = torch.cuda.device_count()

    config = PAPER_CONFIG.copy()
    config['output_dir'] = os.path.join(PROJECT_ROOT, 'outputs', 'paper_results_6src')
    config['train_sources'] = ['biggan', 'adm', 'glide', 'vqdm', 'midjourney', 'sdv4']

    print("=" * 60)
    print("Supplementary Experiments")
    print(f"GPU: {num_gpus}x, Device: {device}")
    print("=" * 60)

    # Phase 1: Train new SOTA methods
    print("\n\n" + "=" * 60)
    print("PHASE 1: Training RINE & SSP")
    print("=" * 60)
    phase1_train_new_sota(config, device, num_gpus)

    # Phase 2: Robustness ablation
    print("\n\n" + "=" * 60)
    print("PHASE 2: Robustness Ablation")
    print("=" * 60)
    phase2_robustness_ablation(config, device)

    # Phase 3: Cross-dataset for new methods
    print("\n\n" + "=" * 60)
    print("PHASE 3: Cross-dataset Evaluation (DRCT + RINE + SSP)")
    print("=" * 60)
    phase3_crossdataset_new_methods(config, device)

    # Phase 4: Robustness for new methods
    print("\n\n" + "=" * 60)
    print("PHASE 4: Robustness for New Methods")
    print("=" * 60)
    phase4_robustness_new_methods(config, device)

    print("\n\n" + "=" * 60)
    print("ALL SUPPLEMENTARY EXPERIMENTS DONE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
