#!/usr/bin/env python3
"""
6源 SOTA 并行训练 — 两张卡各跑一半方法
用法:
  CUDA_VISIBLE_DEVICES=0 python experiments/run_6src_sota_parallel.py --gpu-group 0
  CUDA_VISIBLE_DEVICES=1 python experiments/run_6src_sota_parallel.py --gpu-group 1
"""

import sys
import os
import json
import copy
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'experiments'))

import torch

from run_paper_experiments import (
    PAPER_CONFIG, setup_output_dir, save_json, fmt,
    train_model, evaluate_model,
    create_multi_source_dataloader, create_sota_model,
)

# 10 个方法分成两组
GROUP_0 = ['cnndetection', 'f3net', 'univfd', 'freqnet', 'npr']
GROUP_1 = ['spec', 'dire', 'lare2', 'drct', 'c2pclip']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-group', type=int, required=True, choices=[0, 1],
                        help='0=GPU0跑前5个, 1=GPU1跑后5个')
    args = parser.parse_args()

    device = 'cuda:0'  # 因为 CUDA_VISIBLE_DEVICES 已经限定了单卡
    num_gpus = 1  # 单卡，不用 DataParallel

    config = copy.deepcopy(PAPER_CONFIG)
    config['train_sources'] = ['biggan', 'adm', 'glide', 'vqdm', 'midjourney', 'sdv4']
    config['output_dir'] = os.path.join(PROJECT_ROOT, 'outputs', 'paper_results_6src')

    # 加速参数
    config['sota_epochs'] = 10
    config['batch_size'] = 128
    config['max_samples_per_source'] = 5000  # 5K/源 减少HDD IO瓶颈
    config['num_workers'] = 6  # 每卡分配6个worker

    output_dir = setup_output_dir(config['output_dir'])
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')

    methods = GROUP_0 if args.gpu_group == 0 else GROUP_1

    print(f"=" * 60)
    print(f"GPU Group {args.gpu_group}: Training {methods}")
    print(f"Config: epochs=10, batch=128, single GPU (no DataParallel)")
    print(f"=" * 60)

    # 创建 dataloader (每个 GPU 进程各自加载一份)
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
        max_samples_per_source=1000,
        num_workers=config['num_workers'],
        strong_aug=False,
    )

    results = {}

    for method_name in methods:
        print(f"\n{'=' * 60}")
        print(f"Training SOTA: {method_name} (GPU group {args.gpu_group})")
        print(f"{'=' * 60}")

        # 跳过已完成的
        ckpt_path = os.path.join(checkpoint_dir, f'{method_name}_best.pth')
        if os.path.exists(ckpt_path) and not os.path.islink(ckpt_path):
            try:
                print(f"  -> Checkpoint exists, evaluating: {ckpt_path}")
                model = create_sota_model(method_name, num_classes=2)
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model.load_state_dict(ckpt['model_state_dict'])
                model = model.to(device)
                metrics = evaluate_model(model, val_loader, device, desc=f'{method_name} eval')
                metrics['epoch'] = ckpt.get('epoch', -1)
                results[method_name] = metrics
                print(f"  {method_name} (reused): AUC={metrics['auc']*100:.2f}%")
                del model; torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"  [Warning] Checkpoint reuse failed: {e}, retraining...")

        # 删除旧的 symlink
        if os.path.islink(ckpt_path):
            os.unlink(ckpt_path)

        try:
            model = create_sota_model(method_name, num_classes=2)
            best_metrics = train_model(
                model, train_loader, val_loader, config,
                model_name=method_name,
                output_dir=output_dir,
                device=device, num_gpus=num_gpus,
                is_our_model=False,
            )
            results[method_name] = best_metrics
        except Exception as e:
            print(f"[Error] {method_name}: {e}")
            import traceback; traceback.print_exc()
            results[method_name] = {'error': str(e)}

        torch.cuda.empty_cache()

    # 保存本组结果
    result_file = os.path.join(output_dir, 'results', f'sota_6src_group{args.gpu_group}.json')
    save_json(results, result_file)
    print(f"\nGroup {args.gpu_group} done! Results: {result_file}")

    return results


if __name__ == '__main__':
    main()
