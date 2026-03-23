#!/usr/bin/env python3
"""
6源训练实验: biggan + adm + glide + vqdm + midjourney + sdv4
复用 run_paper_experiments.py 中的所有训练/评估函数
输出到独立目录 outputs/paper_results_6src/
"""

import sys
import os
import json
import copy

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'experiments'))

import torch
import numpy as np

# 复用现有实验框架
from run_paper_experiments import (
    PAPER_CONFIG, get_device, setup_output_dir, save_json, fmt,
    train_model, evaluate_model,
    create_multi_source_dataloader,
    exp_cross_domain, exp_robustness, exp_visualization, exp_efficiency,
)
from models.detector import AIGCDetectorV2


def main():
    device, num_gpus = get_device()

    # 6源配置 — 在原配置基础上修改
    config = copy.deepcopy(PAPER_CONFIG)
    config['train_sources'] = ['biggan', 'adm', 'glide', 'vqdm', 'midjourney', 'sdv4']
    config['output_dir'] = os.path.join(PROJECT_ROOT, 'outputs', 'paper_results_6src')

    output_dir = setup_output_dir(config['output_dir'])

    # 保存配置
    save_json(config, os.path.join(output_dir, 'paper_config_6src.json'))

    print("=" * 60)
    print("6源 V2 训练实验")
    print(f"训练源: {config['train_sources']}")
    print(f"输出: {output_dir}")
    print("=" * 60)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='all',
                        choices=['train', 'cross_domain', 'robustness', 'loo',
                                 'visualization', 'efficiency', 'report', 'all'])
    parser.add_argument('--genimage_root', type=str, default=None)
    parser.add_argument('--diffusion_forensics_root', type=str, default=None)
    parser.add_argument('--cifake_root', type=str, default=None)
    parser.add_argument('--ntire2026_root', type=str, default=None)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    if args.genimage_root:
        config['genimage_root'] = args.genimage_root
    if args.diffusion_forensics_root:
        config['eval_datasets']['diffusion_forensics'] = args.diffusion_forensics_root
    if args.cifake_root:
        config['eval_datasets']['cifake'] = args.cifake_root
    if args.ntire2026_root:
        config['eval_datasets']['ntire2026'] = args.ntire2026_root

    force = args.force

    def _should_skip(result_file, label):
        fpath = os.path.join(output_dir, 'results', result_file)
        if force:
            return False
        if os.path.exists(fpath) and os.path.getsize(fpath) > 10:
            print(f"\n[Skip] {label}: {fpath} exists (use --force)")
            return True
        return False

    # =========================================================================
    # Step 1: 训练 6源 V2 主模型
    # =========================================================================
    if args.mode in ['all', 'train']:
        if not _should_skip('ours_6src_train_results.json', 'train'):
            print("\n" + "=" * 60)
            print("Step 1: Training V2 with 6 sources (20 epochs)")
            print("=" * 60)

            source_to_id = {s: i for i, s in enumerate(config['train_sources'])}

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

            model = AIGCDetectorV2(
                backbone_name=config['backbone'],
                num_classes=2,
                img_size=config['img_size'],
                embed_dim=config['embed_dim'],
                use_hierarchical=config['use_hierarchical'],
                dropout=config['dropout'],
                freeze_backbone=True,
            )

            best_metrics = train_model(
                model, train_loader, val_loader, config,
                model_name='fdaa_net_v2_6src',
                output_dir=output_dir,
                device=device, num_gpus=num_gpus,
                is_our_model=True, source_to_id=source_to_id,
            )

            save_json({'fdaa_net_v2_6src': best_metrics},
                      os.path.join(output_dir, 'results', 'ours_6src_train_results.json'))
            print(f"\n6src V2 Train: {fmt(best_metrics)}")
            del model; torch.cuda.empty_cache()

    # =========================================================================
    # Step 2: Leave-One-Out (6源版)
    # =========================================================================
    if args.mode in ['all', 'loo']:
        if not _should_skip('leave_one_out_6src_results.json', 'loo'):
            print("\n" + "=" * 60)
            print("Step 2: Leave-One-Out (6 sources)")
            print("=" * 60)

            all_sources = list(config['train_sources'])
            loo_epochs = min(config['epochs'], 10)
            results = {}

            for held_out in all_sources:
                train_sources = [s for s in all_sources if s != held_out]
                source_to_id = {s: i for i, s in enumerate(train_sources)}
                model_name = f'loo_6src_held_{held_out}'
                ckpt_path = os.path.join(output_dir, 'checkpoints', f'{model_name}_best.pth')
                log_path = os.path.join(output_dir, 'logs', f'{model_name}_training_log.json')

                print(f"\n--- Held-out: {held_out}, Train on: {train_sources} ---")

                # 跳过已完成的
                need_train = True
                if os.path.exists(ckpt_path) and os.path.exists(log_path):
                    try:
                        with open(log_path) as fp:
                            log_data = json.load(fp)
                        if len(log_data) >= loo_epochs:
                            print(f"  -> Already trained ({len(log_data)} epochs), loading checkpoint...")
                            need_train = False
                    except Exception:
                        pass

                test_loader = create_multi_source_dataloader(
                    genimage_root=config['genimage_root'],
                    sources=[held_out], split='val',
                    batch_size=config['batch_size'], img_size=config['img_size'],
                    max_samples_per_source=5000,
                    num_workers=config['num_workers'], strong_aug=False,
                )

                if test_loader is None or len(test_loader.dataset) == 0:
                    print(f"  [Warning] No test data for {held_out}, skipping")
                    continue

                if need_train:
                    train_loader = create_multi_source_dataloader(
                        genimage_root=config['genimage_root'],
                        sources=train_sources, split='train',
                        batch_size=config['batch_size'], img_size=config['img_size'],
                        max_samples_per_source=config['max_samples_per_source'],
                        num_workers=config['num_workers'], strong_aug=True,
                    )

                    val_loader = create_multi_source_dataloader(
                        genimage_root=config['genimage_root'],
                        sources=train_sources, split='val',
                        batch_size=config['batch_size'], img_size=config['img_size'],
                        max_samples_per_source=2000,
                        num_workers=config['num_workers'], strong_aug=False,
                    )

                    model = AIGCDetectorV2(
                        backbone_name=config['backbone'], num_classes=2,
                        img_size=config['img_size'], embed_dim=config['embed_dim'],
                        use_hierarchical=config['use_hierarchical'],
                        dropout=config['dropout'], freeze_backbone=True,
                    )

                    loo_config = config.copy()
                    loo_config['epochs'] = loo_epochs

                    train_model(
                        model, train_loader, val_loader, loo_config,
                        model_name=model_name,
                        output_dir=output_dir, device=device, num_gpus=num_gpus,
                        is_our_model=True, source_to_id=source_to_id,
                    )
                    del model; torch.cuda.empty_cache()

                # 加载 best checkpoint 测试
                model = AIGCDetectorV2(
                    backbone_name=config['backbone'], num_classes=2,
                    img_size=config['img_size'], embed_dim=config['embed_dim'],
                    use_hierarchical=config['use_hierarchical'],
                    dropout=config['dropout'], freeze_backbone=True,
                )
                if os.path.exists(ckpt_path):
                    ckpt = torch.load(ckpt_path, map_location='cpu')
                    model.load_state_dict(ckpt['model_state_dict'])
                model = model.to(device)
                test_metrics = evaluate_model(model, test_loader, device,
                                              desc=f'LOO-6src test {held_out}')

                results[held_out] = {
                    'train_sources': train_sources,
                    'held_out': held_out,
                    'metrics': test_metrics,
                }
                print(f"  6src Held-out {held_out}: {fmt(test_metrics)}")
                del model; torch.cuda.empty_cache()

            auc_values = [r['metrics']['auc'] for r in results.values()
                          if 'auc' in r.get('metrics', {})]
            if auc_values:
                results['_average'] = {
                    'auc': float(np.mean(auc_values)),
                    'std': float(np.std(auc_values)),
                }
                print(f"\n  6src Average LOO AUC: {np.mean(auc_values)*100:.2f}% "
                      f"± {np.std(auc_values)*100:.2f}%")

            save_json(results, os.path.join(output_dir, 'results',
                                            'leave_one_out_6src_results.json'))

    # =========================================================================
    # Step 3: 跨域评估
    # =========================================================================
    if args.mode in ['all', 'cross_domain']:
        if not _should_skip('cross_domain_results.json', 'cross_domain'):
            print("\n" + "=" * 60)
            print("Step 3: Cross-domain evaluation")
            print("=" * 60)

            # 需要把 4 源的 SOTA checkpoints 拷贝/链接过来
            # 因为 SOTA 方法不需要重训（它们与训练源无关的评估）
            src_ckpt_dir = os.path.join(PROJECT_ROOT, 'outputs', 'paper_results', 'checkpoints')
            dst_ckpt_dir = os.path.join(output_dir, 'checkpoints')

            # 链接 SOTA checkpoints
            for method in config['sota_methods']:
                src = os.path.join(src_ckpt_dir, f'{method}_best.pth')
                dst = os.path.join(dst_ckpt_dir, f'{method}_best.pth')
                if os.path.exists(src) and not os.path.exists(dst):
                    os.symlink(src, dst)
                    print(f"  Linked: {method}")

            # 链接 6源V2 checkpoint 为 exp_cross_domain 期望的名字
            src_6src = os.path.join(dst_ckpt_dir, 'fdaa_net_v2_6src_best.pth')
            dst_v2 = os.path.join(dst_ckpt_dir, 'fdaa_net_v2_best.pth')
            if os.path.exists(src_6src) and not os.path.exists(dst_v2):
                os.symlink(src_6src, dst_v2)
                print("  Linked: fdaa_net_v2_6src -> fdaa_net_v2 (for cross_domain)")

            exp_cross_domain(config, device)

    # =========================================================================
    # Step 4: 鲁棒性测试
    # =========================================================================
    if args.mode in ['all', 'robustness']:
        if not _should_skip('robustness_results.json', 'robustness'):
            print("\n" + "=" * 60)
            print("Step 4: Robustness evaluation")
            print("=" * 60)
            exp_robustness(config, device)

    # =========================================================================
    # Step 5: 可视化 + 效率
    # =========================================================================
    if args.mode in ['all', 'visualization']:
        print("\n" + "=" * 60)
        print("Step 5: Visualization + Efficiency")
        print("=" * 60)
        exp_visualization(config, device)
        exp_efficiency(config, device)

    # =========================================================================
    # Step 6: 生成报告
    # =========================================================================
    if args.mode in ['all', 'report']:
        print("\n" + "=" * 60)
        print("Step 6: Generate report")
        print("=" * 60)

        # 汇总所有结果
        results_dir = os.path.join(output_dir, 'results')
        report_lines = []
        report_lines.append("# FDAA-Net V2 6源实验报告\n")
        report_lines.append(f"训练源: {config['train_sources']}\n")

        # 训练结果
        train_f = os.path.join(results_dir, 'ours_6src_train_results.json')
        if os.path.exists(train_f):
            d = json.load(open(train_f))
            report_lines.append("## 域内训练结果\n")
            for k, v in d.items():
                report_lines.append(f"- {k}: AUC={v['auc']*100:.2f}% Acc={v['accuracy']*100:.2f}%\n")

        # LOO
        loo_f = os.path.join(results_dir, 'leave_one_out_6src_results.json')
        if os.path.exists(loo_f):
            d = json.load(open(loo_f))
            report_lines.append("\n## Leave-One-Out 跨生成器泛化\n")
            report_lines.append("| Held-out | AUC (%) | Acc (%) |\n|---|---|---|\n")
            for src in config['train_sources']:
                if src in d:
                    m = d[src]['metrics']
                    report_lines.append(f"| {src} | {m['auc']*100:.2f} | {m['accuracy']*100:.2f} |\n")
            if '_average' in d:
                report_lines.append(f"| **Average** | **{d['_average']['auc']*100:.2f} ± {d['_average']['std']*100:.2f}** | - |\n")

        report_path = os.path.join(output_dir, 'reports', 'report_6src.md')
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            f.writelines(report_lines)
        print(f"[Report] {report_path}")

    print("\n" + "=" * 60)
    print("6源实验完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
