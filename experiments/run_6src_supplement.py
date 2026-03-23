#!/usr/bin/env python3
"""
6源补充实验: SOTA重训 + 消融实验
在 run_6src.sh 完成后自动执行
"""

import sys
import os
import json
import copy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'experiments'))

import torch
import numpy as np

from run_paper_experiments import (
    PAPER_CONFIG, get_device, setup_output_dir, save_json, fmt,
    train_model, evaluate_model,
    create_multi_source_dataloader,
    exp_train_sota, exp_ablation,
    exp_cross_domain, exp_robustness, exp_visualization, exp_efficiency,
)


def main():
    device, num_gpus = get_device()

    config = copy.deepcopy(PAPER_CONFIG)
    config['train_sources'] = ['biggan', 'adm', 'glide', 'vqdm', 'midjourney', 'sdv4']
    config['output_dir'] = os.path.join(PROJECT_ROOT, 'outputs', 'paper_results_6src')

    # SOTA 训练加速参数 (不影响消融/评估, 它们各自有 dataloader)
    config['sota_epochs'] = 10
    config['sota_val_samples'] = 2000
    # batch_size 保持 64 — 消融实验的 V2 模型大 (323M), 128 会 OOM

    output_dir = setup_output_dir(config['output_dir'])

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='all',
                        choices=['sota', 'ablation', 'cross_domain', 'robustness',
                                 'visualization', 'efficiency', 'report', 'all'])
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

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
    # Step 1: SOTA 方法 6源重训 (公平对比)
    # =========================================================================
    if args.mode in ['all', 'sota']:
        if not _should_skip('sota_6src_train_results.json', 'sota_6src'):
            print("\n" + "=" * 60)
            print("SOTA 6源重训 (10 methods × 20 epochs)")
            print(f"训练源: {config['train_sources']}")
            print("=" * 60)

            # 先删掉旧的 4源 SOTA symlinks，防止被跳过
            ckpt_dir = os.path.join(output_dir, 'checkpoints')
            for method in config['sota_methods']:
                link = os.path.join(ckpt_dir, f'{method}_best.pth')
                if os.path.islink(link):
                    os.unlink(link)
                    print(f"  Removed symlink: {method}")

            exp_train_sota(config, device, num_gpus)

            # 重命名结果文件
            src = os.path.join(output_dir, 'results', 'sota_train_results.json')
            dst = os.path.join(output_dir, 'results', 'sota_6src_train_results.json')
            if os.path.exists(src):
                os.rename(src, dst)
                print(f"  Renamed: {src} -> {dst}")

    # =========================================================================
    # Step 2: 6源消融实验 (4 variants × 10 epochs)
    # =========================================================================
    if args.mode in ['all', 'ablation']:
        if not _should_skip('ablation_6src_results.json', 'ablation_6src'):
            print("\n" + "=" * 60)
            print("6源消融实验 (4 variants × 10 epochs)")
            print("=" * 60)

            # 确保主模型 checkpoint 存在 (用于 full 变体)
            main_ckpt = os.path.join(output_dir, 'checkpoints', 'fdaa_net_v2_6src_best.pth')
            v2_ckpt = os.path.join(output_dir, 'checkpoints', 'fdaa_net_v2_best.pth')

            # exp_ablation 查找 fdaa_net_v2_best.pth
            if os.path.exists(main_ckpt) and not os.path.exists(v2_ckpt):
                os.symlink(main_ckpt, v2_ckpt)

            # 同样处理训练日志
            main_log = os.path.join(output_dir, 'logs', 'fdaa_net_v2_6src_training_log.json')
            v2_log = os.path.join(output_dir, 'logs', 'fdaa_net_v2_training_log.json')
            if os.path.exists(main_log) and not os.path.exists(v2_log):
                os.symlink(main_log, v2_log)

            exp_ablation(config, device, num_gpus)

            # 重命名结果文件
            src = os.path.join(output_dir, 'results', 'ablation_results.json')
            dst = os.path.join(output_dir, 'results', 'ablation_6src_results.json')
            if os.path.exists(src):
                os.rename(src, dst)

    # =========================================================================
    # Step 3: 重新跑跨域评估 (6源模型 + 6源SOTA)
    # =========================================================================
    if args.mode in ['all', 'cross_domain']:
        # 删除旧结果强制重跑
        old = os.path.join(output_dir, 'results', 'cross_domain_results.json')
        if os.path.exists(old):
            os.remove(old)
        print("\n" + "=" * 60)
        print("跨域评估 (6源模型 + 6源SOTA)")
        print("=" * 60)
        exp_cross_domain(config, device)

    # =========================================================================
    # Step 4: 重新跑鲁棒性测试
    # =========================================================================
    if args.mode in ['all', 'robustness']:
        old = os.path.join(output_dir, 'results', 'robustness_results.json')
        if os.path.exists(old):
            os.remove(old)
        print("\n" + "=" * 60)
        print("鲁棒性测试 (6源模型 + 6源SOTA)")
        print("=" * 60)
        exp_robustness(config, device)

    # =========================================================================
    # Step 5: 可视化 + 效率
    # =========================================================================
    if args.mode in ['all', 'visualization', 'efficiency']:
        print("\n" + "=" * 60)
        print("可视化 + 效率分析")
        print("=" * 60)
        exp_visualization(config, device)
        exp_efficiency(config, device)

    # =========================================================================
    # Step 6: 生成完整报告
    # =========================================================================
    if args.mode in ['all', 'report']:
        print("\n" + "=" * 60)
        print("生成完整论文报告")
        print("=" * 60)

        results_dir = os.path.join(output_dir, 'results')
        report_lines = []
        report_lines.append("# FDAA-Net V2 6源完整实验报告\n\n")
        report_lines.append(f"训练源: {config['train_sources']}\n\n")

        # 1. 域内训练结果
        train_f = os.path.join(results_dir, 'ours_6src_train_results.json')
        if os.path.exists(train_f):
            d = json.load(open(train_f))
            report_lines.append("## Table 1: 域内检测性能\n\n")
            report_lines.append("| Method | AUC (%) | AP (%) | Accuracy (%) | EER (%) |\n")
            report_lines.append("|--------|---------|--------|--------------|--------|\n")
            for k, v in d.items():
                report_lines.append(f"| FDAA-Net V2 (Ours) | {v['auc']*100:.2f} | {v['ap']*100:.2f} | {v['accuracy']*100:.2f} | {v['eer']*100:.2f} |\n")

        # SOTA 结果
        sota_f = os.path.join(results_dir, 'sota_6src_train_results.json')
        if os.path.exists(sota_f):
            d = json.load(open(sota_f))
            for method, v in sorted(d.items(), key=lambda x: -x[1].get('auc', 0)):
                if 'auc' in v:
                    report_lines.append(f"| {method} | {v['auc']*100:.2f} | {v['ap']*100:.2f} | {v['accuracy']*100:.2f} | {v['eer']*100:.2f} |\n")
            report_lines.append("\n")

        # 2. 消融实验
        abl_f = os.path.join(results_dir, 'ablation_6src_results.json')
        if os.path.exists(abl_f):
            d = json.load(open(abl_f))
            report_lines.append("## Table 2: 消融实验\n\n")
            report_lines.append("| Variant | FDAA | MGFP | Params (M) | AUC (%) | Acc (%) | ΔAUC |\n")
            report_lines.append("|---------|------|------|-----------|---------|---------|------|\n")
            baseline_auc = d.get('baseline', {}).get('auc', 0)
            for variant in ['baseline', 'baseline+fdaa', 'baseline+mgfp', 'full']:
                if variant in d:
                    v = d[variant]
                    fdaa = '✓' if 'fdaa' in variant or variant == 'full' else '✗'
                    mgfp = '✓' if 'mgfp' in variant or variant == 'full' else '✗'
                    params = v.get('trainable_params_M', 0)
                    delta = f"+{(v['auc']-baseline_auc)*100:.2f}" if variant != 'baseline' else '-'
                    report_lines.append(f"| {variant} | {fdaa} | {mgfp} | {params:.1f} | {v['auc']*100:.2f} | {v['accuracy']*100:.2f} | {delta} |\n")
            report_lines.append("\n")

        # 3. LOO
        loo_f = os.path.join(results_dir, 'leave_one_out_6src_results.json')
        if os.path.exists(loo_f):
            d = json.load(open(loo_f))
            report_lines.append("## Table 3: Leave-One-Out 跨生成器泛化\n\n")
            report_lines.append("| Held-out Generator | Train Sources | AUC (%) | Acc (%) |\n")
            report_lines.append("|-------------------|---------------|---------|--------|\n")
            for src in config['train_sources']:
                if src in d:
                    m = d[src]['metrics']
                    trains = ', '.join(d[src]['train_sources'])
                    report_lines.append(f"| {src} | {trains} | {m['auc']*100:.2f} | {m['accuracy']*100:.2f} |\n")
            if '_average' in d:
                report_lines.append(f"| **Average** | - | **{d['_average']['auc']*100:.2f} ± {d['_average']['std']*100:.2f}** | - |\n")
            report_lines.append("\n")

        # 4. 跨域
        cd_f = os.path.join(results_dir, 'cross_domain_results.json')
        if os.path.exists(cd_f):
            d = json.load(open(cd_f))
            report_lines.append("## Table 4: 跨数据集泛化 — AUC (%)\n\n")
            datasets = ['cifake', 'diffusion_forensics', 'ntire2026']
            models = []
            for ds in datasets:
                if ds in d:
                    for m in d[ds]:
                        if isinstance(d[ds][m], dict) and 'auc' in d[ds][m] and m not in models:
                            models.append(m)
            if models:
                header = "| Dataset | " + " | ".join(models) + " |\n"
                sep = "|---------|" + "|".join(["------" for _ in models]) + "|\n"
                report_lines.append(header)
                report_lines.append(sep)
                for ds in datasets:
                    if ds in d:
                        row = f"| {ds} |"
                        for m in models:
                            auc = d[ds].get(m, {}).get('auc', 0) * 100
                            row += f" {auc:.2f} |"
                        report_lines.append(row + "\n")
            report_lines.append("\n")

        # 5. 鲁棒性
        rob_f = os.path.join(results_dir, 'robustness_results.json')
        if os.path.exists(rob_f):
            report_lines.append("## Table 5: 鲁棒性评估\n\n")
            report_lines.append("(见 robustness_results.json)\n\n")

        report_path = os.path.join(output_dir, 'reports', 'full_paper_report_6src.md')
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            f.writelines(report_lines)
        print(f"[Report] {report_path}")

    print("\n" + "=" * 60)
    print("6源补充实验全部完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
