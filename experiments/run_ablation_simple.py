"""
简化版消融实验脚本

直接使用 HuggingFace 流式数据集运行消融实验

Usage:
    LD_LIBRARY_PATH="" python experiments/run_ablation_simple.py
    LD_LIBRARY_PATH="" python experiments/run_ablation_simple.py --epochs 3 --configs baseline,full
"""

import os
import sys
import json
import argparse
from datetime import datetime
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.detector_ablation import create_ablation_model, ABLATION_CONFIGS
from models.losses import AIGCDetectionLoss
from data.streaming_dataset import create_streaming_dataloader
from utils.metrics import MetricsAccumulator
from utils.logger import AverageMeter


def train_epoch(model, dataloader, criterion, optimizer, device, max_batches=None):
    """训练一个epoch"""
    model.train()
    loss_meter = AverageMeter('Loss', ':.4f')
    acc_meter = AverageMeter('Acc', ':.4f')

    pbar = tqdm(dataloader, desc='Training', leave=False)
    for batch_idx, batch in enumerate(pbar):
        if max_batches and batch_idx >= max_batches:
            break

        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss_dict = criterion(outputs, labels)
        loss = loss_dict['total_loss']

        loss.backward()
        optimizer.step()

        preds = outputs['logits'].argmax(dim=1)
        acc = (preds == labels).float().mean()

        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc.item(), images.size(0))

        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}', 'acc': f'{acc_meter.avg:.4f}'})

    return {'loss': loss_meter.avg, 'acc': acc_meter.avg}


@torch.no_grad()
def validate(model, dataloader, criterion, device, max_batches=None):
    """验证"""
    model.eval()
    loss_meter = AverageMeter('Loss', ':.4f')
    accumulator = MetricsAccumulator()

    pbar = tqdm(dataloader, desc='Validating', leave=False)
    for batch_idx, batch in enumerate(pbar):
        if max_batches and batch_idx >= max_batches:
            break

        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        outputs = model(images)
        loss_dict = criterion(outputs, labels)

        probs = torch.softmax(outputs['logits'], dim=1)[:, 1]
        accumulator.update(probs.cpu(), labels.cpu())
        loss_meter.update(loss_dict['total_loss'].item(), images.size(0))

    metrics = accumulator.compute()
    metrics['loss'] = loss_meter.avg
    return metrics


def run_single_ablation(config_name, args, device):
    """运行单个消融配置"""
    config_info = ABLATION_CONFIGS[config_name]
    print(f"\n{'='*60}")
    print(f"配置: {config_info['name']}")
    print(f"{'='*60}")

    # 创建模型
    model = create_ablation_model(
        config_name,
        num_classes=2,
        img_size=args.img_size,
        embed_dim=args.embed_dim,
        dropout=0.1
    ).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {params:,}")

    # 数据加载器
    train_loader = create_streaming_dataloader(
        dataset_name=args.dataset,
        split='train',
        batch_size=args.batch_size,
        img_size=args.img_size,
        max_samples=args.train_samples
    )

    val_loader = create_streaming_dataloader(
        dataset_name=args.dataset,
        split='train',
        batch_size=args.batch_size,
        img_size=args.img_size,
        max_samples=args.val_samples,
        skip_samples=args.train_samples
    )

    # 损失函数和优化器
    criterion = AIGCDetectionLoss(use_focal=True, use_aux=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练
    best_auc = 0
    best_metrics = {}

    for epoch in range(args.epochs):
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            max_batches=args.train_samples // args.batch_size
        )

        val_metrics = validate(
            model, val_loader, criterion, device,
            max_batches=args.val_samples // args.batch_size
        )

        scheduler.step()

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_metrics = val_metrics.copy()
            best_metrics['epoch'] = epoch

        print(f"  Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, "
              f"Val AUC={val_metrics['auc']:.4f}, Val Acc={val_metrics['accuracy']:.4f}")

    print(f"\n  最佳结果 (Epoch {best_metrics.get('epoch', 0)}):")
    print(f"    AUC: {best_metrics['auc']:.4f}")
    print(f"    Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"    AP: {best_metrics.get('ap', 0):.4f}")

    return {
        'config': config_name,
        'name': config_info['name'],
        'params': params,
        'best_auc': best_metrics['auc'],
        'best_acc': best_metrics['accuracy'],
        'best_ap': best_metrics.get('ap', 0),
        'best_epoch': best_metrics.get('epoch', 0)
    }


def main(args):
    print("="*60)
    print("AIGC Detection - 消融实验")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"数据集: {args.dataset}")
    print(f"训练样本: {args.train_samples}, 验证样本: {args.val_samples}")
    print(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}")

    # 确定要运行的配置
    if args.configs:
        configs = args.configs.split(',')
    else:
        configs = list(ABLATION_CONFIGS.keys())

    print(f"\n将运行 {len(configs)} 个配置: {configs}")

    # 运行消融实验
    results = []
    for config_name in configs:
        if config_name not in ABLATION_CONFIGS:
            print(f"警告: 未知配置 {config_name}, 跳过")
            continue

        result = run_single_ablation(config_name, args, device)
        results.append(result)

    # 输出结果汇总
    print("\n" + "="*60)
    print("消融实验结果汇总")
    print("="*60)
    print(f"{'配置':<35} {'参数量':<12} {'AUC':<8} {'Acc':<8} {'AP':<8}")
    print("-"*71)

    for r in results:
        print(f"{r['name']:<35} {r['params']:<12,} {r['best_auc']:<8.4f} "
              f"{r['best_acc']:<8.4f} {r['best_ap']:<8.4f}")

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    result_file = os.path.join(args.output_dir, f"ablation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(result_file, 'w') as f:
        json.dump({
            'args': vars(args),
            'results': results
        }, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {result_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ablation experiments')

    # 数据集
    parser.add_argument('--dataset', type=str,
                        default='Hemg/AI-Generated-vs-Real-Images-Datasets',
                        help='Dataset name')
    parser.add_argument('--train_samples', type=int, default=3000,
                        help='Number of training samples')
    parser.add_argument('--val_samples', type=int, default=1000,
                        help='Number of validation samples')

    # 模型
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--embed_dim', type=int, default=768)

    # 训练
    parser.add_argument('--epochs', type=int, default=5,
                        help='Epochs per config')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)

    # 消融配置
    parser.add_argument('--configs', type=str, default=None,
                        help='Comma-separated config names (default: all)')

    # 输出
    parser.add_argument('--output_dir', type=str, default='./outputs/ablation')

    args = parser.parse_args()
    main(args)
