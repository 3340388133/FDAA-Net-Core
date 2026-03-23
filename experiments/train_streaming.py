"""
流式数据集训练脚本

使用 Hugging Face 流式数据集训练，不需要下载到本地

Usage:
    python experiments/train_streaming.py
    python experiments/train_streaming.py --epochs 30 --batch_size 16
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import AIGCDetectorLite
from models.losses import AIGCDetectionLoss
from data.streaming_dataset import (
    HFStreamingDataset,
    create_streaming_dataloader,
    get_streaming_transforms
)
from utils.metrics import MetricsAccumulator
from utils.logger import AverageMeter, setup_logger
from utils.checkpoint import save_checkpoint


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, max_batches=None):
    """训练一个epoch"""
    model.train()

    loss_meter = AverageMeter('Loss', ':.4f')
    acc_meter = AverageMeter('Acc', ':.4f')

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

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

        # 计算准确率
        preds = outputs['logits'].argmax(dim=1)
        acc = (preds == labels).float().mean()

        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc.item(), images.size(0))

        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.4f}'
        })

    return {'loss': loss_meter.avg, 'acc': acc_meter.avg}


@torch.no_grad()
def validate(model, dataloader, criterion, device, max_batches=None):
    """验证"""
    model.eval()

    loss_meter = AverageMeter('Loss', ':.4f')
    accumulator = MetricsAccumulator()

    pbar = tqdm(dataloader, desc='Validating')

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


def main(args):
    print("=" * 60)
    print("AIGC Detection - 流式数据集训练")
    print("=" * 60)

    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)

    # 日志
    logger = setup_logger('train', log_dir=os.path.join(args.output_dir, 'logs'))

    # 创建流式数据加载器
    print(f"\n数据集: {args.dataset}")
    print(f"训练样本数: {args.train_samples}")
    print(f"验证样本数: {args.val_samples}")

    train_loader = create_streaming_dataloader(
        dataset_name=args.dataset,
        split='train',
        batch_size=args.batch_size,
        img_size=args.img_size,
        max_samples=args.train_samples
    )

    # 注意：该数据集只有 train split，使用 train split 的不同部分作为验证集
    # 通过 skip 参数跳过训练样本
    val_loader = create_streaming_dataloader(
        dataset_name=args.dataset,
        split='train',  # 使用 train split（数据集没有 test split）
        batch_size=args.batch_size,
        img_size=args.img_size,
        max_samples=args.val_samples,
        skip_samples=args.train_samples  # 跳过训练样本
    )

    # 创建模型
    print("\n创建模型...")
    model = AIGCDetectorLite(
        num_classes=2,
        img_size=args.img_size,
        embed_dim=args.embed_dim,
        num_prototypes=4,
        dropout=0.1
    ).to(device)

    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 损失函数
    criterion = AIGCDetectionLoss(
        use_focal=True,
        use_aux=True,
        aux_weight=0.5
    )

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    # 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    # 训练
    print(f"\n开始训练 ({args.epochs} epochs)...")
    best_auc = 0

    for epoch in range(args.epochs):
        # 训练
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            max_batches=args.train_samples // args.batch_size
        )

        # 验证
        val_metrics = validate(
            model, val_loader, criterion, device,
            max_batches=args.val_samples // args.batch_size
        )

        scheduler.step()

        # 日志
        logger.info(
            f"Epoch {epoch}: "
            f"Train Loss={train_metrics['loss']:.4f}, "
            f"Val Loss={val_metrics['loss']:.4f}, "
            f"Val AUC={val_metrics['auc']:.4f}, "
            f"Val Acc={val_metrics['accuracy']:.4f}"
        )

        print(f"\nEpoch {epoch} 结果:")
        print(f"  训练 Loss: {train_metrics['loss']:.4f}")
        print(f"  验证 Loss: {val_metrics['loss']:.4f}")
        print(f"  验证 AUC:  {val_metrics['auc']:.4f}")
        print(f"  验证 Acc:  {val_metrics['accuracy']:.4f}")

        # 保存最佳模型
        is_best = val_metrics['auc'] > best_auc
        if is_best:
            best_auc = val_metrics['auc']
            print(f"  *** 新最佳 AUC: {best_auc:.4f} ***")

        save_checkpoint(
            state={
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_auc': best_auc,
                'metrics': val_metrics
            },
            is_best=is_best,
            checkpoint_dir=os.path.join(args.output_dir, 'checkpoints')
        )

    print("\n" + "=" * 60)
    print(f"训练完成! 最佳 AUC: {best_auc:.4f}")
    print(f"模型保存在: {args.output_dir}/checkpoints/")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train with streaming dataset')

    # 数据集
    parser.add_argument('--dataset', type=str,
                        default='Hemg/AI-Generated-vs-Real-Images-Datasets',
                        help='Hugging Face dataset name')
    parser.add_argument('--train_samples', type=int, default=5000,
                        help='Number of training samples')
    parser.add_argument('--val_samples', type=int, default=1000,
                        help='Number of validation samples')

    # 模型
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--embed_dim', type=int, default=768)

    # 训练
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)

    # 输出
    parser.add_argument('--output_dir', type=str, default='./outputs_streaming')

    args = parser.parse_args()
    main(args)
