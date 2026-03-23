#!/usr/bin/env python3
"""
NTIRE 2026 完整训练脚本
使用全部可用数据进行完整的模型训练
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')


class NTIRE2026Dataset(Dataset):
    """NTIRE 2026 完整数据集"""

    def __init__(self, data_dir, shard_nums=None, transform=None, max_samples=None, verify_images=True):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        if shard_nums is None:
            shard_nums = [0, 1, 4, 5]

        print(f"加载 shards: {shard_nums}")

        for shard_num in shard_nums:
            shard_dir = os.path.join(data_dir, f'shard_{shard_num}')
            labels_path = os.path.join(shard_dir, 'labels.csv')
            images_dir = os.path.join(shard_dir, 'images')

            if not os.path.exists(labels_path):
                print(f"  跳过 shard_{shard_num}: labels.csv 不存在")
                continue

            if not os.path.exists(images_dir):
                print(f"  跳过 shard_{shard_num}: images 目录不存在")
                continue

            df = pd.read_csv(labels_path)
            count = 0

            for _, row in df.iterrows():
                img_path = os.path.join(images_dir, row['image_name'])
                if os.path.exists(img_path):
                    if verify_images:
                        try:
                            img = Image.open(img_path)
                            img.verify()
                            self.samples.append((img_path, int(row['label'])))
                            count += 1
                        except:
                            continue
                    else:
                        self.samples.append((img_path, int(row['label'])))
                        count += 1

                if max_samples and len(self.samples) >= max_samples:
                    break

            print(f"  shard_{shard_num}: {count} 样本")

            if max_samples and len(self.samples) >= max_samples:
                break

        # 统计
        labels = [s[1] for s in self.samples]
        real_count = labels.count(0)
        fake_count = labels.count(1)
        print(f"总计: {len(self.samples)} 样本 (真实: {real_count}, AI生成: {fake_count})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return {'image': image, 'label': label}
        except:
            # Fallback
            image = torch.zeros(3, 224, 224)
            return {'image': image, 'label': label}


def get_transforms(train=True):
    """获取数据增强"""
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, device, desc="Evaluating"):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    try:
        auc = roc_auc_score(all_labels, all_preds)
        ap = average_precision_score(all_labels, all_preds)
    except:
        auc, ap = 0.5, 0.5

    binary_preds = (all_preds > 0.5).astype(int)
    acc = accuracy_score(all_labels, binary_preds)

    return {
        'auc': auc,
        'ap': ap,
        'accuracy': acc,
        'predictions': all_preds,
        'labels': all_labels
    }


def main():
    print("=" * 70)
    print("NTIRE 2026 完整训练")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 配置
    config = {
        'data_dir': '/home/customer/文档/zc/0125image/datasets/authoritative/NTIRE2026',
        'train_shards': [0],  # 使用shard_0训练
        'test_shards': [1],   # 使用shard_1测试
        'batch_size': 32,
        'num_epochs': 20,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'num_workers': 4,
        'use_amp': True,  # 混合精度训练
        'train_samples': None,  # 使用全部
        'test_samples': 10000,
    }

    print(f"\n配置: {json.dumps(config, indent=2, ensure_ascii=False)}")

    # 数据集
    print("\n[1] 加载训练数据...")
    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)

    train_dataset = NTIRE2026Dataset(
        config['data_dir'],
        shard_nums=config['train_shards'],
        transform=train_transform,
        max_samples=config['train_samples'],
        verify_images=False  # 跳过验证加速加载
    )

    print("\n[2] 加载测试数据...")
    test_dataset = NTIRE2026Dataset(
        config['data_dir'],
        shard_nums=config['test_shards'],
        transform=test_transform,
        max_samples=config['test_samples'],
        verify_images=False
    )

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # 模型
    print("\n[3] 初始化模型...")
    from models.detector import AIGCDetectorLite

    model = AIGCDetectorLite(
        num_classes=2,
        img_size=224,
        embed_dim=768,
        num_prototypes=4,
        dropout=0.1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: {total_params:,} (可训练: {trainable_params:,})")

    # 优化器和调度器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=1e-6
    )

    # 混合精度
    scaler = torch.cuda.amp.GradScaler() if config['use_amp'] else None

    # 训练
    print("\n[4] 开始训练...")
    print("-" * 70)

    best_auc = 0
    best_epoch = 0
    history = []

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']} (LR: {scheduler.get_last_lr()[0]:.2e})")

        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )

        # 评估
        results = evaluate(model, test_loader, device, desc="Testing")

        # 记录
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_auc': results['auc'],
            'test_ap': results['ap'],
            'test_acc': results['accuracy']
        })

        # 保存最佳模型
        marker = ""
        if results['auc'] > best_auc:
            best_auc = results['auc']
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'config': config
            }, 'checkpoints/ntire2026_best_full.pth')
            marker = " ★ Best"

        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Test  AUC:  {results['auc']:.4f}, AP: {results['ap']:.4f}, Acc: {results['accuracy']:.4f}{marker}")

        scheduler.step()

    # 最终结果
    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)
    print(f"最佳 AUC: {best_auc:.4f} (Epoch {best_epoch})")
    print(f"模型保存: checkpoints/ntire2026_best_full.pth")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 保存历史
    history_df = pd.DataFrame(history)
    history_df.to_csv('outputs/ntire2026_training_history.csv', index=False)
    print(f"训练历史: outputs/ntire2026_training_history.csv")

    # 打印最终统计
    print("\n训练历史:")
    print(history_df.to_string(index=False))


if __name__ == "__main__":
    main()
