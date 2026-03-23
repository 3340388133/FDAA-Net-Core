#!/usr/bin/env python3
"""
NTIRE 2026 权威数据集测试脚本
测试AIGCDetector在NTIRE 2026 Robust AI-Generated Image Detection数据集上的表现
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image


class NTIRE2026Dataset(Dataset):
    """NTIRE 2026 Robust AI-generated Image Detection Dataset"""

    def __init__(self, shard_dir, shard_nums=None, transform=None, max_samples=None):
        """
        Args:
            shard_dir: 基础目录，shards存储的位置
            shard_nums: 使用的shard编号列表，默认使用所有
            transform: 图像变换
            max_samples: 最大样本数（用于快速测试）
        """
        self.shard_root_dir = shard_dir
        self.transform = transform

        if shard_nums is None:
            self.shard_dirs = [os.path.join(shard_dir, f'shard_{i}') for i in range(6)]
        else:
            self.shard_dirs = [os.path.join(shard_dir, f'shard_{i}') for i in shard_nums]

        self.shard_dirs = [x for x in self.shard_dirs if os.path.isdir(x)]

        if not self.shard_dirs:
            raise ValueError(f"No valid shards found in {shard_dir}")

        # 加载所有标签
        label_dfs = []
        for shard_dir in self.shard_dirs:
            label_path = os.path.join(shard_dir, 'labels.csv')
            if os.path.exists(label_path):
                ldf = pd.read_csv(label_path, index_col=0)
                ldf['shard_name'] = Path(shard_dir).name
                label_dfs.append(ldf)

        self.label_df = pd.concat(label_dfs, ignore_index=True)

        # 限制样本数
        if max_samples and len(self.label_df) > max_samples:
            self.label_df = self.label_df.sample(n=max_samples, random_state=42)
            self.label_df = self.label_df.reset_index(drop=True)

        print(f"Loaded {len(self.shard_dirs)} shards, {len(self.label_df)} images")

        # 统计标签分布
        label_counts = self.label_df['label'].value_counts()
        print(f"  Real images: {label_counts.get(0, 0)}")
        print(f"  AI-generated: {label_counts.get(1, 0)}")

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.label_df.iloc[idx]
        img_path = os.path.join(
            self.shard_root_dir,
            row['shard_name'],
            'images',
            row['image_name']
        )

        image = Image.open(img_path).convert('RGB')
        label = int(row['label'])

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': label}


def get_transforms():
    """获取图像变换"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform


def evaluate_model(model, dataloader, device, desc="Evaluating"):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 计算指标
    try:
        auc = roc_auc_score(all_labels, all_preds)
        ap = average_precision_score(all_labels, all_preds)
    except ValueError:
        auc, ap = 0.5, 0.5

    binary_preds = (all_preds > 0.5).astype(int)
    acc = accuracy_score(all_labels, binary_preds)
    cm = confusion_matrix(all_labels, binary_preds)

    return {
        'auc': auc,
        'ap': ap,
        'accuracy': acc,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
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

    return total_loss / len(dataloader), correct / total


def main():
    print("="*70)
    print("NTIRE 2026 Robust AI-Generated Image Detection 测试")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # 数据集路径
    data_dir = '/home/customer/文档/zc/0125image/datasets/authoritative/NTIRE2026'

    # 检查数据是否解压
    shard_0_dir = os.path.join(data_dir, 'shard_0')
    if not os.path.exists(shard_0_dir):
        print(f"\n错误: 数据未解压。请先运行:")
        print(f"  cd {data_dir}")
        print(f"  unzip -o shard_*.zip")
        return

    transform = get_transforms()

    # 加载数据集 - 使用shard_0和shard_1作为训练，shard_4作为测试
    print("\n[1] 加载数据集...")
    try:
        train_dataset = NTIRE2026Dataset(
            data_dir,
            shard_nums=[0, 1],
            transform=transform,
            max_samples=10000  # 快速测试用
        )
        test_dataset = NTIRE2026Dataset(
            data_dir,
            shard_nums=[4],
            transform=transform,
            max_samples=5000
        )
    except Exception as e:
        print(f"加载数据失败: {e}")
        return

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 加载模型
    print("\n[2] 初始化模型...")
    from models.detector import AIGCDetectorLite

    model = AIGCDetectorLite(
        num_classes=2,
        img_size=224,
        embed_dim=768,
        num_prototypes=4
    ).to(device)

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数: {total_params:,}")

    # 训练设置
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # 训练
    print("\n[3] 开始训练...")
    num_epochs = 5
    best_auc = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # 验证
        results = evaluate_model(model, test_loader, device, desc="Validating")
        print(f"  Test AUC: {results['auc']:.4f}, AP: {results['ap']:.4f}, Acc: {results['accuracy']:.4f}")

        if results['auc'] > best_auc:
            best_auc = results['auc']
            torch.save(model.state_dict(), 'checkpoints/best_ntire2026.pth')
            print(f"  -> Best model saved!")

        scheduler.step()

    # 最终评估
    print("\n" + "="*70)
    print("最终测试结果")
    print("="*70)

    final_results = evaluate_model(model, test_loader, device, desc="Final Test")
    print(f"\nAUC:      {final_results['auc']:.4f}")
    print(f"AP:       {final_results['ap']:.4f}")
    print(f"Accuracy: {final_results['accuracy']:.4f}")
    print(f"\nConfusion Matrix:")
    print(final_results['confusion_matrix'])


if __name__ == "__main__":
    main()
