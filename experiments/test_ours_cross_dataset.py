#!/usr/bin/env python3
"""Ours模型跨数据集泛化测试

训练: BigGAN + ADM
测试: DiffusionDB (跨数据集泛化)
"""

import sys
import json
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import numpy as np
from tqdm import tqdm

from experiments.run_multisource_sota_comparison import GenImageDataset
from models.detector import AIGCDetectorLite


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, leave=False, desc="Training"):
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
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, desc="Evaluating"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, leave=False, desc=desc):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            outputs = model(images)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    try:
        auc = roc_auc_score(all_labels, all_preds)
        ap = average_precision_score(all_labels, all_preds)
    except ValueError:
        auc, ap = 0.5, 0.5
    acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    return {'auc': auc, 'ap': ap, 'accuracy': acc}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("="*60)
    print("Ours模型跨数据集泛化测试")
    print("训练: BigGAN + ADM | 测试: DiffusionDB")
    print("="*60)

    train_transform, val_transform = get_transforms()

    # 训练数据: BigGAN + ADM
    print("\n[1] 加载训练数据...")
    train_datasets = []
    for name, path in [('BigGAN', 'datasets/genimage_partial/imagenet_ai_0419_biggan'),
                       ('ADM', 'datasets/genimage_partial/imagenet_ai_0508_adm')]:
        try:
            ds = GenImageDataset(path, split='train', transform=train_transform, max_samples=5000)
            train_datasets.append(ds)
            print(f"  {name}: {len(ds)} samples")
        except Exception as e:
            print(f"  {name}: 加载失败 ({e})")

    combined_train = ConcatDataset(train_datasets)
    train_loader = DataLoader(combined_train, batch_size=32, shuffle=True, num_workers=4)
    print(f"  总训练样本: {len(combined_train)}")

    # 测试数据
    print("\n[2] 加载测试数据...")
    test_loaders = {}
    test_configs = [
        ('BigGAN', 'datasets/genimage_partial/imagenet_ai_0419_biggan', False),
        ('BigGAN_JPEG75', 'datasets/genimage_partial/imagenet_ai_0419_biggan', True),
        ('ADM', 'datasets/genimage_partial/imagenet_ai_0508_adm', False),
        ('ADM_JPEG75', 'datasets/genimage_partial/imagenet_ai_0508_adm', True),
        ('DiffusionDB', 'datasets/diffusiondb_test', False),
        ('DiffusionDB_JPEG75', 'datasets/diffusiondb_test', True),
    ]

    for name, path, apply_jpeg in test_configs:
        try:
            ds = GenImageDataset(path, split='val', transform=val_transform,
                               max_samples=1000, apply_jpeg=apply_jpeg, jpeg_quality=75)
            test_loaders[name] = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
            print(f"  {name}: {len(ds)} samples")
        except Exception as e:
            print(f"  {name}: 跳过 ({e})")

    # 创建模型
    print("\n[3] 创建Ours模型...")
    model = AIGCDetectorLite(num_classes=2, img_size=224, embed_dim=768,
                            num_prototypes=8, dropout=0.1)
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  参数量: {params:.2f}M")

    # 训练
    print("\n[4] 训练中...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    start_time = time.time()
    for epoch in range(10):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"  Epoch {epoch+1}/10 - Loss: {loss:.4f}")
    train_time = (time.time() - start_time) / 60
    print(f"  训练时间: {train_time:.1f} 分钟")

    # 测试
    print("\n[5] 测试结果:")
    print("-"*60)
    results = {}
    for name, loader in test_loaders.items():
        metrics = evaluate(model, loader, device, desc=name)
        results[name] = metrics

        # 标记跨数据集测试
        marker = " [跨数据集]" if "DiffusionDB" in name else ""
        print(f"  {name:<20} AUC={metrics['auc']:.4f}  AP={metrics['ap']:.4f}  Acc={metrics['accuracy']:.4f}{marker}")

    # 保存结果
    output = {
        'model': 'Ours (AIGCDetectorLite)',
        'params_M': params,
        'train_time_min': train_time,
        'train_data': ['BigGAN', 'ADM'],
        'results': results
    }

    output_path = 'outputs/multisource_sota_comparison/ours_cross_dataset_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n结果已保存到: {output_path}")

    # 汇总
    print("\n" + "="*60)
    print("跨数据集泛化能力总结")
    print("="*60)
    in_domain = (results.get('BigGAN', {}).get('auc', 0) + results.get('ADM', {}).get('auc', 0)) / 2
    cross_domain = results.get('DiffusionDB', {}).get('auc', 0)
    print(f"  In-domain (BigGAN+ADM) 平均AUC: {in_domain*100:.2f}%")
    print(f"  Cross-domain (DiffusionDB) AUC: {cross_domain*100:.2f}%")
    print(f"  泛化性能下降: {(in_domain - cross_domain)*100:.2f}%")


if __name__ == '__main__':
    main()
