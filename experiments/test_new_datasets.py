#!/usr/bin/env python3
"""在新数据集上测试SOTA方法（训练+测试）"""

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
from models.sota_methods import create_sota_model
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

def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, leave=False, desc="Evaluating"):
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
        # 如果只有一个类，返回0.5
        auc = 0.5
        ap = 0.5
    acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    return {'auc': auc, 'ap': ap, 'accuracy': acc}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    train_transform, val_transform = get_transforms()
    
    # 训练数据: BigGAN + ADM
    print("\n加载训练数据 (BigGAN + ADM)...")
    train_datasets = []

    train_paths = [
        'datasets/genimage_partial/imagenet_ai_0419_biggan',
        'datasets/genimage_partial/imagenet_ai_0508_adm'
    ]

    for path in train_paths:
        try:
            ds = GenImageDataset(path, split='train', transform=train_transform, max_samples=5000)
            train_datasets.append(ds)
            print(f"  {path}: {len(ds)} samples")
        except Exception as e:
            print(f"  {path}: 跳过 ({e})")

    if not train_datasets:
        print("错误：没有可用的训练数据！")
        return

    combined_train = ConcatDataset(train_datasets)
    train_loader = DataLoader(combined_train, batch_size=32, shuffle=True, num_workers=4)
    
    # 测试数据集配置
    # 训练：BigGAN + ADM | 测试：DiffusionDB (跨域泛化) + BigGAN/ADM (in-domain)
    test_configs = {
        # In-domain 测试 (BigGAN, ADM)
        'BigGAN': ('datasets/genimage_partial/imagenet_ai_0419_biggan', False),
        'BigGAN_JPEG75': ('datasets/genimage_partial/imagenet_ai_0419_biggan', True),
        'ADM': ('datasets/genimage_partial/imagenet_ai_0508_adm', False),
        'ADM_JPEG75': ('datasets/genimage_partial/imagenet_ai_0508_adm', True),
        # Cross-domain 泛化测试 (DiffusionDB)
        'DiffusionDB': ('datasets/diffusiondb_test', False),
        'DiffusionDB_JPEG75': ('datasets/diffusiondb_test', True),
    }
    
    test_loaders = {}
    for name, (path, apply_jpeg) in test_configs.items():
        try:
            ds = GenImageDataset(path, split='val', transform=val_transform, 
                               max_samples=1000, apply_jpeg=apply_jpeg, jpeg_quality=75)
            test_loaders[name] = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
            print(f"  {name}: {len(ds)} samples")
        except Exception as e:
            print(f"  {name}: 跳过 ({e})")
    
    # 测试所有方法
    methods = ['ours', 'univfd', 'dire', 'lare2', 'drct', 'cnndetection', 'f3net', 'freqnet', 'c2pclip']
    all_results = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"方法: {method.upper()}")
        print(f"{'='*60}")
        
        try:
            # 创建模型
            if method == 'ours':
                model = AIGCDetectorLite(num_classes=2, img_size=224, embed_dim=768, 
                                        num_prototypes=8, dropout=0.1)
            else:
                model = create_sota_model(method)
            model = model.to(device)
            
            params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"参数量: {params:.2f}M")
            
            # 快速训练 (10 epochs)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            
            start_time = time.time()
            for epoch in range(10):
                train_epoch(model, train_loader, criterion, optimizer, device)
                print(f"  Epoch {epoch+1}/10 完成")
            train_time = (time.time() - start_time) / 60
            
            # 测试所有数据集
            results = {'params_M': params, 'train_time_min': train_time}
            for test_name, loader in test_loaders.items():
                metrics = evaluate(model, loader, device)
                results[test_name] = metrics
                print(f"  {test_name}: AUC={metrics['auc']:.4f}")
            
            all_results[method] = results
            
            # 保存中间结果
            with open(f'outputs/multisource_sota_comparison/result_{method}_new.json', 'w') as f:
                json.dump(results, f, indent=2)
                
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存最终结果
    with open('outputs/multisource_sota_comparison/biggan_adm_train_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # 打印汇总
    print("\n" + "="*100)
    print("结果汇总 (训练: BigGAN + ADM)")
    print("="*100)
    print(f"{'Method':<15} {'BigGAN':>10} {'BigGAN_J75':>12} {'ADM':>10} {'ADM_J75':>10} {'DiffDB':>10} {'DiffDB_J75':>12}")
    print("-"*100)
    for method, res in all_results.items():
        bg = res.get('BigGAN', {}).get('auc', 0) * 100
        bg_j = res.get('BigGAN_JPEG75', {}).get('auc', 0) * 100
        adm = res.get('ADM', {}).get('auc', 0) * 100
        adm_j = res.get('ADM_JPEG75', {}).get('auc', 0) * 100
        db = res.get('DiffusionDB', {}).get('auc', 0) * 100
        db_j = res.get('DiffusionDB_JPEG75', {}).get('auc', 0) * 100
        print(f"{method:<15} {bg:>9.2f}% {bg_j:>11.2f}% {adm:>9.2f}% {adm_j:>9.2f}% {db:>9.2f}% {db_j:>11.2f}%")

if __name__ == '__main__':
    main()
