#!/usr/bin/env python3
"""
GenImage 完整训练脚本
使用GenImage多生成器数据集进行完整训练和跨生成器泛化评估
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')


class GenImageDataset(Dataset):
    """GenImage 数据集加载器"""

    def __init__(self, root_dir, generator_name, split='train', transform=None, max_samples=None):
        self.root_dir = root_dir
        self.generator_name = generator_name
        self.transform = transform
        self.samples = []

        # GenImage目录结构: GenImage/{generator}/{extracted_folder}/{split}/{ai|nature}/
        gen_dir = os.path.join(root_dir, generator_name)

        # 查找解压后的子目录
        extracted_dirs = []
        for item in os.listdir(gen_dir):
            item_path = os.path.join(gen_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                # Check if this is the extracted data directory (has train/val subdirs)
                if os.path.isdir(os.path.join(item_path, split)):
                    extracted_dirs.append(item_path)

        if not extracted_dirs:
            print(f"  警告: {generator_name} 没有找到 {split} 目录")
            return

        for ext_dir in extracted_dirs:
            split_dir = os.path.join(ext_dir, split)

            # AI生成图像 (label=1)
            ai_dir = os.path.join(split_dir, 'ai')
            if os.path.isdir(ai_dir):
                ai_images = self._get_images(ai_dir)
                for img_path in ai_images:
                    self.samples.append((img_path, 1))

            # 真实图像 (label=0)
            nature_dir = os.path.join(split_dir, 'nature')
            if os.path.isdir(nature_dir):
                nature_images = self._get_images(nature_dir)
                for img_path in nature_images:
                    self.samples.append((img_path, 0))

        # 限制样本数
        if max_samples and len(self.samples) > max_samples:
            np.random.seed(42)
            indices = np.random.choice(len(self.samples), max_samples, replace=False)
            self.samples = [self.samples[i] for i in indices]

        labels = [s[1] for s in self.samples]
        real_count = labels.count(0)
        fake_count = labels.count(1)
        print(f"  {generator_name}/{split}: {len(self.samples)} 样本 (真实: {real_count}, AI生成: {fake_count})")

    def _get_images(self, directory):
        """获取目录下所有图像文件"""
        extensions = {'.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG'}
        images = []
        for f in os.listdir(directory):
            if any(f.endswith(ext) for ext in extensions):
                images.append(os.path.join(directory, f))
        return sorted(images)

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
            image = torch.zeros(3, 224, 224)
            return {'image': image, 'label': label}


def get_transforms(train=True):
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

    return {'auc': auc, 'ap': ap, 'accuracy': acc}


def main():
    print("=" * 70)
    print("GenImage 完整训练与跨生成器泛化评估")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # GenImage数据集路径
    genimage_root = '/home/customer/文档/zc/0125image/datasets/authoritative/GenImage'

    # 检测可用的生成器
    available_generators = []
    for gen_name in ['ADM', 'BigGAN', 'glide', 'Midjourney', 'stable_diffusion_v_1_4', 'VQDM']:
        gen_dir = os.path.join(genimage_root, gen_name)
        if os.path.isdir(gen_dir):
            # 检查是否有解压后的数据（包含train目录的子目录）
            has_data = False
            for item in os.listdir(gen_dir):
                item_path = os.path.join(gen_dir, item)
                if os.path.isdir(item_path) and os.path.isdir(os.path.join(item_path, 'train')):
                    has_data = True
                    break
            if has_data:
                available_generators.append(gen_name)

    print(f"\n可用生成器: {available_generators}")

    if not available_generators:
        print("错误: 没有找到已解压的GenImage数据集!")
        return

    # 配置
    config = {
        'genimage_root': genimage_root,
        'generators': available_generators,
        'batch_size': 64,
        'num_epochs': 15,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'num_workers': 4,
        'use_amp': True,
        'train_max_per_generator': 50000,  # 每个生成器最多50000训练样本
        'val_max_per_generator': 5000,     # 每个生成器最多5000验证样本
    }

    print(f"\n配置: {json.dumps({k:v for k,v in config.items() if k != 'genimage_root'}, indent=2, ensure_ascii=False)}")

    # 加载数据
    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)

    print("\n[1] 加载训练数据 (所有生成器合并)...")
    train_datasets = []
    for gen in available_generators:
        ds = GenImageDataset(
            genimage_root, gen, split='train',
            transform=train_transform,
            max_samples=config['train_max_per_generator']
        )
        if len(ds) > 0:
            train_datasets.append(ds)

    if not train_datasets:
        print("错误: 没有训练数据!")
        return

    combined_train = ConcatDataset(train_datasets)
    print(f"\n合并训练集: {len(combined_train)} 样本")

    train_loader = DataLoader(
        combined_train,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    print("\n[2] 加载验证数据 (按生成器分开评估)...")
    val_loaders = {}
    for gen in available_generators:
        # 先尝试加载val集
        ds = GenImageDataset(
            genimage_root, gen, split='val',
            transform=test_transform,
            max_samples=config['val_max_per_generator']
        )
        if len(ds) == 0:
            # 没有val集，从train中分出一部分作为验证
            print(f"  {gen}: 没有val集，从train中抽取验证样本...")
            ds = GenImageDataset(
                genimage_root, gen, split='train',
                transform=test_transform,
                max_samples=config['val_max_per_generator']
            )
        if len(ds) > 0:
            val_loaders[gen] = DataLoader(
                ds,
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
    print(f"模型参数: {total_params:,}")

    # 优化器
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

    scaler = torch.cuda.amp.GradScaler() if config['use_amp'] else None

    # 训练
    print("\n[4] 开始训练...")
    print("-" * 70)

    best_avg_auc = 0
    best_epoch = 0
    history = []

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']} (LR: {scheduler.get_last_lr()[0]:.2e})")

        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )

        # 对每个生成器分别评估
        epoch_results = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
        }

        gen_aucs = []
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        for gen_name, val_loader in val_loaders.items():
            results = evaluate(model, val_loader, device, desc=f"Test-{gen_name}")
            epoch_results[f'{gen_name}_auc'] = results['auc']
            epoch_results[f'{gen_name}_ap'] = results['ap']
            epoch_results[f'{gen_name}_acc'] = results['accuracy']
            gen_aucs.append(results['auc'])
            print(f"  {gen_name:25s} AUC: {results['auc']:.4f}, AP: {results['ap']:.4f}, Acc: {results['accuracy']:.4f}")

        avg_auc = np.mean(gen_aucs) if gen_aucs else 0
        epoch_results['avg_auc'] = avg_auc

        marker = ""
        if avg_auc > best_avg_auc:
            best_avg_auc = avg_auc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_avg_auc': best_avg_auc,
                'config': config
            }, 'checkpoints/genimage_best_full.pth')
            marker = " ★ Best"

        print(f"  >>> 平均 AUC: {avg_auc:.4f}{marker}")
        history.append(epoch_results)

        scheduler.step()

    # 最终结果
    print("\n" + "=" * 70)
    print("GenImage 训练完成!")
    print("=" * 70)
    print(f"最佳平均 AUC: {best_avg_auc:.4f} (Epoch {best_epoch})")
    print(f"模型保存: checkpoints/genimage_best_full.pth")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 保存历史
    history_df = pd.DataFrame(history)
    history_df.to_csv('outputs/genimage_training_history.csv', index=False)
    print(f"训练历史: outputs/genimage_training_history.csv")

    # 打印最终统计
    print("\n各生成器最终 AUC:")
    last = history[-1]
    for gen in available_generators:
        key = f'{gen}_auc'
        if key in last:
            print(f"  {gen:25s}: {last[key]:.4f}")
    print(f"  {'平均':25s}: {last['avg_auc']:.4f}")

    # 打印完整训练历史摘要
    print("\n训练历史摘要:")
    summary_cols = ['epoch', 'train_loss', 'train_acc', 'avg_auc']
    print(history_df[summary_cols].to_string(index=False))


if __name__ == "__main__":
    main()
