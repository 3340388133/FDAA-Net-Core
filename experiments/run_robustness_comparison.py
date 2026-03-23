#!/usr/bin/env python3
"""
鲁棒性对比实验 - 有区分度的SOTA对比

核心思路：在干净数据上训练，在各种扰动下测试
这样能真正体现不同方法的鲁棒性差异

扰动类型:
- JPEG压缩 (Q=50, 70, 90)
- 高斯模糊 (σ=1, 2, 3)
- 高斯噪声 (σ=0.05, 0.1, 0.15)
- 下采样 (0.5x, 0.25x)
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import functional as TF
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from tqdm import tqdm
from PIL import Image
import io

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.sota_methods import create_sota_model, list_sota_methods
from models.detector import AIGCDetectorLite
from data.genimage_dataset import GenImageDataset


class RobustnessTransform:
    """鲁棒性测试变换"""

    def __init__(self, perturbation_type, perturbation_level, img_size=224):
        self.perturbation_type = perturbation_type
        self.perturbation_level = perturbation_level
        self.img_size = img_size
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __call__(self, img):
        # 先resize
        img = TF.resize(img, (self.img_size, self.img_size))

        # 应用扰动
        if self.perturbation_type == 'none':
            pass
        elif self.perturbation_type == 'jpeg':
            img = self._jpeg_compress(img, quality=self.perturbation_level)
        elif self.perturbation_type == 'blur':
            img = TF.gaussian_blur(img, kernel_size=int(self.perturbation_level * 4) * 2 + 1,
                                   sigma=self.perturbation_level)
        elif self.perturbation_type == 'noise':
            img = self._add_noise(img, sigma=self.perturbation_level)
        elif self.perturbation_type == 'resize':
            # 先缩小再放大
            small_size = int(self.img_size * self.perturbation_level)
            img = TF.resize(img, (small_size, small_size))
            img = TF.resize(img, (self.img_size, self.img_size))

        # 转tensor并归一化
        img = TF.to_tensor(img)
        img = self.normalize(img)
        return img

    def _jpeg_compress(self, img, quality):
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=int(quality))
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')

    def _add_noise(self, img, sigma):
        img_array = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(0, sigma, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 1)
        return Image.fromarray((noisy * 255).astype(np.uint8))


def get_train_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def train_model(model, train_loader, epochs, device, method_name):
    """快速训练模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"{method_name} Epoch {epoch+1}/{epochs}", leave=False)
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        print(f"  Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")

    return model


def evaluate_with_perturbation(model, dataset, perturbation_type, perturbation_level,
                                device, batch_size=32):
    """在特定扰动下评估模型"""
    transform = RobustnessTransform(perturbation_type, perturbation_level)

    # 创建新的数据集副本并应用扰动变换
    class PerturbedDataset:
        def __init__(self, base_dataset, transform):
            self.base_dataset = base_dataset
            self.transform = transform
            self.samples = base_dataset.samples

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return {'image': img, 'label': label}

    perturbed_dataset = PerturbedDataset(dataset, transform)
    loader = DataLoader(perturbed_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            labels = batch['label']

            outputs = model(images)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            all_preds.extend(probs)
            all_labels.extend(labels.numpy())

    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))

    return {'auc': auc, 'accuracy': acc}


def main():
    parser = argparse.ArgumentParser(description='鲁棒性对比实验')
    parser.add_argument('--data_root', type=str,
                        default='datasets/genimage_partial/imagenet_ai_0419_biggan')
    parser.add_argument('--train_samples', type=int, default=10000)
    parser.add_argument('--test_samples', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='outputs/robustness_comparison')
    parser.add_argument('--methods', nargs='+',
                        default=['ours', 'cnndetection', 'f3net', 'univfd', 'freqnet'])

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 定义扰动测试配置
    perturbations = [
        ('none', 0, '无扰动'),
        ('jpeg', 50, 'JPEG Q=50'),
        ('jpeg', 70, 'JPEG Q=70'),
        ('blur', 1.0, '模糊 σ=1'),
        ('blur', 2.0, '模糊 σ=2'),
        ('noise', 0.05, '噪声 5%'),
        ('noise', 0.10, '噪声 10%'),
        ('resize', 0.5, '缩放 0.5x'),
        ('resize', 0.25, '缩放 0.25x'),
    ]

    # 加载训练数据
    print(f"\n加载数据集: {args.data_root}")
    train_transform = get_train_transform()

    train_dataset = GenImageDataset(
        root_dir=args.data_root,
        split='train',
        transform=train_transform,
        max_samples=args.train_samples // 2
    )

    # 测试数据集（不带变换，后面动态添加）
    test_dataset_base = GenImageDataset(
        root_dir=args.data_root,
        split='val',
        transform=None,  # 不设置transform
        max_samples=args.test_samples // 2
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    print(f"训练样本: {len(train_dataset)}, 测试样本: {len(test_dataset_base)}")

    # 存储结果
    results = {
        'args': vars(args),
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'perturbations': [(p[0], p[1], p[2]) for p in perturbations],
        'methods': {}
    }

    # 训练和测试每个方法
    for method_name in args.methods:
        print(f"\n{'='*60}")
        print(f"方法: {method_name.upper()}")
        print(f"{'='*60}")

        try:
            # 创建模型
            if method_name == 'ours':
                model = AIGCDetectorLite(num_classes=2, img_size=224, embed_dim=768)
            else:
                model = create_sota_model(method_name, num_classes=2)

            model = model.to(device)

            # 训练
            print(f"训练中...")
            model = train_model(model, train_loader, args.epochs, device, method_name)

            # 在各种扰动下测试
            method_results = {}
            print(f"\n测试鲁棒性:")

            for p_type, p_level, p_name in perturbations:
                metrics = evaluate_with_perturbation(
                    model, test_dataset_base, p_type, p_level, device, args.batch_size
                )
                method_results[p_name] = metrics
                print(f"  {p_name}: AUC={metrics['auc']:.4f}, Acc={metrics['accuracy']:.4f}")

            results['methods'][method_name] = method_results

            # 清理
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
            results['methods'][method_name] = {'error': str(e)}

    # 保存结果
    result_file = output_dir / f"robustness_comparison_{results['timestamp']}.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {result_file}")

    # 打印汇总表格
    print("\n" + "="*100)
    print("鲁棒性对比汇总 (AUC)")
    print("="*100)

    # 表头
    header = f"{'方法':<15}"
    for _, _, p_name in perturbations:
        header += f"{p_name:>12}"
    print(header)
    print("-"*100)

    # 每个方法的结果
    for method_name in args.methods:
        if method_name in results['methods'] and 'error' not in results['methods'][method_name]:
            row = f"{method_name:<15}"
            for _, _, p_name in perturbations:
                auc = results['methods'][method_name][p_name]['auc']
                row += f"{auc:>12.4f}"
            print(row)

    print("="*100)

    # 计算平均鲁棒性下降
    print("\n鲁棒性下降分析 (相对于无扰动):")
    for method_name in args.methods:
        if method_name in results['methods'] and 'error' not in results['methods'][method_name]:
            base_auc = results['methods'][method_name]['无扰动']['auc']
            drops = []
            for _, _, p_name in perturbations[1:]:  # 跳过无扰动
                auc = results['methods'][method_name][p_name]['auc']
                drop = (base_auc - auc) / base_auc * 100
                drops.append(drop)
            avg_drop = np.mean(drops)
            print(f"  {method_name}: 平均下降 {avg_drop:.2f}%")


if __name__ == '__main__':
    main()
