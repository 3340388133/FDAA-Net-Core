#!/usr/bin/env python3
"""在DiffusionDB数据集上测试已有模型"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from tqdm import tqdm

from experiments.run_multisource_sota_comparison import GenImageDataset
from models.sota_methods import create_sota_model
from models.detector import AIGCDetectorLite

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, leave=False):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            outputs = model(images)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    
    auc = roc_auc_score(all_labels, all_preds)
    ap = average_precision_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    return {'auc': auc, 'ap': ap, 'accuracy': acc}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 加载数据
    data_root = 'datasets/diffusiondb_test'
    transform = get_transform()
    
    # 测试数据集
    test_dataset = GenImageDataset(data_root, split='val', transform=transform, max_samples=1000)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    print(f"测试样本数: {len(test_dataset)}")
    
    # JPEG压缩版本
    test_dataset_jpeg = GenImageDataset(data_root, split='val', transform=transform, 
                                         max_samples=1000, apply_jpeg=True, jpeg_quality=75)
    test_loader_jpeg = DataLoader(test_dataset_jpeg, batch_size=32, shuffle=False, num_workers=4)
    
    # 测试所有方法
    methods = ['ours', 'univfd', 'dire', 'lare2', 'drct', 'cnndetection', 'f3net', 'freqnet', 'c2pclip']
    results = {}
    
    for method in methods:
        print(f"\n测试 {method}...")
        try:
            if method == 'ours':
                model = AIGCDetectorLite(num_classes=2, img_size=224, embed_dim=768, num_prototypes=8, dropout=0.1)
            else:
                model = create_sota_model(method)
            model = model.to(device)
            
            # 原始测试
            metrics = evaluate(model, test_loader, device)
            metrics_jpeg = evaluate(model, test_loader_jpeg, device)
            
            results[method] = {
                'DiffusionDB': metrics,
                'DiffusionDB_JPEG75': metrics_jpeg
            }
            print(f"  原始 AUC: {metrics['auc']:.4f}, JPEG AUC: {metrics_jpeg['auc']:.4f}")
            
        except Exception as e:
            print(f"  错误: {e}")
    
    # 保存结果
    output_path = 'outputs/multisource_sota_comparison/diffusiondb_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到: {output_path}")

if __name__ == '__main__':
    main()
