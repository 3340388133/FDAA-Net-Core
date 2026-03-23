"""
FDAA-Net V2 完整实验脚本

功能:
1. 多源训练 (BigGAN + ADM + GLIDE + VQDM)
2. 强增强 (JPEG, Blur, ColorJitter)
3. 双GPU DataParallel
4. Warmup(5 epochs) + CosineAnnealing(30 epochs)
5. 批量训练所有SOTA方法 (公平对比)
6. 域内 + 跨域评估 (CIFAKE, DiffForensics, NTIRE2026)
7. 消融实验 (baseline / +FDAA / +MGFP / Full)
8. 自动生成结果 JSON 和 Markdown 报告

使用方法:
    # 完整实验
    python experiments/run_improved_experiment.py --mode all

    # 仅训练我们的模型
    python experiments/run_improved_experiment.py --mode train_ours

    # 仅训练SOTA对比
    python experiments/run_improved_experiment.py --mode train_sota

    # 仅评估
    python experiments/run_improved_experiment.py --mode evaluate

    # 仅消融
    python experiments/run_improved_experiment.py --mode ablation
"""

import os
import sys
import json
import time
import argparse
import datetime
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.detector import AIGCDetectorV2, AIGCDetectorLite
from models.modules.fdaa import FDAAv2
from models.modules.mgfp import MGFPv2
from models.losses.losses import AIGCDetectionLoss
from models.sota_methods import create_sota_model, SOTA_METHODS
from data.multi_source_dataset import create_multi_source_dataloader, MultiSourceGenImageDataset, get_multi_source_transforms
from data.genimage_dataset import GenImageDataset, get_genimage_transforms
from utils.metrics import compute_metrics, MetricsAccumulator


# =============================================================================
# 配置
# =============================================================================

DEFAULT_CONFIG = {
    # 数据
    'genimage_root': '/home/customer/文档/zc/0125image/datasets/authoritative/GenImage',
    'sources': ['biggan', 'adm', 'glide', 'vqdm'],
    'max_samples_per_source': 10000,

    # 训练
    'batch_size': 64,
    'lr': 5e-4,
    'weight_decay': 1e-4,
    'epochs': 20,
    'warmup_epochs': 5,
    'num_workers': 8,
    'img_size': 224,
    'embed_dim': 1024,  # CLIP ViT-L/14 内部维度

    # 模型
    'backbone': 'ViT-L/14',
    'use_hierarchical': True,
    'dropout': 0.1,

    # 损失
    'use_focal': True,
    'use_contrastive': True,
    'contrastive_weight': 0.3,     # 从0.1提升到0.3
    'aux_weight': 0.3,             # 从0.5降低到0.3
    'label_smoothing': 0.05,       # 从0.1降低到0.05

    # 评估数据集
    'eval_datasets': {
        'cifake': '/home/customer/文档/zc/0125image/datasets/authoritative/CIFAKE',
        'diffusion_forensics': '/home/customer/文档/zc/0125image/datasets/authoritative/DiffusionForensics',
    },

    # 输出
    'output_dir': str(PROJECT_ROOT / 'outputs' / 'v2_improved'),

    # SOTA方法列表 (核心对比方法)
    'sota_methods': ['cnndetection', 'f3net', 'univfd', 'freqnet', 'npr', 'spec'],
    'sota_epochs': 20,  # SOTA方法训练epoch数 — 与Ours相同！
}


# =============================================================================
# 工具函数
# =============================================================================

def get_device():
    """获取可用设备"""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"[Device] {num_gpus} GPU(s) available:")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return 'cuda', num_gpus
    else:
        print("[Device] Using CPU")
        return 'cpu', 0


def setup_output_dir(output_dir):
    """创建输出目录结构"""
    os.makedirs(output_dir, exist_ok=True)
    for subdir in ['checkpoints', 'logs', 'results', 'reports']:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    return output_dir


def save_json(data, filepath):
    """保存JSON"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"[Save] {filepath}")


def format_metrics(metrics):
    """格式化指标显示"""
    parts = []
    for k in ['auc', 'ap', 'accuracy', 'eer']:
        if k in metrics:
            if k == 'eer':
                parts.append(f"{k.upper()}: {metrics[k]*100:.2f}%")
            else:
                parts.append(f"{k.upper()}: {metrics[k]*100:.2f}%")
    return ' | '.join(parts)


# =============================================================================
# 训练核心
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, warmup_epochs=5, base_lr=3e-4):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch}')

    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        # Warmup learning rate
        if epoch < warmup_epochs:
            warmup_factor = (epoch * len(loader) + batch_idx) / (warmup_epochs * len(loader))
            lr = base_lr * warmup_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        optimizer.zero_grad()

        if scaler is not None:
            with autocast('cuda'):
                # 安全调用：先尝试 return_features，SOTA方法有 **kwargs 会忽略
                outputs = model(images, return_features=True)
                features = outputs.get('features', None)
                loss_dict = criterion(outputs, labels, features=features)
                loss = loss_dict['total_loss']

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images, return_features=True)
            features = outputs.get('features', None)
            loss_dict = criterion(outputs, labels, features=features)
            loss = loss_dict['total_loss']

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs['logits'].argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total*100:.1f}%',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

    return {
        'loss': total_loss / total,
        'accuracy': correct / total
    }


@torch.no_grad()
def evaluate_model(model, loader, device, desc='Eval'):
    """评估模型"""
    model.eval()
    accumulator = MetricsAccumulator()

    for batch in tqdm(loader, desc=desc):
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label']

        if device == 'cuda':
            with autocast('cuda'):
                outputs = model(images)
        else:
            outputs = model(images)

        probs = torch.softmax(outputs['logits'].float(), dim=1)[:, 1]
        accumulator.update(probs.cpu(), labels)

    metrics = accumulator.compute()
    return metrics


def train_model(
    model, train_loader, val_loader, config, model_name='model',
    output_dir='./outputs', device='cuda', num_gpus=1,
    resume_checkpoint=None
):
    """
    完整训练流程

    返回: best_metrics, model
    """
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")

    # 断点续训: 加载权重
    start_epoch = 0
    best_auc = 0
    best_metrics = {}
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"[Resume] Loading checkpoint: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location='cpu')
        raw_model = model
        raw_model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        saved_metrics = ckpt.get('metrics', {})
        best_auc = saved_metrics.get('auc', 0)
        best_metrics = saved_metrics
        print(f"[Resume] Resuming from epoch {start_epoch}, best AUC so far: {best_auc*100:.2f}%")

    # DataParallel
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # 损失函数
    criterion = AIGCDetectionLoss(
        use_focal=config.get('use_focal', True),
        use_contrastive=config.get('use_contrastive', True),
        contrastive_weight=config.get('contrastive_weight', 0.3),
        use_aux=config.get('use_hierarchical', False),
        aux_weight=config.get('aux_weight', 0.3),
        label_smoothing=config.get('label_smoothing', 0.05),
    )

    # 优化器 — 分层学习率（对有 fdaa/mgfp/classifier 属性的模型）
    raw_model = model.module if hasattr(model, 'module') else model
    if hasattr(raw_model, 'fdaa') and hasattr(raw_model, 'mgfp'):
        param_groups = [
            {'params': raw_model.fdaa.parameters(), 'lr': config['lr'] * 2},      # 频率分支：较高lr
            {'params': raw_model.mgfp.parameters(), 'lr': config['lr']},           # 融合模块：基础lr
            {'params': raw_model.classifier.parameters(), 'lr': config['lr']},     # 分类头：基础lr
        ]
        if hasattr(raw_model, 'aux_classifier'):
            param_groups.append({'params': raw_model.aux_classifier.parameters(), 'lr': config['lr']})
        optimizer = optim.AdamW(param_groups, weight_decay=config['weight_decay'])
    else:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'] - config.get('warmup_epochs', 5),
        eta_min=1e-6
    )

    # 恢复scheduler状态到正确位置
    if start_epoch > config.get('warmup_epochs', 5):
        for _ in range(start_epoch - config.get('warmup_epochs', 5)):
            scheduler.step()

    # AMP
    scaler = GradScaler('cuda') if device == 'cuda' else None

    if start_epoch >= config['epochs']:
        print(f"[Resume] Training already completed ({start_epoch}/{config['epochs']} epochs)")
        return best_metrics, model

    for epoch in range(start_epoch, config['epochs']):
        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch,
            warmup_epochs=config.get('warmup_epochs', 5),
            base_lr=config['lr']
        )

        # 更新学习率 (warmup 后)
        if epoch >= config.get('warmup_epochs', 5):
            scheduler.step()

        # 验证
        val_metrics = evaluate_model(model, val_loader, device, desc=f'Val E{epoch}')

        print(f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f} Acc={train_metrics['accuracy']*100:.1f}% | "
              f"Val {format_metrics(val_metrics)}")

        # 保存最佳
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_metrics = val_metrics.copy()
            best_metrics['epoch'] = epoch

            save_model = model.module if hasattr(model, 'module') else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': save_model.state_dict(),
                'metrics': best_metrics,
            }, os.path.join(checkpoint_dir, f'{model_name}_best.pth'))
            print(f"  -> New best AUC: {best_auc*100:.2f}%")

    print(f"\n{model_name} training complete. Best AUC: {best_auc*100:.2f}% (epoch {best_metrics.get('epoch', -1)})")
    return best_metrics, model


# =============================================================================
# 数据集加载工具
# =============================================================================

def create_eval_loader(dataset_root, split='test', batch_size=64, img_size=224, num_workers=4):
    """创建评估数据加载器"""
    transform = get_multi_source_transforms(img_size, is_train=False)
    dataset_root = Path(dataset_root)

    # 检测数据集格式
    split_dir = dataset_root / split

    if not split_dir.exists():
        # 尝试没有split的格式
        split_dir = dataset_root

    # 检查 ai/nature 格式 (GenImage)
    if (split_dir / 'ai').exists() and (split_dir / 'nature').exists():
        dataset = GenImageDataset(
            root_dir=str(dataset_root), split=split,
            transform=transform, balance_classes=False
        )
    # 检查 real/fake 格式 (CIFAKE, DiffForensics)
    elif (split_dir / 'real').exists() and (split_dir / 'fake').exists():
        samples = []
        real_dir = split_dir / 'real'
        fake_dir = split_dir / 'fake'
        extensions = {'.png', '.jpg', '.jpeg', '.JPEG', '.JPG', '.PNG', '.bmp', '.webp'}

        for p in real_dir.iterdir():
            if p.suffix in extensions:
                samples.append((p, 0))
        for p in fake_dir.iterdir():
            if p.suffix in extensions:
                samples.append((p, 1))

        dataset = SimpleImageDataset(samples, transform)
    else:
        print(f"[Warning] Unknown dataset format at {dataset_root}")
        return None

    if len(dataset) == 0:
        return None

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )


class SimpleImageDataset(torch.utils.data.Dataset):
    """简单图像数据集"""
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': label, 'path': str(path)}


from PIL import Image


# =============================================================================
# 实验流程
# =============================================================================

def train_our_model(config, device, num_gpus, resume=False):
    """训练 FDAA-Net V2"""
    output_dir = setup_output_dir(config['output_dir'])

    # 检查是否需要resume
    resume_ckpt = None
    if resume:
        ckpt_path = os.path.join(output_dir, 'checkpoints', 'fdaa_net_v2_best.pth')
        if os.path.exists(ckpt_path):
            resume_ckpt = ckpt_path
            print(f"[Resume] Found checkpoint: {ckpt_path}")
        else:
            print("[Resume] No checkpoint found, training from scratch")

    # 创建多源数据加载器
    print("\n[Data] Creating multi-source dataloaders...")
    train_loader = create_multi_source_dataloader(
        genimage_root=config['genimage_root'],
        sources=config['sources'],
        split='train',
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        max_samples_per_source=config['max_samples_per_source'],
        num_workers=config['num_workers'],
        strong_aug=True,
    )

    val_loader = create_multi_source_dataloader(
        genimage_root=config['genimage_root'],
        sources=config['sources'],
        split='val',
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        max_samples_per_source=5000,
        num_workers=config['num_workers'],
        strong_aug=False,
    )

    # 创建模型 (embed_dim=1024 匹配 CLIP ViT-L/14)
    print("\n[Model] Creating AIGCDetectorV2 (embed_dim=1024)...")
    model = AIGCDetectorV2(
        backbone_name=config['backbone'],
        num_classes=2,
        img_size=config['img_size'],
        embed_dim=config.get('embed_dim', 1024),
        use_hierarchical=config['use_hierarchical'],
        dropout=config['dropout'],
        freeze_backbone=True,
    )

    # 训练
    best_metrics, model = train_model(
        model, train_loader, val_loader, config,
        model_name='fdaa_net_v2',
        output_dir=output_dir,
        device=device,
        num_gpus=num_gpus,
        resume_checkpoint=resume_ckpt,
    )

    return best_metrics


def train_sota_methods(config, device, num_gpus):
    """训练所有SOTA方法"""
    output_dir = setup_output_dir(config['output_dir'])
    results = {}

    # 创建多源数据加载器 (相同数据)
    print("\n[Data] Creating multi-source dataloaders for SOTA comparison...")
    train_loader = create_multi_source_dataloader(
        genimage_root=config['genimage_root'],
        sources=config['sources'],
        split='train',
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        max_samples_per_source=config['max_samples_per_source'],
        num_workers=config['num_workers'],
        strong_aug=True,
    )

    val_loader = create_multi_source_dataloader(
        genimage_root=config['genimage_root'],
        sources=config['sources'],
        split='val',
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        max_samples_per_source=5000,
        num_workers=config['num_workers'],
        strong_aug=False,
    )

    for method_name in config.get('sota_methods', []):
        print(f"\n{'='*60}")
        print(f"Training SOTA: {method_name}")
        print(f"{'='*60}")

        try:
            model = create_sota_model(method_name, num_classes=2)

            # SOTA方法使用相同训练配置但不开对比损失，epoch数可调
            sota_config = config.copy()
            sota_config['use_contrastive'] = False
            sota_config['use_hierarchical'] = False
            sota_config['epochs'] = config.get('sota_epochs', config['epochs'])

            best_metrics, _ = train_model(
                model, train_loader, val_loader, sota_config,
                model_name=method_name,
                output_dir=output_dir,
                device=device,
                num_gpus=num_gpus,
            )
            results[method_name] = best_metrics

        except Exception as e:
            print(f"[Error] Failed to train {method_name}: {e}")
            import traceback
            traceback.print_exc()
            results[method_name] = {'error': str(e)}

    save_json(results, os.path.join(output_dir, 'results', 'sota_train_results.json'))
    return results


def cross_dataset_evaluation(config, device):
    """跨数据集评估 (包括跨生成器评估)"""
    output_dir = config['output_dir']
    results = {}
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')

    # 收集所有已训练模型
    model_configs = {}

    # 我们的模型
    ours_ckpt = os.path.join(checkpoint_dir, 'fdaa_net_v2_best.pth')
    if os.path.exists(ours_ckpt):
        model_configs['Ours (FDAA-Net V2)'] = {
            'checkpoint': ours_ckpt,
            'create_fn': lambda: AIGCDetectorV2(
                backbone_name=config['backbone'],
                num_classes=2,
                img_size=config['img_size'],
                embed_dim=config.get('embed_dim', 1024),
                use_hierarchical=config['use_hierarchical'],
                dropout=config['dropout'],
            )
        }

    # SOTA模型
    for method_name in config.get('sota_methods', []):
        ckpt = os.path.join(checkpoint_dir, f'{method_name}_best.pth')
        if os.path.exists(ckpt):
            method = method_name
            model_configs[method_name] = {
                'checkpoint': ckpt,
                'create_fn': lambda m=method: create_sota_model(m, num_classes=2)
            }

    if not model_configs:
        print("[Warning] No trained models found for evaluation")
        return {}

    # ==========================================
    # 1. 跨生成器评估 (GenImage内部)
    # ==========================================
    genimage_root = config['genimage_root']
    train_sources = set(config['sources'])
    all_sources = ['biggan', 'adm', 'glide', 'vqdm']
    # 找出训练中未使用的生成器 (held-out)
    held_out_sources = [s for s in all_sources if s not in train_sources]
    # 也对训练使用的生成器单独测试 (域内)
    in_domain_sources = list(train_sources)

    print(f"\n{'='*60}")
    print(f"Cross-generator evaluation within GenImage")
    print(f"Train sources: {list(train_sources)}")
    print(f"Held-out sources: {held_out_sources}")
    print(f"{'='*60}")

    # 对每个生成器的val集单独评估
    for source_name in all_sources:
        dataset_key = f"GenImage_{source_name}"
        print(f"\n--- Evaluating on {dataset_key} ---")

        loader = create_multi_source_dataloader(
            genimage_root=genimage_root,
            sources=[source_name],
            split='val',
            batch_size=config['batch_size'],
            img_size=config['img_size'],
            max_samples_per_source=5000,
            num_workers=config['num_workers'],
            strong_aug=False,
        )

        if loader is None or len(loader.dataset) == 0:
            print(f"  [Skip] Cannot load {source_name}")
            continue

        is_held_out = source_name not in train_sources
        results[dataset_key] = {'_held_out': is_held_out}

        for model_name, model_cfg in model_configs.items():
            try:
                model = model_cfg['create_fn']()
                ckpt = torch.load(model_cfg['checkpoint'], map_location='cpu')
                model.load_state_dict(ckpt['model_state_dict'])
                model = model.to(device)

                metrics = evaluate_model(model, loader, device, desc=f'{model_name} on {dataset_key}')
                results[dataset_key][model_name] = metrics
                marker = " [HELD-OUT]" if is_held_out else " [IN-DOMAIN]"
                print(f"  {model_name}{marker}: {format_metrics(metrics)}")

                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [Error] {model_name}: {e}")
                results[dataset_key][model_name] = {'error': str(e)}

    # ==========================================
    # 2. 跨数据集评估 (外部数据集)
    # ==========================================
    for dataset_name, dataset_root in config.get('eval_datasets', {}).items():
        print(f"\n{'='*60}")
        print(f"Cross-dataset evaluation: {dataset_name}")
        print(f"{'='*60}")

        loader = create_eval_loader(dataset_root, split='test', batch_size=config['batch_size'])
        if loader is None:
            print(f"[Skip] Cannot load {dataset_name}")
            continue

        results[dataset_name] = {}

        for model_name, model_cfg in model_configs.items():
            try:
                model = model_cfg['create_fn']()
                ckpt = torch.load(model_cfg['checkpoint'], map_location='cpu')
                model.load_state_dict(ckpt['model_state_dict'])
                model = model.to(device)

                metrics = evaluate_model(model, loader, device, desc=f'{model_name} on {dataset_name}')
                results[dataset_name][model_name] = metrics
                print(f"  {model_name}: {format_metrics(metrics)}")

                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [Error] {model_name}: {e}")
                results[dataset_name][model_name] = {'error': str(e)}

    save_json(results, os.path.join(output_dir, 'results', 'cross_dataset_results.json'))
    return results


def run_ablation(config, device, num_gpus):
    """消融实验"""
    output_dir = setup_output_dir(config['output_dir'])
    results = {}

    # 数据加载器
    train_loader = create_multi_source_dataloader(
        genimage_root=config['genimage_root'],
        sources=config['sources'],
        split='train',
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        max_samples_per_source=config['max_samples_per_source'],
        num_workers=config['num_workers'],
        strong_aug=True,
    )

    val_loader = create_multi_source_dataloader(
        genimage_root=config['genimage_root'],
        sources=config['sources'],
        split='val',
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        max_samples_per_source=5000,
        num_workers=config['num_workers'],
        strong_aug=False,
    )

    ablation_config = config.copy()
    ablation_config['epochs'] = min(config['epochs'], 8)  # 消融用更少epoch

    # 消融实验使用 ViT-B/16 (768-dim) Lite 模型进行快速验证
    ablation_dim = 768  # Lite 模型使用 ViT-B/16

    # 1. Baseline: 冻结ViT + 线性分类器 (无FDAA, 无MGFP)
    print("\n[Ablation 1/4] Baseline (ViT + Linear)")
    model_baseline = AIGCDetectorLite(
        num_classes=2, img_size=config['img_size'],
        embed_dim=ablation_dim, pretrained=True, version='v1'
    )
    # 替换FDAA/MGFP为identity
    model_baseline.fdaa = nn.Identity()

    class SimplePool(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
        def forward(self, cls_token, patch_tokens, return_attention=False):
            out = self.norm(cls_token)
            if return_attention:
                B, N, _ = patch_tokens.shape
                return out, {'forgery_map': torch.zeros(B, N, device=cls_token.device)}
            return out

    model_baseline.mgfp = SimplePool(ablation_dim)
    ablation_config['use_contrastive'] = False
    ablation_config['use_hierarchical'] = False

    results['baseline'], _ = train_model(
        model_baseline, train_loader, val_loader, ablation_config,
        model_name='ablation_baseline', output_dir=output_dir,
        device=device, num_gpus=num_gpus
    )

    # 2. Baseline + FDAAv2
    print("\n[Ablation 2/4] Baseline + FDAAv2")
    model_fdaa = AIGCDetectorLite(
        num_classes=2, img_size=config['img_size'],
        embed_dim=ablation_dim, pretrained=True, version='v2'
    )
    # 使用简单聚合替代完整MGFP，仅保留freq分支效果

    class FreqOnlyPool(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
            )
        def forward(self, cls_token, patch_tokens, freq_feat, return_attention=False):
            out = self.mlp(torch.cat([cls_token, freq_feat], dim=-1))
            if return_attention:
                B, N, _ = patch_tokens.shape
                return out, {'forgery_map': torch.zeros(B, N, device=cls_token.device)}
            return out

    model_fdaa.mgfp = FreqOnlyPool(ablation_dim)
    ablation_config['use_contrastive'] = False

    results['baseline+fdaa'], _ = train_model(
        model_fdaa, train_loader, val_loader, ablation_config,
        model_name='ablation_fdaa', output_dir=output_dir,
        device=device, num_gpus=num_gpus
    )

    # 3. Baseline + MGFPv2 (无频率分支)
    print("\n[Ablation 3/4] Baseline + MGFPv2 (no freq)")
    model_mgfp = AIGCDetectorLite(
        num_classes=2, img_size=config['img_size'],
        embed_dim=ablation_dim, pretrained=True, version='v2'
    )
    # 频率分支输出零向量
    _abl_dim = ablation_dim
    model_mgfp.fdaa = type('ZeroFreq', (nn.Module,), {
        '__init__': lambda self: nn.Module.__init__(self),
        'forward': lambda self, x: torch.zeros(x.shape[0], _abl_dim, device=x.device)
    })()

    results['baseline+mgfp'], _ = train_model(
        model_mgfp, train_loader, val_loader, ablation_config,
        model_name='ablation_mgfp', output_dir=output_dir,
        device=device, num_gpus=num_gpus
    )

    # 4. Full model (FDAAv2 + MGFPv2)
    print("\n[Ablation 4/4] Full (FDAAv2 + MGFPv2)")
    model_full = AIGCDetectorLite(
        num_classes=2, img_size=config['img_size'],
        embed_dim=ablation_dim, pretrained=True, version='v2'
    )
    ablation_config['use_contrastive'] = True

    results['full'], _ = train_model(
        model_full, train_loader, val_loader, ablation_config,
        model_name='ablation_full', output_dir=output_dir,
        device=device, num_gpus=num_gpus
    )

    save_json(results, os.path.join(output_dir, 'results', 'ablation_results.json'))
    return results


# =============================================================================
# 报告生成
# =============================================================================

def generate_report(config):
    """生成 Markdown 报告"""
    output_dir = config['output_dir']
    results_dir = os.path.join(output_dir, 'results')
    report_path = os.path.join(output_dir, 'reports', 'experiment_report.md')

    lines = [
        "# FDAA-Net V2 实验报告",
        f"\n生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n## 实验设置",
        f"- 训练源: {', '.join(config['sources'])}",
        f"- 每源样本数: {config['max_samples_per_source']}",
        f"- Batch size: {config['batch_size']}",
        f"- 学习率: {config['lr']}",
        f"- Epochs: {config['epochs']}",
        f"- Backbone: {config['backbone']}",
        "",
    ]

    # SOTA对比结果
    sota_results_path = os.path.join(results_dir, 'sota_train_results.json')
    if os.path.exists(sota_results_path):
        with open(sota_results_path) as f:
            sota_results = json.load(f)

        lines.append("## 域内评估结果 (SOTA对比)")
        lines.append("")
        lines.append("| Method | AUC (%) | AP (%) | Accuracy (%) | EER (%) |")
        lines.append("|--------|---------|--------|--------------|---------|")

        for method, metrics in sorted(sota_results.items(), key=lambda x: x[1].get('auc', 0), reverse=True):
            if 'error' in metrics:
                lines.append(f"| {method} | Error | - | - | - |")
            else:
                lines.append(
                    f"| {method} | {metrics.get('auc', 0)*100:.2f} | "
                    f"{metrics.get('ap', 0)*100:.2f} | "
                    f"{metrics.get('accuracy', 0)*100:.2f} | "
                    f"{metrics.get('eer', 0)*100:.2f} |"
                )
        lines.append("")

    # 跨域结果
    cross_results_path = os.path.join(results_dir, 'cross_dataset_results.json')
    if os.path.exists(cross_results_path):
        with open(cross_results_path) as f:
            cross_results = json.load(f)

        # 分离跨生成器和跨数据集结果
        genimage_results = {k: v for k, v in cross_results.items() if k.startswith('GenImage_')}
        external_results = {k: v for k, v in cross_results.items() if not k.startswith('GenImage_')}

        if genimage_results:
            lines.append("## 跨生成器评估结果 (GenImage)")
            lines.append("")
            lines.append("| Generator | Type | Method | AUC (%) | AP (%) | Accuracy (%) |")
            lines.append("|-----------|------|--------|---------|--------|--------------|")

            for dataset_name, dataset_results in sorted(genimage_results.items()):
                is_held_out = dataset_results.pop('_held_out', False)
                gen_type = "HELD-OUT" if is_held_out else "IN-DOMAIN"
                for method, metrics in sorted(dataset_results.items(), key=lambda x: x[1].get('auc', 0), reverse=True):
                    if 'error' in metrics:
                        lines.append(f"| {dataset_name} | {gen_type} | {method} | Error | - | - |")
                    else:
                        lines.append(
                            f"| {dataset_name} | {gen_type} | {method} | "
                            f"{metrics.get('auc', 0)*100:.2f} | "
                            f"{metrics.get('ap', 0)*100:.2f} | "
                            f"{metrics.get('accuracy', 0)*100:.2f} |"
                        )
            lines.append("")

        if external_results:
            lines.append("## 跨数据集评估结果 (外部数据集)")
            for dataset_name, dataset_results in external_results.items():
                lines.append(f"\n### {dataset_name}")
                lines.append("")
                lines.append("| Method | AUC (%) | AP (%) | Accuracy (%) |")
                lines.append("|--------|---------|--------|--------------|")

                for method, metrics in sorted(dataset_results.items(), key=lambda x: x[1].get('auc', 0), reverse=True):
                    if 'error' in metrics:
                        lines.append(f"| {method} | Error | - | - |")
                    else:
                        lines.append(
                            f"| {method} | {metrics.get('auc', 0)*100:.2f} | "
                            f"{metrics.get('ap', 0)*100:.2f} | "
                            f"{metrics.get('accuracy', 0)*100:.2f} |"
                        )
                lines.append("")

    # 消融结果
    ablation_results_path = os.path.join(results_dir, 'ablation_results.json')
    if os.path.exists(ablation_results_path):
        with open(ablation_results_path) as f:
            ablation_results = json.load(f)

        lines.append("## 消融实验")
        lines.append("")
        lines.append("| Variant | AUC (%) | AP (%) | Accuracy (%) | EER (%) |")
        lines.append("|---------|---------|--------|--------------|---------|")

        for variant, metrics in ablation_results.items():
            if 'error' in metrics:
                lines.append(f"| {variant} | Error | - | - | - |")
            else:
                lines.append(
                    f"| {variant} | {metrics.get('auc', 0)*100:.2f} | "
                    f"{metrics.get('ap', 0)*100:.2f} | "
                    f"{metrics.get('accuracy', 0)*100:.2f} | "
                    f"{metrics.get('eer', 0)*100:.2f} |"
                )
        lines.append("")

    report = '\n'.join(lines)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n[Report] Generated: {report_path}")
    print(report)


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='FDAA-Net V2 Improved Experiment')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'train_ours', 'train_sota', 'evaluate', 'ablation', 'report'],
                        help='Experiment mode')
    parser.add_argument('--genimage_root', type=str, default=None,
                        help='GenImage dataset root')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from best checkpoint')

    args = parser.parse_args()

    # 配置
    config = DEFAULT_CONFIG.copy()
    if args.genimage_root:
        config['genimage_root'] = args.genimage_root
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.epochs:
        config['epochs'] = args.epochs
    if args.lr:
        config['lr'] = args.lr
    if args.max_samples:
        config['max_samples_per_source'] = args.max_samples

    # 设置种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 设备
    device, num_gpus = get_device()

    # 输出目录
    setup_output_dir(config['output_dir'])
    save_json(config, os.path.join(config['output_dir'], 'config.json'))

    print(f"\n{'#'*60}")
    print(f"# FDAA-Net V2 Improved Experiment")
    print(f"# Mode: {args.mode}")
    print(f"# Device: {device} ({num_gpus} GPUs)")
    print(f"# Output: {config['output_dir']}")
    print(f"{'#'*60}")

    start_time = time.time()

    if args.mode in ['all', 'train_ours']:
        train_our_model(config, device, num_gpus, resume=args.resume)

    if args.mode in ['all', 'train_sota']:
        train_sota_methods(config, device, num_gpus)

    if args.mode in ['all', 'evaluate']:
        cross_dataset_evaluation(config, device)

    if args.mode in ['all', 'ablation']:
        run_ablation(config, device, num_gpus)

    if args.mode in ['all', 'report']:
        generate_report(config)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Experiment completed in {elapsed/3600:.2f} hours")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
