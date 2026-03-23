"""
消融实验脚本

对两个创新点进行完整消融实验:
1. FDAA模块消融 (空间Adapter、频率Adapter、跨域交互)
2. MGFP模块消融 (Patch特征、伪造注意力、层级感知)

Usage:
    python experiments/run_ablation.py --config configs/default.yaml --output_dir ./ablation_results
"""

import os
import sys
import argparse
import yaml
import json
import copy
import torch
import torch.nn as nn
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.modules.fdaa import FDAA, SpatialAdapter, FrequencyAdapter
from models.modules.mgfp import MGFP, PatchForgeryAttention, MultiGranularityAggregation
from models import AIGCDetectorLite
from data import create_dataloader
from trainers import Trainer
from models.losses import AIGCDetectionLoss
from utils.logger import setup_logger


class AblationConfig:
    """消融实验配置"""

    # FDAA模块消融配置
    FDAA_ABLATIONS = {
        'baseline': {
            'description': 'Without FDAA (Baseline)',
            'use_fdaa': False,
            'use_spatial': False,
            'use_frequency': False,
            'use_cross_interaction': False
        },
        'spatial_only': {
            'description': 'Spatial Adapter Only',
            'use_fdaa': True,
            'use_spatial': True,
            'use_frequency': False,
            'use_cross_interaction': False
        },
        'frequency_only': {
            'description': 'Frequency Adapter Only',
            'use_fdaa': True,
            'use_spatial': False,
            'use_frequency': True,
            'use_cross_interaction': False
        },
        'dual_domain': {
            'description': 'Spatial + Frequency (No Interaction)',
            'use_fdaa': True,
            'use_spatial': True,
            'use_frequency': True,
            'use_cross_interaction': False
        },
        'full_fdaa': {
            'description': 'Full FDAA (Ours)',
            'use_fdaa': True,
            'use_spatial': True,
            'use_frequency': True,
            'use_cross_interaction': True
        }
    }

    # MGFP模块消融配置
    MGFP_ABLATIONS = {
        'baseline': {
            'description': 'Without MGFP (Global Feature Only)',
            'use_mgfp': False,
            'use_patch_attention': False,
            'use_forgery_prototype': False,
            'use_hierarchical': False
        },
        'patch_only': {
            'description': 'With Patch Features',
            'use_mgfp': True,
            'use_patch_attention': True,
            'use_forgery_prototype': False,
            'use_hierarchical': False
        },
        'with_prototype': {
            'description': 'With Forgery Prototypes',
            'use_mgfp': True,
            'use_patch_attention': True,
            'use_forgery_prototype': True,
            'use_hierarchical': False
        },
        'full_mgfp': {
            'description': 'Full MGFP (Ours)',
            'use_mgfp': True,
            'use_patch_attention': True,
            'use_forgery_prototype': True,
            'use_hierarchical': True
        }
    }


class AblationDetector(nn.Module):
    """
    消融实验专用检测器
    支持灵活开关各个模块
    """
    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 2,
        img_size: int = 224,
        patch_size: int = 16,
        # FDAA配置
        use_fdaa: bool = True,
        use_spatial: bool = True,
        use_frequency: bool = True,
        use_cross_interaction: bool = True,
        # MGFP配置
        use_mgfp: bool = True,
        use_patch_attention: bool = True,
        use_forgery_prototype: bool = True,
        use_hierarchical: bool = True,
        num_prototypes: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_fdaa = use_fdaa
        self.use_mgfp = use_mgfp

        # 简单backbone (模拟ViT输出)
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(4)
        ])

        # FDAA模块 (可选)
        if use_fdaa:
            self.fdaa = self._build_fdaa(
                embed_dim, img_size, patch_size,
                use_spatial, use_frequency, use_cross_interaction, dropout
            )
        else:
            self.fdaa = None

        # MGFP模块 (可选)
        if use_mgfp:
            self.mgfp = self._build_mgfp(
                embed_dim, self.num_patches,
                use_patch_attention, use_forgery_prototype, use_hierarchical,
                num_prototypes, dropout
            )
        else:
            self.mgfp = None

        # 分类头
        final_dim = embed_dim
        if use_mgfp:
            # MGFP输出包含多粒度特征
            final_dim = embed_dim * 3 if use_forgery_prototype else embed_dim

        self.classifier = nn.Linear(final_dim, num_classes)

        # 辅助分类器
        self.aux_classifier = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _build_fdaa(
        self, dim, img_size, patch_size,
        use_spatial, use_frequency, use_cross_interaction, dropout
    ):
        """构建FDAA模块（支持部分组件）"""
        if not use_spatial and not use_frequency:
            return None

        # 使用完整FDAA但可以部分禁用
        fdaa = FDAA(
            dim=dim,
            img_size=img_size,
            patch_size=patch_size,
            dropout=dropout
        )

        # 通过设置标志来控制使用哪些部分
        fdaa.use_spatial = use_spatial
        fdaa.use_frequency = use_frequency
        fdaa.use_cross_interaction = use_cross_interaction

        return fdaa

    def _build_mgfp(
        self, dim, num_patches,
        use_patch_attention, use_forgery_prototype, use_hierarchical,
        num_prototypes, dropout
    ):
        """构建MGFP模块（支持部分组件）"""
        if not use_patch_attention:
            return None

        mgfp = MGFP(
            dim=dim,
            num_patches=num_patches,
            num_prototypes=num_prototypes if use_forgery_prototype else 0,
            use_hierarchical=use_hierarchical,
            dropout=dropout
        )

        return mgfp

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, return_attention=False):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # [B, D, H, W]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]

        # 添加cls token和位置编码
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # FDAA处理
        if self.fdaa is not None:
            # 分离cls和patch tokens
            cls_token = x[:, :1]
            patch_tokens = x[:, 1:]

            # FDAA增强
            fdaa_output = self.fdaa(patch_tokens.transpose(1, 2).view(
                B, -1, int(self.num_patches ** 0.5), int(self.num_patches ** 0.5)
            ).permute(0, 1, 2, 3))

            if isinstance(fdaa_output, dict):
                enhanced_features = fdaa_output['fused']
            else:
                enhanced_features = fdaa_output

            # 更新patch tokens
            if len(enhanced_features.shape) == 4:
                enhanced_features = enhanced_features.flatten(2).transpose(1, 2)
            patch_tokens = patch_tokens + enhanced_features

            x = torch.cat([cls_token, patch_tokens], dim=1)

        # 提取特征
        cls_feature = x[:, 0]  # [B, D]
        patch_features = x[:, 1:]  # [B, N, D]

        # MGFP处理
        outputs = {}
        if self.mgfp is not None:
            mgfp_output = self.mgfp(x, return_attention=return_attention)
            final_feature = mgfp_output['fused_feature']

            if return_attention:
                outputs['forgery_map'] = mgfp_output.get('forgery_attention')
        else:
            final_feature = cls_feature

        # 分类
        outputs['logits'] = self.classifier(final_feature)
        outputs['aux_logits'] = self.aux_classifier(cls_feature)
        outputs['features'] = final_feature

        return outputs


def run_single_ablation(
    config: dict,
    ablation_config: dict,
    train_loader,
    val_loader,
    device: str,
    output_dir: str,
    logger
) -> dict:
    """
    运行单个消融实验

    Args:
        config: 基础配置
        ablation_config: 消融配置
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 设备
        output_dir: 输出目录
        logger: 日志器
    Returns:
        results: 实验结果
    """
    logger.info(f"Running ablation: {ablation_config['description']}")

    # 构建模型
    model = AblationDetector(
        embed_dim=config.get('model', {}).get('embed_dim', 768),
        num_classes=2,
        img_size=config.get('model', {}).get('img_size', 224),
        # FDAA配置
        use_fdaa=ablation_config.get('use_fdaa', True),
        use_spatial=ablation_config.get('use_spatial', True),
        use_frequency=ablation_config.get('use_frequency', True),
        use_cross_interaction=ablation_config.get('use_cross_interaction', True),
        # MGFP配置
        use_mgfp=ablation_config.get('use_mgfp', True),
        use_patch_attention=ablation_config.get('use_patch_attention', True),
        use_forgery_prototype=ablation_config.get('use_forgery_prototype', True),
        use_hierarchical=ablation_config.get('use_hierarchical', True),
        num_prototypes=config.get('model', {}).get('num_prototypes', 4),
        dropout=config.get('model', {}).get('dropout', 0.1)
    )

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 损失函数
    criterion = AIGCDetectionLoss(
        use_focal=True,
        use_aux=True,
        aux_weight=0.5
    )

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('training', {}).get('learning_rate', 1e-4),
        weight_decay=config.get('training', {}).get('weight_decay', 1e-4)
    )

    # 调度器
    num_epochs = config.get('training', {}).get('num_epochs', 30)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
    )

    # 训练器
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        use_amp=True,
        log_interval=50,
        save_interval=10
    )

    # 训练
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs
    )

    # 获取最佳结果
    best_auc = max(history.get('val_auc', [0]))
    best_acc = max(history.get('val_acc', [0]))

    results = {
        'description': ablation_config['description'],
        'config': ablation_config,
        'best_auc': best_auc,
        'best_acc': best_acc,
        'history': history
    }

    return results


def run_ablation_study(args):
    """运行完整消融实验"""

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f'ablation_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # 设置日志
    logger = setup_logger('ablation', log_dir=output_dir)
    logger.info(f"Starting ablation study")
    logger.info(f"Output directory: {output_dir}")

    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # 数据加载器
    data_root = config.get('paths', {}).get('data_root', './datasets')
    train_loader = create_dataloader(data_root, 'train', batch_size=32)
    val_loader = create_dataloader(data_root, 'val', batch_size=32)

    all_results = {
        'fdaa_ablation': {},
        'mgfp_ablation': {}
    }

    # FDAA消融实验
    if not args.skip_fdaa:
        logger.info("=" * 60)
        logger.info("Running FDAA Ablation Study")
        logger.info("=" * 60)

        for name, ablation_config in AblationConfig.FDAA_ABLATIONS.items():
            exp_output_dir = os.path.join(output_dir, 'fdaa', name)
            os.makedirs(exp_output_dir, exist_ok=True)

            # 合并MGFP配置（使用完整MGFP）
            full_config = {**ablation_config}
            full_config.update({
                'use_mgfp': True,
                'use_patch_attention': True,
                'use_forgery_prototype': True,
                'use_hierarchical': True
            })

            results = run_single_ablation(
                config=config,
                ablation_config=full_config,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                output_dir=exp_output_dir,
                logger=logger
            )

            all_results['fdaa_ablation'][name] = {
                'description': results['description'],
                'best_auc': results['best_auc'],
                'best_acc': results['best_acc']
            }

            logger.info(f"  {name}: AUC={results['best_auc']:.4f}, Acc={results['best_acc']:.4f}")

    # MGFP消融实验
    if not args.skip_mgfp:
        logger.info("=" * 60)
        logger.info("Running MGFP Ablation Study")
        logger.info("=" * 60)

        for name, ablation_config in AblationConfig.MGFP_ABLATIONS.items():
            exp_output_dir = os.path.join(output_dir, 'mgfp', name)
            os.makedirs(exp_output_dir, exist_ok=True)

            # 合并FDAA配置（使用完整FDAA）
            full_config = {**ablation_config}
            full_config.update({
                'use_fdaa': True,
                'use_spatial': True,
                'use_frequency': True,
                'use_cross_interaction': True
            })

            results = run_single_ablation(
                config=config,
                ablation_config=full_config,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                output_dir=exp_output_dir,
                logger=logger
            )

            all_results['mgfp_ablation'][name] = {
                'description': results['description'],
                'best_auc': results['best_auc'],
                'best_acc': results['best_acc']
            }

            logger.info(f"  {name}: AUC={results['best_auc']:.4f}, Acc={results['best_acc']:.4f}")

    # 保存结果
    results_path = os.path.join(output_dir, 'ablation_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # 生成结果表格
    from visualization import create_comparison_table

    logger.info("\n" + "=" * 60)
    logger.info("FDAA Ablation Results")
    logger.info("=" * 60)

    fdaa_results = {
        k: {'accuracy': v['best_acc'], 'auc': v['best_auc']}
        for k, v in all_results['fdaa_ablation'].items()
    }
    if fdaa_results:
        table = create_comparison_table(fdaa_results, metrics=['accuracy', 'auc'])
        logger.info("\n" + table)

    logger.info("\n" + "=" * 60)
    logger.info("MGFP Ablation Results")
    logger.info("=" * 60)

    mgfp_results = {
        k: {'accuracy': v['best_acc'], 'auc': v['best_auc']}
        for k, v in all_results['mgfp_ablation'].items()
    }
    if mgfp_results:
        table = create_comparison_table(mgfp_results, metrics=['accuracy', 'auc'])
        logger.info("\n" + table)

    logger.info("\nAblation study completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Ablation Study')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='./ablation_results',
                        help='Output directory')
    parser.add_argument('--skip_fdaa', action='store_true',
                        help='Skip FDAA ablation')
    parser.add_argument('--skip_mgfp', action='store_true',
                        help='Skip MGFP ablation')

    args = parser.parse_args()
    run_ablation_study(args)
