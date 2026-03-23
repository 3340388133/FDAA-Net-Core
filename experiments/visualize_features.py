"""
特征可视化脚本

包含：
1. Grad-CAM 热力图可视化
2. t-SNE 特征分布可视化
3. 伪造注意力图可视化
4. ROC曲线和混淆矩阵

Usage:
    python experiments/visualize_features.py --checkpoint path/to/checkpoint.pth
    python experiments/visualize_features.py --vis_type all
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.detector_ablation import create_ablation_model
from data.genimage_dataset import GenImageDataset


class GradCAM:
    """Grad-CAM可视化"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册hook
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        """生成CAM"""
        self.model.eval()
        self.model.zero_grad()

        # 前向传播
        output = self.model(input_tensor)
        logits = output['logits']

        if target_class is None:
            target_class = logits.argmax(dim=1)

        # 反向传播
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, target_class.view(-1, 1), 1)
        logits.backward(gradient=one_hot, retain_graph=True)

        # 计算CAM
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)

        # 归一化
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam


def visualize_gradcam(model, dataloader, device, output_dir, num_samples=16):
    """Grad-CAM可视化"""
    print("生成Grad-CAM可视化...")

    os.makedirs(output_dir, exist_ok=True)

    # 找到目标层（使用backbone的最后一层）
    target_layer = None
    for name, module in model.named_modules():
        if 'transformer' in name.lower() or 'encoder' in name.lower():
            if hasattr(module, 'norm'):
                target_layer = module.norm
                break
        if 'layer4' in name:  # ResNet
            target_layer = module
            break

    if target_layer is None:
        # 使用模型的某个子模块
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                target_layer = module
                break

    if target_layer is None:
        print("警告: 无法找到合适的目标层，跳过Grad-CAM")
        return

    grad_cam = GradCAM(model, target_layer)

    # 收集样本
    model.eval()
    samples = []
    labels_list = []

    for batch in dataloader:
        images = batch['image'].to(device)
        labels = batch['label']

        for i in range(min(len(images), num_samples - len(samples))):
            samples.append(images[i:i+1])
            labels_list.append(labels[i].item())

        if len(samples) >= num_samples:
            break

    # 生成可视化
    fig, axes = plt.subplots(4, 8, figsize=(24, 12))
    axes = axes.flatten()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for idx, (img_tensor, label) in enumerate(zip(samples, labels_list)):
        if idx >= len(axes):
            break

        # 生成CAM
        try:
            cam = grad_cam.generate_cam(img_tensor)
            cam = cam.cpu().numpy()[0]

            # 上采样到原始大小
            cam = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224)))
            cam = cam / 255.0

        except Exception as e:
            print(f"  样本 {idx} CAM生成失败: {e}")
            cam = np.zeros((224, 224))

        # 反归一化图像
        img = img_tensor[0].cpu()
        img = img * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        # 叠加热力图
        heatmap = plt.cm.jet(cam)[:, :, :3]
        overlay = 0.5 * img + 0.5 * heatmap

        ax = axes[idx * 2] if idx * 2 < len(axes) else axes[idx]
        ax.imshow(img)
        ax.set_title(f"{'Fake' if label else 'Real'}", fontsize=10)
        ax.axis('off')

        if idx * 2 + 1 < len(axes):
            ax2 = axes[idx * 2 + 1]
            ax2.imshow(overlay)
            ax2.set_title("Grad-CAM", fontsize=10)
            ax2.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradcam_visualization.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存到: {output_dir}/gradcam_visualization.png")


def visualize_tsne(model, dataloader, device, output_dir, num_samples=500):
    """t-SNE特征可视化"""
    print("生成t-SNE可视化...")

    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("警告: sklearn未安装，跳过t-SNE")
        return

    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label']

            # 获取特征
            outputs = model(images)

            # 尝试获取中间特征
            if 'features' in outputs:
                feat = outputs['features']
            else:
                # 使用logits前的特征
                feat = outputs['logits']

            features_list.append(feat.cpu().numpy())
            labels_list.extend(labels.numpy())

            if len(labels_list) >= num_samples:
                break

    features = np.concatenate(features_list, axis=0)[:num_samples]
    labels = np.array(labels_list)[:num_samples]

    # t-SNE
    print(f"  运行t-SNE (样本数: {len(features)})...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4))
    features_2d = tsne.fit_transform(features)

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#2ecc71', '#e74c3c']  # 绿色=真实, 红色=伪造
    markers = ['o', 's']

    for label_val in [0, 1]:
        mask = labels == label_val
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=colors[label_val],
            marker=markers[label_val],
            label='Real' if label_val == 0 else 'Fake',
            alpha=0.6,
            s=50
        )

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('Feature Distribution (t-SNE)', fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tsne_visualization.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存到: {output_dir}/tsne_visualization.png")


def visualize_forgery_attention(model, dataloader, device, output_dir, num_samples=8):
    """伪造注意力图可视化"""
    print("生成伪造注意力图可视化...")

    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    samples = []
    labels_list = []
    attention_maps = []

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label']

            outputs = model(images)

            # 检查是否有伪造图
            if 'forgery_map' in outputs:
                for i in range(min(len(images), num_samples - len(samples))):
                    samples.append(images[i].cpu())
                    labels_list.append(labels[i].item())
                    attention_maps.append(outputs['forgery_map'][i].cpu())

            if len(samples) >= num_samples:
                break

    if not attention_maps:
        print("  模型不输出伪造注意力图，跳过此可视化")
        return

    # 绘图
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))

    for idx in range(min(len(samples), num_samples)):
        # 原图
        img = samples[idx]
        img = img * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        axes[0, idx].imshow(img)
        axes[0, idx].set_title(f"{'Fake' if labels_list[idx] else 'Real'}", fontsize=10)
        axes[0, idx].axis('off')

        # 注意力图
        attn = attention_maps[idx].numpy()
        if len(attn.shape) == 1:
            # 假设是14x14的patch
            size = int(np.sqrt(len(attn)))
            attn = attn.reshape(size, size)

        attn_resized = np.array(Image.fromarray((attn * 255).astype(np.uint8)).resize((224, 224))) / 255.0

        axes[1, idx].imshow(attn_resized, cmap='hot')
        axes[1, idx].set_title("Attention", fontsize=10)
        axes[1, idx].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'forgery_attention.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存到: {output_dir}/forgery_attention.png")


def visualize_roc_curve(model, dataloader, device, output_dir):
    """ROC曲线可视化"""
    print("生成ROC曲线...")

    try:
        from sklearn.metrics import roc_curve, auc, confusion_matrix
    except ImportError:
        print("警告: sklearn未安装，跳过ROC曲线")
        return

    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label']

            outputs = model(images)
            probs = torch.softmax(outputs['logits'], dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # ROC曲线
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC曲线
    axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    axes[0].plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
    axes[0].fill_between(fpr, tpr, alpha=0.2)
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC Curve', fontsize=14)
    axes[0].legend(loc='lower right', fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # 混淆矩阵
    preds = (all_probs > 0.5).astype(int)
    cm = confusion_matrix(all_labels, preds)

    im = axes[1].imshow(cm, cmap='Blues')
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(['Real', 'Fake'], fontsize=11)
    axes[1].set_yticklabels(['Real', 'Fake'], fontsize=11)
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('Actual', fontsize=12)
    axes[1].set_title('Confusion Matrix', fontsize=14)

    # 添加数值
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            axes[1].text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=16, color=color, fontweight='bold')

    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_confusion.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存到: {output_dir}/roc_confusion.png")

    return {'auc': roc_auc, 'accuracy': (preds == all_labels).mean()}


def visualize_comparison_results(results_file, output_dir):
    """可视化SOTA对比结果"""
    print("生成SOTA对比图表...")

    import json

    os.makedirs(output_dir, exist_ok=True)

    with open(results_file, 'r') as f:
        results = json.load(f)

    if 'sota_comparison' not in results:
        print("结果文件中没有SOTA对比数据")
        return

    sota_results = results['sota_comparison']

    methods = []
    aucs = []
    accs = []
    params = []

    for method, metrics in sota_results.items():
        if 'error' not in metrics:
            methods.append(method)
            aucs.append(metrics['auc'])
            accs.append(metrics['accuracy'])
            params.append(metrics['params'] / 1e6)  # 转为M

    # 排序按AUC
    sorted_idx = np.argsort(aucs)[::-1]
    methods = [methods[i] for i in sorted_idx]
    aucs = [aucs[i] for i in sorted_idx]
    accs = [accs[i] for i in sorted_idx]
    params = [params[i] for i in sorted_idx]

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # AUC对比
    colors = ['#e74c3c' if 'Ours' in m else '#3498db' for m in methods]
    bars1 = axes[0].barh(methods, aucs, color=colors, alpha=0.8)
    axes[0].set_xlabel('AUC', fontsize=12)
    axes[0].set_title('Detection Performance (AUC)', fontsize=14)
    axes[0].set_xlim(0, 1.05)

    for bar, auc_val in zip(bars1, aucs):
        axes[0].text(auc_val + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{auc_val:.4f}', va='center', fontsize=10)

    # 参数量vs性能
    colors = ['#e74c3c' if 'Ours' in m else '#3498db' for m in methods]
    scatter = axes[1].scatter(params, aucs, c=colors, s=200, alpha=0.7)

    for i, method in enumerate(methods):
        axes[1].annotate(method, (params[i], aucs[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)

    axes[1].set_xlabel('Parameters (M)', fontsize=12)
    axes[1].set_ylabel('AUC', fontsize=12)
    axes[1].set_title('Efficiency vs Performance', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sota_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存到: {output_dir}/sota_comparison.png")


def main():
    parser = argparse.ArgumentParser(description='特征可视化')

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型检查点路径')
    parser.add_argument('--data_root', type=str,
                        default='datasets/genimage_partial/imagenet_ai_0419_biggan',
                        help='数据集根目录')
    parser.add_argument('--vis_type', type=str, default='all',
                        choices=['all', 'gradcam', 'tsne', 'attention', 'roc', 'comparison'],
                        help='可视化类型')
    parser.add_argument('--results_file', type=str, default=None,
                        help='SOTA对比结果文件（用于comparison可视化）')
    parser.add_argument('--output_dir', type=str, default='./outputs/figures',
                        help='输出目录')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='t-SNE样本数')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--embed_dim', type=int, default=768)

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 如果只是比较可视化
    if args.vis_type == 'comparison' and args.results_file:
        visualize_comparison_results(args.results_file, args.output_dir)
        return

    # 加载模型
    print("加载模型...")
    model = create_ablation_model(
        'full',
        num_classes=2,
        img_size=args.img_size,
        embed_dim=args.embed_dim
    ).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"加载检查点: {args.checkpoint}")

    # 准备数据
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = GenImageDataset(
        root_dir=args.data_root,
        split='val',
        transform=transform,
        max_samples=args.num_samples
    )

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4
    )

    # 运行可视化
    if args.vis_type in ['all', 'gradcam']:
        visualize_gradcam(model, dataloader, device, args.output_dir)

    if args.vis_type in ['all', 'tsne']:
        visualize_tsne(model, dataloader, device, args.output_dir, args.num_samples)

    if args.vis_type in ['all', 'attention']:
        visualize_forgery_attention(model, dataloader, device, args.output_dir)

    if args.vis_type in ['all', 'roc']:
        metrics = visualize_roc_curve(model, dataloader, device, args.output_dir)
        if metrics:
            print(f"  AUC: {metrics['auc']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

    print(f"\n所有可视化结果已保存到: {args.output_dir}")


if __name__ == '__main__':
    main()
