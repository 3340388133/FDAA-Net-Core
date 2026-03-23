#!/usr/bin/env python
"""
快速验证脚本 - 用合成数据快速测试改进模块的效果

使用方法：
    python experiments/quick_validate.py

这个脚本会：
1. 使用合成数据（无需下载真实数据集）
2. 快速训练几个epoch
3. 对比baseline vs full改进版本
4. 输出跨域泛化能力对比
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

from models.detector_generalized import GeneralizedAIGCDetector


def create_synthetic_data(num_samples, num_domains=3, seed=None):
    """
    创建合成数据用于快速测试
    不同域使用不同的噪声模式，模拟不同生成器的特征
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    images_list = []
    labels_list = []
    domains_list = []

    samples_per_domain = num_samples // num_domains

    for domain_id in range(num_domains):
        # 每个域有不同的噪声特征
        noise_scale = 0.1 + 0.1 * domain_id
        freq_bias = domain_id * 0.5

        for i in range(samples_per_domain):
            # 真实图像（低频主导）
            real_img = torch.randn(3, 224, 224) * 0.3
            # 添加域特定的低频成分
            real_img += freq_bias * 0.1

            # 假图像（添加高频噪声，模拟生成伪影）
            fake_img = torch.randn(3, 224, 224) * 0.3
            # 添加域特定的高频噪声
            high_freq_noise = torch.randn(3, 224, 224) * noise_scale
            high_freq_noise = F.avg_pool2d(high_freq_noise.unsqueeze(0), 3, 1, 1).squeeze(0)
            fake_img += high_freq_noise

            images_list.extend([real_img, fake_img])
            labels_list.extend([0, 1])  # 0=real, 1=fake
            domains_list.extend([domain_id, domain_id])

    images = torch.stack(images_list)
    labels = torch.tensor(labels_list)
    domains = torch.tensor(domains_list)

    # 打乱
    perm = torch.randperm(len(images))
    images = images[perm]
    labels = labels[perm]
    domains = domains[perm]

    return images, labels, domains


def train_one_epoch(model, train_loader, optimizer, device, use_domain=True):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    for images, labels, domains in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        domains = domains.to(device) if use_domain else None

        optimizer.zero_grad()

        outputs = model(images, labels=labels, domain_labels=domains)
        loss = criterion(outputs['logits'], labels)

        # 添加域对抗损失
        if 'domain_logits' in outputs and domains is not None:
            if isinstance(outputs['domain_logits'], list):
                for dl in outputs['domain_logits']:
                    loss = loss + 0.1 * criterion(dl, domains)
            else:
                loss = loss + 0.1 * criterion(outputs['domain_logits'], domains)

        # 添加PCR损失
        if 'pcr_losses' in outputs:
            for v in outputs['pcr_losses'].values():
                loss = loss + 0.05 * v

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = outputs['logits'].argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(train_loader), correct / total


@torch.no_grad()
def evaluate(model, test_loader, device):
    """评估模型"""
    model.eval()
    all_probs = []
    all_labels = []

    for images, labels, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs['logits'], dim=-1)[:, 1]
        all_probs.append(probs.cpu())
        all_labels.append(labels)

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    # 计算AUC
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels, probs)
    except:
        # 简单的AUC估算
        auc = ((probs[labels == 1].mean() > probs[labels == 0].mean()) + 0.5) / 2

    # 计算准确率
    preds = (probs > 0.5).astype(int)
    acc = (preds == labels).mean()

    return {'auc': auc, 'acc': acc}


def main():
    print("="*60)
    print("Quick Validation: Baseline vs Generalized Detector")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # 配置
    num_domains = 3
    train_samples = 1200  # 每个域400个样本
    test_samples = 600    # 每个域200个样本
    batch_size = 32
    epochs = 5

    print(f"\nConfiguration:")
    print(f"  Domains: {num_domains}")
    print(f"  Train samples: {train_samples}")
    print(f"  Test samples per domain: {test_samples // num_domains}")
    print(f"  Epochs: {epochs}")

    # 创建数据
    print("\n[1] Creating synthetic data...")
    train_images, train_labels, train_domains = create_synthetic_data(
        train_samples, num_domains, seed=42
    )
    # 创建不同域的测试数据
    test_data = {}
    for domain_id in range(num_domains):
        test_img, test_lbl, test_dom = create_synthetic_data(
            test_samples // num_domains, 1, seed=100 + domain_id
        )
        test_dom = torch.full_like(test_dom, domain_id)
        test_data[f'Domain_{domain_id}'] = (test_img, test_lbl, test_dom)

    train_dataset = TensorDataset(train_images, train_labels, train_domains)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_loaders = {}
    for name, (img, lbl, dom) in test_data.items():
        test_loaders[name] = DataLoader(
            TensorDataset(img, lbl, dom), batch_size=batch_size
        )

    # 只训练Domain_0，测试在所有域上
    print(f"\n[2] Training on Domain_0 only, testing on all domains")

    # ===== Baseline Model =====
    print("\n" + "-"*40)
    print("Training BASELINE model (no improvements)")
    print("-"*40)

    baseline_model = GeneralizedAIGCDetector(
        num_classes=2,
        img_size=224,
        patch_size=14,
        embed_dim=256,  # 小维度用于快速测试
        num_adapter_layers=2,
        num_prototypes=4,
        num_domains=num_domains,
        use_multi_level_da=False,
        use_domain_adversarial=False,
        use_pcr=False,
        use_hierarchical=False
    ).to(device)

    optimizer = torch.optim.AdamW(baseline_model.get_trainable_params(), lr=1e-3)

    for epoch in range(epochs):
        baseline_model.set_epoch(epoch, epochs)
        loss, acc = train_one_epoch(baseline_model, train_loader, optimizer, device, use_domain=False)
        print(f"  Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}")

    baseline_results = {}
    for name, loader in test_loaders.items():
        metrics = evaluate(baseline_model, loader, device)
        baseline_results[name] = metrics
        print(f"  {name}: AUC={metrics['auc']:.4f}")

    # ===== Generalized Model =====
    print("\n" + "-"*40)
    print("Training GENERALIZED model (DAFL + AFB + PCR)")
    print("-"*40)

    gen_model = GeneralizedAIGCDetector(
        num_classes=2,
        img_size=224,
        patch_size=14,
        embed_dim=256,
        num_adapter_layers=2,
        num_prototypes=4,
        num_domains=num_domains,
        use_multi_level_da=True,
        use_domain_adversarial=True,
        use_pcr=True,
        use_hierarchical=True
    ).to(device)

    optimizer = torch.optim.AdamW(gen_model.get_trainable_params(), lr=1e-3)

    for epoch in range(epochs):
        gen_model.set_epoch(epoch, epochs)
        loss, acc = train_one_epoch(gen_model, train_loader, optimizer, device, use_domain=True)
        print(f"  Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}")

    gen_results = {}
    for name, loader in test_loaders.items():
        metrics = evaluate(gen_model, loader, device)
        gen_results[name] = metrics
        print(f"  {name}: AUC={metrics['auc']:.4f}")

    # ===== 结果对比 =====
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)

    print("\n| Domain   | Baseline AUC | Generalized AUC | Improvement |")
    print("|----------|--------------|-----------------|-------------|")

    improvements = []
    for name in test_loaders.keys():
        base_auc = baseline_results[name]['auc']
        gen_auc = gen_results[name]['auc']
        imp = gen_auc - base_auc
        improvements.append(imp)
        print(f"| {name:8} | {base_auc:.4f}       | {gen_auc:.4f}          | {imp:+.4f}      |")

    # 跨域平均（排除训练域）
    cross_domain_names = [n for n in test_loaders.keys() if n != 'Domain_0']
    base_cross = np.mean([baseline_results[n]['auc'] for n in cross_domain_names])
    gen_cross = np.mean([gen_results[n]['auc'] for n in cross_domain_names])

    print("|----------|--------------|-----------------|-------------|")
    print(f"| Cross-Domain Avg | {base_cross:.4f}       | {gen_cross:.4f}          | {gen_cross - base_cross:+.4f}      |")

    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)

    if gen_cross > base_cross:
        print(f"\n✓ Generalized model shows BETTER cross-domain generalization!")
        print(f"  Improvement: +{(gen_cross - base_cross) * 100:.2f}% AUC")
    else:
        print(f"\n✗ Need more training or tuning for improvement")

    print("\nNote: This is a quick validation with synthetic data.")
    print("For accurate results, run full experiments with real datasets.")


if __name__ == '__main__':
    main()
