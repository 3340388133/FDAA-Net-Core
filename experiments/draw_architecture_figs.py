#!/usr/bin/env python3
"""绘制FDAA-Net创新点框架图 — 清晰流水线布局，无交叉线"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'paper_results_6src', 'paper_figures')

def save(fig, name):
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'))
    print(f'  [Saved] {name}')
    plt.close(fig)


def box(ax, x, y, w, h, text, fc, fontsize=8, tc='white', bold=False, ls='-', ec='#444444', lw=0.8, alpha=0.92):
    """圆角矩形+居中文字，返回中心坐标"""
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", facecolor=fc,
                        edgecolor=ec, linewidth=lw, alpha=alpha, linestyle=ls, zorder=3)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
            color=tc, weight='bold' if bold else 'normal', zorder=5)
    return (x + w/2, y + h/2)


def arrow(ax, x1, y1, x2, y2, color='#555555', lw=1.0, style='-|>'):
    """简单直线箭头"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw), zorder=2)


def harrow(ax, x1, y, x2, color='#555555', lw=1.0):
    """水平箭头"""
    arrow(ax, x1, y, x2, y, color, lw)


def varrow(ax, x, y1, y2, color='#555555', lw=1.0):
    """垂直箭头"""
    arrow(ax, x, y1, x, y2, color, lw)


# 颜色
C_IN = '#3498DB'
C_CLIP = '#2C3E50'
C_SRM = '#E74C3C'
C_FFT = '#C0392B'
C_FDAA = '#D35400'
C_MGFP = '#27AE60'
C_MGFP2 = '#229954'
C_OUT = '#8E44AD'
C_LOSS = '#E67E22'
C_LIGHT_R = '#FDEDEC'
C_LIGHT_G = '#EAFAF1'


# =====================================================================
# 图1: 整体架构 — 严格从左到右三行流水线
# =====================================================================
def fig_overall_architecture():
    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    ax.set_xlim(-0.5, 14.0)
    ax.set_ylim(-1.0, 6.0)
    ax.axis('off')

    # Title
    ax.text(6.5, 5.6, 'FDAA-Net: Overall Architecture', fontsize=14, ha='center',
            weight='bold', color='#2C3E50')
    ax.text(6.5, 5.2, 'Total: 323M  |  Trainable: 19.11M (5.9%)', fontsize=9, ha='center',
            color='#7F8C8D', style='italic')

    # ========== 左侧：输入 ==========
    box(ax, 0, 1.8, 1.3, 1.6, 'Input\nImage\n224x224x3', C_IN, fontsize=8, bold=True)

    # ========== 上行：CLIP通路 ==========
    box(ax, 2.2, 3.8, 2.4, 1.0, 'CLIP ViT-L/14\n(Frozen, 304M)', C_CLIP, fontsize=9, bold=True)
    box(ax, 5.4, 4.1, 1.3, 0.6, 'CLS [1, D]', '#34495E', fontsize=7)
    box(ax, 5.4, 3.3, 1.5, 0.6, 'Patches [256, D]', '#5D6D7E', fontsize=7)

    # Input → CLIP
    arrow(ax, 1.3, 3.0, 2.2, 4.0, '#555', 1.2)
    # CLIP → CLS / Patches
    harrow(ax, 4.6, 4.4, 5.4, C_CLIP, 1.0)
    harrow(ax, 4.6, 3.9, 5.4, C_CLIP, 1.0)

    # ========== 下行：FDAA通路 ==========
    fdaa_bg = FancyBboxPatch((1.9, -0.4), 5.5, 2.9, boxstyle="round,pad=0.12",
                              facecolor=C_LIGHT_R, edgecolor=C_SRM, linewidth=1.5,
                              linestyle='--', alpha=0.5, zorder=0)
    ax.add_patch(fdaa_bg)
    ax.text(4.65, 2.3, 'FDAA Module (C1) — 3.35M', fontsize=8,
            color=C_SRM, weight='bold', ha='center', style='italic')

    # SRM
    box(ax, 2.2, 1.2, 1.1, 0.8, 'SRM\n30ch', C_SRM, fontsize=7.5)
    ax.text(2.75, 1.05, 'lr x0.1', fontsize=5.5, color=C_SRM, ha='center', style='italic')
    # FFT
    box(ax, 2.2, 0.0, 1.1, 0.8, 'FFT\n6ch', C_FFT, fontsize=7.5)
    ax.text(2.75, -0.15, 'mag+phase', fontsize=5.5, color=C_FFT, ha='center', style='italic')

    # Input → SRM, FFT
    arrow(ax, 1.3, 2.6, 2.2, 1.6, '#555', 1.0)
    arrow(ax, 1.3, 2.2, 2.2, 0.4, '#555', 1.0)

    # Concat
    box(ax, 3.7, 0.5, 0.9, 0.9, 'Cat\n36ch', '#F5B7B1', fontsize=7, tc='#333')
    harrow(ax, 3.3, 1.6, 3.7, C_SRM, 0.8)
    arrow(ax, 3.3, 0.4, 3.7, 0.8, C_FFT, 0.8)

    # 4-stage CNN
    box(ax, 4.9, 0.3, 2.2, 1.2, '4-Stage CNN\n36→64→128→256→D', C_FDAA, fontsize=7.5, bold=True)
    harrow(ax, 4.6, 0.95, 4.9, '#555', 1.0)

    # FDAA输出
    ax.text(7.3, 0.3, 'Freq [1,D]', fontsize=6.5, color=C_FDAA, weight='bold', ha='center')

    # ========== 右侧：MGFP模块 ==========
    mgfp_bg = FancyBboxPatch((7.8, 0.6), 3.6, 3.8, boxstyle="round,pad=0.12",
                              facecolor=C_LIGHT_G, edgecolor=C_MGFP, linewidth=1.5,
                              linestyle='--', alpha=0.5, zorder=0)
    ax.add_patch(mgfp_bg)
    ax.text(9.6, 4.2, 'MGFP Module (C2) — 15.76M', fontsize=8,
            color=C_MGFP, weight='bold', ha='center', style='italic')

    # Cross-Attention
    box(ax, 8.0, 2.7, 3.2, 1.1, 'Cross-Attention (C3)\nQ=Freq  K,V=Patches',
        C_MGFP, fontsize=8, bold=True)

    # Hierarchical + Gated
    box(ax, 8.0, 0.8, 3.2, 1.4, 'Hierarchical Perception\nD → D//2 → D//4\nGated Fusion\n(CLS + Local + Freq)',
        C_MGFP2, fontsize=7)

    # Patches → Cross-Attention (KV)
    harrow(ax, 6.9, 3.6, 8.0, C_CLIP, 1.2)
    ax.text(7.5, 3.8, 'K, V', fontsize=7, color=C_CLIP, weight='bold', ha='center')

    # FDAA → Cross-Attention (Q)
    arrow(ax, 7.5, 1.2, 8.0, 2.7, C_FDAA, 1.2)
    ax.text(7.4, 2.1, 'Q', fontsize=8, color=C_FDAA, weight='bold', ha='left')

    # Cross-Attention → Hierarchical
    varrow(ax, 9.6, 2.7, 2.2, C_MGFP, 1.0)

    # CLS → Gated Fusion (斜线到Hierarchical右侧)
    arrow(ax, 6.7, 4.4, 11.0, 1.8, C_CLIP, 0.7)
    ax.text(9.5, 3.65, 'CLS', fontsize=6, color=C_CLIP, ha='center', style='italic')

    # ========== 最右：输出 ==========
    box(ax, 12.0, 1.8, 1.5, 1.2, 'Classifier\nReal / Fake', C_OUT, fontsize=9, bold=True)
    harrow(ax, 11.2, 1.5, 12.0, C_MGFP, 1.3)
    arrow(ax, 12.0, 1.5, 12.0, 1.8, C_MGFP, 1.3)

    # Loss
    box(ax, 11.8, 0.0, 1.8, 1.0, 'Loss\nFocal + Contr + Aux', C_LOSS, fontsize=7)
    varrow(ax, 12.7, 1.8, 1.0, C_LOSS, 0.8)

    # Aux classifier
    box(ax, 7.3, -0.6, 1.0, 0.6, 'Aux Head', C_LOSS, fontsize=6.5)
    varrow(ax, 7.8, 0.3, -0.6, C_LOSS, 0.7)
    harrow(ax, 8.3, -0.3, 12.0, C_LOSS, 0.6)

    save(fig, 'fig_architecture_overall')


# =====================================================================
# 图2: FDAA模块详图 — 纯水平流水线
# =====================================================================
def fig_fdaa_detail():
    fig, ax = plt.subplots(figsize=(8.0, 2.8))
    ax.set_xlim(-0.3, 11.0)
    ax.set_ylim(-0.5, 3.2)
    ax.axis('off')

    ax.text(5.3, 2.9, 'FDAA: Frequency-Domain Artifact Analysis Module', fontsize=11,
            ha='center', weight='bold', color='#2C3E50')

    # Input
    box(ax, 0, 0.4, 1.0, 1.6, 'Input\n224x224\nx3', C_IN, fontsize=7.5, bold=True)

    # SRM (上行)
    box(ax, 1.6, 1.4, 1.3, 0.8, 'SRM Filters\n30 kernels (5x5)', C_SRM, fontsize=7)
    ax.text(2.25, 1.25, 'lr x 0.1', fontsize=5.5, color=C_SRM, ha='center', style='italic')

    # FFT (下行)
    box(ax, 1.6, 0.2, 1.3, 0.8, 'Real FFT\nMag + Phase', C_FFT, fontsize=7)

    # Input → SRM / FFT
    arrow(ax, 1.0, 1.6, 1.6, 1.8, '#555', 1.0)
    arrow(ax, 1.0, 0.8, 1.6, 0.6, '#555', 1.0)

    # SRM out
    box(ax, 3.3, 1.4, 0.8, 0.8, '30ch', '#F1948A', fontsize=7, tc='#333')
    harrow(ax, 2.9, 1.8, 3.3, C_SRM, 0.8)

    # FFT out
    box(ax, 3.3, 0.2, 0.8, 0.8, '6ch', '#F1948A', fontsize=7, tc='#333')
    harrow(ax, 2.9, 0.6, 3.3, C_FFT, 0.8)

    # Concat
    box(ax, 4.5, 0.6, 0.8, 1.2, 'Concat\n36ch', '#E8DAEF', fontsize=7, tc='#333')
    arrow(ax, 4.1, 1.6, 4.5, 1.4, '#555', 0.8)
    arrow(ax, 4.1, 0.6, 4.5, 0.9, '#555', 0.8)

    # 4 Stages 水平排列
    stage_info = [('S1\n36→64', C_SRM), ('S2\n64→128', '#D35400'), ('S3\n128→256', C_FFT), ('S4\n256→D', '#922B21')]
    for i, (txt, c) in enumerate(stage_info):
        sx = 5.8 + i * 1.1
        box(ax, sx, 0.65, 0.9, 1.1, txt, c, fontsize=7)
        if i == 0:
            harrow(ax, 5.3, 1.2, sx, '#555', 1.0)
        else:
            harrow(ax, sx - 0.2, 1.2, sx, '#555', 0.8)

    # Output
    box(ax, 10.0, 0.7, 0.8, 1.0, 'Freq\n[1,D]\nD=1024', C_OUT, fontsize=7, bold=True)
    harrow(ax, 9.5, 1.2, 10.0, '#555', 1.0)

    # 标注
    ax.annotate('stride=2 each stage\n(no stride-4 gaps)', xy=(7.4, 0.55), xytext=(7.4, -0.2),
                fontsize=7, ha='center', color=C_SRM, style='italic',
                arrowprops=dict(arrowstyle='->', color=C_SRM, lw=0.6))

    save(fig, 'fig_architecture_fdaa')


# =====================================================================
# 图3: MGFP模块详图 — 从左到右水平流
# =====================================================================
def fig_mgfp_detail():
    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    ax.set_xlim(-0.3, 11.5)
    ax.set_ylim(-0.5, 4.0)
    ax.axis('off')

    ax.text(5.7, 3.7, 'MGFP: Multi-Granularity Fusion Perception Module', fontsize=11,
            ha='center', weight='bold', color='#2C3E50')

    # 左侧两个输入 (上下排列)
    box(ax, 0, 2.2, 1.4, 0.9, 'FDAA Output\nFreq [1, D]', C_FDAA, fontsize=7.5, bold=True)
    box(ax, 0, 0.9, 1.4, 0.9, 'CLIP Patches\n[256, D]', C_CLIP, fontsize=7.5, bold=True)
    box(ax, 0, -0.2, 1.4, 0.7, 'CLIP CLS\n[1, D]', '#34495E', fontsize=7)

    # Linear projections
    box(ax, 1.9, 2.3, 0.9, 0.7, 'Linear\n→ Q', '#F5B7B1', fontsize=7, tc='#333')
    box(ax, 1.9, 1.0, 0.9, 0.7, 'Linear\n→ K, V', '#AED6F1', fontsize=7, tc='#333')

    harrow(ax, 1.4, 2.65, 1.9, C_FDAA, 1.0)
    harrow(ax, 1.4, 1.35, 1.9, C_CLIP, 1.0)

    # Cross-Attention 大框
    ca_bg = FancyBboxPatch((3.2, 0.6), 2.6, 2.6, boxstyle="round,pad=0.1",
                            facecolor=C_LIGHT_G, edgecolor=C_MGFP, linewidth=1.5,
                            linestyle='--', alpha=0.5, zorder=0)
    ax.add_patch(ca_bg)
    ax.text(4.5, 3.0, 'Cross-Attention (C3)', fontsize=8.5, color=C_MGFP,
            weight='bold', ha='center', style='italic')

    box(ax, 3.4, 1.8, 1.0, 0.8, 'Q x K^T\n/ sqrt(d)', C_MGFP, fontsize=7)
    box(ax, 4.6, 1.8, 1.0, 0.8, 'Softmax\nx V', C_MGFP2, fontsize=7)
    box(ax, 3.8, 0.8, 1.6, 0.7, 'Freq-Guided\nSemantic [1,D]', C_MGFP, fontsize=7, bold=True)

    # Q → QK^T
    harrow(ax, 2.8, 2.65, 3.4, C_FDAA, 0.8)
    # K → QK^T
    arrow(ax, 2.8, 1.35, 3.4, 2.1, C_CLIP, 0.8)
    # V → Softmax x V
    arrow(ax, 2.8, 1.5, 4.6, 2.0, C_CLIP, 0.6)
    # QK^T → Softmax
    harrow(ax, 4.4, 2.2, 4.6, C_MGFP, 0.8)
    # Softmax → output
    varrow(ax, 5.1, 1.8, 1.5, C_MGFP, 0.8)
    arrow(ax, 5.1, 1.5, 5.4, 1.15, C_MGFP, 0.8)

    # Hierarchical Perception
    box(ax, 6.2, 1.3, 2.0, 1.4, 'Hierarchical\nForgery\nPerception\nD→D//2→D//4', C_MGFP2, fontsize=7.5)

    harrow(ax, 5.8, 1.15, 6.2, C_MGFP, 1.0)
    arrow(ax, 5.8, 1.15, 6.2, 1.8, C_MGFP, 1.0)

    # Gated Fusion
    box(ax, 8.7, 0.8, 1.3, 1.8, 'Gated\nFusion\n\nCLS\n+ Local\n+ Freq', '#1ABC9C', fontsize=7)

    harrow(ax, 8.2, 2.0, 8.7, C_MGFP2, 1.0)
    # CLS → Gated Fusion (底部水平线)
    harrow(ax, 1.4, 0.15, 8.7, '#34495E', 0.7)
    arrow(ax, 8.7, 0.15, 8.7, 0.8, '#34495E', 0.7)
    # FDAA freq_global → Gated Fusion (顶部水平线)
    ax.plot([1.4, 1.6], [2.65, 3.3], '-', color=C_FDAA, lw=0.6, alpha=0.6)
    ax.plot([1.6, 8.7], [3.3, 3.3], '-', color=C_FDAA, lw=0.6, alpha=0.6)
    arrow(ax, 8.7, 3.3, 8.7, 2.6, C_FDAA, 0.7)
    ax.text(5.0, 3.45, 'Freq Global [1,D]', fontsize=6, color=C_FDAA, ha='center', style='italic')

    # Output
    box(ax, 10.3, 1.2, 1.0, 1.0, 'Fused\nFeature\n[1, D]', C_OUT, fontsize=7.5, bold=True)
    harrow(ax, 10.0, 1.7, 10.3, '#555', 1.2)

    save(fig, 'fig_architecture_mgfp')


# =====================================================================
# 图4: 方法对比 — 频域特征覆盖范围
# =====================================================================
def fig_method_comparison():
    fig, axes = plt.subplots(1, 4, figsize=(8.5, 3.0))

    methods = [
        ('F3Net\n(DCT only)', [('DCT Blocks', '#3498DB')],
         'Single local\nfrequency', False),
        ('FreqNet\n(FFT mag only)', [('FFT Magnitude', '#2980B9')],
         'Single global\nno phase info', False),
        ('UnivFD\n(No frequency)', [('CLIP Only', '#95A5A6')],
         'No frequency\nanalysis at all', False),
        ('FDAA-Net (Ours)', [('SRM 30ch', '#E74C3C'), ('FFT 6ch', '#C0392B'), ('CLIP 256p', '#2C3E50')],
         'Multi-source\ncomplementary', True),
    ]

    for ax, (title, blocks, desc, is_ours) in zip(axes, methods):
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4.5)
        ax.axis('off')

        # 标题
        weight = 'bold'
        color = '#C0392B' if is_ours else '#2C3E50'
        ax.text(2.0, 4.2, title, fontsize=7.5, ha='center', weight=weight, color=color)

        # 块
        n = len(blocks)
        total_w = min(3.2, n * 1.1)
        gap = 0.1
        bw = (total_w - (n-1)*gap) / n
        start_x = (4 - total_w) / 2
        for i, (text, c) in enumerate(blocks):
            bx = start_x + i * (bw + gap)
            b = FancyBboxPatch((bx, 1.5), bw, 1.8, boxstyle="round,pad=0.06",
                               facecolor=c, edgecolor='#444', linewidth=0.6, alpha=0.85, zorder=3)
            ax.add_patch(b)
            ax.text(bx + bw/2, 2.4, text, ha='center', va='center', fontsize=6.5,
                    color='white', weight='bold', zorder=5)

        # 描述
        dc = '#C0392B' if is_ours else '#777777'
        dw = 'bold' if is_ours else 'normal'
        ax.text(2.0, 0.7, desc, ha='center', fontsize=7, color=dc, style='italic', weight=dw)

        # Ours边框
        if is_ours:
            rect = FancyBboxPatch((0.15, 0.2), 3.7, 3.8, boxstyle="round,pad=0.1",
                                  facecolor='none', edgecolor='#C0392B', linewidth=1.8, zorder=0)
            ax.add_patch(rect)

    plt.tight_layout(w_pad=0.5)
    save(fig, 'fig_method_comparison')


# =====================================================================
# 图5: 模块贡献分析柱状图
# =====================================================================
def fig_contribution_analysis():
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    modules = ['FDAA\n(C1)', 'MGFP\n(C2)', 'Full\n(C1+C2+C3)']
    domain_delta = [0.20, 0.38, 0.38]
    robust_delta = [1.06, 2.87, 3.26]

    x = np.arange(len(modules))
    w = 0.3

    bars1 = ax.bar(x - w/2, domain_delta, w, label='In-domain ΔAUC', color='#3498DB',
                   edgecolor='white', alpha=0.85, zorder=3)
    bars2 = ax.bar(x + w/2, robust_delta, w, label='Robustness ΔAUC', color='#E74C3C',
                   edgecolor='white', alpha=0.85, zorder=3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'+{bar.get_height():.2f}', ha='center', fontsize=7.5, color='#3498DB', weight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'+{bar.get_height():.2f}', ha='center', fontsize=7.5, color='#E74C3C', weight='bold')

    ratios = [5.3, 7.6, 8.6]
    for i, r in enumerate(ratios):
        ax.annotate(f'{r:.1f}x', xy=(x[i] + w/2, robust_delta[i]),
                    xytext=(x[i] + w/2 + 0.3, robust_delta[i] + 0.3),
                    fontsize=8, color='#C0392B', weight='bold',
                    arrowprops=dict(arrowstyle='->', color='#C0392B', lw=0.7))

    ax.set_xticks(x)
    ax.set_xticklabels(modules, fontsize=9)
    ax.set_ylabel('ΔAUC (pp)', fontsize=10)
    ax.set_title('Module Contribution: In-domain vs Robustness', fontsize=10, weight='bold', pad=8)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_ylim(0, 4.2)

    ax.text(0.5, -0.15, 'Robustness contribution is consistently 5-9x larger than in-domain contribution',
            fontsize=7.5, color='#555555', style='italic', ha='center', transform=ax.transAxes)

    plt.tight_layout()
    save(fig, 'fig_contribution_analysis')


# =====================================================================
# 图6: 融合方式对比 — 4种方式的简洁对比
# =====================================================================
def fig_fusion_comparison():
    fig, axes = plt.subplots(1, 4, figsize=(8.5, 3.0))

    fusions = [
        ('(a) No Fusion\nUnivFD',
         [('CLIP CLS', '#2C3E50', 2.5)],
         'Classifier',
         'Only global CLS\nNo local / No freq'),

        ('(b) Early Fusion\nConcat',
         [('CLIP', '#2C3E50', 1.6), ('Freq', '#E74C3C', 1.6)],
         'Concat + FC',
         'Simple concatenation\nNo cross-modal interaction'),

        ('(c) Late Fusion\nScore Ensemble',
         [('CLIP score', '#2C3E50', 1.6), ('Freq score', '#E74C3C', 1.6)],
         'Weighted Sum',
         'Score-level only\nNo feature interaction'),

        ('(d) Cross-Attention\nOurs (C3)',
         [('Q = Freq', '#E74C3C', 1.6), ('K,V = Patches', '#2C3E50', 1.6)],
         'Guided Agg.',
         'Freq guides semantic\nDeep cross-modal fusion'),
    ]

    for ax, (title, inputs, output, desc) in zip(axes, fusions):
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 5.0)
        ax.axis('off')

        is_ours = 'Ours' in title
        tc = '#C0392B' if is_ours else '#2C3E50'
        ax.text(2.0, 4.7, title, fontsize=7, ha='center', weight='bold', color=tc)

        # Input blocks (上方)
        n = len(inputs)
        for i, (text, color, bw) in enumerate(inputs):
            y = 3.3 - i * 0.9
            bx = (4 - bw) / 2
            b = FancyBboxPatch((bx, y), bw, 0.7, boxstyle="round,pad=0.05",
                               facecolor=color, edgecolor='#444', linewidth=0.5, alpha=0.8, zorder=3)
            ax.add_patch(b)
            ax.text(2.0, y + 0.35, text, ha='center', va='center', fontsize=6.5,
                    color='white', weight='bold', zorder=5)

        # 箭头
        ax.annotate('', xy=(2.0, 1.8), xytext=(2.0, 2.3),
                    arrowprops=dict(arrowstyle='-|>', color='#555', lw=0.8))

        # Output
        ax.text(2.0, 1.5, output, ha='center', fontsize=7.5, color='#333', weight='bold')

        # Description (底部)
        dc = '#C0392B' if is_ours else '#777777'
        dw = 'bold' if is_ours else 'normal'
        ax.text(2.0, 0.5, desc, ha='center', fontsize=6.5, color=dc, style='italic', weight=dw)

        if is_ours:
            rect = FancyBboxPatch((0.1, 0.1), 3.8, 4.5, boxstyle="round,pad=0.1",
                                  facecolor='none', edgecolor='#C0392B', linewidth=1.8, zorder=0)
            ax.add_patch(rect)

    plt.tight_layout(w_pad=0.3)
    save(fig, 'fig_fusion_comparison')


if __name__ == '__main__':
    print("=" * 50)
    print("Drawing architecture figures (clean layout)...")
    print("=" * 50)
    fig_overall_architecture()
    fig_fdaa_detail()
    fig_mgfp_detail()
    fig_method_comparison()
    fig_contribution_analysis()
    fig_fusion_comparison()
    print("\n[Done] 6 figures generated")
