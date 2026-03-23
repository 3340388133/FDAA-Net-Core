#!/usr/bin/env python3
"""
修复3张有问题的图:
  fig3 消融柱状图 → 改为横向热力图（更清晰）
  fig6 散点图     → 用adjustText避免标签重叠，放大拥挤区域
  fig8 SOTA柱状图 → 改为横向热力图（替代拥挤的柱状图）
"""
import os, json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.6,
})

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'paper_results_6src', 'paper_figures')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'paper_results_6src', 'results')

def save(fig, name):
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'))
    print(f'  [Saved] {name}')
    plt.close(fig)


# =====================================================================
# Fig 3: 消融鲁棒性 → 热力图（替代拥挤的分组柱状图）
# =====================================================================
def fix_fig3():
    with open(os.path.join(RESULTS_DIR, 'ablation_robustness_results.json')) as f:
        rob = json.load(f)

    variants = ['abl_baseline', 'abl_baseline+fdaa', 'abl_baseline+mgfp', 'abl_full']
    labels = ['Baseline\n(CLIP+CLS)', '+ FDAA', '+ MGFP', 'Full\n(Ours)']
    tests = ['jpeg_70', 'jpeg_50', 'jpeg_30', 'blur_1.0', 'blur_2.0', 'noise_0.02', 'noise_0.05']
    test_labels = ['JPEG Q=70', 'JPEG Q=50', 'JPEG Q=30', 'Blur σ=1.0', 'Blur σ=2.0', 'Noise σ=.02', 'Noise σ=.05']

    # 构建矩阵
    data = np.array([[rob[t][v]['auc'] * 100 for t in tests] for v in variants])

    # 添加Avg列
    avgs = data.mean(axis=1, keepdims=True)
    data_ext = np.hstack([data, avgs])
    col_labels = test_labels + ['Avg']

    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    im = ax.imshow(data_ext, cmap='RdYlGn', aspect='auto', vmin=88, vmax=100)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=7.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False)

    # 数值标注
    for i in range(len(variants)):
        for j in range(len(col_labels)):
            v = data_ext[i, j]
            is_best = (i == np.argmax(data_ext[:, j]))
            is_avg = (j == len(col_labels) - 1)
            weight = 'bold' if is_best or is_avg else 'normal'
            color = 'white' if v < 92 else 'black'
            ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                    fontsize=7.5 if is_avg else 7, color=color, weight=weight)

    # Avg列加竖线分隔
    ax.axvline(x=6.5, color='black', linewidth=1.0)

    # ΔAUC标注在右侧
    for i, label in enumerate(labels):
        delta = avgs[i, 0] - avgs[0, 0]
        if i > 0:
            ax.text(len(col_labels) - 0.5, i, f'  Δ+{delta:.1f}',
                    ha='left', va='center', fontsize=7, color='#C0392B', weight='bold')

    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.12, aspect=15)
    cbar.set_label('AUC (%)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    save(fig, 'fig3_ablation_robustness_grouped_bar')


# =====================================================================
# Fig 6: 散点图 → 双面板（全局 + 拥挤区域放大）
# =====================================================================
def fix_fig6():
    pts = {
        'FDAA-Net (Ours)': (99.93, 97.79, 19.11, '#C0392B'),
        'F3Net':            (99.75, 94.47, 5.18,  '#2980B9'),
        'CNNDet':           (99.40, 94.08, 23.51, '#27AE60'),
        'FreqNet':          (99.07, 93.86, 23.38, '#8E44AD'),
        'UnivFD':           (99.13, 93.47, 0.39,  '#E67E22'),
        'DIRE':             (99.45, 92.73, 23.53, '#1ABC9C'),
        'LaRE²':            (98.17, 92.07, 37.96, '#95A5A6'),
        'NPR':              (97.47, 90.92, 11.37, '#34495E'),
        'C2P-CLIP':         (89.79, 85.90, 18.37, '#BDC3C7'),
        'SPEC':             (97.12, 79.29, 1.55,  '#7F8C8D'),
    }

    # 手动指定每个标签的偏移方向，避免重叠
    offsets = {
        'FDAA-Net (Ours)': (-55, 8),
        'F3Net':            (8, 6),
        'CNNDet':           (8, -12),
        'FreqNet':          (-55, -8),
        'UnivFD':           (8, -12),
        'DIRE':             (8, 6),
        'LaRE²':            (-45, -10),
        'NPR':              (8, 6),
        'C2P-CLIP':         (8, 6),
        'SPEC':             (8, -8),
    }

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    for name, (dom, rob, par, color) in pts.items():
        is_ours = 'Ours' in name
        s = max(par * 3.5, 40)
        ax.scatter(dom, rob, s=s, c=color,
                   alpha=1.0 if is_ours else 0.7, zorder=10 if is_ours else 5,
                   edgecolor='#C0392B' if is_ours else '#333333',
                   linewidth=1.5 if is_ours else 0.4)

        ox, oy = offsets[name]
        ax.annotate(name, (dom, rob),
                    textcoords='offset points', xytext=(ox, oy),
                    fontsize=7.5 if is_ours else 7,
                    weight='bold' if is_ours else 'normal',
                    color=color if is_ours else '#333333',
                    arrowprops=dict(arrowstyle='-', color='#AAAAAA', lw=0.4,
                                   connectionstyle='arc3,rad=0.1') if not is_ours else None)

    ax.set_xlabel('In-domain AUC (%)', fontsize=9)
    ax.set_ylabel('Avg Robustness AUC (%)', fontsize=9)
    ax.set_xlim(88, 101)
    ax.set_ylim(77, 100)
    ax.set_axisbelow(True)
    ax.grid(True, linestyle='--', alpha=0.25)

    # Pareto标注区域
    ax.fill_between([99.93, 101], 97.79, 100, alpha=0.06, color='#C0392B', zorder=0)
    ax.axhline(y=97.79, color='#C0392B', linestyle=':', linewidth=0.5, alpha=0.3)
    ax.axvline(x=99.93, color='#C0392B', linestyle=':', linewidth=0.5, alpha=0.3)

    # 拥挤区域标注框
    rect_x = [98.8, 100.2]
    rect_y = [92.0, 95.2]
    from matplotlib.patches import Rectangle
    rect = Rectangle((rect_x[0], rect_y[0]), rect_x[1]-rect_x[0], rect_y[1]-rect_y[0],
                      linewidth=0.8, edgecolor='#666666', facecolor='none', linestyle='--', zorder=8)
    ax.add_patch(rect)
    ax.text(rect_x[0] - 0.15, rect_y[0] - 0.8, 'Dense region', fontsize=6, color='#666666', style='italic')

    plt.tight_layout()
    save(fig, 'fig6_performance_robustness_scatter')


# =====================================================================
# Fig 8: SOTA鲁棒性 → 热力图（替代拥挤柱状图）
# =====================================================================
def fix_fig8():
    methods = ['FDAA-Net (Ours)', 'F3Net', 'CNNDet', 'FreqNet', 'UnivFD',
               'DIRE', 'LaRE²', 'NPR', 'C2P-CLIP', 'SPEC']
    tests = ['JPEG Q=70', 'JPEG Q=50', 'JPEG Q=30', 'Blur σ=1.0', 'Blur σ=2.0', 'Noise σ=.02', 'Noise σ=.05']

    data = np.array([
        [99.01, 98.24, 96.33, 99.39, 94.53, 99.08, 97.97],
        [97.04, 95.94, 93.66, 98.96, 89.39, 96.78, 89.49],
        [96.54, 95.65, 93.96, 98.42, 85.66, 96.23, 92.08],
        [96.61, 95.59, 93.40, 97.28, 84.92, 96.17, 93.08],
        [95.18, 93.79, 90.98, 97.04, 88.08, 96.30, 92.95],
        [96.28, 94.81, 92.69, 98.39, 84.94, 94.89, 87.14],
        [95.16, 94.08, 92.28, 95.62, 83.48, 94.60, 89.26],
        [94.84, 93.77, 91.53, 93.55, 81.63, 92.90, 88.23],
        [88.39, 87.52, 86.25, 84.55, 79.40, 89.13, 86.04],
        [81.38, 79.79, 77.43, 89.18, 64.29, 83.69, 79.27],
    ])

    # 添加Avg列
    avgs = data.mean(axis=1, keepdims=True)
    data_ext = np.hstack([data, avgs])
    col_labels = tests + ['Avg']

    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    # 自定义colormap: 红→黄→绿
    im = ax.imshow(data_ext, cmap='RdYlGn', aspect='auto', vmin=62, vmax=100)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=7.5)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=8)
    ax.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False)

    # Avg列竖线
    ax.axvline(x=6.5, color='black', linewidth=1.0)

    # 第一行(Ours)与其他行分隔线
    ax.axhline(y=0.5, color='#C0392B', linewidth=1.2)

    # 数值标注
    for i in range(len(methods)):
        for j in range(len(col_labels)):
            v = data_ext[i, j]
            is_best = (i == np.argmax(data_ext[:, j]))
            is_avg = (j == len(col_labels) - 1)
            weight = 'bold' if is_best else 'normal'
            color = 'white' if v < 82 else 'black'
            fontsize = 7 if not is_avg else 7.5
            text = f'{v:.1f}'
            if is_best and not is_avg:
                text = f'\\textbf{{{v:.1f}}}'
                text = f'{v:.1f}'  # bold via weight
            ax.text(j, i, text, ha='center', va='center',
                    fontsize=fontsize, color=color, weight=weight)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, aspect=20)
    cbar.set_label('AUC (%)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    save(fig, 'fig8_robustness_sota_bar')


if __name__ == '__main__':
    print("Fixing 3 figures...")
    fix_fig3()
    fix_fig6()
    fix_fig8()
    print("[Done]")
