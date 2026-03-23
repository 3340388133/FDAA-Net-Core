#!/usr/bin/env python3
"""修复消融综合大图：上下间距加大，标注精简"""
import os, json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.08,
    'axes.linewidth': 0.6,
})

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'paper_results_6src', 'paper_figures')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'paper_results_6src', 'results')

with open(os.path.join(RESULTS_DIR, 'ablation_robustness_results.json')) as f:
    rob = json.load(f)

# ---- 布局：上面两个柱状图占1份高度，下面热力图占1.2份，中间大间距 ----
fig = plt.figure(figsize=(7.5, 6.5))
gs = GridSpec(2, 2, height_ratios=[1, 1.2], hspace=0.55, wspace=0.4)

variants_short = ['Baseline', '+FDAA', '+MGFP', 'Full']
colors = ['#BDC3C7', '#3498DB', '#2ECC71', '#C0392B']

# ===== (a) 域内消融柱状图 =====
ax_a = fig.add_subplot(gs[0, 0])
domain_auc = [99.49, 99.69, 99.87, 99.87]
ax_a.bar(variants_short, domain_auc, color=colors, edgecolor='white', width=0.55, zorder=3, alpha=0.88)
ax_a.set_ylim(99.2, 100.08)
ax_a.set_ylabel('AUC (%)')
ax_a.set_title('(a) In-domain Detection', fontsize=10, pad=6)
ax_a.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
ax_a.set_axisbelow(True)
ax_a.yaxis.grid(True, linestyle='--', alpha=0.3)
for i, auc in enumerate(domain_auc):
    ax_a.text(i, auc + 0.015, f'{auc:.2f}', ha='center', fontsize=7.5, weight='bold')
# FDAA标注
ax_a.annotate('+0.20', xy=(1, 99.71), xytext=(0.3, 99.95),
              fontsize=7.5, ha='center', color='#3498DB', weight='bold',
              arrowprops=dict(arrowstyle='->', color='#3498DB', lw=0.7))

# ===== (b) 鲁棒性消融柱状图 =====
ax_b = fig.add_subplot(gs[0, 1])
rob_avg = [94.53, 95.59, 97.40, 97.79]
ax_b.bar(variants_short, rob_avg, color=colors, edgecolor='white', width=0.55, zorder=3, alpha=0.88)
ax_b.set_ylim(93.5, 99.0)
ax_b.set_ylabel('Avg Robustness AUC (%)')
ax_b.set_title('(b) Robustness (7 Perturbations Avg)', fontsize=10, pad=6)
ax_b.yaxis.set_major_locator(mticker.MultipleLocator(1))
ax_b.set_axisbelow(True)
ax_b.yaxis.grid(True, linestyle='--', alpha=0.3)
for i, auc in enumerate(rob_avg):
    ax_b.text(i, auc + 0.08, f'{auc:.1f}', ha='center', fontsize=7.5, weight='bold')
# FDAA鲁棒性标注
ax_b.annotate('+1.06 (5.3x)', xy=(1, 95.7), xytext=(1.8, 96.6),
              fontsize=7.5, ha='center', color='#C0392B', weight='bold',
              arrowprops=dict(arrowstyle='->', color='#C0392B', lw=0.7))

# ===== (c) 消融鲁棒性热力图（跨两列） =====
ax_c = fig.add_subplot(gs[1, :])

abl_variants = ['abl_baseline', 'abl_baseline+fdaa', 'abl_baseline+mgfp', 'abl_full']
abl_labels = ['Baseline (CLIP+CLS)', '+ FDAA', '+ MGFP', 'Full (Ours)']
tests = ['jpeg_70', 'jpeg_50', 'jpeg_30', 'blur_1.0', 'blur_2.0', 'noise_0.02', 'noise_0.05']
test_labels = ['JPEG\nQ=70', 'JPEG\nQ=50', 'JPEG\nQ=30', 'Blur\n$\\sigma$=1.0',
               'Blur\n$\\sigma$=2.0', 'Noise\n$\\sigma$=.02', 'Noise\n$\\sigma$=.05']

data_arr = np.array([[rob[t][v]['auc'] * 100 for t in tests] for v in abl_variants])
avgs = data_arr.mean(axis=1, keepdims=True)
data_ext = np.hstack([data_arr, avgs])
col_labels = test_labels + ['Avg']

im = ax_c.imshow(data_ext, cmap='RdYlGn', aspect='auto', vmin=88, vmax=100)
ax_c.set_xticks(range(len(col_labels)))
ax_c.set_xticklabels(col_labels, fontsize=8)
ax_c.set_yticks(range(len(abl_labels)))
ax_c.set_yticklabels(abl_labels, fontsize=8.5)
ax_c.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False)
ax_c.axvline(x=6.5, color='black', linewidth=1.0)
ax_c.set_title('(c) Module Ablation — Robustness Details (AUC %)', fontsize=10, pad=18)

for i in range(len(abl_variants)):
    for j in range(len(col_labels)):
        v = data_ext[i, j]
        is_best = (i == np.argmax(data_ext[:, j]))
        is_avg = (j == len(col_labels) - 1)
        color = 'white' if v < 92 else 'black'
        w = 'bold' if is_best or is_avg else 'normal'
        ax_c.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=7.5, color=color, weight=w)

# Δ标注
for i in range(1, len(abl_labels)):
    delta = avgs[i, 0] - avgs[0, 0]
    ax_c.text(len(col_labels) - 0.4, i, f' $\\Delta$+{delta:.1f}',
              ha='left', va='center', fontsize=8, color='#C0392B', weight='bold')

cbar = fig.colorbar(im, ax=ax_c, shrink=0.75, pad=0.1, aspect=12)
cbar.set_label('AUC (%)', fontsize=8)
cbar.ax.tick_params(labelsize=7)

for ext in ['pdf', 'png']:
    fig.savefig(os.path.join(FIG_DIR, f'fig_combo_ablation.{ext}'))
print('[Saved] fig_combo_ablation')
plt.close(fig)
