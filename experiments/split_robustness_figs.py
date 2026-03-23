#!/usr/bin/env python3
"""把鲁棒性综合大图拆成两张独立图：衰减折线图 + SOTA热力图"""
import os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.08,
    'axes.linewidth': 0.6,
})

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'paper_results_6src', 'paper_figures')

def save(fig, name):
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'))
    print(f'  [Saved] {name}')
    plt.close(fig)


# =====================================================================
# 图1: 鲁棒性衰减折线图 (3子图横排)
# =====================================================================
def fig_degradation_curves():
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8), sharey=True)

    methods = {
        'FDAA-Net (Ours)': {'c': '#C0392B', 'ls': '-',  'm': 'o', 'lw': 1.8},
        'F3Net':            {'c': '#2980B9', 'ls': '--', 'm': 's', 'lw': 1.0},
        'CNNDetection':     {'c': '#27AE60', 'ls': '--', 'm': '^', 'lw': 1.0},
        'FreqNet':          {'c': '#8E44AD', 'ls': '--', 'm': 'D', 'lw': 1.0},
        'UnivFD':           {'c': '#E67E22', 'ls': '--', 'm': 'v', 'lw': 1.0},
    }
    clean = {'FDAA-Net (Ours)': 99.93, 'F3Net': 99.75, 'CNNDetection': 99.40,
             'FreqNet': 99.07, 'UnivFD': 99.13}

    jpeg = {'FDAA-Net (Ours)': [99.01,98.24,96.33], 'F3Net': [97.04,95.94,93.66],
            'CNNDetection': [96.54,95.65,93.96], 'FreqNet': [96.61,95.59,93.40],
            'UnivFD': [95.18,93.79,90.98]}
    blur = {'FDAA-Net (Ours)': [99.39,94.53], 'F3Net': [98.96,89.39],
            'CNNDetection': [98.42,85.66], 'FreqNet': [97.28,84.92], 'UnivFD': [97.04,88.08]}
    noise = {'FDAA-Net (Ours)': [99.08,97.97], 'F3Net': [96.78,89.49],
             'CNNDetection': [96.23,92.08], 'FreqNet': [96.17,93.08], 'UnivFD': [96.30,92.95]}

    datasets = [
        ('(a) JPEG Compression', ['Clean','Q=70','Q=50','Q=30'], jpeg),
        ('(b) Gaussian Blur', ['Clean',r'$\sigma$=1.0',r'$\sigma$=2.0'], blur),
        ('(c) Gaussian Noise', ['Clean',r'$\sigma$=0.02',r'$\sigma$=0.05'], noise),
    ]

    for ax, (title, xlabels, data) in zip(axes, datasets):
        ax.set_title(title, fontsize=10, pad=6)
        for name, cfg in methods.items():
            vals = [clean[name]] + data[name]
            ax.plot(xlabels, vals, color=cfg['c'], linestyle=cfg['ls'],
                    marker=cfg['m'], markersize=4.5, markeredgewidth=0.4,
                    markeredgecolor='white', label=name, linewidth=cfg['lw'],
                    zorder=10 if 'Ours' in name else 5)
        ax.set_ylim(83, 101)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(axis='x', labelsize=7.5)

    axes[0].set_ylabel('AUC (%)')

    # Blur图标注Ours与次优的差距
    axes[1].annotate('', xy=(2, 94.53), xytext=(2, 89.39),
                     arrowprops=dict(arrowstyle='<->', color='#C0392B', lw=0.8))
    axes[1].text(2.15, 91.6, r'$\Delta$5.1', fontsize=7.5, color='#C0392B', weight='bold')

    # 图例放在顶部
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08),
               ncol=5, frameon=True, fancybox=False, edgecolor='#CCCCCC',
               columnspacing=1.2, handletextpad=0.4, fontsize=8)

    plt.tight_layout()
    save(fig, 'fig_robustness_degradation_curves')


# =====================================================================
# 图2: SOTA鲁棒性热力图
# =====================================================================
def fig_sota_robustness_heatmap():
    method_names = ['FDAA-Net (Ours)', 'F3Net', 'CNNDet', 'FreqNet', 'UnivFD',
                    'DIRE', 'LaRE²', 'NPR', 'C2P-CLIP', 'SPEC']
    test_labels = ['JPEG\nQ=70', 'JPEG\nQ=50', 'JPEG\nQ=30', 'Blur\n$\\sigma$=1.0',
                   'Blur\n$\\sigma$=2.0', 'Noise\n$\\sigma$=.02', 'Noise\n$\\sigma$=.05']
    data_arr = np.array([
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

    avgs = data_arr.mean(axis=1, keepdims=True)
    data_ext = np.hstack([data_arr, avgs])
    col_labels = test_labels + ['Avg']

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    im = ax.imshow(data_ext, cmap='RdYlGn', aspect='auto', vmin=62, vmax=100)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(method_names)))
    ax.set_yticklabels(method_names, fontsize=8.5)
    ax.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False)

    # Avg列竖线
    ax.axvline(x=6.5, color='black', linewidth=1.0)
    # Ours与其他行分隔线
    ax.axhline(y=0.5, color='#C0392B', linewidth=1.2)

    # 数值标注
    for i in range(len(method_names)):
        for j in range(len(col_labels)):
            v = data_ext[i, j]
            is_best = (i == np.argmax(data_ext[:, j]))
            is_avg = (j == len(col_labels) - 1)
            color = 'white' if v < 82 else 'black'
            w = 'bold' if is_best or is_avg else 'normal'
            fs = 7.5 if is_avg else 7
            ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                    fontsize=fs, color=color, weight=w)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, aspect=20)
    cbar.set_label('AUC (%)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    save(fig, 'fig_sota_robustness_heatmap')


if __name__ == '__main__':
    print("Splitting robustness combo into 2 separate figures...")
    fig_degradation_curves()
    fig_sota_robustness_heatmap()
    print("[Done]")
