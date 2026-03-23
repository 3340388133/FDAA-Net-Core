#!/usr/bin/env python3
"""
修复散点图标签重叠 + 做组合大图
"""
import os, json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle, FancyArrowPatch, ConnectionPatch
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5,
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
# Fig 6: 散点图 — 主图 + 右上角放大插图
# =====================================================================
def fix_fig6_scatter():
    pts = [
        ('FDAA-Net\n(Ours)', 99.93, 97.79, 19.11, '#C0392B'),
        ('F3Net',            99.75, 94.47, 5.18,  '#2980B9'),
        ('CNNDet',           99.40, 94.08, 23.51, '#27AE60'),
        ('FreqNet',          99.07, 93.86, 23.38, '#8E44AD'),
        ('UnivFD',           99.13, 93.47, 0.39,  '#E67E22'),
        ('DIRE',             99.45, 92.73, 23.53, '#1ABC9C'),
        ('LaRE²',            98.17, 92.07, 37.96, '#95A5A6'),
        ('NPR',              97.47, 90.92, 11.37, '#34495E'),
        ('C2P-CLIP',         89.79, 85.90, 18.37, '#BDC3C7'),
        ('SPEC',             97.12, 79.29, 1.55,  '#7F8C8D'),
    ]

    fig, ax_main = plt.subplots(figsize=(6.0, 4.5))

    # 画所有点（主图）
    for name, dom, rob, par, color in pts:
        is_ours = 'Ours' in name
        s = max(par * 3, 35)
        ax_main.scatter(dom, rob, s=s, c=color,
                        alpha=1.0 if is_ours else 0.7,
                        zorder=10 if is_ours else 5,
                        edgecolor='#C0392B' if is_ours else '#555555',
                        linewidth=1.5 if is_ours else 0.4)

    # 主图标签：只标注不拥挤的点
    non_dense = ['FDAA-Net\n(Ours)', 'C2P-CLIP', 'SPEC', 'NPR', 'LaRE²']
    label_offsets = {
        'FDAA-Net\n(Ours)': (-15, 10),
        'C2P-CLIP': (10, 5),
        'SPEC': (10, 5),
        'NPR': (10, 5),
        'LaRE²': (10, -12),
    }
    for name, dom, rob, par, color in pts:
        if name in non_dense:
            ox, oy = label_offsets[name]
            is_ours = 'Ours' in name
            ax_main.annotate(name.replace('\n', ' '), (dom, rob),
                             textcoords='offset points', xytext=(ox, oy),
                             fontsize=8 if is_ours else 7,
                             weight='bold' if is_ours else 'normal',
                             color='#C0392B' if is_ours else '#333333',
                             ha='center' if is_ours else 'left')

    ax_main.set_xlabel('In-domain AUC (%)', fontsize=10)
    ax_main.set_ylabel('Avg Robustness AUC (%)', fontsize=10)
    ax_main.set_xlim(88, 101)
    ax_main.set_ylim(77, 100)
    ax_main.set_axisbelow(True)
    ax_main.grid(True, linestyle='--', alpha=0.2)
    ax_main.axhline(y=97.79, color='#C0392B', linestyle=':', linewidth=0.5, alpha=0.3)
    ax_main.axvline(x=99.93, color='#C0392B', linestyle=':', linewidth=0.5, alpha=0.3)

    # 拥挤区域虚线框
    zx1, zx2 = 98.8, 100.1
    zy1, zy2 = 92.0, 95.2
    rect = Rectangle((zx1, zy1), zx2 - zx1, zy2 - zy1,
                      linewidth=1.0, edgecolor='#555555', facecolor='none',
                      linestyle='--', zorder=8)
    ax_main.add_patch(rect)

    # ===== 放大插图 =====
    ax_zoom = inset_axes(ax_main, width="45%", height="45%",
                         loc='center left', bbox_to_anchor=(0.05, 0.55, 1, 1),
                         bbox_transform=ax_main.transAxes)

    # 放大区域的点
    dense_pts = [p for p in pts if zx1 <= p[1] <= zx2 and zy1 <= p[2] <= zy2]
    for name, dom, rob, par, color in dense_pts:
        s = max(par * 5, 50)
        ax_zoom.scatter(dom, rob, s=s, c=color, alpha=0.85, zorder=5,
                        edgecolor='#555555', linewidth=0.5)

    # 放大图里的标签 — 每个点手动设置偏移，保证不重叠
    zoom_offsets = {
        'F3Net':   (8, 6),
        'CNNDet':  (8, -10),
        'FreqNet': (-50, 6),
        'UnivFD':  (-50, -10),
        'DIRE':    (8, 6),
    }
    for name, dom, rob, par, color in dense_pts:
        ox, oy = zoom_offsets.get(name, (8, 0))
        ax_zoom.annotate(name, (dom, rob),
                         textcoords='offset points', xytext=(ox, oy),
                         fontsize=8, color=color, weight='bold',
                         arrowprops=dict(arrowstyle='->', color=color, lw=0.8,
                                         connectionstyle='arc3,rad=0.15'))

    ax_zoom.set_xlim(zx1 - 0.1, zx2 + 0.1)
    ax_zoom.set_ylim(zy1 - 0.3, zy2 + 0.3)
    ax_zoom.set_axisbelow(True)
    ax_zoom.grid(True, linestyle='--', alpha=0.3)
    ax_zoom.set_title('Zoomed dense region', fontsize=7.5, style='italic', pad=3)
    ax_zoom.tick_params(labelsize=6.5)
    for spine in ax_zoom.spines.values():
        spine.set_edgecolor('#555555')
        spine.set_linewidth(0.8)

    # 连接线
    mark_inset(ax_main, ax_zoom, loc1=2, loc2=4, fc="none", ec="#555555",
               lw=0.6, linestyle='--')

    plt.tight_layout()
    save(fig, 'fig6_performance_robustness_scatter')


# =====================================================================
# 组合大图1: 鲁棒性综合分析 (折线图 + SOTA热力图)
# =====================================================================
def combo_fig_robustness():
    """大图: 上=衰减折线图(3子图), 下=SOTA热力图"""
    fig = plt.figure(figsize=(7.5, 7.0))
    gs = GridSpec(2, 1, height_ratios=[1, 1.3], hspace=0.35)

    # === 上半: 衰减折线图 (3子图) ===
    gs_top = gs[0].subgridspec(1, 3, wspace=0.12)
    axes_top = [fig.add_subplot(gs_top[0, i]) for i in range(3)]

    methods = {
        'FDAA-Net (Ours)': {'c': '#C0392B', 'ls': '-',  'm': 'o', 'lw': 1.6},
        'F3Net':            {'c': '#2980B9', 'ls': '--', 'm': 's', 'lw': 0.9},
        'CNNDetection':     {'c': '#27AE60', 'ls': '--', 'm': '^', 'lw': 0.9},
        'FreqNet':          {'c': '#8E44AD', 'ls': '--', 'm': 'D', 'lw': 0.9},
        'UnivFD':           {'c': '#E67E22', 'ls': '--', 'm': 'v', 'lw': 0.9},
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

    for ax, (title, xlabels, data) in zip(axes_top, datasets):
        ax.set_title(title, fontsize=9, pad=4)
        for name, cfg in methods.items():
            vals = [clean[name]] + data[name]
            ax.plot(xlabels, vals, color=cfg['c'], linestyle=cfg['ls'],
                    marker=cfg['m'], markersize=4, markeredgewidth=0.4,
                    markeredgecolor='white', label=name, linewidth=cfg['lw'],
                    zorder=10 if 'Ours' in name else 5)
        ax.set_ylim(83, 101)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)

    axes_top[0].set_ylabel('AUC (%)')
    # Blur图标注差距
    axes_top[1].annotate('', xy=(2, 94.53), xytext=(2, 89.39),
                          arrowprops=dict(arrowstyle='<->', color='#C0392B', lw=0.8))
    axes_top[1].text(2.15, 91.6, r'$\Delta$5.1', fontsize=7, color='#C0392B', weight='bold')

    handles, labels = axes_top[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02),
               ncol=5, frameon=True, fancybox=False, edgecolor='#CCCCCC',
               columnspacing=1.0, handletextpad=0.3, fontsize=7.5)

    # === 下半: SOTA热力图 ===
    ax_heat = fig.add_subplot(gs[1])

    method_names = ['FDAA-Net (Ours)', 'F3Net', 'CNNDet', 'FreqNet', 'UnivFD',
                    'DIRE', 'LaRE²', 'NPR', 'C2P-CLIP', 'SPEC']
    test_labels = ['JPEG\nQ=70', 'JPEG\nQ=50', 'JPEG\nQ=30', 'Blur\nσ=1.0',
                   'Blur\nσ=2.0', 'Noise\nσ=.02', 'Noise\nσ=.05']
    data_arr = np.array([
        [99.01,98.24,96.33,99.39,94.53,99.08,97.97],
        [97.04,95.94,93.66,98.96,89.39,96.78,89.49],
        [96.54,95.65,93.96,98.42,85.66,96.23,92.08],
        [96.61,95.59,93.40,97.28,84.92,96.17,93.08],
        [95.18,93.79,90.98,97.04,88.08,96.30,92.95],
        [96.28,94.81,92.69,98.39,84.94,94.89,87.14],
        [95.16,94.08,92.28,95.62,83.48,94.60,89.26],
        [94.84,93.77,91.53,93.55,81.63,92.90,88.23],
        [88.39,87.52,86.25,84.55,79.40,89.13,86.04],
        [81.38,79.79,77.43,89.18,64.29,83.69,79.27],
    ])
    avgs = data_arr.mean(axis=1, keepdims=True)
    data_ext = np.hstack([data_arr, avgs])
    col_labels = test_labels + ['Avg']

    im = ax_heat.imshow(data_ext, cmap='RdYlGn', aspect='auto', vmin=62, vmax=100)
    ax_heat.set_xticks(range(len(col_labels)))
    ax_heat.set_xticklabels(col_labels, fontsize=7.5)
    ax_heat.set_yticks(range(len(method_names)))
    ax_heat.set_yticklabels(method_names, fontsize=8)
    ax_heat.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False)
    ax_heat.axvline(x=6.5, color='black', linewidth=1.0)
    ax_heat.axhline(y=0.5, color='#C0392B', linewidth=1.2)
    ax_heat.set_title('(d) Robustness Comparison Heatmap (AUC %)', fontsize=9, pad=15)

    for i in range(len(method_names)):
        for j in range(len(col_labels)):
            v = data_ext[i, j]
            is_best = (i == np.argmax(data_ext[:, j]))
            color = 'white' if v < 82 else 'black'
            ax_heat.text(j, i, f'{v:.1f}', ha='center', va='center',
                         fontsize=6.5, color=color, weight='bold' if is_best else 'normal')

    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.7, pad=0.02, aspect=20)
    cbar.set_label('AUC (%)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    save(fig, 'fig_combo_robustness')


# =====================================================================
# 组合大图2: 消融综合分析 (域内vs鲁棒性 + 消融热力图)
# =====================================================================
def combo_fig_ablation():
    """大图: 上=域内vs鲁棒性柱状对比, 下=消融鲁棒性热力图"""
    with open(os.path.join(RESULTS_DIR, 'ablation_robustness_results.json')) as f:
        rob = json.load(f)

    fig = plt.figure(figsize=(7.5, 5.5))
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.4)

    # === 上半: 域内 vs 鲁棒性柱状图 ===
    gs_top = gs[0].subgridspec(1, 2, wspace=0.35)
    ax_dom = fig.add_subplot(gs_top[0, 0])
    ax_rob = fig.add_subplot(gs_top[0, 1])

    variants = ['Baseline', '+ FDAA', '+ MGFP', 'Full']
    colors = ['#BDC3C7', '#3498DB', '#2ECC71', '#C0392B']
    domain_auc = [99.49, 99.69, 99.87, 99.87]
    rob_avg = [94.53, 95.59, 97.40, 97.79]

    # (a) In-domain
    ax_dom.bar(variants, domain_auc, color=colors, edgecolor='white', width=0.6, zorder=3, alpha=0.88)
    ax_dom.set_ylim(99.2, 100.05)
    ax_dom.set_ylabel('AUC (%)')
    ax_dom.set_title('(a) In-domain Detection', fontsize=9, pad=4)
    ax_dom.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax_dom.set_axisbelow(True)
    ax_dom.yaxis.grid(True, linestyle='--', alpha=0.3)
    for i, auc in enumerate(domain_auc):
        ax_dom.text(i, auc + 0.02, f'{auc:.2f}', ha='center', fontsize=7, weight='bold')
    ax_dom.annotate(r'$\Delta$+0.20', xy=(1, 99.72), xytext=(1.8, 99.92),
                    fontsize=7, ha='center', color='#3498DB', weight='bold',
                    arrowprops=dict(arrowstyle='->', color='#3498DB', lw=0.6))
    ax_dom.tick_params(axis='x', rotation=15)

    # (b) Robustness
    ax_rob.bar(variants, rob_avg, color=colors, edgecolor='white', width=0.6, zorder=3, alpha=0.88)
    ax_rob.set_ylim(93, 99.5)
    ax_rob.set_ylabel('Avg Robustness AUC (%)')
    ax_rob.set_title('(b) Robustness (Avg 7 Perturbations)', fontsize=9, pad=4)
    ax_rob.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax_rob.set_axisbelow(True)
    ax_rob.yaxis.grid(True, linestyle='--', alpha=0.3)
    for i, auc in enumerate(rob_avg):
        ax_rob.text(i, auc + 0.1, f'{auc:.1f}', ha='center', fontsize=7, weight='bold')
    ax_rob.annotate(r'$\Delta$+1.06', xy=(1, 95.75), xytext=(1, 96.8),
                    fontsize=7, ha='center', color='#3498DB', weight='bold',
                    arrowprops=dict(arrowstyle='->', color='#3498DB', lw=0.6))
    ax_rob.annotate(r'5.3$\times$ larger', xy=(1.1, 97.1), xytext=(2.6, 95.8),
                    fontsize=7, ha='center', color='#C0392B', weight='bold',
                    arrowprops=dict(arrowstyle='->', color='#C0392B', lw=0.6))
    ax_rob.tick_params(axis='x', rotation=15)

    # === 下半: 消融鲁棒性热力图 ===
    ax_heat = fig.add_subplot(gs[1])

    abl_variants = ['abl_baseline', 'abl_baseline+fdaa', 'abl_baseline+mgfp', 'abl_full']
    abl_labels = ['Baseline (CLIP+CLS)', '+ FDAA', '+ MGFP', 'Full (Ours)']
    tests = ['jpeg_70', 'jpeg_50', 'jpeg_30', 'blur_1.0', 'blur_2.0', 'noise_0.02', 'noise_0.05']
    test_labels = ['JPEG\nQ=70', 'JPEG\nQ=50', 'JPEG\nQ=30', 'Blur\nσ=1.0',
                   'Blur\nσ=2.0', 'Noise\nσ=.02', 'Noise\nσ=.05']

    data_arr = np.array([[rob[t][v]['auc'] * 100 for t in tests] for v in abl_variants])
    avgs = data_arr.mean(axis=1, keepdims=True)
    data_ext = np.hstack([data_arr, avgs])
    col_labels = test_labels + ['Avg']

    im = ax_heat.imshow(data_ext, cmap='RdYlGn', aspect='auto', vmin=88, vmax=100)
    ax_heat.set_xticks(range(len(col_labels)))
    ax_heat.set_xticklabels(col_labels, fontsize=7.5)
    ax_heat.set_yticks(range(len(abl_labels)))
    ax_heat.set_yticklabels(abl_labels, fontsize=8)
    ax_heat.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False)
    ax_heat.axvline(x=6.5, color='black', linewidth=1.0)
    ax_heat.set_title('(c) Module Ablation under Perturbations (AUC %)', fontsize=9, pad=15)

    for i in range(len(abl_variants)):
        for j in range(len(col_labels)):
            v = data_ext[i, j]
            is_best = (i == np.argmax(data_ext[:, j]))
            is_avg = (j == len(col_labels) - 1)
            color = 'white' if v < 92 else 'black'
            ax_heat.text(j, i, f'{v:.1f}', ha='center', va='center',
                         fontsize=7, color=color, weight='bold' if is_best or is_avg else 'normal')

    # ΔAUC
    for i in range(1, len(abl_labels)):
        delta = avgs[i, 0] - avgs[0, 0]
        ax_heat.text(len(col_labels) - 0.5, i, f'  Δ+{delta:.1f}',
                     ha='left', va='center', fontsize=7, color='#C0392B', weight='bold')

    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.85, pad=0.12, aspect=15)
    cbar.set_label('AUC (%)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    save(fig, 'fig_combo_ablation')


# =====================================================================
# 组合大图3: 泛化综合分析 (雷达图 + 双热力图 + LOO)
# =====================================================================
def combo_fig_generalization():
    """大图: 2×2布局"""
    fig = plt.figure(figsize=(7.5, 7.0))
    gs = GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # === (a) 雷达图 ===
    ax_radar = fig.add_subplot(gs[0, 0], polar=True)
    categories = ['UniversalFakeDetect', 'ForenSynths', 'Synthbuster', 'DiffForensics']
    radar_data = {
        'FDAA-Net (Ours)': [96.31, 94.2, 82.2, 95.65],
        'UnivFD':           [88.63, 90.8, 79.6, 92.92],
        'F3Net':            [89.88, 83.6, 78.3, 88.89],
        'DIRE':             [91.47, 74.7, 71.0, 95.03],
        'CNNDetection':     [91.04, 77.5, 75.3, 92.84],
    }
    radar_colors = ['#C0392B', '#E67E22', '#2980B9', '#8E44AD', '#27AE60']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

    for (name, vals), color in zip(radar_data.items(), radar_colors):
        values = vals + vals[:1]
        lw = 1.8 if 'Ours' in name else 0.9
        ms = 4.5 if 'Ours' in name else 2.5
        ax_radar.plot(angles, values, 'o-', color=color, linewidth=lw, markersize=ms,
                      markeredgecolor='white', markeredgewidth=0.3, label=name)
        if 'Ours' in name:
            ax_radar.fill(angles, values, alpha=0.08, color=color)

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories, fontsize=6.5)
    ax_radar.set_ylim(60, 100)
    ax_radar.set_yticks([70, 80, 90, 100])
    ax_radar.set_yticklabels(['70', '80', '90', '100'], fontsize=6)
    ax_radar.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax_radar.set_title('(a) Cross-dataset Overview', fontsize=9, pad=12)

    # === (b) LOO ===
    ax_loo = fig.add_subplot(gs[0, 1])
    generators = ['BigGAN', 'GLIDE', 'VQDM', 'SDv4', 'ADM', 'MJ']
    aucs = [99.66, 99.37, 99.37, 98.80, 91.84, 81.25]
    avg = np.mean(aucs)
    bar_colors = ['#27AE60' if a > 95 else '#E67E22' if a > 85 else '#E74C3C' for a in aucs]
    ax_loo.bar(generators, aucs, color=bar_colors, edgecolor='white', width=0.6, zorder=3, alpha=0.85)
    ax_loo.axhline(y=avg, color='#2C3E50', linestyle='--', linewidth=0.8, alpha=0.5)
    ax_loo.text(5.4, avg + 0.5, f'Avg:{avg:.1f}%', fontsize=6.5, color='#2C3E50')
    for i, auc in enumerate(aucs):
        ax_loo.text(i, auc + 0.5, f'{auc:.1f}', ha='center', fontsize=6.5, weight='bold')
    ax_loo.set_ylabel('AUC (%)')
    ax_loo.set_ylim(75, 103)
    ax_loo.set_axisbelow(True)
    ax_loo.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax_loo.set_title('(b) Leave-One-Out', fontsize=9, pad=4)
    ax_loo.tick_params(axis='x', rotation=15)

    # === (c) ForenSynths热力图 ===
    ax_fs = fig.add_subplot(gs[1, 0])
    mths = ['FDAA-Net', 'UnivFD', 'F3Net', 'DIRE', 'CNNDet']
    gens_a = ['ProGAN', 'StyleGAN', 'StyleGAN2', 'CycleGAN', 'GauGAN', 'BigGAN']
    data_a = np.array([
        [99.5,94.5,96.6,94.7,99.9,100.0],
        [98.3,84.7,89.6,91.1,99.2,99.9],
        [78.2,73.3,67.0,81.5,74.9,78.5],
        [82.3,68.1,71.2,68.2,43.8,66.7],
        [79.8,70.7,78.2,68.2,50.8,73.2],
    ])
    im1 = ax_fs.imshow(data_a, cmap='RdYlGn', aspect='auto', vmin=40, vmax=100)
    ax_fs.set_xticks(range(len(gens_a)))
    ax_fs.set_xticklabels(gens_a, rotation=35, ha='right', fontsize=6.5)
    ax_fs.set_yticks(range(len(mths)))
    ax_fs.set_yticklabels(mths, fontsize=7.5)
    ax_fs.set_title('(c) ForenSynths (GAN)', fontsize=9, pad=4)
    for i in range(len(mths)):
        for j in range(len(gens_a)):
            v = data_a[i, j]
            ax_fs.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=6,
                       color='white' if v < 65 else 'black', weight='bold' if i == 0 else 'normal')

    # === (d) Synthbuster热力图 ===
    ax_sb = fig.add_subplot(gs[1, 1])
    gens_b = ['SD 1.3', 'SD 1.4', 'SD 2.0', 'SD XL', 'MJ v5']
    data_b = np.array([
        [78.5,76.6,73.6,90.8,93.1],
        [74.0,72.9,72.1,88.1,87.7],
        [73.9,75.7,60.4,89.4,93.2],
        [64.3,64.0,58.6,81.6,86.7],
        [70.5,72.3,66.8,85.7,85.9],
    ])
    im2 = ax_sb.imshow(data_b, cmap='RdYlGn', aspect='auto', vmin=40, vmax=100)
    ax_sb.set_xticks(range(len(gens_b)))
    ax_sb.set_xticklabels(gens_b, rotation=35, ha='right', fontsize=6.5)
    ax_sb.set_yticks(range(len(mths)))
    ax_sb.set_yticklabels(mths, fontsize=7.5)
    ax_sb.set_title('(d) Synthbuster (Diffusion)', fontsize=9, pad=4)
    for i in range(len(mths)):
        for j in range(len(gens_b)):
            v = data_b[i, j]
            ax_sb.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=6,
                       color='white' if v < 62 else 'black', weight='bold' if i == 0 else 'normal')

    # 图例放在雷达图下面
    handles, labels = ax_radar.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.28, 0.52),
               ncol=1, frameon=True, fancybox=False, edgecolor='#CCCCCC', fontsize=7)

    save(fig, 'fig_combo_generalization')


if __name__ == '__main__':
    print("=" * 50)
    print("Fixing figures v2...")
    print("=" * 50)
    fix_fig6_scatter()
    combo_fig_robustness()
    combo_fig_ablation()
    combo_fig_generalization()
    print("\n[Done] 4 figures generated")
