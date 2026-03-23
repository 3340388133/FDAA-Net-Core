# FDAA-Net Core Code

FDAA-Net 是一个面向 AI 生成图像检测的研究项目，核心目标是提升 **真实后处理扰动下的鲁棒性** 与 **跨生成器泛化能力**。

本仓库是「仅核心代码」版本，适合开源与复现：
- 保留：模型、训练与评估脚本、配置、复现实验入口。
- 不包含：权重文件、实验输出、原始数据集、大体积中间结果。

## 1. 这个项目做了什么

FDAA-Net 在冻结 CLIP ViT-L/14 语义骨干上，加入两个关键模块：
- **FDAA**（Frequency-Domain Artifact Analysis）
  - 用 `SRM 30 通道` + `FFT 幅度/相位 6 通道` 提取频域伪影。
- **MGFP**（Multi-Granularity Fusion Perception）
  - 用频域引导的交叉注意力，把频域线索和 CLIP patch 语义做多粒度融合。

根据论文与 `paper_draft_report_v2.md` 统计，主要结果包括：
- 域内 AUC：`99.93%`
- 鲁棒性平均 AUC：`97.79%`
- 4 个跨数据集基准均排名第一（覆盖 26+ 未见生成器）
- 可训练参数：`19.11M`（冻结 CLIP 后约占总参数 5.9%）

详细结果解读见 `REPORT.md`。

## 2. 环境安装

建议 Python 3.8+，CUDA 11.8+。

```bash
conda create -n fdaa-net python=3.8 -y
conda activate fdaa-net

pip install -r requirements.txt
# 跨数据集评估需要
pip install pyarrow huggingface_hub
```

## 3. 数据准备

### 3.1 训练数据（必需）

使用 GenImage 的 6 个源：
- `biggan`, `adm`, `glide`, `vqdm`, `midjourney`, `sdv4`

建议目录：

```text
datasets/authoritative/GenImage/
```

具体格式可参考 `DATASET_README.md`。

### 3.2 可选评估数据

用于本地跨域评估（Table 6 对应入口）:
- `DiffusionForensics`
- `CIFAKE`
- `NTIRE2026`

建议目录：

```text
datasets/authoritative/
├── DiffusionForensics/
├── CIFAKE/
└── NTIRE2026/
```

## 4. 复现步骤（论文 V2/6src）

### 4.1 推荐：用环境变量指定数据路径

```bash
export FDAA_DATASETS_ROOT=/abs/path/to/datasets/authoritative
# 可选：单独覆盖
# export FDAA_GENIMAGE_ROOT=/abs/path/to/datasets/authoritative/GenImage
# export FDAA_DIFFUSION_FORENSICS_ROOT=/abs/path/to/datasets/authoritative/DiffusionForensics
# export FDAA_CIFAKE_ROOT=/abs/path/to/datasets/authoritative/CIFAKE
# export FDAA_NTIRE2026_ROOT=/abs/path/to/datasets/authoritative/NTIRE2026
```

### 4.2 按模块复现

```bash
# 1) 6 源训练
python experiments/run_6src_experiments.py --mode train

# 2) Leave-One-Out
python experiments/run_6src_experiments.py --mode loo

# 3) 鲁棒性评估（JPEG/Blur/Noise）
python experiments/run_6src_experiments.py --mode robustness

# 4) 跨域评估（本地评估集）
python experiments/run_6src_experiments.py --mode cross_domain

# 5) 汇总报告
python experiments/run_6src_experiments.py --mode report
```

### 4.3 跨数据集（HF Arrow）

```bash
# UniversalFakeDetect + ForenSynths
python experiments/run_crossdataset_eval.py

# Synthbuster
python experiments/run_synthbuster_hf_eval.py
```

## 5. 结果文件与表格对应（V2）

- Table 1：`outputs/paper_results_6src/results/ours_6src_train_results.json` + `sota_6src_group*.json`
- Table 2：`outputs/paper_results_6src/results/robustness_results.json`
- Table 3：`outputs/paper_results_6src/results/crossdataset_ojha_results.json`
- Table 4：`outputs/paper_results_6src/results/crossdataset_forensynths_results.json`
- Table 5：`outputs/paper_results_6src/results/crossdataset_synthbuster_results.json`
- Table 6：`outputs/paper_results_6src/results/cross_domain_results.json`
- Table 7：`outputs/paper_results_6src/results/leave_one_out_6src_results.json`
- Table 8/9/10：`outputs/paper_results_6src/results/ablation_*.json`
- Table 11：`outputs/paper_results_6src/results/efficiency_results.json`

## 6. 仓库结构

```text
configs/         # 训练与评估配置
data/            # 数据加载与增广
models/          # FDAA-Net 主体与模块
trainers/        # 训练器
utils/           # 指标、日志、检查点工具
experiments/     # 论文实验入口脚本
visualization/   # 可视化辅助
docs/report_v2/  # 完整图文报告与全部图表
REPORT.md        # 项目报告摘要（本仓库新增）
```

## 7. 论文与报告

- 完整图文报告：`docs/report_v2/paper_draft_report_v2.md`
- 报告图表目录：`docs/report_v2/`
- 本仓库摘要版：`REPORT.md`

## 8. 图文报告（README 内嵌关键图）

完整图文版见：[`docs/report_v2/paper_draft_report_v2.md`](docs/report_v2/paper_draft_report_v2.md)

### 8.1 方法与架构

![FDAA-Net 架构图](docs/report_v2/fig01_architecture_overview.png)
![频域特征覆盖对比](docs/report_v2/fig02_method_comparison.png)
![融合方式对比](docs/report_v2/fig13_fusion_comparison.png)

### 8.2 鲁棒性与跨数据集结果

![鲁棒性退化曲线](docs/report_v2/fig03_robustness_degradation.png)
![鲁棒性热力图](docs/report_v2/fig04_robustness_heatmap.png)
![跨数据集雷达图](docs/report_v2/fig06_cross_dataset_radar.png)

### 8.3 消融、效率与训练动态

![模块消融](docs/report_v2/fig05_ablation_combined.png)
![损失函数消融](docs/report_v2/fig09_loss_ablation.png)
![效率对比](docs/report_v2/fig08_efficiency_scatter.png)
![训练曲线](docs/report_v2/fig10_training_curves.png)
