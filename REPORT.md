# FDAA-Net 项目报告（摘要版）

## 1. 研究目标

本项目针对 AI 生成图像检测中的两个痛点：
- 真实后处理（JPEG、模糊、噪声）下性能大幅下降。
- 在未见生成器上的泛化能力不足。

目标是在保持高域内精度的同时，显著提升鲁棒性和跨数据集泛化。

## 2. 方法概述

### 2.1 FDAA：频域伪影分析
- SRM 高通残差（30 通道）提取多方向高频痕迹。
- FFT 幅度 + 相位（6 通道）提取全局频谱信息。
- 通过 4-stage CNN 编码到频域 token 表示。

### 2.2 MGFP：多粒度融合感知
- 频域特征作为 Query，引导 CLIP patch 语义（Key/Value）进行交叉注意力。
- 1x1 / 2x2 / 4x4 多粒度伪造感知。
- 三路门控融合（全局语义、局部融合、频域证据）。

## 3. 实验设置（V2 报告口径）

- 训练集：GenImage 6 源（BigGAN、ADM、GLIDE、VQDM、Midjourney、SD v1.4）
- 对比方法：9 种 SOTA，统一数据与训练设置重训
- 鲁棒性测试：7 种扰动（JPEG/Blur/Noise）
- 跨数据集：Ojha、ForenSynths、Synthbuster、DiffusionForensics
- 指标：AUC、AP、Accuracy、EER

## 4. 关键结果

- 域内性能：AUC `99.93%`，EER `1.10%`
- 鲁棒性：平均 AUC `97.79%`，领先第二名 `3.32pp`
- 跨数据集：4 个基准均第一
- 参数效率：可训练参数 `19.11M`（冻结 CLIP 后约总参数 `5.9%`）
- 速度：约 `17.35ms/图`（约 `58 FPS`）

## 5. 结论

FDAA-Net 的核心收益不是“再提高一点域内上限”，而是把性能优势集中在更有实际价值的维度：
- 后处理扰动鲁棒性
- 未见生成器泛化能力

这也是本项目相较常规空间域检测器的主要工程与研究价值。

## 6. 复现入口

- 主流程（6src）：`experiments/run_6src_experiments.py`
- 跨数据集：`experiments/run_crossdataset_eval.py`、`experiments/run_synthbuster_hf_eval.py`
- 消融补充：`experiments/run_ablation_robustness.py`

详见 `README.md` 的复现步骤与结果文件映射。

## 7. 说明

本文件是面向 GitHub 的摘要版报告。完整材料请参考：
- `FDAA-Net_Paper_CN.pdf`
- `paper_draft_report_v2.md`

