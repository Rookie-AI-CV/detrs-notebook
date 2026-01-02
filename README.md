# DETR 系列算法

## 项目简介

本仓库收集了 DETR (Detection Transformer) 系列算法的主要论文和代码实现。DETR 是首个将 Transformer 架构应用于目标检测的端到端检测器，消除了对 Anchor、NMS 等手工设计组件的依赖。

## 项目内容

- **论文资源** (`paper/`): DETR 系列算法论文，包含原版、中英文对照版和中文翻译版
- **代码实现** (`code/`): 通过 Git Submodule 集成的各算法官方实现
- **技术分享** (`DETR系列算法-从原型到SOTA.ipynb`): 基于 RISE 的 Jupyter PPT 演示文档，包含算法原理、代码实现和实验分析

## 快速开始

```bash
# 克隆仓库
git clone git@github.com:Rookie-AI-CV/detrs-notebook.git
cd detrs-notebook

# 初始化子模块
git submodule init
git submodule update

# 安装依赖
pip install -r requirements.txt

# 启用 RISE 扩展（PPT 演示）
jupyter-nbextension install rise --py --sys-prefix
jupyter-nbextension enable rise --py --sys-prefix
```

## 项目结构

```
detrs-notebook/
├── README.md
├── DETR系列算法-从原型到SOTA.ipynb
├── paper/                    # 论文资源
│   ├── detr/                 # DETR 系列算法
│   ├── backbone/             # Backbone 网络论文
│   └── related/              # 相关研究
├── code/                     # 代码实现（Git Submodule）
│   ├── detr/
│   ├── ConditionalDETR/
│   ├── DAB-DETR/
│   ├── DINO/
│   ├── RT-DETR/
│   ├── D-FINE/
│   ├── DEIM/
│   ├── AnchorDETR/
│   └── detrex/
├── imgs/
└── utils.py
```

## 论文资源

### DETR 系列算法（按年份排序）

| 年份 | 算法名称 | PDF链接 | 中英文对照版 | 中文版 | 代码仓库 |
|------|---------|---------|-------------|--------|---------|
| 2020 | DETR | [End-to-End Object Detection with Transformers](./paper/detr/End-to-End%20Object%20Detection%20with%20Transformers.pdf) | [dual](./paper/detr/End-to-End%20Object%20Detection%20with%20Transformers-dual.pdf) | [mono](./paper/detr/End-to-End%20Object%20Detection%20with%20Transformers-mono.pdf) | [detr](./code/detr) |
| 2021 | Deformable DETR | [DEFORMABLE DETR](./paper/detr/DEFORMABLE%20DETR.pdf) | [dual](./paper/detr/DEFORMABLE%20DETR-dual.pdf) | [mono](./paper/detr/DEFORMABLE%20DETR-mono.pdf) | |
| 2021 | Conditional DETR | [Conditional DETR for Fast Training Convergence](./paper/detr/Conditional%20DETR%20for%20Fast%20Training%20Convergence.pdf) | [dual](./paper/detr/Conditional%20DETR%20for%20Fast%20Training%20Convergence-dual.pdf) | [mono](./paper/detr/Conditional%20DETR%20for%20Fast%20Training%20Convergence-mono.pdf) | [ConditionalDETR](./code/ConditionalDETR) |
| 2022 | DAB-DETR | [DAB-DETR](./paper/detr/DAB-DETR.pdf) | [dual](./paper/detr/DAB-DETR-dual.pdf) | [mono](./paper/detr/DAB-DETR-mono.pdf) | [DAB-DETR](./code/DAB-DETR) |
| 2022 | DN-DETR | [DN-DETR](./paper/detr/DN-DETR.pdf) | [dual](./paper/detr/DN-DETR-dual.pdf) | [mono](./paper/detr/DN-DETR-mono.pdf) | |
| 2022 | AnchorDETR | | | | [AnchorDETR](./code/AnchorDETR) |
| 2023 | DINO | [DINO - DETR with Improved DeNoising Anchor](./paper/detr/DINO%20-%20DETR%20with%20Improved%20DeNoising%20Anchor.pdf) | [dual](./paper/detr/DINO%20-%20DETR%20with%20Improved%20DeNoising%20Anchor-dual.pdf) | [mono](./paper/detr/DINO%20-%20DETR%20with%20Improved%20DeNoising%20Anchor-mono.pdf) | [DINO](./code/DINO) |
| 2023 | RT-DETR | [RT-DETR](./paper/detr/RT-DETR.pdf) | [dual](./paper/detr/RT-DETR-dual.pdf) | [mono](./paper/detr/RT-DETR-mono.pdf) | [RT-DETR](./code/RT-DETR) |
| 2023 | RT-DETRv3 | [RT-DETRv3](./paper/detr/RT-DETRv3.pdf) | [dual](./paper/detr/RT-DETRv3-dual.pdf) | [mono](./paper/detr/RT-DETRv3-mono.pdf) | |
| 2023 | D-FINE | [D-FINE](./paper/detr/D-FINE.pdf) | [dual](./paper/detr/D-FINE-dual.pdf) | [mono](./paper/detr/D-FINE-mono.pdf) | [D-FINE](./code/D-FINE) |
| 2023 | DEIM | | | | [DEIM](./code/DEIM) |
| | LW-DETR | [LW-DETR .. A Transformer Replacement to](./paper/detr/LW-DETR%20..%20A%20Transformer%20Replacement%20to.pdf) | [dual](./paper/detr/LW-DETR%20..%20A%20Transformer%20Replacement%20to-dual.pdf) | [mono](./paper/detr/LW-DETR%20..%20A%20Transformer%20Replacement%20to-mono.pdf) | |
| | RF-DETR | [RF-DETR.vsYOLO12](./paper/detr/RF-DETR.vsYOLO12.pdf) | [dual](./paper/detr/RF-DETR.vs.YOLO12-dual.pdf) | [mono](./paper/detr/RF-DETR.vs.YOLO12-mono.pdf) | |
| | detrex | | | | [detrex](./code/detrex) |

### Backbone 网络论文

| 算法名称 | PDF链接 | 中英文对照版 | 中文版 | 代码仓库 |
|---------|---------|-------------|--------|---------|
| DINOv1 | [DINOv1 - Emerging Properties in Self-Supervised Vision Transformers](./paper/backbone/DINOv1%20-%20Emerging%20Properties%20in%20Self-Supervised%20Vision%20Transformers.pdf) | [dual](./paper/backbone/DINOv1%20-%20Emerging%20Properties%20in%20Self-Supervised%20Vision%20Transformers-dual.pdf) | [mono](./paper/backbone/DINOv1%20-%20Emerging%20Properties%20in%20Self-Supervised%20Vision%20Transformers-mono.pdf) | |
| DINOv2 | [DINOv2 - Learning Robust Visual Features](./paper/backbone/DINOv2%20-%20Learning%20Robust%20Visual%20Features.pdf) | | | |
| On the Relationship between Self-Attention and Convolutional Layers | [ON THE RELATIONSHIP BETWEEN SELF-ATTENTION](./paper/backbone/ON%20THE%20RELATIONSHIP%20BETWEEN%20SELF-ATTENTION.pdf) | [dual](./paper/backbone/ON%20THE%20RELATIONSHIP%20BETWEEN%20SELF-ATTENTION-dual.pdf) | [mono](./paper/backbone/ON%20THE%20RELATIONSHIP%20BETWEEN%20SELF-ATTENTION-mono.pdf) | |

### 相关研究

| 年份 | 算法名称 | PDF链接 | 中英文对照版 | 中文版 | 代码仓库 |
|------|---------|---------|-------------|--------|---------|
| 2021 | UP-DETR | [UP-DETR](./paper/related/UP-DETR.pdf) | | | |
| 2021 | YOLOS | [YOLOS](./paper/related/YOLOS.pdf) | | | |
| 2021 | TSP | [TSP](./paper/related/TSP.pdf) | [dual](./paper/related/TSP-dual.pdf) | [mono](./paper/related/TSP-mono.pdf) | |
| 2021 | Rethinking Transformer-based Set Prediction (ICCV) | [Sun_Rethinking_Transformer-Based_Set_Prediction_for_Object_Detection_ICCV_2021_paper](./paper/related/Sun_Rethinking_Transformer-Based_Set_Prediction_for_Object_Detection_ICCV_2021_paper.pdf) | [dual](./paper/related/Sun_Rethinking_Transformer-Based_Set_Prediction_for_Object_Detection_ICCV_2021_paper-dual.pdf) | [mono](./paper/related/Sun_Rethinking_Transformer-Based_Set_Prediction_for_Object_Detection_ICCV_2021_paper-mono.pdf) | |
| 2021 | You Only Look at One Sequence (NeurIPS) | [NeurIPS-2021-you-only-look-at-one-sequence-rethinking-transformer-in-vision-through-object-detection-Paper](./paper/related/NeurIPS-2021-you-only-look-at-one-sequence-rethinking-transformer-in-vision-through-object-detection-Paper.pdf) | [dual](./paper/related/NeurIPS-2021-you-only-look-at-one-sequence-rethinking-transformer-in-vision-through-object-detection-Paper-dual.pdf) | [mono](./paper/related/NeurIPS-2021-you-only-look-at-one-sequence-rethinking-transformer-in-vision-through-object-detection-Paper-mono.pdf) | |
| 2021 | PIX2SEQ | [PIX2SEQ](./paper/related/PIX2SEQ.pdf) | [dual](./paper/related/PIX2SEQ-dual.pdf) | [mono](./paper/related/PIX2SEQ-mono.pdf) | |

## 说明

- **PDF 版本**: 原版、中英文对照版 (dual)、中文翻译版 (mono)
- **代码仓库**: 通过 Git Submodule 管理，位于 `code/` 目录

## 运行 PPT 演示

安装依赖并启用 RISE 扩展后，打开 `DETR系列算法-从原型到SOTA.ipynb`，点击工具栏中的 "Enter/Exit RISE Slideshow" 按钮（或按 `Alt+R`）即可进入演示模式。

演示快捷键：
- `Space` / `→`: 下一张幻灯片
- `Shift+Space` / `←`: 上一张幻灯片
- `Esc`: 退出演示模式

## 相关链接

- GitHub: [Rookie-AI-CV/detrs-notebook](https://github.com/Rookie-AI-CV/detrs-notebook)
- 技术分享: `DETR系列算法-从原型到SOTA.ipynb`
- RISE 文档: [RISE - Reveal.js Jupyter/IPython Slideshow Extension](https://rise.readthedocs.io/)
