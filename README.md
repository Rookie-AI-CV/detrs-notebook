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
│   ├── two-stage/            # Two-stage 检测器论文
│   ├── one-stage/             # One-stage 检测器论文
│   │   ├── anchor-based/     # Anchor-based 检测器
│   │   └── anchor-free/      # Anchor-free 检测器
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

### 第一部分：传统检测器

#### Two-Stage 检测器

| 年份 | 算法名称 | PDF链接 | 中英文对照版 | 中文版 | 代码仓库 |
|------|---------|---------|-------------|--------|---------|
| 2014 | R-CNN | [R-CNN - Rich feature hierarchies for accurate object detection and semantic segmentation](./paper/two-stage/R-CNN%20-%20Rich%20feature%20hierarchies%20for%20accurate%20object%20detection%20and%20semantic%20segmentation.pdf) | [dual](./paper/two-stage/R-CNN%20-%20Rich%20feature%20hierarchies%20for%20accurate%20object%20detection%20and%20semantic%20segmentation-dual.pdf) | [mono](./paper/two-stage/R-CNN%20-%20Rich%20feature%20hierarchies%20for%20accurate%20object%20detection%20and%20semantic%20segmentation-mono.pdf) | |
| 2015 | Faster R-CNN | [Faster R-CNN - Towards Real-Time Object](./paper/two-stage/Faster%20R-CNN%20-%20Towards%20Real-Time%20Object.pdf) | [dual](./paper/two-stage/Faster%20R-CNN%20-%20Towards%20Real-Time%20Object-dual.pdf) | [mono](./paper/two-stage/Faster%20R-CNN%20-%20Towards%20Real-Time%20Object-mono.pdf) | |
| 2017 | Mask R-CNN | [Mask R-CNN](./paper/two-stage/Mask%20R-CNN.pdf) | [dual](./paper/two-stage/Mask%20R-CNN-dual.pdf) | [mono](./paper/two-stage/Mask%20R-CNN-mono.pdf) | |
| 2018 | Cascade R-CNN | [Cascade R-CNN - Delving into High Quality Object Detection](./paper/two-stage/Cascade%20R-CNN%20-%20Delving%20into%20High%20Quality%20Object%20Detection.pdf) | [dual](./paper/two-stage/Cascade%20R-CNN%20-%20Delving%20into%20High%20Quality%20Object%20Detection-dual.pdf) | [mono](./paper/two-stage/Cascade%20R-CNN%20-%20Delving%20into%20High%20Quality%20Object%20Detection-mono.pdf) | |

#### One-Stage 检测器 - Anchor-based

| 年份 | 算法名称 | PDF链接 | 中英文对照版 | 中文版 | 代码仓库 |
|------|---------|---------|-------------|--------|---------|
| 2016 | YOLO-V1 | [YOLO-V1 - You Only Look Once - Unified,Real-Time Object Detection](./paper/one-stage/anchor‑based/YOLO-V1%20-%20You%20Only%20Look%20Once%20-%20Unified,Real-Time%20Object%20Detection.pdf) | [dual](./paper/one-stage/anchor‑based/YOLO-V1%20-%20You%20Only%20Look%20Once%20-%20Unified,Real-Time%20Object%20Detection-dual.pdf) | [mono](./paper/one-stage/anchor‑based/YOLO-V1%20-%20You%20Only%20Look%20Once%20-%20Unified,Real-Time%20Object%20Detection-mono.pdf) | |
| 2016 | SSD | [SSD - Single Shot MultiBox Detector](./paper/one-stage/anchor‑based/SSD%20-%20Single%20Shot%20MultiBox%20Detector.pdf) | [dual](./paper/one-stage/anchor‑based/SSD%20-%20Single%20Shot%20MultiBox%20Detector-dual.pdf) | [mono](./paper/one-stage/anchor‑based/SSD%20-%20Single%20Shot%20MultiBox%20Detector-mono.pdf) | |
| 2018 | YOLOv3 | [YOLOv3 - An Incremental Improvement](./paper/one-stage/anchor‑based/YOLOv3%20-%20An%20Incremental%20Improvement.pdf) | [dual](./paper/one-stage/anchor‑based/YOLOv3%20-%20An%20Incremental%20Improvement-dual.pdf) | [mono](./paper/one-stage/anchor‑based/YOLOv3%20-%20An%20Incremental%20Improvement-mono.pdf) | |
| 2018 | RetinaNet | [RetinaNet - Focal Loss for Dense Object Detection](./paper/one-stage/anchor‑based/RetinaNet%20-%20Focal%20Loss%20for%20Dense%20Object%20Detection.pdf) | [dual](./paper/one-stage/anchor‑based/RetinaNet%20-%20Focal%20Loss%20for%20Dense%20Object%20Detection-dual.pdf) | [mono](./paper/one-stage/anchor‑based/RetinaNet%20-%20Focal%20Loss%20for%20Dense%20Object%20Detection-mono.pdf) | |

#### One-Stage 检测器 - Anchor-free

| 年份 | 算法名称 | PDF链接 | 中英文对照版 | 中文版 | 代码仓库 |
|------|---------|---------|-------------|--------|---------|
| 2018 | CornerNet | [CornerNet - Detecting Objects as Paired Keypoints](./paper/one-stage/anchor‑free/CornerNet%20-%20Detecting%20Objects%20as%20Paired%20Keypoints.pdf) | [dual](./paper/one-stage/anchor‑free/CornerNet%20-%20Detecting%20Objects%20as%20Paired%20Keypoints-dual.pdf) | [mono](./paper/one-stage/anchor‑free/CornerNet%20-%20Detecting%20Objects%20as%20Paired%20Keypoints-mono.pdf) | |
| 2019 | CenterNet | [CenterNet - Keypoint Triplets for Object Detection](./paper/one-stage/anchor‑free/CenterNet%20-%20Keypoint%20Triplets%20for%20Object%20Detection.pdf) | [dual](./paper/one-stage/anchor‑free/CenterNet%20-%20Keypoint%20Triplets%20for%20Object%20Detection-dual.pdf) | [mono](./paper/one-stage/anchor‑free/CenterNet%20-%20Keypoint%20Triplets%20for%20Object%20Detection-mono.pdf) | |
| 2019 | FCOS | [FCOS - Fully Convolutional One-Stage Object Detection](./paper/one-stage/anchor‑free/FCOS%20-%20Fully%20Convolutional%20One-Stage%20Object%20Detection.pdf) | [dual](./paper/one-stage/anchor‑free/FCOS%20-%20Fully%20Convolutional%20One-Stage%20Object%20Detection-dual.pdf) | [mono](./paper/one-stage/anchor‑free/FCOS%20-%20Fully%20Convolutional%20One-Stage%20Object%20Detection-mono.pdf) | |
| 2019 | RepPoints | [RepPoints - Point Set Representation for Object Detection](./paper/one-stage/anchor‑free/RepPoints%20-%20Point%20Set%20Representation%20for%20Object%20Detection.pdf) | [dual](./paper/one-stage/anchor‑free/RepPoints%20-%20Point%20Set%20Representation%20for%20Object%20Detection-dual.pdf) | [mono](./paper/one-stage/anchor‑free/RepPoints%20-%20Point%20Set%20Representation%20for%20Object%20Detection-mono.pdf) | |
| 2020 | ATSS | [ATSS - Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection](./paper/one-stage/anchor‑free/ATSS%20-%20Bridging%20the%20Gap%20Between%20Anchor-based%20and%20Anchor-free%20Detection%20via%20Adaptive%20Training%20Sample%20Selection.pdf) | [dual](./paper/one-stage/anchor‑free/ATSS%20-%20Bridging%20the%20Gap%20Between%20Anchor-based%20and%20Anchor-free%20Detection%20via%20Adaptive%20Training%20Sample%20Selection-dual.pdf) | [mono](./paper/one-stage/anchor‑free/ATSS%20-%20Bridging%20the%20Gap%20Between%20Anchor-based%20and%20Anchor-free%20Detection%20via%20Adaptive%20Training%20Sample%20Selection-mono.pdf) | |

### 第二部分：DETR 系列

| 年份 | 算法名称 | PDF链接 | 中英文对照版 | 中文版 | 代码仓库 |
|------|---------|---------|-------------|--------|---------|
| 2020 | **DETR** | [End-to-End Object Detection with Transformers](./paper/detr/End-to-End%20Object%20Detection%20with%20Transformers.pdf) | [dual](./paper/detr/End-to-End%20Object%20Detection%20with%20Transformers-dual.pdf) | [mono](./paper/detr/End-to-End%20Object%20Detection%20with%20Transformers-mono.pdf) | [detr](./code/detr) |
| 2021 | **Deformable DETR** | [DEFORMABLE DETR](./paper/detr/DEFORMABLE%20DETR.pdf) | [dual](./paper/detr/DEFORMABLE%20DETR-dual.pdf) | [mono](./paper/detr/DEFORMABLE%20DETR-mono.pdf) | |
| 2021 | **Conditional DETR** | [Conditional DETR for Fast Training Convergence](./paper/detr/Conditional%20DETR%20for%20Fast%20Training%20Convergence.pdf) | [dual](./paper/detr/Conditional%20DETR%20for%20Fast%20Training%20Convergence-dual.pdf) | [mono](./paper/detr/Conditional%20DETR%20for%20Fast%20Training%20Convergence-mono.pdf) | [ConditionalDETR](./code/ConditionalDETR) |
| 2022 | **DAB-DETR** | [DAB-DETR](./paper/detr/DAB-DETR.pdf) | [dual](./paper/detr/DAB-DETR-dual.pdf) | [mono](./paper/detr/DAB-DETR-mono.pdf) | [DAB-DETR](./code/DAB-DETR) |
| 2022 | **DN-DETR** | [DN-DETR](./paper/detr/DN-DETR.pdf) | [dual](./paper/detr/DN-DETR-dual.pdf) | [mono](./paper/detr/DN-DETR-mono.pdf) | |
| 2022 | AnchorDETR | | | | [AnchorDETR](./code/AnchorDETR) |
| 2023 | **DINO** | [DINO - DETR with Improved DeNoising Anchor](./paper/detr/DINO%20-%20DETR%20with%20Improved%20DeNoising%20Anchor.pdf) | [dual](./paper/detr/DINO%20-%20DETR%20with%20Improved%20DeNoising%20Anchor-dual.pdf) | [mono](./paper/detr/DINO%20-%20DETR%20with%20Improved%20DeNoising%20Anchor-mono.pdf) | [DINO](./code/DINO) |
| 2023 | **RT-DETR** | [RT-DETR](./paper/detr/RT-DETR.pdf) | [dual](./paper/detr/RT-DETR-dual.pdf) | [mono](./paper/detr/RT-DETR-mono.pdf) | [RT-DETR](./code/RT-DETR) |
| 2023 | RT-DETRv2 | [RT-DETRv2 - Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer](./paper/detr/RT-DETRv2%20-%20Improved%20Baseline%20with%20Bag-of-Freebies%20for%20Real-Time%20Detection%20Transformer.pdf) | [dual](./paper/detr/RT-DETRv2%20-%20Improved%20Baseline%20with%20Bag-of-Freebies%20for%20Real-Time%20Detection%20Transformer-dual.pdf) | [mono](./paper/detr/RT-DETRv2%20-%20Improved%20Baseline%20with%20Bag-of-Freebies%20for%20Real-Time%20Detection%20Transformer-mono.pdf) | |
| 2023 | RT-DETRv3 | [RT-DETRv3](./paper/detr/RT-DETRv3.pdf) | [dual](./paper/detr/RT-DETRv3-dual.pdf) | [mono](./paper/detr/RT-DETRv3-mono.pdf) | |
| 2023 | RT-DETRv4 | [RT-DETRv4 - Painlessly Furthering Real-Time Object Detection with Vision Foundation Models](./paper/detr/RT-DETRv4%20-%20%20Painlessly%20Furthering%20Real-Time%20Object%20Detection%20with%20Vision%20Foundation%20Models.pdf) | [dual](./paper/detr/RT-DETRv4%20-%20%20Painlessly%20Furthering%20Real-Time%20Object%20Detection%20with%20Vision%20Foundation%20Models-dual.pdf) | [mono](./paper/detr/RT-DETRv4%20-%20%20Painlessly%20Furthering%20Real-Time%20Object%20Detection%20with%20Vision%20Foundation%20Models-mono.pdf) | |
| 2023 | **D-FINE** | [D-FINE](./paper/detr/D-FINE.pdf) | [dual](./paper/detr/D-FINE-dual.pdf) | [mono](./paper/detr/D-FINE-mono.pdf) | [D-FINE](./code/D-FINE) |
| 2023 | DEIM | | | | [DEIM](./code/DEIM) |
| | LW-DETR | [LW-DETR .. A Transformer Replacement to](./paper/detr/LW-DETR%20..%20A%20Transformer%20Replacement%20to.pdf) | [dual](./paper/detr/LW-DETR%20..%20A%20Transformer%20Replacement%20to-dual.pdf) | [mono](./paper/detr/LW-DETR%20..%20A%20Transformer%20Replacement%20to-mono.pdf) | |
| | detrex | | | | [detrex](./code/detrex) |

### 第三部分：Backbone 网络（扩展阅读）

| 算法名称 | PDF链接 | 中英文对照版 | 中文版 | 代码仓库 |
|---------|---------|-------------|--------|---------|
| On the Relationship between Self-Attention and Convolutional Layers | [ON THE RELATIONSHIP BETWEEN SELF-ATTENTION](./paper/backbone/ON%20THE%20RELATIONSHIP%20BETWEEN%20SELF-ATTENTION.pdf) | [dual](./paper/backbone/ON%20THE%20RELATIONSHIP%20BETWEEN%20SELF-ATTENTION-dual.pdf) | [mono](./paper/backbone/ON%20THE%20RELATIONSHIP%20BETWEEN%20SELF-ATTENTION-mono.pdf) | |
| DINOv1 | [DINOv1 - Emerging Properties in Self-Supervised Vision Transformers](./paper/backbone/DINOv1%20-%20Emerging%20Properties%20in%20Self-Supervised%20Vision%20Transformers.pdf) | [dual](./paper/backbone/DINOv1%20-%20Emerging%20Properties%20in%20Self-Supervised%20Vision%20Transformers-dual.pdf) | [mono](./paper/backbone/DINOv1%20-%20Emerging%20Properties%20in%20Self-Supervised%20Vision%20Transformers-mono.pdf) | |
| DINOv2 | [DINOv2 - Learning Robust Visual Features](./paper/backbone/DINOv2%20-%20Learning%20Robust%20Visual%20Features.pdf) | [dual](./paper/backbone/DINOv2%20-%20Learning%20Robust%20Visual%20Features-dual.pdf) | [mono](./paper/backbone/DINOv2%20-%20Learning%20Robust%20Visual%20Features-mono.pdf) | |

### 第四部分：相关研究（扩展阅读）

| 年份 | 算法名称 | PDF链接 | 中英文对照版 | 中文版 | 代码仓库 |
|------|---------|---------|-------------|--------|---------|
| 2021 | UP-DETR | [UP-DETR](./paper/related/UP-DETR.pdf) | [dual](./paper/related/UP-DETR-dual.pdf) | [mono](./paper/related/UP-DETR-mono.pdf) | |
| 2021 | YOLOS | [YOLOS](./paper/related/YOLOS.pdf) | [dual](./paper/related/YOLOS-dual.pdf) | [mono](./paper/related/YOLOS-mono.pdf) | |
| 2021 | TSP | [TSP](./paper/related/TSP.pdf) | [dual](./paper/related/TSP-dual.pdf) | [mono](./paper/related/TSP-mono.pdf) | |
| 2021 | Rethinking Transformer-based Set Prediction (ICCV) | [Sun_Rethinking_Transformer-Based_Set_Prediction_for_Object_Detection_ICCV_2021_paper](./paper/related/Sun_Rethinking_Transformer-Based_Set_Prediction_for_Object_Detection_ICCV_2021_paper.pdf) | [dual](./paper/related/Sun_Rethinking_Transformer-Based_Set_Prediction_for_Object_Detection_ICCV_2021_paper-dual.pdf) | [mono](./paper/related/Sun_Rethinking_Transformer-Based_Set_Prediction_for_Object_Detection_ICCV_2021_paper-mono.pdf) | |
| 2021 | You Only Look at One Sequence (NeurIPS) | [NeurIPS-2021-you-only-look-at-one-sequence-rethinking-transformer-in-vision-through-object-detection-Paper](./paper/related/NeurIPS-2021-you-only-look-at-one-sequence-rethinking-transformer-in-vision-through-object-detection-Paper.pdf) | [dual](./paper/related/NeurIPS-2021-you-only-look-at-one-sequence-rethinking-transformer-in-vision-through-object-detection-Paper-dual.pdf) | [mono](./paper/related/NeurIPS-2021-you-only-look-at-one-sequence-rethinking-transformer-in-vision-through-object-detection-Paper-mono.pdf) | |
| 2021 | PIX2SEQ | [PIX2SEQ](./paper/related/PIX2SEQ.pdf) | [dual](./paper/related/PIX2SEQ-dual.pdf) | [mono](./paper/related/PIX2SEQ-mono.pdf) | |

## 说明

- **PDF 版本**: 原版、中英文对照版 (dual)、中文翻译版 (mono)
- **代码仓库**: 通过 Git Submodule 管理，位于 `code/` 目录
