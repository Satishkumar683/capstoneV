# Cross-View Human Gait Recognition Through Attention Guided Residual Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://www.tensorflow.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey.svg)](#installation)
[![Dataset](https://img.shields.io/badge/Dataset-CASIA--B-green.svg)](http://www.cbsr.ia.ac.cn/english/GaitDatabases.asp)
[![Architecture](https://img.shields.io/badge/Backbone-ResNet--50%20%2B%20CBAM%20%2B%20HPM-orange.svg)](#architecture-overview)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A robust, two-stage deep learning framework for cross-view human gait recognition using **Attention Guided Residual Learning**. This framework couples a **ResNet-50** residual backbone with the **Convolutional Block Attention Module (CBAM)** and **Horizontal Pyramid Mapping (HPM)** to achieve superior recognition accuracy across varying camera viewpoints, apparel variations, and carrying conditions.

Fully integrated and optimized for **Windows OS** with native **Visual Studio Code** support, multi-core CPU and NVIDIA GPU auto-configuration, and automatic dataset extraction.

---

## Table of Contents
- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
- [Supported Gait Representations](#supported-gait-representations)
- [Dataset (CASIA-B)](#dataset-casia-b)
- [Project Structure](#project-structure)
- [Installation & Environment Setup](#installation--environment-setup)
- [How to Run](#how-to-run)
  - [Option 1: In Visual Studio Code (F5)](#option-1-in-visual-studio-code-f5-recommended)
  - [Option 2: Windows One-Click Batch Runner](#option-2-windows-one-click-batch-runner)
  - [Option 3: PowerShell / Command Prompt (CLI)](#option-3-powershell--command-prompt-cli)
- [Detailed Model Components](#detailed-model-components)
  - [ResNet-50 Backbone](#1-resnet-50-backbone)
  - [CBAM (Channel + Spatial Attention)](#2-cbam-convolutional-block-attention-module)
  - [Horizontal Pyramid Mapping (HPM)](#3-horizontal-pyramid-mapping-hpm)
- [Experimental Results](#experimental-results)
- [Applications](#applications)
- [Future Work](#future-work)
- [Acknowledgements & References](#acknowledgements--references)

---

## Key Features

- **Cross-View Invariance**: Robust identification across 11 camera viewpoints ($0^\circ$ to $180^\circ$ in $18^\circ$ intervals).
- **Attention-Guided Residual Learning**: Combines deep residual representations (ResNet-50) with adaptive feature recalibration via CBAM.
- **Dual Attention Mechanism**: Dynamically focuses on salient anatomical regions using channel and spatial attention maps.
- **Multi-Scale Part Representation (HPM)**: Horizontally strips feature maps (6 parts) to capture fine-grained limb movements and global body silhouette simultaneously.
- **Multi-Template Support**: Native pipelines for **GEI** (Gait Energy Image), **GMI** (Gait Motion Image), and **MFEI** (Motion Flow Energy Image).
- **Two-Stage Transfer Protocol with Few-Shot Adaptation**: Pre-training on Subjects 1–74, gallery transfer to Subjects 75–124, and few-shot fine-tuning on bag (`bg`) and coat (`cl`) variations.
- **Test-Time Augmentation (TTA) & BatchNorm Adaptation**: Recalibrates running statistics and evaluates multi-transformed inputs for enhanced test accuracy.
- **Windows-Native & VS Code Ready**: Auto-extracts `Casia-B.zip`, enables GPU memory growth, and generates formatted Excel (`.xlsx`) workbooks with embedded accuracy plots.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STAGE 1: GAIT TEMPLATE GENERATION                      │
│                                                                             │
│   Video / Silhouette Sequences ──► Aspect Ratio & Cycle Detection           │
│                                           │                                 │
│                                           ▼                                 │
│                   Temporal Aggregation: [ GEI | GMI | MFEI ]                │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: ATTENTION-GUIDED DEEP LEARNING                  │
│                                                                             │
│      Gait Template ──► [ ResNet-50 Residual Feature Extractor ]             │
│                                           │                                 │
│                                           ▼                                 │
│                        [ CBAM Dual Attention Module ]                       │
│                        ├─ Channel Attention (What to focus on)              │
│                        └─ Spatial Attention (Where to focus)                │
│                                           │                                 │
│                                           ▼                                 │
│                     [ Horizontal Pyramid Mapping (HPM) ]                    │
│                      (Multi-scale horizontal stripe pooling)                │
│                                           │                                 │
│                                           ▼                                 │
│                     [ Fully-Connected Metric Embedding Layer ]              │
│                                           │                                 │
│                                           ▼                                 │
│                       Cosine / Euclidean Distance Metric                    │
│                                  (Match Rank)                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Supported Gait Representations

| Template | Name | Formulation / Characteristic | Primary Strength |
| :---: | :---: | :--- | :--- |
| **GEI** | **Gait Energy Image** | Temporal average of binary silhouettes across a full gait cycle:<br>$$\text{GEI}(x, y) = \frac{1}{N}\sum_{t=1}^{N} B(x, y, t)$$ | Preserves static silhouette shape and dynamic walking envelope with minimal computational overhead. |
| **GMI** | **Gait Motion Image** | Weights silhouette frames according to motion progression and directional transitions. | Highlights temporal velocity and limb acceleration across step phases. |
| **MFEI**| **Motion Flow Energy Image** | Fuses dense optical flow vectors ($u, v$) with temporal silhouette energy maps. | Distinguishes subtle stride dynamics even under clothing occlusion. |

---

## Dataset (CASIA-B)

The model is developed and benchmarked on the benchmark **CASIA-B Multiview Gait Dataset**:
- **Subjects**: 124 individuals.
- **Angles**: 11 viewpoints per subject ($0^\circ, 18^\circ, 36^\circ, 54^\circ, 72^\circ, 90^\circ, 108^\circ, 126^\circ, 144^\circ, 162^\circ, 180^\circ$).
- **Walking Variations**:
  - **NM (Normal Walk)**: 6 sequences per subject (NM-01 to NM-06).
  - **BG (Carrying Bag)**: 2 sequences per subject (BG-01 to BG-02).
  - **CL (Wearing Coat)**: 2 sequences per subject (CL-01 to CL-02).
- **Archive Format**: Place `Casia-B.zip` directly in the project root; the pipeline automatically extracts it into `./dataset/` upon first run.

---

## Project Structure

```
capstoneV/
├── .vscode/
│   ├── launch.json          # Pre-configured VS Code Run/Debug profiles (F5)
│   └── settings.json        # Workspace execution and IntelliSense settings
│
├── models/
│   ├── __init__.py
│   ├── attention_module.py  # CBAM: Channel and Spatial Attention
│   ├── hpm.py               # Horizontal Pyramid Mapping (6-parts)
│   └── resnet50_cbam.py     # Complete ResNet-50 + CBAM + HPM model builder
│
├── training/
│   ├── __init__.py
│   ├── utils.py             # Windows auto-unzip, tf.data loader, TTA, BN adapt
│   ├── train.py             # Two-stage training and few-shot fine-tuning
│   └── test.py              # Cross-view evaluation matrix & Excel export
│
├── outputs/
│   ├── checkpoints/         # Saved model weights (.weights.h5)
│   └── results/             # Accuracy plots (.png) and Excel sheets (.xlsx)
│
├── Casia-B.zip              # Dataset archive
├── main.py                  # Master command-line driver
├── run_windows.bat          # 1-click Windows interactive launcher
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```

---

## Installation & Environment Setup

### 1. Clone the Project
```bash
git clone https://github.com/Satishkumar683/capstoneV.git
cd capstoneV
```

### 2. Create Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## How to Run

### Option 1: In Visual Studio Code (F5) — *Recommended*
1. Open this folder in VS Code (`File` $\rightarrow$ `Open Folder...`).
2. Press **`Ctrl + Shift + D`** (or click the **Run and Debug** icon on the left bar).
3. Select any profile from the top dropdown and press **`F5`** (or click the green Play button):
   * 🟢 **`Python: Fast Test Run (2 Epochs, CPU)`**: Quick sanity check on 2 epochs.
   * 🟢 **`Python: Balanced Run (12 Epochs)`**: Optimal 12-epoch balanced run (~1 to 1.5 hrs).
   * 🟢 **`Python: Full Pipeline (25 Epochs)`**: Full 25-epoch benchmark training on CASIA-B.
   * 🟢 **`Python: Evaluation Only`**: Runs cross-view evaluation using saved weights.

### Option 2: Windows One-Click Batch Runner
Double-click [`run_windows.bat`](file:///c:/Users/raman/Downloads/capstoneV-main/run_windows.bat) or run from PowerShell:
```cmd
.\run_windows.bat
```
Select `1` for balanced 12-epoch training, `2` for evaluation only, or `3` for a fast 2-epoch test run.

### Option 3: PowerShell / Command Prompt (CLI)
* **Balanced 12-Epoch Training & Evaluation (Default)**:
  ```powershell
  python main.py --mode all
  ```
* **Full Benchmark (25 Epochs)**:
  ```powershell
  python main.py --mode all --epochs 25 --fs_epochs 10
  ```
* **Fast Test Run on CPU (2 Epochs)**:
  ```powershell
  python main.py --mode all --cpu --epochs 2 --fs_epochs 1
  ```
* **Evaluation Only (using pre-trained weights)**:
  ```powershell
  python main.py --mode eval --weights outputs/checkpoints/final_fewshot_model.weights.h5
  ```

### Smart Checkpointing & Auto-Resume
The training pipeline automatically saves checkpoints to `outputs/checkpoints/`:
* `stage1_model.weights.h5`: Saved upon Stage 1 completion.
* `stage2_model.weights.h5`: Saved upon Stage 2 completion.
* `final_fewshot_model.weights.h5`: Saved upon Few-Shot adaptation completion.
* **Auto-Resume**: If training is ever interrupted, running the command again will automatically detect the saved checkpoints, load the weights, and resume seamlessly from the next stage without repeating previous stages.

---

## Detailed Model Components

### 1. ResNet-50 Backbone
Extracts rich hierarchical 2D representations from the gait template. Residual shortcut connections allow gradients to backpropagate cleanly without vanishing across deep layers.

### 2. CBAM (Convolutional Block Attention Module)
Applies sequential attention along two independent dimensions:
* **Channel Attention**: Employs both Average-Pooling and Max-Pooling followed by a shared MLP to evaluate **what** feature maps convey unique identity characteristics.
* **Spatial Attention**: Gathers inter-channel statistics through pooling operations and a convolutional layer with a $7\times 7$ kernel to emphasize **where** salient bodily movement is occurring (e.g., knee bends, arm swing trajectories).

### 3. Horizontal Pyramid Mapping (HPM)
Human walking patterns have strong vertical structural dependency (head $\rightarrow$ torso $\rightarrow$ legs $\rightarrow$ feet):
* HPM slices the high-level feature map into 6 horizontal stripes.
* Global Average Pooling and Global Max Pooling are applied per stripe, capturing fine-grained local limb kinematics without losing global body proportions.

---

## Experimental Results

The framework demonstrates superior recognition performance on CASIA-B under the two-stage transfer protocol (Subjects 001–074 for Stage 1 pretraining; Subjects 075–124 for Stage 2 & Few-Shot adaptation across all 11 view angles):

| Condition | Description | Probe Sequences | Gallery Reference | Diagonal (Identical View) | Mean Cross-View Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Normal Walk (`NM`)** | Natural clothing & gait | `nm-05`, `nm-06` | `nm-01` to `nm-04` | **98.75%** | **90.16%** |
| **Carrying Bag (`BG`)** | Backpack / shoulder bag | `bg-01`, `bg-02` | `nm-01` to `nm-04` | **95.33%** | **86.29%** |
| **Wearing Coat (`CL`)** | Bulky apparel occlusion | `cl-01`, `cl-02` | `nm-01` to `nm-04` | **89.42%** | **79.26%** |

### Generated Output Artifacts
All reports and high-resolution curves are saved in [`outputs/results/`](outputs/results/):
* 📊 **[`results_nm_05_06.xlsx`](outputs/results/results_nm_05_06.xlsx)**: Formatted $11 \times 11$ cross-view accuracy matrix with embedded plot.
* 📊 **[`results_bg_01_02.xlsx`](outputs/results/results_bg_01_02.xlsx)**: Bag carrying cross-view evaluation workbook.
* 📊 **[`results_cl_01_02.xlsx`](outputs/results/results_cl_01_02.xlsx)**: Coat wearing cross-view evaluation workbook.
* 📊 **[`summary_metrics.xlsx`](outputs/results/summary_metrics.xlsx)**: Comprehensive condition comparison table.
* 📈 **[`crossview_comparison_heatmap.png`](outputs/results/crossview_comparison_heatmap.png)**: Multi-panel cross-view heatmap across NM, BG, and CL conditions.

---

## Applications

* **Long-Range Surveillance & Law Enforcement**: Suspect identification in low-resolution CCTV footage where facial features are obscured, masked, or distant.
* **Person Re-Identification (Re-ID)**: Tracking target individuals across disparate non-overlapping cameras in airports, stadiums, and train stations.
* **Contactless Access Control**: Non-intrusive authentication for smart buildings and secure entry points.
* **Clinical Gait Assessment**: Quantitative monitoring of pathological locomotion, recovery tracking, and sports biomechanics.

---

## Acknowledgements & References

1. **CASIA-B Dataset**: Provided by the Center for Biometrics and Security Research (CBSR), Institute of Automation, Chinese Academy of Sciences.
2. **CBAM**: Woo et al., *"CBAM: Convolutional Block Attention Module"*, ECCV 2018.
3. **GaitSet / HPM**: Chao et al., *"GaitSet: Regarding Gait as a Set for Cross-View Gait Recognition"*, AAAI 2019.
4. **ResNet**: He et al., *"Deep Residual Learning for Image Recognition"*, CVPR 2016.

---

## License

This project is licensed under the [MIT License](LICENSE).
