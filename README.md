# Reproducible Semantic Segmentation Experiment: Baseline U-Net vs. Dilated U-Net

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete implementation and empirical evaluation for **Assignment B: Build a Reproducible Segmentation Experiment** for the *Advanced Computer Vision (CS107203CS)* course.

---

## 1. Project Overview & Scientific Question

In semantic segmentation, capturing multi-scale context while preserving fine spatial boundary details is a primary architectural challenge. 

This experiment evaluates a controlled architectural modification:
> **Research Question:** *Does substituting standard bottleneck convolutions in a canonical U-Net with Dilated (Atrous) Convolutions improve semantic segmentation accuracy (mIoU) and boundary contour localization under strict parameter parity ($\Delta \text{Params} = 0$) and identical computational budgets?*

---

## 2. The Data Contract

| Contract Element | Specification |
| :--- | :--- |
| **Dataset** | Oxford-IIIT Pet Dataset (Parkhi et al., CVPR 2012) |
| **Target Task** | 3-Class Semantic Segmentation (Trimap) |
| **Class Mappings** | `Class 0`: Foreground (Pet Body)<br>`Class 1`: Background<br>`Class 2`: Boundary / Contour |
| **Ignored Labels** | None (`ignore_index = -100`). All 3 classes are scored to rigorously assess edge preservation. |
| **Splits & Seed** | **70% Train** (5,144 imgs) \| **15% Val** (1,102 imgs) \| **15% Test** (1,104 imgs)<br>Locked with deterministic seed: `42`. |
| **Resolution** | $256 \times 256$ pixels |
| **Transforms** | **Image:** Bilinear resize $\rightarrow$ ImageNet normalization ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$)<br>**Mask:** Nearest-neighbor resize $\rightarrow$ remapped to $\{0, 1, 2\}$ LongTensor. |
| **Augmentation** | Synchronized Random Horizontal Flip ($p = 0.5$) during training only. |

---

## 3. Architecture & Controlled Ablation

```
               PROPOSED ARCHITECTURAL COMPARISON

      [Input: 3 x 256 x 256]
                 │
           Encoder Level 1 (32 ch)  ────── Skip 1 ──────┐
                 │ MaxPool2d(2)                         │
           Encoder Level 2 (64 ch)  ────── Skip 2 ──┐   │
                 │ MaxPool2d(2)                     │   │
           Encoder Level 3 (128 ch) ─── Skip 3 ─┐   │   │
                 │ MaxPool2d(2)                 │   │   │
           Encoder Level 4 (256 ch) ─ Skip 4 ┐  │   │   │
                 │ MaxPool2d(2)              │  │   │   │
  ═══════════════╪═══════════════════════════╪══╪═══╪═══╪═════════════════════
                 ▼                           │  │   │   │
  ┌─────────────────────────────────────┐    │  │   │   │
  │         BOTTLENECK STAGE            │    │  │   │   │
  │ Baseline: Conv3x3 (r=1, p=1)        │    │  │   │   │
  │ Proposed: Dilated Conv3x3 (r=2, 4)  │    │  │   │   │
  └──────────────────┬──────────────────┘    │  │   │   │
  ═══════════════════╪═══════════════════════╪══╪═══╪═══╪═════════════════════
                     ▼                       │  │   │   │
           Decoder Level 4 (256 ch) ◄────────┘  │   │   │
                 │ ConvTranspose2d(2)           │   │   │
           Decoder Level 3 (128 ch) ◄───────────┘   │   │
                 │ ConvTranspose2d(2)               │   │
           Decoder Level 2 (64 ch)  ◄───────────────┘   │
                 │ ConvTranspose2d(2)                   │
           Decoder Level 1 (32 ch)  ◄───────────────────┘
                 │ Conv2d(1x1) Head
                 ▼
      [Output Logits: 3 x 256 x 256]
```

### Architectural Parity:
* **Baseline U-Net:** Standard U-Net (Ronneberger et al., 2015) with standard $3 \times 3$ convolutions ($r=1$) in the bottleneck.
* **Dilated U-Net (Proposed):** Replaces bottleneck convolutions with atrous convolutions ($r_1=2, p_1=2$ and $r_2=4, p_2=4$).
* **Exact Parameter Parity:** Both models contain **exactly 7,763,107 trainable parameters ($\Delta \text{Params} = 0$)**.

---

## 4. Controlled Training Budget

Both models were trained under identical conditions:
* **Epochs:** 15 epochs per model
* **Batch Size:** 16 images
* **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $= 10^{-4}$)
* **Learning Rate Schedule:** `CosineAnnealingLR` ($\eta_{\max} = 10^{-3} \rightarrow \eta_{\min} = 10^{-6}$)
* **Loss Function:** Multi-Class Cross-Entropy Loss
* **Stopping Rule:** Fixed 15 epochs with best checkpoint saved on Validation mIoU.

---

## 5. Empirical Results on the Unseen Test Set (1,104 Images)

The evaluation was conducted under standardized hardware conditions (batch size 1, 30 warm-up runs, 100 timed iterations using hardware synchronization):

| Metric | Baseline U-Net | Dilated U-Net (Proposed) | Improvement / Impact |
| :--- | :---: | :---: | :---: |
| **Trainable Parameters** | 7,763,107 | 7,763,107 | **0 (Exact 100% Parity)** |
| **Inference Latency ($B=1$)** | 14.13 ± 0.12 ms | 14.68 ± 0.12 ms | +0.55 ms (Negligible) |
| **Throughput (FPS)** | 70.8 FPS | 68.1 FPS | Real-time (>60 FPS) |
| **Test Mean IoU (mIoU)** | **0.7400** | **0.7560** | **+1.60% absolute gain (+160 bps)** |
| **Test Pixel Accuracy** | **0.9021** | **0.9100** | **+0.79% improvement** |
| **Foreground (Pet) IoU** | 0.8187 | **0.8359** | **+1.72% improvement** |
| **Background IoU** | 0.8987 | **0.9090** | **+1.03% improvement** |
| **Boundary Contour IoU** | 0.5026 | **0.5233** | **+2.07% improvement (Largest gain)** |

### Key Findings:
1. **Boundary Precision:** Dilated convolution achieved its highest relative gain on the **Boundary class (+2.07%)**, confirming that a broader receptive field helps resolve ambiguous silhouette pixels.
2. **Faster Convergence:** Dilated U-Net reached a validation mIoU of `0.7403` at Epoch 11, surpassing the Baseline's peak 15-epoch score (`0.7378`) four epochs earlier.
3. **Real-time Throughput:** Running at **68.1 FPS**, the proposed model maintains real-time processing capability with negligible latency overhead.

---

## 6. What the Experiment Does NOT Establish (Limitations)

1. **Resolution Dependence:** Results were established at $256 \times 256$. Dilation rates of $r=2, 4$ may need scaling for ultra-high resolutions ($1080\text{p}, 4\text{K}$).
2. **Domain Generalization:** Tested on domestic animals (Oxford-IIIT Pet). Findings do not guarantee identical gains in medical histopathology or autonomous driving.
3. **Gridding Effects:** Sparse sampling in atrous convolutions can lead to gridding artifacts on ultra-fine textures without Hybrid Dilated Convolution (HDC) smoothing.
4. **Pretrained Backbone Interaction:** Both models were trained from scratch; interaction with pre-trained vision backbones (e.g., ImageNet ResNet-50) remains unexamined.

---

## 7. Repository Structure

```
├── README.md                      # Comprehensive project documentation
├── requirements.txt               # Dependency specifications
├── result_colab.ipynb             # Master executable notebook with outputs & plots
├── Project_Report.pdf             # Formal project report (named by Roll Number)
├── image.png                      # High-resolution architectural diagram
└── src/                           # Modular source code
    ├── __init__.py
    ├── dataset.py                 # Data Contract & synchronized transforms
    ├── models.py                  # Baseline & Dilated U-Net implementations
    ├── metrics.py                 # Per-class IoU, mIoU & latency measurement
    ├── train.py                   # Standalone training script with JSON logging
    └── evaluate.py                # Standalone test evaluation script
```

---

## 8. How to Reproduce

### Quick Start via Colab:
Open and run `result_colab.ipynb` on Google Colab with a free T4 GPU.

### Local Execution:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train Baseline U-Net
python -m src.train --model baseline --epochs 15

# 3. Train Dilated U-Net
python -m src.train --model dilated --epochs 15

# 4. Run Evaluation on Test Set
python -m src.evaluate
```

---

## References
1. **Ronneberger, O., Fischer, P., & Brox, T. (2015).** *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI.
2. **Chen, L. C., et al. (2017).** *DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs.* IEEE TPAMI.
3. **Parkhi, O. M., et al. (2012).** *Cats and Dogs.* IEEE CVPR.
4. **Yu, F., & Koltun, V. (2016).** *Multi-Scale Context Aggregation by Dilated Convolutions.* ICLR.
