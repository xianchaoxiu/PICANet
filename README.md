# PICA-Net

The code in this toolbox implements "PICANet: Physics-Informed Cascaded Asymmetric Network for Infrared Small Target Detection" by <i>J. Liu, Y. Han, X. Xiu, J. Zhang, W. Liu</i>.



## Overview

Infrared small targets occupy only a few pixels and often have weak contrast, incomplete contours, and limited semantic information. PICANet addresses these issues with three complementary components:

- **Hierarchical Prior Decoupling Module (HPDM):** extracts two physical constraint priors from the input image using frozen multi-directional feature aggregation calculation blocks (MFACBs).
  - `$\mathrm{PCP}_1$`: low-level geometric/spatial prior for target localization.
  - `$\mathrm{PCP}_2$`: high-level semantic/contour prior for feature embedding.
- **Dual Priors Interactive Fusion Module (DPIFM):** injects the two priors into multi-scale backbone features through bidirectional prior-guided feature enhancement (BPGFE).
- **Multi-level Cross-Feature Attention Module (MCFAM):** aligns high-level semantic features and low-level spatial details with a cascaded asymmetric ISC-A mechanism.

## Network framework

Add the PICANet framework figure at the following path:

![](C:\Users\25292\AppData\Roaming\marktext\images\2026-09-06-23-31-44-image.png)

The framework follows this processing pipeline:

```text
Input infrared image
          │
          ▼
┌─────────────────────────────────────────────┐
│ HPDM: Hierarchical Prior Decoupling Module  │
│  MFACBs → PH1/PH2 → PCP1, PCP2              │
└─────────────────────────────────────────────┘
          │ physical priors
          ▼
┌─────────────────────────────────────────────┐
│ Backbone (U-Net or ResNet-FPN)              │
└─────────────────────────────────────────────┘
          │ multi-scale features + priors
          ▼
┌─────────────────────────────────────────────┐
│ DPIFM: Dual Priors Interactive Fusion       │
│  BPGFE: prior-to-feature / feature-to-prior │
└─────────────────────────────────────────────┘
          │ refined multi-level features
          ▼
┌─────────────────────────────────────────────┐
│ MCFAM: Multi-level Cross-Feature Attention  │
│  ISC-A cascaded asymmetric alignment        │
└─────────────────────────────────────────────┘
          │
          ▼
Collaborative multi-level prediction → target maskPICANet is designed to be integrated with different backbones. The paper reports two variants:
```

- **PICANet (U):** PICANet integrated with U-Net.
- **PICANet (F):** PICANet integrated with a ResNet-FPN backbone.

## Main contributions

1. A plug-and-play physics-informed architecture that can be inserted into mainstream ISTD backbones.
2. A hierarchical prior decoupling strategy that separates low-level geometric information from high-level semantic information.
3. A dual-prior interactive fusion mechanism that uses physical priors as spatial and semantic guidance rather than simply concatenating handcrafted maps.
4. A cascaded asymmetric cross-feature attention mechanism that improves the alignment between semantic context and fine spatial detail.
5. A collaborative multi-level prediction strategy for combining predictions from different decoding stages.

## Installation

The reference implementation is based on Python and PyTorch.

```bash
git clone https://github.com/xianchaoxiu/PICANet.git
cd PICANet

conda create -n picanet python=3.10 -y
conda activate picanet
pip install -r requirements.txt
```

If the release uses a different environment file, replace the last command with the corresponding setup command, for example `pip install -e .`.

## Datasets

The experiments in the paper use the following public datasets:

- [NUDT-SIRST](https://github.com/YeRen123455/Infrared-Small-Target-Detection)
- [IRSTD-1k](https://github.com/RuiZhang97/ISNet)
- [SIRST-Aug](https://github.com/Tianfang-Zhang/AGPCNet)

After downloading the datasets, arrange them according to the repository's expected directory structure. A typical layout is:

```text
data/
├── NUDT-SIRST/
├── IRSTD-1k/
└── SIRST-Aug/
```

Update the dataset paths in the configuration file before training or evaluation.

## Training

The paper trains the models for 400 epochs with a batch size of 8. Replace the commands below with the actual entry points in the code release:

```bash
# Train PICANet with a U-Net backbone
python train_UNet.py

# Train PICANet with a ResNet-FPN backbone
python train_FPN.py 
```

The training objective combines multi-scale binary cross-entropy, Soft-IoU, and a masked physical-prior consistency loss:

```text
L_total = ω_bce(α) · L_mBCE
        + ω_iou(α) · L_SoftIoU
        + ω_prior · L_MaskedMSE
```

The current paper setting uses a fixed prior regularization weight
\(\omega_{\mathrm{prior}}=1.0\). The BCE and IoU weights are dynamically adjusted according to the training ratio \(\alpha\).

## Evaluation

```bash
python t_models.py 
```

The reported metrics are:

- mIoU (mean intersection over union)
- $F_1$-score
- $P_d$ (probability of detection)
- $F_a$ (false alarm rate)

## Acknowledgement
Please contact Y. Han for more details.
