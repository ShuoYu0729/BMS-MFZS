# BMS-MFZS

---

## Introduction

Reliable background suppression remains a key challenge in infrared imaging for space and aerial scientific visual learning. Infrared images are often affected by environmental variability, radiometric inconsistency, sensor noise, non-uniform background radiation, and star-like interference. These factors make it difficult for conventional models to learn stable background characteristics and maintain robust target-background discrimination.

This repository provides the core implementation of **BMS-MFZS**, a background modeling and suppression method based on **Multi-Feature Generalized Zero-Shot Learning**. The proposed method introduces Generalized Zero-Shot Learning into infrared background modeling. By jointly utilizing image features, semantic embeddings, and physical auxiliary information, the model learns transferable background representations and improves unseen-background discrimination capability under complex infrared conditions.

The method also introduces temporal and spectral physical features as auxiliary information within the GZSL framework. These auxiliary features improve target-background discrimination by providing motion continuity and physical response differences. A joint semantic-pixel background reconstruction module is further designed to combine global semantic background modeling with local pixel-level refinement.

---

## Main Components

BMS-MFZS contains the following modules:

- Image feature and semantic label feature acquisition
- Temporal auxiliary feature extraction
- Spectral auxiliary feature extraction
- GZSL-based background feature modeling
- Semantic-pixel background reconstruction
- Adaptive infrared background suppression
- Physical feature-space interpretability analysis

---

## Project Structure

```text
BMS-MFZS/
│
├── README.md
├── requirements.txt
│
├── train.py
├── model.py
├── test_zero_shot.py
├── con_test.py
│
├── image_feature_extraction.py
├── text_feature_extraction.py
├── time_feature_extraction.py
├── spectrum_feature_extraction.py
├── transformer_network.py
│
├── background_model.py
├── background_reconstruction.py
├── physical_feature_analysis.py
│
├── utils.py
├── bmp-jpg.py
│
├── checkpoints/
│   └── pretrained_model.pth
│
└── demo/
    ├── input/
    ├── reconstruction_results/
    └── suppression_results/
```

---

## Code Description

The following table gives a brief description of the released code files and their correspondence to the proposed BMS-MFZS framework.

### Core Framework

| File | Description |
|---|---|
| `model.py` | Defines the main BMS-MFZS framework, including image feature extraction, semantic embedding, temporal and spectral auxiliary feature fusion, GZSL-based background modeling, and adaptive background suppression. |
| `train.py` | Training script of BMS-MFZS. It loads training data, builds the network, computes optimization losses, updates parameters, and saves trained checkpoints. |
| `test_zero_shot.py` | Testing and generalized zero-shot inference script. It performs background modeling, unseen-category inference, and infrared background suppression evaluation. |
| `con_test.py` | Evaluation script for suppression performance comparison. It generates suppression results and computes contrast-related evaluation metrics. |
| `utils.py` | Utility functions used for data loading, feature processing, model initialization, evaluation, and result saving. |

### Feature Extraction Modules

| File | Description |
|---|---|
| `image_feature_extraction.py` | Implements infrared image feature extraction using convolutional feature encoding to capture grayscale distribution and local spatial structures. |
| `text_feature_extraction.py` | Implements semantic label feature extraction and semantic embedding construction for target and background categories. |
| `time_feature_extraction.py` | Implements temporal auxiliary feature extraction to model inter-frame motion continuity and temporal variation characteristics of infrared targets. |
| `spectrum_feature_extraction.py` | Implements spectral auxiliary feature extraction to enhance target-background discrimination using spectral response characteristics. |
| `transformer_network.py` | Transformer-based feature modeling module used for spectral sequence encoding and feature interaction learning. |

### Background Modeling and Reconstruction

| File | Description |
|---|---|
| `background_model.py` | Implements the GZSL-based background feature modeling module, including semantic mapping, background representation learning, and unseen background category expansion. |
| `background_reconstruction.py` | Implements semantic-pixel background reconstruction, including semantic-guided coarse reconstruction, soft-mask generation, residual refinement, pseudo-ground-truth generation, confidence-map estimation, and confidence-weighted reconstruction loss. |

### Analysis and Preprocessing

| File | Description |
|---|---|
| `physical_feature_analysis.py` | Performs physical auxiliary feature analysis, including t-SNE visualization and quantitative separability evaluation using Fisher Ratio, Silhouette Score, and Inter/Intra Class Distance Ratio. |
| `bmp-jpg.py` | Performs infrared image format conversion and basic preprocessing operations. |

---

## Methodology Correspondence

| Paper Module | Corresponding Code |
|---|---|
| Image Feature and Label Semantic Feature Acquisition | `image_feature_extraction.py`, `text_feature_extraction.py`, `model.py` |
| Auxiliary Features Acquisition | `time_feature_extraction.py`, `spectrum_feature_extraction.py`, `transformer_network.py` |
| Background Feature Model Establishment based on GZSL | `background_model.py`, `model.py` |
| Semantic-Pixel Background Reconstruction | `background_reconstruction.py` |
| Background Suppression | `test_zero_shot.py`, `con_test.py` |
| Physical Interpretability Analysis | `physical_feature_analysis.py` |

---

## Installation

```bash
git clone https://github.com/yourname/BMS-MFZS.git
cd BMS-MFZS

conda create -n bms_mfzs python=3.10
conda activate bms_mfzs

pip install -r requirements.txt
```

Recommended environment:

```text
Python >= 3.9
PyTorch >= 1.12
CUDA >= 11.3
```

Main dependencies:

```text
torch
torchvision
numpy
opencv-python
Pillow
matplotlib
scikit-learn
transformers
tqdm
```

---

## Training

Run:

```bash
python train.py
```

Before training, please modify the dataset paths and checkpoint saving paths in `train.py`.

Typical settings include:

```python
image_dir = "path/to/images"
json_dir = "path/to/labels"
save_path = "checkpoints/pretrained_model.pth"
```

---

## Testing

Run:

```bash
python "test_zero_shot.py"
```

Typical settings include:

```python
model_path = "checkpoints/pretrained_model.pth"
test_image_dir = "path/to/test/images"
result_dir = "demo/suppression_results"
```

---

## Background Reconstruction

The semantic-pixel background reconstruction module is implemented in:

```text
background_reconstruction.py
```

This module includes:

- Semantic-guided coarse background reconstruction
- Soft-mask generation
- Residual-based fine reconstruction
- Non-local pseudo-ground-truth generation
- Temporal pseudo-ground-truth generation
- Confidence-weighted reconstruction loss
- Adaptive infrared background suppression

Example usage:

```python
from background_reconstruction import SemanticPixelBackgroundReconstruction

reconstructor = SemanticPixelBackgroundReconstruction(kappa=5.0)

coarse_background, mask, refined_background = reconstructor(
    image=image_tensor,
    pixel_features=feature_tensor,
    background_bank=background_feature_bank
)
```

---

## Physical Feature Analysis

Run:

```bash
python physical_feature_analysis.py \
  --image_dir "path/to/images" \
  --json_dir "path/to/labels" \
  --model_path "checkpoints/pretrained_model.pth" \
  --save_dir "results/physical_analysis"
```

Supported feature spaces include:

- Base-model feature space
- Temporal-only feature space
- Spectral-only feature space
- Spectral-temporal feature space

Supported quantitative metrics include:

- Fisher Ratio
- Silhouette Score
- Inter/Intra Class Distance Ratio

---

## Evaluation Metrics

The repository supports the following infrared background suppression metrics:

| Metric | Description |
|---|---|
| BSF | Background Suppression Factor |
| SCR | Signal-to-Clutter Ratio |
| CNR | Contrast-to-Noise Ratio |
| CG | Contrast Gain |
| PSNR | Peak Signal-to-Noise Ratio |
| SSIM | Structural Similarity |
| MAE | Mean Absolute Error |
| FSIM | Feature Similarity |
| BRR | Background Retention Rate |

---

## Dataset and Simulation Notes

The experiments in the paper include:

- Simulated infrared datasets
- Semi-physical simulation datasets
- Public infrared sequence datasets
- Public infrared small-target datasets

The complete semi-physical simulation dataset and raw measurement data are not fully released in the current version. Representative configuration examples and demo samples will be gradually organized.

Released configuration examples may include:

- Background type
- Target size
- Target intensity range
- Motion pattern
- Initial SCR/SNR settings
- Star-field density settings
- Noise perturbation settings

