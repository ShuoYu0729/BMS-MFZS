# BMS-MFZS

Background Modeling and Suppression based on Multi-Feature Generalized Zero-Shot Learning for infrared weak-target images.

## Introduction

Reliable background suppression remains a key challenge in infrared imaging for space and aerial scientific visual learning. Infrared images are often affected by environmental variability, radiometric inconsistency, sensor noise, non-uniform background radiation, and star-like interference. These factors make it difficult for conventional models to learn stable background characteristics and maintain robust target-background discrimination.

This repository provides the core implementation of **BMS-MFZS**, a background modeling and suppression method based on **Multi-Feature Generalized Zero-Shot Learning**. The proposed method introduces Generalized Zero-Shot Learning into infrared background modeling. By using target features and seen background features, the model infers background representations and suppresses the influence of complex backgrounds on target recognition.

The method also introduces temporal and spectral physical features as auxiliary information within the GZSL framework. These auxiliary features improve target-background discrimination by providing motion continuity and physical response differences. A joint semantic-pixel background reconstruction module is further designed to combine global semantic background modeling with local pixel-level refinement.

The released code includes the main model, training and testing scripts, physical auxiliary feature analysis, semantic-pixel background reconstruction, and background suppression evaluation.

## Main Components

BMS-MFZS contains the following modules:

- Image feature and semantic label feature acquisition
- Temporal auxiliary feature extraction
- Spectral auxiliary feature extraction
- GZSL-based background feature modeling
- Semantic-pixel background reconstruction
- Adaptive infrared background suppression
- Physical feature-space interpretability analysis

## Project Structure

```text
BMS-MFZS/
│
├── README.md
├── requirements.txt
│
├── train.py
├── model.py
├── test-zero shot.py
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

## Code Description

### Core Framework

| File | Description |
|---|---|
| `model.py` | Defines the main BMS-MFZS network. It integrates image features, semantic label features, temporal features, spectral features, cross-modal alignment, GZSL mapping, and final background suppression branches. |
| `train.py` | Training script for BMS-MFZS. It loads the training data, builds the model, computes the training loss, updates network parameters, and saves trained checkpoints. |
| `test-zero shot.py` | Testing script for generalized zero-shot inference. It loads the trained model, predicts seen and unseen background categories, and outputs background suppression results. |
| `con_test.py` | Evaluation and comparison script for background suppression. It is mainly used to calculate contrast-related indicators and generate suppression results for experimental comparison. |
| `utils.py` | Provides common utility functions used in training, testing, data loading, feature processing, and result saving. |

### Feature Extraction Modules

| File | Description |
|---|---|
| `image_feature_extraction.py` | Implements infrared image feature extraction. The module extracts spatial grayscale and local structural features from infrared images using convolutional feature encoding. |
| `text_feature_extraction.py` | Implements semantic label feature extraction. Target and background labels are embedded into a semantic feature space and used for GZSL-based background modeling. |
| `time_feature_extraction.py` | Implements temporal feature extraction. The module models inter-frame variation and temporal continuity of infrared targets and backgrounds. |
| `spectrum_feature_extraction.py` | Implements spectral auxiliary feature extraction. It encodes spectral or emissivity-related physical features to enhance target-background discrimination. |
| `transformer_network.py` | Provides the Transformer-based encoder used for spectral sequence modeling and feature representation learning. |

### Background Modeling and Reconstruction

| File | Description |
|---|---|
| `background_model.py` | Implements the background feature modeling module. It constructs background feature mappings under the GZSL framework and updates background representations using newly identified unseen categories. |
| `background_reconstruction.py` | Implements the semantic-pixel background reconstruction module. It includes semantic-guided coarse reconstruction, soft-mask generation, residual refinement, pseudo-ground-truth generation, confidence-map estimation, and confidence-weighted reconstruction loss. |

### Suppression, Analysis, and Preprocessing

| File | Description |
|---|---|
| `physical_feature_analysis.py` | Performs physical auxiliary feature analysis. It extracts image, temporal, spectral, and spectral-temporal features, generates t-SNE visualizations, and computes feature separability metrics such as Fisher Ratio, Silhouette Score, and Inter/Intra Class Distance Ratio. |
| `bmp-jpg.py` | Converts image formats and performs simple preprocessing for infrared image inputs. |
