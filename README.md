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
