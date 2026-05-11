"""
model.py

Public implementation of BMS-MFZS.

This file corresponds to:
1. Image Feature and Label Semantic Feature Acquisition
2. Auxiliary Features Acquisition
3. Background Feature Model Establishment based on GZSL
4. Background Suppression interface
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (torch.norm(x, p=2, dim=dim, keepdim=True) + eps)


class Mish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class DilatedImageFeatureExtractor(nn.Module):
    """
    VGG-like image feature extractor with dilated convolutions.

    Corresponds to Eq. (1) in the methodology.
    """

    def __init__(self, in_channels: int = 3, out_dim: int = 2048):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, dilation=1),
            nn.InstanceNorm2d(32, affine=True),
            Mish(),
            nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2),
            nn.InstanceNorm2d(64, affine=True),
            Mish(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2),
            nn.InstanceNorm2d(128, affine=True),
            Mish(),
            nn.Conv2d(128, 256, kernel_size=3, padding=4, dilation=4),
            nn.InstanceNorm2d(256, affine=True),
            Mish(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=4, dilation=4),
            nn.InstanceNorm2d(512, affine=True),
            Mish(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.proj = nn.Linear(512, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x).flatten(1)
        feat = self.proj(feat)
        return feat


class SemanticLabelEncoder(nn.Module):
    """
    Semantic label feature encoder.

    For public release, this module uses learnable semantic embeddings.
    If BERT embeddings are available, users can replace the embedding table
    with pre-computed BERT label features.
    """

    def __init__(self, num_classes: int, text_dim: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, text_dim)
        self.proj = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            Mish(),
            nn.Linear(text_dim, text_dim),
        )

    def forward(self, label_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            label_ids: [B] or [B, C] multi-hot labels.

        Returns:
            semantic feature: [B, text_dim]
        """
        if label_ids.dim() == 1:
            emb = self.embedding(label_ids)
        else:
            weight = label_ids.float()
            emb_table = self.embedding.weight.unsqueeze(0)
            emb = torch.matmul(weight, emb_table.squeeze(0))
            denom = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
            emb = emb / denom

        return self.proj(emb)


class TemporalTCN(nn.Module):
    """
    Temporal feature extraction using causal dilated temporal convolutions.

    Corresponds to Eq. (2) and Eq. (3).
    """

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 1024, output_dim: int = 512):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.tcn = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=1),
            nn.InstanceNorm1d(hidden_dim, affine=True),
            Mish(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=2),
            nn.InstanceNorm1d(hidden_dim, affine=True),
            Mish(),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=3, padding=8, dilation=4),
            nn.InstanceNorm1d(output_dim, affine=True),
            Mish(),
        )

    def forward(self, image_seq_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_seq_features: [B, T, D]

        Returns:
            temporal features: [B, T, output_dim]
        """
        x = self.input_proj(image_seq_features)
        x = x.transpose(1, 2)
        x = self.tcn(x)

        # remove extra right-side length caused by causal padding approximation
        x = x[:, :, :image_seq_features.size(1)]
        x = x.transpose(1, 2)
        return x


class SpectralTransformerEncoder(nn.Module):
    """
    Spectral auxiliary feature encoder.

    Corresponds to Eq. (4) to Eq. (8).
    """

    def __init__(
        self,
        spectrum_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(spectrum_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectrum:
                [B, T, D] or [B, D]

        Returns:
            spectral feature: [B, T, hidden_dim] or [B, hidden_dim]
        """
        single_frame = False
        if spectrum.dim() == 2:
            spectrum = spectrum.unsqueeze(1)
            single_frame = True

        x = self.input_proj(spectrum)
        x = self.encoder(x)
        x = self.output_proj(x)

        if single_frame:
            x = x.squeeze(1)
        return x


class CrossAttentionAligner(nn.Module):
    """
    Cross-attention based feature alignment.

    Corresponds to Eq. (9).
    """

    def __init__(self, query_dim: int, kv_dim: int, hidden_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.q_proj = nn.Linear(query_dim, hidden_dim)
        self.k_proj = nn.Linear(kv_dim, hidden_dim)
        self.v_proj = nn.Linear(kv_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [B, D] or [B, T, D]
            key_value: [B, D] or [B, T, D]

        Returns:
            aligned feature: [B, hidden_dim] or [B, T, hidden_dim]
        """
        q_single = False
        if query.dim() == 2:
            query = query.unsqueeze(1)
            q_single = True
        if key_value.dim() == 2:
            key_value = key_value.unsqueeze(1)

        q = self.q_proj(query)
        k = self.k_proj(key_value)
        v = self.v_proj(key_value)

        out, _ = self.attn(q, k, v)
        out = self.norm(out + q)

        if q_single:
            out = out.squeeze(1)
        return out


class MultiModalSimilarityLoss(nn.Module):
    """
    Similarity-aware contrastive loss.

    Corresponds to Eq. (11) and the following contrastive loss.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.25, gamma: float = 0.25, tau: float = 0.07):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def similarity(self, image_feat: torch.Tensor, semantic_feat: torch.Tensor,
                   temporal_feat: torch.Tensor, spectral_feat: torch.Tensor) -> torch.Tensor:
        image_feat = l2_normalize(image_feat)
        semantic_feat = l2_normalize(semantic_feat)
        temporal_feat = l2_normalize(temporal_feat)
        spectral_feat = l2_normalize(spectral_feat)

        sim_il = torch.sum(image_feat * semantic_feat, dim=-1)
        sim_it = torch.sum(image_feat * temporal_feat, dim=-1)
        sim_is = torch.sum(image_feat * spectral_feat, dim=-1)

        return self.alpha * sim_il + self.beta * sim_it + self.gamma * sim_is

    def forward(self, image_feat: torch.Tensor, semantic_feat: torch.Tensor,
                temporal_feat: torch.Tensor, spectral_feat: torch.Tensor) -> torch.Tensor:
        image_feat = l2_normalize(image_feat)
        semantic_feat = l2_normalize(semantic_feat)
        temporal_feat = l2_normalize(temporal_feat)
        spectral_feat = l2_normalize(spectral_feat)

        logits = (
            self.alpha * image_feat @ semantic_feat.t()
            + self.beta * image_feat @ temporal_feat.t()
            + self.gamma * image_feat @ spectral_feat.t()
        ) / self.tau

        targets = torch.arange(image_feat.size(0), device=image_feat.device)
        return F.cross_entropy(logits, targets)


class BMSMFZS(nn.Module):
    """
    BMS-MFZS main network.

    This public version keeps the complete methodological structure:
    image feature extraction, semantic feature encoding, temporal branch,
    spectral branch, cross-attention alignment, GZSL mapping, and classification.
    """

    def __init__(
        self,
        num_classes: int,
        image_dim: int = 2048,
        semantic_dim: int = 512,
        temporal_dim: int = 512,
        spectral_dim: int = 512,
        hidden_dim: int = 512,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.image_encoder = DilatedImageFeatureExtractor(out_dim=image_dim)
        self.semantic_encoder = SemanticLabelEncoder(num_classes=num_classes, text_dim=semantic_dim)
        self.temporal_encoder = TemporalTCN(input_dim=image_dim, output_dim=temporal_dim)
        self.spectral_encoder = SpectralTransformerEncoder(spectrum_dim=spectral_dim, hidden_dim=spectral_dim)

        self.align_image = CrossAttentionAligner(image_dim, image_dim, hidden_dim)
        self.align_semantic = CrossAttentionAligner(semantic_dim, image_dim, hidden_dim)
        self.align_temporal = CrossAttentionAligner(temporal_dim, image_dim, hidden_dim)
        self.align_spectral = CrossAttentionAligner(spectral_dim, image_dim, hidden_dim)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            Mish(),
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.semantic_mapper = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            Mish(),
            nn.Linear(hidden_dim, semantic_dim),
        )

    def forward(
        self,
        images: torch.Tensor,
        label_ids: Optional[torch.Tensor] = None,
        spectrum: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: [B, T, C, H, W]
            label_ids: [B] or [B, C], optional
            spectrum: [B, T, 512] or [B, 512], optional

        Returns:
            dictionary containing logits and aligned features.
        """
        b, t, c, h, w = images.shape
        images_flat = images.reshape(b * t, c, h, w)

        image_feat_flat = self.image_encoder(images_flat)
        image_seq_feat = image_feat_flat.reshape(b, t, -1)
        image_feat = image_seq_feat.mean(dim=1)

        temporal_seq = self.temporal_encoder(image_seq_feat)
        temporal_feat = temporal_seq.mean(dim=1)

        if spectrum is None:
            spectrum = torch.zeros(b, t, 512, device=images.device)
        if spectrum.dim() == 2:
            spectrum = spectrum.unsqueeze(1).repeat(1, t, 1)
        spectral_seq = self.spectral_encoder(spectrum)
        spectral_feat = spectral_seq.mean(dim=1)

        if label_ids is None:
            label_ids = torch.zeros(b, dtype=torch.long, device=images.device)
        semantic_feat = self.semantic_encoder(label_ids)

        image_aligned = self.align_image(image_feat, image_feat)
        semantic_aligned = self.align_semantic(semantic_feat, image_feat)
        temporal_aligned = self.align_temporal(temporal_feat, image_feat)
        spectral_aligned = self.align_spectral(spectral_feat, image_feat)

        fused = torch.cat([image_aligned, semantic_aligned, temporal_aligned, spectral_aligned], dim=-1)
        fused = self.fusion(fused)

        logits = self.classifier(fused)
        mapped_semantic = self.semantic_mapper(fused)

        return {
            "logits": logits,
            "fused": fused,
            "mapped_semantic": mapped_semantic,
            "image_feature": image_aligned,
            "semantic_feature": semantic_aligned,
            "temporal_feature": temporal_aligned,
            "spectral_feature": spectral_aligned,
            "raw_image_feature": image_feat,
        }


if __name__ == "__main__":
    model = BMSMFZS(num_classes=6)
    images = torch.randn(2, 5, 3, 224, 224)
    labels = torch.tensor([0, 1])
    spectrum = torch.randn(2, 5, 512)

    out = model(images, labels, spectrum)
    for k, v in out.items():
        print(k, v.shape)