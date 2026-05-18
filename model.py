from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import BertModel, BertTokenizer
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    BertModel = None
    BertTokenizer = None
    _TRANSFORMERS_AVAILABLE = False


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (torch.norm(x, p=2, dim=dim, keepdim=True) + eps)


class Mish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class DilatedImageFeatureExtractor(nn.Module):
    """
    VGG-like dilated convolution image feature extractor.

    Input:
        x: [B, C, H, W]
    Output:
        global_feature: [B, out_dim]
        pixel_feature: [B, pixel_dim, H/4, W/4]
    """

    def __init__(self, in_channels: int = 3, out_dim: int = 2048, pixel_dim: int = 512):
        super().__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, dilation=1),
            nn.InstanceNorm2d(32, affine=True),
            Mish(),
            nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2),
            nn.InstanceNorm2d(64, affine=True),
            Mish(),
            nn.MaxPool2d(2),
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2),
            nn.InstanceNorm2d(128, affine=True),
            Mish(),
            nn.Conv2d(128, 256, kernel_size=3, padding=4, dilation=4),
            nn.InstanceNorm2d(256, affine=True),
            Mish(),
            nn.MaxPool2d(2),
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(256, pixel_dim, kernel_size=3, padding=4, dilation=4),
            nn.InstanceNorm2d(pixel_dim, affine=True),
            Mish(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(pixel_dim, out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stage1(x)
        x = self.stage2(x)
        pixel_feature = self.stage3(x)
        global_feature = self.pool(pixel_feature).flatten(1)
        global_feature = self.proj(global_feature)
        return global_feature, pixel_feature


class SemanticLabelEncoder(nn.Module):
    """
    Semantic label encoder.

    The Methodology states that label semantic features are obtained using BERT.
    This class uses HuggingFace BERT when available. For machines without the
    transformers package or local BERT weights, it automatically falls back to a
    learnable embedding so the code remains executable. For manuscript-consistent
    experiments, install transformers and set bert_model_name to a local or
    downloadable BERT model.
    """

    def __init__(
        self,
        num_classes: int,
        output_dim: int = 512,
        bert_model_name: str = "bert-base-uncased",
        use_bert: bool = True,
        freeze_bert: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.output_dim = output_dim
        self.use_bert = use_bert and _TRANSFORMERS_AVAILABLE

        if self.use_bert:
            self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
            self.bert = BertModel.from_pretrained(bert_model_name)
            if freeze_bert:
                for p in self.bert.parameters():
                    p.requires_grad = False
            bert_dim = self.bert.config.hidden_size
            self.proj = nn.Sequential(
                nn.Linear(bert_dim, output_dim),
                Mish(),
                nn.Linear(output_dim, output_dim),
            )
        else:
            self.embedding = nn.Embedding(num_classes, output_dim)
            self.proj = nn.Sequential(
                nn.Linear(output_dim, output_dim),
                Mish(),
                nn.Linear(output_dim, output_dim),
            )

    def forward(
        self,
        label_ids: Optional[torch.Tensor] = None,
        label_texts: Optional[Sequence[str]] = None,
    ) -> torch.Tensor:
        if self.use_bert:
            if label_texts is None:
                if label_ids is None:
                    raise ValueError("Either label_texts or label_ids must be provided.")
                label_texts = [f"class {int(i)}" for i in label_ids.detach().cpu().view(-1).tolist()]

            device = next(self.parameters()).device
            encoded = self.tokenizer(
                list(label_texts),
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            outputs = self.bert(**encoded)
            cls_feature = outputs.last_hidden_state[:, 0, :]
            return self.proj(cls_feature)

        if label_ids is None:
            if label_texts is None:
                raise ValueError("Either label_ids or label_texts must be provided.")
            # Stable hash fallback for text labels.
            ids = [abs(hash(t)) % self.num_classes for t in label_texts]
            label_ids = torch.tensor(ids, dtype=torch.long, device=next(self.parameters()).device)

        if label_ids.dim() == 1:
            emb = self.embedding(label_ids.long())
        else:
            weight = label_ids.float()
            emb = torch.matmul(weight, self.embedding.weight)
            denom = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
            emb = emb / denom
        return self.proj(emb)


class TemporalTCN(nn.Module):
    """
    Temporal feature extraction using causal dilated temporal convolutions.

    Input:
        image_seq_features: [B, T, D]
    Output:
        temporal_feature_seq: [B, T, output_dim]
    """

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 1024, output_dim: int = 512):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2)
        self.conv3 = nn.Conv1d(hidden_dim, output_dim, kernel_size=3, dilation=4)
        self.norm1 = nn.InstanceNorm1d(hidden_dim, affine=True)
        self.norm2 = nn.InstanceNorm1d(hidden_dim, affine=True)
        self.norm3 = nn.InstanceNorm1d(output_dim, affine=True)
        self.act = Mish()

    @staticmethod
    def _causal_conv(x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        left_pad = (conv.kernel_size[0] - 1) * conv.dilation[0]
        x = F.pad(x, (left_pad, 0))
        return conv(x)

    def forward(self, image_seq_features: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(image_seq_features)  # [B,T,H]
        x = x.transpose(1, 2)  # [B,H,T]
        x = self.act(self.norm1(self._causal_conv(x, self.conv1)))
        x = self.act(self.norm2(self._causal_conv(x, self.conv2)))
        x = self.act(self.norm3(self._causal_conv(x, self.conv3)))
        return x.transpose(1, 2)


class SpectralTransformerEncoder(nn.Module):
    """
    Transformer encoder for spectral features.

    Input:
        spectrum: [B, T, D] or [B, D]
    Output:
        spectral feature sequence [B, T, hidden_dim] or [B, hidden_dim]
    """

    def __init__(
        self,
        spectrum_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.input_proj = nn.Linear(spectrum_dim, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
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
        single_frame = False
        if spectrum.dim() == 2:
            spectrum = spectrum.unsqueeze(1)
            single_frame = True

        x = self.input_proj(spectrum)
        t = x.size(1)
        x = x + self.position_embedding[:, :t, :]
        x = self.encoder(x)
        x = self.output_proj(x)

        if single_frame:
            x = x.squeeze(1)
        return x


class CrossAttentionAligner(nn.Module):
    """
    Cross-attention alignment module M_CA(Q, K).
    """

    def __init__(self, query_dim: int, kv_dim: int, hidden_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.q_proj = nn.Linear(query_dim, hidden_dim)
        self.k_proj = nn.Linear(kv_dim, hidden_dim)
        self.v_proj = nn.Linear(kv_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, alpha: float = 0.5, beta: float = 0.25, gamma: float = 0.25, tau: float = 0.07):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def forward(
        self,
        image_feat: torch.Tensor,
        semantic_feat: torch.Tensor,
        temporal_feat: torch.Tensor,
        spectral_feat: torch.Tensor,
    ) -> torch.Tensor:
        image_feat = l2_normalize(image_feat)
        semantic_feat = l2_normalize(semantic_feat)
        temporal_feat = l2_normalize(temporal_feat)
        spectral_feat = l2_normalize(spectral_feat)

        sim_il = image_feat @ semantic_feat.t()
        sim_it = image_feat @ temporal_feat.t()
        sim_is = image_feat @ spectral_feat.t()
        sim = self.alpha * sim_il + self.beta * sim_it + self.gamma * sim_is
        sim = sim / self.tau

        target = torch.arange(image_feat.size(0), device=image_feat.device)
        return F.cross_entropy(sim, target)


class BMSMFZS(nn.Module):
    """
    Background Modeling and Suppression Method based on Multi-Feature GZSL.
    """

    def __init__(
        self,
        num_classes: int,
        image_dim: int = 2048,
        hidden_dim: int = 512,
        spectrum_dim: int = 512,
        image_channels: int = 3,
        bert_model_name: str = "bert-base-uncased",
        use_bert: bool = True,
        seen_threshold: float = 0.75,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.seen_threshold = seen_threshold

        self.image_encoder = DilatedImageFeatureExtractor(
            in_channels=image_channels,
            out_dim=image_dim,
            pixel_dim=hidden_dim,
        )
        self.semantic_encoder = SemanticLabelEncoder(
            num_classes=num_classes,
            output_dim=hidden_dim,
            bert_model_name=bert_model_name,
            use_bert=use_bert,
        )
        self.temporal_encoder = TemporalTCN(input_dim=image_dim, hidden_dim=1024, output_dim=hidden_dim)
        self.spectral_encoder = SpectralTransformerEncoder(spectrum_dim=spectrum_dim, hidden_dim=hidden_dim)

        self.align_image = CrossAttentionAligner(image_dim, image_dim, hidden_dim)
        self.align_semantic = CrossAttentionAligner(hidden_dim, image_dim, hidden_dim)
        self.align_temporal = CrossAttentionAligner(hidden_dim, image_dim, hidden_dim)
        self.align_spectral = CrossAttentionAligner(hidden_dim, image_dim, hidden_dim)

        self.multimodal_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            Mish(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            Mish(),
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

        # Semantic/background prototype bank used for threshold-based seen/unseen decision
        # and semantic-pixel reconstruction. It is learnable and updated during training.
        self.background_feature_bank = nn.Parameter(torch.randn(num_classes, hidden_dim) * 0.02)

        self.pixel_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

    @staticmethod
    def identity_auxiliary(batch_size: int, sequence_length: int, feature_dim: int, device: torch.device) -> torch.Tensor:
        x = torch.zeros(batch_size, sequence_length, feature_dim, device=device)
        diag_len = min(sequence_length, feature_dim)
        for i in range(diag_len):
            x[:, i, i] = 1.0
        return x

    def forward(
        self,
        image_sequence: torch.Tensor,
        label_ids: Optional[torch.Tensor] = None,
        spectrum: Optional[torch.Tensor] = None,
        label_texts: Optional[Sequence[str]] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            image_sequence: [B, T, C, H, W]
            label_ids: [B] or [B, num_classes]
            spectrum: [B, T, 512] or None. If None, identity-like auxiliary vectors are used.
            label_texts: optional semantic label texts for BERT.
            threshold: seen/unseen threshold. If None, self.seen_threshold is used.
        """
        if image_sequence.dim() != 5:
            raise ValueError("image_sequence must have shape [B, T, C, H, W].")

        b, t, c, h, w = image_sequence.shape
        device = image_sequence.device

        global_features = []
        pixel_features = []
        for ti in range(t):
            g, p = self.image_encoder(image_sequence[:, ti])
            global_features.append(g)
            pixel_features.append(p)

        image_seq_features = torch.stack(global_features, dim=1)  # [B,T,2048]
        current_image_feature = image_seq_features[:, -1]  # i'
        current_pixel_feature = pixel_features[-1]  # [B,512,H/4,W/4]

        semantic_feature_raw = self.semantic_encoder(label_ids=label_ids, label_texts=label_texts)  # l_la'
        temporal_seq = self.temporal_encoder(image_seq_features)  # ti'
        temporal_feature_raw = temporal_seq[:, -1]

        if spectrum is None:
            spectrum = self.identity_auxiliary(b, t, 512, device)
        spectral_seq = self.spectral_encoder(spectrum)  # se'
        spectral_feature_raw = spectral_seq[:, -1] if spectral_seq.dim() == 3 else spectral_seq

        #cross-attention feature alignment guided by image features.
        aligned_image = self.align_image(current_image_feature, current_image_feature)
        aligned_semantic = self.align_semantic(semantic_feature_raw, current_image_feature)
        aligned_temporal = self.align_temporal(temporal_feature_raw, current_image_feature)
        aligned_spectral = self.align_spectral(spectral_feature_raw, current_image_feature)

        # explicit multimodal representation F_ms = [i, l, t, s].
        multimodal_representation = torch.cat(
            [aligned_image, aligned_semantic, aligned_temporal, aligned_spectral], dim=-1
        )
        fused_feature = self.multimodal_fusion(multimodal_representation)
        logits = self.classifier(fused_feature)

        #threshold-based seen/unseen decision.
        prototypes = l2_normalize(self.background_feature_bank, dim=-1)
        query = l2_normalize(fused_feature, dim=-1)
        similarity_to_bank = query @ prototypes.t()
        max_similarity, predicted_class = torch.max(similarity_to_bank, dim=1)
        th = self.seen_threshold if threshold is None else threshold
        seen_mask = max_similarity > th
        unseen_mask = ~seen_mask

        pixel_feature_aligned = self.pixel_proj(current_pixel_feature)

        return {
            "logits": logits,
            "image_feature": aligned_image,
            "semantic_feature": aligned_semantic,
            "temporal_feature": aligned_temporal,
            "spectral_feature": aligned_spectral,
            "raw_image_feature": current_image_feature,
            "raw_semantic_feature": semantic_feature_raw,
            "raw_temporal_feature": temporal_feature_raw,
            "raw_spectral_feature": spectral_feature_raw,
            "multimodal_representation": multimodal_representation,
            "fused_feature": fused_feature,
            "pixel_feature": pixel_feature_aligned,
            "background_bank": self.background_feature_bank,
            "similarity_to_bank": similarity_to_bank,
            "max_similarity": max_similarity,
            "predicted_class": predicted_class,
            "seen_mask": seen_mask,
            "unseen_mask": unseen_mask,
        }
