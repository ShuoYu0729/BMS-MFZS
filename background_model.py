"""
background_model.py

GZSL-based background feature modeling for BMS-MFZS.

This file corresponds to:
1. Background modeling based on GZSL
2. Similarity-based seen/unseen recognition
3. Background feature mapping group update
4. LDA-based feature separability analysis
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def cosine_similarity_matrix(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x / (torch.norm(x, p=2, dim=-1, keepdim=True) + eps)
    y = y / (torch.norm(y, p=2, dim=-1, keepdim=True) + eps)
    return torch.matmul(x, y.t())


class BackgroundFeatureBank:
    """
    Background feature mapping group.

    This module stores background semantic prototypes and updates them
    when newly identified unseen background categories are incorporated.
    """

    def __init__(self, feature_dim: int = 512, momentum: float = 0.9):
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.features: List[torch.Tensor] = []
        self.labels: List[int] = []
        self.names: List[str] = []

    def __len__(self) -> int:
        return len(self.features)

    def add(self, feature: torch.Tensor, label: int, name: Optional[str] = None) -> None:
        feature = feature.detach().float().cpu()
        if feature.dim() > 1:
            feature = feature.mean(dim=0)
        self.features.append(feature)
        self.labels.append(int(label))
        self.names.append(name if name is not None else f"class_{label}")

    def update(self, feature: torch.Tensor, label: int, name: Optional[str] = None) -> None:
        feature = feature.detach().float().cpu()
        if feature.dim() > 1:
            feature = feature.mean(dim=0)

        if label in self.labels:
            idx = self.labels.index(label)
            self.features[idx] = self.momentum * self.features[idx] + (1.0 - self.momentum) * feature
        else:
            self.add(feature, label, name)

    def as_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        if len(self.features) == 0:
            raise RuntimeError("BackgroundFeatureBank is empty.")
        x = torch.stack(self.features, dim=0)
        if device is not None:
            x = x.to(device)
        return x

    def save(self, path: str) -> None:
        obj = {
            "features": self.features,
            "labels": self.labels,
            "names": self.names,
            "feature_dim": self.feature_dim,
            "momentum": self.momentum,
        }
        torch.save(obj, path)

    @classmethod
    def load(cls, path: str) -> "BackgroundFeatureBank":
        obj = torch.load(path, map_location="cpu")
        bank = cls(feature_dim=obj.get("feature_dim", 512), momentum=obj.get("momentum", 0.9))
        bank.features = obj["features"]
        bank.labels = obj["labels"]
        bank.names = obj["names"]
        return bank


class GZSLBackgroundModeler:
    """
    GZSL-based background model establishment.

    The model compares mapped image-semantic-auxiliary features with
    background semantic prototypes and decides whether a sample belongs
    to seen or unseen categories according to a cosine-similarity threshold.
    """

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    def classify_seen_unseen(
        self,
        query_feature: torch.Tensor,
        background_bank: BackgroundFeatureBank,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            query_feature: [B, D]
            background_bank: stored semantic/background prototypes

        Returns:
            max similarity, predicted label, seen/unseen mask
        """
        bank = background_bank.as_tensor(device=query_feature.device)
        sim = cosine_similarity_matrix(query_feature, bank)

        max_sim, idx = torch.max(sim, dim=1)
        bank_labels = torch.tensor(background_bank.labels, device=query_feature.device)
        pred_labels = bank_labels[idx]

        is_seen = max_sim >= self.threshold
        is_unseen = ~is_seen

        return {
            "similarity": sim,
            "max_similarity": max_sim,
            "pred_labels": pred_labels,
            "is_seen": is_seen,
            "is_unseen": is_unseen,
        }

    def update_unseen_groups(
        self,
        query_feature: torch.Tensor,
        unseen_mask: torch.Tensor,
        background_bank: BackgroundFeatureBank,
        start_label: int = 1000,
    ) -> BackgroundFeatureBank:
        """
        Add newly detected unseen samples into the background feature bank.
        """
        unseen_features = query_feature[unseen_mask].detach().cpu()
        for i, feat in enumerate(unseen_features):
            background_bank.add(feat, label=start_label + i, name=f"unseen_{start_label + i}")
        return background_bank


class LDAFeatureAnalyzer:
    """
    LDA-based intra-class and inter-class feature analysis.

    Corresponds to Eq. (13).
    """

    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.lda = None

    def fit_transform(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        n_classes = len(np.unique(labels))
        n_features = features.shape[1]
        if self.n_components is None:
            n_components = min(n_classes - 1, n_features)
        else:
            n_components = self.n_components

        self.lda = LinearDiscriminantAnalysis(n_components=n_components)
        return self.lda.fit_transform(features, labels)

    @staticmethod
    def fisher_ratio(features: np.ndarray, labels: np.ndarray, eps: float = 1e-8) -> float:
        classes = np.unique(labels)
        global_mean = np.mean(features, axis=0)

        sb = 0.0
        sw = 0.0

        for c in classes:
            x_c = features[labels == c]
            if x_c.shape[0] == 0:
                continue
            mean_c = np.mean(x_c, axis=0)
            sb += x_c.shape[0] * np.sum((mean_c - global_mean) ** 2)
            sw += np.sum((x_c - mean_c) ** 2)

        return float(sb / (sw + eps))


def build_background_bank_from_features(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_names: Optional[List[str]] = None,
) -> BackgroundFeatureBank:
    """
    Build background feature mapping groups from labeled features.
    """
    bank = BackgroundFeatureBank(feature_dim=features.shape[-1])

    unique_labels = labels.unique().tolist()
    for lab in unique_labels:
        mask = labels == lab
        proto = features[mask].mean(dim=0)
        name = class_names[int(lab)] if class_names is not None and int(lab) < len(class_names) else None
        bank.add(proto, int(lab), name)

    return bank


if __name__ == "__main__":
    features = torch.randn(20, 512)
    labels = torch.randint(0, 4, (20,))

    bank = build_background_bank_from_features(features, labels)
    modeler = GZSLBackgroundModeler(threshold=0.75)

    query = torch.randn(5, 512)
    result = modeler.classify_seen_unseen(query, bank)

    print("Predicted labels:", result["pred_labels"])
    print("Seen mask:", result["is_seen"])