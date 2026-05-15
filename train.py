from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from model import BMSMFZS, MultiModalSimilarityLoss
from background_reconstruction import (
    NonLocalPseudoGT,
    TemporalPseudoGT,
    SemanticPixelBackgroundReconstruction,
    ReconstructionLoss,
)


CONFIG = {
    # ------------------------- Data paths -------------------------
    "train_image_dir": r"E:\starry-data\train\images",
    "train_label_dir": r"E:\starry-data\train\labels",

    # Optional. Set to None if spectral features are unavailable.
    # Each spectral file should be .npy with shape [T, 512] or [512].
    "spectral_dir": None,  # r"E:\starry-data\train\spectral"

    # Optional. If real GT/pseudo-GT backgrounds are available, place them here.
    # Each file should have the same stem as the input image.
    "gt_background_dir": None,  # r"E:\starry-data\train\gt_background"

    # Output checkpoint.
    "save_path": r"E:\starry-data\checkpoint\bms_mfzs.pth",

    # ------------------------- Model settings -------------------------
    "num_classes": 200,
    "sequence_length": 5,
    "image_size": 500,
    "image_channels": 3,
    "spectrum_dim": 512,
    "hidden_dim": 512,
    "bert_model_name": "bert-base-uncased",
    "use_bert": True,
    "seen_threshold": 0.75,

    # ------------------------- Training settings -------------------------
    "batch_size": 2,
    "epochs": 50,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "device": "cuda",

    # similarity weights.
    "alpha": 0.5,
    "beta": 0.25,
    "gamma": 0.25,
    "tau_s": 0.07,

    # Total loss weights.
    "classification_loss_weight": 1.0,
    "similarity_loss_weight": 0.1,
    "reconstruction_loss_weight": 1.0,
    "gradient_loss_weight": 1.0,

    # Pseudo-GT settings.
    "use_temporal_pseudo_gt": True,
    "use_nonlocal_pseudo_gt": True,
}


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def identity_auxiliary(sequence_length: int, feature_dim: int = 512) -> torch.Tensor:
    """
    Identity-like auxiliary matrix for missing spectral features.
    Shape: [T, D].
    """
    x = torch.zeros(sequence_length, feature_dim, dtype=torch.float32)
    diag_len = min(sequence_length, feature_dim)
    for i in range(diag_len):
        x[i, i] = 1.0
    return x


def find_image_by_stem(directory: Optional[str], stem: str) -> Optional[str]:
    if directory is None:
        return None
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        path = os.path.join(directory, stem + ext)
        if os.path.exists(path):
            return path
    return None


class InfraredSequenceDataset(Dataset):
    """
    Dataset output:
        sequence: [T, 3, H, W]
        label: [num_classes]
        spectrum: [T, 512]
        label_text: str
        gt_background: [1, H, W]
        has_gt: bool
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        spectral_dir: Optional[str],
        gt_background_dir: Optional[str],
        num_classes: int,
        sequence_length: int,
        image_size: int,
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.spectral_dir = spectral_dir
        self.gt_background_dir = gt_background_dir
        self.num_classes = num_classes
        self.sequence_length = sequence_length

        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"train_image_dir does not exist: {image_dir}")
        if not os.path.isdir(label_dir):
            raise FileNotFoundError(f"train_label_dir does not exist: {label_dir}")

        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        ])
        if not self.image_files:
            raise RuntimeError(f"No images found in {image_dir}")

        self.transform_rgb = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.transform_gray = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.image_files)

    def _load_image(self, idx: int) -> torch.Tensor:
        idx = max(0, min(idx, len(self.image_files) - 1))
        path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(path).convert("RGB")
        return self.transform_rgb(img)

    def _read_label_json(self, image_name: str) -> Tuple[torch.Tensor, str]:
        stem = os.path.splitext(image_name)[0]
        json_path = os.path.join(self.label_dir, stem + ".json")
        label = torch.zeros(self.num_classes, dtype=torch.float32)
        default_text = "infrared dim target background"

        if not os.path.exists(json_path):
            label[0] = 1.0
            return label, default_text

        with open(json_path, "r", encoding="utf-8") as f:
            obj: Dict = json.load(f)

        if "labels" in obj:
            values = obj["labels"]
            n = min(len(values), self.num_classes)
            label[:n] = torch.tensor(values[:n], dtype=torch.float32)
        elif "label" in obj:
            idx = int(obj["label"])
            if 0 <= idx < self.num_classes:
                label[idx] = 1.0
            else:
                label[0] = 1.0
        else:
            label[0] = 1.0

        if "semantic_text" in obj:
            text = str(obj["semantic_text"])
        elif "label_text" in obj:
            text = str(obj["label_text"])
        elif "attributes" in obj and isinstance(obj["attributes"], list):
            text = " ".join([str(x) for x in obj["attributes"]])
        else:
            # Fallback semantic description. For manuscript-consistent experiments,
            # each JSON should contain semantic_text, e.g.
            active = torch.nonzero(label > 0.5, as_tuple=False).view(-1).tolist()
            text = " ".join([f"class {i}" for i in active]) if active else default_text

        return label, text

    def _load_spectral(self, image_name: str) -> torch.Tensor:
        if self.spectral_dir is None:
            return identity_auxiliary(self.sequence_length, 512)

        stem = os.path.splitext(image_name)[0]
        path = os.path.join(self.spectral_dir, stem + ".npy")
        if not os.path.exists(path):
            return identity_auxiliary(self.sequence_length, 512)

        arr = np.load(path).astype(np.float32)
        if arr.ndim == 1:
            arr = np.tile(arr[None, :], (self.sequence_length, 1))
        if arr.shape[0] < self.sequence_length:
            pad = np.repeat(arr[-1:, :], self.sequence_length - arr.shape[0], axis=0)
            arr = np.concatenate([arr, pad], axis=0)
        arr = arr[:self.sequence_length]

        if arr.shape[1] != 512:
            fixed = np.zeros((self.sequence_length, 512), dtype=np.float32)
            d = min(arr.shape[1], 512)
            fixed[:, :d] = arr[:, :d]
            arr = fixed
        return torch.from_numpy(arr)

    def _load_gt_background(self, image_name: str, image_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        stem = os.path.splitext(image_name)[0]
        path = find_image_by_stem(self.gt_background_dir, stem)
        if path is None:
            dummy = torch.zeros(1, image_size, image_size, dtype=torch.float32)
            return dummy, torch.tensor(False)
        img = Image.open(path).convert("L")
        gt = self.transform_gray(img)
        return gt, torch.tensor(True)

    def __getitem__(self, idx: int):
        start = idx - self.sequence_length + 1
        frames = [self._load_image(i) for i in range(start, idx + 1)]
        sequence = torch.stack(frames, dim=0)

        image_name = self.image_files[idx]
        label, label_text = self._read_label_json(image_name)
        spectrum = self._load_spectral(image_name)
        gt_background, has_gt = self._load_gt_background(image_name, sequence.shape[-1])

        return sequence, label, spectrum, label_text, gt_background, has_gt


def _to_gray_sequence(images: torch.Tensor) -> torch.Tensor:
    """[B,T,3,H,W] -> [B,T,1,H,W]"""
    return images.mean(dim=2, keepdim=True)


def train() -> None:
    cfg = CONFIG
    save_dir = os.path.dirname(cfg["save_path"])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    dataset = InfraredSequenceDataset(
        image_dir=cfg["train_image_dir"],
        label_dir=cfg["train_label_dir"],
        spectral_dir=cfg["spectral_dir"],
        gt_background_dir=cfg["gt_background_dir"],
        num_classes=cfg["num_classes"],
        sequence_length=cfg["sequence_length"],
        image_size=cfg["image_size"],
    )
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)

    model = BMSMFZS(
        num_classes=cfg["num_classes"],
        spectrum_dim=cfg["spectrum_dim"],
        hidden_dim=cfg["hidden_dim"],
        image_channels=cfg["image_channels"],
        bert_model_name=cfg["bert_model_name"],
        use_bert=cfg["use_bert"],
        seen_threshold=cfg["seen_threshold"],
    ).to(device)

    reconstructor = SemanticPixelBackgroundReconstruction(
        image_channels=1,
        feature_dim=cfg["hidden_dim"],
        kappa=5.0,
    ).to(device)

    nonlocal_pseudo_gt = NonLocalPseudoGT(top_k=3).to(device)
    temporal_pseudo_gt = TemporalPseudoGT().to(device)

    cls_loss_fn = nn.BCEWithLogitsLoss()
    sim_loss_fn = MultiModalSimilarityLoss(
        alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"], tau=cfg["tau_s"]
    )
    reco_loss_fn = ReconstructionLoss(gradient_weight=cfg["gradient_loss_weight"]).to(device)

    params = list(model.parameters()) + list(reconstructor.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    for epoch in range(cfg["epochs"]):
        model.train()
        reconstructor.train()
        total_loss = 0.0
        total_cls = 0.0
        total_sim = 0.0
        total_reco = 0.0

        for images, labels, spectrum, label_texts, gt_background, has_gt in tqdm(
            loader, desc=f"Epoch {epoch + 1}/{cfg['epochs']}"
        ):
            images = images.to(device)  # [B,T,3,H,W]
            labels = labels.to(device)  # [B,C]
            spectrum = spectrum.to(device)  # [B,T,512]
            gt_background = gt_background.to(device)  # [B,1,H,W]
            has_gt = has_gt.to(device).bool()

            output = model(
                image_sequence=images,
                label_ids=labels,
                spectrum=spectrum,
                label_texts=list(label_texts),
                threshold=cfg["seen_threshold"],
            )

            # Classification + Eq. (11)-(12) similarity-aware contrastive loss.
            loss_cls = cls_loss_fn(output["logits"], labels)
            loss_sim = sim_loss_fn(
                image_feat=output["image_feature"],
                semantic_feat=output["semantic_feature"],
                temporal_feat=output["temporal_feature"],
                spectral_feat=output["spectral_feature"],
            )

            # Semantic-pixel reconstruction.
            gray_seq = _to_gray_sequence(images)
            current_gray = gray_seq[:, -1]  # [B,1,H,W]
            adjacent_gray = gray_seq[:, :-1] if gray_seq.size(1) > 1 else None

            coarse_bg, mask, refined_bg, similarity_map = reconstructor(
                image=current_gray,
                pixel_features=output["pixel_feature"],
                background_bank=output["background_bank"],
            )

            # Pseudo-GT generation. If true GT/pseudo-GT file exists, use it;
            # otherwise build pseudo-GT from non-local and temporal consistency.
            pseudo_nonlocal, conf_nonlocal = nonlocal_pseudo_gt(current_gray, output["pixel_feature"].detach())

            if cfg["use_temporal_pseudo_gt"] and adjacent_gray is not None and adjacent_gray.size(1) > 0:
                pseudo_temporal, conf_temporal = temporal_pseudo_gt(current_gray, adjacent_gray)
                pseudo_generated = 0.5 * pseudo_nonlocal + 0.5 * pseudo_temporal
                conf_generated = torch.clamp(0.5 * conf_nonlocal + 0.5 * conf_temporal, 0.0, 1.0)
            else:
                pseudo_generated = pseudo_nonlocal
                conf_generated = conf_nonlocal

            if torch.any(has_gt):
                pseudo_gt = pseudo_generated.clone()
                confidence_map = conf_generated.clone()
                pseudo_gt[has_gt] = gt_background[has_gt]
                confidence_map[has_gt] = torch.ones_like(confidence_map[has_gt])
            else:
                pseudo_gt = pseudo_generated
                confidence_map = conf_generated

            # Eq. (20)/(25): confidence-weighted reconstruction + gradient loss.
            loss_reco = reco_loss_fn(
                pred=refined_bg,
                target=pseudo_gt,
                confidence=confidence_map,
            )

            loss = (
                cfg["classification_loss_weight"] * loss_cls
                + cfg["similarity_loss_weight"] * loss_sim
                + cfg["reconstruction_loss_weight"] * loss_reco
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()

            total_loss += float(loss.item())
            total_cls += float(loss_cls.item())
            total_sim += float(loss_sim.item())
            total_reco += float(loss_reco.item())

        n = max(len(loader), 1)
        print(
            f"Epoch {epoch + 1}: "
            f"loss={total_loss / n:.6f}, "
            f"cls={total_cls / n:.6f}, "
            f"sim={total_sim / n:.6f}, "
            f"reco={total_reco / n:.6f}"
        )

        torch.save(
            {
                "model": model.state_dict(),
                "reconstructor": reconstructor.state_dict(),
                "config": cfg,
            },
            cfg["save_path"],
        )

    print(f"Training finished. Model saved to: {cfg['save_path']}")


if __name__ == "__main__":
    train()
