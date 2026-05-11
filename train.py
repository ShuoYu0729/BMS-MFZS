"""
train.py

Simplified public training script for BMS-MFZS.

This script keeps the training logic consistent with the paper:
1. Image feature extraction
2. Semantic label feature learning
3. Temporal and spectral auxiliary feature modeling
4. GZSL-based contrastive alignment
5. Multi-label classification training
"""

import argparse
import json
import os
from typing import List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from model import BMSMFZS, MultiModalSimilarityLoss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--label_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="checkpoints/pretrained_model.pth")
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--sequence_length", type=int, default=5)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


class InfraredSequenceDataset(Dataset):
    """
    Public dataset loader.

    Label files are expected to be JSON files containing either:
    {"label": int}
    or
    {"labels": [0, 1, 0, ...]}
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        num_classes: int,
        sequence_length: int = 5,
        image_size: int = 224,
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.num_classes = num_classes
        self.sequence_length = sequence_length

        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in [".jpg", ".png", ".bmp", ".jpeg", ".tif"]
        ])

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def _read_label(self, image_name: str) -> torch.Tensor:
        stem = os.path.splitext(image_name)[0]
        json_path = os.path.join(self.label_dir, stem + ".json")

        label = torch.zeros(self.num_classes)

        if not os.path.exists(json_path):
            label[0] = 1.0
            return label

        with open(json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if "labels" in obj:
            labels = obj["labels"]
            label[:len(labels)] = torch.tensor(labels).float()
        elif "label" in obj:
            label[int(obj["label"])] = 1.0
        else:
            label[0] = 1.0

        return label

    def _load_image(self, idx: int) -> torch.Tensor:
        idx = max(0, min(idx, len(self.image_files) - 1))
        path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx: int):
        start = idx - self.sequence_length + 1
        frames = []
        for i in range(start, idx + 1):
            frames.append(self._load_image(i))
        sequence = torch.stack(frames, dim=0)

        label = self._read_label(self.image_files[idx])

        # Public version: if spectral features are unavailable, use placeholder.
        spectrum = torch.zeros(self.sequence_length, 512)

        return sequence, label, spectrum


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = InfraredSequenceDataset(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        num_classes=args.num_classes,
        sequence_length=args.sequence_length,
        image_size=args.image_size,
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = BMSMFZS(num_classes=args.num_classes).to(device)

    cls_loss_fn = nn.BCEWithLogitsLoss()
    sim_loss_fn = MultiModalSimilarityLoss(alpha=0.5, beta=0.25, gamma=0.25, tau=0.07)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for images, labels, spectrum in tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            images = images.to(device)
            labels = labels.to(device)
            spectrum = spectrum.to(device)

            output = model(images, labels, spectrum)

            logits = output["logits"]
            loss_cls = cls_loss_fn(logits, labels)

            loss_sim = sim_loss_fn(
                image_feat=output["image_feature"],
                semantic_feat=output["semantic_feature"],
                temporal_feat=output["temporal_feature"],
                spectral_feat=output["spectral_feature"],
            )

            loss = loss_cls + 0.1 * loss_sim

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        print(f"Epoch {epoch + 1}: loss={avg_loss:.6f}")

        torch.save(model.state_dict(), args.save_path)

    print(f"Training finished. Model saved to {args.save_path}")


if __name__ == "__main__":
    main()