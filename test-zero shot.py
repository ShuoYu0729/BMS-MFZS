"""
test-zero shot.py

Public testing script for BMS-MFZS.

This script performs:
1. Model loading
2. Zero-shot inference
3. Background reconstruction
4. Background suppression
5. Result saving
"""

import argparse
import os
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import BMSMFZS
from background_reconstruction import SemanticPixelBackgroundReconstruction, background_suppression


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--sequence_length", type=int, default=5)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def list_images(image_dir: str) -> List[str]:
    exts = [".jpg", ".png", ".bmp", ".jpeg", ".tif"]
    files = []
    for name in sorted(os.listdir(image_dir)):
        if os.path.splitext(name)[1].lower() in exts:
            files.append(os.path.join(image_dir, name))
    return files


def load_sequence(paths: List[str], transform, sequence_length: int) -> torch.Tensor:
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        imgs.append(transform(img))

    while len(imgs) < sequence_length:
        imgs.append(imgs[-1])

    imgs = imgs[:sequence_length]
    return torch.stack(imgs, dim=0)


def save_tensor_image(tensor: torch.Tensor, path: str):
    arr = tensor.detach().cpu().squeeze().numpy()
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-8)
    arr = (arr * 255).astype(np.uint8)
    cv2.imwrite(path, arr)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    model = BMSMFZS(num_classes=args.num_classes).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    reconstructor = SemanticPixelBackgroundReconstruction(image_channels=1, kappa=5.0).to(device)
    reconstructor.eval()

    image_files = list_images(args.image_dir)
    if len(image_files) == 0:
        raise RuntimeError("No images found.")

    with torch.no_grad():
        for idx in range(len(image_files)):
            start = max(0, idx - args.sequence_length + 1)
            seq_paths = image_files[start: idx + 1]
            sequence = load_sequence(seq_paths, transform, args.sequence_length)
            sequence = sequence.unsqueeze(0).to(device)

            dummy_label = torch.zeros(1, dtype=torch.long, device=device)
            dummy_spectrum = torch.zeros(1, args.sequence_length, 512, device=device)

            output = model(sequence, dummy_label, dummy_spectrum)
            logits = output["logits"]
            pred = torch.argmax(logits, dim=1)

            current_rgb = sequence[:, -1]
            current_gray = current_rgb.mean(dim=1, keepdim=True)

            # Construct public-version pixel features from current image feature.
            b, _, h, w = current_gray.shape
            raw_feat = output["raw_image_feature"]
            pixel_features = raw_feat.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)

            # Use class prototypes from classifier weights as background bank.
            background_bank = model.classifier.weight.detach()

            # Match feature dimension if necessary.
            if background_bank.shape[1] != pixel_features.shape[1]:
                proj = torch.nn.Linear(background_bank.shape[1], pixel_features.shape[1]).to(device)
                background_bank = proj(background_bank)

            coarse, mask, refined = reconstructor(current_gray, pixel_features, background_bank)
            suppressed = background_suppression(current_gray, refined, mask)

            base = os.path.splitext(os.path.basename(image_files[idx]))[0]
            save_tensor_image(suppressed, os.path.join(args.save_dir, f"{base}_suppressed.png"))
            save_tensor_image(refined, os.path.join(args.save_dir, f"{base}_background.png"))
            save_tensor_image(mask, os.path.join(args.save_dir, f"{base}_mask.png"))

            print(f"[{idx + 1}/{len(image_files)}] saved, pred={pred.item()}")


if __name__ == "__main__":
    main()