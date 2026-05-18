import csv
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from local_contrast import contrast_algorithm
from background_reconstruction import SemanticPixelBackgroundReconstruction, background_suppression
from con_test import evaluate_suppression, read_gray, average_metrics
from model import BMSMFZS


CONFIG = {
    "image_dir": r"E:\starry-data\test",
    "checkpoint": r"E:\starry-data\checkpoint\bms_mfzs.pth",
    "save_dir": r"E:\starry-data\result",

    "label_dir": None,
    "spectral_dir": None,

    "annotation_path": r"E:\starry-data\annotation.json",
    "gt_background_dir": None,
    "background_mask_dir": None,

    "num_classes": 200,
    "sequence_length": 5,
    "image_size": 500,
    "device": "cuda",
    "kappa": 5.0,

    # Local contrast is used to identify background regions before reconstruction.
    "use_local_contrast": True,
}


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
Box = Tuple[int, int, int, int]


def identity_auxiliary(sequence_length: int, feature_dim: int = 512) -> torch.Tensor:
    x = torch.zeros(sequence_length, feature_dim)
    diag_len = min(sequence_length, feature_dim)
    for i in range(diag_len):
        x[i, i] = 1.0
    return x


def list_images(image_dir: str) -> List[str]:
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"image_dir does not exist: {image_dir}")

    return sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    ])


def load_sequence(paths: List[str], transform, sequence_length: int) -> torch.Tensor:
    imgs = []

    for p in paths:
        img = Image.open(p).convert("RGB")
        imgs.append(transform(img))

    while len(imgs) < sequence_length:
        imgs.insert(0, imgs[0])

    imgs = imgs[-sequence_length:]

    return torch.stack(imgs, dim=0)


def save_tensor_image(tensor: torch.Tensor, path: str) -> None:
    arr = tensor.detach().cpu().squeeze().numpy()
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-8)
    arr = (arr * 255).astype(np.uint8)
    cv2.imwrite(path, arr)


def load_label(
    label_dir: Optional[str],
    stem: str,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    if label_dir is None:
        label = torch.zeros(1, num_classes, dtype=torch.float32, device=device)
        label[0, 0] = 1.0
        return label

    path = os.path.join(label_dir, stem + ".json")
    label = torch.zeros(num_classes, dtype=torch.float32)

    if not os.path.exists(path):
        label[0] = 1.0
        return label.unsqueeze(0).to(device)

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if "labels" in obj:
        values = obj["labels"]
        n = min(len(values), num_classes)
        label[:n] = torch.tensor(values[:n], dtype=torch.float32)
    elif "label" in obj:
        idx = int(obj["label"])
        if 0 <= idx < num_classes:
            label[idx] = 1.0
        else:
            label[0] = 1.0
    else:
        label[0] = 1.0

    return label.unsqueeze(0).to(device)


def load_spectral(
    spectral_dir: Optional[str],
    stem: str,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    if spectral_dir is None:
        return identity_auxiliary(sequence_length, 512).unsqueeze(0).to(device)

    path = os.path.join(spectral_dir, stem + ".npy")

    if not os.path.exists(path):
        return identity_auxiliary(sequence_length, 512).unsqueeze(0).to(device)

    arr = np.load(path).astype(np.float32)

    if arr.ndim == 1:
        arr = np.tile(arr[None, :], (sequence_length, 1))

    if arr.shape[0] < sequence_length:
        pad = np.repeat(arr[-1:, :], sequence_length - arr.shape[0], axis=0)
        arr = np.concatenate([arr, pad], axis=0)

    arr = arr[:sequence_length]

    if arr.shape[1] != 512:
        fixed = np.zeros((sequence_length, 512), dtype=np.float32)
        d = min(arr.shape[1], 512)
        fixed[:, :d] = arr[:, :d]
        arr = fixed

    return torch.from_numpy(arr).unsqueeze(0).to(device)


def load_annotations(path: Optional[str]) -> Dict[str, Dict[str, Box]]:
    if path is None or not os.path.exists(path):
        return {}

    ext = os.path.splitext(path)[1].lower()
    anns: Dict[str, Dict[str, Box]] = {}

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        for name, item in obj.items():
            stem = os.path.splitext(name)[0]
            anns[stem] = {
                "target_box": tuple(item["target_box"]),
                "background_box": tuple(item["background_box"]),
            }

        return anns

    if ext == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                stem = os.path.splitext(row["name"])[0]
                anns[stem] = {
                    "target_box": (
                        int(row["target_x1"]),
                        int(row["target_y1"]),
                        int(row["target_x2"]),
                        int(row["target_y2"]),
                    ),
                    "background_box": (
                        int(row["bg_x1"]),
                        int(row["bg_y1"]),
                        int(row["bg_x2"]),
                        int(row["bg_y2"]),
                    ),
                }

        return anns

    raise ValueError("annotation_path must be .json or .csv")


def find_image(directory: Optional[str], stem: str) -> Optional[str]:
    if directory is None or not os.path.isdir(directory):
        return None

    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        path = os.path.join(directory, stem + ext)
        if os.path.exists(path):
            return path

    return None


def write_metrics_csv(save_dir: str, rows: List[Dict[str, float]], avg: Dict[str, float]) -> None:
    if not rows:
        return

    path = os.path.join(save_dir, "metrics.csv")
    fieldnames = ["name"] + sorted([k for k in rows[0].keys() if k != "name"])

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

        avg_row = {"name": "Average"}
        avg_row.update(avg)
        writer.writerow(avg_row)

    print(f"Metric CSV saved to: {path}")


def build_background_region_mask(
    candidate_masks,
    idx: int,
    h: int,
    w: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    candidate_mask = 1: high local-contrast candidate region.
    background_region_mask = 1 - candidate_mask: reliable background region.

    contrast_algorithm uses three consecutive frames.
    Mask k approximately corresponds to image index k+1.
    """
    if candidate_masks is None or len(candidate_masks) == 0:
        return None

    mask_idx = idx - 1
    mask_idx = max(0, min(mask_idx, len(candidate_masks) - 1))

    local_mask = candidate_masks[mask_idx]
    local_mask = cv2.resize(local_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    local_mask = local_mask.astype(np.float32)

    background_mask = 1.0 - local_mask
    background_mask = torch.from_numpy(background_mask).float()
    background_mask = background_mask.unsqueeze(0).unsqueeze(0).to(device)

    return background_mask


def apply_background_mask_to_pixel_features(
    pixel_features: torch.Tensor,
    background_region_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if background_region_mask is None:
        return pixel_features

    if pixel_features.dim() == 4:
        # [B, C, H, W]
        if pixel_features.shape[-2:] == background_region_mask.shape[-2:]:
            return pixel_features * background_region_mask

        # [B, H, W, C]
        if pixel_features.shape[1:3] == background_region_mask.shape[-2:]:
            mask = background_region_mask.squeeze(1).unsqueeze(-1)
            return pixel_features * mask

        return pixel_features

    if pixel_features.dim() == 3:
        # [B, N, C], where N = H * W
        b, n, c = pixel_features.shape
        _, _, h, w = background_region_mask.shape

        if n == h * w:
            mask = background_region_mask.view(b, 1, h * w).transpose(1, 2)
            return pixel_features * mask

        return pixel_features

    return pixel_features


def main() -> None:
    cfg = CONFIG

    os.makedirs(cfg["save_dir"], exist_ok=True)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
    ])

    model = BMSMFZS(num_classes=cfg["num_classes"]).to(device)

    if not os.path.exists(cfg["checkpoint"]):
        raise FileNotFoundError(f"checkpoint does not exist: {cfg['checkpoint']}")

    state = torch.load(cfg["checkpoint"], map_location=device)

    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"], strict=False)
    else:
        model.load_state_dict(state, strict=False)

    model.eval()

    reconstructor = SemanticPixelBackgroundReconstruction(
        image_channels=1,
        kappa=cfg["kappa"],
    ).to(device)

    if isinstance(state, dict) and "reconstructor" in state:
        reconstructor.load_state_dict(state["reconstructor"], strict=False)

    reconstructor.eval()

    image_files = list_images(cfg["image_dir"])

    if cfg.get("use_local_contrast", True):
        test_pattern = os.path.join(cfg["image_dir"], "*.*")
        candidate_masks = contrast_algorithm(test_pattern)
        print(f"Generated {len(candidate_masks)} local contrast candidate masks.")
    else:
        candidate_masks = []

    annotations = load_annotations(cfg["annotation_path"])
    metric_rows: List[Dict[str, float]] = []
    inference_times: List[float] = []

    with torch.no_grad():
        for idx, image_path in enumerate(image_files):
            stem = os.path.splitext(os.path.basename(image_path))[0]

            start = max(0, idx - cfg["sequence_length"] + 1)
            seq_paths = image_files[start:idx + 1]

            sequence = load_sequence(
                seq_paths,
                transform,
                cfg["sequence_length"],
            ).unsqueeze(0).to(device)

            labels = load_label(
                cfg["label_dir"],
                stem,
                cfg["num_classes"],
                device,
            )

            spectrum = load_spectral(
                cfg["spectral_dir"],
                stem,
                cfg["sequence_length"],
                device,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()

            tic = time.perf_counter()

            output = model(sequence, labels, spectrum)

            logits = output["logits"]
            pred = torch.argmax(logits, dim=1)

            current_rgb = sequence[:, -1]
            current_gray = current_rgb.mean(dim=1, keepdim=True)

            b, _, h, w = current_gray.shape

            # The semantic-pixel reconstruction stage uses the fused representation
            # as a pixel-level semantic feature.
            pixel_features = output["fused"].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)

            if cfg.get("use_local_contrast", True) and len(candidate_masks) > 0:
                background_region_mask = build_background_region_mask(
                    candidate_masks=candidate_masks,
                    idx=idx,
                    h=h,
                    w=w,
                    device=device,
                )

                pixel_features = apply_background_mask_to_pixel_features(
                    pixel_features=pixel_features,
                    background_region_mask=background_region_mask,
                )

            background_bank = model.classifier.weight.detach()

            coarse, mask, refined = reconstructor(
                current_gray,
                pixel_features,
                background_bank,
            )

            suppressed = background_suppression(
                current_gray,
                refined,
                mask,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()

            toc = time.perf_counter()
            inference_times.append(toc - tic)

            suppressed_path = os.path.join(cfg["save_dir"], f"{stem}_suppressed.png")
            background_path = os.path.join(cfg["save_dir"], f"{stem}_background.png")
            mask_path = os.path.join(cfg["save_dir"], f"{stem}_mask.png")

            save_tensor_image(suppressed, suppressed_path)
            save_tensor_image(refined, background_path)
            save_tensor_image(mask, mask_path)

            if stem in annotations:
                input_img = read_gray(image_path)
                output_img = read_gray(suppressed_path)

                gt_path = find_image(cfg["gt_background_dir"], stem)
                mask_gt_path = find_image(cfg["background_mask_dir"], stem)

                gt_img = read_gray(gt_path) if gt_path else None
                bg_mask = read_gray(mask_gt_path) > 0 if mask_gt_path else None

                m = evaluate_suppression(
                    input_image=input_img,
                    output_image=output_img,
                    target_box=annotations[stem]["target_box"],
                    background_box=annotations[stem]["background_box"],
                    gt_background=gt_img,
                    background_mask=bg_mask,
                )

                row = {"name": stem}
                row.update(m)
                metric_rows.append(row)

            print(f"[{idx + 1}/{len(image_files)}] saved: {stem}, pred={int(pred.item())}")

    avg_time = float(np.mean(inference_times)) if inference_times else 0.0
    fps = 1.0 / avg_time if avg_time > 0 else 0.0

    print(f"Average inference latency: {avg_time * 1000:.4f} ms/frame")
    print(f"FPS: {fps:.4f}")

    if metric_rows:
        avg = average_metrics([
            {k: v for k, v in r.items() if k != "name"}
            for r in metric_rows
        ])

        print("Average metrics:", {k: round(v, 4) for k, v in avg.items()})
        write_metrics_csv(cfg["save_dir"], metric_rows, avg)


if __name__ == "__main__":
    main()