"""
con_test.py

Evaluation metrics for infrared background suppression.

This file provides BSF, SCR, CNR, CG, PSNR, MAE, and SSIM calculation.
"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def to_float_image(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return img


def crop_region(img: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return img[y1:y2, x1:x2]


def scr(
    image: np.ndarray,
    target_box: Tuple[int, int, int, int],
    background_box: Tuple[int, int, int, int],
    eps: float = 1e-8,
) -> float:
    image = to_float_image(image)
    target = crop_region(image, target_box)
    bg = crop_region(image, background_box)

    mu_t = np.mean(target)
    mu_b = np.mean(bg)
    sigma_b = np.std(bg)

    return float((mu_t - mu_b) / (sigma_b + eps))


def cnr(
    image: np.ndarray,
    target_box: Tuple[int, int, int, int],
    background_box: Tuple[int, int, int, int],
    eps: float = 1e-8,
) -> float:
    return scr(image, target_box, background_box, eps)


def bsf(
    input_image: np.ndarray,
    output_image: np.ndarray,
    background_box: Tuple[int, int, int, int],
    eps: float = 1e-8,
) -> float:
    input_image = to_float_image(input_image)
    output_image = to_float_image(output_image)

    bg_in = crop_region(input_image, background_box)
    bg_out = crop_region(output_image, background_box)

    return float(np.std(bg_in) / (np.std(bg_out) + eps))


def contrast_gain(
    input_image: np.ndarray,
    output_image: np.ndarray,
    target_box: Tuple[int, int, int, int],
    background_box: Tuple[int, int, int, int],
    eps: float = 1e-8,
) -> float:
    input_image = to_float_image(input_image)
    output_image = to_float_image(output_image)

    target_in = crop_region(input_image, target_box)
    bg_in = crop_region(input_image, background_box)
    target_out = crop_region(output_image, target_box)
    bg_out = crop_region(output_image, background_box)

    c_in = abs(np.mean(target_in) - np.mean(bg_in))
    c_out = abs(np.mean(target_out) - np.mean(bg_out))

    return float(c_out / (c_in + eps))


def psnr(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    pred = to_float_image(pred)
    gt = to_float_image(gt)
    mse = np.mean((pred - gt) ** 2)
    return float(10.0 * np.log10(1.0 / (mse + eps)))


def mae(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = to_float_image(pred)
    gt = to_float_image(gt)
    return float(np.mean(np.abs(pred - gt)))


def ssim_simple(pred: np.ndarray, gt: np.ndarray, c1: float = 0.01 ** 2, c2: float = 0.03 ** 2) -> float:
    pred = to_float_image(pred)
    gt = to_float_image(gt)

    mu_x = np.mean(pred)
    mu_y = np.mean(gt)
    sigma_x = np.var(pred)
    sigma_y = np.var(gt)
    sigma_xy = np.mean((pred - mu_x) * (gt - mu_y))

    value = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    )
    return float(value)


def evaluate_suppression(
    input_image: np.ndarray,
    output_image: np.ndarray,
    target_box: Tuple[int, int, int, int],
    background_box: Tuple[int, int, int, int],
    gt_background: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    result = {
        "BSF": bsf(input_image, output_image, background_box),
        "SCR": scr(output_image, target_box, background_box),
        "CNR": cnr(output_image, target_box, background_box),
        "CG": contrast_gain(input_image, output_image, target_box, background_box),
    }

    if gt_background is not None:
        result["PSNR"] = psnr(output_image, gt_background)
        result["MAE"] = mae(output_image, gt_background)
        result["SSIM"] = ssim_simple(output_image, gt_background)

    return result


def read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


if __name__ == "__main__":
    input_img = np.random.rand(256, 256)
    output_img = np.random.rand(256, 256)

    target_box = (120, 120, 126, 126)
    background_box = (110, 110, 140, 140)

    metrics = evaluate_suppression(input_img, output_img, target_box, background_box)
    print(metrics)