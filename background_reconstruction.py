"""
background_reconstruction.py

Semantic-pixel background reconstruction module for BMS-MFZS.

This file corresponds to:
1. Coarse background reconstruction based on feature matching
2. Soft mask generation
3. Fine background reconstruction based on residual network
4. Non-local self-reconstruction pseudo-GT
5. Temporal consistency-guided pseudo-GT
6. Background suppression
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (torch.norm(x, p=2, dim=dim, keepdim=True) + eps)


class SoftMaskGenerator(nn.Module):
    """
    M(p) = sigmoid(kappa * s_{p,k})
    """

    def __init__(self, kappa: float = 5.0):
        super().__init__()
        self.kappa = kappa

    def forward(self, similarity_map: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.kappa * similarity_map)


class CoarseBackgroundReconstruction(nn.Module):
    """
    Coarse semantic-guided background reconstruction.

    Eq. (15), Eq. (16), and Eq. (17).
    """

    def __init__(self, kappa: float = 5.0):
        super().__init__()
        self.mask_generator = SoftMaskGenerator(kappa=kappa)

    def forward(
        self,
        image: torch.Tensor,
        pixel_features: torch.Tensor,
        background_bank: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            image: [B, C, H, W]
            pixel_features: [B, D, H, W]
            background_bank: [K, D]

        Returns:
            coarse_background, mask, similarity_map
        """
        b, d, h, w = pixel_features.shape

        feat = pixel_features.permute(0, 2, 3, 1).reshape(b, h * w, d)
        feat = l2_normalize(feat, dim=-1)
        bank = l2_normalize(background_bank, dim=-1)

        similarity = torch.matmul(feat, bank.t())
        max_similarity, _ = torch.max(similarity, dim=-1)
        similarity_map = max_similarity.reshape(b, 1, h, w)

        mask = self.mask_generator(similarity_map)

        # public version: use local smoothed image as estimated background
        estimated_background = F.avg_pool2d(image, kernel_size=5, stride=1, padding=2)
        coarse_background = mask * estimated_background + (1.0 - mask) * image

        return coarse_background, mask, similarity_map


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
            Mish(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
        )
        self.act = Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ResidualRefinementNet(nn.Module):
    """
    Fine background reconstruction based on residual learning.

    Eq. (18).
    """

    def __init__(self, image_channels: int = 1, base_channels: int = 32, num_blocks: int = 4):
        super().__init__()

        in_channels = image_channels * 2 + 1

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_channels, affine=True),
            Mish(),
        )

        self.body = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(base_channels, image_channels, kernel_size=3, padding=1)

    def forward(self, image: torch.Tensor, coarse_background: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([image, coarse_background, mask], dim=1)
        feat = self.head(x)
        feat = self.body(feat)
        residual = self.tail(feat)
        return coarse_background + residual


class NonLocalPseudoGT(nn.Module):
    """
    Non-local self-reconstruction pseudo-GT for single-frame images.

    Eq. (20), Eq. (21), Eq. (22), and Eq. (23).
    """

    def __init__(self, search_kernel: int = 11, top_k: int = 3, tau: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.search_kernel = search_kernel
        self.top_k = top_k
        self.tau = tau
        self.eps = eps

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = image.shape
        pad = self.search_kernel // 2

        patches = F.unfold(image, kernel_size=self.search_kernel, padding=pad)
        patches = patches.view(b, c, self.search_kernel * self.search_kernel, h, w)

        center = image.unsqueeze(2)

        local_mean = F.avg_pool2d(image, kernel_size=3, stride=1, padding=1)
        local_var = F.avg_pool2d((image - local_mean) ** 2, kernel_size=3, stride=1, padding=1)
        local_var = local_var.unsqueeze(2)

        score = -((center - patches) ** 2) / (2.0 * (local_var + self.eps))
        score = score.mean(dim=1, keepdim=True)

        weights = torch.softmax(score / self.tau, dim=2)
        top_weights, top_indices = torch.topk(weights, k=self.top_k, dim=2)

        patches_gray = patches.mean(dim=1, keepdim=True)
        top_values = torch.gather(patches_gray, dim=2, index=top_indices)

        pseudo_gt = torch.median(top_values, dim=2).values

        dominant = torch.sum(top_weights, dim=2)
        remaining = torch.clamp(1.0 - dominant, min=self.eps)
        confidence = torch.sigmoid(dominant / remaining - local_var.mean(dim=2))

        return pseudo_gt, confidence


class TemporalPseudoGT(nn.Module):
    """
    Temporal consistency-guided pseudo-GT for multi-frame images.

    Eq. (25) and Eq. (26) are implemented in a compact public version.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, current_frame: torch.Tensor, adjacent_frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            current_frame: [B, C, H, W]
            adjacent_frames: [B, T, C, H, W]
        """
        frames = torch.cat([current_frame.unsqueeze(1), adjacent_frames], dim=1)
        pseudo_gt = torch.median(frames, dim=1).values
        temporal_var = torch.var(frames, dim=1)
        confidence = torch.exp(-temporal_var / (temporal_var.mean() + self.eps))
        confidence = torch.clamp(confidence, 0.0, 1.0)
        return pseudo_gt, confidence


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss and gradient loss.

    Eq. (19) and Eq. (24).
    """

    @staticmethod
    def gradient(x: torch.Tensor) -> torch.Tensor:
        grad_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        grad_y = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        return grad_x + grad_y

    def forward(self, pred: torch.Tensor, target: torch.Tensor, confidence: Optional[torch.Tensor] = None) -> torch.Tensor:
        if confidence is None:
            confidence = torch.ones_like(pred)

        loss_re = torch.abs(pred - target)
        loss_grad = torch.abs(self.gradient(pred) - self.gradient(target))
        return (confidence * (loss_re + loss_grad)).mean()


class SemanticPixelBackgroundReconstruction(nn.Module):
    """
    Complete semantic-pixel background reconstruction module.
    """

    def __init__(self, image_channels: int = 1, kappa: float = 5.0):
        super().__init__()
        self.coarse = CoarseBackgroundReconstruction(kappa=kappa)
        self.refinement = ResidualRefinementNet(image_channels=image_channels)

    def forward(
        self,
        image: torch.Tensor,
        pixel_features: torch.Tensor,
        background_bank: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coarse_background, mask, similarity_map = self.coarse(image, pixel_features, background_bank)
        refined_background = self.refinement(image, coarse_background, mask)
        return coarse_background, mask, refined_background


def background_suppression(
    image: torch.Tensor,
    reconstructed_background: torch.Tensor,
    confidence: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    I_sup(p) = T(p) * |I(p) - B_s(p)|

    Eq. (27).
    """
    diff = torch.abs(image - reconstructed_background)
    if confidence is not None:
        return confidence * diff
    return diff


if __name__ == "__main__":
    image = torch.rand(2, 1, 128, 128)
    pixel_features = torch.rand(2, 64, 128, 128)
    background_bank = torch.rand(6, 64)

    model = SemanticPixelBackgroundReconstruction(image_channels=1, kappa=5.0)
    coarse, mask, refined = model(image, pixel_features, background_bank)
    sup = background_suppression(image, refined, mask)

    print(coarse.shape, mask.shape, refined.shape, sup.shape)