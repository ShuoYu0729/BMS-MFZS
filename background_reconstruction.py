from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (torch.norm(x, p=2, dim=dim, keepdim=True) + eps)


class Mish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class SoftMaskGenerator(nn.Module):
    def __init__(self, kappa: float = 5.0):
        super().__init__()
        self.kappa = kappa

    def forward(self, similarity_map: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.kappa * similarity_map)


class CoarseBackgroundReconstruction(nn.Module):
    """
    Coarse semantic-guided background reconstruction.

    s_{p,k} = <f_p, b_k> / (||f_p|| ||b_k||)
    M(p) = sigmoid(kappa * s_{p,k})
    I~(p) = M(p)*I_hat(p) + (1-M(p))*I(p)
    """

    def __init__(self, image_channels: int = 1, feature_dim: int = 512, kappa: float = 5.0):
        super().__init__()
        self.mask_generator = SoftMaskGenerator(kappa=kappa)
        self.background_decoder = nn.Sequential(
            nn.Conv2d(feature_dim, 256, kernel_size=3, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            Mish(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            Mish(),
            nn.Conv2d(128, image_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        image: torch.Tensor,
        pixel_features: torch.Tensor,
        background_bank: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if image.dim() != 4:
            raise ValueError("image must be [B,C,H,W]")
        if pixel_features.dim() != 4:
            raise ValueError("pixel_features must be [B,D,Hf,Wf]")

        b, d, hf, wf = pixel_features.shape
        _, _, h, w = image.shape

        feat = pixel_features.permute(0, 2, 3, 1).reshape(b, hf * wf, d)
        feat_n = l2_normalize(feat, dim=-1)
        bank_n = l2_normalize(background_bank, dim=-1)

        similarity = feat_n @ bank_n.t()
        max_similarity, best_idx = torch.max(similarity, dim=-1)
        similarity_map = max_similarity.reshape(b, 1, hf, wf)

        selected_bank = background_bank[best_idx.reshape(-1)]
        selected_bank = selected_bank.reshape(b, hf, wf, d).permute(0, 3, 1, 2)

        estimated_background_low = self.background_decoder(selected_bank)
        estimated_background = F.interpolate(
            estimated_background_low,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        similarity_map = F.interpolate(
            similarity_map,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        mask = self.mask_generator(similarity_map)
        coarse_background = mask * estimated_background + (1.0 - mask) * image

        return coarse_background, mask, similarity_map, estimated_background


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)
        self.act = Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class ResidualRefinementNet(nn.Module):
    """
    Fine background reconstruction network R(I, I~, M).
    """

    def __init__(self, image_channels: int = 1, hidden_channels: int = 64, num_blocks: int = 4):
        super().__init__()
        in_channels = image_channels * 2 + 1
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(hidden_channels, affine=True),
            Mish(),
        )
        self.body = nn.Sequential(*[ResidualBlock(hidden_channels) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(hidden_channels, image_channels, kernel_size=3, padding=1)

    def forward(self, image: torch.Tensor, coarse_background: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([image, coarse_background, mask], dim=1)
        feat = self.head(x)
        residual = self.tail(self.body(feat))
        refined = torch.clamp(coarse_background + residual, 0.0, 1.0)
        return refined


class SemanticPixelBackgroundReconstruction(nn.Module):
    """
    Complete semantic-pixel background reconstruction module.
    """

    def __init__(self, image_channels: int = 1, feature_dim: int = 512, kappa: float = 5.0):
        super().__init__()
        self.coarse = CoarseBackgroundReconstruction(
            image_channels=image_channels,
            feature_dim=feature_dim,
            kappa=kappa,
        )
        self.refinement = ResidualRefinementNet(image_channels=image_channels)

    def forward(
        self,
        image: torch.Tensor,
        pixel_features: torch.Tensor,
        background_bank: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        coarse_background, mask, similarity_map, estimated_background = self.coarse(
            image=image,
            pixel_features=pixel_features,
            background_bank=background_bank,
        )
        refined_background = self.refinement(image, coarse_background, mask)
        return coarse_background, mask, refined_background, similarity_map


class NonLocalPseudoGT(nn.Module):
    """
    Non-local self-reconstruction pseudo-GT for single-frame images.

    """

    def __init__(
        self,
        kernel_size: int = 7,
        top_k: int = 3,
        tau: float = 0.1,
        sigma_f: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.top_k = top_k
        self.tau = tau
        self.sigma_f = sigma_f
        self.eps = eps

    def forward(
        self,
        image: torch.Tensor,
        feature_map: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if image.dim() != 4:
            raise ValueError("image must be [B,C,H,W]")
        b, c, h, w = image.shape
        pad = self.kernel_size // 2
        k2 = self.kernel_size * self.kernel_size

        img_patches = F.unfold(image, kernel_size=self.kernel_size, padding=pad)
        img_patches = img_patches.view(b, c, k2, h, w)
        center = image.unsqueeze(2)

        local_var = torch.var(img_patches, dim=2, keepdim=True).mean(dim=1, keepdim=True)
        pixel_dist = torch.mean((center - img_patches) ** 2, dim=1, keepdim=True)
        pixel_score = -pixel_dist / (2.0 * (local_var + self.eps))

        if feature_map is not None:
            feature_resized = F.interpolate(feature_map, size=(h, w), mode="bilinear", align_corners=False)
            _, df, _, _ = feature_resized.shape
            feat_patches = F.unfold(feature_resized, kernel_size=self.kernel_size, padding=pad)
            feat_patches = feat_patches.view(b, df, k2, h, w)
            feat_center = feature_resized.unsqueeze(2)
            feat_dist = torch.mean((feat_center - feat_patches) ** 2, dim=1, keepdim=True)
            feature_score = -feat_dist / (2.0 * (self.sigma_f ** 2))
            score = feature_score + pixel_score
        else:
            score = pixel_score

        score_flat = score.view(b, 1, k2, h, w)
        weights = torch.softmax(score_flat / self.tau, dim=2)
        top_weights, top_indices = torch.topk(weights, k=min(self.top_k, k2), dim=2)

        gray_patches = img_patches.mean(dim=1, keepdim=True)
        top_values = torch.gather(gray_patches, dim=2, index=top_indices)
        pseudo_gt = torch.median(top_values, dim=2).values

        dominant = torch.sum(top_weights, dim=2)
        remaining = torch.clamp(1.0 - dominant, min=self.eps)
        var_norm = local_var.squeeze(2) / (torch.amax(local_var.squeeze(2), dim=(-2, -1), keepdim=True) + self.eps)
        confidence = torch.sigmoid(dominant / remaining - var_norm)
        confidence = torch.clamp(confidence, 0.0, 1.0)
        return pseudo_gt, confidence


class TemporalPseudoGT(nn.Module):
    """
    Temporal consistency-guided pseudo-GT for multi-frame images.

    This core implementation uses temporal median aggregation and a temporal
    variance confidence map. Optical-flow consistency can be added when a flow
    module is available, but this remains consistent with the manuscript's
    median-based temporal pseudo-GT principle.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, current_frame: torch.Tensor, adjacent_frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if adjacent_frames is None or adjacent_frames.numel() == 0:
            return current_frame, torch.ones_like(current_frame)
        frames = torch.cat([current_frame.unsqueeze(1), adjacent_frames], dim=1)
        pseudo_gt = torch.median(frames, dim=1).values
        temporal_var = torch.var(frames, dim=1)
        confidence = torch.exp(-temporal_var / (temporal_var.mean(dim=(-2, -1), keepdim=True) + self.eps))
        confidence = torch.clamp(confidence, 0.0, 1.0)
        return pseudo_gt, confidence


class ReconstructionLoss(nn.Module):
    """
    Confidence-weighted reconstruction and gradient loss.

    Loss_total = mean_p T(p)(|I* - I_gt| + lambda_g |grad I* - grad I_gt|)
    """

    def __init__(self, gradient_weight: float = 1.0):
        super().__init__()
        self.gradient_weight = gradient_weight

    @staticmethod
    def gradient(x: torch.Tensor) -> torch.Tensor:
        grad_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        grad_y = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        return grad_x + grad_y

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pred.shape != target.shape:
            target = F.interpolate(target, size=pred.shape[-2:], mode="bilinear", align_corners=False)
        if confidence is None:
            confidence = torch.ones_like(pred)
        elif confidence.shape != pred.shape:
            confidence = F.interpolate(confidence, size=pred.shape[-2:], mode="bilinear", align_corners=False)

        loss_re = torch.abs(pred - target)
        loss_gra = torch.abs(self.gradient(pred) - self.gradient(target))
        return (confidence * (loss_re + self.gradient_weight * loss_gra)).mean()


def background_suppression(image: torch.Tensor, background: torch.Tensor, confidence: torch.Tensor,) -> torch.Tensor:
    if background.shape[-2:] != image.shape[-2:]:
        background = F.interpolate(background, size=image.shape[-2:], mode="bilinear", align_corners=False)
    if confidence.shape[-2:] != image.shape[-2:]:
        confidence = F.interpolate(confidence, size=image.shape[-2:], mode="bilinear", align_corners=False)
    return confidence * torch.abs(image - background)
