"""Standalone TKR segmentation + image-validity inference.

The checkpoint contains one shared EfficientNet-B3 encoder, a U-Net
segmentation decoder, and an image-level valid-TKR head.  The calibrated gate
is evaluated before the final segmentation mask is returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
_DEFAULT_WEIGHT = (
    Path(__file__).resolve().parent
    / "weights"
    / "tkr_seg_validity_gate_b3_256.pt"
)


class ValidityOODHead(nn.Module):
    """Image-level P(valid TKR) head with padding-aware global pooling."""

    def __init__(self, in_channels: int, hidden_dim: int = 128,
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def masked_pool(features: torch.Tensor,
                    valid_mask: torch.Tensor | None) -> torch.Tensor:
        if valid_mask is None:
            return features.mean(dim=(-2, -1))
        if valid_mask.ndim == 3:
            valid_mask = valid_mask.unsqueeze(1)
        mask = F.interpolate(
            valid_mask.float(), size=features.shape[-2:], mode="nearest"
        ).to(device=features.device, dtype=features.dtype)
        numerator = (features * mask).sum(dim=(-2, -1))
        denominator = mask.sum(dim=(-2, -1)).clamp_min(1.0)
        return numerator / denominator

    def forward(self, features: torch.Tensor,
                valid_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        embedding = self.masked_pool(features, valid_mask)
        logit = self.classifier(embedding).squeeze(1)
        valid_probability = torch.sigmoid(logit)
        return {
            "valid_tkr_probability": valid_probability,
            "ood_score_mean": 1.0 - valid_probability,
        }


class EfficientUnetValidity(nn.Module):
    """Exact inference architecture used by the packaged checkpoint."""

    def __init__(self, *, encoder_name: str = "efficientnet-b3",
                 attention: str = "scse", decoder_dropout: float = 0.4,
                 validity_hidden_dim: int = 128,
                 validity_dropout: float = 0.3) -> None:
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=1,
            decoder_attention_type=attention,
            activation=None,
        )
        for block in self.model.decoder.blocks:
            block.add_module("dropout", nn.Dropout2d(p=decoder_dropout))
        self.bottleneck_dropout = nn.Dropout2d(p=decoder_dropout)
        self.ood_head = ValidityOODHead(
            in_channels=self.model.encoder.out_channels[-1],
            hidden_dim=validity_hidden_dim,
            dropout=validity_dropout,
        )

    def forward(self, image: torch.Tensor,
                valid_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.model.encoder(image)
        features[-1] = self.bottleneck_dropout(features[-1])
        gate = self.ood_head(features[-1], valid_mask=valid_mask)
        decoded = self.model.decoder(features)
        gate["seg_logits"] = self.model.segmentation_head(decoded)
        return gate


@dataclass(frozen=True)
class InferenceResult:
    """One image's outputs. Arrays use the original input resolution."""

    accepted: bool
    decision: str
    ood_score: float
    valid_tkr_probability: float
    ood_threshold: float
    mask: np.ndarray
    bbox_xyxy: tuple[int, int, int, int] | None
    masked_bgr: np.ndarray
    overlay_bgr: np.ndarray
    raw_mask_non_empty: bool
    mask_non_empty: bool
    image_shape_hw: tuple[int, int]

    def metadata(self) -> dict[str, Any]:
        """Return the JSON-safe portion of the result."""
        return {
            "accepted": self.accepted,
            "decision": self.decision,
            "ood_score": self.ood_score,
            "valid_tkr_probability": self.valid_tkr_probability,
            "ood_threshold": self.ood_threshold,
            "score_direction": "higher_is_more_ood",
            "mask_non_empty": self.mask_non_empty,
            "raw_mask_non_empty": self.raw_mask_non_empty,
            "bbox_xyxy": list(self.bbox_xyxy) if self.bbox_xyxy else None,
            "image_shape_hw": list(self.image_shape_hw),
        }


def _pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _infer_content_mask(image_rgb: np.ndarray) -> np.ndarray:
    nonzero = np.any(image_rgb != 0, axis=2)
    rows = np.flatnonzero(nonzero.any(axis=1))
    cols = np.flatnonzero(nonzero.any(axis=0))
    valid = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    if rows.size and cols.size:
        valid[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1] = 1
    else:
        valid.fill(1)
    return valid


def _letterbox(image: np.ndarray, size: int, *, nearest: bool = False
               ) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width, new_height = round(width * scale), round(height * scale)
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    shape = (size, size) if image.ndim == 2 else (size, size, image.shape[2])
    boxed = np.zeros(shape, dtype=image.dtype)
    valid = np.zeros((size, size), dtype=np.uint8)
    x0, y0 = (size - new_width) // 2, (size - new_height) // 2
    boxed[y0:y0 + new_height, x0:x0 + new_width] = resized
    valid[y0:y0 + new_height, x0:x0 + new_width] = 1
    return boxed, valid


def _letterbox_with_content_mask(image_rgb: np.ndarray, size: int
                                 ) -> tuple[np.ndarray, np.ndarray]:
    source_valid = _infer_content_mask(image_rgb)
    boxed, geometric_valid = _letterbox(image_rgb, size)
    boxed_source_valid, _ = _letterbox(source_valid, size, nearest=True)
    valid = geometric_valid * (boxed_source_valid > 0).astype(np.uint8)
    return boxed, valid


def _restore_mask(mask: np.ndarray, valid_mask: np.ndarray,
                  original_shape: tuple[int, int]) -> np.ndarray:
    rows = np.flatnonzero(valid_mask.any(axis=1))
    cols = np.flatnonzero(valid_mask.any(axis=0))
    if not rows.size or not cols.size:
        raise ValueError("valid content mask is empty")
    cropped = mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    height, width = original_shape
    return cv2.resize(
        cropped.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    )


def _postprocess(mask: np.ndarray, *, min_area: int = 50) -> np.ndarray:
    work = np.where(mask > 0, 255, 0).astype(np.uint8)
    work = cv2.GaussianBlur(work, (7, 7), 0.0)
    _, work = cv2.threshold(work, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    clean = np.zeros_like(work)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) >= min_area:
            cv2.drawContours(clean, [largest], -1, 255, thickness=cv2.FILLED)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    return (clean > 0).astype(np.uint8)


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if not xs.size:
        return None
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


class TKRSegmentationPipeline:
    """Load once, then run the gated segmentation pipeline on many images."""

    def __init__(self, checkpoint: str | Path | None = None, *,
                 device: str = "auto", mask_threshold: float = 0.5) -> None:
        self.checkpoint_path = Path(checkpoint) if checkpoint else _DEFAULT_WEIGHT
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        self.device = _pick_device(device)
        self.mask_threshold = float(mask_threshold)

        checkpoint_data = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        cfg = checkpoint_data.get("model_cfg") or {}
        expected = {
            "model": "efficientunet",
            "encoder_name": "efficientnet-b3",
            "attention": "scse",
            "classes": 1,
            "ood_enabled": True,
            "ood_mode": "validity",
            "validity_hidden_dim": 128,
        }
        wrong = {key: (cfg.get(key), value) for key, value in expected.items()
                 if cfg.get(key) != value}
        if wrong:
            raise ValueError(f"unsupported checkpoint architecture: {wrong}")
        calibration = checkpoint_data.get("ood_calibration") or {}
        if calibration.get("status") != "calibrated":
            raise ValueError("checkpoint has no validation-calibrated OOD threshold")
        if calibration.get("test_data_used") is not False:
            raise ValueError("refusing a checkpoint whose threshold used test data")

        self.ood_threshold = float(calibration["threshold"])
        self.image_size = 256
        self.checkpoint_epoch = int(checkpoint_data.get("epoch", -1))
        self.model = EfficientUnetValidity(
            encoder_name=cfg["encoder_name"],
            attention=cfg["attention"],
            validity_hidden_dim=int(cfg["validity_hidden_dim"]),
            validity_dropout=float(cfg.get("validity_dropout", 0.3)),
        )
        self.model.load_state_dict(checkpoint_data["state_dict"], strict=True)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, image: str | Path | np.ndarray) -> InferenceResult:
        """Predict from an image path or an OpenCV-style BGR uint8 array."""
        if isinstance(image, (str, Path)):
            image_bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError(f"cannot read image: {image}")
        else:
            image_bgr = np.asarray(image)
            if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
                raise ValueError("NumPy input must be an HxWx3 BGR image")
            if image_bgr.dtype != np.uint8:
                raise ValueError("NumPy input must have dtype uint8")
            image_bgr = image_bgr.copy()

        original_shape = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        boxed, valid = _letterbox_with_content_mask(image_rgb, self.image_size)
        normalized = (boxed.astype(np.float32) / 255.0 - _MEAN) / _STD
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
        valid_tensor = torch.from_numpy(valid.astype(np.float32))[None, None]
        tensor = tensor.to(self.device)
        valid_tensor = valid_tensor.to(self.device)

        output = self.model(tensor, valid_tensor)
        ood_score = float(output["ood_score_mean"][0].item())
        valid_probability = float(output["valid_tkr_probability"][0].item())
        accepted = ood_score < self.ood_threshold
        raw_boxed_mask = (
            torch.sigmoid(output["seg_logits"])[0, 0] > self.mask_threshold
        ).to(torch.uint8).cpu().numpy()
        raw_non_empty = bool(raw_boxed_mask.any())

        if accepted:
            boxed_mask = _postprocess(raw_boxed_mask, min_area=50)
        else:
            boxed_mask = np.zeros_like(raw_boxed_mask, dtype=np.uint8)
        mask = _restore_mask(boxed_mask, valid, original_shape)
        mask_non_empty = bool(mask.any())
        bbox = _mask_bbox(mask)
        if not accepted:
            decision = "REJECT_OOD"
        elif mask_non_empty:
            decision = "ACCEPT_TKR_WOUND_FOUND"
        else:
            decision = "ACCEPT_TKR_NO_WOUND_MASK"

        masked = cv2.bitwise_and(image_bgr, image_bgr, mask=mask * 255)
        overlay = image_bgr.copy()
        if mask_non_empty:
            red = np.zeros_like(image_bgr)
            red[:, :, 2] = 255
            blended = cv2.addWeighted(image_bgr, 0.55, red, 0.45, 0)
            overlay[mask.astype(bool)] = blended[mask.astype(bool)]

        return InferenceResult(
            accepted=accepted,
            decision=decision,
            ood_score=ood_score,
            valid_tkr_probability=valid_probability,
            ood_threshold=self.ood_threshold,
            mask=mask,
            bbox_xyxy=bbox,
            masked_bgr=masked,
            overlay_bgr=overlay,
            raw_mask_non_empty=raw_non_empty,
            mask_non_empty=mask_non_empty,
            image_shape_hw=original_shape,
        )
