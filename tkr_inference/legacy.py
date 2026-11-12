"""June exp2 pure segmentation, sharing the current 512px image processing."""
from pathlib import Path

import torch

from .pipeline import (
    A, _MEAN, _STD, _pick_device, EfficientUnetValidity, TKRSegmentationPipeline,
)


class EfficientUnetLegacy(EfficientUnetValidity):
    """Same encoder/decoder, with no validity head or gate computation."""

    def __init__(self):
        super().__init__()
        del self.ood_head

    def forward(self, image, valid_mask):
        features = self.model.encoder(image)
        features[-1] = self.bottleneck_dropout(features[-1])
        decoded = self.model.decoder(features)
        return {"seg_logits": self.model.segmentation_head(decoded)}


class LegacySegmentationPipeline(TKRSegmentationPipeline):
    """Load the original pure segmentation checkpoint strictly, without a gate."""

    gate_enabled = False

    def __init__(self, checkpoint=None, *, device="auto", mask_threshold=0.5, profile="letterbox"):
        self.checkpoint_path = Path(checkpoint) if checkpoint else (
            Path(__file__).resolve().parents[1]
            / "outputs/checkpoints/tkr-finetune-v1/exp2/best.pt"
        )
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        self.device = _pick_device(device)
        self.mask_threshold = float(mask_threshold)
        data = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state = data.get("state_dict", data)
        if any(key.startswith("ood_head.") for key in state):
            raise ValueError("legacy mode requires a pure segmentation checkpoint without an OOD head")
        self.model = EfficientUnetLegacy()
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.checkpoint_epoch = int(data.get("epoch", -1))
        self.ood_threshold = None
        self.image_size = 512
        self.checkpoint_postprocess = dict(blur_kernel=9, blur_sigma=0.0, closing_kernel=9,
                                open_kernel=5, min_area=200, keep_largest=True)
        self._configure_profile(profile)
        self.normalize = A.Normalize(mean=tuple(_MEAN), std=tuple(_STD))
