"""Minimal example: put TKR segmentation before an existing classifier."""

from pathlib import Path
from typing import Any, Protocol

import numpy as np

from tkr_inference import TKRSegmentationPipeline


class ExistingClassifier(Protocol):
    def predict(self, image_bgr: np.ndarray) -> Any:
        """Your classmate's existing BGR-image classification API."""


class SegmentationThenClassification:
    def __init__(self, classifier: ExistingClassifier,
                 segmentation_device: str = "auto") -> None:
        self.segmentation = TKRSegmentationPipeline(
            device=segmentation_device
        )
        self.classifier = classifier

    def predict(self, image_path: str | Path) -> dict:
        segmentation = self.segmentation.predict(image_path)

        # Rejection is terminal: do not call the classifier.
        if not segmentation.accepted:
            return {
                "status": "rejected_non_tkr",
                "classification": None,
                "segmentation": segmentation.metadata(),
            }

        # A valid-looking TKR image can still have no usable wound mask.
        if not segmentation.should_classify:
            return {
                "status": "accepted_but_no_wound_roi",
                "classification": None,
                "segmentation": segmentation.metadata(),
            }

        # Replace this one line if the classifier wants RGB, a file path, or
        # the masked full-size image instead of the BGR wound crop.
        class_result = self.classifier.predict(segmentation.roi_bgr)
        return {
            "status": "classified",
            "classification": class_result,
            "segmentation": segmentation.metadata(),
        }
