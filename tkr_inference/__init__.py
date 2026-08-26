"""Standalone TKR segmentation and validity-gate inference package."""

from .pipeline import InferenceResult, TKRSegmentationPipeline

__version__ = "2.0.0"

__all__ = ["InferenceResult", "TKRSegmentationPipeline", "__version__"]
