"""Offline package integrity and smoke test (does not use patient images)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from tkr_inference import TKRSegmentationPipeline, __version__


def main() -> None:
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    weight = root / manifest["weight_file"]
    if not weight.is_file():
        raise SystemExit(
            "model weight is missing. Place the privately supplied file at "
            f"{weight} and run verify again."
        )
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    if digest != manifest["weight_sha256"]:
        raise SystemExit(
            f"weight checksum mismatch: expected {manifest['weight_sha256']}, got {digest}"
        )

    checkpoint = torch.load(weight, map_location="cpu", weights_only=False)
    if int(checkpoint["epoch"]) != int(manifest["checkpoint_epoch"]):
        raise SystemExit("checkpoint epoch does not match manifest")
    checkpoint_threshold = float(checkpoint["ood_calibration"]["threshold"])
    if abs(checkpoint_threshold - float(manifest["ood_threshold"])) > 1e-12:
        raise SystemExit("checkpoint OOD threshold does not match manifest")

    pipeline = TKRSegmentationPipeline(device="cpu")
    # Deterministic synthetic pixels verify loading, preprocessing and forward
    # execution only. They are not used as a clinical or accuracy test.
    gradient = np.linspace(0, 255, 256, dtype=np.uint8)
    synthetic = np.dstack([
        np.tile(gradient, (256, 1)),
        np.tile(gradient[:, None], (1, 256)),
        np.full((256, 256), 127, dtype=np.uint8),
    ])
    result = pipeline.predict(synthetic)
    print("package verification: PASS")
    print(f"package version: {__version__}")
    print(f"weight SHA-256: {digest}")
    print(json.dumps(result.metadata(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
