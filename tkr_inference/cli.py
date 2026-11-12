"""Command-line interface for standalone gated TKR segmentation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2

from .pipeline import TKRSegmentationPipeline

_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        item for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in _SUFFIXES
    )


def _output_name(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{path.stem}_{digest}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TKR validity gate + wound segmentation inference"
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="an image or a directory (searched recursively)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("validity", "legacy"), default="validity",
                        help="validity: current gated model; legacy: June pure segmentation")
    parser.add_argument("--profile", choices=("historical-roi", "letterbox"), default=None,
                        help="validity default: historical-roi; pure legacy default: letterbox")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto",
                        help="auto, cpu, cuda, cuda:0, or mps")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output directory: {args.output}"
        )
    paths = _collect(args.input)
    if not paths:
        raise FileNotFoundError(f"no supported images under {args.input}")

    pipeline_class = TKRSegmentationPipeline
    if args.model == "legacy":
        from .legacy import LegacySegmentationPipeline
        pipeline_class = LegacySegmentationPipeline
    pipeline = pipeline_class(
        checkpoint=args.checkpoint,
        device=args.device,
        **({"profile": args.profile} if args.profile is not None else {}),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    summary = []
    for index, path in enumerate(paths, 1):
        result = pipeline.predict(path)
        image_dir = args.output / _output_name(path)
        image_dir.mkdir()
        for name, pixels in (("mask.png", result.mask * 255), ("overlay.png", result.overlay_bgr),
                             ("masked.png", result.masked_bgr)):
            if not cv2.imwrite(str(image_dir / name), pixels):
                raise OSError(f"failed to save {image_dir / name}")
        if result.roi_bgr is not None:
            if not cv2.imwrite(str(image_dir / "roi.jpg"), result.roi_bgr,
                               [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"failed to save {image_dir / 'roi.jpg'}")
        record = {"input": str(path), **result.metadata(),
                  "roi_file": "roi.jpg" if result.roi_bgr is not None else None}
        (image_dir / "result.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        summary.append(record)
        gate_text = (f" ood={result.ood_score:.6f} threshold={result.ood_threshold:.6f}"
                     if pipeline.gate_enabled else " (no validity gate)")
        print(f"[{index}/{len(paths)}] {path.name}: {result.decision}{gate_text}", flush=True)

    accepted = sum(item["accepted"] is True for item in summary) if pipeline.gate_enabled else None
    mask_non_empty = sum(item["mask_non_empty"] for item in summary)
    report = {
        "input": str(args.input),
        "images": len(summary),
        "accepted": accepted,
        "rejected": len(summary) - accepted if pipeline.gate_enabled else None,
        "model_variant": args.model,
        "processing": pipeline.processing_config(),
        "roi_images": sum(item["roi_available"] for item in summary),
        "validity_gate_enabled": pipeline.gate_enabled,
        "checkpoint_path": str(pipeline.checkpoint_path.resolve()),
        "checkpoint_sha256": hashlib.sha256(pipeline.checkpoint_path.read_bytes()).hexdigest(),
        "image_size": pipeline.image_size,
        "mask_threshold": pipeline.mask_threshold,
        "postprocess": pipeline.postprocess,
        "mask_non_empty": mask_non_empty,
        "device": str(pipeline.device),
        "checkpoint_epoch": pipeline.checkpoint_epoch,
        "ood_threshold": pipeline.ood_threshold,
        "results": summary,
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
