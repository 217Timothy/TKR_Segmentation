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
        description="TKR validity gate + wound segmentation + ROI extraction"
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="an image or a directory (searched recursively)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto",
                        help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--roi-padding-pixels", type=int, default=40)
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

    pipeline = TKRSegmentationPipeline(
        checkpoint=args.checkpoint,
        device=args.device,
        roi_padding_pixels=args.roi_padding_pixels,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    summary = []
    for index, path in enumerate(paths, 1):
        result = pipeline.predict(path)
        image_dir = args.output / _output_name(path)
        image_dir.mkdir()
        cv2.imwrite(str(image_dir / "mask.png"), result.mask * 255)
        cv2.imwrite(str(image_dir / "overlay.png"), result.overlay_bgr)
        cv2.imwrite(str(image_dir / "masked.png"), result.masked_bgr)
        if result.roi_bgr is not None:
            cv2.imwrite(str(image_dir / "roi.png"), result.roi_bgr)
        record = {"input": str(path), **result.metadata()}
        (image_dir / "result.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        summary.append(record)
        print(
            f"[{index}/{len(paths)}] {path.name}: {result.decision} "
            f"ood={result.ood_score:.6f} threshold={result.ood_threshold:.6f}"
        )

    accepted = sum(item["accepted"] for item in summary)
    classified = sum(item["should_classify"] for item in summary)
    report = {
        "input": str(args.input),
        "images": len(summary),
        "accepted": accepted,
        "rejected": len(summary) - accepted,
        "ready_for_classification": classified,
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
