#!/usr/bin/env python3
"""Run the banner-replacement pipeline from a YAML config.

Usage
-----
    python scripts/run_pipeline.py --config configs/default.yaml
    python scripts/run_pipeline.py --config configs/default.yaml --override fitter.type=lp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the src/ directory is importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2

from banner_pipeline.pipeline import load_config, run


def _apply_overrides(config: dict, overrides: list[str]) -> None:
    """Apply dot-notation overrides like ``fitter.type=lp`` to *config*."""
    for ov in overrides:
        key_path, _, raw = ov.partition("=")
        keys = key_path.split(".")
        d = config.setdefault("pipeline", {})
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        # Try to parse as int/float/bool/null, else keep as string.
        parsed: object = raw
        for parser in (int, float):
            try:
                parsed = parser(raw)
                break
            except ValueError:
                continue
        if raw == "true":
            parsed = True
        elif raw == "false":
            parsed = False
        elif raw == "null":
            parsed = None
        d[keys[-1]] = parsed


def main():
    parser = argparse.ArgumentParser(description="Run the banner-replacement pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override a config value, e.g. --override fitter.type=lp (repeatable)",
    )
    parser.add_argument("--save", default=None, help="Save result (image path or video path)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.override:
        _apply_overrides(config, args.override)

    mode = config.get("pipeline", {}).get("mode", "image")
    output_path = args.save or ("output.mp4" if mode == "video" else None)

    results = run(config, config_path=args.config, output_path=output_path or "output.mp4")

    if mode == "image":
        if args.save and results.get("composited") is not None:
            cv2.imwrite(args.save, results["composited"])
            print(f"Saved: {args.save}")
        elif args.save and results.get("frame") is not None:
            cv2.imwrite(args.save, results["frame"])
            print(f"Saved (no compositing): {args.save}")
    else:
        if results.get("output_path"):
            print(f"Video saved: {results['output_path']}")

    # Print metrics summary.
    metrics = results.get("metrics", {})
    if metrics:
        print("\n--- Metrics ---")
        for k, v in sorted(metrics.items()):
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
