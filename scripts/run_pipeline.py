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

from banner_pipeline.pipeline import load_config, run_pipeline


def _apply_overrides(config: dict, overrides: list[str]) -> None:
    """Apply dot-notation overrides like ``fitter.type=lp`` to *config*."""
    for ov in overrides:
        key_path, _, value = ov.partition("=")
        keys = key_path.split(".")
        d = config.setdefault("pipeline", {})
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        # Try to parse as int/float/bool, else keep as string.
        for parser in (int, float):
            try:
                value = parser(value)
                break
            except ValueError:
                continue
        if value == "true":
            value = True
        elif value == "false":
            value = False
        elif value == "null":
            value = None
        d[keys[-1]] = value


def main():
    parser = argparse.ArgumentParser(description="Run the banner-replacement pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--override", action="append", default=[],
        help="Override a config value, e.g. --override fitter.type=lp (repeatable)",
    )
    parser.add_argument("--save", default=None, help="Save composited result to this path")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.override:
        _apply_overrides(config, args.override)

    results = run_pipeline(config, config_path=args.config)

    if args.save and results["composited"] is not None:
        cv2.imwrite(args.save, results["composited"])
        print(f"Saved: {args.save}")
    elif args.save and results["frame"] is not None:
        cv2.imwrite(args.save, results["frame"])
        print(f"Saved (no compositing): {args.save}")

    # Print metrics summary.
    metrics = results.get("metrics", {})
    if metrics:
        print("\n--- Metrics ---")
        for k, v in sorted(metrics.items()):
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
