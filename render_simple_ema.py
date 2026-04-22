from __future__ import annotations
import argparse, json, os, subprocess, sys
import cv2, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from banner_segment import fit_quadrilateral, composite_logo

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--logo-long", default=None,
                    help="Logo for long-axis banners")
    ap.add_argument("--logo-short", default=None,
                    help="Logo for short-axis banners")
    ap.add_argument("--logo", default=None,
                    help="Fallback logo for all banners (if --logo-long/--logo-short not set)")
    ap.add_argument("--ema-alpha", type=float, default=0.02)
    ap.add_argument("--out", default="output_simple_ema.mp4")
    args = ap.parse_args()

    logo_long = args.logo_long or args.logo
    logo_short = args.logo_short or args.logo
    if not logo_long and not logo_short:
        print("ERROR: provide --logo or --logo-long / --logo-short")
        return

    cache = args.cache_dir
    meta = json.load(open(os.path.join(cache, "meta.json")))
    n_frames, fps = meta["n_frames"], meta["fps"]
    w, h = meta["width"], meta["height"]
    player_ids = meta["player_ids"]
    frames_dir = os.path.join(cache, "frames")
    masks_dir = os.path.join(cache, "player_masks")
    alpha = args.ema_alpha

    # Map each banner ID to its axis + logo
    long_ids = set(meta.get("banner_long_ids", []))
    short_ids = set(meta.get("banner_short_ids", []))
    banner_ids = meta["banner_ids"]

    # If old cache without long/short split, treat all as short
    if not long_ids and not short_ids:
        short_ids = set(banner_ids)

    def axis_for(bid):
        return "long" if bid in long_ids else "short"

    def logo_for(bid):
        return logo_long if bid in long_ids else logo_short

    print(f"  Long-axis banners: {sorted(long_ids)} → {logo_long}")
    print(f"  Short-axis banners: {sorted(short_ids)} → {logo_short}")

    ema = {}
    raw = args.out.rsplit(".", 1)[0] + "_raw.mp4"
    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for fi in range(n_frames):
        frame = cv2.imread(os.path.join(frames_dir, f"{fi:05d}.jpg"))
        data = np.load(os.path.join(masks_dir, f"{fi:05d}.npz"))

        occ = np.zeros((h, w), dtype=bool)
        for pid in player_ids:
            k = f"obj{pid}"
            if k in data:
                occ |= data[k].astype(bool)

        result = frame
        for bid in banner_ids:
            k = f"obj{bid}"
            if k not in data:
                continue
            bmask = data[k].astype(np.uint8)
            if bmask.sum() < 50:
                continue

            logo = logo_for(bid)
            if not logo:
                continue

            corners = fit_quadrilateral(bmask, axis=axis_for(bid),
                                        debug=False, verbose=False)
            if corners is None:
                if bid in ema:
                    smooth = ema[bid].astype(np.float32)
                else:
                    continue
            else:
                if bid not in ema:
                    ema[bid] = corners.astype(np.float64)
                else:
                    ema[bid] = alpha * corners.astype(np.float64) + (1 - alpha) * ema[bid]
                smooth = ema[bid].astype(np.float32)

            result = composite_logo(result, smooth, logo, mask=bmask,
                                    occlusion_mask=occ if occ.any() else None,
                                    save_path=None, debug=False, verbose=False)

        writer.write(result)
        if (fi + 1) % 50 == 0 or fi == n_frames - 1:
            print(f"  {fi + 1}/{n_frames}", flush=True)

    writer.release()
    subprocess.run(["ffmpeg", "-i", raw, "-c:v", "libx264", "-preset", "fast",
                    "-crf", "23", "-pix_fmt", "yuv420p", args.out, "-y",
                    "-loglevel", "error"], check=True)
    os.remove(raw)
    print(f"Done: {args.out}")

if __name__ == "__main__":
    main()
