"""
Lunar Crater Detection via Hough Circle Transform (M1)
======================================================
Consumes the edge map produced by `EdgeDetection.adaptive_canny` and runs
multi-scale Hough Circle detection to locate:

  1. The lunar limb (one large circle ~ disk outline)
  2. Crater rims (many small-to-medium circles inside the disk)

This is the first downstream stage of the OpNav pipeline, sitting directly
on top of EdgeDetection.py. It outputs circles in *pixel space* only --
projection to (lat, lon) is the job of M2 (Projection.py).

Usage:
    # Run on the default NASA full-disk mosaic:
    python CraterDetection.py

    # Run on a local image:
    python CraterDetection.py --image path/to/moon.png

    # Tune Hough sensitivity:
    python CraterDetection.py --dp 1.2 --param2 25 --min-radius 6 --max-radius 60

Outputs (written to ./img/ by default):
    img/lunar_crater_detections.png  -- overlay figure
    img/lunar_crater_detections.csv  -- x_px, y_px, r_px, score, kind
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from EdgeDetection import (
    NASA_SOURCES,
    adaptive_canny,
    fetch_image,
    load_local,
)

IMG_DIR = Path(__file__).resolve().parent.parent / "img"


# ---------------------------------------------------------------------------
# Limb detection
# ---------------------------------------------------------------------------

def find_limb(blurred: np.ndarray,
              min_frac: float = 0.35,
              max_frac: float = 0.55) -> tuple[int, int, int] | None:
    """
    Fit the lunar disk as a single large Hough circle.

    Strategy:
      - The Moon fills most of a full-disk mosaic frame.
      - Its radius in pixels is ~ 0.35 - 0.55 * min(H, W).
      - We use a *very* large min_dist so only one circle can win.
      - High param2 keeps false positives down.

    Returns (cx, cy, R) in pixels, or None if no confident limb fit.
    """
    h, w = blurred.shape[:2]
    short = min(h, w)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=short,                  # force at most one circle
        param1=100,                      # Canny high threshold (internal)
        param2=30,                       # accumulator threshold
        minRadius=int(min_frac * short),
        maxRadius=int(max_frac * short),
    )

    if circles is None:
        return None

    # HoughCircles already sorts by accumulator score; take the top hit.
    cx, cy, r = np.round(circles[0, 0]).astype(int)
    return int(cx), int(cy), int(r)


# ---------------------------------------------------------------------------
# Crater detection
# ---------------------------------------------------------------------------

def detect_craters(blurred: np.ndarray,
                   limb: tuple[int, int, int] | None,
                   dp: float = 1.2,
                   param1: int = 100,
                   param2: int = 25,
                   min_radius: int = 6,
                   max_radius: int = 60,
                   min_dist_frac: float = 1.0,
                   inside_margin: float = 0.98) -> np.ndarray:
    """
    Multi-scale Hough crater detection.

    Parameters
    ----------
    blurred       : adaptive-blurred grayscale image (from EdgeDetection)
    limb          : (cx, cy, R) lunar disk; used to reject off-disk hits
    dp            : inverse accumulator resolution. 1.0 = full res, >1 = coarser/faster
    param1        : internal Canny high threshold
    param2        : accumulator vote threshold. Lower = more (noisier) circles.
    min_radius    : smallest crater radius in pixels
    max_radius    : largest crater radius in pixels
    min_dist_frac : min center-to-center distance as a multiple of min_radius
    inside_margin : reject circles whose center is > (margin * R) from limb center

    Returns
    -------
    ndarray of shape (N, 4): columns = [x_px, y_px, r_px, score].
    `score` is a rim-contrast proxy in [0, 1] computed post-hoc (Hough itself
    doesn't expose vote counts through the OpenCV API).
    """
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=max(4, int(min_dist_frac * min_radius)),
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return np.empty((0, 4), dtype=np.float32)

    circles = circles[0]  # shape (N, 3): x, y, r

    # Reject anything outside the lunar disk (if we know where it is).
    if limb is not None:
        lx, ly, lr = limb
        d = np.hypot(circles[:, 0] - lx, circles[:, 1] - ly)
        keep = d < inside_margin * lr
        circles = circles[keep]

    if len(circles) == 0:
        return np.empty((0, 4), dtype=np.float32)

    # Post-hoc rim-contrast score: mean gradient magnitude on the rim ring.
    # This gives a usable ranking signal since OpenCV doesn't expose votes.
    scores = _rim_contrast_scores(blurred, circles)

    out = np.column_stack([circles, scores]).astype(np.float32)
    # Sort by score descending so top-N is "best" craters.
    out = out[np.argsort(-out[:, 3])]
    return out


def _rim_contrast_scores(img: np.ndarray, circles: np.ndarray) -> np.ndarray:
    """
    For each circle, sample intensity along its rim vs. just inside it and
    return a normalized contrast proxy in [0, 1].

    A real crater typically has a bright rim and shadowed interior (or the
    reverse depending on sun angle). Either way, |rim - interior| is large.
    Flat highlands yield near-zero contrast.
    """
    h, w = img.shape[:2]
    scores = np.zeros(len(circles), dtype=np.float32)

    # 32 samples around the rim is enough for a stable mean.
    thetas = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)

    img_f = img.astype(np.float32)

    for i, (x, y, r) in enumerate(circles):
        # Rim samples (at radius r)
        rx = np.clip(np.round(x + r * cos_t).astype(int), 0, w - 1)
        ry = np.clip(np.round(y + r * sin_t).astype(int), 0, h - 1)
        rim_vals = img_f[ry, rx]

        # Interior samples (at radius 0.5 * r)
        ix = np.clip(np.round(x + 0.5 * r * cos_t).astype(int), 0, w - 1)
        iy = np.clip(np.round(y + 0.5 * r * sin_t).astype(int), 0, h - 1)
        int_vals = img_f[iy, ix]

        scores[i] = np.abs(rim_vals.mean() - int_vals.mean()) / 255.0

    return np.clip(scores, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Visualization + export
# ---------------------------------------------------------------------------

def overlay(img: np.ndarray,
            limb: tuple[int, int, int] | None,
            craters: np.ndarray,
            title: str,
            out_path: str | Path = IMG_DIR / "lunar_crater_detections.png") -> str:
    """
    Plot the image with the limb circle + crater circles overlaid.
    Crater color is shaded by rim-contrast score.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="#0a0a0f")
    ax.imshow(img, cmap="gray", interpolation="nearest")

    if limb is not None:
        lx, ly, lr = limb
        ax.add_patch(plt.Circle((lx, ly), lr,
                                fill=False, edgecolor="#4af",
                                linewidth=1.5, linestyle="--",
                                label=f"Limb  R={lr}px"))
        ax.plot(lx, ly, "+", color="#4af", markersize=10)

    if len(craters):
        # Normalize score → colormap
        s = craters[:, 3]
        s_norm = (s - s.min()) / (s.max() - s.min() + 1e-6)
        cmap = plt.cm.plasma
        for (x, y, r, score), sn in zip(craters, s_norm):
            ax.add_patch(plt.Circle((x, y), r,
                                    fill=False,
                                    edgecolor=cmap(sn),
                                    linewidth=0.8, alpha=0.85))

    ax.set_title(
        f"{title}\n"
        f"Limb: {'yes' if limb else 'NOT FOUND'}   "
        f"Craters: {len(craters)}",
        color="white", fontsize=11, pad=10,
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Saved overlay  -> {out_path}")
    plt.show()
    return str(out_path)


def export_csv(limb: tuple[int, int, int] | None,
               craters: np.ndarray,
               out_path: str | Path = IMG_DIR / "lunar_crater_detections.csv") -> str:
    """Write detections to CSV: one row per circle (limb first, then craters)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "x_px", "y_px", "r_px", "score"])
        if limb is not None:
            lx, ly, lr = limb
            w.writerow(["limb", lx, ly, lr, 1.0])
        for x, y, r, score in craters:
            w.writerow(["crater", f"{x:.1f}", f"{y:.1f}",
                        f"{r:.1f}", f"{score:.4f}"])
    print(f"  Saved CSV      -> {out_path}")
    return str(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Hough-based lunar crater detection (M1)")
    p.add_argument("--image", type=str, default=None,
                   help="Local image path. If omitted, downloads from NASA.")
    p.add_argument("--source", type=str,
                   default="lro_nac_mosaic_1024",
                   choices=list(NASA_SOURCES.keys()),
                   help="Which NASA source to download")
    p.add_argument("--blur", type=int, default=5,
                   help="Base Gaussian kernel size for adaptive Canny")
    p.add_argument("--otsu-ratio", type=float, default=0.5,
                   help="Canny low/high ratio (passed to adaptive_canny)")
    p.add_argument("--dp", type=float, default=1.2,
                   help="Hough accumulator resolution (higher = faster, coarser)")
    p.add_argument("--param1", type=int, default=100,
                   help="Hough internal Canny high threshold")
    p.add_argument("--param2", type=int, default=25,
                   help="Hough accumulator vote threshold (lower = more circles)")
    p.add_argument("--min-radius", type=int, default=6,
                   help="Smallest crater radius in px")
    p.add_argument("--max-radius", type=int, default=60,
                   help="Largest crater radius in px")
    p.add_argument("--top", type=int, default=None,
                   help="Keep only top-N craters by rim-contrast score")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n=== Lunar Crater Detection (Hough) ===\n")

    # Stage 0: load
    if args.image:
        print(f"Loading local image: {args.image}")
        img = load_local(args.image)
        title = Path(args.image).name
    else:
        print(f"Fetching NASA source: {args.source}")
        img = fetch_image(args.source)
        title = NASA_SOURCES[args.source]["desc"]
    print(f"  Image shape: {img.shape}  dtype: {img.dtype}")

    # Stage 1: reuse EdgeDetection.adaptive_canny
    print(f"\nStage 1 -- Adaptive Canny (blur={args.blur}, "
          f"otsu_ratio={args.otsu_ratio})")
    result = adaptive_canny(img, blur_ksize=args.blur,
                            otsu_ratio=args.otsu_ratio)
    blurred = result["blurred"]
    lo, hi = result["thresholds"]
    print(f"  Canny thresholds: low={lo}  high={hi}")

    # Stage 2a: limb fit
    print("\nStage 2a -- Limb fit (single large Hough circle)")
    limb = find_limb(blurred)
    if limb:
        print(f"  Limb: center=({limb[0]}, {limb[1]})  R={limb[2]}px")
    else:
        print("  Limb NOT found -- crater filter will not reject off-disk hits.")

    # Stage 2b: craters
    print(f"\nStage 2b -- Crater detection (r in [{args.min_radius}, "
          f"{args.max_radius}] px)")
    craters = detect_craters(
        blurred, limb,
        dp=args.dp,
        param1=args.param1,
        param2=args.param2,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
    )
    print(f"  Detected {len(craters)} crater candidates")

    if args.top is not None and len(craters) > args.top:
        craters = craters[: args.top]
        print(f"  Keeping top {args.top} by rim-contrast score")

    # Stage 3: export + visualize
    print("\nStage 3 -- Export")
    export_csv(limb, craters)
    overlay(img, limb, craters, title=title)

    print("\nDone. Pixel-space detections ready. "
          "Next: M2 (pixel -> lat/lon projection).\n")

    return {"limb": limb, "craters": craters}


if __name__ == "__main__":
    main()
