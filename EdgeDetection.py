"""
Lunar Adaptive Canny Edge Detection
====================================
Fetches NASA LRO lunar imagery directly and runs adaptive Canny edge detection.
Designed as the preprocessing stage feeding into a Hough Transform pipeline.

Images sourced from:
  - NASA SVS LRO NAC Mosaic (full disk, 1024px)
  - USGS Astropedia LOLA DEM browse image
  - NASA SVS thumbnail (smaller, faster to test with)

Usage:
    python lunar_adaptive_canny.py

    # Or point at your own image:
    python lunar_adaptive_canny.py --image path/to/your/lunar.png

    # Adjust Canny sensitivity:
    python lunar_adaptive_canny.py --otsu-ratio 0.4 --blur 5
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# NASA public image sources — no auth required
# ---------------------------------------------------------------------------
NASA_SOURCES = {
    "lro_nac_mosaic_1024": {
        "url": "https://svs.gsfc.nasa.gov/vis/a000000/a005000/a005001/moon_mosaic_print.jpg",
        "desc": "LRO NAC Full-Disk Mosaic 1024px (NASA SVS)",
    },
    "lola_dem_browse": {
        "url": "https://astrogeology.usgs.gov/ckan/dataset/8d95ec67-e637-4b48-88ed-84e1f95660fc/resource/6ce79c01-d8a2-4b54-bae1-31bf4efc6e4b/download/moon_lro_lola_global_ldem_1024.jpg",
        "desc": "LOLA DEM Global Browse Image 1024px (USGS)",
    },
}

CACHE_DIR = Path("/tmp/lunar_cache")


# ---------------------------------------------------------------------------
# Image acquisition
# ---------------------------------------------------------------------------

def fetch_image(key: str) -> np.ndarray:
    """Download a NASA lunar image (cached locally after first fetch)."""
    CACHE_DIR.mkdir(exist_ok=True)
    source = NASA_SOURCES[key]
    cache_path = CACHE_DIR / f"{key}.jpg"

    if not cache_path.exists():
        print(f"  Downloading: {source['desc']}")
        print(f"  URL: {source['url']}")
        req = urllib.request.Request(
            source["url"],
            headers={"User-Agent": "Mozilla/5.0 (lunar-nav-research)"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        cache_path.write_bytes(data)
        print(f"  Saved to {cache_path} ({len(data) / 1024:.1f} KB)")
    else:
        print(f"  Using cached: {cache_path}")

    img = cv2.imread(str(cache_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"cv2 could not load {cache_path}")
    return img


def load_local(path: str) -> np.ndarray:
    """Load a local image file (supports 8-bit and 16-bit)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {path}")
    # 16-bit → 8-bit
    if img.dtype == np.uint16:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Color → gray
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# ---------------------------------------------------------------------------
# Adaptive Canny implementation
# ---------------------------------------------------------------------------

def adaptive_gaussian_blur(img: np.ndarray, base_ksize: int = 5) -> np.ndarray:
    """
    Pixel-wise adaptive Gaussian blur.

    Core idea from the adaptive Canny literature:
      - Compute local mean in a window around each pixel
      - Pixels that deviate strongly from local mean = likely at an edge
        → apply LESS blur to preserve the edge
      - Pixels close to local mean = smooth region / noise
        → apply MORE blur to suppress noise

    In practice we approximate this with two fixed-scale blurs
    and blend them using a local variance map.
    This avoids per-pixel kernel computation (too slow for flight prototype)
    while capturing the same intent.

    Returns: blended blur image (uint8)
    """
    ksize_small = max(3, base_ksize - 2) | 1   # e.g. 3 — preserve edges
    ksize_large = (base_ksize + 2) | 1          # e.g. 7 — suppress noise

    blur_soft = cv2.GaussianBlur(img, (ksize_small, ksize_small), 0)
    blur_hard = cv2.GaussianBlur(img, (ksize_large, ksize_large), 0)

    # Local variance map — high variance = likely edge region
    img_f = img.astype(np.float32)
    local_mean = cv2.GaussianBlur(img_f, (ksize_large, ksize_large), 0)
    local_sq_mean = cv2.GaussianBlur(img_f ** 2, (ksize_large, ksize_large), 0)
    variance = local_sq_mean - local_mean ** 2
    variance = np.clip(variance, 0, None)

    # Normalize variance to [0, 1] — this is the blend weight
    # High variance → use soft blur (edge preservation)
    # Low variance  → use hard blur (noise suppression)
    var_norm = variance / (variance.max() + 1e-6)

    blended = (var_norm * blur_soft.astype(np.float32)
               + (1.0 - var_norm) * blur_hard.astype(np.float32))
    return np.clip(blended, 0, 255).astype(np.uint8)


def otsu_canny_thresholds(img: np.ndarray,
                           low_ratio: float = 0.5) -> tuple[int, int]:
    """
    Compute Canny thresholds automatically from the image gradient histogram.

    Otsu finds the intensity that best separates foreground/background.
    We use it on the blurred image as the HIGH threshold and derive
    the LOW threshold as a fraction of it.

    low_ratio: 0.4–0.5 is typical. Lower = more edges (more noise risk).
    """
    high, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low = int(low_ratio * high)
    return low, int(high)


def adaptive_canny(img: np.ndarray,
                   blur_ksize: int = 5,
                   otsu_ratio: float = 0.5) -> dict:
    """
    Full adaptive Canny pipeline.

    Returns a dict of all intermediate stages so you can inspect
    exactly what each step contributed — important for tuning and
    for F Prime telemetry later.

    Stages:
        raw        → input image
        blurred    → adaptive gaussian output
        edges      → final binary edge map (input to Hough)
        thresholds → (low, high) used by Canny
        variance   → local variance map (shows where adaptation fired)
    """
    # Stage 1: Adaptive Gaussian
    blurred = adaptive_gaussian_blur(img, base_ksize=blur_ksize)

    # Stage 2: Auto-threshold via Otsu
    low_thresh, high_thresh = otsu_canny_thresholds(blurred, low_ratio=otsu_ratio)

    # Stage 3: Canny on adaptively-blurred image
    edges = cv2.Canny(blurred, low_thresh, high_thresh)

    # Compute variance map for visualization
    img_f = img.astype(np.float32)
    ksize = (blur_ksize + 2) | 1
    local_mean = cv2.GaussianBlur(img_f, (ksize, ksize), 0)
    local_sq   = cv2.GaussianBlur(img_f ** 2, (ksize, ksize), 0)
    variance   = np.clip(local_sq - local_mean ** 2, 0, None)
    var_vis    = cv2.normalize(variance, None, 0, 255,
                               cv2.NORM_MINMAX).astype(np.uint8)

    return {
        "raw":        img,
        "blurred":    blurred,
        "edges":      edges,
        "thresholds": (low_thresh, high_thresh),
        "variance":   var_vis,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize(result: dict, title: str = "Adaptive Canny — Lunar Image"):
    """
    4-panel figure:
      [Raw] [Variance Map] [Adaptively Blurred] [Edge Map]

    The variance map shows WHERE the adaptive blur applied soft vs hard
    smoothing — bright = edge regions (soft blur applied).
    """
    fig = plt.figure(figsize=(18, 5), facecolor="#0a0a0f")
    fig.suptitle(title, color="white", fontsize=13, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(1, 4, wspace=0.04)

    panels = [
        ("Raw Input",           result["raw"],      "gray",    None),
        ("Local Variance Map",  result["variance"], "inferno", "→ bright = edge zone\n(soft blur applied)"),
        ("Adaptive Blur",       result["blurred"],  "gray",    None),
        ("Edge Map → Hough",    result["edges"],    "gray",    f"Canny thresholds:\nlo={result['thresholds'][0]}  hi={result['thresholds'][1]}"),
    ]

    for i, (label, data, cmap, note) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        ax.imshow(data, cmap=cmap, interpolation="nearest")
        ax.set_title(label, color="white", fontsize=10, pad=6)
        ax.axis("off")

        if note:
            ax.text(0.5, -0.04, note, transform=ax.transAxes,
                    ha="center", va="top", fontsize=7.5,
                    color="#aaaacc", style="italic")

        # Highlight the output panel
        if i == 3:
            for spine in ax.spines.values():
                spine.set_edgecolor("#4af")
                spine.set_linewidth(1.5)
                spine.set_visible(True)

    # Edge pixel stats
    edge_pct = 100 * np.count_nonzero(result["edges"]) / result["edges"].size
    fig.text(0.5, -0.02,
             f"Edge pixels: {np.count_nonzero(result['edges']):,}  "
             f"({edge_pct:.2f}% of image)  |  "
             f"Image size: {result['raw'].shape[1]}×{result['raw'].shape[0]}",
             ha="center", color="#888", fontsize=9)

    plt.tight_layout()
    out_path = "/tmp/lunar_adaptive_canny_result.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  Saved figure → {out_path}")
    plt.show()
    return out_path


def compare_fixed_vs_adaptive(img: np.ndarray,
                               blur_ksize: int = 5,
                               otsu_ratio: float = 0.5):
    """
    Side-by-side comparison: fixed Gaussian Canny vs Adaptive Canny.
    Useful for justifying the adaptive approach.
    """
    # Fixed Canny
    fixed_blur = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    lo, hi = otsu_canny_thresholds(fixed_blur, otsu_ratio)
    fixed_edges = cv2.Canny(fixed_blur, lo, hi)

    # Adaptive Canny
    adaptive_result = adaptive_canny(img, blur_ksize, otsu_ratio)
    adaptive_edges = adaptive_result["edges"]

    # Diff map — what did adaptive recover that fixed missed?
    diff = cv2.bitwise_and(adaptive_edges,
                           cv2.bitwise_not(fixed_edges))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                              facecolor="#0a0a0f")
    fig.suptitle("Fixed vs Adaptive Canny Comparison",
                 color="white", fontsize=12, fontweight="bold")

    data = [
        ("Fixed Gaussian + Otsu Canny", fixed_edges),
        ("Adaptive Gaussian + Otsu Canny", adaptive_edges),
        ("Edges in Adaptive NOT in Fixed\n(what adaptive recovered)", diff),
    ]
    for ax, (label, img_data) in zip(axes, data):
        ax.imshow(img_data, cmap="gray")
        ax.set_title(label, color="white", fontsize=9, pad=6)
        ax.axis("off")
        count = np.count_nonzero(img_data)
        ax.text(0.5, 0.01, f"{count:,} edge px",
                transform=ax.transAxes, ha="center",
                color="#88ccff", fontsize=8)

    plt.tight_layout()
    out_path = "/tmp/lunar_canny_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Saved comparison → {out_path}")
    plt.show()
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Adaptive Canny for Lunar Imagery")
    p.add_argument("--image", type=str, default=None,
                   help="Path to local image. If omitted, downloads from NASA.")
    p.add_argument("--source", type=str,
                   default="lro_nac_mosaic_1024",
                   choices=list(NASA_SOURCES.keys()),
                   help="Which NASA source to download (default: lro_nac_mosaic_1024)")
    p.add_argument("--blur", type=int, default=5,
                   help="Base Gaussian kernel size (odd int, default 5)")
    p.add_argument("--otsu-ratio", type=float, default=0.5,
                   help="Low/high threshold ratio for Canny (default 0.5)")
    p.add_argument("--compare", action="store_true",
                   help="Also show fixed vs adaptive comparison figure")
    p.add_argument("--crop", type=str, default=None,
                   help="Crop region as x,y,w,h e.g. --crop 200,100,400,400")
    return p.parse_args()


def main():
    args = parse_args()

    # ---- Load image ----
    print("\n=== Lunar Adaptive Canny Edge Detection ===\n")
    if args.image:
        print(f"Loading local image: {args.image}")
        img = load_local(args.image)
        title = Path(args.image).name
    else:
        print(f"Fetching NASA source: {args.source}")
        img = fetch_image(args.source)
        title = NASA_SOURCES[args.source]["desc"]

    print(f"  Image shape: {img.shape}  dtype: {img.dtype}")

    # ---- Optional crop ----
    if args.crop:
        x, y, w, h = map(int, args.crop.split(","))
        img = img[y:y+h, x:x+w]
        print(f"  Cropped to: {img.shape}")
        title += f" [crop {x},{y},{w},{h}]"

    # ---- Run adaptive Canny ----
    print(f"\nRunning adaptive Canny  (blur={args.blur}, otsu_ratio={args.otsu_ratio})")
    result = adaptive_canny(img, blur_ksize=args.blur, otsu_ratio=args.otsu_ratio)

    lo, hi = result["thresholds"]
    edge_count = np.count_nonzero(result["edges"])
    print(f"  Otsu thresholds → low={lo}  high={hi}")
    print(f"  Edge pixels: {edge_count:,}  ({100*edge_count/result['edges'].size:.2f}%)")
    print(f"  Edge map shape: {result['edges'].shape}  ← this feeds Hough Transform")

    # ---- Visualize ----
    visualize(result, title=title)

    if args.compare:
        print("\nGenerating fixed vs adaptive comparison...")
        compare_fixed_vs_adaptive(img, args.blur, args.otsu_ratio)

    print("\nDone. Edge map is in result['edges'] — ready for Hough input.\n")
    return result


if __name__ == "__main__":
    main()