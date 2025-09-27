#!/usr/bin/env python3
"""
Single-image Trackpy with circular ROI selection.
- First run: shows image, lets you drag a circle ROI.
- ROI center/radius saved to roi.json.
- Next runs: auto-uses saved ROI, no dragging needed.
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import EllipseSelector
import trackpy as tp
from skimage.io import imread

# --- Config ---
IMAGE_FILENAME = "1758927055861_frame_000000.jpg"
ROI_FILE = "roi.json"
OUTPUT_DIR = "tp_output"
DIAMETER = 15
MINMASS = 150
INVERT = False
# -------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

def select_circular_roi(img):
    """Click once for center, once for edge -> returns cx, cy, r"""
    roi = {}

    pts = []

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        pts.append((event.xdata, event.ydata))
        if len(pts) == 2:
            cx, cy = pts[0]
            ex, ey = pts[1]
            r = int(((ex - cx)**2 + (ey - cy)**2)**0.5)
            roi['cx'], roi['cy'], roi['r'] = int(cx), int(cy), r
            plt.close()

    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.set_title("Click center, then edge of circle ROI")
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return roi


def apply_circular_mask(img, cx, cy, r):
    """Return masked image with circle ROI"""
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    mask = (xx - cx)**2 + (yy - cy)**2 <= r**2
    return np.where(mask, img, 0)

def main():
    frame = imread(IMAGE_FILENAME, as_gray=True)
    if frame.ndim == 3:
        frame = np.mean(frame, axis=2).astype(frame.dtype)

    # Load or select ROI
    if os.path.exists(ROI_FILE):
        with open(ROI_FILE, "r") as f:
            roi = json.load(f)
        print(f"[tp] Loaded ROI: {roi}")
    else:
        roi = select_circular_roi(frame)
        with open(ROI_FILE, "w") as f:
            json.dump(roi, f)
        print(f"[tp] Saved ROI to {ROI_FILE}: {roi}")

    # Apply circular mask
    masked = apply_circular_mask(frame, roi['cx'], roi['cy'], roi['r'])

    # Locate features
    f = tp.locate(masked, diameter=DIAMETER, minmass=MINMASS, invert=INVERT)
    print(f"[tp] Found {len(f)} particles in circular ROI")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(masked, cmap='gray')
    # tp.annotate(f, masked, ax=ax, color='yellow')
    ax.plot(f['x'], f['y'], 'ro', markersize=2)  # centroid dots
    circle = plt.Circle((roi['cx'], roi['cy']), roi['r'],
                        color='cyan', fill=False, linewidth=1.5)
    ax.add_patch(circle)
    ax.set_title(f"{len(f)} particles (ROI circle d={DIAMETER}, minmass={MINMASS})")
    ax.axis("off")
    out_path = os.path.join(OUTPUT_DIR, "annot_circular_roi.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"[tp] Wrote {out_path}")

    # Save features CSV
    csv_path = os.path.join(OUTPUT_DIR, "features_circular_roi.csv")
    f.to_csv(csv_path, index=False)
    print(f"[tp] Wrote {csv_path}")

if __name__ == "__main__":
    main()
