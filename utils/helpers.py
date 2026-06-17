import cv2
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import io


def generate_live_histogram(img: np.ndarray) -> Image.Image:

    if len(img.shape) == 3 and img.shape[2] == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = img.copy()

    flattened_pixels = gray_img.ravel() / 255.0

    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=100)

    counts, bins, patches = ax.hist(
        flattened_pixels,
        bins=256,
        range=(0.0, 1.0),
        color="#fca35d",
        histtype='stepfilled',
        edgecolor='none',
        alpha=0.9
    )

    ax.set_xlabel("Intensity", fontsize=9, fontname="sans-serif")
    ax.set_ylabel("Count", fontsize=9, fontname="sans-serif")
    ax.set_xlim(-0.03, 1.03)  # Slight padding so endpoints don't clip tightly

    max_count = counts.max() if len(counts) > 0 else 1.0
    ax.set_ylim(0, max_count * 1.05)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')

    ax.tick_params(axis='both', colors='#333333', labelsize=8)
    fig.tight_layout()

    buf = io.BytesIO()
    canvas = fig.canvas
    canvas.print_png(buf)
    buf.seek(0)

    plt.close(fig)

    return Image.open(buf)