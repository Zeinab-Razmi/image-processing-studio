import os
import cv2
import math
import matplotlib.pyplot as plt

import src.noise_models as ns
import src.morphological as morph
import src.smoothing_filters as sm
import src.edge_detection as ed
import src.segmentation as seg
import src.colorspace as cs
import src.enhancement as enh


OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread("image/ballons.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_comparison_figure(original, results_dict, category_name):
    os.makedirs("output", exist_ok=True)

    total_images = len(results_dict) + 1
    cols = 3
    rows = math.ceil(total_images / cols)

    plt.figure(figsize=(18, 6 * rows))

    # Original image
    plt.subplot(rows, cols, 1)

    if original.ndim == 2:
        plt.imshow(original, cmap="gray")
    else:
        plt.imshow(original)

    plt.title("Original")
    plt.axis("off")

    for idx, (title, img_proc) in enumerate(results_dict.items(), start=2):

        plt.subplot(rows, cols, idx)

        if img_proc.ndim == 2:
            plt.imshow(img_proc, cmap="gray")
        else:
            plt.imshow(img_proc)

        plt.title(title)
        plt.axis("off")

    plt.suptitle(category_name, fontsize=18)
    plt.tight_layout()

    filename = f"{category_name.replace(' ', '_')}_Comparison.png"

    plt.savefig(
        os.path.join("output", filename),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: output/{filename}")


def save_histogram_comparison(original, results_dict, category_name):
    """
    Save histogram-only comparison figure:
    Original + all processed outputs
    """

    os.makedirs("output", exist_ok=True)

    total = len(results_dict) + 1
    cols = 3
    rows = math.ceil(total / cols)

    plt.figure(figsize=(18, 5 * rows))

    def plot_hist(img, title, idx):
        plt.subplot(rows, cols, idx)

        if img.ndim == 2:
            plt.hist(img.ravel(), bins=256, color="black", alpha=0.8)
        else:
            colors = ["r", "g", "b"]
            for c, col in enumerate(colors):
                plt.hist(img[:, :, c].ravel(), bins=256, color=col, alpha=0.4)

        plt.title(title)
        plt.axis("off")

    # Original histogram
    plot_hist(original, "Original", 1)

    # Processed histograms
    for i, (title, img_proc) in enumerate(results_dict.items(), start=2):
        plot_hist(img_proc, title, i)

    plt.suptitle(category_name + " (Histogram Comparison)", fontsize=18)
    plt.tight_layout()

    filename = f"{category_name.replace(' ', '_')}_Histograms.png"

    plt.savefig(
        os.path.join("output", filename),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: output/{filename}")


# Morphological Operations
morph_dict = {
    "Erosion": morph.erosion(img, 5),
    "Dilation": morph.dilation(img, 5),
    "Opening": morph.opening(img, 5),
    "Closing": morph.closing(img, 5),
    "Gradient": morph.morphological_gradient(img, 5),
    "Top Hat": morph.top_hat(img, 5),
    "Black Hat": morph.black_hat(img, 5),
}

save_comparison_figure(img, morph_dict, "Morphological Operations")
save_histogram_comparison(img, morph_dict, "Morphological Operations")


# Edge Detection
edge_dict = {
    "Sobel": ed.sobel_edges(img),
    "Prewitt": ed.prewitt_edges(img),
    "Roberts": ed.roberts_edges(img),
    "Canny": ed.canny_edges(img, 50, 150),
    "LoG": ed.log_edges(img, sigma=1.0, threshold=10),
}

save_comparison_figure(img, edge_dict, "Edge Detection Filters")
save_histogram_comparison(img, edge_dict, "Edge Detection Filters")


# Enhancement
enh_dict = {
    "Gamma Correction": enh.gamma_correction(img, gamma=1.5),
    "Negative / Inversion": enh.inversion(img),
    "Log Transform": enh.log_transform(img, 1.0),
    "Histogram Equalization": enh.histogram_equalization(img),
    "CLAHE": enh.apply_clahe(img, clip_limit=2.0, tile_grid_size=8),
    "Contrast Stretching": enh.contrast_stretching(img, 70, 0, 180, 255),
}

save_comparison_figure(img, enh_dict, "Intensity Transforms & Enhancements")
save_histogram_comparison(img, enh_dict, "Intensity Transforms & Enhancements")


# Smoothing Filters
smooth_dict = {
    "Gaussian Filter": sm.gaussian_filter(img, 5, sigma=0),
    "Mean Filter": sm.mean_filter(img, 5),
    "Median Filter": sm.median_filter(img, 5),
    "Bilateral Filter": sm.bilateral_filter(img, 5),
}

save_comparison_figure(img, smooth_dict, "Smoothing Filters")
save_histogram_comparison(img, smooth_dict, "Smoothing Filters")


# Noise Models
noise_dict = {
    "Gaussian Noise": ns.add_gaussian_noise(img, sigma=15),
    "Salt & Pepper": ns.add_salt_and_pepper(img, density=0.05),
    "Speckle Noise": ns.add_speckle_noise(img, variance=0.05),
    "Periodic Noise": ns.add_periodic_noise(img, frequency=20),
}

save_comparison_figure(img, noise_dict, "Noise Models")
save_histogram_comparison(img, noise_dict, "Noise Models")


# Color Spaces
color_dict = {
    "RGB to Grayscale": cs.rgb_to_grayscale(img),
    "Simulate Deuteranopia": cs.simulate_deuteranopia(img),
    "RGB to HSV": cs.rgb_to_hsv_view(img),
    "White Balance": cs.white_balance(img),
}

save_comparison_figure(img, color_dict, "Color Space & Transforms")
save_histogram_comparison(img, color_dict, "Color Space & Transforms")


# Segmentation
seg_dict = {
    "Multi-Otsu": seg.multi_otsu(img, target_classes=3),
    "Global Thresholding": seg.global_threshold(img, thresh=127),
    "Otsu's Method": seg.otsu_threshold(img),
    "Adaptive Gaussian": seg.adaptive_threshold(img, method_type="gaussian", block_size=11),
    "Adaptive Mean": seg.adaptive_threshold(img, method_type="mean", block_size=11),
    "K-Means Clustering": seg.kmeans_segmentation(img, k=3),
}

save_comparison_figure(img, seg_dict, "Segmentation & Clustered Bins")
save_histogram_comparison(img, seg_dict, "Segmentation & Clustered Bins")
