import cv2
import numpy as np
from skimage import color as sk_color


def rgb_to_grayscale(image):
    img = image.astype(np.float32)

    gray = (
        0.299 * img[:, :, 0] +
        0.587 * img[:, :, 1] +
        0.114 * img[:, :, 2]
    )
    return np.clip(gray, 0, 255).astype(np.uint8)

def simulate_deuteranopia(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    is_scaled = False
    if img.max() > 1.0:
        img /= 255.0
        is_scaled = True

    matrix = np.array([
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501, 0.047413],
        [-0.011820, 0.042940, 0.968881],
    ])
    simulated = img @ matrix.T
    simulated = np.clip(simulated, 0, 1)

    if is_scaled:
        simulated = (simulated * 255.0).astype(np.uint8)
    return simulated


def rgb_to_hsv_view(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    hsv = sk_color.rgb2hsv(img)
    return (hsv * 255.0).astype(np.uint8)


def white_balance(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    means = img.mean(axis=(0, 1))
    global_mean = means.mean()
    scale = global_mean / (means + 1e-6)
    balanced = img * scale[np.newaxis, np.newaxis, :]
    return np.clip(balanced, 0, 255).astype(image.dtype)