import cv2
import numpy as np


def inversion(img: np.ndarray) -> np.ndarray:
    return 255 - img

def log_transform(img: np.ndarray, c: float) -> np.ndarray:

    img_float = img.astype(np.float64)
    result = c * (np.log1p(img_float) / np.log(256)) * 255
    return np.clip(result, 0, 255).astype(np.uint8)

def gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
    table = np.array([(i / 255.0) ** gamma * 255.0 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def histogram_equalization(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return cv2.equalizeHist(img)
    # Color: equalize only the Y (luminance) channel
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

def apply_clahe(img, clip_limit=2.0, tile_grid_size=8):
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid_size), int(tile_grid_size)))

    if len(img.shape) == 2 or img.shape[2] == 1:
        return clahe.apply(img)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    cl = clahe.apply(l_channel)

    merged_lab = cv2.merge((cl, a_channel, b_channel))
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)


def contrast_stretching(img: np.ndarray, r1: int = 70, s1: int = 0, r2: int = 180, s2: int = 255):
    def stretch(pixel):
        if pixel < r1:
            return int((s1 / max(r1, 1)) * pixel)
        elif pixel < r2:
            return int(s1 + ((pixel - r1) / max(r2 - r1, 1)) * (s2 - s1))
        else:
            return int(s2 + ((pixel - r2) / max(255 - r2, 1)) * (255 - s2))

    table = np.array([stretch(i) for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)



