import cv2
import logging
import numpy as np
from skimage.filters import threshold_multiotsu

logger = logging.getLogger(__name__)


def global_threshold(img, thresh=127):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    _, result = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)


def otsu_threshold(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)


def adaptive_threshold(img, method_type="gaussian", block_size=11, c_val=2):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img

    if block_size % 2 == 0:
        block_size = max(3, block_size + 1)

    method = (cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method_type.lower() == "gaussian"
              else cv2.ADAPTIVE_THRESH_MEAN_C)

    result = cv2.adaptiveThreshold(gray, 255, method, cv2.THRESH_BINARY, block_size, c_val)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)


def multi_otsu(img, target_classes=3):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    try:

        thresholds = threshold_multiotsu(gray, classes=target_classes)
        regions = np.digitize(gray, bins=thresholds)
        normalized = np.uint8(255 * regions / regions.max())
        colorful_segments = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        return cv2.cvtColor(colorful_segments, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.warning(f"Multi-Otsu distribution mismatch. Error: {e}. Returning source matrix.")
        return img

def kmeans_segmentation(img, k=3):
    assert len(img.shape) == 3, "K-Means execution module requires multi-channel color maps"

    data = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    try:
        _, labels, centers = cv2.kmeans(data, int(k), None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = centers.astype(np.uint8)
        return centers[labels.flatten()].reshape(img.shape)
    except Exception as e:
        logger.error(f"Clustering matrix initialization error: {e}")
        return img