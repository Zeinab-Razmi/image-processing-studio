import cv2
import numpy as np

def sobel_edges(img, ksize=3):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    Gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    Gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    out = np.clip(magnitude, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)

def prewitt_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    gray_f = gray.astype(np.float64)
    Kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    Ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    Gx = cv2.filter2D(gray_f, -1, Kx)
    Gy = cv2.filter2D(gray_f, -1, Ky)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    out = np.clip(magnitude, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)

def roberts_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    gray_f = gray.astype(np.float64)
    Kx = np.array([[1, 0], [0, -1]], dtype=np.float64)
    Ky = np.array([[0, 1], [-1, 0]], dtype=np.float64)
    Gx = cv2.filter2D(gray_f, -1, Kx)
    Gy = cv2.filter2D(gray_f, -1, Ky)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    out = np.clip(magnitude, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)

def canny_edges(img, low=50, high=150):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    out = cv2.Canny(gray, low, high)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)

def log_edges(img, sigma=1.0, threshold=10):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    log = cv2.Laplacian(blurred.astype(np.float64), cv2.CV_64F)
    result = np.abs(log)
    out = (result > threshold).astype(np.uint8) * 255
    return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)