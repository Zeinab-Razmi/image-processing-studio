import cv2
import numpy as np


def add_gaussian_noise(img: np.ndarray, mean: float = 0.0, sigma: float = 25.0) -> np.ndarray:
    noise = np.random.normal(mean, sigma, img.shape).astype(np.float64)
    noisy = img.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_and_pepper(img: np.ndarray, density: float = 0.05) -> np.ndarray:
    noisy = img.copy()

    rows, cols = img.shape[0], img.shape[1]
    num_pixels = rows * cols

    num_salt = int(num_pixels * density / 2)
    num_pepper = int(num_pixels * density / 2)

    salt_r = np.random.randint(0, rows, num_salt)
    salt_c = np.random.randint(0, cols, num_salt)
    noisy[salt_r, salt_c] = 255

    pepper_r = np.random.randint(0, rows, num_pepper)
    pepper_c = np.random.randint(0, cols, num_pepper)
    noisy[pepper_r, pepper_c] = 0

    return noisy


def add_speckle_noise(img: np.ndarray, variance: float = 0.04) -> np.ndarray:
    noise = np.random.randn(*img.shape) * np.sqrt(variance)
    noisy = img.astype(np.float64) + img.astype(np.float64) * noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_periodic_noise(img: np.ndarray, frequency: float = 30.0, amplitude: float = 30.0) -> np.ndarray:
    rows, cols = img.shape[:2]
    x = np.arange(cols)
    pattern = amplitude * np.sin(2 * np.pi * frequency * x / cols)
    noise_2d = np.tile(pattern, (rows, 1))

    if len(img.shape) == 3:
        noise_2d = np.stack([noise_2d] * 3, axis=2)

    noisy = img.astype(np.float64) + noise_2d
    return np.clip(noisy, 0, 255).astype(np.uint8)