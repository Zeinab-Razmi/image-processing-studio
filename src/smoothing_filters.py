import cv2

def mean_filter(img, kernel_size=5):
    return cv2.blur(img, (kernel_size, kernel_size))

def gaussian_filter(img, kernel_size=5, sigma=0):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

def median_filter(img, kernel_size=5):
    return cv2.medianBlur(img, kernel_size)

def bilateral_filter(img, kernel_size=5, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(img, kernel_size, sigma_color, sigma_space)