import cv2
import numpy as np

def get_kernel(shape="rect", size=3):
    shapes = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS,
    }
    return cv2.getStructuringElement(shapes.get(shape, cv2.MORPH_RECT), (size, size))

def erosion(img, kernel_size=5):
    return cv2.erode(img, get_kernel("ellipse", kernel_size))

def dilation(img, kernel_size=5):
    return cv2.dilate(img, get_kernel("ellipse", kernel_size))

def opening(img, kernel_size=5):
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, get_kernel("ellipse", kernel_size))

def closing(img, kernel_size=5):
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, get_kernel("ellipse", kernel_size))

def morphological_gradient(img, kernel_size=3):
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, get_kernel("rect", kernel_size))

def top_hat(img, kernel_size=9):
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, get_kernel("rect", kernel_size))

def black_hat(img, kernel_size=9):
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, get_kernel("rect", kernel_size))


