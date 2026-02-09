"""
Spatial Domain Image Blurring Module

This module implements image blurring using spatial domain convolution
with a Gaussian blur kernel.
"""

import numpy as np
from typing import Tuple


def create_gaussian_kernel(kernel_size: int, sigma: float = 1.0) -> np.ndarray:
    """
    Create a 2D Gaussian blur kernel.

    Parameters:
    -----------
    kernel_size : int
        Size of the kernel (must be odd).
    sigma : float
        Standard deviation of the Gaussian distribution.

    Returns:
    --------
    np.ndarray
        2D Gaussian kernel normalized to sum to 1.
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")

    # Create a grid of coordinates
    x = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    xx, yy = np.meshgrid(x, x)

    # Compute Gaussian function
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))

    # Normalize to sum to 1 (preserves image brightness)
    kernel = kernel / np.sum(kernel)

    return kernel


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Perform 2D convolution of an image with a kernel using spatial domain.

    This is a manual implementation of convolution for educational purposes.

    Parameters:
    -----------
    image : np.ndarray
        Input image (grayscale or single channel).
    kernel : np.ndarray
        Convolution kernel.

    Returns:
    --------
    np.ndarray
        Convolved image (same size as input).
    """
    # Get dimensions
    i_height, i_width = image.shape
    k_height, k_width = kernel.shape

    # Calculate padding needed to maintain image size
    pad_height = k_height // 2
    pad_width = k_width // 2

    # Pad the image
    padded = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)),
                    mode='constant', constant_values=0)

    # Prepare output
    output = np.zeros_like(image, dtype=np.float64)

    # Perform convolution
    for i in range(i_height):
        for j in range(i_width):
            # Extract the region of interest
            roi = padded[i:i + k_height, j:j + k_width]
            # Apply kernel
            output[i, j] = np.sum(roi * kernel)

    return output


def spatial_blur(image: np.ndarray, kernel_size: int = 15,
                 sigma: float = 2.0) -> np.ndarray:
    """
    Apply Gaussian blur to an image using spatial domain convolution.

    Parameters:
    -----------
    image : np.ndarray
        Input image (grayscale or RGB).
    kernel_size : int
        Size of the Gaussian kernel (must be odd).
    sigma : float
        Standard deviation of the Gaussian.

    Returns:
    --------
    np.ndarray
        Blurred image with same dimensions as input.
    """
    # Create Gaussian kernel
    kernel = create_gaussian_kernel(kernel_size, sigma)

    # Handle color images
    if len(image.shape) == 3:
        # Process each channel separately
        blurred = np.zeros_like(image, dtype=np.float64)
        for c in range(image.shape[2]):
            blurred[:, :, c] = convolve2d(image[:, :, c].astype(np.float64),
                                          kernel)
        return np.clip(blurred, 0, 255).astype(image.dtype)
    else:
        # Grayscale image
        blurred = convolve2d(image.astype(np.float64), kernel)
        return np.clip(blurred, 0, 255).astype(image.dtype)


def spatial_blur_cv(image: np.ndarray, kernel_size: int = 15,
                    sigma: float = 2.0) -> np.ndarray:
    """
    Apply Gaussian blur using OpenCV's optimized implementation.

    This is provided for comparison and validation of the manual implementation.

    Parameters:
    -----------
    image : np.ndarray
        Input image.
    kernel_size : int
        Size of the Gaussian kernel (must be odd).
    sigma : float
        Standard deviation of the Gaussian.

    Returns:
    --------
    np.ndarray
        Blurred image.
    """
    try:
        import cv2
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    except ImportError:
        return spatial_blur(image, kernel_size, sigma)
