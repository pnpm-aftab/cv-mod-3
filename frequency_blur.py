"""
Frequency Domain Image Blurring Module

This module implements image blurring using frequency domain filtering.
According to the convolution theorem, convolution in the spatial domain
is equivalent to multiplication in the frequency domain.

Convolution Theorem:
    f(x,y) * h(x,y) <=> F(u,v) * H(u,v)

Where:
    * denotes convolution
    <=> denotes Fourier transform pair
    F(u,v) is the FFT of the image
    H(u,v) is the FFT of the kernel
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

    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)

    return kernel


def pad_to_shape(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Pad an image to a target shape for FFT operations.

    Padding is necessary because FFT assumes circular convolution.
    We pad to avoid edge effects (wraparound artifacts).

    Parameters:
    -----------
    image : np.ndarray
        Input image.
    target_shape : Tuple[int, int]
        Target shape (height, width).

    Returns:
    --------
    np.ndarray
        Padded image.
    """
    # Calculate padding amounts
    pad_height = target_shape[0] - image.shape[0]
    pad_width = target_shape[1] - image.shape[1]

    # Pad symmetrically
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode='constant', constant_values=0)

    return padded


def frequency_blur(image: np.ndarray, kernel_size: int = 15,
                   sigma: float = 2.0) -> np.ndarray:
    """
    Apply Gaussian blur to an image using frequency domain filtering.

    Method:
    1. Compute FFT of the image
    2. Create and pad the Gaussian kernel
    3. Compute FFT of the kernel
    4. Multiply in frequency domain
    5. Compute inverse FFT
    6. Crop back to original size

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
    # Get original image shape
    orig_height, orig_width = image.shape[:2]

    # Handle color images
    if len(image.shape) == 3:
        # Process each channel separately
        blurred = np.zeros_like(image, dtype=np.float64)
        for c in range(image.shape[2]):
            blurred[:, :, c] = _frequency_blur_single_channel(
                image[:, :, c].astype(np.float64),
                orig_height, orig_width, kernel_size, sigma
            )
        return np.clip(blurred, 0, 255).astype(image.dtype)
    else:
        # Grayscale image
        blurred = _frequency_blur_single_channel(
            image.astype(np.float64),
            orig_height, orig_width, kernel_size, sigma
        )
        return np.clip(blurred, 0, 255).astype(image.dtype)


def _frequency_blur_single_channel(image: np.ndarray,
                                    orig_height: int, orig_width: int,
                                    kernel_size: int, sigma: float) -> np.ndarray:
    """
    Apply frequency domain blur to a single channel.

    Parameters:
    -----------
    image : np.ndarray
        Single channel image.
    orig_height : int
        Original image height for cropping.
    orig_width : int
        Original image width for cropping.
    kernel_size : int
        Size of the Gaussian kernel.
    sigma : float
        Standard deviation of the Gaussian.

    Returns:
    --------
    np.ndarray
        Blurred single channel image.
    """
    # Size for FFT (same size as image - padded to avoid wraparound)
    # For proper linear convolution via FFT, we need size = image_size + kernel_size - 1
    fft_height = orig_height + kernel_size - 1
    fft_width = orig_width + kernel_size - 1

    # Step 1: Compute FFT of the (padded) image
    padded_image = pad_to_shape(image, (fft_height, fft_width))
    image_fft = np.fft.fft2(padded_image)

    # Step 2: Create and pad the Gaussian kernel
    kernel = create_gaussian_kernel(kernel_size, sigma)
    padded_kernel = np.zeros((fft_height, fft_width))
    padded_kernel[0:kernel_size, 0:kernel_size] = kernel

    # Step 3: Compute FFT of the kernel
    # Shift so that the center of the kernel (kernel_size//2) is at (0,0)
    kernel_shifted = np.roll(padded_kernel, (-(kernel_size // 2), -(kernel_size // 2)), axis=(0, 1))
    kernel_fft = np.fft.fft2(kernel_shifted)

    # Step 4: Multiply in frequency domain
    # This is equivalent to convolution in spatial domain
    filtered_fft = image_fft * kernel_fft

    # Step 5: Compute inverse FFT
    filtered = np.fft.ifft2(filtered_fft)

    # Take real part (imaginary parts should be negligible, result is real)
    filtered = np.real(filtered)

    # Step 6: Crop back to original size
    # The convolution result is centered, so we need to crop appropriately
    start_y = (kernel_size - 1) // 2
    start_x = (kernel_size - 1) // 2
    cropped = filtered[start_y:start_y + orig_height,
                       start_x:start_x + orig_width]

    return cropped


def get_frequency_spectrum(image: np.ndarray) -> np.ndarray:
    """
    Compute the magnitude spectrum of an image for visualization.

    Uses log transformation for better visualization of frequency components.

    Parameters:
    -----------
    image : np.ndarray
        Input image.

    Returns:
    --------
    np.ndarray
        Log magnitude spectrum (shifted so DC component is at center).
    """
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    # Log transform for visualization
    spectrum = np.log(magnitude + 1)
    return spectrum


def get_gaussian_frequency_filter(shape: Tuple[int, int], sigma: float = 2.0) -> np.ndarray:
    """
    Create a Gaussian low-pass filter directly in the frequency domain.

    This is an alternative method that creates the filter in frequency domain
    rather than transforming a spatial kernel.

    Parameters:
    -----------
    shape : Tuple[int, int]
        Shape of the filter (height, width).
    sigma : float
        Standard deviation of the Gaussian.

    Returns:
    --------
    np.ndarray
        Gaussian low-pass filter in frequency domain.
    """
    rows, cols = shape
    center_y, center_x = rows // 2, cols // 2

    # Create coordinate grid
    y = np.arange(rows)
    x = np.arange(cols)
    xx, yy = np.meshgrid(x, y)

    # Compute distance from center (normalized by sigma)
    d_squared = (xx - center_x)**2 + (yy - center_y)**2

    # Gaussian filter in frequency domain
    # Note: sigma in frequency domain is inverse of spatial sigma
    filter_freq = np.exp(-d_squared / (2 * (sigma * sigma)))

    return filter_freq
