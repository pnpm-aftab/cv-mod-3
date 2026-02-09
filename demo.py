#!/usr/bin/env python3
"""
Image Blurring Demo: Spatial Domain vs Frequency Domain

This script demonstrates that blurring an image using spatial domain convolution
produces the same result as frequency domain filtering, according to the
convolution theorem.

Usage:
    CHANGE THE IMAGE PATH IN THE main() FUNCTION TO YOUR IMAGE FILE
    Then run:
        python demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Try importing PIL/Pillow, fall back to creating a test image if not available
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL/Pillow not found. Will create a synthetic test image.")

# Import our blur modules
from spatial_blur import spatial_blur, create_gaussian_kernel
from frequency_blur import frequency_blur, get_frequency_spectrum


def create_test_image(size: tuple = (256, 256)) -> np.ndarray:
    """
    Create a synthetic test image with various patterns for blurring.

    Parameters:
    -----------
    size : tuple
        Size of the image (height, width).

    Returns:
    --------
    np.ndarray
        Test image as numpy array.
    """
    height, width = size

    # Create a blank image
    image = np.zeros((height, width), dtype=np.uint8)

    # Add various patterns for testing blur
    # 1. Checkerboard pattern
    square_size = 32
    for i in range(0, height, square_size):
        for j in range(0, width, square_size):
            if (i // square_size + j // square_size) % 2 == 0:
                image[i:i+square_size, j:j+square_size] = 255

    # 2. Add a white square
    image[80:160, 80:160] = 255

    # 3. Add a black circle
    center_y, center_x = height // 2, width // 2
    y, x = np.ogrid[:height, :width]
    mask = (x - center_x)**2 + (y - center_y)**2 <= 30**2
    image[mask] = 0

    # 4. Add some high-frequency noise lines
    image[170:200, :] = np.random.randint(100, 200, (30, width), dtype=np.uint8)

    return image


def load_image(image_path: str, grayscale: bool = True) -> np.ndarray:
    """
    Load an image from file.

    Parameters:
    -----------
    image_path : str
        Path to the image file.
    grayscale : bool
        If True, convert to grayscale.

    Returns:
    --------
    np.ndarray
        Image as numpy array.
    """
    if not HAS_PIL:
        print("PIL not available, creating test image instead.")
        return create_test_image()

    img = Image.open(image_path)

    if grayscale:
        img = img.convert('L')

    return np.array(img)


def visualize_results(original: np.ndarray, spatial_result: np.ndarray,
                     frequency_result: np.ndarray, difference: np.ndarray,
                     kernel: np.ndarray, save_path: str = None):
    """
    Create a comprehensive visualization of the blur comparison.

    Parameters:
    -----------
    original : np.ndarray
        Original image.
    spatial_result : np.ndarray
        Result from spatial domain blur.
    frequency_result : np.ndarray
        Result from frequency domain blur.
    difference : np.ndarray
        Difference between spatial and frequency results.
    kernel : np.ndarray
        Gaussian kernel used.
    save_path : str, optional
        Path to save the figure.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Spatial Domain vs Frequency Domain Image Blurring',
                 fontsize=16, fontweight='bold')

    # Row 1: Images
    # Original
    im0 = axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    # Spatial blur result
    im1 = axes[0, 1].imshow(spatial_result, cmap='gray')
    axes[0, 1].set_title('Spatial Domain Blur\n(Convolution)')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # Frequency blur result
    im2 = axes[0, 2].imshow(frequency_result, cmap='gray')
    axes[0, 2].set_title('Frequency Domain Blur\n(FFT Multiplication)')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    # Row 2: Analysis
    # Difference image
    im3 = axes[1, 0].imshow(difference, cmap='seismic', vmin=-1, vmax=1)
    axes[1, 0].set_title(f'Difference\n(Min: {difference.min():.2e}, Max: {difference.max():.2e})')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    # Gaussian kernel
    im4 = axes[1, 1].imshow(kernel, cmap='viridis')
    axes[1, 1].set_title('Gaussian Kernel')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

    # Histogram of differences
    axes[1, 2].hist(difference.flatten(), bins=50, edgecolor='black')
    axes[1, 2].set_title('Distribution of Differences')
    axes[1, 2].set_xlabel('Pixel Value Difference')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def visualize_frequency_domain(image: np.ndarray, kernel: np.ndarray,
                               save_path: str = None):
    """
    Visualize the frequency domain representation.

    Parameters:
    -----------
    image : np.ndarray
        Original image.
    kernel : np.ndarray
        Gaussian kernel.
    save_path : str, optional
        Path to save the figure.
    """
    # Compute frequency spectra
    image_spectrum = get_frequency_spectrum(image.astype(np.float64))

    # Compute kernel spectrum
    kernel_padded = np.zeros_like(image, dtype=np.float64)
    k_height, k_width = kernel.shape
    pad_top = (image.shape[0] - k_height) // 2
    pad_left = (image.shape[1] - k_width) // 2
    kernel_padded[pad_top:pad_top+k_height, pad_left:pad_left+k_width] = kernel

    kernel_spectrum = get_frequency_spectrum(kernel_padded)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Frequency Domain Analysis', fontsize=16, fontweight='bold')

    # Image spectrum
    im0 = axes[0].imshow(image_spectrum, cmap='gray')
    axes[0].set_title('Image Frequency Spectrum\n(log magnitude)')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # Kernel spectrum
    im1 = axes[1].imshow(kernel_spectrum, cmap='gray')
    axes[1].set_title('Gaussian Kernel Frequency Spectrum\n(log magnitude)')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Product (filtered spectrum)
    combined = image_spectrum * kernel_spectrum / kernel_spectrum.max()
    im2 = axes[2].imshow(combined, cmap='gray')
    axes[2].set_title('Filtered Spectrum\n(Image x Kernel)')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Frequency domain figure saved to: {save_path}")

    plt.show()


def main():
    # Configuration
    IMAGE_PATH = "4.1.05.tiff"
    KERNEL_SIZE = 15
    SIGMA = 2.0
    OUTPUT_DIR = "output"

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load image
    print(f"Loading image from: {IMAGE_PATH}")
    image = load_image(IMAGE_PATH, grayscale=True)

    print(f"Image shape: {image.shape}")
    print(f"Kernel size: {KERNEL_SIZE}, Sigma: {SIGMA}")

    # Create Gaussian kernel
    kernel = create_gaussian_kernel(KERNEL_SIZE, SIGMA)

    # Apply spatial domain blur
    print("\nApplying spatial domain blur (convolution)...")
    spatial_blurred = spatial_blur(image, KERNEL_SIZE, SIGMA)

    # Apply frequency domain blur
    print("Applying frequency domain blur (FFT multiplication)...")
    frequency_blurred = frequency_blur(image, KERNEL_SIZE, SIGMA)

    # Compute difference
    difference = spatial_blurred.astype(np.float64) - frequency_blurred.astype(np.float64)

    # Calculate statistics
    max_diff = np.max(np.abs(difference))
    mean_diff = np.mean(np.abs(difference))
    rmse = np.sqrt(np.mean(difference ** 2))

    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    print(f"Maximum absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference:    {mean_diff:.6e}")
    print(f"Root Mean Square Error:      {rmse:.6e}")
    print("="*60)

    # The small numerical differences are due to floating-point precision
    # and FFT boundary handling
    if max_diff < 1.0:
        print("\nSUCCESS: Both methods produce essentially the same result!")
        print("(Small differences are due to floating-point precision)")
    else:
        print("\nNote: Larger differences may be due to boundary handling.")

    # Save individual results
    output_dir = Path(OUTPUT_DIR)

    # Save numerical results for the report
    stats_path = output_dir / "comparison_stats.txt"
    with stats_path.open("w", encoding="utf-8") as stats_file:
        stats_file.write("Spatial vs Frequency Blur Comparison\n")
        stats_file.write(f"Image: {IMAGE_PATH}\n")
        stats_file.write(f"Kernel size: {KERNEL_SIZE}\n")
        stats_file.write(f"Sigma: {SIGMA}\n")
        stats_file.write(f"Maximum absolute difference: {max_diff:.6e}\n")
        stats_file.write(f"Mean absolute difference:    {mean_diff:.6e}\n")
        stats_file.write(f"Root Mean Square Error:      {rmse:.6e}\n")

    # Save as PNG
    from PIL import Image as PILImage
    PILImage.fromarray(image).save(output_dir / "original.png")
    PILImage.fromarray(spatial_blurred.astype(np.uint8)).save(output_dir / "spatial_blurred.png")
    PILImage.fromarray(frequency_blurred.astype(np.uint8)).save(output_dir / "frequency_blurred.png")

    # Save difference image (scaled for visibility)
    diff_scaled = np.abs(difference)
    if diff_scaled.max() > 0:
        diff_scaled = (diff_scaled / diff_scaled.max() * 255)
    else:
        diff_scaled = np.zeros_like(diff_scaled)
    
    PILImage.fromarray(diff_scaled.astype(np.uint8)).save(output_dir / "difference.png")

    print(f"\nResults saved to: {output_dir}")
    print(f"Comparison stats saved to: {stats_path}")

    # Visualize
    print("\nDisplaying comparison...")
    visualize_results(
        image, spatial_blurred, frequency_blurred,
        difference, kernel,
        save_path=str(output_dir / "comparison.png")
    )

    # Frequency domain visualization
    print("\nDisplaying frequency domain analysis...")
    visualize_frequency_domain(
        image, kernel,
        save_path=str(output_dir / "frequency_analysis.png")
    )


if __name__ == "__main__":
    main()
