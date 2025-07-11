import os
import math
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2
import matplotlib.pyplot as plt

def apply_baseline_model(lr_image):
    """Apply algorithmic baseline model to the low-resolution image."""
    return cv2.resize(lr_image, (256, 256), interpolation = cv2.INTER_CUBIC)

def compute_psnr(img1, img2):
    """Compute PSNR between two images."""

    mse = ((img1 - img2) ** 2).mean()
    if mse == 0:
        return float('inf')  # No difference
    return 20 * math.log10(255.0 / mse)

def compute_ssim(img1, img2):
    """Compute SSIM between two images, expects numpy arrays of HxWxC."""
    return ssim(img1, img2, channel_axis=2, data_range=255)

def evaluate_baseline(data_dir='data/chestxrays', num_images=4):
    """Evaluate the baseline model on a small dataset. Assumes images are processed."""
    
    # Load the dataset
    hr_dir = os.path.join(data_dir, "HR_256")
    lr_dir = os.path.join(data_dir, "LR_64")
    filenames = sorted(os.listdir(hr_dir)[:num_images])

    psnr_values = []
    ssim_values = []

    # Prepare a single figure for all images
    fig, axes = plt.subplots(num_images, 3, figsize=(8, 1.5 * num_images))
    fig.suptitle("Baseline Model Results", fontsize=14)

    for i, filename in enumerate(filenames):
        hr_path = os.path.join(hr_dir, filename)
        lr_path = os.path.join(lr_dir, filename)
        if not os.path.exists(hr_path) or not os.path.exists(lr_path):
            continue

        hr_image = plt.imread(hr_path)
        lr_image = plt.imread(lr_path)

        # If images are grayscale, convert to 3-channel
        if hr_image.ndim == 2:
            hr_image = np.stack([hr_image]*3, axis=-1)
        if lr_image.ndim == 2:
            lr_image = np.stack([lr_image]*3, axis=-1)

        # Ensure images are uint8
        if hr_image.dtype != np.uint8:
            hr_image = (hr_image * 255).astype(np.uint8)
        if lr_image.dtype != np.uint8:
            lr_image = (lr_image * 255).astype(np.uint8)

        # Apply the baseline model
        hr_baseline = apply_baseline_model(lr_image)

        # Compute PSNR and SSIM
        psnr_value = compute_psnr(hr_baseline, hr_image)
        ssim_value = compute_ssim(hr_baseline, hr_image)

        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)

        # Show images in the same figure, first 5 only
        axes[i, 0].imshow(lr_image, vmin=0, vmax=255, cmap='gray')
        axes[i, 0].set_title('Low Resolution')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(hr_image, vmin=0, vmax=255, cmap='gray')
        axes[i, 1].set_title('High Resolution')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(hr_baseline, vmin=0, vmax=255, cmap='gray')
        axes[i, 2].set_title(f'Baseline HR\nPSNR: {psnr_value:.2f} SSIM: {ssim_value:.3f}')
        axes[i, 2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")

if __name__ == "__main__":
    evaluate_baseline()