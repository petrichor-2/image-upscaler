"""
Inference script for trained model
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

from UNetLite import UNetLite
from process_data import get_data_loaders
from train_latent_diffusion import LatentDiffusionSuperResolution

def compute_psnr(img1, img2):
    """Compute PSNR between two images (expects uint8 numpy arrays)"""
    mse = ((img1 - img2) ** 2).mean()
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def compute_ssim(img1, img2):
    """Compute SSIM between two images (expects uint8 numpy arrays of HxWxC)"""
    return ssim(img1, img2, channel_axis=2, data_range=255)

def tensor_to_uint8_numpy(tensor):
    """Convert tensor to uint8 numpy for metric calculation"""
    tensor = tensor.cpu()
    tensor = (tensor + 1) / 2  # Denormalize from [-1,1] to [0,1]
    tensor = torch.clamp(tensor, 0, 1)
    tensor = tensor.permute(0, 2, 3, 1)  # BCHW -> BHWC
    numpy_img = (tensor.numpy() * 255).astype(np.uint8)
    return numpy_img

def reverse_diffusion_sample(model, lr_latent, T, betas, device):
    """Reverse diffusion to generate HR from LR"""
    # Start from noise
    hr_latent = torch.randn_like(lr_latent)
    
    # Constants
    alphas = 1. - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_one_minus_alphas_bar = torch.sqrt(1 - alphas_bar)
    
    # Reverse process
    for t in tqdm(reversed(range(T)), desc="Generating"):
        # Create timestep tensor for batch since unet expects [batch_size] timesteps, one per image
        t_tensor = torch.full((lr_latent.shape[0],), t, device=device, dtype=torch.long)
        
        # Concatenate noisy HR and LR for unet input
        x = torch.cat([hr_latent, lr_latent], dim=1)
        
        # Predict noise
        predicted_noise = model(x, t_tensor)
        
        # DDPM reverse diffusion formula: removes noise incrementally, not all at once
        # UNet predicts total accumulated noise, but we only remove the incremental portion
        # for this timestep to maintain mathematical stability and generation quality
        hr_latent = sqrt_recip_alphas[t] * (hr_latent - betas[t] * predicted_noise / sqrt_one_minus_alphas_bar[t])
        
        # Add noise (except for last step)
        # The og paper does this, it seems counterintuituve, because 
        # whole point of reverse diffusion was to remove noise, so why do 
        # we add some noise again? Stochastic mathematical stuff, but basically 
        # leads to better results
        if t > 0:
            noise = torch.randn_like(hr_latent)
            hr_latent = hr_latent + torch.sqrt(betas[t]) * noise
    
    return hr_latent

def generate_super_resolution(model_path, data_dir, num_samples=4, device="cuda"):
    """
    Generate super-resolution images using trained model
    """
    print(f"Loading model from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    T = checkpoint['T']
    betas = checkpoint['betas']
    
    # Initialize model
    ldsr = LatentDiffusionSuperResolution(data_dir, device, use_pretrained_vae=True)
    ldsr.unet.load_state_dict(checkpoint['unet_state_dict'])
    ldsr.unet.eval()
    
    # Load VAE if saved (ie if we are not using pretrained vae since it doesnt load)
    if 'vae_state_dict' in checkpoint:
        ldsr.vae.load_state_dict(checkpoint['vae_state_dict'])
    
    # Get some test samples
    test_loader = get_data_loaders(data_dir, batch_size=num_samples, get_train=False, get_val=False, get_test=True)['test']
    lr_images, hr_images = next(iter(test_loader))
    
    lr_images = lr_images.to(device)
    hr_images = hr_images.to(device)
    
    print("Generating super-resolution images...")
    
    with torch.no_grad():
        # Encode LR images
        lr_latents = ldsr.encode_images(lr_images)
        
        # Generate HR latents using reverse diffusion
        generated_hr_latents = reverse_diffusion_sample(
            ldsr.unet, lr_latents, T, betas, device
        )
        
        # Decode to images
        generated_hr_images = ldsr.decode_latents(generated_hr_latents)
        
        # Also get bicubic upsampling for comparison
        lr_upsampled = F.interpolate(lr_images, size=(256, 256), mode='bicubic', align_corners=False)
    
    # Visualize results with metrics
    visualize_results_with_metrics(lr_images, hr_images, generated_hr_images, lr_upsampled, num_samples)

def visualize_results_with_metrics(lr_images, hr_images, generated_images, bicubic_images, num_samples):
    """
    Visualize super-resolution results with PSNR and SSIM metrics
    """
    def tensor_to_numpy(tensor):
        """Convert tensor to numpy for visualization"""
        tensor = tensor.cpu()
        tensor = (tensor + 1) / 2  # Denormalize from [-1,1] to [0,1]
        tensor = torch.clamp(tensor, 0, 1)
        return tensor.permute(0, 2, 3, 1).numpy()
    
    # Convert tensors to numpy arrays for visualization
    lr_np = tensor_to_numpy(lr_images)
    hr_np = tensor_to_numpy(hr_images)
    generated_np = tensor_to_numpy(generated_images)
    bicubic_np = tensor_to_numpy(bicubic_images)
    
    # Convert to uint8 for metric calculations
    hr_uint8 = tensor_to_uint8_numpy(hr_images)
    generated_uint8 = tensor_to_uint8_numpy(generated_images)
    bicubic_uint8 = tensor_to_uint8_numpy(bicubic_images)
    
    fig, axes = plt.subplots(4, num_samples, figsize=(4*num_samples, 16))
    
    # Track metrics for averaging
    bicubic_psnr_values = []
    bicubic_ssim_values = []
    diffusion_psnr_values = []
    diffusion_ssim_values = []
    
    for i in range(num_samples):
        # Calculate metrics for this sample
        bicubic_psnr = compute_psnr(bicubic_uint8[i], hr_uint8[i])
        bicubic_ssim = compute_ssim(bicubic_uint8[i], hr_uint8[i])
        diffusion_psnr = compute_psnr(generated_uint8[i], hr_uint8[i])
        diffusion_ssim = compute_ssim(generated_uint8[i], hr_uint8[i])
        
        # Store for averaging
        bicubic_psnr_values.append(bicubic_psnr)
        bicubic_ssim_values.append(bicubic_ssim)
        diffusion_psnr_values.append(diffusion_psnr)
        diffusion_ssim_values.append(diffusion_ssim)
        
        # Print metrics for this sample
        print(f"\nSample {i+1}:")
        print(f"  Bicubic  - PSNR: {bicubic_psnr:.2f} dB, SSIM: {bicubic_ssim:.4f}")
        print(f"  Diffusion - PSNR: {diffusion_psnr:.2f} dB, SSIM: {diffusion_ssim:.4f}")
        
        # LR image
        axes[0, i].imshow(lr_np[i], cmap='gray' if lr_np[i].shape[-1] == 1 else None)
        axes[0, i].set_title(f'LR Input (64x64)')
        axes[0, i].axis('off')
        
        # Bicubic upsampling with metrics
        axes[1, i].imshow(bicubic_np[i], cmap='gray' if bicubic_np[i].shape[-1] == 1 else None)
        axes[1, i].set_title(f'Bicubic\nPSNR: {bicubic_psnr:.2f} dB\nSSIM: {bicubic_ssim:.3f}')
        axes[1, i].axis('off')
        
        # Generated HR with metrics
        axes[2, i].imshow(generated_np[i], cmap='gray' if generated_np[i].shape[-1] == 1 else None)
        axes[2, i].set_title(f'Diffusion\nPSNR: {diffusion_psnr:.2f} dB\nSSIM: {diffusion_ssim:.3f}')
        axes[2, i].axis('off')
        
        # Ground truth HR
        axes[3, i].imshow(hr_np[i], cmap='gray' if hr_np[i].shape[-1] == 1 else None)
        axes[3, i].set_title(f'Ground Truth HR (256x256)')
        axes[3, i].axis('off')
    
    # Print average metrics
    avg_bicubic_psnr = np.mean(bicubic_psnr_values)
    avg_bicubic_ssim = np.mean(bicubic_ssim_values)
    avg_diffusion_psnr = np.mean(diffusion_psnr_values)
    avg_diffusion_ssim = np.mean(diffusion_ssim_values)
    
    print(f"\n{'='*50}")
    print("AVERAGE METRICS:")
    print(f"  Bicubic  - PSNR: {avg_bicubic_psnr:.2f} dB, SSIM: {avg_bicubic_ssim:.4f}")
    print(f"  Diffusion - PSNR: {avg_diffusion_psnr:.2f} dB, SSIM: {avg_diffusion_ssim:.4f}")
    print(f"{'='*50}")
    
    plt.tight_layout()
    plt.savefig('super_resolution_results.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main inference function"""
    model_path = input("Enter path to trained model (.pt file): ").strip()
    data_dir = input("Enter path to data directory: ").strip()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    generate_super_resolution(model_path, data_dir, num_samples=4, device=device)

if __name__ == "__main__":
    main()
