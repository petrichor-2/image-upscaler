"""
Inference script for trained model
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

from UNetLite import UNetLite
from process_data import get_data_loaders
from train_latent_diffusion import LatentDiffusionSuperResolution
from PIL import Image
import time

from run_trt import TRTUNet

def calculate_psnr(img1, img2, max_value=1.0):
    """
    Calculate PSNR between two images
    Args:
        img1, img2: Tensor images in range [0, 1]
        max_value: Maximum possible pixel value (1.0 for normalized images)
    Returns:
        PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100  # Perfect match
    psnr = 20 * torch.log10(max_value / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(img1, img2):
    """
    Calculate SSIM between two images
    Args:
        img1, img2: Tensor images in range [0, 1] with shape (C, H, W)
    Returns:
        SSIM value
    """
    # Convert tensors to numpy arrays
    img1_np = img1.cpu().numpy()
    img2_np = img2.cpu().numpy()
    
    # Convert from (C, H, W) to (H, W, C) for skimage
    if img1_np.shape[0] == 3:  # RGB
        img1_np = np.transpose(img1_np, (1, 2, 0))
        img2_np = np.transpose(img2_np, (1, 2, 0))
        # Use multichannel=True for RGB images
        return ssim(img1_np, img2_np, multichannel=True, data_range=1.0)
    else:  # Grayscale
        img1_np = img1_np.squeeze()
        img2_np = img2_np.squeeze()
        return ssim(img1_np, img2_np, data_range=1.0)

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

def reverse_diffusion_sample_trt(model_path, lr_latent, T, betas, device):
    """Reverse diffusion to generate HR from LR using the TRT model"""
    model = TRTUNet(engine_path=model_path)  # Load TRT model

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
        predicted_noise = model.infer(x, t_tensor, device)
        
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

def generate_super_resolution(model_path, data_dir, trt_path="None", use_trt=False, num_samples=4, device="cuda"):
    """
    Generate super-resolution images using trained model
    """
    print(f"Loading model from {model_path}")
    

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    T = checkpoint['T']
    betas = checkpoint['betas']

    # Detect model size from checkpoint
    unet_base_channels = 32  # default
    if 'unet_state_dict' in checkpoint:
        # Check the first layer to determine model size
        first_layer_weight = checkpoint['unet_state_dict']['downBlock1.conv.conv1.weight']
        if first_layer_weight.shape[0] == 64:  # Large model
            unet_base_channels = 64
            print("Detected large UNet model (base_channels=64)")
        else:
            print("Detected standard UNet model (base_channels=32)")

        # Initialize model with correct size
        ldsr = LatentDiffusionSuperResolution(
            data_dir, 
            device, 
            use_pretrained_vae=True,
            unet_base_channels=unet_base_channels
        )
        ldsr.unet.load_state_dict(checkpoint['unet_state_dict'])
        ldsr.unet.eval()
    
    # Load VAE if saved (ie if we are not using pretrained vae since it doesnt load)
    if 'vae_state_dict' in checkpoint:
        ldsr.vae.load_state_dict(checkpoint['vae_state_dict'])
    
    if data_dir == "None":
        # Load specific LR and HR images
        hr_image_path = "/home/nazmus/Desktop/EdgeDiff SR/image-upscaler/src/Data/HR_256/00000001_000 (1).png"
        lr_image_path = "/home/nazmus/Desktop/EdgeDiff SR/image-upscaler/src/Data/LR_64/00000001_000.png"

        # Open and convert to RGB
        lr_image = Image.open(lr_image_path).convert("RGB")
        hr_image = Image.open(hr_image_path).convert("RGB")

        # Transform to tensor and normalize to [-1, 1]
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),  # [0,1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [0,1] -> [-1,1]
        ])

        lr_images = transform(lr_image).unsqueeze(0).to(device)
        hr_images = transform(hr_image).unsqueeze(0).to(device)
  
    else:
        # Get some test samples
        test_loader = get_data_loaders(data_dir, batch_size=num_samples, get_train=False, get_val=False, get_test=True)['test']
        lr_images, hr_images = next(iter(test_loader))
        
        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)
    
    print("Generating super-resolution images...")
    
    with torch.no_grad():
        # Encode LR images
        # 1) upsample LR (64x64) to 256x256 to match training
        lr_images_resized = F.interpolate(lr_images, size=(256, 256), mode='bicubic', align_corners=False)

        # 2) encode to latents -> (B,4,32,32)
        lr_latents = ldsr.encode_images(lr_images_resized)

        # (optional) sanity check
        assert lr_latents.shape[-2:] == (32, 32), f"Expected 32x32 latents, got {lr_latents.shape[-2:]}"


        

        if use_trt:
            start_time = time.time()
            # Generate HR latents using reverse diffusion with TRT model
            generated_hr_latents = reverse_diffusion_sample_trt(
                trt_path, lr_latents, T, betas, device
            )
            end_time = time.time()
            print(f"TRT UNet: Generated HR latents in {end_time - start_time:.2f} seconds")
        else:
            start_time = time.time()
            # Generate HR latents using reverse diffusion
            generated_hr_latents = reverse_diffusion_sample(
                ldsr.unet, lr_latents, T, betas, device
            )
            end_time = time.time()
            print(f"Standard UNet: Generated HR latents in {end_time - start_time:.2f} seconds")
        
        
        # Decode to images
        generated_hr_images = ldsr.decode_latents(generated_hr_latents)
        
        # Debug: Print shapes to identify issues
        print(f"Debug - LR images shape: {lr_images.shape}")
        print(f"Debug - HR images shape: {hr_images.shape}")
        print(f"Debug - Generated HR images shape: {generated_hr_images.shape}")
        
        # Fix size mismatch if generated images are wrong size
        if generated_hr_images.shape[-2:] != hr_images.shape[-2:]:
            print(f"Size mismatch detected! Resizing generated images from {generated_hr_images.shape[-2:]} to {hr_images.shape[-2:]}")
            generated_hr_images = F.interpolate(
                generated_hr_images, 
                size=hr_images.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
            print(f"After resize - Generated HR images shape: {generated_hr_images.shape}")
        
        # Also get bicubic upsampling for comparison
        lr_upsampled = F.interpolate(lr_images, size=(256, 256), mode='bicubic', align_corners=False)
    
    
    elapsed = end_time - start_time
    print(f"Time taken to generate {num_samples} image(s): {elapsed:.2f} seconds")

    # Visualize results
    visualize_results(lr_images, hr_images, generated_hr_images, lr_upsampled, num_samples)

def visualize_results(lr_images, hr_images, generated_images, bicubic_images, num_samples):
    """
    Visualize super-resolution results with PSNR calculations
    """
    def tensor_to_numpy(tensor):
        """Convert tensor to numpy for visualization"""
        tensor = tensor.cpu()
        tensor = (tensor + 1) / 2  # Denormalize from [-1,1] to [0,1]
        tensor = torch.clamp(tensor, 0, 1)
        return tensor.permute(0, 2, 3, 1).numpy()
    
    # Convert for visualization
    lr_np = tensor_to_numpy(lr_images)
    hr_np = tensor_to_numpy(hr_images)
    generated_np = tensor_to_numpy(generated_images)
    bicubic_np = tensor_to_numpy(bicubic_images)
    
    # Convert for PSNR calculation (keep as tensors, normalized to [0,1])
    hr_norm = torch.clamp((hr_images + 1) / 2, 0, 1)
    gen_norm = torch.clamp((generated_images + 1) / 2, 0, 1)
    bicubic_norm = torch.clamp((bicubic_images + 1) / 2, 0, 1)
    
    fig, axes = plt.subplots(4, num_samples, figsize=(4*num_samples, 16))
    if num_samples == 1:
        axes = axes.reshape(4, 1)
    
    # Calculate and print PSNR and SSIM values
    print("\n=== PSNR and SSIM Results ===")
    bicubic_psnrs = []
    diffusion_psnrs = []
    bicubic_ssims = []
    diffusion_ssims = []
    
    for i in range(num_samples):
        # Calculate PSNR for each image
        bicubic_psnr = calculate_psnr(bicubic_norm[i], hr_norm[i])
        diffusion_psnr = calculate_psnr(gen_norm[i], hr_norm[i])
        
        # Calculate SSIM for each image
        bicubic_ssim = calculate_ssim(bicubic_norm[i], hr_norm[i])
        diffusion_ssim = calculate_ssim(gen_norm[i], hr_norm[i])
        
        bicubic_psnrs.append(bicubic_psnr)
        diffusion_psnrs.append(diffusion_psnr)
        bicubic_ssims.append(bicubic_ssim)
        diffusion_ssims.append(diffusion_ssim)
        
        print(f"Image {i+1}: Bicubic PSNR = {bicubic_psnr:.2f} dB, SSIM = {bicubic_ssim:.4f}")
        print(f"         Diffusion PSNR = {diffusion_psnr:.2f} dB, SSIM = {diffusion_ssim:.4f}")
        
        # Plot images with PSNR and SSIM in titles
        axes[0, i].imshow(lr_np[i], cmap='gray' if lr_np[i].shape[-1] == 1 else None)
        axes[0, i].set_title(f'LR Input (64x64)')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(bicubic_np[i], cmap='gray' if bicubic_np[i].shape[-1] == 1 else None)
        axes[1, i].set_title(f'Bicubic Upsampling\nPSNR: {bicubic_psnr:.2f} dB, SSIM: {bicubic_ssim:.3f}')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(generated_np[i], cmap='gray' if generated_np[i].shape[-1] == 1 else None)
        axes[2, i].set_title(f'Generated HR (Diffusion)\nPSNR: {diffusion_psnr:.2f} dB, SSIM: {diffusion_ssim:.3f}')
        axes[2, i].axis('off')
        
        axes[3, i].imshow(hr_np[i], cmap='gray' if hr_np[i].shape[-1] == 1 else None)
        axes[3, i].set_title(f'Ground Truth HR (256x256)')
        axes[3, i].axis('off')
    
    # Print average PSNR and SSIM
    avg_bicubic_psnr = np.mean(bicubic_psnrs)
    avg_diffusion_psnr = np.mean(diffusion_psnrs)
    avg_bicubic_ssim = np.mean(bicubic_ssims)
    avg_diffusion_ssim = np.mean(diffusion_ssims)
    
    print(f"\nAverage Results:")
    print(f"Bicubic:   PSNR = {avg_bicubic_psnr:.2f} dB, SSIM = {avg_bicubic_ssim:.4f}")
    print(f"Diffusion: PSNR = {avg_diffusion_psnr:.2f} dB, SSIM = {avg_diffusion_ssim:.4f}")
    
    plt.tight_layout()
    plt.savefig('super_resolution_results.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main inference function"""
    model_path = input("Enter path to trained model (.pt file): ").strip()
    data_dir = input("Enter path to data directory: ").strip()
    trt_path = input("Enter path to TensorRT engine (.trt file) or 'None' if not using TRT: ").strip()
    use_trt = trt_path.lower() != "none"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    generate_super_resolution(model_path, data_dir, trt_path, use_trt, num_samples=1, device=device)

if __name__ == "__main__":
    main()
