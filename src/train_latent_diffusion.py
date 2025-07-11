"""
Training script for latent diffusion super-resolution
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Import our modules
from UNetLite import UNetLite
from process_data import get_data_loaders
from diffusion_loop import (
    beta_schedule, extract_timestep_coefficients, 
    forward_diffusion_sample, get_loss
)

# Try to import diffusers VAE, fallback to simple VAE if not available
try:
    from diffusers import AutoencoderKL
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Diffusers not available, using simple VAE")

class SimpleVAE(nn.Module):
    """Simple VAE if diffusers not available"""
    def __init__(self, in_channels=3, latent_dim=4):
        super().__init__()
        
        # Encoder: 256x256 -> 32x32x4
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),  # 256->128
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),          # 128->64
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),         # 64->32
            nn.ReLU(),
            nn.Conv2d(128, latent_dim, 1),       # 32->32, reduce channels
        )
        
        # Decoder: 32x32x4 -> 256x256
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, 128, 1),       # Keep 32x32, expand channels
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # 32->64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 64->128
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 4, 2, 1), # 128->256
            nn.Tanh()  # Output in [-1, 1] range
        )
    
    def encode(self, x):
        """Encode image to latent space"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent to image"""
        return self.decoder(z)

class LatentDiffusionSuperResolution:
    """Main training class"""
    
    def __init__(self, data_dir, device="cuda", use_pretrained_vae=True, unet_time_emb_dim=256, unet_base_channels=32, **kwargs):
        self.device = device
        self.data_dir = data_dir
        
        # Diffusion parameters
        self.T = 300
        self.betas = beta_schedule(timesteps=self.T)
        self.setup_diffusion_constants()
        
        # Initialize VAE
        self.vae = self.setup_vae(use_pretrained_vae)
        
        # Initialize UNet (input: 8 channels = 4 for noisy HR + 4 for LR condition)
        self.unet = UNetLite(in_channels=8, out_channels=4, time_emb_dim=unet_time_emb_dim, base_channels=unet_base_channels).to(device)
        
        # Setup optimizer
        self.optimizer = Adam(self.unet.parameters(), lr=1e-4)
        
        # Setup data loaders
        self.data_loaders = get_data_loaders(
            data_dir, 
            batch_size=4,  # Start small for testing
            get_train=True, 
            get_val=True, 
            get_test=False,
            augment_train=True
        )
        
        print(f"Initialized on {device}")
        print(f"Train samples: {len(self.data_loaders['train'].dataset)}")
        print(f"Val samples: {len(self.data_loaders['val'].dataset)}")
    
    def setup_vae(self, use_pretrained_vae):
        """Setup VAE"""
        if use_pretrained_vae and DIFFUSERS_AVAILABLE:
            try:
                print("Loading pretrained VAE...")
                vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
                vae = vae.to(self.device)
                vae.eval()
                print("Loaded pretrained VAE")
                return vae
            except Exception as e:
                print(f"Failed to load pretrained VAE: {e}")
                print("Falling back to simple VAE")
        
        print("Using simple VAE")
        vae = SimpleVAE().to(self.device)
        return vae
    
    def setup_diffusion_constants(self):
        """Setup diffusion constants"""
        alphas = 1. - self.betas
        self.alphas_bar = torch.cumprod(alphas, axis=0)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1 - self.alphas_bar)
    
    def encode_images(self, images):
        """Encode images to latent space using VAE"""
        with torch.no_grad():
            if hasattr(self.vae, 'encode'):
                # Diffusers VAE
                latents = self.vae.encode(images).latent_dist.sample()
                latents = latents * 0.18215  # Scaling factor
            else:
                # Simple VAE
                latents = self.vae.encode(images)
        return latents
    
    def decode_latents(self, latents):
        """Decode latents to images using VAE"""
        with torch.no_grad():
            if hasattr(self.vae, 'decode'):
                # Diffusers VAE
                latents = latents / 0.18215
                images = self.vae.decode(latents).sample
            else:
                # Simple VAE
                images = self.vae.decode(latents)
        return images
    
    """
    VAE: (batch, 3, 256, 256) -> (batch, 4, 32, 32) -> (batch, 3, 256, 256)
    8x spatial reduction, 3->4->3 channels
    """
    """
    But then isnt the unet predicting how much noise to remove at once relative to the t, and not sequentially remove the t because it never learns that? 
    """
    def forward_diffusion_sample(self, hr_latent, t):
        """Add noise to HR latent"""
        noise = torch.randn_like(hr_latent)
        sqrt_alphas_bar_t = extract_timestep_coefficients(
            self.sqrt_alphas_bar, t, hr_latent.shape
        )
        sqrt_one_minus_alphas_bar_t = extract_timestep_coefficients(
            self.sqrt_one_minus_alphas_bar, t, hr_latent.shape
        )
        
        # Return noisy latent and the noise
        noisy_latent = (sqrt_alphas_bar_t.to(self.device) * hr_latent + 
                       sqrt_one_minus_alphas_bar_t.to(self.device) * noise.to(self.device))
        return noisy_latent, noise.to(self.device)
    
    def get_loss(self, lr_latent, hr_latent, t):
        """Calculate loss"""
        # Add noise to HR latent
        noisy_hr_latent, noise = self.forward_diffusion_sample(hr_latent, t)
        
        # Concatenate noisy HR and LR latents as input to UNet
        x = torch.cat([noisy_hr_latent, lr_latent], dim=1)
        
        # Predict noise
        # Note, unet is designed to handle batchess
        predicted_noise = self.unet(x, t)
        
        # MSE loss between predicted and actual noise
        return F.mse_loss(predicted_noise, noise)
    
    def train_step(self, lr_images, hr_images):
        """Single training step"""
        batch_size = lr_images.shape[0]
        
        # Resize LR to match HR size before encoding  
        lr_images_resized = F.interpolate(lr_images, size=(256, 256), mode='bicubic', align_corners=False)
        
        # Encode to latent space
        lr_latents = self.encode_images(lr_images_resized)
        hr_latents = self.encode_images(hr_images)
        
        # Random timesteps (generates random ints between the range)
        t = torch.randint(0, self.T, (batch_size,), device=self.device).long()
        
        # Calculate loss
        loss = self.get_loss(lr_latents, hr_latents, t)
        
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self):
        """Validation step"""
        self.unet.eval() #set to eval mode 
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for lr_images, hr_images in self.data_loaders['val']:
                lr_images = lr_images.to(self.device)
                hr_images = hr_images.to(self.device)
                
                # Resize LR to match HR size
                lr_images_resized = F.interpolate(lr_images, size=(256, 256), mode='bicubic', align_corners=False)
                
                batch_size = lr_images.shape[0]
                lr_latents = self.encode_images(lr_images_resized)
                hr_latents = self.encode_images(hr_images)
                
                t = torch.randint(0, self.T, (batch_size,), device=self.device).long()
                loss = self.get_loss(lr_latents, hr_latents, t)
                
                total_loss += loss.item()
                num_batches += 1
        
        self.unet.train() #set back to train mode 
        return total_loss / num_batches if num_batches > 0 else 0
    
    def train(self, epochs=100, save_every=10, validate_every=5):
        """Main training loop"""
        print(f"Starting training for {epochs} epochs...")
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            epoch_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(self.data_loaders['train'], desc=f"Epoch {epoch+1}/{epochs}")
            
            for lr_images, hr_images in progress_bar:
                lr_images = lr_images.to(self.device)
                hr_images = hr_images.to(self.device)
                
                loss = self.train_step(lr_images, hr_images)
                epoch_loss += loss
                num_batches += 1
                
                progress_bar.set_postfix({'loss': f'{loss:.4f}'})
            
            #since recall get_loss() gets the per batch loss 
            avg_train_loss = epoch_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # validation
            if (epoch + 1) % validate_every == 0:
                val_loss = self.validate()
                val_losses.append(val_loss)
                print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")
            
            # Save model
            if (epoch + 1) % save_every == 0:
                self.save_model(f"model_epoch_{epoch+1}.pt")
        
        # Plot training curves
        self.plot_training_curves(train_losses, val_losses, validate_every)
        
        return train_losses, val_losses
    
    def save_model(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'unet_state_dict': self.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'T': self.T,
            'betas': self.betas,
        }
        
        # Also save VAE if it's trainable
        if not hasattr(self.vae, 'encode') or not DIFFUSERS_AVAILABLE:
            checkpoint['vae_state_dict'] = self.vae.state_dict()
        
        torch.save(checkpoint, filename)
        print(f"Model saved to {filename}")
    
    def plot_training_curves(self, train_losses, val_losses, validate_every):
        """Plot training and validation curves"""
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        
        if val_losses:
            plt.subplot(1, 2, 2)
            val_epochs = list(range(validate_every-1, len(train_losses), validate_every))
            plt.plot(val_epochs, val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Validation Loss')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.show()

def main():
    """Main function to run training"""
    
    # Configuration
    data_dir = input("Enter path to your processed data directory: ").strip()
    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        return
    
    # Choose UNet size
    use_large_unet = input("Use larger UNet? (y/n, default=n): ").strip().lower()
    unet_base_channels = 64 if use_large_unet == 'y' else 32
    unet_size_name = "large" if use_large_unet == 'y' else "standard"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"UNet size: {unet_size_name} (base_channels={unet_base_channels})")
    
    # Initialize model
    model = LatentDiffusionSuperResolution(
        data_dir=data_dir,
        device=device,
        use_pretrained_vae=True,
        unet_base_channels=unet_base_channels
    )
    
    
    # Train
    train_losses, val_losses = model.train(
        epochs=50,
        save_every=10,
        validate_every=5
    )
    
    print("Training complete!")

if __name__ == "__main__":
    main()
