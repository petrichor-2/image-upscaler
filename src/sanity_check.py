"""
Sanity check script: Overfit on tiny dataset to verify pipeline works
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from torch.utils.data import DataLoader

# Import existing modules
from train_latent_diffusion import LatentDiffusionSuperResolution
from process_data import get_data_loaders
from pakhi_data import PairDataset

class TinyDataset(PairDataset):
    """Dataset limited to small number of images for overfitting test"""
    def __init__(self, data_dir, max_images=10, **kwargs):
        super().__init__(data_dir, **kwargs)
        self.filenames = self.filenames[:max_images]
        print(f"Sanity check: Using only {len(self.filenames)} images")

def create_tiny_dataloaders(data_dir, tiny_size=10, batch_size=4):
    """Create dataloaders with small dataset"""
    from pakhi_data import generate_downsampled_pairs
    
    generate_downsampled_pairs(data_dir)
    
    datasets = {
        'train': TinyDataset(data_dir, max_images=tiny_size, mode='train', augment=False)
    }
    
    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=False)
    }
    
    return dataloaders

class SanityChecker:
    
    def __init__(self, data_dir, device="cuda", tiny_size=10):
        self.device = device
        self.data_dir = data_dir
        self.tiny_size = tiny_size
        
        print("Initializing LatentDiffusionSuperResolution...")
        self.model = LatentDiffusionSuperResolution(
            data_dir=data_dir,
            device=device,
            use_pretrained_vae=True,
            unet_time_emb_dim=512  # Larger time embedding dimension for bigger UNet
        )
        
        # Replace UNet with larger version
        from UNetLite import UNetLite
        self.model.unet = UNetLite(in_channels=8, out_channels=4, time_emb_dim=512, base_channels=64).to(device)
        self.model.optimizer = torch.optim.Adam(self.model.unet.parameters(), lr=1e-3)
        
        self.model.data_loaders = create_tiny_dataloaders(
            data_dir, 
            tiny_size=tiny_size, 
            batch_size=min(4, tiny_size)
        )
        
    
    def run_sanity_check(self, epochs=500, plot_every=50):
        """Run sanity check training"""
        print(f"\n=== SANITY CHECK: Overfitting on tiny dataset ===")
        print(f"Train samples: {len(self.model.data_loaders['train'].dataset)}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {self.model.optimizer.param_groups[0]['lr']}")
        
        train_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            # Train on same batch multiple times per epoch for faster overfitting
            for repeat in range(3):  # Reduced from 5 to speed up with more epochs
                for lr_images, hr_images in self.model.data_loaders['train']:
                    lr_images = lr_images.to(self.device)
                    hr_images = hr_images.to(self.device)
                    
                    loss = self.model.train_step(lr_images, hr_images)
                    epoch_loss += loss
                    num_batches += 1
            
            avg_train_loss = epoch_loss / num_batches
            train_losses.append(avg_train_loss)
            
            if epoch % plot_every == 0 or epoch < 10:
                print(f"Epoch {epoch:3d}: Train = {avg_train_loss:.6f}")
        
        print(f"\nFinal train loss: {train_losses[-1]:.6f}")
        print(f"Loss reduction: {train_losses[0]:.6f} -> {train_losses[-1]:.6f}")
        
        self.plot_training_curve(train_losses)
        self.test_inference_with_existing_model()
        
        return train_losses
    
    def plot_training_curve(self, train_losses):
        """Plot training curve"""
        plt.figure(figsize=(10, 6))
        
        plt.plot(train_losses, 'b-', linewidth=2, label='Train Loss')
        plt.title('Training Loss Over 500 Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('sanity_check_training_curve.png', dpi=150)
        plt.show()
        
        final_train = train_losses[-1]
        initial_train = train_losses[0]
        
        print(f"\n=== TRAINING RESULTS ===")
        print(f"Initial train loss: {initial_train:.6f}")
        print(f"Final train loss: {final_train:.6f}")
        
    
    def test_inference_with_existing_model(self):
        """Test inference-------------------"""
        from inference import reverse_diffusion_sample
        
        print("\nTesting Inference:")
        
        lr_images, hr_images = next(iter(self.model.data_loaders['train']))
        lr_images = lr_images.to(self.device)
        hr_images = hr_images.to(self.device)
        
        lr_sample = lr_images[:4]
        hr_sample = hr_images[:4]
        
        print(f"Testing inference on {lr_sample.shape[0]} images...")
        
        with torch.no_grad():
            lr_sample_resized = torch.nn.functional.interpolate(
                lr_sample, size=(256, 256), mode='bicubic', align_corners=False
            )
            lr_latents = self.model.encode_images(lr_sample_resized)
            
            generated_hr_latents = reverse_diffusion_sample(
                self.model.unet, 
                lr_latents, 
                self.model.T, 
                self.model.betas, 
                self.device
            )
            
            generated_hr_images = self.model.decode_latents(generated_hr_latents)
            
            bicubic_hr = torch.nn.functional.interpolate(
                lr_sample, size=(256, 256), mode='bicubic', align_corners=False
            )
        
        self.visualize_inference_results(lr_sample, hr_sample, generated_hr_images, bicubic_hr)
    
    def visualize_inference_results(self, lr_images, hr_images, generated_images, bicubic_images):
        """Visualize inference results"""
        def tensor_to_numpy(tensor):
            tensor = tensor.cpu()
            tensor = torch.clamp((tensor + 1) / 2, 0, 1)
            return tensor.permute(0, 2, 3, 1).numpy()
        
        lr_np = tensor_to_numpy(lr_images)
        hr_np = tensor_to_numpy(hr_images)
        gen_np = tensor_to_numpy(generated_images)
        bicubic_np = tensor_to_numpy(bicubic_images)
        
        num_samples = lr_images.shape[0]
        
        fig, axes = plt.subplots(4, num_samples, figsize=(4*num_samples, 16))
        
        for i in range(num_samples):
            axes[0, i].imshow(lr_np[i])
            axes[0, i].set_title(f'LR Input #{i+1} (64x64)')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(bicubic_np[i])
            axes[1, i].set_title(f'Bicubic Upsampling #{i+1}')
            axes[1, i].axis('off')
            
            axes[2, i].imshow(gen_np[i])
            axes[2, i].set_title(f'Generated HR #{i+1} (Diffusion)')
            axes[2, i].axis('off')
            
            axes[3, i].imshow(hr_np[i])
            axes[3, i].set_title(f'Ground Truth HR #{i+1}')
            axes[3, i].axis('off')
        
        plt.tight_layout()
        plt.savefig('sanity_check_inference_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Inference test complete - results saved")

def main():
    """Run sanity check"""
    print("=== DIFFUSION SUPER-RESOLUTION SANITY CHECK ===")
    
    data_dir = input("Enter path to your processed data directory: ").strip()
    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        return
    
    hr_dir = os.path.join(data_dir, "HR_256")
    lr_dir = os.path.join(data_dir, "LR_64")
    
    if not os.path.exists(hr_dir) or not os.path.exists(lr_dir):
        print("HR_256 and LR_64 folders not found. Run process_data.py first!")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    checker = SanityChecker(data_dir, device=device, tiny_size=10)
    
    try:
        train_losses = checker.run_sanity_check(epochs=500, plot_every=50)
        
        print("\n=== SANITY CHECK SUMMARY ===")
        print(f"Train loss: {train_losses[0]:.6f} -> {train_losses[-1]:.6f}")
        print(f"Training curve and inference results saved")
        
        final_train = train_losses[-1]
        initial_train = train_losses[0]
        
    except Exception as e:
        print(f"SANITY CHECK FAILED: {e}")
        print("Check your data directory and model setup")

if __name__ == "__main__":
    main()
