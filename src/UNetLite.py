import torch
import torch.nn as nn
import math

# paper: U-Net: Convolutional Networks for Biomedical Image Segmentation
# https://arxiv.org/pdf/1505.04597


class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Add time embedding projection if specified
        self.time_mlp = None
        if time_emb_dim is not None:
            self.time_mlp = nn.Linear(time_emb_dim, out_channels)
    
    def forward(self, x, t_emb=None):
        h = self.relu(self.conv1(x))
        
        # Add time embedding after first conv if provided
        if self.time_mlp is not None and t_emb is not None:
            time_emb = self.time_mlp(t_emb)
            # Reshape time embedding to be compatible with conv features
            time_emb = time_emb[(..., ) + (None, ) * 2]  # Add spatial dims to be broadcastable with image feature maps
            h = h + time_emb
            
        h = self.conv2(h)
        return self.relu(h)
    
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.conv = DoubleConvolution(in_channels, out_channels, time_emb_dim)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x, t_emb=None):
        conv = self.conv(x, t_emb)
        pool = self.pool(conv)
        return conv, pool
    
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.upConv = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConvolution(in_channels, out_channels, time_emb_dim)

    def forward(self, upConv, skip, t_emb=None):
        upConv = self.upConv(upConv)
        # add skip connection to the up conv
        x = torch.cat([upConv, skip], 1)
        return self.conv(x, t_emb)
    
# This module creates sinusoidal position/time-step embeddings
# It maps an integer time step to a high-dimensional vector using sine and cosine functions.
# This encoding is then passed through learnable MLP layers and added to UNet blocks to inform the model which timestep it’s denoising
class PositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000.0) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        # Positional encoding format: [sin, cos] concatenation for each dimension
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    
class UNetLite(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=256):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            PositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.ReLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        
        # Note: in_channels is now 2 * latent_dim due to concatenated HR and LR latents
        self.downBlock1 = DownSample(in_channels, 32, time_emb_dim)
        self.downBlock2 = DownSample(32, 64, time_emb_dim)

        self.bottleNeck = DoubleConvolution(64, 128, time_emb_dim)

        self.upBlock1 = UpSample(128, 64, time_emb_dim)
        self.upBlock2 = UpSample(64, 32, time_emb_dim)

        self.out = nn.Conv2d(32, out_channels, kernel_size=1)
    
    def forward(self, x, timestep):
        # Get time embeddings
        t_emb = self.time_mlp(timestep)
        
        # Forward pass with time embeddings
        conv1, pool1 = self.downBlock1(x, t_emb)
        conv2, pool2 = self.downBlock2(pool1, t_emb)

        bottle_neck = self.bottleNeck(pool2, t_emb)

        up1 = self.upBlock1(bottle_neck, conv2, t_emb)
        up2 = self.upBlock2(up1, conv1, t_emb)

        return self.out(up2)