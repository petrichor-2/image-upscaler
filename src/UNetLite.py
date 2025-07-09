import torch
import torch.nn as nn

# paper: U-Net: Convolutional Networks for Biomedical Image Segmentation
# https://arxiv.org/pdf/1505.04597

class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))
    
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super.__init__()
        self.conv = DoubleConvolution(in_channels, out_channels)
        self.pool = nn.MaxPoole2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        conv = self.conv(x)
        pool = self.pool(conv)

        return conv, pool
    
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upConv = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConvolution(in_channels, out_channels)

    def forward(self, upConv, skip):
        upConv = self.upConv(upConv)
        # add skip connection to the up conv
        x = torch.cat([upConv, skip], 1)
        return self.conv(x)
    
class UNetLite(nn.Module):
    # in_channels should equal out_channels which should equal the latent dim of the VAE encoder
    def __init__(self, in_channels, out_channels):
        self.downBlock1 = DownSample(in_channels, 32)
        self.downBlock2 = DownSample(32, 64)

        self.bottleNeck = DoubleConvolution(64, 128)

        self.upBlock1 = UpSample(128, 64)
        self.upBlock2 = UpSample(64, 32)

        self.out = nn.Conv2d(32, out_channels, kernel_size=1)
    
    def forward(self, x):
        conv1, pool1 = self.downBlock1(x)
        conv2, pool2 = self.downBlock2(pool1)

        bottle_neck = self.bottleNeck(pool2)

        # conv1 and conv2 are the skip connections
        up1 = self.upBlock1(bottle_neck, conv2)
        up2 = self.upBlock2(up1, conv1)

        return self.out(up2)