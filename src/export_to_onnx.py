import torch
from UNetLite import UNetLite

# Load model checkpoint
checkpoint_path = "/home/nazmus/Desktop/EdgeDiff SR/image-upscaler/src/model_epoch_50.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
T = checkpoint['T']
betas = checkpoint['betas']
unet_state = checkpoint['unet_state_dict']

# Detect base channels
base_channels = 64 if unet_state['downBlock1.conv.conv1.weight'].shape[0] == 64 else 32
print(f"Detected base_channels = {base_channels}")

# Instantiate model and load weights
unet = UNetLite(in_channels=8, out_channels=4, base_channels=base_channels)
unet.load_state_dict(unet_state)
unet.eval()

# Dummy input (batch size 1, 8 channels, 32x32) and timestep
x_dummy = torch.randn(1, 8, 32, 32)       # hr_latent + lr_latent concatenated
t_dummy = torch.tensor([10], dtype=torch.long)  # e.g. timestep 10

# Export to ONNX
torch.onnx.export(
    unet,
    (x_dummy, t_dummy),
    "unet.onnx",
    input_names=["x", "timestep"],
    output_names=["predicted_noise"],
    dynamic_axes={"x": {0: "batch_size"}, "timestep": {0: "batch_size"}, "predicted_noise": {0: "batch_size"}},
    opset_version=14
)

print("✅ Exported UNet to unet.onnx")
