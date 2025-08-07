# Latent Diffusion for Chest X-Ray Super-Resolution

This project implements a **Latent Diffusion Model (LDM)** pipeline to perform **4× super-resolution** on chest X-ray images. We learn to recover high-resolution (HR) images from noisy latent representations using a **UNet-based denoiser** conditioned on low-resolution (LR) latents and timestep embeddings.

---

## Dataset

We use the **NIH ChestX-ray14** dataset, containing over 100,000 frontal-view radiographs.  
For training, we preprocess and randomly sample **10,000 paired images**, resizing:
- LR images to **64×64**, then bicubically upsampling to **256×256**
- HR images to **256×256**

Both are passed through a pretrained **VAE encoder (AutoencoderKL from Hugging Face)** to extract:
- LR latent: `4×32×32`  
- HR latent: `4×32×32`  
Then the forward diffusion process begins.

---

## Model Architecture

Our LDM pipeline consists of three main components:

### 1. Variational Autoencoder (VAE)
- Pretrained AutoencoderKL encodes RGB images to latent space (`3×256×256 → 4×32×32`) and decodes back.
- Used to encode both LR and HR inputs during training, and decode outputs during inference.

### 2. Forward Diffusion
- Noise is added to the **HR latent** over `T=300` steps using a **beta schedule** from the DDPM paper.
- Closed-form sampling is used to directly obtain `z_t` from `z_0` and timestep `t`.
- LR latent is **not noised** and is used as a conditional input.
- Timesteps are sampled uniformly and used to condition the model via **sinusoidal embeddings**.

### 3. UNet Denoiser
- Lightweight encoder-decoder UNet with skip connections.
- Takes as input:  
  `concat([noisy HR latent, clean LR latent]) → [8×32×32]`  
  along with the timestep `t`.
- Predicts the **noise ε** added during forward diffusion.
- Trained with **MSE loss** between predicted and true noise.

---

## Inference (Reverse Diffusion)

- Start from pure noise `z_T ∼ N(0, I)` in HR latent space.
- At each step `t ∈ [T, ..., 0]`:
  - Predict noise `ε̂_t = UNet(z_t, LR_latent, t)`
  - Use closed-form DDPM reverse update to compute `z_{t-1}`
- After all steps, decode the final latent using the VAE decoder to get a **super-resolved 256×256 image**.

---

## Evaluation

- Super-resolved images are compared to ground-truth HR images.
- **Peak Signal-to-Noise Ratio (PSNR)** and **Structural Similarity Index (SSIM)** are used as evaluation metrics.
- Our model significantly outperforms bicubic interpolation on both metrics.

---

## Edge Deployment with TensorRT

To optimize for fast deployment on NVIDIA GPUs:

- Exported the trained UNet model to **ONNX** format (`.onnx`)
- Converted ONNX to a **TensorRT engine** (`.trt`) for runtime execution
- Implemented CUDA-based inference pipeline using **PyCUDA + TensorRT**
- Achieved **~3.8× faster inference** with negligible quality loss (lower precision enabled by TRT)
- Integrated into main inference pipeline with seamless fallback between PyTorch and TensorRT

---

## Project Status

- VAE integration for image ↔ latent mapping  
- Forward diffusion process (closed-form)  
- Custom UNet with LR + timestep conditioning  
- Training loop (PyTorch)  
- Reverse diffusion and inference sampling  
- Evaluation (PSNR, SSIM)  
- ONNX export and TensorRT deployment  
- End-to-end super-resolution working on GPU  
- Modular PyTorch & TensorRT inference wrappers

---

## Dependencies

- Python 3.8+
- PyTorch ≥ 1.13  
- Hugging Face Transformers / Diffusers (AutoencoderKL)
- NumPy
- TensorRT 8.x / 10.x
- PyCUDA

---

## References

- **Denoising Diffusion Probabilistic Models**  
  [https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
- **U-Net: Biomedical Image Segmentation**  
  [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)
- **High-Resolution Image Synthesis with Latent Diffusion Models**  
  [https://arxiv.org/abs/2112.10752](https://arxiv.org/abs/2112.10752)
