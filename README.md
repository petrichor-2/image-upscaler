# Latent Diffusion for Chest X-Ray Super-Resolution

This project implements a super-resolution pipeline on X-ray images using **Latent Diffusion Models (LDMs)**. The goal is to recover high-resolution (HR) details from noisy latent representations, using a denoising UNet conditioned on low-resolution (LR) latent inputs.

---

## 📁 Dataset

We use the publicly available **NIH ChestX-ray14** dataset, which contains over 100,000 frontal-view chest radiographs. The images were preprocessed and downsampled to generate HR/LR latent pairs for training.

---

## 🧠 Model Overview

We follow a **latent denoising diffusion process** inspired by recent advancements in generative modeling.

### 🔄 Forward Diffusion Process
- A **beta schedule** is applied over `T=300` timesteps.
- Gaussian noise is incrementally added to the **high-resolution latent** to simulate a degradation process.
- The noisy latent is paired with a clean **low-resolution latent**.

### 🧩 UNet Architecture
- We use a custom lightweight **UNet**, inspired by the original biomedical segmentation paper.
- The UNet predicts the noise component to remove from the noisy HR latent.
- It is **conditioned on the LR latent**, concatenated channel-wise at the input.
- **Sinusoidal time-step embeddings** are added to convolutional blocks to guide the denoising process over time.

### 🎯 Loss Function
- The model is trained to minimize the **mean squared error (MSE)** between the predicted noise and the true noise used in the forward process.

---

## ✅ Current Progress

- ✅ Data preprocessing from ChestX-ray14  
- ✅ Forward diffusion implementation  
- ✅ UNet with LR-latent conditioning and timestep embeddings  
- ✅ Loss function and training loop  

---

## 🚧 Coming Soon

- [ ] VAE encoder and decoder for image-to-latent and latent-to-image mapping  
- [ ] Sampling (reverse diffusion loop) for inference  
- [ ] Evaluation metrics (PSNR, SSIM, etc.)  
- [ ] Training on full dataset with image reconstructions

---

## 📚 References

- **U-Net: Convolutional Networks for Biomedical Image Segmentation**  
  [https://arxiv.org/pdf/1505.04597](https://arxiv.org/pdf/1505.04597)

- **Denoising Diffusion Probabilistic Models**  
  [https://arxiv.org/pdf/2006.11239](https://arxiv.org/pdf/2006.11239)

---

## 🛠️ Dependencies

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib (optional for visualization)


