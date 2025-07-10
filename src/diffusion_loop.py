import torch
import torch.nn.functional as F
from UNetLite import UNetLite
from torch.utils.data import DataLoader, Dataset

# this will return a tensor of our betas increasing linearly for timesteps steps
def beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def extract_timestep_coefficients(coeffs, t, shape):
    """
    Extracts the t-th value in some list and returns a tensor broadcasted
    to match the shape of the image tensor so we can multiply the t-th
    value elementwise with the image tensor.
    """

    batch_size = t.shape[0]
    values = coeffs.gather(-1, t.cpu()) # gets appropriate value from list of coeffs for each t
    return values.reshape(batch_size, *((1,) * (len(shape) - 1))).to(t.device)


# Beta Schedule
T = 300
betas = beta_schedule(timesteps=T)

# Calculate different parts of the equations
alphas = 1. - betas
alphas_bar = torch.cumprod(alphas, dim=0)
alphas_bar_prev = F.pad(alphas_bar[:-1], (1,0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0/ alphas)
sqrt_alphas_bar = torch.sqrt(alphas_bar)
sqrt_one_minus_alphas_bar = torch.sqrt(1 - alphas_bar)
posterior_variance = betas * (1. - alphas_bar_prev) / (1. - alphas_bar)

def forward_diffusion_sample(hr_latent, t, device="cpu"):
    """
    Takes a high resolution latent image and timestep as input 
    and returns the noisy version of it
    """
    noise = torch.randn_like(hr_latent)
    sqrt_alphas_bar_t = extract_timestep_coefficients(sqrt_alphas_bar, t, hr_latent.shape)
    sqrt_one_minus_aplhas_bar_t = extract_timestep_coefficients(sqrt_one_minus_alphas_bar, t, hr_latent.shape)

    # return the noisy image and the noise
    return sqrt_alphas_bar_t.to(device) * hr_latent.to(device) + sqrt_one_minus_aplhas_bar_t.to(device) * noise.to(device) , noise.to(device)

def get_loss(model, lr_latent, hr_latent, t, device="cpu"):
    """
    Takes a model, low resolution latent image, high resolution latent image and timestep as input
    and returns the loss of the model
    """
    # get the noisy version of the high resolution latent image
    noisy_hr_latent, noise = forward_diffusion_sample(hr_latent, t, device)

    # concatenate the noisy high resolution latent image and low resolution latent image
    # to create the input for the UNet model
    x = torch.cat([noisy_hr_latent, lr_latent], dim=1)  # Concatenate noisy HR latent and LR latent
    
    # get the predicted noise from the model
    predicted_noise = model(x, t.to(device))

    # calculate the loss
    return F.mse_loss(predicted_noise, noise.to(device))


# training loop
from torch.optim import Adam

def train_basic_diffusion(dataloader, epochs=100):
    """
    Basic training loop for diffusion model
    This is a simple example - use train_latent_diffusion.py for the full pipeline
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetLite(in_channels=2 * 4, out_channels=4, time_emb_dim=256)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # Assuming batch is a tuple: (lr_latent, hr_latent)
            lr_latent = batch[0].to(device)
            hr_latent = batch[1].to(device)
            BATCH_SIZE = lr_latent.shape[0]

            t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
            loss = get_loss(model, lr_latent, hr_latent, t, device)
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0 and step == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")

if __name__ == "__main__":
    # Example usage - only runs when script is executed directly
    print("This is a module with diffusion functions.")
    print("Use train_latent_diffusion.py for the complete training pipeline.")