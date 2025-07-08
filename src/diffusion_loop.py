import torch
import torch.nn.functional as F

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
alphas_bar = torch.cumprod(alphas, axis=0)
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

