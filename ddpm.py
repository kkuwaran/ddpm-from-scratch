from typing import Dict, List, Union, Tuple

import torch
import torch.nn.functional as F



def create_linear_schedule(start: float, end: float, num_timesteps: int, device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """
    Creates a linear beta schedule and precomputes terms used in the DDPM sampling process
    - start: Starting value of beta at timestep 0
    - end: Ending value of beta at the final timestep
    - num_timesteps: Number of diffusion timesteps (T)
    """

    # Generate linearly spaced beta values
    betas = torch.linspace(start=start, end=end, steps=num_timesteps, device=device)

    # Calculate alpha values: 1 - beta
    alphas = 1.0 - betas
    # Compute the cumulative product of alphas: bar{alpha}
    alphas_bar = torch.cumprod(alphas, dim=0)
    # Shift alphas_bar by one timestep to compute alphas_bar_prev
    alphas_bar_prev = F.pad(alphas_bar[:-1], (1, 0), value=1.0)

    # Precompute frequently used terms for sampling
    sqrt_alphas_bar = torch.sqrt(alphas_bar)  # sqrt{bar{alpha}}
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)  # 1 / sqrt{alpha}
    sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - alphas_bar)  # sqrt{1 - bar{alpha}}
    posteriors_std = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)  # sigma_t

    # Return all precomputed schedule values in a dictionary
    schedule = {
        'betas': betas,
        'alphas': alphas,
        'alphas_bar': alphas_bar,
        'sqrt_alphas_bar': sqrt_alphas_bar,
        'sqrt_recip_alphas': sqrt_recip_alphas,
        'sqrt_one_minus_alphas_bar': sqrt_one_minus_alphas_bar,
        'posteriors_std': posteriors_std
    }
    return schedule



def ddpm_forward(images: torch.Tensor, timesteps: torch.Tensor, schedule: Dict[str, torch.Tensor], 
                 device: str = 'cpu') -> Tuple[torch.Tensor]:
    """
    Forward diffusion process for DDPM
    - images: input batch of images; shape (batch_size, channels, height, width)
    - timesteps: timesteps for each image in the batch; shape (batch_size, )
    - schedule['sqrt_alphas_bar']: precomputed sqrt{ bar{alpha} }; shape (T, )
    - schedule['sqrt_one_minus_alphas_bar']: precomputed sqrt{ 1 - bar{alpha} }; shape (T, )
    """

    # Extract variables
    batch_size = images.shape[0]
    sqrt_alphas_bar: torch.Tensor = schedule['sqrt_alphas_bar']
    sqrt_one_minus_alphas_bar: torch.Tensor = schedule['sqrt_one_minus_alphas_bar']

    # Validate input shapes
    assert timesteps.shape == (batch_size, ), "incorrect timesteps"
    assert sqrt_alphas_bar.ndim == 1, "incorrect alpha_bars"
    assert sqrt_one_minus_alphas_bar.ndim == 1, "The incorrect sqrt_one_minus_alphas_bar"

    # Generate random Gaussian noise
    noises = torch.randn_like(images, device=device)

    # Apply the forward diffusion formula
    noisy_images = (
        sqrt_alphas_bar[timesteps].view(batch_size, 1, 1, 1) * images +
        sqrt_one_minus_alphas_bar[timesteps].view(batch_size, 1, 1, 1) * noises
    )
    return noisy_images, noises



@torch.no_grad()
def ddpm_inference(input_noise: torch.Tensor, model: torch.nn.Module, schedule: Dict[str, torch.Tensor], 
                   return_all : bool = False, device: str = 'cpu') -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Implements the DDPM sampling algorithm
    - input_noise: Initial Gaussian noise; shape (batch_size, channels, height, width)
    - model: Trained noise prediction model; predicts added noise given input and timestep
    - schedule: Precomputed schedule variables
    - return_all: If True, returns images at all noise levels; otherwise, only the final denoised image
    """
    
    # Extract input variables
    x = input_noise
    batch_size = x.shape[0]

    # Extract schedule variables
    betas: torch.Tensor = schedule['betas']
    sqrt_recip_alphas: torch.Tensor = schedule['sqrt_recip_alphas']
    sqrt_one_minus_alphas_bar: torch.Tensor = schedule['sqrt_one_minus_alphas_bar']
    posteriors_std: torch.Tensor = schedule['posteriors_std']
    T = betas.numel()
    
    # reverse sampling process
    imgs = []
    for time_step in reversed(range(T)):
        # Sample noise only if it is not the last timestep
        noise = torch.randn_like(x) if time_step > 0 else 0
        
        # Create a timestep tensor for the batch
        t = torch.full((batch_size, ), time_step, device=device, dtype=torch.long)
        
        # Predict the clean image at the current timestep
        predicted_clean = sqrt_recip_alphas[t].view(batch_size, 1, 1, 1) * (
            x - betas[t].view(batch_size, 1, 1, 1) * model(x, t) / 
            sqrt_one_minus_alphas_bar[t].view(batch_size, 1, 1, 1)
        ) + torch.sqrt(posteriors_std[t].view(batch_size, 1, 1, 1)) * noise

        # Update the noisy image
        x = predicted_clean
        
        # Clamp the output to [-1, 1]
        x_clamp = torch.clamp(x, -1, 1)
        imgs.append(x_clamp)
    
    # Return results
    if return_all:
        return imgs  # List of images across timesteps
    else:
        return imgs[-1]  # Final denoised image