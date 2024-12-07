import os
from typing import Dict

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import torch
import torchvision

from ddpm import ddpm_forward, ddpm_inference



def viz_batch(batch: torch.Tensor, path: str = None, nrow: int = 8, dpi: float = 75) -> None:
    """
    Visualizes a batch of images as a grid and optionally saves it to a file
    - batch: A batch of images with shape (batch_size, channels, height, width)
    - path: File path to save the visualization (e.g., 'output/grid.png')
    - nrow: Number of images per row in the grid. Defaults to 8
    """

    # Detach the batch from GPU, move to CPU, and ensure it is in NumPy-compatible format
    batch_cpu = batch.detach().cpu()

    # Create a grid of images with padding and normalization for better visualization
    image_grid = torchvision.utils.make_grid(batch_cpu, nrow=nrow, padding=2, normalize=True, scale_each=True)

    # Rearrange dimensions to (height, width, channels) for display
    image_grid_np = image_grid.permute(1, 2, 0).numpy()

    # Create a figure to display the grid
    fig, ax = plt.subplots(dpi=dpi)
    ax.imshow(image_grid_np)
    ax.axis("off")

    # Save the image grid to the specified path
    if path:
        dir_name = os.path.dirname(path)
        os.makedirs(dir_name, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", pad_inches=0)
        
    # Display the image grid, and close the figure to free memory
    plt.show()
    plt.close(fig)



def viz_forward_diffusion(images: torch.Tensor, num_timesteps: int, schedule: Dict[str, torch.Tensor], 
                          num_checkpoints: int = 16, interleave: bool = False, dpi: float = 75, 
                          device: str = 'cpu') -> None:
    """
    Generate and visualize the forward diffusion process for DDPM
    - images: The input images to diffuse, shape (batch_size, channels, height, width)
    - num_timesteps: Total number of diffusion timesteps, T
    - schedule: Precomputed schedule values for the diffusion process
    - num_checkpoints: Number of timesteps to visualize evenly spaced across the diffusion process
    - interleave: If True, interleave noisy images and the noise added for visualization
    """

    # Extract shape information
    batch_size, *image_shape = images.shape

    # Compute the step size to sample checkpoints evenly
    step_size = max(1, num_timesteps // num_checkpoints)

    # Lists to store the series of noisy images and the corresponding added noises
    noisy_images_series = []
    added_noise_series = []

    with torch.no_grad():
        for timestep in range(0, num_timesteps, step_size):
            # Create a tensor for the current timestep
            t = torch.full((batch_size, ), timestep, device=device, dtype=torch.long)

            # Forward diffusion process for the current timestep
            noisy_images, added_noise = ddpm_forward(images, t, schedule, device=device)

            # Collect the generated noisy images and noises
            noisy_images_series.append(noisy_images)
            added_noise_series.append(added_noise)

    # Convert the series to tensors with shape (batch_size, num_checkpoints, *)
    noisy_images_batch = torch.stack(noisy_images_series, dim=1)  # Shape: (batch_size, num_checkpoints, *)
    print(noisy_images_batch.shape)

    # If interleave is True, interleave noisy images and added noise for visualization
    if interleave:
        # Stack images and noises along a new dimension, then interleave
        added_noise_batch = torch.stack(added_noise_series, dim=1)  # Shape: (batch_size, num_checkpoints, *)
        interleaved = torch.stack((noisy_images_batch, added_noise_batch), dim=2)  # Shape: (batch_size, num_checkpoints, 2, *)
        noisy_images_batch = interleaved.view(batch_size, 2 * num_checkpoints, *image_shape)  # Shape: (batch_size, 2 * num_checkpoints, *)

    # Flatten sequence of images to shape (batch_size * num_steps, *) where num_steps = num_checkpoints or 2*num_checkpoints
    flattened_noisy_images = noisy_images_batch.view(-1, *image_shape)

    # Visualize the batch of images
    num_steps = noisy_images_batch.shape[1]
    viz_batch(flattened_noisy_images, nrow=num_steps, dpi=dpi)



def viz_inference_steps(initial_noises: torch.Tensor, model: torch.nn.Module, 
                        schedule: Dict[str, torch.Tensor], path: str = None, device: str = 'cpu') -> None:
    """
    Visualizes the denoising steps during DDPM inference and saves the output as an image grid
    - initial_noises: The initial noise; shape (num_images, C, H, W).
    - model: The DDPM model used for denoising
    - schedule: The DDPM noise schedule containing betas, alphas, etc
    - epoch: The current epoch number (used in naming the output file)
    """

    # Perform DDPM inference and obtain images at each step
    images_sequence = ddpm_inference(initial_noises, model, schedule, return_all=True, device=device)

    # Stack images: include the initial noise and select intermediate steps at equal intervals
    steps_to_visualize = [images_sequence[0]] + [img for img in images_sequence[63::64]]
    images_stack = torch.stack(steps_to_visualize, dim=1)  # shape: (num_images, num_steps, C, H, W)

    # Flatten the sequence for visualization
    num_steps = images_stack.shape[1]
    flattened_images = images_stack.view(-1, *images_stack.shape[2:])  # shape: (num_images * num_steps, C, H, W)

    # Visualize the batch as a grid image and save it
    viz_batch(flattened_images, path=path, nrow=num_steps, dpi=150)



def plot_loss_and_lr(train_loss_per_epoch: list, lr_per_epoch: list, path: str = None) -> None:
    """Plots training loss and learning rate over epochs using dual y-axes"""

    # Typing annotations
    fig: Figure
    ax1: Axes
    ax2: Axes

    epochs = range(1, len(train_loss_per_epoch) + 1)
    
    # Create the figure and axis
    fig, ax1 = plt.subplots(figsize=(6, 4.5))
    
    # Plot training loss on the left y-axis
    ax1.plot(epochs, train_loss_per_epoch, 'b-o', label='Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Create a twin y-axis for the learning rate
    ax2 = ax1.twinx()
    ax2.plot(epochs, lr_per_epoch, 'r-s', label='Learning Rate')
    ax2.set_ylabel('Learning Rate', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Add a title and legends
    fig.suptitle('Training Loss and Learning Rate per Epoch')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Adjust layout
    fig.tight_layout()

    # Save the plot to the specified path
    if path:
        dir_name = os.path.dirname(path)
        os.makedirs(dir_name, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", pad_inches=0)

    plt.show()