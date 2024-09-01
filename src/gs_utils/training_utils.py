import torch
from torch import nn
import numpy as np
    
    


def loss_fn(outputs, inputs, weight):
    """
    Match output and input points that, position wise, are closest together
    Once matched, the loss is the sum of the mean squared errors
    """

    # Rearrange outputs to match closest input points
    rearranged_outputs = _rearrange_outputs(outputs, inputs)
    
    # Compute element-wise squared error and apply weight
    squared_error = (rearranged_outputs - inputs) ** 2
    weighted_squared_error = squared_error * weight.unsqueeze(1).unsqueeze(2)
    
    # Compute mean of weighted squared error
    mse_loss = torch.mean(weighted_squared_error)
    
    return mse_loss


def _rearrange_outputs(outputs, inputs):
    batch_size, num_points, feature_dim = outputs.shape
    
    # Compute pairwise distances between output and input points within each batch
    distances = torch.stack([torch.cdist(outputs[i], inputs[i]) for i in range(batch_size)])
    
    # Find the closest input point for each output point
    closest_indices = torch.argmin(distances, dim=2)
    
    # Rearrange the outputs to match the closest input points
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, num_points)
    point_indices = torch.arange(num_points).unsqueeze(0).expand(batch_size, -1)
    
    rearranged_outputs = outputs[batch_indices, closest_indices, :]
    
    return rearranged_outputs


def add_noise(batch, 
        max_level_pos=1,
        max_level_color=1,
        max_level_opacity=1,
        max_level_scale=1,
        max_level_rot=1
    ):

    diffusion_steps, alphas_cumprod = cosine_beta_schedule(250)

    # Add Gaussian noise to the batch data
    t = torch.randint(0, 250, (batch.shape[0],))
    noise_level = diffusion_steps[t]
    noise_level = torch.tensor(noise_level, device=batch.device).unsqueeze(1).unsqueeze(2)

    # Inverse loss for noise level
    alpha = torch.tensor(alphas_cumprod[t], device=batch.device)
    weight = 1 / torch.sqrt(alpha)
    
    # Create noise vector with the same batch size and length as batch
    batch_size, seq_length, _ = batch.shape
    noise = torch.zeros(batch_size, seq_length, 14, device=batch.device)

    # Apply noise to specific elements
    noise[:, :, 0:3] = torch.randn_like(noise[:, :, 0:3]) * (noise_level * max_level_pos)
    noise[:, :, 3:6] = torch.randn_like(noise[:, :, 3:6]) * (noise_level * max_level_color)
    noise[:, :, 6:7] = torch.randn_like(noise[:, :, 6:7]) * (noise_level * max_level_opacity)
    noise[:, :, 7:10] = torch.randn_like(noise[:, :, 7:10]) * (noise_level * max_level_scale)
    noise[:, :, 10:14] = torch.randn_like(noise[:, :, 10:14]) * (noise_level * max_level_rot)
    return batch + noise, weight


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Noise schedule for the diffusion model
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999), alphas_cumprod


class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # Extract points from each element in the batch
        point_lists = [item['points'] for item in batch]
        
        # Pad sequences to the same length
        max_points = max(len(points) for points in point_lists)
        padded_points = [points + [[0.0] * 14] * (max_points - len(points)) for points in point_lists]
        
        # Convert to tensor and reshape
        tensor_points = torch.tensor(padded_points)
        
        # Reshape to (batch_size, num_points, 14)
        tensor_points = tensor_points.view(-1, max_points, 14)
        
        # Move to the appropriate device
        tensor_points = tensor_points.to(self.tokenizer.device)
        
        # Return the processed tensor
        return tensor_points
