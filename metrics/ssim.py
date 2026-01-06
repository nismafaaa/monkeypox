import torch
import torch.nn.functional as F
from typing import Tuple

def gaussian_window(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Creates a 2D Gaussian window for SSIM weighting."""
    coords = torch.arange(size, dtype=torch.float32, device=device)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.unsqueeze(1) @ g.unsqueeze(0)

def ssim_map(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the SSIM map between two images (or batches).
    Expects inputs to be (N, C, H, W) in range [0, 1].
    """
    channel = img1.size(1)
    
    # Create window
    window = gaussian_window(window_size, sigma, img1.device)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean([1, 2, 3])  # Average over channels and pixels

def compute_ssim(real_tensor: torch.Tensor, gen_tensor: torch.Tensor) -> float:
    """
    Computes the mean SSIM between two batches of images.
    Args:
        real_tensor: (N, 3, H, W) float32 tensor in [0, 1]
        gen_tensor:  (N, 3, H, W) float32 tensor in [0, 1]
    Returns:
        float: The average SSIM score (0.0 to 1.0)
    """
    # Ensure inputs are on the same device
    device = real_tensor.device
    gen_tensor = gen_tensor.to(device)

    # Calculate per-image SSIM
    scores = ssim_map(real_tensor, gen_tensor)
    
    # Return average across the batch
    return scores.mean().item()