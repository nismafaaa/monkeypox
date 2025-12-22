import torch
import torch.nn.functional as F

def _gaussian_kernel(size: int, sigma: float, device: torch.device):
    coords = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    kernel1d = g.view(1, 1, -1)
    kernel2d = kernel1d.transpose(-1, -2) @ kernel1d
    return kernel2d

def _window(channels: int, size: int, sigma: float, device: torch.device):
    k = _gaussian_kernel(size, sigma, device)
    k = k.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return k.repeat(channels, 1, 1, 1)

def compute_ssim(a: torch.Tensor, b: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> float:
    """
    a,b: (N,3,H,W) in [0,1]
    """
    assert a.shape == b.shape
    device = a.device
    window = _window(a.shape[1], window_size, sigma, device)
    pad = window_size // 2

    mu1 = F.conv2d(a, window, padding=pad, groups=a.shape[1])
    mu2 = F.conv2d(b, window, padding=pad, groups=b.shape[1])
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2

    sigma1_sq = F.conv2d(a * a, window, padding=pad, groups=a.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(b * b, window, padding=pad, groups=b.shape[1]) - mu2_sq
    sigma12 = F.conv2d(a * b, window, padding=pad, groups=a.shape[1]) - mu1_mu2

    C1 = (0.01**2)
    C2 = (0.03**2)
    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / (den + 1e-8)
    return ssim_map.mean().item()
