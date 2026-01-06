import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
from typing import Sequence, Tuple
import numpy as np
from scipy import linalg  # Standard for FID calculation

def get_inception(device: torch.device):
    """Loads the Inception V3 model for feature extraction."""
    weights = Inception_V3_Weights.IMAGENET1K_V1
    base = inception_v3(weights=weights, aux_logits=True)
    return_nodes = {"avgpool": "pool"}
    feat_model = create_feature_extractor(base, return_nodes=return_nodes)
    feat_model.eval().to(device)
    for p in feat_model.parameters():
        p.requires_grad_(False)
    return feat_model

def get_transforms(resize_size=299):
    """
    Standard transforms for InceptionV3.
    CRITICAL FIX: Added Normalization to match ImageNet stats.
    Without this, the model sees 'wrong' colors and outputs garbage features.
    """
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

@torch.no_grad()
def compute_activations(images: torch.Tensor, model: nn.Module, device: torch.device, batch_size: int = 32) -> np.ndarray:
    """Calculates features (activations) for a batch of images."""
    model.to(device)
    acts = []
    
    # Process in batches to save VRAM
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size].to(device)
        out = model(batch)["pool"]  # (N, 2048, 1, 1)
        out = out.squeeze(3).squeeze(2) # Flatten to (N, 2048)
        acts.append(out.cpu().numpy())
    
    return np.concatenate(acts, axis=0)

def calculate_fid_from_stats(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Numpy implementation of the Frechet Distance.
    Uses scipy.linalg.sqrtm to avoid the 'matrix not symmetric' crash.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        print("fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight complex component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def compute_stats_from_paths(paths, model, device):
    """Loads images, runs the model, and computes mean/covariance."""
    tf = get_transforms()
    imgs = []
    
    # Load all images
    for p in paths:
        from PIL import Image
        img = Image.open(p).convert("RGB")
        imgs.append(tf(img))
    
    # Stack into one tensor
    batch = torch.stack(imgs)
    
    # Get features
    acts = compute_activations(batch, model, device)
    
    # Calculate stats in Numpy
    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)
    return mu, sigma

# --- Main Interface ---
def compute_fid_score(real_paths, fake_paths, device_str="cuda"):
    """
    The main function you call from generate_synth.py
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model = get_inception(device)
    
    print(f"Computing stats for Real images ({len(real_paths)})...")
    m1, s1 = compute_stats_from_paths(real_paths, model, device)
    
    print(f"Computing stats for Fake images ({len(fake_paths)})...")
    m2, s2 = compute_stats_from_paths(fake_paths, model, device)
    
    fid_value = calculate_fid_from_stats(m1, s1, m2, s2)
    return fid_value