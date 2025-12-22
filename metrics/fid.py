import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3
from torchvision.models import Inception_V3_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
from typing import Sequence, Tuple
import math

def get_inception(device: torch.device):
    weights = Inception_V3_Weights.IMAGENET1K_V1
    base = inception_v3(weights=weights, aux_logits=True)  # aux_logits must be True for these weights
    return_nodes = {"avgpool": "pool"}  # grab pooled 2048-d features
    feat_model = create_feature_extractor(base, return_nodes=return_nodes)
    feat_model.eval().to(device)
    for p in feat_model.parameters():
        p.requires_grad_(False)
    return feat_model

@torch.no_grad()
def _prep(images: torch.Tensor) -> torch.Tensor:
    # images: (N,3,H,W) in [0,1]; resize to 299
    if images.shape[-2:] != (299, 299):
        images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
    return images

@torch.no_grad()
def compute_activations(images: torch.Tensor, model: nn.Module, device: torch.device, batch_size: int = 32) -> torch.Tensor:
    images = images.to(device)
    acts = []
    for i in range(0, images.shape[0], batch_size):
        batch = _prep(images[i : i + batch_size])
        out = model(batch)["pool"]          # (N,2048,1,1)
        acts.append(out.view(out.size(0), -1))
    return torch.cat(acts, dim=0)  # (N,2048)

@torch.no_grad()
def calculate_statistics(acts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = acts.mean(0)
    # torch.cov requires float dtype
    sigma = torch.cov(acts.T)
    return mu, sigma

def _sqrtm_product(sigma1: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
    # Approximate sqrt of (sigma1 * sigma2) via eigen decomposition of the product
    prod = sigma1 @ sigma2
    # Symmetrize to reduce numerical issues
    prod = (prod + prod.T) * 0.5
    eigvals, eigvecs = torch.linalg.eigh(prod)
    eigvals = torch.clamp(eigvals, min=0)
    sqrt_prod = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T
    return sqrt_prod

@torch.no_grad()
def frechet_distance(mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor, sigma2: torch.Tensor) -> float:
    diff = mu1 - mu2
    covmean = _sqrtm_product(sigma1, sigma2)
    fid = diff.dot(diff) + torch.trace(sigma1 + sigma2 - 2 * covmean)
    return fid.item()

@torch.no_grad()
def compute_ref_stats(image_paths: Sequence, device: torch.device, model: nn.Module, max_images: int = None, resize_size: int = 299) -> Tuple[torch.Tensor, torch.Tensor]:
    to_tensor = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ToTensor(),
    ])
    imgs = []
    for p in image_paths[: max_images or len(image_paths)]:
        from PIL import Image
        img = Image.open(p).convert("RGB")
        imgs.append(to_tensor(img))
    images = torch.stack(imgs, dim=0)  # [N,3,H,W] in [0,1]
    acts = compute_activations(images, model, device)
    return calculate_statistics(acts)

@torch.no_grad()
def compute_fid(gen_images_tensor: torch.Tensor, ref_mu: torch.Tensor, ref_sigma: torch.Tensor, model: nn.Module, device: torch.device) -> float:
    acts_gen = compute_activations(gen_images_tensor, model, device)
    mu_gen, sigma_gen = calculate_statistics(acts_gen)
    return frechet_distance(ref_mu, ref_sigma, mu_gen, sigma_gen)
