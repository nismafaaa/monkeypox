import os
import sys
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from metrics.fid import compute_fid_score
    from metrics.ssim import compute_ssim
    from torchvision import transforms
    from PIL import Image
    import torch
except ImportError as e:
    print(f"\nCRITICAL ERROR: {e}")
    print(f"Python is looking in: {sys.path}")
    print("Please verify you have a folder named 'metrics' containing 'fid.py' and 'ssim.py'\n")
    sys.exit(1)

REAL_DIR = os.path.expanduser("~/mpox-dm/dataset")
FAKE_DIR = os.path.expanduser("~/mpox-dm/mpox-sd15-lora/synth")
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

def load_tensor_batch(paths, device):
    """Helper to load images for SSIM"""
    transform = transforms.Compose([
        transforms.Resize((512, 512)), 
        transforms.ToTensor(),
    ])
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        imgs.append(transform(img))
    return torch.stack(imgs).to(device)

def main():
    print(f"--- Evaluation Mode Only ---")
    print(f"Real Images: {REAL_DIR}")
    print(f"Fake Images: {FAKE_DIR}")

    real_path_obj = Path(REAL_DIR)
    fake_path_obj = Path(FAKE_DIR)
    
    if not real_path_obj.exists():
        print(f"Error: Real dataset dir not found at {REAL_DIR}")
        return
    if not fake_path_obj.exists():
        print(f"Error: Synth dataset dir not found at {FAKE_DIR}")
        return

    real_paths = sorted([str(p) for p in real_path_obj.iterdir() if p.suffix.lower() in ('.jpg', '.png', '.jpeg')])
    fake_paths = sorted([str(p) for p in fake_path_obj.iterdir() if p.suffix.lower() in ('.jpg', '.png', '.jpeg')])

    limit = min(len(real_paths), len(fake_paths))
    print(f"\nFound {len(real_paths)} real and {len(fake_paths)} fake images.")
    print(f"Evaluating on {limit} images...")
    
    real_paths = real_paths[:limit]
    fake_paths = fake_paths[:limit]

    print("\n[1/2] Calculating FID... (This loads the Inception model)")
    fid_val = compute_fid_score(real_paths, fake_paths, device_str=DEVICE)

    print("[2/2] Calculating SSIM...")
    real_tensor = load_tensor_batch(real_paths, DEVICE)
    fake_tensor = load_tensor_batch(fake_paths, DEVICE)
    ssim_val = compute_ssim(real_tensor, fake_tensor)

    print("\n" + "="*40)
    print(f"FINAL METRICS (N={limit})")
    print(f"FID:  {fid_val:.4f}  (Lower is better)")
    print(f"SSIM: {ssim_val:.4f} (Target ~0.3-0.5)")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()