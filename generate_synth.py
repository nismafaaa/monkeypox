import torch, os
from diffusers import StableDiffusionPipeline
from pathlib import Path
from tqdm import trange
import sys

MODEL_NAME = "runwayml/stable-diffusion-v1-5"
LORA_DIR   = os.path.expanduser("~/mpox-sd15-lora/weights-sd15-mpox")
OUT_DIR    = os.path.expanduser("~/mpox-sd15-lora/synth")
PROMPT     = "clinical photo of a mpox skin lesion, high detail, clinical lighting"
NEG_PROMPT = "watermark, text, logo, artifacts, unrealistic, drawing, cartoon, nsfw, nude"

NUM_IMAGES = 500
GUIDANCE   = 7.5
STEPS      = 30
SEED       = 1234

os.makedirs(OUT_DIR, exist_ok=True)
torch_dtype = torch.float16
device = "cuda"

# 🔧 Disable safety checker to prevent black images
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch_dtype,
    safety_checker=None
)
# For older/newer versions that still expect the flag:
pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
# Some versions also check this attribute:
pipe.requires_safety_checker = False

pipe = pipe.to(device)

# Load LoRA
pipe.load_lora_weights(LORA_DIR)
pipe.fuse_lora()

g = torch.Generator(device=device).manual_seed(SEED)

generated_paths = []
for i in trange(NUM_IMAGES):
    img = pipe(
        prompt=PROMPT,
        negative_prompt=NEG_PROMPT,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        generator=g
    ).images[0]
    img_path = os.path.join(OUT_DIR, f"mpox_synth_{i:05d}.jpg")
    img.save(img_path)
    generated_paths.append(img_path)

# --- Optional metrics after generation ---
if EVAL_REAL_DIR := os.getenv("EVAL_REAL_DIR"):  # directory of real images to compare; optional
    EVAL_MAX = int(os.getenv("EVAL_MAX", "200"))
    # Add metrics import path
    sys.path.append(str(Path(__file__).resolve().parent))
    try:
        from metrics.fid import get_inception, compute_ref_stats, compute_fid
        from metrics.ssim import compute_ssim
        from torchvision import transforms
    except Exception:
        pass

    real_dir = Path(EVAL_REAL_DIR)
    real_paths = sorted([p for p in real_dir.iterdir() if p.suffix.lower() in ('.jpg', '.png', '.jpeg')])
    gen_subset = generated_paths[:EVAL_MAX]
    to_tensor = transforms.ToTensor()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    fid_model = get_inception(device)
    fid_mu, fid_sigma = compute_ref_stats(real_paths, device, fid_model, max_images=EVAL_MAX)
    # Load generated subset
    from PIL import Image
    gen_imgs = [to_tensor(Image.open(p).convert("RGB")) for p in gen_subset]
    gen_tensor = torch.stack(gen_imgs).to(device)
    fid_val = compute_fid(gen_tensor, fid_mu, fid_sigma, fid_model, device)
    # SSIM (match counts)
    real_imgs_match = [to_tensor(Image.open(p).convert("RGB")) for p in real_paths[: len(gen_subset)]]
    real_tensor = torch.stack(real_imgs_match).to(device)
    ssim_val = compute_ssim(real_tensor, gen_tensor)
    print(f"[Metrics] FID: {fid_val:.3f} | SSIM: {ssim_val:.4f} (N={len(gen_subset)})")