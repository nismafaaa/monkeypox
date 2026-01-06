import torch, os
from diffusers import StableDiffusionPipeline
from pathlib import Path
from tqdm import trange
import sys

# --- CONFIGURATION ---
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
LORA_DIR   = os.path.expanduser("~/mpox-dm/mpox-sd15-lora/weights-sd15-mpox")
OUT_DIR    = os.path.expanduser("~/mpox-dm/mpox-sd15-lora/synth")

PROMPT     = "clinical photo of a monkeypox skin lesion, 4k, high resolution, sharp focus" 
NEG_PROMPT = "watermark, text, logo, artifacts, unrealistic, drawing, cartoon, nsfw, nude"

NUM_IMAGES = 500
GUIDANCE   = 7.5
STEPS      = 30
SEED       = 1234

# --- GENERATION SETUP ---
os.makedirs(OUT_DIR, exist_ok=True)
torch_dtype = torch.bfloat16 
device = "cuda"

print(f"[Setup] Loading Pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch_dtype,
    safety_checker=None
)
pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
pipe.requires_safety_checker = False

pipe = pipe.to(device)

print(f"[Setup] Loading LoRA from {LORA_DIR}...")
pipe.load_lora_weights(LORA_DIR)
pipe.fuse_lora()

# --- GENERATION LOOP ---
g = torch.Generator(device=device).manual_seed(SEED)

generated_paths = []
print(f"[Generation] Generating {NUM_IMAGES} images...")
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

# --- METRICS EVALUATION BLOCK (UPDATED) ---
if EVAL_REAL_DIR := os.getenv("EVAL_REAL_DIR"):
    print(f"\n[Evaluation] Starting metrics calculation against: {EVAL_REAL_DIR}")

    sys.path.append(os.getcwd()) 
    try:
        from metrics.fid import compute_fid_score
        from metrics.ssim import compute_ssim
        from torchvision import transforms
        from PIL import Image
    except ImportError:
        print("Error: Could not import metrics. Ensure 'metrics/fid.py' and 'metrics/ssim.py' exist.")
        sys.exit(1)

    real_dir = Path(EVAL_REAL_DIR)
    fake_dir = Path(OUT_DIR)
    
    real_paths = sorted([str(p) for p in real_dir.iterdir() if p.suffix.lower() in ('.jpg', '.png', '.jpeg')])
    fake_paths = sorted([str(p) for p in fake_dir.iterdir() if p.suffix.lower() in ('.jpg', '.png', '.jpeg')])

    eval_max_env = os.getenv("EVAL_MAX")
    limit = int(eval_max_env) if eval_max_env else min(len(real_paths), len(fake_paths))
    
    print(f"[Evaluation] Comparing {limit} images...")
    real_paths = real_paths[:limit]
    fake_paths = fake_paths[:limit]

    print("[Metrics] Calculating FID...")
    fid_val = compute_fid_score(real_paths, fake_paths, device_str="cuda")

    print("[Metrics] Calculating SSIM...")

    transform = transforms.Compose([
        transforms.Resize((512, 512)), 
        transforms.ToTensor(),
    ])
    
    def load_tensor_batch(paths):
        imgs = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            imgs.append(transform(img))
        return torch.stack(imgs).to(device)

    real_tensor = load_tensor_batch(real_paths)
    gen_tensor = load_tensor_batch(fake_paths)
    
    ssim_val = compute_ssim(real_tensor, gen_tensor)

    print("-" * 40)
    print(f"FINAL RESULTS (N={limit})")
    print(f"FID:  {fid_val:.4f} (Lower is better)")
    print(f"SSIM: {ssim_val:.4f} (Target: ~0.3-0.5)")
    print("-" * 40)