import os, cv2, glob
import numpy as np
from tqdm import tqdm
from pathlib import Path
import albumentations as A

SRC_DIR = os.path.expanduser("~/mpox-dataset")
DST_DIR = os.path.expanduser("~/mpox-augmented")

# How many augmented variants per source image
AUG_PER_IMAGE = 5

# Medical-safe-ish aug policy:
# - modest geometry changes
# - mild photometric jitter
# - mild blur/noise
# - slight elastic distort (low alpha)
# Avoid extreme color shifts that could change lesion appearance
transform = A.Compose([
    A.OneOf([
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.RandomRotate90(p=1.0),
    ], p=0.6),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.6),
    A.RandomResizedCrop(height=512, width=512, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5),
    A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02, p=0.4),
    A.GaussNoise(var_limit=(5.0, 25.0), p=0.4),
    A.GaussianBlur(blur_limit=(3,5), p=0.25),
    A.ElasticTransform(alpha=5, sigma=30, alpha_affine=5, border_mode=cv2.BORDER_REFLECT_101, p=0.2),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.15),
    A.Normalize(mean=(0.0,0.0,0.0), std=(1.0,1.0,1.0), p=0.0)  # noop placeholder if you later add tensors
], p=1.0)

def is_image(path):
    ext = path.lower().split('.')[-1]
    return ext in ['jpg','jpeg','png','bmp','tif','tiff','webp']

def all_images(root):
    return [p for p in glob.glob(str(Path(root) / "**" / "*"), recursive=True) if is_image(p)]

def dst_path(src_path):
    rel = os.path.relpath(src_path, SRC_DIR)
    return os.path.join(DST_DIR, rel)

def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def read_bgr(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read {path}")
    return img

def to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def to_bgr(img_rgb):
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

def main():
    imgs = all_images(SRC_DIR)
    print(f"Found {len(imgs)} images in {SRC_DIR}")

    for src in tqdm(imgs):
        try:
            img = read_bgr(src)
            # Save a resized 512x512 "baseline" copy too (optional, helps standardize)
            base_out = dst_path(src)
            ensure_parent(base_out)
            base = cv2.resize(img, (512,512), interpolation=cv2.INTER_AREA)
            cv2.imwrite(base_out, base, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Generate AUG_PER_IMAGE variants
            for k in range(AUG_PER_IMAGE):
                aug = transform(image=to_rgb(img))["image"]
                aug = to_bgr(cv2.resize(aug, (512,512), interpolation=cv2.INTER_AREA))
                stem, ext = os.path.splitext(dst_path(src))
                out_path = f"{stem}__aug{k+1}{ext}"
                cv2.imwrite(out_path, aug, [cv2.IMWRITE_JPEG_QUALITY, 95])

        except Exception as e:
            print(f"[WARN] {src}: {e}")

if __name__ == "__main__":
    main()