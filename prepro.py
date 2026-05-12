"""
Waste Classification – Image-Only Pipeline
Loads → Validates → Resizes → Splits → Augments → Saves

OUTPUT:
    data_processed/
        train/  glass/  metal/  paper/  plastic/
        val/    ...
        test/   ...
"""

import random
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "raw_root":    "waste_dataset",
    "output_root": "data_processed",
    "img_size":    (224, 224),
    "aug_copies":  3,
    "random_seed": 42,
    "train_ratio": 0.70,
    "val_ratio":   0.15,
    "test_ratio":  0.15,
}
random.seed(CONFIG["random_seed"])

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Helpers ───────────────────────────────────────────────────────────────────
def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except:
        return False

def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")

def resize_image(img: Image.Image) -> Image.Image:
    return img.resize(CONFIG["img_size"], Image.LANCZOS)

# ── Augmentation ──────────────────────────────────────────────────────────────
def augment_pil(img: Image.Image) -> list[Image.Image]:
    return [
        ImageOps.mirror(img),
        img.rotate(random.uniform(-20, 20), fillcolor=(128, 128, 128)),
        ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.4)),
        img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5))),
    ][:CONFIG["aug_copies"]]

# ── Split ─────────────────────────────────────────────────────────────────────
def split_paths(paths: list) -> dict:
    """Shuffle then cut into train / val / test."""
    paths = paths.copy()
    random.shuffle(paths)
    n        = len(paths)
    n_train  = int(n * CONFIG["train_ratio"])
    n_val    = int(n * CONFIG["val_ratio"])
    return {
        "train": paths[:n_train],
        "val":   paths[n_train : n_train + n_val],
        "test":  paths[n_train + n_val :],
    }

# ── Core Pipeline ─────────────────────────────────────────────────────────────
def process_all(src_root: Path, dst_root: Path, augment: bool = True):
    """
    For every class-folder inside src_root:
      1. Collect valid images
      2. Split → train / val / test
      3. Resize & save under dst_root/<split>/<class>/
      4. Augment train images only
    """
    class_dirs = [d for d in sorted(src_root.iterdir()) if d.is_dir()]
    if not class_dirs:
        print("No class folders found!"); return

    total_saved = total_aug = total_skip = 0

    for cls_dir in class_dirs:
        cls_name = cls_dir.name

        # ── collect valid images ──────────────────────────────────────────────
        all_imgs = [p for p in sorted(cls_dir.iterdir())
                    if p.suffix.lower() in VALID_EXT and is_valid_image(p)]
        invalid  = [p for p in cls_dir.iterdir()
                    if p.suffix.lower() in VALID_EXT and not is_valid_image(p)]

        print(f"\n[{cls_name}]  total={len(all_imgs)}  corrupt={len(invalid)}")
        total_skip += len(invalid)

        splits = split_paths(all_imgs)   # {"train": [...], "val": [...], "test": [...]}

        for split_name, img_paths in splits.items():
            out_dir = dst_root / split_name / cls_name
            out_dir.mkdir(parents=True, exist_ok=True)

            for img_path in img_paths:
                img      = resize_image(load_image(img_path))
                out_path = out_dir / img_path.name
                img.save(out_path, "JPEG", quality=95)
                total_saved += 1

                # augment TRAIN only
                if augment and split_name == "train":
                    for i, aug_img in enumerate(augment_pil(img)):
                        aug_path = out_dir / f"{img_path.stem}_aug{i}.jpg"
                        aug_img.save(aug_path, "JPEG", quality=90)
                        total_aug += 1

            print(f"  {split_name:5s} → {len(img_paths)} images", end="")
            if split_name == "train" and augment:
                print(f"  (+{len(img_paths) * CONFIG['aug_copies']} aug)", end="")
            print()

    print(f"\n{'='*40}")
    print(f"  Saved    : {total_saved}")
    print(f"  Augmented: {total_aug}")
    print(f"  Skipped  : {total_skip}")
    print(f"{'='*40}")

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    process_all(
        src_root = Path(CONFIG["raw_root"]),
        dst_root = Path(CONFIG["output_root"]),
        augment  = True,
    )
    print("\nDone.")