"""
Waste Classification Dataset - Preprocessing Pipeline
Classes: glass | metal | paper | plastic

FOLDER STRUCTURE EXPECTED:
    data2/
        glass/    0.jpg, 1.jpg, 2.jpg ...
        metal/    7.jpg, 8.jpg, 9.jpg ...
        paper/   25.jpg, 26.jpg, 27.jpg ...
        plastic/ 18.jpg, 19.jpg, 20.jpg ...

HOW TO RUN:
    pip install Pillow numpy pandas matplotlib scikit-learn
    python waste_preprocessing.py
"""

import os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    import torch, torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

CONFIG = {
    "raw_root": "final_data", "output_root": "data_processed",
    "csv_output": "final_data.csv", "report_dir": "reports",
    "classes": ["glass", "metal", "paper", "plastic"],
    "img_size": (224, 224), "img_channels": "RGB",
    "train_ratio": 0.70, "val_ratio": 0.15, "test_ratio": 0.15,
    "random_seed": 42,
    "norm_mean": [0.485, 0.456, 0.406],
    "norm_std":  [0.229, 0.224, 0.225],
    "augment": True, "aug_copies": 3,
}
CLASSES = CONFIG["classes"]; IMG_SIZE = CONFIG["img_size"]; SEED = CONFIG["random_seed"]
random.seed(SEED); np.random.seed(SEED)

# ── Helpers ───────────────────────────────────────────────────────────────────
def is_valid_image(path):
    try:
        with Image.open(path) as img: img.verify()
        return True
    except: return False

def load_image(path):
    return Image.open(path).convert(CONFIG["img_channels"])

def resize_image(img, size=IMG_SIZE):
    return img.resize(size, Image.LANCZOS)

def normalize_array(arr):
    arr = arr.astype(np.float32) / 255.0
    return (arr - np.array(CONFIG["norm_mean"], np.float32)) / np.array(CONFIG["norm_std"], np.float32)

# ── Step 1: Build Manifest ────────────────────────────────────────────────────
def build_manifest(raw_root):
    records = []; le = LabelEncoder(); le.fit(CLASSES)
    for cls in CLASSES:
        cls_dir = Path(raw_root) / cls
        if not cls_dir.exists(): continue
        for img_path in sorted(p for p in cls_dir.glob("*")
                               if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}):
            corrupt = not is_valid_image(str(img_path))
            w = h = kb = None
            if not corrupt:
                with Image.open(str(img_path)) as im: w, h = im.size
                kb = round(img_path.stat().st_size / 1024, 2)
            records.append({"image_path": str(img_path), "label": cls,
                "label_id": int(le.transform([cls])[0]),
                "width": w, "height": h, "file_size_kb": kb, "corrupt": corrupt})
    df = pd.DataFrame(records)
    print(f"\n[Step 1] {len(df)} files found\n{df['label'].value_counts().to_string()}")
    return df

# ── Step 2: Clean ─────────────────────────────────────────────────────────────
def clean_manifest(df):
    n = len(df)
    df = df[~df["corrupt"]].drop_duplicates("image_path")
    df = df[(df["width"] >= 32) & (df["height"] >= 32)]
    print(f"\n[Step 2] Removed {n - len(df)} bad files. Remaining: {len(df)}")
    return df.reset_index(drop=True)

# ── Step 3: Split ─────────────────────────────────────────────────────────────
def split_dataset(df):
    df = df.copy(); df["split"] = ""
    train_val, test = train_test_split(df, test_size=CONFIG["test_ratio"],
                                       stratify=df["label"], random_state=SEED)
    val_adj = CONFIG["val_ratio"] / (1 - CONFIG["test_ratio"])
    train, val = train_test_split(train_val, test_size=val_adj,
                                  stratify=train_val["label"], random_state=SEED)
    df.loc[train.index,"split"]="train"; df.loc[val.index,"split"]="val"; df.loc[test.index,"split"]="test"
    print(f"\n[Step 3] train={len(train)}, val={len(val)}, test={len(test)}")
    return df

# ── Step 4: Preprocess & Save ─────────────────────────────────────────────────
def preprocess_and_save(df, out_root):
    Path(out_root).mkdir(parents=True, exist_ok=True)
    df = df.copy(); df["processed_path"] = ""
    for idx, row in df.iterrows():
        out_dir = Path(out_root) / row["split"] / row["label"]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / Path(row["image_path"]).name
        resize_image(load_image(row["image_path"])).save(str(out_path), "JPEG", quality=95)
        df.at[idx, "processed_path"] = str(out_path)
    print(f"\n[Step 4] Saved {len(df)} resized images → {out_root}/")
    return df

# ── Step 5: Augmentation ──────────────────────────────────────────────────────
def augment_pil(img):
    return [
        ImageOps.mirror(img),
        img.rotate(random.uniform(-20,20), fillcolor=(128,128,128)),
        ImageEnhance.Brightness(img).enhance(random.uniform(0.6,1.4)),
        img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5,1.5))),
    ][:CONFIG["aug_copies"]]

def augment_dataset(df, out_root):
    new_rows = []
    for _, row in df[df["split"]=="train"].iterrows():
        img = load_image(row["processed_path"])
        for i, aug_img in enumerate(augment_pil(img)):
            out_path = Path(row["processed_path"]).parent / f"{Path(row['processed_path']).stem}_aug{i}.jpg"
            aug_img.save(str(out_path), "JPEG", quality=90)
            nr = row.to_dict(); nr["processed_path"] = nr["image_path"] = str(out_path)
            new_rows.append(nr)
    if new_rows:
        aug_df = pd.DataFrame(new_rows); aug_df["augmented"] = True
        df["augmented"] = False
        result = pd.concat([df, aug_df], ignore_index=True)
    else:
        df["augmented"] = False; result = df
    print(f"\n[Step 5] Added {len(new_rows)} augmented images")
    return result

# ── Step 6: EDA Plots ─────────────────────────────────────────────────────────
def run_eda_plots(df, report_dir):
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    orig = df[~df.get("augmented", pd.Series([False]*len(df))).fillna(False)]
    colors = ["#4CAF50","#2196F3","#FF9800","#9C27B0"]

    # 1. Class distribution
    counts = orig["label"].value_counts()
    fig, ax = plt.subplots(figsize=(8,5))
    bars = ax.bar(counts.index, counts.values, color=colors[:len(counts)], edgecolor="white")
    ax.bar_label(bars, fmt="%d", padding=4, fontsize=11, fontweight="bold")
    ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Class"); ax.set_ylabel("Count"); ax.set_ylim(0, counts.max()*1.15)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{report_dir}/class_distribution.png", dpi=150); plt.close()

    # 2. Split balance
    fig, axes = plt.subplots(1,3, figsize=(14,4))
    for ax, split in zip(axes, ["train","val","test"]):
        subset = orig[orig["split"]==split]; sub = subset["label"].value_counts()
        ax.pie(sub.values, labels=sub.index, autopct="%1.1f%%", colors=colors[:len(sub)], startangle=140)
        ax.set_title(f"{split.capitalize()} (n={len(subset)})", fontweight="bold")
    plt.suptitle("Class Balance per Split", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{report_dir}/split_distribution.png", dpi=150); plt.close()

    # 3. Image size
    clean = orig.dropna(subset=["width","height"])
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,4))
    ax1.hist(clean["width"],  bins=30, color="#2196F3", edgecolor="white"); ax1.set_title("Width (px)")
    ax2.hist(clean["height"], bins=30, color="#4CAF50", edgecolor="white"); ax2.set_title("Height (px)")
    for ax in (ax1,ax2): ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{report_dir}/image_size_distribution.png", dpi=150); plt.close()

    # 4. Augmentation examples
    col = "processed_path" if "processed_path" in df.columns else "image_path"
    rows = orig[orig["split"]=="train"]
    if not rows.empty:
        orig_img = load_image(rows.iloc[0][col]); aug_imgs = augment_pil(orig_img)
        all_imgs = [orig_img]+aug_imgs; titles = ["Original"]+[f"Aug #{i+1}" for i in range(len(aug_imgs))]
        fig, axes = plt.subplots(1,len(all_imgs), figsize=(4*len(all_imgs),4))
        for ax, img, t in zip(axes, all_imgs, titles):
            ax.imshow(img); ax.set_title(t, fontweight="bold"); ax.axis("off")
        plt.suptitle("Augmentation Examples", fontsize=14, fontweight="bold")
        plt.tight_layout(); plt.savefig(f"{report_dir}/augmentation_examples.png", dpi=150); plt.close()

    print(f"\n[Step 6] Saved 4 EDA plots → {report_dir}/")

# ── Step 7: Pixel Stats ───────────────────────────────────────────────────────
def compute_pixel_stats(df, split="train", n_sample=200):
    subset = df[df["split"]==split]
    if len(subset) > n_sample: subset = subset.sample(n_sample, random_state=SEED)
    col = "processed_path" if "processed_path" in df.columns else "image_path"
    arrays = []
    for _, row in subset.iterrows():
        try: arrays.append(np.array(load_image(row[col]).resize((64,64)))/255.0)
        except: pass
    if not arrays: return {"mean_RGB": CONFIG["norm_mean"], "std_RGB": CONFIG["norm_std"]}
    stack = np.stack(arrays)
    stats = {"mean_RGB": [round(v,4) for v in stack.mean(axis=(0,1,2)).tolist()],
             "std_RGB":  [round(v,4) for v in stack.std(axis=(0,1,2)).tolist()]}
    print(f"\n[Step 7] mean={stats['mean_RGB']}  std={stats['std_RGB']}")
    return stats

# ── PyTorch Transforms ────────────────────────────────────────────────────────
def get_torch_transforms(split, mean=None, std=None):
    if not TORCH_AVAILABLE: return None
    mean = mean or CONFIG["norm_mean"]; std = std or CONFIG["norm_std"]
    if split == "train":
        return T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(0.5),
                          T.RandomVerticalFlip(0.1), T.ColorJitter(0.3,0.3,0.2,0.1),
                          T.RandomRotation(20), T.ToTensor(), T.Normalize(mean, std)])
    return T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean, std)])

# ── Main ──────────────────────────────────────────────────────────────────────
def run_pipeline():
    print("="*60+"\n  WASTE CLASSIFICATION PREPROCESSING\n"+"="*60)
    df = build_manifest(CONFIG["raw_root"])
    df.to_csv("manifest_raw.csv", index=False)
    df = clean_manifest(df)
    df = split_dataset(df)
    df = preprocess_and_save(df, CONFIG["output_root"])
    if CONFIG["augment"]: df = augment_dataset(df, CONFIG["output_root"])
    else: df["augmented"] = False
    run_eda_plots(df, CONFIG["report_dir"])
    stats = compute_pixel_stats(df)
    df.to_csv(CONFIG["csv_output"], index=False)
    print("\nDone! Update CONFIG['norm_mean'] and CONFIG['norm_std'] with the pixel stats above.")
    return df, stats

if __name__ == "__main__":
    df, stats = run_pipeline()