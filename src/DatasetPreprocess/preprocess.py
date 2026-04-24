import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Settings ──────────────────────────────────────────────
INPUT_FOLDER  = "dataset"
OUTPUT_FOLDER = "processed_dataset"
IMAGE_SIZE    = (64, 64)
SPLITS        = ["train", "val", "test"]

# ── Helper Functions ───────────────────────────────────────

def preprocess_image(img):
    """Apply all preprocessing steps to one image."""
    # 1. BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Resize to 64x64
    img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    # 3. Gaussian Blur (noise reduction)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # 4. CLAHE contrast fix (works on L channel in LAB)
    lab     = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l       = clahe.apply(l)
    img     = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

    # Note: Normalization (img/255.0) removed from here 
    # and should be done during model training for better storage quality.
    return img


def augment_image(img):
    """Apply random augmentation to raw image. Input must be uint8."""
    # Random horizontal flip
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)

    # Random rotation ±15°
    angle = np.random.uniform(-15, 15)
    # Get center for rotation based on original image size
    h, w = img.shape[:2]
    M    = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    img  = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Random brightness
    brightness = np.random.uniform(0.75, 1.25)
    img = np.clip(img.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

    return img


# ── Main Processing Loop ───────────────────────────────────

records = []

for split in SPLITS:
    input_split  = os.path.join(INPUT_FOLDER,  split)
    output_split = os.path.join(OUTPUT_FOLDER, split)

    if not os.path.exists(input_split):
        print(f"Skipping '{split}' — folder not found.")
        continue

    classes = sorted(os.listdir(input_split))
    print(f"\nProcessing: {split.upper()} ({len(classes)} classes)")

    for class_name in tqdm(classes, desc=split):
        class_in  = os.path.join(input_split,  class_name)
        class_out = os.path.join(output_split, class_name)

        if not os.path.isdir(class_in):
            continue

        os.makedirs(class_out, exist_ok=True)

        for filename in os.listdir(class_in):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".ppm", ".bmp")):
                continue

            img_path = os.path.join(class_in, filename)
            img_raw  = cv2.imread(img_path)

            if img_raw is None:
                print(f"  Warning: could not read {img_path}")
                continue

            # --- Step 1: Preprocess Original Image ---
            processed_orig = preprocess_image(img_raw)
            save_name = os.path.splitext(filename)[0] + ".png"
            save_path = os.path.join(class_out, save_name)
            
            # Save back as BGR for imwrite
            cv2.imwrite(save_path, cv2.cvtColor(processed_orig, cv2.COLOR_RGB2BGR))

            records.append({
                "split":     split,
                "class":     class_name,
                "filename":  save_name,
                "augmented": False,
            })

            # --- Step 2: Augmentation — ONLY for train ---
            if split == "train":
                for i in range(1): 
                    # Augment the RAW image first
                    aug_raw = augment_image(img_raw)
                    # Then Preprocess the augmented result
                    processed_aug = preprocess_image(aug_raw)
                    
                    aug_name = os.path.splitext(filename)[0] + f"_aug{i}.png"
                    aug_path = os.path.join(class_out, aug_name)
                    
                    # Save back as BGR for imwrite
                    cv2.imwrite(aug_path, cv2.cvtColor(processed_aug, cv2.COLOR_RGB2BGR))

                    records.append({
                        "split":     split,
                        "class":     class_name,
                        "filename":  aug_name,
                        "augmented": True,
                    })

# ── Save CSV ───────────────────────────────────────────────
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
csv_path = os.path.join(OUTPUT_FOLDER, "processed_metadata.csv")
pd.DataFrame(records).to_csv(csv_path, index=False)

# ── Verify Output ──────────────────────────────────────────
print("\nVerifying output images")
errors = 0

for split in SPLITS:
    split_dir = os.path.join(OUTPUT_FOLDER, split)
    if not os.path.exists(split_dir):
        continue
    for root, _, files in os.walk(split_dir):
        for f in files:
            if not f.endswith(".png"):
                continue
            path = os.path.join(root, f)
            img  = cv2.imread(path)
            if img is None:
                print(f"  Corrupt: {path}")
                errors += 1
            elif img.shape[:2] != IMAGE_SIZE:
                print(f"  Wrong size {img.shape[:2]}: {path}")
                errors += 1

if errors == 0:
    print("All images OK — 64x64, no corruption.")
else:
    print(f"{errors} issues found. Check above.")

print(f"\nDone! Saved to: {OUTPUT_FOLDER}/")
print(f"CSV saved to:   {csv_path}")