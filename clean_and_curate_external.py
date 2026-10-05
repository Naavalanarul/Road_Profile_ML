"""
clean_and_curate_external.py

Curation pipeline for CRACK500 and Pothole-600 external datasets.
Outputs clean JPEG images into Dataset/Cleaned_External/{Normal,Rough,Pothole}/.

Rules:
1. Recursively purge .DS_Store, Thumbs.db, __MACOSX, and unneeded .txt files.
2. Pothole-600 Sanitization:
   - Filter out disparity/transformed/depth/label frames.
   - Keep only valid RGB road images from rgb/ subdirectories.
   - Save to Dataset/Cleaned_External/Pothole/pothole600_XXXXX.jpg.
3. CRACK500 Quantization:
   - Pair RGB image with corresponding binary crack mask.
   - ratio = crack_pixels / total_pixels.
   - ratio < 0.025 -> Dataset/Cleaned_External/Normal/crack500_normal_XXXXX.jpg.
   - ratio >= 0.025 -> Dataset/Cleaned_External/Rough/crack500_rough_XXXXX.jpg.
"""

import os
import shutil
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
CLEANED_DIR = os.path.join(DATASET_DIR, "Cleaned_External")
CRACK500_DIR = os.path.join(DATASET_DIR, "CRACK500")
POTHOLE600_DIR = os.path.join(DATASET_DIR, "Pothole600")

OUT_DIRS = {
    "Normal": os.path.join(CLEANED_DIR, "Normal"),
    "Rough": os.path.join(CLEANED_DIR, "Rough"),
    "Pothole": os.path.join(CLEANED_DIR, "Pothole"),
}

DISCARD_KEYWORDS = ["disp", "transformed", "trans_disp", "depth", "label", "tdisp"]


def purge_system_artifacts():
    print("\n--- Purging System Artifacts ---")
    purged_files = 0
    purged_dirs = 0

    for root_target in [DATASET_DIR]:
        for root, dirs, files in os.walk(root_target, topdown=False):
            for d in list(dirs):
                if d in ["__MACOSX", ".ipynb_checkpoints"]:
                    full_d = os.path.join(root, d)
                    shutil.rmtree(full_d, ignore_errors=True)
                    dirs.remove(d)
                    purged_dirs += 1

            for f in files:
                f_lower = f.lower()
                if f_lower in [".ds_store", "thumbs.db"] or f_lower.endswith(".txt"):
                    # Check if unneeded metadata file
                    full_f = os.path.join(root, f)
                    try:
                        os.remove(full_f)
                        purged_files += 1
                    except OSError:
                        pass

    print(f"Purged {purged_files} artifact files and {purged_dirs} artifact directories.")


def process_pothole600():
    print("\n--- Sanitizing Pothole-600 ---")
    pothole_out = OUT_DIRS["Pothole"]
    os.makedirs(pothole_out, exist_ok=True)

    saved_count = 0
    skipped_count = 0

    for root, dirs, files in sorted(os.walk(POTHOLE600_DIR)):
        rel_root = os.path.relpath(root, POTHOLE600_DIR).lower()
        # Discard disparity, transformed, or label directories
        if any(kw in rel_root for kw in DISCARD_KEYWORDS):
            continue

        # Only process RGB image directories
        if not rel_root.endswith("rgb"):
            continue

        for f in sorted(files):
            f_lower = f.lower()
            if not f_lower.endswith((".png", ".jpg", ".jpeg")):
                continue
            if any(kw in f_lower for kw in DISCARD_KEYWORDS):
                continue

            src_path = os.path.join(root, f)
            try:
                with Image.open(src_path) as img:
                    w, h = img.size
                    if w < 100 or h < 100:
                        skipped_count += 1
                        continue
                    rgb_img = img.convert("RGB")
                    dst_filename = f"pothole600_{saved_count:05d}.jpg"
                    dst_path = os.path.join(pothole_out, dst_filename)
                    rgb_img.save(dst_path, "JPEG", quality=95)
                    saved_count += 1
            except Exception as e:
                print(f"Error processing {src_path}: {e}")
                skipped_count += 1

    print(f"Pothole-600 complete: saved {saved_count} images to {pothole_out} (skipped {skipped_count}).")
    return saved_count


def process_crack500():
    print("\n--- Quantizing CRACK500 ---")
    os.makedirs(OUT_DIRS["Normal"], exist_ok=True)
    os.makedirs(OUT_DIRS["Rough"], exist_ok=True)

    normal_count = 0
    rough_count = 0
    no_mask_count = 0

    # Locate all images in train/val/test images directories
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(CRACK500_DIR, split, "images")
        mask_dir = os.path.join(CRACK500_DIR, split, "masks")

        if not os.path.exists(img_dir):
            continue

        for f in sorted(os.listdir(img_dir)):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            src_img_path = os.path.join(img_dir, f)
            src_mask_path = os.path.join(mask_dir, f)

            try:
                with Image.open(src_img_path) as img:
                    w, h = img.size
                    if w < 100 or h < 100:
                        continue
                    rgb_img = img.convert("RGB")

                ratio = 0.0
                if os.path.exists(src_mask_path):
                    with Image.open(src_mask_path) as mask_img:
                        mask_arr = np.array(mask_img.convert("L"))
                        crack_pixels = np.sum(mask_arr > 127)
                        total_pixels = mask_arr.size
                        ratio = crack_pixels / total_pixels
                else:
                    no_mask_count += 1

                # Quantize: < 0.025 -> Normal, >= 0.025 -> Rough
                if ratio < 0.025:
                    dst_name = f"crack500_normal_{normal_count:05d}.jpg"
                    dst_path = os.path.join(OUT_DIRS["Normal"], dst_name)
                    rgb_img.save(dst_path, "JPEG", quality=95)
                    normal_count += 1
                else:
                    dst_name = f"crack500_rough_{rough_count:05d}.jpg"
                    dst_path = os.path.join(OUT_DIRS["Rough"], dst_name)
                    rgb_img.save(dst_path, "JPEG", quality=95)
                    rough_count += 1

            except Exception as e:
                print(f"Error processing {src_img_path}: {e}")

    print(f"CRACK500 complete:")
    print(f"  Normal (< 2.5% crack area): {normal_count} images -> {OUT_DIRS['Normal']}")
    print(f"  Rough  (>= 2.5% crack area): {rough_count} images -> {OUT_DIRS['Rough']}")
    if no_mask_count:
        print(f"  Images without masks (assigned Normal): {no_mask_count}")

    return normal_count, rough_count


def main():
    print(f"{'='*70}\n  CLEAN AND CURATE EXTERNAL DATASETS\n{'='*70}")
    purge_system_artifacts()
    pothole_cnt = process_pothole600()
    normal_cnt, rough_cnt = process_crack500()

    print(f"\n{'='*70}")
    print("  EXTERNAL CURATION SUMMARY:")
    print(f"    Normal : {normal_cnt:5d}")
    print(f"    Rough  : {rough_cnt:5d}")
    print(f"    Pothole: {pothole_cnt:5d}")
    print(f"    Total  : {normal_cnt + rough_cnt + pothole_cnt:5d}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
