"""
inspect_raw_external.py

Standalone discovery script to inspect raw Dataset/CRACK500 and Dataset/Pothole600.
Reports:
  - File counts and extensions
  - Disparity/depth/mask keywords
  - Image resolutions and unreadable/corrupted files
DOES NOT ALTER OR DELETE ANY FILES.
"""

import os
from collections import Counter, defaultdict
from PIL import Image

DATASET_ROOTS = {
    "CRACK500": "Dataset/CRACK500",
    "Pothole600": "Dataset/Pothole600"
}

KEYWORDS = ["disp", "disparity", "transformed", "tdisp", "depth", "mask", "label"]

def inspect_dataset(name, root_path):
    print(f"\n{'='*70}")
    print(f"  INSPECTING DATASET: {name} ({root_path})")
    print(f"{'='*70}")

    if not os.path.exists(root_path):
        print(f"Error: {root_path} does not exist!")
        return

    ext_counts = Counter()
    keyword_matches = defaultdict(int)
    keyword_samples = defaultdict(list)
    resolutions = Counter()
    modes = Counter()
    corrupt_files = []
    total_files = 0
    image_count = 0

    for dirpath, dirnames, filenames in os.walk(root_path, followlinks=True):
        for f in filenames:
            total_files += 1
            full_path = os.path.join(dirpath, f)
            rel_path = os.path.relpath(full_path, root_path)
            ext = os.path.splitext(f)[1].lower() or "[no_ext]"
            ext_counts[ext] += 1

            # Keyword matching on relative path
            path_lower = rel_path.lower()
            for kw in KEYWORDS:
                if kw in path_lower:
                    keyword_matches[kw] += 1
                    if len(keyword_samples[kw]) < 3:
                        keyword_samples[kw].append(rel_path)

            # Check images
            if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]:
                image_count += 1
                try:
                    with Image.open(full_path) as img:
                        img.verify()  # verify integrity
                    with Image.open(full_path) as img:
                        resolutions[img.size] += 1
                        modes[img.mode] += 1
                except Exception as e:
                    corrupt_files.append((rel_path, str(e)))

    print(f"Total files found: {total_files}")
    print(f"\n--- File Extensions ---")
    for ext, count in ext_counts.most_common():
        print(f"  {ext:10s}: {count:6d}")

    print(f"\n--- Disparity / Depth / Mask Keyword Matches in Paths ---")
    for kw in KEYWORDS:
        count = keyword_matches[kw]
        print(f"  '{kw}': {count} files")
        if keyword_samples[kw]:
            for sample in keyword_samples[kw]:
                print(f"      sample: {sample}")

    print(f"\n--- Image Modes (Color Channels) ---")
    for mode, count in modes.most_common():
        print(f"  Mode {mode:6s}: {count:6d}")

    print(f"\n--- Image Resolutions (Top 10) ---")
    for res, count in resolutions.most_common(10):
        print(f"  {res[0]}x{res[1]}: {count:6d}")

    print(f"\n--- Corrupted / Unreadable Files ---")
    if corrupt_files:
        print(f"  WARNING: Found {len(corrupt_files)} corrupted files!")
        for path, err in corrupt_files[:10]:
            print(f"    {path}: {err}")
    else:
        print("  None found! All images readable and verified.")

def main():
    for name, root in DATASET_ROOTS.items():
        inspect_dataset(name, root)

if __name__ == "__main__":
    main()
