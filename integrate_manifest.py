"""
integrate_manifest.py

Integrates Dataset/Cleaned_External into manifest.csv and generates updated splits.

Steps:
1. Reads all images from Dataset/Cleaned_External/{Normal,Rough,Pothole}/.
2. Formats records to match manifest.csv schema:
   image_path,label,severity_score,num_boxes,country,label_full,ambiguous
   country="External_Curated", ambiguous=0
3. Appends records to manifest.csv (avoiding duplicates).
4. Runs split_manifest.py to produce updated train.csv, val.csv, test.csv.
5. Prints class distribution across all splits.
"""

import csv
import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANED_DIR = os.path.join(BASE_DIR, "Dataset", "Cleaned_External")
TRAINING_DIR = os.path.join(BASE_DIR, "Training_Files")
MANIFEST_PATH = os.path.join(TRAINING_DIR, "manifest.csv")


def integrate_external_records():
    print(f"\n--- Reading existing manifest from {MANIFEST_PATH} ---")
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(f"Manifest not found at {MANIFEST_PATH}")

    with open(MANIFEST_PATH, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        existing_rows = list(reader)

    print(f"Existing rows in manifest: {len(existing_rows)}")
    existing_paths = {r["image_path"] for r in existing_rows}

    new_rows = []
    classes = ["Normal", "Rough", "Pothole"]

    for cls_name in classes:
        cls_dir = os.path.join(CLEANED_DIR, cls_name)
        if not os.path.exists(cls_dir):
            continue

        for f in sorted(os.listdir(cls_dir)):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            abs_path = os.path.abspath(os.path.join(cls_dir, f))
            if abs_path in existing_paths:
                continue

            row = {
                "image_path": abs_path,
                "label": cls_name,
                "severity_score": "0.0",
                "num_boxes": "0",
                "country": "External_Curated",
                "label_full": cls_name,
                "ambiguous": "0"
            }
            new_rows.append(row)

    print(f"New external records to add: {len(new_rows)}")
    by_class = {}
    for r in new_rows:
        by_class[r["label"]] = by_class.get(r["label"], 0) + 1
    for c, cnt in sorted(by_class.items()):
        print(f"  {c:8s}: {cnt:5d}")

    all_rows = existing_rows + new_rows
    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Updated manifest saved: total {len(all_rows)} rows.")
    return len(new_rows)


def run_split_manifest():
    print("\n--- Running split_manifest.py ---")
    split_script = os.path.join(TRAINING_DIR, "split_manifest.py")
    result = subprocess.run(
        [sys.executable, "split_manifest.py"],
        cwd=TRAINING_DIR,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print("Errors / Warnings:\n", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"split_manifest.py failed with return code {result.returncode}")


def verify_splits():
    print("\n--- Verifying Updated Dataset Splits ---")
    for split_name in ["train", "val", "test"]:
        csv_path = os.path.join(TRAINING_DIR, f"{split_name}.csv")
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))

        counts = {}
        for r in rows:
            counts[r["label"]] = counts.get(r["label"], 0) + 1

        print(f"\n{split_name.upper()} ({len(rows)} images):")
        total = len(rows)
        for c in ["Smooth", "Normal", "Rough", "Pothole"]:
            cnt = counts.get(c, 0)
            pct = cnt / total * 100 if total else 0.0
            print(f"  {c:8s}: {cnt:6d} ({pct:5.1f}%)")


def main():
    print(f"{'='*70}\n  INTEGRATE EXTERNAL CURATED DATASET\n{'='*70}")
    integrate_external_records()
    run_split_manifest()
    verify_splits()


if __name__ == "__main__":
    main()
