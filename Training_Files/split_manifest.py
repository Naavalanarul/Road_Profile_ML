"""
split_manifest.py

Splits manifest.csv into train/val/test.

SPLIT_MODE = "block" (default): images are grouped into blocks of
SPLIT_BLOCK_SIZE consecutive indices per country (India_000000..000099 is one
group, etc.) and whole groups are assigned to a split. RDD2022 images come
from driving video, so neighbouring frames are near-duplicates; a plain random
split puts them in both train and val and inflates validation accuracy.

This assumes the numbering follows capture order. If it does not, the split
is still valid, just no better than random.
"""

import csv
import os
import random
from collections import defaultdict

import config


def group_key(row):
    stem = os.path.splitext(os.path.basename(row["image_path"]))[0]
    try:
        idx = int(stem.rsplit("_", 1)[1])
        return (row["country"], idx // config.SPLIT_BLOCK_SIZE)
    except (IndexError, ValueError):
        return (row["country"], stem)  # unparseable name -> its own group


def block_split(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[group_key(r)].append(r)

    keys = list(groups.keys())
    random.shuffle(keys)

    n = len(rows)
    targets = {k: config.SPLIT_RATIOS[k] * n for k in ("train", "val", "test")}
    sizes = {k: 0 for k in targets}
    out = {k: [] for k in targets}

    # Greedy: give each group to the split that is furthest below its target.
    for key in keys:
        split = max(targets, key=lambda k: (targets[k] - sizes[k]) / targets[k])
        out[split].extend(groups[key])
        sizes[split] += len(groups[key])

    for k in out:
        random.shuffle(out[k])
    return out["train"], out["val"], out["test"]


def random_split(rows):
    random.shuffle(rows)
    n = len(rows)
    n_train = int(n * config.SPLIT_RATIOS["train"])
    n_val = int(n * config.SPLIT_RATIOS["val"])
    return rows[:n_train], rows[n_train:n_train + n_val], rows[n_train + n_val:]


def main():
    with open(config.MANIFEST_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    random.seed(config.RANDOM_SEED)
    if config.SPLIT_MODE == "block":
        train_rows, val_rows, test_rows = block_split(rows)
    else:
        train_rows, val_rows, test_rows = random_split(rows)
    print(f"Split mode: {config.SPLIT_MODE}")

    for name, split_rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        path = getattr(config, f"{name.upper()}_CSV")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(split_rows)

        counts = {c: 0 for c in config.CLASSES}
        for r in split_rows:
            counts[r["label"]] += 1
        print(f"{name}: {len(split_rows)} images")
        for c in config.CLASSES:
            pct = 100.0 * counts[c] / len(split_rows) if split_rows else 0.0
            print(f"  {c:8s}: {counts[c]:6d}  ({pct:5.1f}%)")

    print(f"\nWrote {config.TRAIN_CSV}, {config.VAL_CSV}, {config.TEST_CSV}")


if __name__ == "__main__":
    main()
