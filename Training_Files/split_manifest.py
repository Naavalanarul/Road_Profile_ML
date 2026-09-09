import csv
import random

import config


def main():
    with open(config.MANIFEST_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    random.seed(config.RANDOM_SEED)
    random.shuffle(rows)

    n = len(rows)
    n_train = int(n * config.SPLIT_RATIOS["train"])
    n_val = int(n * config.SPLIT_RATIOS["val"])

    train_rows = rows[:n_train]
    val_rows = rows[n_train:n_train + n_val]
    test_rows = rows[n_train + n_val:]

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