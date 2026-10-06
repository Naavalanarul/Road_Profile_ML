"""
build_yolo_dataset.py

Merges RDD2022 (Pascal VOC xml) and the Kaggle pothole-cracks-and-openmanhole
set (YOLO txt) into one YOLO detection dataset with a shared class list:

    0 pothole          RDD D40          | Kaggle 0 pothole
    1 crack            RDD D00, D10     | Kaggle 1 cracks
    2 alligator_crack  RDD D20          | -
    3 open_manhole     -                | Kaggle 2 open_manhole

Other RDD codes (D01, D11, D43, D44 closed manhole cover, D50, Repair, ...)
are dropped. Images are hard-linked (same drive), not copied.

RDD splits are made in blocks of consecutive frames so near-duplicate dashcam
frames cannot land in both train and test. Kaggle train -> train, Kaggle
valid -> alternately val / test.

    python build_yolo_dataset.py
"""

import collections
import glob
import os
import random
import shutil
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RDD_ROOT = os.path.join(ROOT, "Dataset", "RDD2022")
KAGGLE_ROOT = os.path.join(ROOT, "pothole-cracks-and-openmanhole", "dataset", "dataset")
OUT = os.path.join(ROOT, "Dataset", "merged_yolo")

NAMES = ["pothole", "crack", "alligator_crack", "open_manhole"]
RDD_MAP = {"D40": 0, "D00": 1, "D10": 1, "D20": 2}
KAGGLE_MAP = {0: 0, 1: 1, 2: 3}

RDD_COUNTRIES = ["India", "Japan", "Czech", "United_States", "China_MotorBike"]
BLOCK = 50                      # consecutive frames kept together in one split
RDD_SPLIT = (0.8, 0.1, 0.1)     # train / val / test
BACKGROUND_FRACTION = 0.15      # max share of no-box images in each split
SEED = 42


def voc_to_yolo(xml_path, dropped):
    root = ET.parse(xml_path).getroot()
    w = float(root.find("size/width").text)
    h = float(root.find("size/height").text)
    lines = []
    for obj in root.findall("object"):
        code = obj.find("name").text.strip()
        if code not in RDD_MAP:
            dropped[code] += 1
            continue
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = max(0.0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(h, max(y1, y2))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        lines.append(f"{RDD_MAP[code]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                     f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return lines


def collect_rdd(dropped):
    """Return list of (split, img_path, label_lines, tag)."""
    rng = random.Random(SEED)
    items = []
    for country in RDD_COUNTRIES:
        imgs = sorted(glob.glob(os.path.join(RDD_ROOT, country, "train", "images", "*.jpg")))
        if not imgs:
            print(f"[warn] no RDD images for {country}")
            continue
        blocks = [imgs[i:i + BLOCK] for i in range(0, len(imgs), BLOCK)]
        rng.shuffle(blocks)
        n_train = round(len(blocks) * RDD_SPLIT[0])
        n_val = round(len(blocks) * RDD_SPLIT[1])
        for bi, block in enumerate(blocks):
            split = "train" if bi < n_train else "val" if bi < n_train + n_val else "test"
            for img in block:
                stem = os.path.splitext(os.path.basename(img))[0]
                xml = os.path.join(RDD_ROOT, country, "train", "annotations", "xmls", stem + ".xml")
                lines = voc_to_yolo(xml, dropped) if os.path.isfile(xml) else []
                items.append((split, img, lines, f"rdd_{country}"))
    return items


def collect_kaggle():
    items = []
    for src, splits in (("train", None), ("valid", ("val", "test"))):
        imgs = sorted(glob.glob(os.path.join(KAGGLE_ROOT, src, "images", "*.jpg")))
        for i, img in enumerate(imgs):
            stem = os.path.splitext(os.path.basename(img))[0]
            txt = os.path.join(KAGGLE_ROOT, src, "labels", stem + ".txt")
            lines = []
            if os.path.isfile(txt):
                for l in open(txt):
                    p = l.split()
                    if len(p) == 5:
                        lines.append(" ".join([str(KAGGLE_MAP[int(p[0])])] + p[1:]))
            split = "train" if splits is None else splits[i % 2]
            items.append((split, img, lines, "kaggle"))
    return items


def cap_background(items):
    """Keep all labelled images; keep background images up to BACKGROUND_FRACTION per split."""
    rng = random.Random(SEED)
    out = []
    for split in ("train", "val", "test"):
        pos = [it for it in items if it[0] == split and it[2]]
        neg = [it for it in items if it[0] == split and not it[2]]
        rng.shuffle(neg)
        max_neg = int(len(pos) * BACKGROUND_FRACTION / (1 - BACKGROUND_FRACTION))
        out += pos + neg[:max_neg]
    return out


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)

    dropped = collections.Counter()
    items = cap_background(collect_rdd(dropped) + collect_kaggle())

    stats = collections.defaultdict(collections.Counter)
    for split, img, lines, tag in items:
        name = f"{tag}_{os.path.basename(img)}"
        img_out = os.path.join(OUT, "images", split, name)
        lbl_out = os.path.join(OUT, "labels", split, os.path.splitext(name)[0] + ".txt")
        os.makedirs(os.path.dirname(img_out), exist_ok=True)
        os.makedirs(os.path.dirname(lbl_out), exist_ok=True)
        os.link(img, img_out)
        with open(lbl_out, "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

        stats[split]["images"] += 1
        stats[split]["background"] += not lines
        stats[split][f"images_{tag}"] += 1
        for l in lines:
            stats[split][NAMES[int(l.split()[0])]] += 1

    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write(f"path: {OUT.replace(os.sep, '/')}\n"
                "train: images/train\nval: images/val\ntest: images/test\n\n"
                f"nc: {len(NAMES)}\nnames: {NAMES}\n")

    for split in ("train", "val", "test"):
        print(f"\n{split}:")
        for k, v in sorted(stats[split].items()):
            print(f"  {k:28s} {v}")
    print(f"\nDropped RDD codes: {dict(dropped)}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
