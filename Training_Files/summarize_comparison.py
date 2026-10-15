"""
summarize_comparison.py

Reads the evaluation outputs written by compare_models.py and prints one
markdown comparison table (also saved to runs/compare/COMPARISON.md).

    python summarize_comparison.py
"""

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "runs", "compare")
MODELS = [("v8s", "YOLOv8s (current)"), ("y11s", "YOLO11s"), ("v8m", "YOLOv8m")]


def read(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read().replace("\r", "\n")


def box_row(text, cls):
    """Return (P, R, mAP50) for a class row of a `yolo val` output."""
    m = re.search(rf"^\s+{cls}\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+\s*$", text, re.M)
    return tuple(float(x) for x in m.groups()) if m else None


def speed(text):
    m = re.search(r"Speed: [\d.]+ms preprocess, ([\d.]+)ms inference", text)
    return float(m.group(1)) if m else None


def rsi_all(text):
    """Accuracy and per-class recall from the 'test - all' block of evaluate_rsi.py."""
    block = text.split("=== test - RDD2022 only")[0]
    acc = re.search(r"accuracy ([\d.]+)", block)
    rec = {c: float(r) for c, r in re.findall(r"^(Smooth|Normal|Rough|Pothole)\s+(?:\d+\s+){4}([\d.]+)", block, re.M)}
    prec = re.search(r"precision Pothole\s+([\d.]+)", block)
    return (float(acc.group(1)) if acc else None, rec, float(prec.group(1)) if prec else None)


def fmt(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def main():
    rows = []
    for key, label in MODELS:
        bt, bk, rt = read(f"{key}_box_test.txt"), read(f"{key}_box_kaggle.txt"), read(f"{key}_rsi_test.txt")
        all_t, pot_t = box_row(bt, "all"), box_row(bt, "pothole")
        pot_k = box_row(bk, "pothole")
        acc, rec, pprec = rsi_all(rt)
        rows.append((label,
                     fmt(all_t[2] if all_t else None), fmt(pot_t[1] if pot_t else None),
                     fmt(pot_k[1] if pot_k else None),
                     fmt(acc), fmt(rec.get("Pothole")), fmt(pprec), fmt(rec.get("Normal")), fmt(rec.get("Rough")),
                     fmt(speed(bt), 1)))

    head = ("Model", "Box mAP50 (test)", "Pothole box recall (test)", "Pothole box recall (Kaggle)",
            "Road-class accuracy", "Pothole img recall", "Pothole img precision", "Normal recall",
            "Rough recall", "Inference ms/img (RTX 5060)")
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    table = "\n".join(lines)
    print(table)
    with open(os.path.join(OUT, "COMPARISON.md"), "w", encoding="utf-8") as f:
        f.write("# Model comparison (same merged dataset, same rule from rsi_detector.py: CONF=0.15, union crack area >= 0.07; images with open manholes excluded)\n\n")
        f.write(table + "\n")


if __name__ == "__main__":
    main()
