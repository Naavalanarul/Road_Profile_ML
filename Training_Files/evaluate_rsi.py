"""
evaluate_rsi.py

Image-level evaluation of the detector -> road class pipeline.
Ground-truth class = rsi_detector.classify() applied to the label boxes;
predicted class    = rsi_detector.classify() applied to YOLO detections.

    python evaluate_rsi.py --weights ../runs/detect/merged_v8s/weights/best.pt --split test
    python evaluate_rsi.py --weights ... --split val --sweep     # tune CONF / ROUGH_CRACK_AREA
"""

import argparse
import collections
import glob
import os

from rsi_detector import CLASSES, CONF, ROUGH_CRACK_AREA, classify

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Dataset", "merged_yolo")


def load_gt(label_path):
    boxes = []
    if os.path.isfile(label_path):
        for line in open(label_path):
            p = line.split()
            if len(p) == 5:
                boxes.append((int(p[0]), 1.0, float(p[3]), float(p[4])))
    return boxes


def report(rows, title):
    """rows: list of (source, gt_class, pred_class)."""
    cm = collections.Counter((g, p) for _, g, p in rows)
    n = len(rows)
    acc = sum(cm[(c, c)] for c in CLASSES) / n if n else 0.0
    print(f"\n=== {title}: {n} images, accuracy {acc:.3f} ===")
    print("GT \\ Pred   " + "".join(f"{c:>9s}" for c in CLASSES) + "   recall")
    for g in CLASSES:
        tot = sum(cm[(g, p)] for p in CLASSES)
        rec = cm[(g, g)] / tot if tot else float("nan")
        print(f"{g:11s}" + "".join(f"{cm[(g, p)]:9d}" for p in CLASSES) + f"   {rec:.3f}")
    for p in CLASSES:
        tot = sum(cm[(g, p)] for g in CLASSES)
        prec = cm[(p, p)] / tot if tot else float("nan")
        print(f"  precision {p:8s} {prec:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO
    from rsi_detector import boxes_from_result

    model = YOLO(args.weights)
    images = sorted(glob.glob(os.path.join(DATA, "images", args.split, "*.jpg")))

    cached = []  # (source, gt_class, predicted boxes at low conf)
    # A list source is predicted as one batch, so feed it in chunks.
    results = (r for i in range(0, len(images), 16)
               for r in model.predict(images[i:i + 16], conf=0.05, verbose=False))
    for result in results:
        name = os.path.basename(result.path)
        label = os.path.join(DATA, "labels", args.split, os.path.splitext(name)[0] + ".txt")
        source = "kaggle" if name.startswith("kaggle_") else "rdd"
        cached.append((source, classify(load_gt(label)), boxes_from_result(result)))

    if args.sweep:
        print("conf  rough_area  accuracy  pothole_recall  pothole_precision  rough_recall")
        for conf in (0.15, 0.2, 0.25, 0.3, 0.4):
            for area in (0.02, 0.05, 0.08, 0.12):
                rows = [(s, g, classify(b, conf, area)) for s, g, b in cached]
                acc = sum(g == p for _, g, p in rows) / len(rows)
                gp = [p for _, g, p in rows if g == "Pothole"]
                pp = [g for _, g, p in rows if p == "Pothole"]
                gr = [p for _, g, p in rows if g == "Rough"]
                print(f"{conf:4.2f}  {area:10.2f}  {acc:8.3f}  "
                      f"{sum(p == 'Pothole' for p in gp) / max(1, len(gp)):14.3f}  "
                      f"{sum(g == 'Pothole' for g in pp) / max(1, len(pp)):17.3f}  "
                      f"{sum(p == 'Rough' for p in gr) / max(1, len(gr)):12.3f}")
        return

    rows = [(s, g, classify(b)) for s, g, b in cached]
    print(f"Using CONF={CONF}, ROUGH_CRACK_AREA={ROUGH_CRACK_AREA}")
    report(rows, f"{args.split} - all")
    report([r for r in rows if r[0] == "rdd"], f"{args.split} - RDD2022 only")
    report([r for r in rows if r[0] == "kaggle"], f"{args.split} - Kaggle only")


if __name__ == "__main__":
    main()
