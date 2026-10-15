"""
compare_models.py

Trains YOLO11s and YOLOv8m on Dataset/merged_yolo with the same settings as
the existing YOLOv8s run (runs/detect/merged_v8s), then evaluates all three
the same way. Safe to re-run: finished runs are skipped and interrupted
runs resume from their last.pt.

    python compare_models.py

Outputs:
    runs/detect/merged_<name>/          training runs
    runs/compare/<name>_*.txt           evaluation outputs
    runs/compare/logs/*.log             raw logs
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV = os.path.join(ROOT, "pothole-cracks-and-openmanhole", ".venv", "Scripts")
YOLO = os.path.join(VENV, "yolo.exe")
PY = os.path.join(VENV, "python.exe")
DATA = os.path.join(ROOT, "Dataset", "merged_yolo", "data.yaml")
KAGGLE = os.path.join(ROOT, "Dataset", "merged_yolo", "kaggle_eval.yaml")
RUNS = os.path.join(ROOT, "runs", "detect")
OUT = os.path.join(ROOT, "runs", "compare")
LOGS = os.path.join(OUT, "logs")

# name, starting weights, batch. v8m uses batch 8 to fit in 8 GB of GPU memory;
# Ultralytics accumulates gradients to a nominal batch of 64 either way.
NEW_RUNS = [
    ("y11s", "yolo11s.pt", 16),
    ("v8m", "yolov8m.pt", 8),
]
ALL_MODELS = ["v8s"] + [n for n, _, _ in NEW_RUNS]
COMMON = ["epochs=60", "imgsz=640", "device=0", "patience=15", "workers=2"]


def run(cmd, log_name):
    log = os.path.join(LOGS, log_name)
    print(f">> {' '.join(cmd)}\n   log: {log}", flush=True)
    with open(log, "a", encoding="utf-8") as f:
        return subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode


def weights(name):
    return os.path.join(RUNS, f"merged_{name}", "weights")


def train(name, model, batch):
    done = os.path.join(RUNS, f"merged_{name}", "DONE")
    if os.path.exists(done):
        print(f"{name}: already trained, skipping", flush=True)
        return
    last = os.path.join(weights(name), "last.pt")
    if os.path.exists(last):
        cmd = [YOLO, "task=detect", "mode=train", "resume=True", f"model={last}", "workers=2"]
    else:
        cmd = [YOLO, "task=detect", "mode=train", f"model={model}", f"data={DATA}",
               f"batch={batch}", f"project={RUNS}", f"name=merged_{name}", "exist_ok=True"] + COMMON
    rc = run(cmd, f"train_{name}.log")
    if rc != 0:
        sys.exit(f"{name}: training failed (exit {rc}); re-run this script to resume")
    open(done, "w").close()


def evaluate(name):
    best = os.path.join(weights(name), "best.pt")
    ev = os.path.join(ROOT, "Training_Files", "evaluate_rsi.py")
    jobs = [
        ([YOLO, "task=detect", "mode=val", f"model={best}", f"data={DATA}", "split=test",
          "batch=8", "workers=0", f"project={OUT}", f"name={name}_box_test", "exist_ok=True"],
         f"{name}_box_test.txt"),
        ([YOLO, "task=detect", "mode=val", f"model={best}", f"data={KAGGLE}",
          "batch=8", "workers=0", f"project={OUT}", f"name={name}_box_kaggle", "exist_ok=True"],
         f"{name}_box_kaggle.txt"),
        ([PY, ev, "--weights", best, "--split", "test"], f"{name}_rsi_test.txt"),
        ([PY, ev, "--weights", best, "--split", "val", "--sweep"], f"{name}_rsi_val_sweep.txt"),
    ]
    for cmd, out in jobs:
        path = os.path.join(OUT, out)
        if os.path.exists(path):
            continue
        tmp = path + ".part"
        with open(tmp, "w", encoding="utf-8") as f:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        if rc == 0:
            os.replace(tmp, path)
        else:
            print(f"   eval failed ({rc}): see {tmp}", flush=True)


def main():
    os.makedirs(LOGS, exist_ok=True)
    for name, model, batch in NEW_RUNS:
        train(name, model, batch)
    for name in ALL_MODELS:
        print(f"evaluating {name}", flush=True)
        evaluate(name)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
