"""
distill_export.py - distil the K-fold teacher into ONE MobileNetV2 for the Jetson Nano,
tune precision-oriented decision rules, and export to ONNX.

Pipeline (no test leakage):
  1. Student trains on train.csv using the OOF teacher logits from train_kfold_clean.py
     (soft targets, temperature T) + hard labels.
  2. val.csv is used ONLY to tune logit biases and the Pothole probability gate.
  3. test.csv is evaluated once at the end.
  4. Exports road_severity_student.onnx (normalisation baked in: input is float RGB 0-255,
     already cropped to the bottom ROI and resized to 192x512) and deploy_config.json.

Usage: python distill_export.py --pothole-precision 0.90
"""
import argparse, functools, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

print = functools.partial(print, flush=True)

import config
import train_kfold as tk
from eval_utils import report, tune_bias, pick_pothole_threshold, gate_pothole

KF = "kfold_clean"


def resolve_path(p):
    if os.path.exists(p):
        return os.path.abspath(p)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(script_dir, p)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    return os.path.abspath(p)


class DistillDS(Dataset):
    def __init__(self, items, teacher_logits):
        self.items, self.t = items, torch.tensor(teacher_logits)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        pil, label, _, _ = self.items[i]
        return tk.TRAIN_AUGMENTATION(pil), label, self.t[i]


class Deploy(nn.Module):
    """Takes float RGB in 0..255 (NCHW); normalises inside the graph."""
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)

    def forward(self, x):
        return self.m((x - self.mean) / self.std)


@torch.no_grad()
def logits_of(model, items, device, bs):
    ds = tk.FastMemoryRoadDataset(items, is_train=False)
    return tk.predict_loader(model, DataLoader(ds, batch_size=bs), device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.7, help="weight of the distillation term")
    ap.add_argument("--pothole-precision", type=float, default=0.90)
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    kf_dir = os.path.join(script_dir, KF)

    csvs = [resolve_path(p) for p in (config.TRAIN_CSV, config.VAL_CSV, config.TEST_CSV)]
    oof_path = os.path.join(kf_dir, "oof_logits.npy")
    n_train_path = os.path.join(kf_dir, "n_train.npy")

    if not os.path.exists(oof_path) or not os.path.exists(n_train_path):
        raise FileNotFoundError(f"Missing teacher outputs in {kf_dir}. Run train_kfold_clean.py first.")

    oof = np.load(oof_path)
    n_train = int(np.load(n_train_path)[0])
    os.makedirs(kf_dir, exist_ok=True)
    device = tk.get_acceleration_device()
    print(f"Using device: {device}")

    tr_items = tk.preload_dataset_into_memory(csvs[0])
    va_items = tk.preload_dataset_into_memory(csvs[1])
    assert len(tr_items) == n_train and len(tr_items) + len(va_items) == len(oof), "CSV order changed"

    model, _ = tk.build_model()
    model.to(device)
    counts = [sum(1 for it in tr_items if it[1] == c) for c in range(tk.NUM_CLASSES)]
    if config.USE_BALANCED_SAMPLER:
        per_class_w = [1.0 / c if c else 0.0 for c in counts]
        sample_w = [per_class_w[it[1]] for it in tr_items]
        sampler = torch.utils.data.WeightedRandomSampler(sample_w, num_samples=len(tr_items), replacement=True)
        loader = DataLoader(DistillDS(tr_items, oof[:n_train]), batch_size=args.batch_size, sampler=sampler)
        ce = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
        print("Using class-balanced sampler for student (CE class weights disabled)")
    else:
        loader = DataLoader(DistillDS(tr_items, oof[:n_train]), batch_size=args.batch_size, shuffle=True)
        ce = nn.CrossEntropyLoss(weight=tk.compute_smoothed_class_weights(counts).to(device),
                                 label_smoothing=config.LABEL_SMOOTHING)
    head = list(model.classifier.parameters())
    hid = {id(p) for p in head}
    body = [p for p in model.parameters() if id(p) not in hid]
    opt = torch.optim.AdamW([{"params": body, "lr": 2e-4}, {"params": head, "lr": 1e-3}],
                            weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for x, y, t in loader:
            x, y, t = x.to(device), y.to(device), t.to(device)
            opt.zero_grad()
            s = model(x)
            kd = F.kl_div(F.log_softmax(s / args.T, 1), F.softmax(t / args.T, 1),
                          reduction="batchmean") * (args.T ** 2)
            loss = args.alpha * kd + (1 - args.alpha) * ce(s, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * x.size(0)
        sched.step()
        print(f"[student {ep+1}/{args.epochs}] loss={tot/len(loader.dataset):.4f}")
    torch.save(model.state_dict(), os.path.join(kf_dir, "road_severity_student.pt"))

    # --- tune decision rules on val only ---
    v_lg, v_lab = logits_of(model, va_items, device, args.batch_size)
    report(v_lg.argmax(1), v_lab, "VAL student (raw)")
    bias, f1 = tune_bias(v_lg, v_lab)
    print(f"tuned biases {bias.tolist()} -> val macroF1 {f1*100:.1f}%")
    pick = pick_pothole_threshold(v_lg + bias, v_lab, args.pothole_precision)
    thr = pick[0] if pick else 0.0
    print("Pothole gate:", f"thr={pick[0]:.2f} prec={pick[1]*100:.1f}% recall={pick[2]*100:.1f}%" if pick
          else f"target precision {args.pothole_precision} not reachable on val; gate disabled")

    # --- single look at test ---
    te_items = tk.preload_dataset_into_memory(csvs[2])
    t_lg, t_lab = logits_of(model, te_items, device, args.batch_size)
    report(t_lg.argmax(1), t_lab, "TEST student (raw, single pass, no TTA)")
    report((t_lg + bias).argmax(1), t_lab, "TEST student + tuned biases")
    report(gate_pothole(t_lg + bias, thr), t_lab, "TEST student + biases + Pothole gate")
    ens_path = os.path.join(kf_dir, "ensemble_test_logits.npy")
    if os.path.exists(ens_path):
        report(torch.tensor(np.load(ens_path)).argmax(1), t_lab, "TEST teacher ensemble (10 passes/frame)")

    # --- export ---
    model.eval().cpu()
    dep = Deploy(model).eval()
    dummy = torch.rand(1, 3, config.INPUT_HEIGHT, config.INPUT_WIDTH) * 255
    kw = dict(input_names=["image"], output_names=["logits"], opset_version=11)
    onnx_path = os.path.join(kf_dir, "road_severity_student.onnx")
    try:
        torch.onnx.export(dep, dummy, onnx_path, dynamo=False, **kw)
    except Exception:
        torch.onnx.export(dep, dummy, onnx_path, **kw)
    deploy_cfg_path = os.path.join(kf_dir, "deploy_config.json")
    json.dump({"classes": config.CLASSES, "logit_bias": bias.tolist(), "pothole_threshold": thr,
               "input": "RGB float32 0-255 NCHW, bottom %.2f crop, resized to %dx%d (HxW), bilinear" % (
                   config.ROI_BOTTOM_FRACTION, config.INPUT_HEIGHT, config.INPUT_WIDTH)},
              open(deploy_cfg_path, "w"), indent=2)
    print(f"\nOn the Jetson:\n  /usr/src/tensorrt/bin/trtexec --onnx=road_severity_student.onnx "
          f"--fp16 --saveEngine=road_student_fp16.engine\n(INT8 gives no speed-up on the Nano's Maxwell GPU - stay on FP16.)")


if __name__ == "__main__":
    main()
