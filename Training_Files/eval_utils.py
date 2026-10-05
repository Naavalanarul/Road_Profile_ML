"""eval_utils.py - precision/recall, logit-bias tuning and a Pothole gate."""
import functools
import numpy as np
import torch
import config

print = functools.partial(print, flush=True)

N = len(config.CLASSES)


def confusion(preds, labels):
    preds = torch.as_tensor(preds).cpu().long()
    labels = torch.as_tensor(labels).cpu().long()
    return torch.bincount(labels * N + preds, minlength=N * N).view(N, N)


def prec_rec(preds, labels):
    c = confusion(preds, labels).double().numpy()
    tp = np.diag(c)
    return tp / np.maximum(c.sum(0), 1), tp / np.maximum(c.sum(1), 1)


def macro_f1(preds, labels):
    p, r = prec_rec(preds, labels)
    return float(np.mean(2 * p * r / np.maximum(p + r, 1e-9)))


def report(preds, labels, title):
    preds = torch.as_tensor(preds).cpu()
    labels = torch.as_tensor(labels).cpu()
    p, r = prec_rec(preds, labels)
    acc = (preds == labels).float().mean().item()
    print(f"\n--- {title} (n={len(labels)}) acc={acc*100:.1f}% macroF1={macro_f1(preds, labels)*100:.1f}% ---")
    for i, c in enumerate(config.CLASSES):
        print(f"  {c:8s} precision={p[i]*100:5.1f}%  recall={r[i]*100:5.1f}%")
    print("  confusion (rows=true, cols=pred):")
    cm = confusion(preds, labels)
    for i, c in enumerate(config.CLASSES):
        print(f"   {c[:7]:7s} " + " ".join(f"{v:6d}" for v in cm[i].tolist()))


def tune_bias(logits, labels, grid=np.linspace(-1.5, 1.5, 31), rounds=3):
    """Coordinate ascent on an additive per-class logit bias to maximise macro-F1."""
    logits = torch.as_tensor(logits).cpu()
    labels = torch.as_tensor(labels).cpu()
    b = torch.zeros(N, dtype=logits.dtype)
    best = macro_f1((logits + b).argmax(1), labels)
    for _ in range(rounds):
        for c in range(N):
            for v in grid:
                t = b.clone()
                t[c] = float(v)
                s = macro_f1((logits + t).argmax(1), labels)
                if s > best + 1e-6:
                    best, b = s, t
    return b, best


def gate_pothole(logits, thr):
    """Only report Pothole if its probability >= thr; otherwise fall back to the best other class."""
    logits = torch.as_tensor(logits).cpu()
    probs = torch.softmax(logits, 1)
    pred = probs.argmax(1)
    m = (pred == N - 1) & (probs[:, N - 1] < thr)
    if m.any():
        pred[m] = probs[m][:, : N - 1].argmax(1)
    return pred


def pick_pothole_threshold(logits, labels, target_precision):
    """Smallest threshold whose Pothole precision >= target. Returns None if unreachable."""
    logits = torch.as_tensor(logits).cpu()
    labels = torch.as_tensor(labels).cpu()
    for thr in np.arange(0.30, 0.99, 0.01):
        p, r = prec_rec(gate_pothole(logits, thr), labels)
        if p[N - 1] >= target_precision:
            return float(thr), float(p[N - 1]), float(r[N - 1])
    return None
