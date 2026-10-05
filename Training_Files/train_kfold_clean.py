"""
train_kfold_clean.py - leak-free version of train_kfold.py.

Differences from train_kfold.py:
  * Folds are drawn from train.csv + val.csv ONLY. test.csv is never loaded
    until the very end, and is never part of any training fold.
  * Stratifies on (class, country) so each fold keeps both mixes.
  * Saves out-of-fold (OOF) logits for every dev image. Each OOF logit comes
    from a model that never saw that image -> honest teacher targets for
    distill_export.py and an honest CV estimate.
  * Reports per-class PRECISION as well as recall.

Needs train_kfold.py and eval_utils.py in the same folder.
Usage: python train_kfold_clean.py [--k 5] [--p1-epochs 4] [--p2-epochs 14]
Outputs go to ./kfold_clean/
"""
import argparse, functools, os
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

print = functools.partial(print, flush=True)

import config
import train_kfold as tk
from eval_utils import report

OUT = "kfold_clean"


def resolve_path(p):
    if os.path.exists(p):
        return os.path.abspath(p)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(script_dir, p)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    return os.path.abspath(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--p1-epochs", type=int, default=4)
    ap.add_argument("--p2-epochs", type=int, default=14)
    ap.add_argument("--no-tta", dest="tta", action="store_false")
    ap.set_defaults(tta=True)
    args = ap.parse_args()

    train_csv = resolve_path(config.TRAIN_CSV)
    val_csv = resolve_path(config.VAL_CSV)
    test_csv = resolve_path(config.TEST_CSV)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, OUT)
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(out_dir)  # fold checkpoints are written here, not over your old ones

    device = tk.get_acceleration_device()
    print(f"Using device: {device}")

    dev_items = tk.preload_dataset_into_memory(train_csv) + tk.preload_dataset_into_memory(val_csv)
    n_train = sum(1 for _ in open(train_csv)) - 1
    labels = np.array([it[1] for it in dev_items])
    countries = sorted({it[2] for it in dev_items})
    strat = np.array([labels[i] * 10 + countries.index(dev_items[i][2]) for i in range(len(dev_items))])
    print(f"Dev set (train+val): {len(dev_items)} | test.csv is NOT used for folds")

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=config.RANDOM_SEED)
    oof = np.zeros((len(dev_items), tk.NUM_CLASSES), dtype=np.float32)
    fold_weights = []
    for f, (tr, va) in enumerate(skf.split(np.zeros(len(dev_items)), strat)):
        tr_items = [dev_items[i] for i in tr]
        va_items = [dev_items[i] for i in va]
        _, w, _ = tk.train_single_fold(f, args.k, tr_items, va_items, device, args)
        fold_weights.append(w)
        m, _ = tk.build_model()
        m.load_state_dict(w)
        m.to(device)
        lg, _ = tk.predict_with_tta(m, va_items, device, args.batch_size) if args.tta else \
            tk.predict_loader(m, torch.utils.data.DataLoader(
                tk.FastMemoryRoadDataset(va_items, is_train=False), batch_size=args.batch_size), device)
        oof[va] = lg.numpy()
        del m

    np.save("oof_logits.npy", oof)
    np.save("dev_labels.npy", labels)
    np.save("n_train.npy", np.array([n_train]))  # first n_train dev rows == train.csv
    t_oof = torch.tensor(oof)
    t_lab = torch.tensor(labels)
    report(t_oof.argmax(1), t_lab, "OUT-OF-FOLD on dev (honest CV)")

    # Single final look at the untouched test split
    test_items = tk.preload_dataset_into_memory(test_csv)
    logits = []
    for w in fold_weights:
        m, _ = tk.build_model()
        m.load_state_dict(w)
        m.to(device)
        lg, test_labels = tk.predict_with_tta(m, test_items, device, args.batch_size) if args.tta else \
            tk.predict_loader(m, torch.utils.data.DataLoader(
                tk.FastMemoryRoadDataset(test_items, is_train=False), batch_size=args.batch_size), device)
        logits.append(lg)
        del m
    ens = torch.stack(logits).mean(0)
    np.save("ensemble_test_logits.npy", ens.numpy())
    report(ens.argmax(1), test_labels, f"TEST - {args.k}-fold ensemble (leak-free)")


if __name__ == "__main__":
    main()
