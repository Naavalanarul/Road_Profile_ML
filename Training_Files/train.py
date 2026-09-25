"""
train.py

Two-phase transfer learning for the 4-class road severity classifier.

Phase 1: frozen backbone, train the new head.
Phase 2: unfreeze (part of) the backbone, fine-tune with discriminative LRs
         (small for backbone, larger for head) and a cosine schedule.

Changes vs. the original:
  * frozen BatchNorm layers are kept in eval mode (before, their running
    statistics kept drifting while the weights were "frozen")
  * loss = cross-entropy (class weights, label smoothing) + lambda * EMD,
    with no double class-weighting inside the EMD term
  * optional class-balanced sampler
  * Phase 2 LR raised (backbone 1e-4, head 3e-4), AdamW, cosine decay, 30 epochs
  * richer evaluation: within-1 accuracy, MAE, balanced accuracy, confusion
    matrix, per-country accuracy, accuracy on non-ambiguous labels
  * best checkpoint is re-evaluated on the TEST split at the end

Usage:
    python train.py                 # train, then evaluate best model on test
    python train.py --test-only     # just evaluate the saved checkpoint
    python train.py --test-only --split val
"""

import argparse
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models

import config
from dataset import RoadSeverityDataset

NUM_CLASSES = len(config.CLASSES)


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------
class OrdinalEMDLoss(nn.Module):
    """
    Squared Earth Mover's Distance loss for ordinal classification
    (Hou et al., 2016): compares the CDF of the predicted softmax with the
    CDF of the one-hot target, so predictions far from the truth on the
    severity scale are penalised more.
    """

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        pred_cdf = torch.cumsum(probs, dim=1)
        target_cdf = torch.cumsum(
            F.one_hot(targets, num_classes=self.num_classes).float(), dim=1)
        return torch.sum((pred_cdf - target_cdf) ** 2, dim=1).mean()


class CombinedLoss(nn.Module):
    """CE (optionally class-weighted, label-smoothed) + lambda * EMD."""

    def __init__(self, class_weights=None, use_emd=True,
                 emd_lambda=None, label_smoothing=None):
        super().__init__()
        # Read config values at call time (getattr with a fallback), not as
        # function-default expressions. Defaults are evaluated once when the
        # class is defined, so if this class is defined in a notebook cell
        # that ran before config.py had these attributes, the stale default
        # would stick even after config.py is fixed and reloaded.
        if emd_lambda is None:
            emd_lambda = getattr(config, "EMD_LAMBDA", 0.5)
        if label_smoothing is None:
            label_smoothing = getattr(config, "LABEL_SMOOTHING", 0.0)
        self.ce = nn.CrossEntropyLoss(weight=class_weights,
                                      label_smoothing=label_smoothing)
        self.emd = OrdinalEMDLoss(NUM_CLASSES) if use_emd else None
        self.emd_lambda = emd_lambda

    def forward(self, logits, targets):
        loss = self.ce(logits, targets)
        if self.emd is not None:
            loss = loss + self.emd_lambda * self.emd(logits, targets)
        return loss


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_model():
    if config.BACKBONE == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if config.PRETRAINED else None
        backbone = models.mobilenet_v2(weights=weights)
    elif config.BACKBONE == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if config.PRETRAINED else None
        backbone = models.efficientnet_b0(weights=weights)
    else:
        raise ValueError(f"Unknown backbone: {config.BACKBONE}")

    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, NUM_CLASSES),
    )
    return backbone, backbone.features


def set_backbone_trainable(feature_blocks, trainable, unfreeze_last_n=0):
    blocks = list(feature_blocks)
    for i, block in enumerate(blocks):
        make_trainable = trainable or (i >= len(blocks) - unfreeze_last_n)
        for p in block.parameters():
            p.requires_grad = make_trainable


def set_train_mode(model, feature_blocks):
    """model.train(), but keep fully frozen backbone blocks (and therefore
    their BatchNorm running stats) in eval mode."""
    model.train()
    for block in feature_blocks:
        if not any(p.requires_grad for p in block.parameters()):
            block.eval()


def compute_class_weights(counts):
    total = sum(counts)
    n_classes = len(counts)
    weights = [total / (n_classes * c) if c > 0 else 0.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


# --------------------------------------------------------------------------
# Train / evaluate
# --------------------------------------------------------------------------
def train_one_epoch(model, feature_blocks, loader, criterion, optimizer, device):
    set_train_mode(model, feature_blocks)
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        all_logits.append(model(images.to(device)).cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def compute_metrics(logits, labels, criterion, countries, ambiguous):
    preds = logits.argmax(1)
    n = labels.numel()
    m = {"loss": criterion(logits, labels).item(), "acc": (preds == labels).float().mean().item()}

    diff = (preds - labels).abs()
    m["within1"] = (diff <= 1).float().mean().item()
    m["mae"] = diff.float().mean().item()

    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for t, p in zip(labels.tolist(), preds.tolist()):
        conf[t, p] += 1
    m["confusion"] = conf
    per_class = []
    for c in range(NUM_CLASSES):
        row = conf[c].sum().item()
        per_class.append(conf[c, c].item() / row if row else float("nan"))
    m["per_class"] = per_class
    valid = [a for a in per_class if a == a]
    m["balanced_acc"] = sum(valid) / len(valid) if valid else float("nan")

    by_country = defaultdict(lambda: [0, 0])
    for c, ok in zip(countries, (preds == labels).tolist()):
        by_country[c][0] += int(ok)
        by_country[c][1] += 1
    m["per_country"] = {c: a / t for c, (a, t) in by_country.items()}

    clean = torch.tensor([a == 0 for a in ambiguous])
    m["clean_acc"] = (preds[clean] == labels[clean]).float().mean().item() if clean.any() else float("nan")
    m["n_clean"] = int(clean.sum())
    m["n"] = n
    return m


def short_line(m):
    pc = ", ".join(f"{c[:4]}={a:.2f}" for c, a in zip(config.CLASSES, m["per_class"]))
    return (f"val_loss={m['loss']:.4f} acc={m['acc']:.3f} within1={m['within1']:.3f} "
            f"bal_acc={m['balanced_acc']:.3f} | {pc}")


def full_report(m, title):
    print(f"\n===== {title} (n={m['n']}) =====")
    print(f"accuracy            : {m['acc']:.3f}")
    print(f"within-1 accuracy   : {m['within1']:.3f}")
    print(f"mean abs. error     : {m['mae']:.3f} classes")
    print(f"balanced accuracy   : {m['balanced_acc']:.3f}")
    print(f"acc, non-ambiguous  : {m['clean_acc']:.3f} (n={m['n_clean']})")
    print("per-class recall    : " + ", ".join(
        f"{c}={a:.2f}" for c, a in zip(config.CLASSES, m["per_class"])))
    print("per-country accuracy: " + ", ".join(
        f"{c}={a:.2f}" for c, a in sorted(m["per_country"].items())))
    print("confusion matrix (rows=true, cols=pred):")
    print("            " + " ".join(f"{c[:7]:>8s}" for c in config.CLASSES))
    for i, c in enumerate(config.CLASSES):
        print(f"  {c[:9]:9s} " + " ".join(f"{v:8d}" for v in m["confusion"][i].tolist()))


def make_loader(ds, shuffle, sampler=None):
    kw = dict(batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
    if config.NUM_WORKERS > 0:
        kw["persistent_workers"] = True
    if sampler is not None:
        return DataLoader(ds, sampler=sampler, **kw)
    return DataLoader(ds, shuffle=shuffle, **kw)


def evaluate_split(model, csv_path, criterion, device, title):
    ds = RoadSeverityDataset(csv_path, train=False)
    loader = make_loader(ds, shuffle=False)
    logits, labels = predict(model, loader, device)
    m = compute_metrics(logits, labels, criterion, ds.countries(), ds.ambiguous_mask())
    full_report(m, title)
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-only", action="store_true",
                        help="skip training, evaluate the saved checkpoint")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    model, feature_blocks = build_model()
    model = model.to(device)

    if args.test_only:
        model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location=device))
        csv_path = config.TEST_CSV if args.split == "test" else config.VAL_CSV
        evaluate_split(model, csv_path, CombinedLoss(None, config.USE_ORDINAL_LOSS),
                       device, f"{args.split.upper()} (checkpoint)")
        return

    train_ds = RoadSeverityDataset(config.TRAIN_CSV, train=True)
    val_ds = RoadSeverityDataset(config.VAL_CSV, train=False)
    val_loader = make_loader(val_ds, shuffle=False)

    counts = train_ds.class_counts()
    class_weights = compute_class_weights(counts)
    print(f"Train class counts ({config.CLASSES}): {counts}")

    if config.USE_BALANCED_SAMPLER:
        per_class_w = [1.0 / c if c else 0.0 for c in counts]
        sample_w = [per_class_w[y] for y in train_ds.labels()]
        sampler = WeightedRandomSampler(sample_w, num_samples=len(train_ds), replacement=True)
        train_loader = make_loader(train_ds, shuffle=False, sampler=sampler)
        ce_weights = None
        print("Using class-balanced sampler (CE class weights disabled)")
    else:
        train_loader = make_loader(train_ds, shuffle=True)
        ce_weights = class_weights.to(device)
        print(f"CE class weights: {[round(w, 3) for w in class_weights.tolist()]}")

    criterion = CombinedLoss(ce_weights, use_emd=config.USE_ORDINAL_LOSS)
    # unweighted criterion for comparable validation loss numbers
    eval_criterion = CombinedLoss(None, use_emd=config.USE_ORDINAL_LOSS)
    print(f"Loss: CE(label_smoothing={config.LABEL_SMOOTHING})"
          + (f" + {config.EMD_LAMBDA} * EMD" if config.USE_ORDINAL_LOSS else ""))

    best_val_acc = 0.0

    def run_val():
        logits, labels = predict(model, val_loader, device)
        return compute_metrics(logits.cpu(), labels, eval_criterion,
                               val_ds.countries(), val_ds.ambiguous_mask())

    # ---- Phase 1: head only ----
    print("\n=== Phase 1: training head, backbone frozen ===")
    set_backbone_trainable(feature_blocks, trainable=False)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.PHASE1_LR, weight_decay=config.WEIGHT_DECAY)

    for epoch in range(config.PHASE1_EPOCHS):
        tr_loss, tr_acc = train_one_epoch(model, feature_blocks, train_loader,
                                          criterion, optimizer, device)
        m = run_val()
        print(f"[P1 {epoch+1}/{config.PHASE1_EPOCHS}] train_loss={tr_loss:.4f} "
              f"train_acc={tr_acc:.3f} | {short_line(m)}")
        if m["acc"] > best_val_acc:
            best_val_acc = m["acc"]
            torch.save(model.state_dict(), config.CHECKPOINT_PATH)

    # ---- Phase 2: fine-tune backbone ----
    print("\n=== Phase 2: fine-tuning backbone ===")
    if config.UNFREEZE_LAST_N_BLOCKS is None:
        set_backbone_trainable(feature_blocks, trainable=True)
    else:
        set_backbone_trainable(feature_blocks, trainable=False,
                               unfreeze_last_n=config.UNFREEZE_LAST_N_BLOCKS)

    backbone_params = [p for b in feature_blocks for p in b.parameters() if p.requires_grad]
    head_params = list(model.classifier.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": config.PHASE2_BACKBONE_LR},
         {"params": head_params, "lr": config.PHASE2_HEAD_LR}],
        weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.PHASE2_EPOCHS, eta_min=1e-6)

    for epoch in range(config.PHASE2_EPOCHS):
        tr_loss, tr_acc = train_one_epoch(model, feature_blocks, train_loader,
                                          criterion, optimizer, device)
        scheduler.step()
        m = run_val()
        print(f"[P2 {epoch+1}/{config.PHASE2_EPOCHS}] train_loss={tr_loss:.4f} "
              f"train_acc={tr_acc:.3f} | {short_line(m)}")
        if m["acc"] > best_val_acc:
            best_val_acc = m["acc"]
            torch.save(model.state_dict(), config.CHECKPOINT_PATH)

    print(f"\nBest val acc: {best_val_acc:.3f}. Model saved to {config.CHECKPOINT_PATH}")

    # ---- Final evaluation of the best checkpoint ----
    model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location=device))
    evaluate_split(model, config.VAL_CSV, eval_criterion, device, "VAL (best checkpoint)")
    evaluate_split(model, config.TEST_CSV, eval_criterion, device, "TEST (best checkpoint)")


if __name__ == "__main__":
    main()
