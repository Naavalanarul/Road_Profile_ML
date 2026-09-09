"""
train.py

Two-phase transfer learning for the 4-class road severity classifier.

Phase 1: freeze the pretrained backbone, train only the new head.
Phase 2: unfreeze the last few backbone blocks, fine-tune at a low LR.

Uses class-weighted cross-entropy so the (likely rare) Pothole class
isn't drowned out.

Usage:
    python train.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

import config
from dataset import RoadSeverityDataset


class OrdinalEMDLoss(nn.Module):
    """
    Squared Earth Mover's Distance loss for ordinal classification.

    The 4 classes (Smooth < Normal < Rough < Pothole) sit on an ordered
    severity scale, not independent categories -- standard cross-entropy
    penalizes "Smooth predicted as Pothole" the same as "Smooth predicted
    as Normal", which is the wrong error signal for suspension tuning.

    This loss compares the cumulative distribution of the predicted
    softmax against the cumulative distribution of the one-hot target,
    so predictions that are further away on the severity scale are
    penalized more heavily. Reference: Hou et al., "Squared Earth
    Mover's Distance-based Loss for Training Deep Neural Networks", 2016.

    class_weights (optional) still up-weights the rare Pothole class on
    top of the ordinal penalty.
    """

    def __init__(self, num_classes, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        pred_cdf = torch.cumsum(probs, dim=1)

        target_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
        target_cdf = torch.cumsum(target_onehot, dim=1)

        per_sample_loss = torch.sum((pred_cdf - target_cdf) ** 2, dim=1)

        if self.class_weights is not None:
            sample_weights = self.class_weights[targets]
            per_sample_loss = per_sample_loss * sample_weights

        return per_sample_loss.mean()


def build_model():
    if config.BACKBONE == "mobilenet_v2":
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, len(config.CLASSES)),
        )
        feature_blocks = backbone.features
    elif config.BACKBONE == "efficientnet_b0":
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, len(config.CLASSES)),
        )
        feature_blocks = backbone.features
    else:
        raise ValueError(f"Unknown backbone: {config.BACKBONE}")

    return backbone, feature_blocks


def set_backbone_trainable(feature_blocks, trainable: bool, unfreeze_last_n: int = 0):
    blocks = list(feature_blocks)
    for i, block in enumerate(blocks):
        make_trainable = trainable or (i >= len(blocks) - unfreeze_last_n)
        for p in block.parameters():
            p.requires_grad = make_trainable


def compute_class_weights(counts):
    total = sum(counts)
    n_classes = len(counts)
    weights = [total / (n_classes * c) if c > 0 else 0.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    per_class_correct = [0] * len(config.CLASSES)
    per_class_total = [0] * len(config.CLASSES)

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for c in range(len(config.CLASSES)):
                mask = labels == c
                per_class_total[c] += mask.sum().item()
                per_class_correct[c] += (preds[mask] == c).sum().item()

    avg_loss = total_loss / total
    acc = correct / total
    per_class_acc = [
        (per_class_correct[c] / per_class_total[c]) if per_class_total[c] > 0 else float("nan")
        for c in range(len(config.CLASSES))
    ]
    return avg_loss, acc, per_class_acc


def main():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = RoadSeverityDataset(config.TRAIN_CSV, train=True)
    val_ds = RoadSeverityDataset(config.VAL_CSV, train=False)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

    class_weights = compute_class_weights(train_ds.class_counts()).to(device)
    print(f"Class weights ({config.CLASSES}): {class_weights.tolist()}")

    if config.USE_ORDINAL_LOSS:
        criterion = OrdinalEMDLoss(len(config.CLASSES), class_weights=class_weights)
        print("Using ordinal EMD loss (severity-distance-aware)")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    model, feature_blocks = build_model()
    model = model.to(device)

    best_val_acc = 0.0

    # ---- Phase 1: train head only ----
    print("\n=== Phase 1: training head, backbone frozen ===")
    set_backbone_trainable(feature_blocks, trainable=False)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config.PHASE1_LR
    )

    for epoch in range(config.PHASE1_EPOCHS):
        train_loss, train_acc, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_per_class = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"[P1 {epoch+1}/{config.PHASE1_EPOCHS}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.CHECKPOINT_PATH)

    # ---- Phase 2: fine-tune top backbone blocks ----
    print("\n=== Phase 2: fine-tuning top backbone blocks ===")
    set_backbone_trainable(feature_blocks, trainable=False, unfreeze_last_n=config.UNFREEZE_LAST_N_BLOCKS)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config.PHASE2_LR
    )

    for epoch in range(config.PHASE2_EPOCHS):
        train_loss, train_acc, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_per_class = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        per_class_str = ", ".join(
            f"{c}={a:.2f}" if a == a else f"{c}=n/a"
            for c, a in zip(config.CLASSES, val_per_class)
        )
        print(f"[P2 {epoch+1}/{config.PHASE2_EPOCHS}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | per-class: {per_class_str}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.CHECKPOINT_PATH)

    print(f"\nBest val acc: {best_val_acc:.3f}. Model saved to {config.CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()