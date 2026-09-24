"""
Central configuration for the Road Severity CNN pipeline.
Edit the paths and thresholds here — nothing else needs touching
for a first run.
"""

import os

RDD2022_ROOT = os.environ.get("RDD2022_ROOT", "/Users/naavalanarul/Documents/Projects/Road_Profile_ML/Dataset/21431547/RDD2022")

COUNTRIES = ["India", "Japan", "United_States", "Czech", "Norway"]

MANIFEST_CSV = "manifest.csv"

TRAIN_CSV = "train.csv"
VAL_CSV = "val.csv"
TEST_CSV = "test.csv"

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42

CLASSES = ["Smooth", "Normal", "Rough", "Pothole"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

POTHOLE_CODES = {"D40"}
SEVERE_CRACK_CODES = {"D20"}
MINOR_CRACK_CODES = {"D00", "D01", "D10", "D11"}
IGNORED_CODES = {"D43", "D44", "D50"}

DAMAGE_WEIGHTS = {
    "pothole": 3.0,
    "severe_crack": 1.5,
    "minor_crack": 1.0,
}

NORMAL_MAX_SCORE = 0.02

ROI_BOTTOM_FRACTION = 0.35

IMG_SIZE = 224
BATCH_SIZE = 32
BACKBONE = "mobilenet_v2"   # or "efficientnet_b0"
PHASE1_EPOCHS = 8           # frozen backbone, train head only
PHASE2_EPOCHS = 12          # fine-tune top backbone layers
PHASE1_LR = 1e-3
PHASE2_LR = 1e-5
UNFREEZE_LAST_N_BLOCKS = 3
CHECKPOINT_PATH = "road_severity_model.pt"

def get_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_device()

USE_ORDINAL_LOSS = True

SMOOTHING_WINDOW = 8          # frames averaged for de-escalation decisions
CONFIDENCE_THRESHOLD = 0.65   # min probability to allow any switch
MIN_DWELL_FRAMES = 10         # min frames before a de-escalation is allowed

SEVERITY_INDEX = {
    "Smooth": 0,
    "Normal": 1,
    "Rough": 2,
    "Pothole": 3,
}
