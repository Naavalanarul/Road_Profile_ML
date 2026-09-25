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

# "block": consecutive image indices (per country) are kept together so
#          near-duplicate neighbouring video frames don't leak across
#          train/val/test. Assumes filenames like India_000123.jpg are in
#          capture order (verify on your data).
# "random": old behaviour (leaky if frames come from videos).
SPLIT_MODE = "block"
SPLIT_BLOCK_SIZE = 100

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

# Severity score is now measured INSIDE the ROI (the strip the model sees),
# normalised by ROI area. Boxes are clipped to the ROI; boxes outside it are
# ignored. Because the denominator shrank by ROI_BOTTOM_FRACTION, the old
# 0.02 (full-image units) becomes roughly 0.02 / 0.35 ~= 0.057. Re-run
# build_manifest.py and check the printed class distribution; tune this
# number so Normal/Rough are sensible for your data.
NORMAL_MAX_SCORE = 0.05

# Pothole label needs at least this much (clipped) pothole area inside the
# ROI, as a fraction of ROI area. 0.0 = any visible D40 box counts (old
# behaviour). Try 0.003 to stop tiny, hard-to-see potholes forcing "Pothole".
MIN_POTHOLE_ROI_AREA = 0.0

# Boxes with less than this many pixels of height left after clipping are
# dropped (a sliver of a box at the crop edge is not visible damage).
MIN_CLIPPED_BOX_PX = 4

# Images whose score is within +-AMBIGUITY_MARGIN of NORMAL_MAX_SCORE are
# flagged `ambiguous` in the manifest; evaluation also reports accuracy on
# the non-ambiguous subset so you can see how much label noise costs you.
AMBIGUITY_MARGIN = 0.015

ROI_BOTTOM_FRACTION = 0.35

# Model input (height, width). The ROI strip is wide, so keep its aspect
# ratio instead of squashing it into a square. Cracks are thin: more
# resolution helps. 192x512 is a good start; try 224x640 if memory allows.
INPUT_HEIGHT = 192
INPUT_WIDTH = 512
IMG_SIZE = (INPUT_HEIGHT, INPUT_WIDTH)   # kept for backwards compatibility

BATCH_SIZE = 32
NUM_WORKERS = 4
BACKBONE = "mobilenet_v2"   # or "efficientnet_b0"
PRETRAINED = True           # set False for offline smoke tests
PHASE1_EPOCHS = 5           # frozen backbone, train head only
PHASE2_EPOCHS = 30          # fine-tune backbone
PHASE1_LR = 1e-3
PHASE2_HEAD_LR = 3e-4
PHASE2_BACKBONE_LR = 1e-4   # was 1e-5, which barely moved the weights
WEIGHT_DECAY = 1e-4
UNFREEZE_LAST_N_BLOCKS = 8  # None = unfreeze the whole backbone
LABEL_SMOOTHING = 0.05
EMD_LAMBDA = 0.5            # loss = CE + EMD_LAMBDA * EMD (when ordinal loss on)
USE_BALANCED_SAMPLER = False  # if True, class-balanced sampling and the CE
                              # class weights are turned off (no double count)
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
