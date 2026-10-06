# Road Perception: Merged Dataset, YOLO Detector and Road Severity Index

**Date:** 6 October 2026
**Scope:** the road-perception (CNN) part of the *Edge AI-based Adaptive Suspension System* project only. The STSMC controller, Simulink model and Jetson hardware were not touched.

---

## 1. Summary

- **Goal:** combine several road-damage datasets, train one model, and turn its output into the 4 road classes (**Smooth, Normal, Rough, Pothole**) and the **Road Severity Index (RSI)** from the proposal.
- **Approach chosen:** one **YOLOv8s object detector** for **potholes, cracks and alligator cracks**, trained on **RDD2022 + a Kaggle pothole & crack dataset**. Each frame's detections are converted into a road class and RSI by a simple rule.
- **Main results:**

| What is measured | Result |
|---|---|
| Road-class accuracy, test set (2,378 images) | **0.828** (0.808 before the Normal/Rough fix in section 7.5) |
| Normal-road recall, test set | **0.71** (0.56 before the fix) |
| Pothole recall, image level (test set) | **0.86**, from 0.84 on RDD2022 dashcam images to 0.92 on Kaggle images |
| Pothole precision, image level (test set) | **0.86** |
| Pothole recall per box, on the old Kaggle validation images | **0.59** (old model: 0.55) |
| Detector mAP50 (pothole, crack, alligator crack), merged validation set | **0.66** |
| 3 test videos | All ran correctly; potholes and cracks found; a few brief false potholes (section 9) |

---

## 2. Starting point

| Item | State before this work |
|---|---|
| Teammate's model (`Training_Files/train.py`) | MobileNetV2 4-class image classifier trained on RDD2022 only. Reported to be weak at detecting potholes. Never scored on its test set. Data paths pointed to a Mac. |
| Our earlier model (Kaggle only) | YOLOv8s trained on the Kaggle dataset only (2,236 images). **Pothole recall only 0.55** (missed almost half of potholes). |
| RDD2022 images on this PC | None. |

### The two dataset links that were provided
- `https://rmets.onlinelibrary.wiley.com/doi/10.1002/gdj3.260` is the journal article describing **RDD2022**.
- `https://github.com/sekilab/RoadDamageDetector` is the official download page for the same **RDD2022** dataset.

**They are the same dataset**, so we used RDD2022 together with the Kaggle dataset.

---

## 3. Decisions made (with the user)

| Question | Choice | Reason |
|---|---|---|
| Which RDD2022 countries to download? | India, Japan, Czech, United States, China_MotorBike (**Norway skipped**) | Norway is 9.9 GB of mostly wide highway shots where damage is tiny. The other 5 are about 2.4 GB. |
| Classifier or detector? | **Detector, then RSI** | A whole-image classifier shrinks the image, so small potholes blur away. A detector looks for each defect separately, and the result converts to RSI by a clear rule. YOLO also runs fast on Jetson (TensorRT). |
| Which defects count? | **Potholes, cracks and alligator cracks only** | The project only considers road-surface defects. |

---

## 4. Downloading RDD2022

**Problem 1:** every download link on the GitHub page (Amazon S3, `bigdatacup.s3...`) now returns **403 Access Denied**.

**Fix:** used the official **figshare** copy instead (article 21431547). Figshare only offers it as **one 13.3 GB zip**, but that zip simply holds one uncompressed zip per country. The script asks the server for just the byte ranges of the 5 countries we need, so Norway was never downloaded.

**Problem 2:** the Python `requests` library stalled forever on large downloads. **Fix:** the script calls `curl` instead, with automatic retries.

**Problem 3:** one country at a time was slow (about 17 MB/min). **Fix:** downloaded all 5 countries in parallel.

**Result** (`Dataset/RDD2022/`, 4.9 GB including the zips):

| Country | Training images | Annotation files |
|---|---|---|
| Japan | 10,506 | 10,506 |
| India | 7,706 | 7,706 |
| United States | 4,805 | 4,805 |
| Czech | 2,829 | 2,829 |
| China_MotorBike | 1,977 | 1,977 |
| **Total** | **27,823** | |

Script: `Training_Files/download_rdd2022.py`. It can resume, and `--countries` picks which countries to fetch.

---

## 5. Building the merged dataset

Script: `Training_Files/build_yolo_dataset.py`. Output: `Dataset/merged_yolo/` (524 MB; images are hard links, so no copies are made).

### 5.1 Shared class list

| Class | From RDD2022 | From Kaggle |
|---|---|---|
| 0 `pothole` | D40 | pothole |
| 1 `crack` | D00 (along the road), D10 (across the road) | cracks |
| 2 `alligator_crack` | D20 | – |

The detector file also contains one extra class from the Kaggle data that this project does not use. The pipeline never asks the model for it, and images containing it are left out of every evaluation in this report.

RDD2022 codes **not used:**

| Code | Boxes | Meaning |
|---|---|---|
| D44 | 5,057 | utility cover (not a defect) |
| D50 | 3,581 | road marking |
| D43 | 793 | crosswalk blur |
| Repair | 277 | patched area |
| D01 | 179 | rare crack sub-code |
| D11 | 45 | rare crack sub-code |
| D0w0 | 1 | typo |

### 5.2 Converting and splitting
- RDD2022 boxes were converted from Pascal VOC XML into YOLO format. Boxes are clipped to the image, and any box under 1 pixel is dropped.
- **RDD2022 split:** 80% train, 10% validation, 10% test. Frames are kept together in **blocks of 50 consecutive frames**, so near-identical dashcam frames can't end up in both train and test. That would inflate the scores.
- **Kaggle split:** Kaggle *train* goes to train. Kaggle *valid* is divided alternately between validation and test.
- **Background images** (images with no defect boxes) are capped at 15% of each split. All of them would swamp the training.

### 5.3 Final dataset

| | Train | Val | Test |
|---|---|---|---|
| Images | 20,370 | 2,522 | 2,452 |
| – of which Kaggle | 2,186 | 240 | 240 |
| – of which background | 3,055 | 378 | 367 |
| Pothole boxes | **5,880** | 726 | 896 |
| Crack boxes | 20,759 | 2,662 | 2,474 |
| Alligator-crack boxes | 7,859 | 981 | 1,015 |

Evaluations in sections 7–8 use the 2,378 test images (and 2,448 validation images) that contain only these three classes.

The model now has **5,880 pothole boxes to train on**, up from **1,080** in the Kaggle-only model, about 5.4× more.

---

## 6. Training

| Setting | Value |
|---|---|
| Model | YOLOv8s, starting from pretrained COCO weights |
| Epochs | 60 (stop early if no improvement for 15) |
| Image size | 640 |
| Batch size | 16 |
| GPU | NVIDIA RTX 5060 Laptop (8 GB) |
| Time per epoch | about 3.7 minutes |
| Output | `runs/detect/merged_v8s/` |

### Problems during training
1. **Crash after epoch 1:** "The paging file is too small". With 4 data-loading workers, the laptop ran out of memory (15 GB RAM, nearly all in use). **Fix:** restarted with `workers=2`.
2. **Interrupted at epoch 51:** at 19:57 the Claude desktop app updated itself and the GPU driver reset; the laptop had also gone into standby at 19:46. The training process was stopped. **Fix:** resumed from `last.pt` (epoch 50); nothing was lost. The best checkpoint before the resume is backed up as `weights/best_epoch1-50.pt`.

### Progress (merged validation set, averaged over all trained classes)

| Epoch | Recall | mAP50 |
|---|---|---|
| 1 | 0.47 | 0.46 |
| 10 | 0.58 | 0.59 |
| 20 | 0.63 | 0.68 |
| 30–50 | ~0.68 | 0.72–0.73 |
| **59 (best)** | **0.69** | **0.728** |

The curve is flat after about epoch 40.

### Final validation results per class (box level, merged validation set)

| Class | Precision | Recall | mAP50 |
|---|---|---|---|
| Average of the three | 0.69 | 0.60 | 0.66 |
| Pothole | 0.69 | 0.54 | 0.60 |
| Crack | 0.66 | 0.62 | 0.66 |
| Alligator crack | 0.72 | 0.65 | 0.72 |

Pothole recall per box looks low here because the RDD2022 dashcam frames contain many small, distant potholes. The suspension needs the **image-level** result (section 8), not every single box.

---

## 7. From detections to road class and RSI

Script: `Training_Files/rsi_detector.py`

### 7.1 The rule (applied to each frame)
Only boxes with confidence ≥ **0.15** count.
1. Any **pothole** → **Pothole**
2. Otherwise, any **alligator crack**, or crack boxes together covering ≥ **7%** of the frame (overlaps counted once) → **Rough**
3. Otherwise, any crack at all → **Normal**
4. No defects → **Smooth**

### 7.2 RSI values (from the proposal's RSI diagnostic matrix)

| Class | RSI |
|---|---|
| Smooth | 0.10 |
| Normal | 0.35 |
| Rough | 0.65 |
| Pothole | 0.95 |

### 7.3 Smoothing over time
The rule reuses the teammate's `SeverityStateMachine` from `infer_smooth.py`:
- **Moving to a worse class happens instantly**, because a pothole is only in view for a few frames.
- **Moving to a better class** needs the smoothed result to be confident **and** at least **10 frames** in the current class, so the suspension doesn't soften before the wheel has passed the defect.

### 7.4 Choosing the confidence threshold (validation set)

| Confidence | Pothole recall | Pothole precision |
|---|---|---|
| **0.15** | **0.82** | 0.79 |
| 0.25 (old default) | 0.74 | 0.87 |
| 0.40 | 0.65 | 0.93 |

**Chosen:** confidence 0.15, the best pothole recall. For the suspension, **missing a pothole is worse than a false alarm**. The cost is more false pothole alarms.

### 7.5 Fixing Normal vs Rough
With the first rule (summed crack area ≥ 5%), only **56%** of true-Normal test frames were classified as Normal. Checking every one of the 486 true-Normal test frames showed why:

| What happened to true-Normal frames | Frames | Share |
|---|---|---|
| Correct | 272 | 56% |
| → Rough: predicted crack area crossed the 5% line | 129 | **27%** |
| → Smooth: thin cracks missed | 39 | 8% |
| → Rough: a false alligator crack was detected | 34 | 7% |
| → Pothole: false pothole | 12 | 2% |

**Cause of the biggest error:** in these frames the model's predicted crack area was about **1.8× the labelled area** (median 3.4% vs 1.9%), and more than 2× in 176 frames. Two reasons:
1. The rule **added up box areas**, so overlapping crack boxes were counted twice.
2. The 5% line sits close to real Normal crack areas: 10% of true-Normal frames already cover 4.3%.

**Fix (no retraining):** predictions now use the **combined area covered by the crack boxes** (overlaps counted once) with a **7%** line. Both were tuned on the validation set from 24 combinations (summed vs combined area × crack confidence 0.15 / 0.25 / 0.35 × line 5 / 7 / 9 / 12%), keeping Rough recall ≥ 0.85.

The **true** class is still computed from the labelled boxes with the original definition (summed area ≥ 5%), so before and after are scored against the same answers.

| Test set | Accuracy | Smooth | Normal | Rough | Pothole |
|---|---|---|---|---|---|
| Before (summed area, 5%) | 0.808 | 0.79 | 0.56 | 0.91 | 0.86 |
| **After (combined area, 7%)** | **0.828** | 0.79 | **0.71** | 0.89 | 0.86 |

**Trade-off:** slightly more Rough frames are now called Normal (57, up from 34), so Rough recall drops from 0.91 to 0.89. Potholes are unaffected.

**Considered and not done yet: adding Norway.** Norway has the most Normal images of any RDD2022 country (1,291 of 8,161), but it would mainly help the 8% of Normal frames where thin cracks are missed. It doesn't address the biggest cause above, costs 9.9 GB, and its damage is very small after resizing to 640 px. It's worth trying later together with a higher training resolution.

---

## 8. Test-set results (image level)

Script: `Training_Files/evaluate_rsi.py --split test`, with the fixed rule from section 7.5.

### All 2,378 test images. Accuracy **0.828**

| True \ Predicted | Smooth | Normal | Rough | Pothole | Recall |
|---|---|---|---|---|---|
| Smooth | **289** | 41 | 23 | 14 | 0.79 |
| Normal | 39 | **343** | 92 | 12 | **0.71** |
| Rough | 32 | 57 | **962** | 36 | 0.89 |
| Pothole | 22 | 8 | 33 | **375** | **0.86** |
| **Precision** | 0.76 | 0.76 | 0.87 | **0.86** | |

### Which way the errors go
For the suspension, calling a road worse than it is (too stiff) is safer than calling it better (too soft).

| | Frames | Share |
|---|---|---|
| Correct | 1,969 | 82.8% |
| Over-estimate (too stiff, less comfort) | 218 | 9.2% |
| Under-estimate by 1 level | 129 | 5.4% |
| Under-estimate by 2+ levels (e.g. pothole treated as Smooth) | 62 | 2.6% |

Of the 63 missed pothole frames, 33 were called Rough (RSI 0.65, still stiffened); 30 (7% of pothole frames) went to Smooth or Normal.

### By data source

| | Accuracy | Pothole recall | Pothole precision |
|---|---|---|---|
| RDD2022 dashcam (2,212 images) | 0.82 | 0.84 | 0.83 |
| Kaggle (166 images) | 0.93 | 0.92 | 0.98 |

### Like-for-like with the old model
Same Kaggle validation images, box level:

| | Old (Kaggle only) | New (merged) |
|---|---|---|
| Pothole recall | 0.548 | **0.588** |
| Pothole mAP50 | 0.678 | 0.681 |
| Crack mAP50 | 0.761 | 0.744 |

**What this means:** on the old benchmark the box scores barely changed. The real gains are elsewhere:
1. The model now works on **dashcam road footage** (RDD2022), which the old model was never trained on.
2. At image level it flags **86% of pothole images**, which is what the controller uses.

---

## 9. Test-video results

Run on the 3 road-surface test videos that came with the Kaggle dataset (`testvideo1`, `testvideo2` and `testvideo6` in its `test/video/` folder). The videos have no labels, so they were checked by eye on 6 sample frames each.

| Video | Content | Time in each class | Verdict |
|---|---|---|---|
| 1 | Badly potholed dirt road | Pothole 100% | ✅ Correct; many potholes boxed |
| 2 | Wet rural road, dashcam | Pothole 57%, Rough 25%, Smooth 14%, Normal 3% | ✅ Mostly right: alligator crack → Rough, water-filled potholes → Pothole. Class changes often (31 switches) |
| 6 | Asphalt cracks, close up | Rough 93%, Pothole 7% | ✅ Cracks → Rough; 2 brief false potholes on a wide crack |

Annotated videos (boxes plus a coloured banner showing road class and RSI): `runs/detect/merged_v8s_videos/testvideo1_rsi.mp4`, `testvideo2_rsi.mp4`, `testvideo6_rsi.mp4`.

---

## 10. Known limitations

1. **Normal vs Rough is still the weakest split** (Normal recall 0.71 after the fix in section 7.5, up from 0.56). The boundary is a crack-area line, not a measured roughness.
2. **The "true" road class is computed, not labelled by hand.** It comes from the annotation boxes. RDD2022 also doesn't annotate every defect, so some "Smooth" images contain damage.
3. **False potholes** appear on wide cracks, a side effect of the low 0.15 threshold.
4. **Small, distant potholes are the hardest boxes** (per-box recall 0.54), so how far ahead a pothole is detected is not known yet.
5. **Only one test video is from a dashcam.** None tests the real camera position on our vehicle.
6. **Not tested on Jetson yet.** No ONNX/TensorRT export, and no speed (FPS) measurement on Jetson.

---

## 11. Files

### New scripts in `Training_Files/`

| File | Purpose | How to run |
|---|---|---|
| `download_rdd2022.py` | Downloads the chosen RDD2022 countries from figshare | `python download_rdd2022.py --out ../Dataset` |
| `build_yolo_dataset.py` | Builds `Dataset/merged_yolo/` (the dataset the current model was trained on) | `python build_yolo_dataset.py` |
| `rsi_detector.py` | Video or camera → road class + RSI; `--save` writes an annotated video. Only requests pothole, crack and alligator-crack boxes | `python rsi_detector.py --weights ../runs/detect/merged_v8s/weights/best.pt --source video.mp4 --save out.mp4` |
| `evaluate_rsi.py` | Road-class confusion matrix; `--sweep` tests thresholds. Scores against the label definition (summed crack area ≥ 5%) | `python evaluate_rsi.py --weights ... --split test` |

Run them with the Python environment at `pothole-cracks-and-openmanhole/.venv`.

### Data and results (not in git)

| Path | Contents |
|---|---|
| `Dataset/RDD2022/` | 5 RDD2022 countries + downloaded zips + download logs |
| `Dataset/merged_yolo/` | Merged dataset + `data.yaml`; also `kaggle_eval.yaml` for the like-for-like check |
| `runs/detect/merged_v8s/weights/best.pt` | ⭐ **Final model** |
| `runs/detect/merged_v8s/weights/best_epoch1-50.pt` | Backup from before the resume |
| `runs/detect/merged_v8s/` | Training curves, confusion matrix, `results.csv` |
| `runs/detect/merged_v8s_kaggle_val/` | Like-for-like evaluation output |
| `runs/detect/merged_v8s_videos/` | Annotated test videos |
| `runs/merged_v8s_train_part1.log`, `runs/merged_v8s_train.log` | Training logs (before and after the resume) |

The scripts and this report are on the git branch `road-perception-merged-yolo`. `Dataset/` and `runs/` are not in git.

---

## 12. Suggested next steps

1. **Improve Normal vs Rough further:** define the classes from real ride measurements (accelerometer on the Quanser rig or a phone in the car); try a separate confidence threshold for alligator cracks (7% of Normal errors); train at 960 px and/or **YOLOv8m**, optionally adding Norway (helps thin cracks).
2. **Export for Jetson:** ONNX, then TensorRT (FP16), and measure FPS on the Jetson Orin.
3. **Record a real dashcam clip** on local roads and test with `rsi_detector.py --save`.
4. **Connect to the controller:** send the RSI from `rsi_detector.py` (serial, ROS or socket) to the STSMC gain scheduler.
5. **Resolve the RSI mismatch:** `config.py`/`infer_smooth.py` use 0–3, while the proposal and `rsi_detector.py` use 0.10–0.95.
6. **Optional: retrain with only the three classes**, so the model file contains nothing the project doesn't use.
