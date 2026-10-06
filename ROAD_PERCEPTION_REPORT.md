# Road Perception: Merged Dataset, YOLO Detector and Road Severity Index

**Date:** 6 October 2026
**Scope:** the road-perception (CNN) part of the *Edge AI-based Adaptive Suspension System* project only. The STSMC controller, Simulink model and Jetson hardware were not touched.

---

## 1. Summary

- **Goal:** combine several road-damage datasets, train one model, and turn its output into the 4 road classes (**Smooth, Normal, Rough, Pothole**) and the **Road Severity Index (RSI)** from the proposal.
- **Approach chosen:** one **YOLOv8s object detector** trained on **RDD2022 + the Kaggle pothole-cracks-and-openmanhole dataset**. Each frame's detections are converted into a road class and RSI by a simple rule.
- **Main results:**

| What is measured | Result |
|---|---|
| Road-class accuracy, test set (2,452 images) | **0.815** |
| Pothole recall, image level (test set) | **0.88**, from 0.84 on RDD2022 dashcam images to 0.98 on Kaggle images |
| Pothole precision, image level (test set) | **0.88** |
| Pothole recall per box, on the old Kaggle validation images | **0.59** (old model: 0.55) |
| Detector mAP50, merged validation set | **0.728** |
| 7 test videos | All ran correctly; potholes, open manholes and cracks found; a few false alarms (section 9) |

---

## 2. Starting point

| Item | State before this work |
|---|---|
| Teammate's model (`Training_Files/train.py`) | MobileNetV2 4-class image classifier trained on RDD2022 only. Reported to be weak at detecting potholes. Never scored on its test set. Data paths pointed to a Mac. |
| Our earlier model (`pothole-cracks-and-openmanhole/`) | YOLOv8s trained on the Kaggle dataset only (2,236 images). mAP50 0.79, but **pothole recall only 0.55** (missed almost half of potholes). |
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

| New class | From RDD2022 | From Kaggle |
|---|---|---|
| 0 `pothole` | D40 | 0 pothole |
| 1 `crack` | D00 (along the road), D10 (across the road) | 1 cracks |
| 2 `alligator_crack` | D20 | – |
| 3 `open_manhole` | – | 2 open_manhole |

RDD2022 codes **not used:**

| Code | Boxes | Meaning |
|---|---|---|
| D44 | 5,057 | closed manhole cover (not a defect) |
| D50 | 3,581 | road marking |
| D43 | 793 | crosswalk blur |
| Repair | 277 | patched area |
| D01 | 179 | rare crack sub-code |
| D11 | 45 | rare crack sub-code |
| D0w0 | 1 | typo |

### 5.2 Converting and splitting
- RDD2022 boxes were converted from Pascal VOC XML into YOLO format. Boxes are clipped to the image, and any box under 1 pixel is dropped.
- **RDD2022 split:** 80% train, 10% validation, 10% test. Frames are kept together in **blocks of 50 consecutive frames**, so near-identical dashcam frames can't end up in both train and test. That would inflate the scores.
- **Kaggle split:** Kaggle *train* goes to train. Kaggle *valid* (480 images) is divided alternately between validation and test.
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
| Open-manhole boxes | 680 | 74 | 74 |

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

### Progress (merged validation set, all classes)

| Epoch | Recall | mAP50 |
|---|---|---|
| 1 | 0.47 | 0.46 |
| 10 | 0.58 | 0.59 |
| 20 | 0.63 | 0.68 |
| 30–50 | ~0.68 | 0.72–0.73 |
| **59 (best)** | **0.69** | **0.728** |

### Final validation results per class (box level, merged validation set)

| Class | Precision | Recall | mAP50 |
|---|---|---|---|
| All | 0.75 | 0.68 | 0.725 |
| Pothole | 0.69 | 0.54 | 0.60 |
| Crack | 0.66 | 0.62 | 0.66 |
| Alligator crack | 0.72 | 0.65 | 0.72 |
| Open manhole | 0.92 | 0.92 | 0.92 |

Pothole recall per box looks low here because the RDD2022 dashcam frames contain many small, distant potholes. The suspension needs the **image-level** result (section 8), not every single box.

---

## 7. From detections to road class and RSI

Script: `Training_Files/rsi_detector.py`

### 7.1 The rule (applied to each frame)
Only boxes with confidence ≥ **0.15** count.
1. Any **pothole** or **open manhole** → **Pothole**
2. Otherwise, any **alligator crack**, or cracks covering ≥ **5%** of the frame → **Rough**
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

### 7.4 Choosing the two thresholds (validation set)
`evaluate_rsi.py --sweep` tested 20 combinations:

| Confidence | Crack area | Accuracy | Pothole recall | Pothole precision |
|---|---|---|---|---|
| **0.15** | **0.05** | **0.776** | **0.849** | 0.817 |
| 0.25 (old default) | 0.05 | 0.775 | 0.786 | 0.892 |
| 0.40 | 0.05 | 0.718 | 0.702 | 0.942 |

**Chosen:** confidence 0.15, crack area 0.05. This gives the best pothole recall with almost no loss of accuracy. For the suspension, **missing a pothole is worse than a false alarm**.

---

## 8. Test-set results (image level)

Script: `Training_Files/evaluate_rsi.py --split test`

### All 2,452 test images. Accuracy **0.815**

| True \ Predicted | Smooth | Normal | Rough | Pothole | Recall |
|---|---|---|---|---|---|
| Smooth | **289** | 40 | 24 | 14 | 0.79 |
| Normal | 39 | **272** | 163 | 12 | **0.56** |
| Rough | 32 | 34 | **985** | 36 | 0.91 |
| Pothole | 19 | 7 | 34 | **452** | **0.88** |
| **Precision** | 0.76 | 0.77 | 0.82 | **0.88** | |

### By data source

| | Accuracy | Pothole recall | Pothole precision |
|---|---|---|---|
| RDD2022 dashcam (2,212 images) | 0.80 | 0.84 | 0.83 |
| Kaggle (240 images) | 0.96 | 0.98 | 0.99 |

### Like-for-like with the old model
Same 480 Kaggle validation images, box level:

| | Old (Kaggle only) | New (merged) |
|---|---|---|
| Pothole recall | 0.548 | **0.588** |
| Pothole mAP50 | 0.678 | 0.681 |
| Crack mAP50 | 0.761 | 0.744 |
| Open manhole mAP50 | 0.922 | 0.916 |
| Overall mAP50 | 0.787 | 0.780 |

**What this means:** on the old benchmark the box scores barely changed. The real gains are elsewhere:
1. The model now works on **dashcam road footage** (RDD2022), which the old model was never trained on.
2. At image level it flags **88% of pothole images**, which is what the controller uses.

---

## 9. Test-video results

Run on the 7 videos in `pothole-cracks-and-openmanhole/dataset/dataset/test/video/`. The videos have no labels, so they were checked by eye on 6 sample frames each.

| Video | Content | Time in each class | Verdict |
|---|---|---|---|
| 1 | Badly potholed dirt road | Pothole 100% | ✅ Correct; many potholes boxed |
| 2 | Wet rural road, dashcam | Pothole 57%, Rough 25%, Smooth 14%, Normal 3% | ✅ Mostly right. Class changes often (30 switches) |
| 3 | Inside an open manhole, close up | Pothole 55%, Smooth 45% | ✅ Smooth only when the camera looks into the pit |
| 4 | Worker lifting a manhole cover | Pothole 71%, Smooth 29% | ⚠️ Open manhole correct; **one false pothole** on the concrete around the closed cover |
| 5 | City street with an open manhole | Pothole 50%, Smooth 21%, Rough 16%, Normal 12% | ✅ Found at mid-distance; **missed while far away** |
| 6 | Asphalt cracks, close up | Rough 93%, Pothole 7% | ✅ Cracks → Rough; 2 brief false potholes on a wide crack |
| 7 | CCTV: children near an open manhole | Pothole 83%, Smooth 17% | ✅ Steady detection; Smooth once it's covered |

Annotated videos (boxes plus a coloured banner showing road class and RSI): `runs/detect/merged_v8s_videos/testvideo1_rsi.mp4` … `testvideo7_rsi.mp4`.

---

## 10. Known limitations

1. **Normal vs Rough is the weakest split** (Normal recall 0.56). The boundary is a hand-picked 5% crack area, not a measured roughness. The error usually goes toward Rough, which is the safer direction for the suspension.
2. **The "true" road class is computed, not labelled by hand.** It comes from the annotation boxes using the same rule as the prediction. RDD2022 also doesn't annotate every defect, so some "Smooth" images contain damage.
3. **False potholes** appear on rough concrete and wide cracks, a side effect of the low 0.15 threshold.
4. **Distant objects are detected late**, which reduces the time the suspension has to react.
5. **The test videos aren't dashcam footage** (mostly internet clips: handheld or CCTV). They don't test the real camera position on the vehicle.
6. **Not tested on Jetson yet.** No ONNX/TensorRT export, and no speed (FPS) measurement on Jetson.

---

## 11. Files

### New scripts in `Training_Files/`

| File | Purpose | How to run |
|---|---|---|
| `download_rdd2022.py` | Downloads the chosen RDD2022 countries from figshare | `python download_rdd2022.py --out ../Dataset` |
| `build_yolo_dataset.py` | Builds `Dataset/merged_yolo/` | `python build_yolo_dataset.py` |
| `rsi_detector.py` | Video or camera → road class + RSI; `--save` writes an annotated video | `python rsi_detector.py --weights ../runs/detect/merged_v8s/weights/best.pt --source video.mp4 --save out.mp4` |
| `evaluate_rsi.py` | Road-class confusion matrix; `--sweep` tests thresholds | `python evaluate_rsi.py --weights ... --split test` |

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
| `runs/detect/merged_v8s_videos/` | 7 annotated test videos |
| `runs/merged_v8s_train_part1.log`, `runs/merged_v8s_train.log` | Training logs (before and after the resume) |

Nothing has been committed to git yet. `Dataset/` is already ignored by git.

---

## 12. Suggested next steps

1. **Improve Normal vs Rough:** tune the crack-area threshold against real ride measurements, or train **YOLOv8m** (more accurate, about 2× slower).
2. **Export for Jetson:** ONNX, then TensorRT (FP16), and measure FPS on the Jetson Orin.
3. **Record a real dashcam clip** on local roads and test with `rsi_detector.py --save`.
4. **Connect to the controller:** send the RSI from `rsi_detector.py` (serial, ROS or socket) to the STSMC gain scheduler.
5. **Resolve the RSI mismatch:** `config.py`/`infer_smooth.py` use 0–3, while the proposal and `rsi_detector.py` use 0.10–0.95.
