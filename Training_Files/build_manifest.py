"""
build_manifest.py

Builds manifest.csv from RDD2022 annotations.

KEY CHANGE vs. the original version
-----------------------------------
The model only ever sees the bottom `ROI_BOTTOM_FRACTION` of each image
(dataset.BottomCrop). The old script scored damage over the WHOLE image, so
images could be labelled Pothole/Rough because of damage in the cropped-away
top of the frame -- something the model can never see. Now every box is
clipped to the same ROI and the score is normalised by the ROI area, so the
label describes exactly what the model receives.

The old full-image label is still written as `label_full` so you can see how
many labels change (printed at the end).
"""

import os
import csv
import glob
import xml.etree.ElementTree as ET

import config


def parse_annotation(xml_path):
    """Return list of (damage_code, xmin, ymin, xmax, ymax) and (img_w, img_h)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    boxes = []
    for obj in root.findall("object"):
        code = obj.find("name").text.strip()
        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)
        boxes.append((code, xmin, ymin, xmax, ymax))

    return boxes, img_w, img_h


def damage_weight(code):
    if code in config.POTHOLE_CODES:
        return config.DAMAGE_WEIGHTS["pothole"]
    if code in config.SEVERE_CRACK_CODES:
        return config.DAMAGE_WEIGHTS["severe_crack"]
    # minor cracks and unknown codes (treated conservatively as minor)
    return config.DAMAGE_WEIGHTS["minor_crack"]


def clip_box_to_roi(box, img_w, img_h, roi_fraction):
    """
    Clip a (code, xmin, ymin, xmax, ymax) box to the bottom-`roi_fraction`
    strip of the image. Returns the clipped box, or None if too little of it
    remains inside the ROI to count as visible damage.
    """
    code, xmin, ymin, xmax, ymax = box
    roi_top = img_h * (1.0 - roi_fraction)

    ymin = max(ymin, roi_top)
    ymax = min(ymax, float(img_h))
    xmin = max(xmin, 0.0)
    xmax = min(xmax, float(img_w))

    if (ymax - ymin) < config.MIN_CLIPPED_BOX_PX or (xmax - xmin) <= 0:
        return None
    return (code, xmin, ymin, xmax, ymax)


def weighted_severity_score(boxes, img_w, img_h, roi_fraction=None):
    """
    Sum weighted normalised bbox area.

    roi_fraction=None  -> whole image (legacy score, normalised by image area)
    roi_fraction=f     -> boxes clipped to the bottom f of the image and
                          normalised by the ROI area (what the model sees)

    Returns (score, has_pothole, num_relevant_boxes, pothole_area_frac).
    IGNORED_CODES (crosswalk / white-line blur / manhole) are skipped.
    """
    if roi_fraction is None:
        norm_area_total = float(img_w * img_h)
    else:
        norm_area_total = float(img_w * img_h * roi_fraction)

    score = 0.0
    pothole_area = 0.0
    num_relevant_boxes = 0

    for box in boxes:
        code = box[0]
        if code in config.IGNORED_CODES:
            continue

        if roi_fraction is not None:
            box = clip_box_to_roi(box, img_w, img_h, roi_fraction)
            if box is None:
                continue
            code = box[0]

        _, xmin, ymin, xmax, ymax = box
        area = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
        norm_area = area / norm_area_total if norm_area_total > 0 else 0.0

        score += damage_weight(code) * norm_area
        if code in config.POTHOLE_CODES:
            pothole_area += norm_area
        num_relevant_boxes += 1

    has_pothole = pothole_area > 0.0
    return score, has_pothole, num_relevant_boxes, pothole_area


def assign_label(score, has_pothole, num_boxes, pothole_area=None,
                 normal_max=None, min_pothole_area=None):
    normal_max = config.NORMAL_MAX_SCORE if normal_max is None else normal_max
    min_pothole_area = (config.MIN_POTHOLE_ROI_AREA
                        if min_pothole_area is None else min_pothole_area)
    if num_boxes == 0:
        return "Smooth"
    if has_pothole and (pothole_area is None or pothole_area >= min_pothole_area):
        return "Pothole"
    if score < normal_max:
        return "Normal"
    return "Rough"


def find_pairs(country_dir):
    """Yield (image_path, xml_path_or_None) for a country's train split."""
    img_dir = os.path.join(country_dir, "train", "images")
    xml_dir = os.path.join(country_dir, "train", "annotations", "xmls")

    if not os.path.isdir(img_dir):
        print(f"  [skip] no images dir at {img_dir}")
        return

    for img_path in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        xml_path = os.path.join(xml_dir, stem + ".xml")
        yield img_path, (xml_path if os.path.isfile(xml_path) else None)


def main():
    rows = []
    counts = {c: 0 for c in config.CLASSES}
    changed = 0
    ambiguous = 0

    for country in config.COUNTRIES:
        country_dir = os.path.join(config.RDD2022_ROOT, country)
        if not os.path.isdir(country_dir):
            print(f"[warn] country folder not found, skipping: {country_dir}")
            continue

        print(f"Processing {country} ...")
        n_country = 0
        for img_path, xml_path in find_pairs(country_dir):
            if xml_path is None:
                boxes, img_w, img_h = [], 0, 0
            else:
                boxes, img_w, img_h = parse_annotation(xml_path)

            # legacy full-image label, kept only for comparison
            s_full, p_full, n_full, pa_full = weighted_severity_score(
                boxes, img_w, img_h, roi_fraction=None)
            label_full = assign_label(s_full, p_full, n_full, pa_full,
                                      normal_max=0.02, min_pothole_area=0.0)

            # ROI label: what the model can actually see
            score, has_pothole, num_boxes, pothole_area = weighted_severity_score(
                boxes, img_w, img_h, roi_fraction=config.ROI_BOTTOM_FRACTION)
            label = assign_label(score, has_pothole, num_boxes, pothole_area)

            is_ambiguous = (
                num_boxes > 0 and not has_pothole and
                abs(score - config.NORMAL_MAX_SCORE) <= config.AMBIGUITY_MARGIN
            )

            counts[label] += 1
            changed += int(label != label_full)
            ambiguous += int(is_ambiguous)
            n_country += 1

            rows.append({
                "image_path": img_path,
                "label": label,
                "severity_score": round(score, 5),
                "num_boxes": num_boxes,
                "country": country,
                "label_full": label_full,
                "ambiguous": int(is_ambiguous),
            })

        print(f"  -> {n_country} images")

    if not rows:
        print("No images found. Check RDD2022_ROOT and COUNTRIES in config.py.")
        return

    with open(config.MANIFEST_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total = sum(counts.values())
    print("\nClass distribution (ROI labels):")
    for c in config.CLASSES:
        pct = 100.0 * counts[c] / total if total else 0.0
        print(f"  {c:8s}: {counts[c]:6d}  ({pct:5.1f}%)")
    print(f"\nLabels that changed vs. the old full-image labelling: "
          f"{changed} / {total} ({100.0 * changed / total:.1f}%)")
    print(f"Ambiguous (near Normal/Rough threshold): {ambiguous} "
          f"({100.0 * ambiguous / total:.1f}%)")
    print(f"\nWrote {total} rows to {config.MANIFEST_CSV}")

    if counts["Pothole"] / total < 0.03:
        print("\n[note] Pothole class is under 3% of the data. Consider "
              "class-balanced sampling (config.USE_BALANCED_SAMPLER).")


if __name__ == "__main__":
    main()
