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


def weighted_severity_score(boxes, img_w, img_h):
    """
    Sum normalized bbox area, weighted by damage-type severity.
    Returns (score, has_pothole, num_relevant_boxes).

    num_relevant_boxes excludes IGNORED_CODES (crosswalk blur, white-line
    blur, manhole cover) -- an image annotated with only those is still
    physically Smooth road surface, not "Normal" damage.
    """
    img_area = float(img_w * img_h)
    score = 0.0
    has_pothole = False
    num_relevant_boxes = 0

    for code, xmin, ymin, xmax, ymax in boxes:
        if code in config.IGNORED_CODES:
            continue

        area = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
        norm_area = area / img_area if img_area > 0 else 0.0

        if code in config.POTHOLE_CODES:
            weight = config.DAMAGE_WEIGHTS["pothole"]
            has_pothole = True
        elif code in config.SEVERE_CRACK_CODES:
            weight = config.DAMAGE_WEIGHTS["severe_crack"]
        elif code in config.MINOR_CRACK_CODES:
            weight = config.DAMAGE_WEIGHTS["minor_crack"]
        else:
            # Unknown code: treat conservatively as a minor crack so it
            # isn't silently dropped from the score.
            weight = config.DAMAGE_WEIGHTS["minor_crack"]

        score += weight * norm_area
        num_relevant_boxes += 1

    return score, has_pothole, num_relevant_boxes


def assign_label(score, has_pothole, num_boxes):
    if num_boxes == 0:
        return "Smooth"
    if has_pothole:
        return "Pothole"
    if score < config.NORMAL_MAX_SCORE:
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

    for country in config.COUNTRIES:
        country_dir = os.path.join(config.RDD2022_ROOT, country)
        if not os.path.isdir(country_dir):
            print(f"[warn] country folder not found, skipping: {country_dir}")
            continue

        print(f"Processing {country} ...")
        n_country = 0
        for img_path, xml_path in find_pairs(country_dir):
            if xml_path is None:
                boxes, num_boxes, score, has_pothole = [], 0, 0.0, False
            else:
                boxes, img_w, img_h = parse_annotation(xml_path)
                score, has_pothole, num_boxes = weighted_severity_score(boxes, img_w, img_h)

            label = assign_label(score, has_pothole, num_boxes)
            counts[label] += 1
            n_country += 1

            rows.append({
                "image_path": img_path,
                "label": label,
                "severity_score": round(score, 5),
                "num_boxes": num_boxes,
                "country": country,
            })

        print(f"  -> {n_country} images")

    if not rows:
        print("No images found. Check RDD2022_ROOT and COUNTRIES in config.py.")
        return

    with open(config.MANIFEST_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nClass distribution:")
    total = sum(counts.values())
    for c in config.CLASSES:
        pct = 100.0 * counts[c] / total if total else 0.0
        print(f"  {c:8s}: {counts[c]:6d}  ({pct:5.1f}%)")
    print(f"\nWrote {total} rows to {config.MANIFEST_CSV}")

    if counts["Pothole"] / total < 0.03:
        print("\n[note] Pothole class is under 3% of the data. Consider "
              "oversampling it or using class-weighted loss in training.")


if __name__ == "__main__":
    main()
