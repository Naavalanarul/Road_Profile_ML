"""
rsi_detector.py

YOLO detections -> road class -> Road Severity Index (RSI), using the
proposal's RSI diagnostic matrix. Model is trained on Dataset/merged_yolo
(classes: 0 pothole, 1 crack, 2 alligator_crack, 3 open_manhole).

Live use on video / camera, with the same peak-hold state machine as
infer_smooth.py (instant escalation, dwell-gated de-escalation):

    python rsi_detector.py --weights ../runs/detect/merged_v8s/weights/best.pt --source video.mp4
    python rsi_detector.py --weights ... --source 0
"""

import argparse

CLASSES = ["Smooth", "Normal", "Rough", "Pothole"]
RSI = {"Smooth": 0.10, "Normal": 0.35, "Rough": 0.65, "Pothole": 0.95}

POTHOLE, CRACK, ALLIGATOR, MANHOLE = 0, 1, 2, 3
CONF = 0.15               # min detection confidence to count a box
ROUGH_CRACK_AREA = 0.05   # crack boxes covering >= this fraction of the frame -> Rough


def classify(boxes, conf=CONF, rough_area=ROUGH_CRACK_AREA):
    """
    boxes: iterable of (cls, conf, w, h) with w, h normalised to 0-1.
    Ground-truth boxes can be passed with conf=1.0.
    Returns one of CLASSES.
    """
    kept = [(c, w * h) for c, p, w, h in boxes if p >= conf]
    if any(c in (POTHOLE, MANHOLE) for c, _ in kept):
        return "Pothole"
    crack_area = sum(a for c, a in kept if c in (CRACK, ALLIGATOR))
    if any(c == ALLIGATOR for c, _ in kept) or crack_area >= rough_area:
        return "Rough"
    if kept:
        return "Normal"
    return "Smooth"


def boxes_from_result(result):
    """Ultralytics Result -> list of (cls, conf, w, h) normalised."""
    b = result.boxes
    return [(int(c), float(p), float(wh[0]), float(wh[1]))
            for c, p, wh in zip(b.cls, b.conf, b.xywhn[:, 2:])]


def main():
    import torch
    from ultralytics import YOLO
    from infer_smooth import SeverityStateMachine

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", default="0", help="video path or camera index")
    ap.add_argument("--save", help="optional output .mp4 with boxes and RSI overlay")
    args = ap.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    model = YOLO(args.weights)
    # infer_smooth's state machine is written against config.CLASSES, which
    # has the same Smooth < Normal < Rough < Pothole order as CLASSES here.
    sm = SeverityStateMachine()
    writer, frames_in = None, {c: 0 for c in CLASSES}
    colours = {"Smooth": (80, 175, 76), "Normal": (0, 200, 255), "Rough": (0, 140, 255), "Pothole": (40, 40, 220)}

    for i, result in enumerate(model.predict(source, stream=True, conf=CONF, verbose=False)):
        frame_class = classify(boxes_from_result(result))
        probs = torch.zeros(len(CLASSES))
        probs[CLASSES.index(frame_class)] = 1.0
        stable, _, switched, reason = sm.update(probs)
        frames_in[stable] += 1
        if switched:
            print(f"[frame {i}] {stable}  RSI={RSI[stable]:.2f}  ({reason})")

        if args.save:
            import cv2
            img = result.plot()
            if writer is None:
                h, w = img.shape[:2]
                fps = cv2.VideoCapture(source).get(cv2.CAP_PROP_FPS) or 30
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            cv2.rectangle(img, (0, 0), (img.shape[1], 50), colours[stable], -1)
            cv2.putText(img, f"{stable}   RSI {RSI[stable]:.2f}   (frame: {frame_class})", (12, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(img)

    if writer is not None:
        writer.release()
    total = sum(frames_in.values())
    print("Time in each road class: " + ", ".join(
        f"{c} {100 * n / total:.0f}%" for c, n in frames_in.items()) if total else "No frames read.")


if __name__ == "__main__":
    main()
