"""
infer_smooth.py

Real-time-style inference loop:
    frame -> CNN softmax -> smoothed probability (moving average)
          -> hysteresis + min-dwell-time gate -> stable class
          -> STSMC preset lookup

This is the piece that prevents controller "chattering" -- rapid
switching between preset parameter sets caused by frame-to-frame
prediction noise (lighting, motion blur, etc).

Run against a video file or webcam index:
    python infer_smooth.py --source path/to/video.mp4
    python infer_smooth.py --source 0
"""

import argparse
from collections import deque

import cv2
import torch
import torch.nn.functional as F

import config
from train import build_model
from dataset import get_transforms


class SeverityStateMachine:
    """
    Asymmetric hysteresis ("peak hold"):

    - ESCALATION (moving to a MORE severe class) triggers instantly off
      the raw, single-frame prediction, bypassing smoothing and dwell
      time entirely. A physical pothole is only in frame for ~3-8
      frames at highway speed; averaging it into an 8-frame window
      would dilute the spike below detection and the suspension would
      not stiffen in time.
    - DE-ESCALATION (moving to a LESS severe class) still requires the
      smoothed prediction to clear the confidence threshold AND a
      minimum dwell time, so the suspension doesn't soften again before
      the rear axle has actually cleared the defect.

    Requires config.CLASSES to be ordered from least to most severe
    (Smooth, Normal, Rough, Pothole).
    """

    def __init__(self, window=config.SMOOTHING_WINDOW,
                 conf_threshold=config.CONFIDENCE_THRESHOLD,
                 min_dwell=config.MIN_DWELL_FRAMES):
        self.window = window
        self.conf_threshold = conf_threshold
        self.min_dwell = min_dwell

        self.prob_history = deque(maxlen=window)
        self.current_class = config.CLASSES[0]  # start assuming Smooth
        self.frames_in_current_class = 0

    def update(self, probs):
        """
        probs: 1D tensor of length len(config.CLASSES), raw per-frame
               softmax output (not smoothed).
        Returns: (stable_class, smoothed_probs, switched: bool, reason: str)
        """
        self.prob_history.append(probs)
        smoothed = sum(self.prob_history) / len(self.prob_history)

        current_idx = config.CLASSES.index(self.current_class)
        raw_idx = int(probs.argmax())
        raw_conf = float(probs[raw_idx])

        self.frames_in_current_class += 1
        switched = False
        reason = ""

        # --- Escalation: instant, bypasses smoothing/dwell ---
        if raw_idx > current_idx and raw_conf >= self.conf_threshold:
            self.current_class = config.CLASSES[raw_idx]
            self.frames_in_current_class = 0
            switched = True
            reason = "escalation (peak hold, raw frame)"
            return self.current_class, smoothed, switched, reason

        # --- De-escalation: smoothed + dwell gated ---
        candidate_idx = int(smoothed.argmax())
        candidate_conf = float(smoothed[candidate_idx])
        if candidate_idx < current_idx:
            enough_dwell = self.frames_in_current_class >= self.min_dwell
            enough_conf = candidate_conf >= self.conf_threshold
            if enough_dwell and enough_conf:
                self.current_class = config.CLASSES[candidate_idx]
                self.frames_in_current_class = 0
                switched = True
                reason = "de-escalation (smoothed + dwell)"

        return self.current_class, smoothed, switched, reason


def load_model(device):
    model, _ = build_model()
    state_dict = torch.load(config.CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess_frame(frame_bgr, transform):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    from PIL import Image
    image = Image.fromarray(frame_rgb)
    return transform(image)


def get_infer_transform():
    # Reuse the exact same pipeline used at training time (including the
    # ROI bottom-crop) so inference sees the same geometry the model was
    # trained on.
    return get_transforms(train=False)


def emit_severity_index(stable_class):
    """
    This code does not hold the STSMC preset values themselves -- it
    only emits the classified condition as a Road Severity Index (RSI).
    The controller receives this index and applies its own preset
    lookup for (k1, k2, lambda, ...).

    Replace the print with your actual output channel: serial write,
    ROS publish, shared-memory/socket update, etc.
    """
    rsi = config.SEVERITY_INDEX[stable_class]
    print(f"  -> RSI out: {rsi} ({stable_class})")
    # e.g. serial_port.write(f"{rsi}\n".encode())
    # e.g. ros_publisher.publish(Int32(data=rsi))
    return rsi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0",
                         help="Video file path, or camera index (e.g. 0)")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    device = torch.device(config.DEVICE)
    model = load_model(device)
    transform = get_infer_transform()
    state_machine = SeverityStateMachine()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tensor = preprocess_frame(frame, transform).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu()

        stable_class, smoothed, switched, reason = state_machine.update(probs)

        if switched:
            print(f"[frame {frame_idx}] class switch -> {stable_class} [{reason}] "
                  f"(smoothed probs: {[round(float(p), 2) for p in smoothed]})")
            emit_severity_index(stable_class)

        frame_idx += 1

    cap.release()
    print(f"\nDone. Processed {frame_idx} frames. Final class: {state_machine.current_class}")


if __name__ == "__main__":
    main()
