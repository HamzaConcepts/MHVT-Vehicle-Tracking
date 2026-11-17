"""Single-target visual tracker (VitTrack) with an honest LOST state, plus an
optional YOLO detector overlay used for click-to-snap selection."""

from __future__ import annotations

from pathlib import Path

import cv2

VIT_MODEL = Path(__file__).resolve().parent.parent / "models" / "object_tracking_vittrack_2023sep.onnx"
VEHICLE_NAMES = {"vehicle", "car", "van", "truck", "bus", "motor", "motorcycle", "tricycle", "awning-tricycle"}


class TargetTracker:
    """Wraps OpenCV VitTrack. VitTrack exposes a confidence score; CSRT/KCF do not
    (they keep drawing a box on the wrong object), so VitTrack is required for a
    trustworthy TRACKING/LOST status."""

    def __init__(self, score_thresh: float = 0.30, lost_frames: int = 6):
        if not hasattr(cv2, "TrackerVit_create"):
            raise RuntimeError("OpenCV build lacks TrackerVit; install opencv-contrib-python>=4.9")
        if not VIT_MODEL.is_file():
            raise RuntimeError(f"Missing tracker model: {VIT_MODEL}")
        self.score_thresh = score_thresh
        self.lost_frames = lost_frames
        self.tracker = None
        self.bbox: tuple[int, int, int, int] | None = None
        self.score = 0.0
        self.low = 0
        self.lost = True

    def init(self, frame, bbox: tuple[int, int, int, int]) -> None:
        params = cv2.TrackerVit_Params()
        params.net = str(VIT_MODEL)
        self.tracker = cv2.TrackerVit_create(params)
        self.tracker.init(frame, bbox)
        self.bbox, self.score, self.low, self.lost = bbox, 1.0, 0, False

    def update(self, frame) -> tuple[bool, tuple[int, int, int, int] | None, float]:
        """Returns (tracking_ok, bbox, score). Once LOST it stays lost until init()."""
        if self.lost or self.tracker is None:
            return False, None, 0.0
        ok, box = self.tracker.update(frame)
        self.score = float(self.tracker.getTrackingScore())
        h, w = frame.shape[:2]
        x, y, bw, bh = (int(v) for v in box)
        cx, cy = x + bw / 2, y + bh / 2
        off_frame = not (0 <= cx < w and 0 <= cy < h)
        self.low = self.low + 1 if (not ok or self.score < self.score_thresh) else 0
        if self.low >= self.lost_frames or off_frame:
            self.lost = True
            return False, None, self.score
        self.bbox = (x, y, bw, bh)
        return True, self.bbox, self.score


class Detector:
    """Optional YOLO detector (use a drone-view/VisDrone fine-tuned model)."""

    def __init__(self, weights: str, imgsz: int = 1280, conf: float = 0.25):
        from ultralytics import YOLO  # imported lazily: only needed with --model

        self.model = YOLO(weights)
        self.imgsz, self.conf = imgsz, conf
        names = self.model.names
        self.classes = [i for i, n in names.items() if n in VEHICLE_NAMES] or None

    def detect(self, frame) -> list[tuple[int, int, int, int]]:
        r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, classes=self.classes,
                               device="cpu", verbose=False)[0]
        return [(int(x1), int(y1), int(x2 - x1), int(y2 - y1)) for x1, y1, x2, y2 in r.boxes.xyxy.tolist()]


def snap_to_detection(point: tuple[int, int], boxes) -> tuple[int, int, int, int] | None:
    best = None
    for (x, y, w, h) in boxes:
        if x <= point[0] <= x + w and y <= point[1] <= y + h:
            if best is None or w * h < best[2] * best[3]:
                best = (x, y, w, h)
    return best
