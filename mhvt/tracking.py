"""Target followers for the demo.

  DetFollower  YOLO detector + ByteTrack/BoT-SORT (Ultralytics) give every vehicle a persistent ID; the
               user's click picks one ID. `TargetManager` (prediction.py) keeps the target through
               occlusions (bridges) by predicting its path and re-binding it when it re-appears.
  VitFollower  fallback without a detector: OpenCV VitTrack follows the single picked box.

Both return the same `FollowResult` so main.py does not care which one runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .prediction import TargetManager, Track

ROOT = Path(__file__).resolve().parent.parent
VIT_MODEL = ROOT / "models" / "object_tracking_vittrack_2023sep.onnx"
DEFAULT_WEIGHTS = ROOT / "weights" / "vehicle_visdrone_n_bootstrap.pt"
BYTETRACK_CFG = ROOT / "bytetrack_mhvt.yaml"
BOTSORT_CFG = "botsort.yaml"  # Ultralytics' built-in (with camera-motion compensation)
VEHICLE_NAMES = {"vehicle", "car", "van", "truck", "bus", "motor", "motorcycle", "tricycle", "awning-tricycle"}


@dataclass
class FollowResult:
    ok: bool                              # a target position exists (observed or predicted)
    bbox: tuple | None = None
    score: float = 0.0
    predicted: bool = False               # position is dead-reckoned, not observed
    lost: bool = False                    # final: the target is gone until re-selected
    target_id: int | None = None
    std: float = 0.0                      # uncertainty radius of a predicted position (px)
    occluded_s: float = 0.0
    others: list = field(default_factory=list)   # [(id, bbox)] other tracked vehicles
    events: list = field(default_factory=list)   # [(level, text)]


def hsv_hist(frame: np.ndarray, bbox) -> np.ndarray | None:
    """Colour signature of a vehicle: hue/saturation histogram + coarse brightness, normalised."""
    x, y, w, h = (int(v) for v in bbox)
    mx, my = int(w * 0.15), int(h * 0.15)  # centre crop: ignore road pixels at the edges
    x0, y0 = max(0, x + mx), max(0, y + my)
    x1, y1 = min(frame.shape[1], x + w - mx), min(frame.shape[0], y + h - my)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    hsv = cv2.cvtColor(np.ascontiguousarray(frame[y0:y1, x0:x1]), cv2.COLOR_BGR2HSV)
    hs = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten()
    v = cv2.calcHist([hsv], [2], None, [4], [0, 256]).flatten()
    f = np.concatenate([hs / max(hs.sum(), 1e-6), v / max(v.sum(), 1e-6)]) / 2.0
    return f.astype(np.float32)


# ------------------------------------------------------------------ detector + tracker
class YoloTracks:
    """Ultralytics YOLO + a multi-object tracker -> list[Track] per frame."""

    def __init__(self, weights: str, tracker_cfg: str, imgsz: int = 1280, conf: float = 0.20):
        from ultralytics import YOLO  # imported lazily: heavy, and only needed for this backend

        self.model = YOLO(str(weights))
        self.tracker_cfg, self.imgsz, self.conf = str(tracker_cfg), imgsz, conf
        names = self.model.names
        self.classes = [i for i, n in names.items() if n in VEHICLE_NAMES] or None

    def tracks(self, frame: np.ndarray, first: bool) -> list[Track]:
        """first=True starts a fresh tracker (new IDs); first=False continues the sequence."""
        r = self.model.track(frame, persist=not first, tracker=self.tracker_cfg, classes=self.classes,
                             conf=self.conf, imgsz=self.imgsz, device="cpu", verbose=False)[0]
        if r.boxes is None or r.boxes.id is None:
            return []
        out = []
        for (x1, y1, x2, y2), tid, cf in zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist(), r.boxes.conf.tolist()):
            bb = (x1, y1, x2 - x1, y2 - y1)
            out.append(Track(int(tid), bb, float(cf), feature_fn=lambda b=bb, f=frame: hsv_hist(f, b)))
        return out


class DetFollower:
    def __init__(self, provider: YoloTracks, fps: float, name: str, frame_size=(1280, 720),
                 max_occlusion_s: float = 8.0):
        self.provider, self.name = provider, name
        self.mgr = TargetManager(fps, frame_size, max_occlusion_s=max_occlusion_s)
        self.lost = True
        self.occlusions: list[dict] = []

    def preview(self, frame) -> list[tuple]:
        """Boxes of every detected vehicle (for the pick-a-vehicle step)."""
        return [t.bbox for t in self.provider.tracks(frame, first=True)]

    def init(self, frame, bbox) -> None:
        tracks = self.provider.tracks(frame, first=True)
        self.mgr.start(tuple(float(v) for v in bbox), tracks)
        self.lost = False

    def update(self, frame, cam: tuple[float, float] = (0.0, 0.0)) -> FollowResult:
        if self.lost:
            return FollowResult(False, lost=True)
        tracks = self.provider.tracks(frame, first=False)
        st = self.mgr.update(tracks, cam)
        self.occlusions = self.mgr.occlusions
        self.lost = st.state == "LOST"
        others = [(t.id, t.bbox) for t in tracks if t.id != st.target_id]
        bbox = tuple(int(round(v)) for v in st.bbox) if st.bbox else None
        return FollowResult(bbox is not None, bbox, st.score, st.predicted, self.lost, st.target_id, st.std,
                            st.occluded_s, others, st.events)


# ------------------------------------------------------------------ single-object fallback
class VitFollower:
    """OpenCV VitTrack. It exposes a confidence score; CSRT/KCF do not (they keep drawing a box on the
    wrong object), so VitTrack is required for a trustworthy TRACKING/LOST status."""

    name = "vittrack"

    def __init__(self, score_thresh: float = 0.30, lost_frames: int = 6):
        if not hasattr(cv2, "TrackerVit_create"):
            raise RuntimeError("OpenCV build lacks TrackerVit; install opencv-contrib-python>=4.9")
        if not VIT_MODEL.is_file():
            raise RuntimeError(f"Missing tracker model: {VIT_MODEL}")
        self.score_thresh, self.lost_frames = score_thresh, lost_frames
        self.occlusions: list = []
        self.tracker = None
        self.low = 0
        self.lost = True

    def preview(self, frame) -> list[tuple]:
        return []

    def init(self, frame, bbox) -> None:
        params = cv2.TrackerVit_Params()
        params.net = str(VIT_MODEL)
        self.tracker = cv2.TrackerVit_create(params)
        self.tracker.init(frame, tuple(int(v) for v in bbox))
        self.low, self.lost = 0, False

    def update(self, frame, cam: tuple[float, float] = (0.0, 0.0)) -> FollowResult:
        if self.lost or self.tracker is None:
            return FollowResult(False, lost=True)
        ok, box = self.tracker.update(frame)
        score = float(self.tracker.getTrackingScore())
        h, w = frame.shape[:2]
        x, y, bw, bh = (int(v) for v in box)
        off = not (0 <= x + bw / 2 < w and 0 <= y + bh / 2 < h)
        self.low = self.low + 1 if (not ok or score < self.score_thresh) else 0
        if self.low >= self.lost_frames or off:
            self.lost = True
            return FollowResult(False, None, score, lost=True, events=[("err", "Target LOST (tracker confidence)")])
        return FollowResult(True, (x, y, bw, bh), score)


def snap_to_detection(point: tuple[int, int], boxes) -> tuple[int, int, int, int] | None:
    best = None
    for (x, y, w, h) in boxes:
        if x <= point[0] <= x + w and y <= point[1] <= y + h:
            if best is None or w * h < best[2] * best[3]:
                best = (int(x), int(y), int(w), int(h))
    return best


def build_follower(kind: str, fps: float, weights: str | None = None, imgsz: int = 1280, conf: float = 0.20,
                   max_occlusion_s: float = 8.0):
    """kind: auto | bytetrack | botsort | vit. 'auto' = YOLO+ByteTrack when it can be loaded, else VitTrack.
    Returns (follower, description)."""
    if kind == "vit":
        return VitFollower(), "OpenCV VitTrack"
    w = Path(weights) if weights else DEFAULT_WEIGHTS
    try:
        if not w.is_file():
            raise FileNotFoundError(f"detector weights not found: {w}")
        cfg = BOTSORT_CFG if kind == "botsort" else BYTETRACK_CFG
        provider = YoloTracks(w, cfg, imgsz, conf)
        label = "botsort" if kind == "botsort" else "bytetrack"
        return DetFollower(provider, fps, f"yolo+{label}", max_occlusion_s=max_occlusion_s), \
            f"YOLOv8 ({w.name}) + {'BoT-SORT' if kind == 'botsort' else 'ByteTrack'} + path prediction"
    except Exception as e:  # missing ultralytics/torch, blocked DLL, missing weights ...
        if kind != "auto":
            raise SystemExit(f"Cannot start the {kind} tracker: {e}")
        print(f"[tracker] YOLO backend unavailable ({type(e).__name__}: {e}); using OpenCV VitTrack instead.")
        return VitFollower(), "OpenCV VitTrack (YOLO backend unavailable)"
