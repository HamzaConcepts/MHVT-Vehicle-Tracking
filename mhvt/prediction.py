"""Path prediction and target re-acquisition (pure numpy, no OpenCV/YOLO).

When the followed vehicle disappears (drives under a bridge, behind a building, or the tracker simply
changes its ID) the system must not just give up. `TargetManager` keeps a Kalman motion model of the
target and, while the detector cannot see it,

  * PREDICTS where it should be (dead reckoning, with a growing uncertainty radius), and
  * RE-ACQUIRES it when a plausible vehicle appears, then re-binds the target to that tracker ID.

All positions are handled in *scene* coordinates (picture position minus the accumulated camera shift
`cam`), so the drone speeding up or slowing down does not corrupt the prediction.

A candidate is accepted only if it (1) lies inside the uncertainty gate - narrow sideways (vehicles keep
their lane), wide along the road (speed changes) - 
(2) has a similar size, (3) arrived from the right direction, (4) looks like the target (colour
histogram, when available), (5) is *new* - a vehicle that was already being tracked continuously
is another car, not the target - and (6) keeps passing for `confirm_frames` consecutive frames, and (7) is clearly better than the
runner-up: if two vehicles look about equally plausible the manager refuses to guess.
If nothing plausible appears within `max_occlusion_s`, or the prediction leaves the picture,
the target is declared LOST (final).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

BBox = tuple[float, float, float, float]  # x, y, w, h


@dataclass
class Track:
    id: int
    bbox: BBox
    conf: float = 1.0
    feature: np.ndarray | None = None
    feature_fn: Callable[[], np.ndarray | None] | None = None  # computed lazily (appearance is costly)

    def get_feature(self) -> np.ndarray | None:
        if self.feature is None and self.feature_fn is not None:
            self.feature = self.feature_fn()
            self.feature_fn = None
        return self.feature


def center(b: BBox) -> tuple[float, float]:
    return b[0] + b[2] / 2, b[1] + b[3] / 2


def iou(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def appearance_distance(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """Bhattacharyya distance between two normalised histograms (0 = identical, 1 = disjoint)."""
    if a is None or b is None:
        return None
    bc = float(np.sum(np.sqrt(np.clip(a, 0, None) * np.clip(b, 0, None))))
    return math.sqrt(max(0.0, 1.0 - bc))


class MotionModel:
    """Constant-velocity Kalman filter on the target centre, in picture pixels per frame."""

    F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)

    def __init__(self, cx: float, cy: float, accel_std: float = 0.35, meas_std: float = 3.0,
                 lateral_accel_std: float = 0.10):
        self.x = np.array([cx, cy, 0.0, 0.0])
        self.P = np.diag([meas_std ** 2, meas_std ** 2, 25.0, 25.0])  # velocity unknown at the start
        self.qa, self.ql = accel_std, lateral_accel_std
        self.R = np.eye(2) * meas_std ** 2

    def _Q(self) -> np.ndarray:
        """Process noise. Vehicles speed up and brake along the road (accel_std) but hold their lane, so
        sideways acceleration (lateral_accel_std) is much smaller. Below 1 px/frame there is no direction."""
        vx, vy = self.x[2], self.x[3]
        sp = math.hypot(vx, vy)
        if sp < 1.0:
            A = np.eye(2) * self.qa ** 2
        else:
            u = np.array([vx, vy]) / sp
            Rm = np.array([u, [-u[1], u[0]]])
            A = Rm.T @ np.diag([self.qa ** 2, self.ql ** 2]) @ Rm
        return np.block([[A / 4, A / 2], [A / 2, A]])

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self._Q()

    def update(self, cx: float, cy: float) -> None:
        z = np.array([cx, cy])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P

    @property
    def pos(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def vel(self) -> tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

    def std(self) -> float:
        return float(math.sqrt(max(np.linalg.eigvalsh(self.P[:2, :2]).max(), 0.0)))

    def mahalanobis2(self, cx: float, cy: float, extra_var: float = 0.0) -> float:
        S = self.P[:2, :2] + self.R + np.eye(2) * extra_var
        d = np.array([cx - self.x[0], cy - self.x[1]])
        return float(d @ np.linalg.solve(S, d))

    def gate_d2(self, cx: float, cy: float, size: float, along_tol: float = 0.30, lat_tol: float = 0.12) -> float:
        """Distance of (cx, cy) from the prediction in units of its uncertainty, measured along / across the
        direction of travel: tolerant along the road (speed changes), strict across it (lane keeping)."""
        vx, vy = self.vel
        sp = math.hypot(vx, vy)
        d = np.array([cx - self.x[0], cy - self.x[1]])
        S = self.P[:2, :2] + self.R
        if sp < 1.0:
            return float(d @ np.linalg.solve(S + np.eye(2) * (0.35 * size) ** 2, d))
        u = np.array([vx, vy]) / sp
        Rm = np.array([u, [-u[1], u[0]]])
        Sr = Rm @ S @ Rm.T + np.diag([(along_tol * size) ** 2, (lat_tol * size) ** 2])
        dr = Rm @ d
        return float(dr @ np.linalg.solve(Sr, dr))


@dataclass
class TargetStatus:
    state: str                       # TRACKING | PREDICTING | LOST
    bbox: BBox | None
    predicted: bool
    std: float = 0.0                 # position uncertainty radius (px), for drawing
    target_id: int | None = None
    occluded_s: float = 0.0
    score: float = 0.0
    events: list[tuple[str, str]] = field(default_factory=list)  # (level, text)

    @property
    def center(self) -> tuple[float, float] | None:
        return center(self.bbox) if self.bbox else None


class TargetManager:
    def __init__(self, fps: float, frame_size: tuple[int, int] = (1280, 720), max_occlusion_s: float = 8.0,
                 confirm_frames: int = 4, gate_chi2: float = 9.2, accept_cost: float = 0.45,
                 appearance_max: float = 0.5, size_ratio: tuple[float, float] = (0.7, 1.45),
                 min_direction_cos: float = 0.2, accel_std: float = 0.35, announce_after_s: float = 0.25,
                 ambiguity_margin: float = 0.08, lateral_accel_std: float = 0.10):
        self.fps = fps
        self.frame_size = frame_size
        self.max_occlusion_s = max_occlusion_s
        self.confirm_frames = confirm_frames
        self.gate_chi2 = gate_chi2
        self.accept_cost = accept_cost
        self.appearance_max = appearance_max
        self.size_ratio = size_ratio
        self.min_direction_cos = min_direction_cos
        self.accel_std = accel_std
        self.announce_after_s = announce_after_s
        self.ambiguity_margin = ambiguity_margin
        self.lateral_accel_std = lateral_accel_std
        self.ambiguous_frames = 0
        self.cam = (0.0, 0.0)
        self.announced = True
        self.occlusions: list[dict] = []   # one record per occlusion (for the run report)
        self.state = "LOST"
        self.target_id: int | None = None

    # ------------------------------------------------------------------ setup
    def _s(self, c: tuple[float, float]) -> tuple[float, float]:
        """picture position -> scene position (camera motion removed)"""
        return c[0] - self.cam[0], c[1] - self.cam[1]

    def start(self, bbox: BBox, tracks: list[Track], cam: tuple[float, float] = (0.0, 0.0)) -> None:
        self.cam = cam
        self.frame = 0
        self.first_seen = {t.id: 0 for t in tracks}
        cx, cy = self._s(center(bbox))
        self.model = MotionModel(cx, cy, self.accel_std, lateral_accel_std=self.lateral_accel_std)
        self.w, self.h = float(bbox[2]), float(bbox[3])
        self.occlusions = []
        self.pending: dict[int, int] = {}
        self.prev_centers: dict[int, tuple[float, float]] = {}
        self.feature: np.ndarray | None = None
        self.last_obs = (cx, cy)
        self.last_obs_frame = 0
        self.occ_start = 0
        self.init_mode = False
        self.target_id = None
        self.state = "TRACKING"
        best, best_iou = None, 0.0
        for t in tracks:
            v = iou(bbox, t.bbox)
            tx, ty = center(t.bbox)  # picture coordinates (same frame as bbox)
            inside = bbox[0] <= tx <= bbox[0] + bbox[2] and bbox[1] <= ty <= bbox[1] + bbox[3]
            if v > best_iou or (best is None and inside):
                best, best_iou = t, v
        if best is not None and (best_iou >= 0.2 or best_iou == 0.0):
            self.target_id = best.id
            self.feature = best.get_feature()
        else:  # nothing under the click: bind to the first plausible vehicle that appears nearby
            self.state, self.init_mode, self.occ_start = "PREDICTING", True, 0
            self.model.P[2:, 2:] = np.diag([4.0, 4.0])

    # ----------------------------------------------------------------- update
    def update(self, tracks: list[Track], cam: tuple[float, float] = (0.0, 0.0)) -> TargetStatus:
        if self.state == "LOST":
            return self._status([])
        self.cam = cam
        prev_centers, self.prev_centers = self.prev_centers, {t.id: self._s(center(t.bbox)) for t in tracks}
        self._prev = prev_centers
        self.frame += 1
        for t in tracks:
            self.first_seen.setdefault(t.id, self.frame)
        self.model.predict()
        events: list[tuple[str, str]] = []
        by_id = {t.id: t for t in tracks}

        if self.state == "TRACKING":
            tr = by_id.get(self.target_id)
            if tr is not None and self._near_prediction(tr, loose=True):
                self._observe(tr)
                return self._status(events, score=tr.conf)
            self.state, self.occ_start = "PREDICTING", self.frame
            self.pending = {}
            self.occlusions.append({"start_frame": self.frame, "lost_id": self.target_id, "reacquired": False,
                                    "duration_s": None, "new_id": None})
            self.announced = False

        # ---- PREDICTING
        if not self.announced and (self.frame - self.occ_start) / self.fps >= self.announce_after_s:
            self.announced = True  # only talk about real occlusions, not one-frame detector flicker
            events.append(("warn", f"Target #{self.target_id} not visible - predicting its path"))
        tr = by_id.get(self.target_id)
        if tr is not None and self._near_prediction(tr, loose=True) and not self.init_mode:
            return self._reacquired(tr, tr.id, events, same=True)
        cand = self._best_candidate(tracks)
        if cand is not None:
            tr, cost = cand
            self.pending = {tr.id: self.pending.get(tr.id, 0) + 1}
            need = 1 if self.init_mode else self.confirm_frames
            if self.pending[tr.id] >= need:
                return self._reacquired(tr, tr.id, events, same=False)
        else:
            self.pending = {}
        occ_s = (self.frame - self.occ_start) / self.fps
        cx, cy = self.model.pos[0] + self.cam[0], self.model.pos[1] + self.cam[1]  # picture coordinates
        W, H = self.frame_size
        margin = 0.5 * max(self.w, self.h)
        if not (-margin <= cx <= W + margin and -margin <= cy <= H + margin):
            return self._lose("predicted path left the picture", events, occ_s)
        if occ_s > self.max_occlusion_s:
            return self._lose(f"not found again within {self.max_occlusion_s:.0f}s", events, occ_s)
        return self._status(events)

    # --------------------------------------------------------------- internals
    def _size(self) -> float:
        return max(1.0, math.sqrt(self.w * self.h))

    def _near_prediction(self, t: Track, loose: bool = False) -> bool:
        cx, cy = self._s(center(t.bbox))
        return self.model.gate_d2(cx, cy, self._size()) <= self.gate_chi2 * (3 if loose else 1)

    def _observe(self, t: Track) -> None:
        cx, cy = self._s(center(t.bbox))
        self.model.update(cx, cy)
        self.w = 0.8 * self.w + 0.2 * t.bbox[2]
        self.h = 0.8 * self.h + 0.2 * t.bbox[3]
        self.last_obs, self.last_obs_frame = (cx, cy), self.frame
        if t.conf >= 0.4:
            f = t.get_feature()
            if f is not None:
                self.feature = f if self.feature is None else 0.9 * self.feature + 0.1 * f

    def _best_candidate(self, tracks: list[Track]):
        scored: list[tuple[float, Track]] = []
        n = max(1, self.frame - self.last_obs_frame)
        vx, vy = self.model.vel
        speed = math.hypot(vx, vy)
        for t in tracks:
            if t.id == self.target_id:
                continue
            if not self.init_mode and self.first_seen.get(t.id, self.frame) < self.occ_start - 1:
                continue  # an already-established vehicle is another car, not the target
            cx, cy = self._s(center(t.bbox))
            d2 = self.model.gate_d2(cx, cy, self._size())
            if d2 > self.gate_chi2:
                continue
            ratio = math.sqrt(max(t.bbox[2] * t.bbox[3], 1e-6)) / self._size()
            if not self.size_ratio[0] <= ratio <= self.size_ratio[1]:
                continue
            if speed > 1.0 and not self.init_mode:
                ix, iy = (cx - self.last_obs[0]) / n, (cy - self.last_obs[1]) / n
                im = math.hypot(ix, iy)
                if im > 0.5 and (ix * vx + iy * vy) / (im * speed) < self.min_direction_cos:
                    continue  # it came from the wrong direction
            pc = self._prev.get(t.id)
            if pc is not None and speed > 1.0:  # the candidate's own motion must match the target's direction
                cvx, cvy = cx - pc[0], cy - pc[1]
                cs = math.hypot(cvx, cvy)
                if cs > 1.0 and (cvx * vx + cvy * vy) / (cs * speed) < self.min_direction_cos:
                    continue
            app = appearance_distance(self.feature, t.get_feature()) if self.feature is not None else None
            if app is not None and app > self.appearance_max:
                continue
            size_term = abs(math.log(ratio)) / math.log(self.size_ratio[1])
            pos_term = d2 / self.gate_chi2
            cost = (0.55 * pos_term + 0.30 * app + 0.15 * size_term) if app is not None else (0.75 * pos_term + 0.25 * size_term)
            if cost <= self.accept_cost:
                scored.append((cost, t))
        if not scored:
            return None
        scored.sort(key=lambda c: c[0])
        if len(scored) > 1 and not self.init_mode and scored[1][0] - scored[0][0] < self.ambiguity_margin:
            self.ambiguous_frames += 1  # two vehicles look equally plausible: do not guess
            return None
        return scored[0][1], scored[0][0]

    def _reacquired(self, t: Track, new_id: int, events, same: bool) -> TargetStatus:
        occ_s = (self.frame - self.occ_start) / self.fps
        old = self.target_id
        self.target_id = new_id
        self.init_mode = False
        self.state = "TRACKING"
        self.pending = {}
        self._observe(t)
        if self.occlusions:
            self.occlusions[-1].update(reacquired=True, duration_s=round(occ_s, 3), new_id=new_id)
        what = "same ID" if same else f"ID {old} -> {new_id}"
        if self.announced:
            events.append(("ok", f"Target re-acquired after {occ_s:.1f}s ({what})"))
        return self._status(events, score=t.conf)

    def _lose(self, why: str, events, occ_s: float) -> TargetStatus:
        self.state = "LOST"
        if self.occlusions and not self.occlusions[-1]["reacquired"]:
            self.occlusions[-1]["duration_s"] = round(occ_s, 3)
        events.append(("err", f"Target LOST: {why}"))
        return self._status(events)

    def _status(self, events, score: float = 0.0) -> TargetStatus:
        if self.state == "LOST":
            return TargetStatus("LOST", None, False, 0.0, self.target_id, 0.0, 0.0, events)
        cx, cy = self.model.pos[0] + self.cam[0], self.model.pos[1] + self.cam[1]
        bbox = (cx - self.w / 2, cy - self.h / 2, self.w, self.h)
        pred = self.state == "PREDICTING"
        occ = (self.frame - self.occ_start) / self.fps if pred else 0.0
        return TargetStatus(self.state, bbox, pred, self.model.std() if pred else 0.0, self.target_id, occ, score, events)
