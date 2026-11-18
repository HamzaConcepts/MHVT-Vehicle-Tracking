"""Automatic zone layout: equal-area, overlapping UAV zones without drawing them by hand.

Layouts
  vertical    n bands left -> right   (best when traffic flows horizontally in the picture)
  horizontal  n bands top -> bottom   (traffic flows vertically)
  grid        rows x cols cells       (4 = 2x2, 6 = 2x3)
  auto        picks vertical/horizontal from the dominant motion in the video

Every zone has exactly the same size. With overlap fraction o, n zones of width w tile a length L as
    n*w - (n-1)*o*w = L   ->   w = L / (n - (n-1)*o)
so neighbours share o*w and the first/last zones are not clipped smaller than the rest.
"""

from __future__ import annotations

from pathlib import Path

Rect = tuple[float, float, float, float]
LAYOUTS = ("vertical", "horizontal", "grid", "auto")
GRIDS = {4: (2, 2), 6: (2, 3)}
MAX_ZONES = 6


def _spans(length: float, n: int, overlap: float) -> list[tuple[float, float]]:
    if n == 1:
        return [(0.0, length)]
    w = length / (n - (n - 1) * overlap)
    step = w * (1 - overlap)
    return [(i * step, i * step + w) for i in range(n)]


def equal_zones(width: float, height: float, n: int, layout: str = "vertical", overlap: float = 0.15) -> list[Rect]:
    if not 1 <= n <= MAX_ZONES:
        raise ValueError(f"zone count must be 1..{MAX_ZONES}")
    if not 0.0 <= overlap < 0.9:
        raise ValueError("overlap must be in [0, 0.9)")
    if layout == "grid" and n in GRIDS:
        rows, cols = GRIDS[n]
        ys, xs = _spans(height, rows, overlap), _spans(width, cols, overlap)
        return [(x1, y1, x2, y2) for (y1, y2) in ys for (x1, x2) in xs]
    if layout == "horizontal":
        return [(0.0, a, width, b) for a, b in _spans(height, n, overlap)]
    return [(a, 0.0, b, height) for a, b in _spans(width, n, overlap)]  # vertical (and grid fallback)


def zones_area_equal(zones: list[Rect], tol: float = 1e-6) -> bool:
    areas = [(z[2] - z[0]) * (z[3] - z[1]) for z in zones]
    return max(areas) - min(areas) <= tol * max(areas)


def dominant_axis(prev_gray, gray, top_fraction: float = 0.08) -> str:
    """'x' or 'y': the axis along which the fastest-moving pixels move between two frames."""
    import cv2
    import numpy as np

    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag = np.hypot(flow[..., 0], flow[..., 1])
    k = max(1, int(mag.size * top_fraction))
    idx = np.argpartition(mag.ravel(), -k)[-k:]
    fx, fy = np.abs(flow[..., 0]).ravel()[idx].sum(), np.abs(flow[..., 1]).ravel()[idx].sum()
    return "x" if fx >= fy else "y"


def suggest_layout_from_video(path: Path, pairs: int = 5, gap: int = 3) -> str:
    """Looks at a few frame pairs across the clip and returns 'vertical' (traffic moves along x)
    or 'horizontal' (traffic moves along y). Falls back to 'vertical'."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    votes = {"x": 0, "y": 0}
    for i in range(pairs):
        start = int((i + 0.5) / pairs * max(1, n - gap - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        ok1, a = cap.read()
        for _ in range(gap - 1):
            cap.read()
        ok2, b = cap.read()
        if not (ok1 and ok2):
            continue
        g = [cv2.cvtColor(cv2.resize(f, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY) for f in (a, b)]
        votes[dominant_axis(*g)] += 1
    cap.release()
    return "horizontal" if votes["y"] > votes["x"] else "vertical"
