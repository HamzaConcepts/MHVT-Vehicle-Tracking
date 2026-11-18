"""Run metrics (report-style numbers) computed from a finished or running coordinator.

Definitions (see report FR6 / NFR1-3):
  tracking_continuity  share of the run during which the vision tracker still had the target
  coverage_continuity  share of the *tracked* time during which exactly one UAV owned the target
  handover success     committed handovers / (committed + failed)
  handover latency     handover start -> COMMIT
  delivery ratio       1 - dropped / sent coordination messages (simulated link)
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

from .coordinator import Coordinator

Center = tuple[float, float] | None


def owner_at(timeline: list[tuple[float, int | None]], t: float) -> int | None:
    owner = None
    for ts, o in timeline:
        if ts > t + 1e-9:
            break
        owner = o
    return owner


def summarize(coord: Coordinator, track_log: list[Center], fps: float) -> dict:
    n = len(track_log)
    tracked = [c is not None for c in track_log]
    tracked_frames = sum(tracked)
    lost_frame = next((i for i, ok in enumerate(tracked) if not ok), None)
    owned = sum(1 for i, ok in enumerate(tracked) if ok and owner_at(coord.timeline, i / fps) is not None)
    lat = [h["latency_s"] for h in coord.handover_log if h["ok"]]
    ok, failed = coord.ok, coord.failed
    sent, dropped = coord.msgs_sent, coord.msgs_lost
    return {
        "run_duration_s": round(n / fps, 3),
        "tracked_s": round(tracked_frames / fps, 3),
        "target_lost_at_s": None if lost_frame is None else round(lost_frame / fps, 3),
        "tracking_continuity": round(tracked_frames / n, 4) if n else None,
        "coverage_continuity": round(owned / tracked_frames, 4) if tracked_frames else None,
        "handovers_ok": ok,
        "handovers_failed": failed,
        "handovers_aborted_target_lost": coord.aborted,
        "handover_success_rate": round(ok / (ok + failed), 4) if ok + failed else None,
        "handover_latency_mean_s": round(statistics.mean(lat), 3) if lat else None,
        "handover_latency_max_s": round(max(lat), 3) if lat else None,
        "messages_sent": sent,
        "messages_dropped": dropped,
        "delivery_ratio": round(1 - dropped / sent, 4) if sent else None,
        "owner_changes": [{"t": round(t, 3), "owner": o} for t, o in coord.timeline],
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def write_report(path: Path, meta: dict, coord: Coordinator, track_log: list[Center], fps: float) -> Path:
    """Writes <path> (JSON) and <path stem>.handovers.csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"meta": meta, "metrics": summarize(coord, track_log, fps),
              "handovers": coord.handover_log,
              "log": [{"t": round(e.t, 3), "level": e.level, "text": e.text.strip()} for e in coord.events]}
    path.write_text(json.dumps(report, indent=1))
    rows = coord.handover_log
    with open(path.with_suffix(".handovers.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id"])
        w.writeheader()
        w.writerows(rows)
    return path
