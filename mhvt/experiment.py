"""Offline experiments: replay a recorded target path through the coordinator many times.

The vision tracker is deterministic, so it only has to run once per scenario; the coordinator
(which is where loss, latency and faults act) is replayed for every seed / loss level.
`replay()` mirrors main.App exactly, so a replay with the scenario's own seed, loss and events
reproduces the live run (tested)."""

from __future__ import annotations

import statistics

from .coordinator import Config, Coordinator
from .metrics import Center, summarize, wilson

NFR3_LOSS, NFR3_TARGET = 0.10, 0.90


def replay(zones_px, track_log: list[Center], fps: float, seed: int, loss: float,
           events: list[dict] | None = None, config: Config | None = None) -> Coordinator:
    """track_log[i] is the target centre after i tracked frames (None once lost)."""
    cfg = config or Config()
    cfg.seed, cfg.loss_prob = seed, loss
    c = Coordinator(zones_px, cfg)
    pending = sorted(events or [], key=lambda e: e["frame"])
    lost = False
    for i, center in enumerate(track_log):
        while i and pending and pending[0]["frame"] <= i - 1:  # same rule as App.advance_track
            _apply(c, pending.pop(0), i - 1, fps)
        t = i / fps
        if center is None and not lost:
            lost = True
            c.mark_target_lost(t)
        c.update(t, center)
    return c


def _apply(c: Coordinator, ev: dict, frames_done: int, fps: float) -> None:
    t = frames_done / fps  # same timestamp App.apply_event uses
    if ev["type"] == "loss":
        c.set_loss(float(ev["value"]))
        c.log(t, f"Message loss set to {int(float(ev['value']) * 100)}%", "warn")
    elif ev["type"] == "fault" and 1 <= ev["drone"] <= len(c.drones):
        c.set_silent(ev["drone"], bool(ev["silent"]), t)


def sweep(zones_px, track_log, fps, seeds: int, losses: list[float], fault_events=None) -> list[dict]:
    rows = []
    for loss in losses:
        for seed in range(seeds):
            c = replay(zones_px, track_log, fps, seed, loss, fault_events)
            m = summarize(c, track_log, fps)
            rows.append({"loss": loss, "seed": seed, **{k: v for k, v in m.items() if k != "owner_changes"}})
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    out = []
    for loss in sorted({r["loss"] for r in rows}):
        rs = [r for r in rows if r["loss"] == loss]
        ok = sum(r["handovers_ok"] for r in rs)
        failed = sum(r["handovers_failed"] for r in rs)
        n = ok + failed
        lat = [r["handover_latency_mean_s"] for r in rs if r["handover_latency_mean_s"] is not None]
        lo, hi = wilson(ok, n)
        pdr = [r["delivery_ratio"] for r in rs if r["delivery_ratio"] is not None]
        out.append({
            "loss": loss, "runs": len(rs), "handover_attempts": n, "ok": ok, "failed": failed,
            "success_rate": ok / n if n else None, "ci95_low": lo if n else None, "ci95_high": hi if n else None,
            "latency_mean_s": statistics.mean(lat) if lat else None,
            "delivery_ratio": statistics.mean(pdr) if pdr else None,
            "meets_nfr3": (ok / n >= NFR3_TARGET) if (n and abs(loss - NFR3_LOSS) < 1e-9) else None,
        })
    return out
