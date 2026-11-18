import math

from mhvt.coordinator import auto_zones
from mhvt.experiment import aggregate, replay, sweep
from mhvt.metrics import summarize, wilson

ZONES = auto_zones(1280, 720, 4)
FPS = 24.0
TRACK = [(50 + 1180 * i / 479, 360.0) for i in range(480)]  # sweep across all zones in 20 s


def test_wilson_bounds():
    lo, hi = wilson(9, 10)
    assert 0 <= lo < 0.9 < hi <= 1
    assert wilson(0, 0)[0] != wilson(0, 0)[0]  # NaN when no data
    assert wilson(10, 10)[1] == 1.0


def test_replay_is_deterministic_and_summarises():
    a = summarize(replay(ZONES, TRACK, FPS, 5, 0.1), TRACK, FPS)
    b = summarize(replay(ZONES, TRACK, FPS, 5, 0.1), TRACK, FPS)
    assert a == b
    assert a["tracking_continuity"] == 1.0
    assert a["handovers_ok"] + a["handovers_failed"] >= 3


def test_target_lost_lowers_continuity_and_aborts():
    log = TRACK[:200] + [None] * 100
    m = summarize(replay(ZONES, log, FPS, 1, 0.0), log, FPS)
    assert math.isclose(m["tracking_continuity"], 200 / 300, abs_tol=1e-3)
    assert m["target_lost_at_s"] == round(200 / FPS, 3)


def test_sweep_meets_nfr3_on_clean_path():
    rows = sweep(ZONES, TRACK, FPS, 40, [0.0, 0.10])
    agg = {r["loss"]: r for r in aggregate(rows)}
    assert agg[0.0]["success_rate"] == 1.0
    assert agg[0.10]["success_rate"] >= 0.90 and agg[0.10]["meets_nfr3"] is True
    assert agg[0.10]["delivery_ratio"] < 1.0


def test_fault_events_are_applied():
    ev = [{"frame": 5, "type": "fault", "drone": 2, "silent": True}]
    c = replay(ZONES, TRACK, FPS, 1, 0.0, ev)
    assert any("FAULT injected: UAV-2" in e.text for e in c.events)
