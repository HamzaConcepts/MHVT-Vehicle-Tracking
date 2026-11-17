import random

from mhvt.coordinator import Config, Coordinator, DState, auto_zones

ZONES = auto_zones(1280, 720, 4)  # 4 overlapping vertical bands


def run(coord, path, fps=24.0, start=0.0):
    """Drive the coordinator along a list of (x, y) target positions."""
    for i, p in enumerate(path):
        t = start + i / fps
        coord.update(t, p)
        owners = [d for d in coord.drones if d.state in (DState.TRACKING, DState.HANDING_OVER)]
        assert len(owners) <= 1, "more than one primary owner"  # FR4 / NFR4
    return start + len(path) / fps


def sweep(x0, x1, n, y=360):
    return [(x0 + (x1 - x0) * i / (n - 1), y) for i in range(n)]


def test_initial_assignment_picks_covering_drone():
    c = Coordinator(ZONES, Config(seed=1))
    run(c, [(100, 300)] * 5)
    assert c.owner == 1


def test_successful_handover_across_all_zones():
    c = Coordinator(ZONES, Config(seed=1))
    run(c, sweep(50, 1230, 24 * 20))
    assert c.owner == 4
    assert c.ok == 3 and c.failed == 0
    assert c.summary()["mean_latency_s"] < 1.5


def test_handover_survives_10_percent_loss_most_of_the_time():  # NFR3 >= 90%
    ok = failed = 0
    for seed in range(30):
        c = Coordinator(ZONES, Config(seed=seed, loss_prob=0.10))
        run(c, sweep(50, 1230, 24 * 20))
        ok += c.ok
        failed += c.failed
    assert ok / (ok + failed) >= 0.90


def test_dead_successor_triggers_fallback_or_safe_failure():
    # Overlapping 3-zone setup; kill the best candidate -> single owner is kept
    c = Coordinator(auto_zones(1280, 720, 3, overlap=0.6), Config(seed=2))
    c.update(0.0, (100, 300))
    c.set_silent(2, True, 0.0)
    run(c, sweep(100, 700, 24 * 15), start=0.04)
    owners = [d.id for d in c.drones if d.state == DState.TRACKING]
    assert len(owners) == 1
    assert c.drone(2).state == DState.OFFLINE  # declared after ~3 s (NFR5)


def test_total_message_loss_fails_safely():
    c = Coordinator(ZONES, Config(seed=3, loss_prob=1.0))
    run(c, sweep(50, 400, 60))  # < 3 s, before beacons time out
    assert c.ok == 0
    # sending drone still owns the target; no silent ownership change
    assert c.owner == 1


def test_target_lost_and_reselect():
    c = Coordinator(ZONES, Config(seed=1))
    run(c, [(100, 300)] * 5)
    c.mark_target_lost(1.0)
    assert c.drone(1).state == DState.SEARCHING
    c.reset_target()
    run(c, [(900, 300)] * 5, start=2.0)
    assert c.owner in (3, 4)


def test_low_battery_owner_hands_over_when_possible():
    c = Coordinator(auto_zones(1280, 720, 2, overlap=0.8), Config(seed=1))
    run(c, [(640, 300)] * 3)
    owner = c.drone(c.owner)
    owner.battery = 10.0
    run(c, [(640, 300)] * 24 * 3, start=1.0)
    assert c.owner != owner.id


def test_reproducible_with_same_seed():
    def go():
        c = Coordinator(ZONES, Config(seed=11, loss_prob=0.2))
        run(c, sweep(50, 1230, 24 * 20))
        return [(round(e.t, 3), e.text) for e in c.events]
    assert go() == go()
