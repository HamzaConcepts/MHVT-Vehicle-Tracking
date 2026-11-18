import math

import numpy as np

from mhvt.prediction import MotionModel, TargetManager, Track, appearance_distance, center

FPS = 24.0
W, H = 1280, 720
RED = np.array([0.7, 0.2, 0.1] + [0.0] * 5)
BLUE = np.array([0.0, 0.1, 0.1, 0.1, 0.7] + [0.0] * 3)


def car(i, x0=100.0, y=300.0, vx=6.0, w=50.0, h=28.0):
    """Centre position of a car at frame i, as a bbox."""
    cx = x0 + vx * i
    return (cx - w / 2, y - h / 2, w, h)


def run(mgr, scenario, n):
    """scenario(i) -> list[Track]. Returns the list of statuses."""
    return [mgr.update(scenario(i)) for i in range(1, n + 1)]


def make(tracks0=None, bbox=None, **kw):
    mgr = TargetManager(FPS, (W, H), **kw)
    mgr.start(bbox or car(0), tracks0 if tracks0 is not None else [Track(7, car(0), 0.9, RED)])
    return mgr


def test_motion_model_learns_velocity_and_predicts():
    m = MotionModel(100, 300)
    for i in range(1, 30):
        m.predict()
        m.update(100 + 6 * i, 300 + 1 * i)
    assert abs(m.vel[0] - 6) < 0.3 and abs(m.vel[1] - 1) < 0.3
    for _ in range(20):
        m.predict()
    assert abs(m.pos[0] - (100 + 6 * 49)) < 15
    assert m.std() > 3  # uncertainty grows while only predicting


def test_bridge_prediction_and_reacquisition_with_new_id():
    # visible 0-40, hidden 41-70 (under a bridge), reappears as a *new* track id 21
    def scen(i):
        if i <= 40:
            return [Track(7, car(i), 0.9, RED)]
        if i <= 70:
            return []
        return [Track(21, car(i), 0.8, RED)]

    mgr = make()
    st = run(mgr, scen, 90)
    hidden = st[40:70]
    assert all(s.state == "PREDICTING" and s.predicted for s in hidden)
    # dead-reckoned position stays close to the truth under the bridge
    errs = [math.dist(center(s.bbox), center(car(i))) for i, s in zip(range(41, 71), hidden)]
    assert max(errs) < 25, max(errs)
    assert st[-1].state == "TRACKING" and st[-1].target_id == 21
    assert any("re-acquired" in e[1] for s in st for e in s.events)
    assert mgr.occlusions[0]["reacquired"] and mgr.occlusions[0]["new_id"] == 21
    assert 1.0 < mgr.occlusions[0]["duration_s"] < 1.6


def test_decoy_with_different_appearance_is_rejected():
    # at the exit two vehicles appear: a blue decoy right at the predicted point, the red target slightly off
    def scen(i):
        if i <= 40:
            return [Track(7, car(i), 0.9, RED)]
        if i <= 70:
            return []
        t = car(i)
        decoy = (t[0] + 4, t[1] - 3, t[2], t[3])
        offset = (t[0] - 18, t[1] + 12, t[2], t[3])
        return [Track(30, decoy, 0.9, BLUE), Track(31, offset, 0.8, RED)]

    st = run(make(), scen, 90)
    assert st[-1].target_id == 31


def test_established_other_vehicle_is_never_hijacked():
    # a different car is visible for the whole time right where the target "should" be
    other = lambda i: Track(99, car(i, x0=60, vx=6.0), 0.9, RED)  # follows the same path 40 px behind

    def scen(i):
        if i <= 40:
            return [Track(7, car(i), 0.9, RED), other(i)]
        return [other(i)]

    mgr = make(tracks0=[Track(7, car(0), 0.9, RED), other(0)])
    st = run(mgr, scen, 60)
    assert st[-1].state == "PREDICTING" and st[-1].target_id == 7


def test_wrong_direction_candidate_is_rejected():
    def scen(i):
        if i <= 40:
            return [Track(7, car(i), 0.9, RED)]
        if i <= 70:
            return []
        # appears near the prediction but its implied path came from the front, not from behind
        t = car(71 + 40)  # far ahead of the prediction, then drives back towards it
        return [Track(40, (t[0] - 6 * (i - 71), t[1], t[2], t[3]), 0.9, RED)]

    st = run(make(), scen, 80)
    assert all(s.target_id != 40 for s in st)


def test_same_id_returning_resumes_immediately():
    def scen(i):
        return [] if 41 <= i <= 55 else [Track(7, car(i), 0.9, RED)]

    st = run(make(), scen, 70)
    assert st[-1].state == "TRACKING" and st[-1].target_id == 7
    assert any("same ID" in e[1] for s in st for e in s.events)


def test_id_switch_without_occlusion_is_recovered_quickly():
    def scen(i):
        return [Track(7 if i <= 30 else 55, car(i), 0.9, RED)]

    st = run(make(), scen, 40)
    assert st[-1].state == "TRACKING" and st[-1].target_id == 55
    assert sum(s.predicted for s in st) <= 4  # a few frames of prediction at most


def test_timeout_when_never_found_again():
    def scen(i):
        return [Track(7, car(i), 0.9, RED)] if i <= 20 else []

    mgr = make(max_occlusion_s=2.0)
    st = run(mgr, scen, 120)
    assert st[-1].state == "LOST" and st[-1].bbox is None
    assert any("not found again" in e[1] for s in st for e in s.events)
    assert mgr.update([]).state == "LOST"  # LOST is final


def test_prediction_leaving_the_picture_is_lost():
    def scen(i):  # fast car drives off the right edge while still "tracked", then vanishes
        return [Track(7, car(i, x0=1100, vx=12.0), 0.9, RED)] if i <= 10 else []

    st = run(make(tracks0=[Track(7, car(0, x0=1100, vx=12.0), 0.9, RED)], bbox=car(0, x0=1100, vx=12.0)), scen, 60)
    assert st[-1].state == "LOST"
    assert any("left the picture" in e[1] for s in st for e in s.events)


def test_start_without_a_track_binds_to_first_nearby_vehicle():
    mgr = make(tracks0=[], bbox=car(0))

    def scen(i):
        return [Track(5, car(i, vx=0.0), 0.8, None)] if i >= 3 else []

    st = run(mgr, scen, 8)
    assert st[-1].state == "TRACKING" and st[-1].target_id == 5


def test_appearance_distance():
    assert appearance_distance(RED, RED) < 1e-6
    assert appearance_distance(RED, BLUE) > 0.7
    assert appearance_distance(None, RED) is None


# ---------------------------------------------------------------- camera motion, lanes, ambiguity
def cam_at(i):
    """Camera shift of the scene in the picture: steady -8 px/frame, then the drone speeds up to -22 px/frame
    while the vehicle is hidden."""
    if i <= 40:
        return (-8.0 * i, 0.0)
    return (-8.0 * 40 - 22.0 * (i - 40), 0.0)


def scene_car(i, w=50.0, h=28.0):
    """Vehicle drifting through the *scene* at +3 px/frame; returns its bbox in picture coordinates."""
    cx = 2000 + 3.0 * i + cam_at(i)[0]
    return (cx - w / 2, 300 - h / 2, w, h)


def test_camera_compensation_keeps_prediction_accurate_when_the_drone_accelerates():
    def scen(i):
        if i <= 40:
            return [Track(7, scene_car(i), 0.9, RED)]
        return [] if i <= 62 else [Track(21, scene_car(i), 0.8, RED)]

    def run_with(use_cam):
        mgr = TargetManager(FPS, (4000, 720))
        mgr.start(scene_car(0), [Track(7, scene_car(0), 0.9, RED)])
        out = [mgr.update(scen(i), cam_at(i) if use_cam else (0.0, 0.0)) for i in range(1, 80)]
        return mgr, out

    mgr, st = run_with(True)
    errs = [math.dist(center(s.bbox), center(scene_car(i))) for i, s in zip(range(41, 63), st[40:62])]
    assert max(errs) < 20, max(errs)
    assert st[-1].state == "TRACKING" and st[-1].target_id == 21

    # without the camera information the same prediction is hopelessly off (the drone sped up)
    _, st_no = run_with(False)
    errs_no = [math.dist(center(s.bbox), center(scene_car(i))) for i, s in zip(range(41, 63), st_no[40:62]) if s.bbox]
    assert errs_no and max(errs_no) > 100


def test_vehicle_in_the_next_lane_is_not_mistaken_for_the_target():
    # no appearance information at all: only position / lane decides
    def scen(i):
        if i <= 40:
            return [Track(7, car(i), 0.9, None)]
        if i <= 70:
            return []
        t = car(i)
        return [Track(60, (t[0] + 10, t[1] - 62, t[2], t[3]), 0.9, None),   # adjacent lane, slightly ahead
                Track(61, (t[0] - 14, t[1] + 3, t[2], t[3]), 0.9, None)]    # same lane

    st = run(make(tracks0=[Track(7, car(0), 0.9, None)]), scen, 90)
    assert st[-1].target_id == 61


def test_two_equally_plausible_vehicles_refuse_to_guess():
    def scen(i):
        if i <= 40:
            return [Track(7, car(i), 0.9, None)]
        if i <= 70:
            return []
        t = car(i)
        return [Track(70, (t[0] - 12, t[1] - 2, t[2], t[3]), 0.9, None),
                Track(71, (t[0] + 12, t[1] + 2, t[2], t[3]), 0.9, None)]  # mirror images around the prediction

    mgr = make(tracks0=[Track(7, car(0), 0.9, None)])
    st = run(mgr, scen, 90)
    assert st[-1].state == "PREDICTING" and st[-1].target_id == 7  # waits instead of picking one
    assert mgr.ambiguous_frames > 0
