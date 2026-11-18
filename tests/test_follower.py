import numpy as np

from mhvt.prediction import Track
from mhvt.tracking import DetFollower, FollowResult, snap_to_detection

FPS = 24.0
FRAME = np.zeros((720, 1280, 3), np.uint8)


class FakeProvider:
    """Stands in for YOLO + ByteTrack: returns a scripted list of tracks per call."""

    def __init__(self, script):
        self.script, self.calls, self.firsts = script, 0, []

    def tracks(self, frame, first):
        self.firsts.append(first)
        out = self.script(self.calls)
        self.calls += 1
        return out


def bbox(i, vx=6.0, w=50.0, h=28.0, y=300.0, x0=100.0):
    cx = x0 + vx * i
    return (cx - w / 2, y - h / 2, w, h)


def follower(script, **kw):
    f = DetFollower(FakeProvider(script), FPS, "yolo+bytetrack", **kw)
    f.init(FRAME, bbox(0))
    return f


def test_follows_the_clicked_vehicle_and_lists_the_others():
    other = Track(9, (900, 100, 50, 28), 0.8)
    f = follower(lambda i: [Track(7, bbox(i), 0.9), other])
    r = f.update(FRAME)
    assert isinstance(r, FollowResult) and r.ok and not r.predicted and not r.lost
    assert r.target_id == 7 and [o[0] for o in r.others] == [9]
    assert f.provider.firsts[0] is True and f.provider.firsts[1] is False  # tracker restarted on init only


def test_bridge_gives_predicted_positions_then_rebinds_to_the_new_id():
    def script(i):
        if i <= 30:
            return [Track(7, bbox(i), 0.9)]
        return [] if i <= 50 else [Track(21, bbox(i), 0.8)]

    f = follower(script)
    res = [f.update(FRAME) for _ in range(70)]
    mid = res[40]
    assert mid.ok and mid.predicted and mid.occluded_s > 0.2 and mid.std > 0
    assert abs((mid.bbox[0] + mid.bbox[2] / 2) - (bbox(41)[0] + bbox(41)[2] / 2)) < 20
    assert res[-1].target_id == 21 and not res[-1].predicted
    assert any("re-acquired" in t for r in res for _, t in r.events)
    assert f.occlusions and f.occlusions[0]["reacquired"]


def test_gives_up_and_stays_lost():
    f = follower(lambda i: [Track(7, bbox(i), 0.9)] if i <= 10 else [], max_occlusion_s=1.0)
    res = [f.update(FRAME) for _ in range(80)]
    assert res[-1].lost and not res[-1].ok and f.lost
    assert f.update(FRAME).lost  # final until init() is called again
    f.init(FRAME, bbox(0))       # re-selecting restarts it
    assert not f.lost


def test_camera_shift_is_used_for_the_prediction():
    def scene_box(i):
        cx = 2000 + 3.0 * i - 8.0 * i - (14.0 * (i - 20) if i > 20 else 0)  # drone speeds up after frame 20
        return (cx - 25, 286, 50, 28)

    def cam(i):
        return (-8.0 * i - (14.0 * (i - 20) if i > 20 else 0), 0.0)

    f = DetFollower(FakeProvider(lambda i: [Track(7, scene_box(i), 0.9)] if i <= 20 else []), FPS, "x", frame_size=(4000, 720))
    f.init(FRAME, scene_box(0))
    res = [f.update(FRAME, cam(i + 1)) for i in range(30)]
    last = res[-1]
    assert last.predicted
    assert abs((last.bbox[0] + last.bbox[2] / 2) - (scene_box(30)[0] + 25)) < 25


def test_snap_to_detection_picks_the_smallest_box_containing_the_click():
    boxes = [(0, 0, 200, 200), (50, 50, 40, 20)]
    assert snap_to_detection((60, 60), boxes) == (50, 50, 40, 20)
    assert snap_to_detection((150, 150), boxes) == (0, 0, 200, 200)
    assert snap_to_detection((500, 500), boxes) is None
