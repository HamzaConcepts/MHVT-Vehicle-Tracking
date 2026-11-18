import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("PIL")

from mhvt.coordinator import Config, Coordinator, auto_zones
from mhvt.render import UI, Layout, Scene


def scene(stage, n=4, lost=False):
    zones = auto_zones(1280, 720, n)
    c = Coordinator(zones, Config(seed=1))
    work = np.full((720, 1280, 3), 90, np.uint8)
    full = np.full((1440, 2560, 3), 90, np.uint8)
    if stage == "track":
        c.update(0.0, (300, 300))
    return Scene(stage=stage, work=work, full=full, zones=zones, coord=c, video_name="clip.mp4",
                 bbox=None if lost else (280, 290, 50, 28), score=0.6, lost=lost, trail=[(250, 300), (300, 300)],
                 drag=(10, 10, 200, 200) if stage == "zones" else None, toast="Saved clip.scenario.json",
                 nframes=830, t=3.2)


@pytest.mark.parametrize("size", [(1000, 563), (1600, 900), (1920, 1080), (1400, 900)])
@pytest.mark.parametrize("stage,lost", [("zones", True), ("select", True), ("track", False), ("track", True)])
def test_renders_at_any_window_size(size, stage, lost):
    ui = UI(Layout.fit(*size))
    img = ui.render(scene(stage, lost=lost))
    assert img.shape == (size[1], size[0], 3) and img.dtype == np.uint8
    assert img.std() > 5  # something was actually drawn


@pytest.mark.parametrize("size", [(1000, 563), (1600, 900)])
def test_predicted_state_with_bridge_and_other_tracks_renders(size):
    sc = scene("track")
    sc.predicted, sc.std, sc.occluded_s = True, 30.0, 1.2
    sc.occluders = [(500, 0, 700, 720)]
    sc.others = [(5, (100, 100, 50, 28)), (6, (900, 400, 55, 30))]
    sc.backend = "YOLO + ByteTrack"
    ui = UI(Layout.fit(*size))
    plain = UI(Layout.fit(*size)).render(scene("track")).copy()
    img = ui.render(sc)
    assert img.shape == plain.shape and not np.array_equal(img, plain)
    x1, y1, x2, y2 = ui.L.video_px()
    amber = (img[y1:y2, x1:x2] == np.array([32, 176, 255])).all(axis=2).sum()  # AMBER (BGR) pixels: ghost box / circle
    assert amber > 50


def test_six_drones_and_text_is_drawn():
    ui = UI(Layout.fit(1600, 900))
    img = ui.render(scene("track", n=6))
    x1, y1, x2, y2 = ui.L.rect(*ui.L.TEL)
    assert (img[y1:y2, x1:x2] > 200).any()  # bright text pixels in the telemetry panel


def test_mouse_mapping_roundtrip():
    L = Layout.fit(1333, 777)
    x1, y1, x2, y2 = L.video_px()
    k = L.video_scale()
    wx, wy = 640, 360
    px, py = x1 + wx * k, y1 + wy * k
    assert abs((px - x1) / k - wx) < 1e-6 and x1 <= px < x2 and y1 <= py < y2
