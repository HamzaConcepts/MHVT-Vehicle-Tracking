import numpy as np
import pytest

from mhvt.zoning import dominant_axis, equal_zones, zones_area_equal


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("layout", ["vertical", "horizontal", "grid"])
@pytest.mark.parametrize("overlap", [0.0, 0.15, 0.3])
def test_zones_are_equal_and_cover_the_frame(n, layout, overlap):
    z = equal_zones(1280, 720, n, layout, overlap)
    assert len(z) == n and zones_area_equal(z)
    assert min(r[0] for r in z) == 0 and min(r[1] for r in z) == 0
    assert max(r[2] for r in z) == pytest.approx(1280) and max(r[3] for r in z) == pytest.approx(720)
    # no gaps: every column of pixels is inside some zone
    xs = np.arange(0, 1280, 8)
    ys = np.arange(0, 720, 8)
    covered = np.zeros((len(ys), len(xs)), bool)
    for x1, y1, x2, y2 in z:
        covered |= ((xs[None, :] >= x1) & (xs[None, :] <= x2) & (ys[:, None] >= y1) & (ys[:, None] <= y2))
    assert covered.all()


def test_overlap_amount_is_exact():
    z = equal_zones(1000, 500, 4, "vertical", 0.2)
    w = z[0][2] - z[0][0]
    assert z[1][0] == pytest.approx(z[0][2] - 0.2 * w)
    assert z[0][2] - z[1][0] == pytest.approx(0.2 * w)


def test_grid_layout_shapes():
    assert len(equal_zones(1280, 720, 4, "grid", 0.1)) == 4
    z = equal_zones(1280, 720, 6, "grid", 0.0)
    assert sorted({round(r[1]) for r in z}) == [0, 360] and len({round(r[0]) for r in z}) == 3
    # 5 has no grid -> falls back to vertical bands
    assert all(r[1] == 0 and r[3] == 720 for r in equal_zones(1280, 720, 5, "grid"))


def test_invalid_arguments():
    with pytest.raises(ValueError):
        equal_zones(100, 100, 7)
    with pytest.raises(ValueError):
        equal_zones(100, 100, 3, overlap=0.95)


def test_dominant_axis_detects_motion_direction():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.integers(0, 255, (180, 320), dtype=np.uint8), (0, 0), 2.5)
    sx = np.roll(base, 4, axis=1)   # content moves along x
    sy = np.roll(base, 4, axis=0)   # content moves along y
    assert dominant_axis(base, sx) == "x"
    assert dominant_axis(base, sy) == "y"
