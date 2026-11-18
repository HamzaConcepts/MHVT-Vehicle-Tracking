import json

import pytest

from mhvt.scenario import Scenario


def make(**kw):
    base = dict(video="a.mp4", frames=100, fps=24.0, start_frame=5, zones=[[0, 0, .5, 1], [.4, 0, 1, 1]],
                target=[100, 120, 50, 30], seed=3, loss=0.1, zoom=4.0,
                events=[{"frame": 10, "type": "loss", "value": 0.3}, {"frame": 10, "type": "fault", "drone": 2, "silent": True}])
    base.update(kw)
    return Scenario(**base)


def test_roundtrip(tmp_path):
    p = tmp_path / "s.json"
    s = make()
    s.save(p)
    assert Scenario.load(p) == s


def test_same_frame_events_keep_order(tmp_path):
    p = tmp_path / "s.json"
    s = make(events=[{"frame": 9, "type": "zoom", "value": 2}, {"frame": 3, "type": "loss", "value": 0.1},
                     {"frame": 9, "type": "fault", "drone": 1, "silent": True}])
    s.save(p)
    ev = Scenario.load(p).events
    assert [e["frame"] for e in ev] == [3, 9, 9]
    assert [e["type"] for e in ev][1:] == ["zoom", "fault"]


@pytest.mark.parametrize("bad", [
    dict(zones=[]), dict(zones=[[0, 0, 1]]), dict(target=[1, 2, 0, 5]), dict(events=[{"frame": 1, "type": "boom"}]),
])
def test_rejects_invalid(tmp_path, bad):
    with pytest.raises(ValueError):
        make(**bad).save(tmp_path / "x.json")


def test_rejects_newer_version(tmp_path):
    p = tmp_path / "s.json"
    make().save(p)
    d = json.loads(p.read_text())
    d["version"] = 99
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        Scenario.load(p)
