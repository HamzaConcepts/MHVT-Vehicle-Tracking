"""A saved demo run: everything needed to replay the same test exactly.

Stored as JSON next to the video (<video>.scenario.json). Zones are normalised (0-1) so the
file survives a different working resolution; the target box is in 1280x720 working px.
`events` are the live actions taken during the run (message-loss changes, UAV faults, zoom),
stamped with the number of tracked frames at which they happened, so a replay applies them
at the same moments.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

VERSION = 1
EVENT_TYPES = {"loss", "fault", "zoom"}


@dataclass
class Scenario:
    video: str
    frames: int
    fps: float
    start_frame: int
    zones: list[list[float]]          # normalised x1,y1,x2,y2
    target: list[int]                 # x,y,w,h in working px
    seed: int = 7
    loss: float = 0.0
    zoom: float = 3.0
    events: list[dict] = field(default_factory=list)
    version: int = VERSION

    def validate(self) -> None:
        if len(self.zones) < 1 or any(len(z) != 4 for z in self.zones):
            raise ValueError("scenario needs at least one zone of 4 numbers")
        if len(self.target) != 4 or self.target[2] <= 0 or self.target[3] <= 0:
            raise ValueError("scenario target must be x,y,w,h with positive size")
        for ev in self.events:
            if ev.get("type") not in EVENT_TYPES or "frame" not in ev:
                raise ValueError(f"bad event in scenario: {ev}")

    def save(self, path: Path) -> None:
        self.validate()
        path.write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path: Path) -> "Scenario":
        data = json.loads(Path(path).read_text())
        if data.get("version", 1) > VERSION:
            raise ValueError(f"scenario version {data['version']} is newer than this program")
        s = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        s.events = sorted(s.events, key=lambda e: e["frame"])  # stable: keeps same-frame order
        s.validate()
        return s


def default_path(video: Path) -> Path:
    return video.with_suffix(".scenario.json")


def last_run_path(video: Path) -> Path:
    return video.with_suffix(".last_run.json")
