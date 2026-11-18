"""MHVT demo: pick a video, draw one zone per virtual UAV, click a vehicle, and watch
it get tracked with a zoomed 'drone camera' feed and simulated multi-UAV telemetry.

    python main.py [video]                  interactive
    python main.py video --replay           re-run the saved scenario (video.scenario.json) exactly
    python main.py video --scenario f.json  re-run a specific scenario file
    python main.py video --zones auto:3 --target 640,330,52,28 --headless --save out.mp4

Keys (interactive)
  zones   : drag = add zone | 2-6 = auto bands | Backspace = undo | C = clear | Enter = continue | O = load scenario
  select  : drag/click on a vehicle | Space = play/pause | , . = step frame | Esc = back to zones | O = load scenario
  track   : Space pause | T restart run | K save scenario | Z zoom | L message loss 0/10/30%
            1-6 toggle UAV fault | R reselect | S screenshot | O load scenario
  anywhere: Q quit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from collections import deque
from pathlib import Path

import cv2

from mhvt import render as R
from mhvt.render import UI, Layout, Scene
from mhvt.coordinator import Config, Coordinator, auto_zones
from mhvt.metrics import write_report
from mhvt.scenario import Scenario, default_path, last_run_path
from mhvt.tracking import Detector, TargetTracker, snap_to_detection

ROOT = Path(__file__).resolve().parent
WINDOW = "MHVT - Multi-Drone Handover Demo"
ZOOMS = [2.0, 3.0, 4.0, 6.0]
LOSSES = [0.0, 0.10, 0.30]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?", type=Path, help="input video (file dialog if omitted)")
    ap.add_argument("--zones", help="'auto:N' or path to a zones .json (default: <video>.zones.json if present)")
    ap.add_argument("--target", help="x,y,w,h in 1280x720 working px; skips manual selection")
    ap.add_argument("--scenario", type=Path, help="scenario .json to replay (zones, start frame, target, events)")
    ap.add_argument("--replay", action="store_true", help="replay <video>.scenario.json")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--headless", action="store_true", help="no window (needs --scenario/--replay or --zones + --target)")
    ap.add_argument("--save", type=Path, help="write the composite window to this mp4")
    ap.add_argument("--screenshots", default="", help="comma list of tracked-frame numbers to save as PNG (headless)")
    ap.add_argument("--model", help="optional YOLO weights (drone-view/VisDrone) for click-to-snap detections")
    ap.add_argument("--loss", type=float, default=0.0, help="initial coordinator message loss probability")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--zoom", type=float, default=3.0)
    return ap.parse_args()


def pick_video() -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(title="Select traffic video",
                                          filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")])
        root.destroy()
        if path:
            return Path(path)
    except Exception:
        pass
    return ROOT / "traffic.mp4"


def letterbox_169(frame):
    h, w = frame.shape[:2]
    if abs(w / h - 16 / 9) < 0.01:
        return frame
    if w / h > 16 / 9:
        nh = int(round(w * 9 / 16))
        pad = (nh - h) // 2
        return cv2.copyMakeBorder(frame, pad, nh - h - pad, 0, 0, cv2.BORDER_CONSTANT)
    nw = int(round(h * 16 / 9))
    pad = (nw - w) // 2
    return cv2.copyMakeBorder(frame, 0, 0, pad, nw - w - pad, cv2.BORDER_CONSTANT)


class App:
    def __init__(self, args):
        self.args = args
        self.video = args.video or pick_video()
        if not self.video.is_file() or self.video.stat().st_size < 1000:
            sys.exit(f"Cannot read video: {self.video}\n(If traffic.mp4 is tiny, run: git lfs pull)")
        self.cap = cv2.VideoCapture(str(self.video))
        if not self.cap.isOpened():
            sys.exit(f"OpenCV could not open: {self.video}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.detector = Detector(args.model) if args.model else None
        self.tracker = TargetTracker()
        self.zoom = args.zoom
        self.seed = args.seed
        self.loss = args.loss
        self.stage = "zones"
        self.zones: list[tuple[float, float, float, float]] = []
        self.sidecar = self.video.with_suffix(".zones.json")
        self.scenario_path = args.scenario or default_path(self.video)
        self.coord: Coordinator | None = None
        self.paused = True
        self.ended = False
        self.frame_no = -1
        self.full = self.work = None
        self.drag0 = self.drag1 = None
        self.trail: deque = deque(maxlen=90)
        self.zoom_c = (R.WORK_W / 2, R.WORK_H / 2)
        self.bbox = None
        self.score = 0.0
        self.track_frames = 0
        self.detections = []
        self.writer = None
        self._msg, self._msg_until = "", 0.0
        self.layout = Layout.fit(R.DESIGN_W, R.DESIGN_H)
        self.ui = UI(self.layout)
        self.rec_ui: UI | None = None         # fixed-size renderer for --save recordings
        self.fullscreen = False
        # run definition (what a scenario file stores)
        self.run_start = 0
        self.run_bbox: tuple[int, int, int, int] | None = None
        self.run_loss = self.loss
        self.run_zoom = self.zoom
        self.events: list[dict] = []          # events recorded in this run
        self.replay_events: list[dict] = []   # scripted events being replayed
        self.pending: list[dict] = []
        self.replaying = False
        self.track_log: list = []          # target centre per tracked frame (None once lost)
        self.report_done = True
        self.shots = {int(s) for s in args.screenshots.split(",") if s}
        (ROOT / "output").mkdir(exist_ok=True)

    @property
    def msg(self) -> str:
        return self._msg if time.time() < self._msg_until else ""

    @msg.setter
    def msg(self, v: str) -> None:
        self._msg, self._msg_until = v, (time.time() + 4.0 if v else 0.0)

    # ------------------------------------------------------------ frames
    def read(self) -> bool:
        ok, f = self.cap.read()
        if not ok:
            return False
        self.frame_no += 1
        self.full = letterbox_169(f)
        self.work = cv2.resize(self.full, (R.WORK_W, R.WORK_H), interpolation=cv2.INTER_AREA)
        return True

    def seek(self, n: int) -> None:
        n = max(0, min(self.nframes - 1, n))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        self.frame_no = n - 1
        self.read()
        self.detections = []

    # -------------------------------------------------------------- zones
    def load_zones(self, spec: str | None) -> None:
        if spec and spec.startswith("auto:"):
            self.zones = auto_zones(R.WORK_W, R.WORK_H, int(spec[5:]))
            return
        path = Path(spec) if spec else (self.sidecar if self.sidecar.is_file() else None)
        if path and path.is_file():
            norm = json.loads(path.read_text())
            self.zones = [(a * R.WORK_W, b * R.WORK_H, c * R.WORK_W, d * R.WORK_H) for a, b, c, d in norm]
            self.msg = f"Loaded {len(self.zones)} saved zones (C to clear)"

    def norm_zones(self) -> list[list[float]]:
        return [[a / R.WORK_W, b / R.WORK_H, c / R.WORK_W, d / R.WORK_H] for a, b, c, d in self.zones]

    def save_zones(self) -> None:
        self.sidecar.write_text(json.dumps(self.norm_zones()))

    # ----------------------------------------------------------- scenario
    def make_scenario(self) -> Scenario:
        return Scenario(video=self.video.name, frames=self.nframes, fps=round(self.fps, 3),
                        start_frame=self.run_start, zones=self.norm_zones(), target=list(self.run_bbox),
                        seed=self.seed, loss=self.run_loss, zoom=self.run_zoom,
                        events=list(self.replay_events if self.replaying else self.events))

    def save_scenario(self, path: Path | None = None) -> Path | None:
        if self.run_bbox is None:
            self.msg = "Nothing to save yet - select a vehicle first"
            return None
        path = path or self.scenario_path
        self.make_scenario().save(path)
        return path

    def autosave(self) -> None:
        if self.run_bbox is not None:
            try:
                self.make_scenario().save(last_run_path(self.video))
            except OSError:
                pass

    def load_scenario(self, path: Path) -> bool:
        try:
            sc = Scenario.load(path)
        except (OSError, ValueError, TypeError, KeyError) as e:
            self.msg = f"Cannot load scenario: {e}"
            print(self.msg)
            return False
        warn = []
        if sc.video != self.video.name:
            warn.append(f"scenario was made for '{sc.video}'")
        if sc.frames != self.nframes:
            warn.append(f"frame count differs ({sc.frames} vs {self.nframes})")
        if warn:
            print("WARNING:", "; ".join(warn))
        self.zones = [(a * R.WORK_W, b * R.WORK_H, c * R.WORK_W, d * R.WORK_H) for a, b, c, d in sc.zones]
        self.seed, self.loss, self.zoom = sc.seed, sc.loss, sc.zoom
        self.seek(sc.start_frame)
        self.begin_run(tuple(sc.target), events=sc.events, replaying=True)
        self.coord.log(0.0, f"Scenario loaded: {Path(path).name}"
                            + (f" ({len(sc.events)} scripted events)" if sc.events else ""), "ok")
        return True

    # ------------------------------------------------------------- mouse
    def to_work(self, x, y):
        vx1, vy1, vx2, vy2 = self.layout.video_px()
        if not (vx1 <= x < vx2 and vy1 <= y < vy2):
            return None
        k = self.layout.video_scale()
        return (x - vx1) / k, (y - vy1) / k

    def on_mouse(self, event, x, y, flags, _):
        if self.stage not in ("zones", "select"):
            return
        pt = self.to_work(x, y)
        if event == cv2.EVENT_LBUTTONDOWN and pt:
            self.drag0 = self.drag1 = pt
        elif event == cv2.EVENT_MOUSEMOVE and self.drag0 and pt:
            self.drag1 = pt
        elif event == cv2.EVENT_LBUTTONUP and self.drag0:
            end = pt or self.drag1
            self.finish_drag(self.drag0, end)
            self.drag0 = self.drag1 = None

    def finish_drag(self, a, b):
        x1, x2 = sorted((a[0], b[0]))
        y1, y2 = sorted((a[1], b[1]))
        small = (x2 - x1) < 12 or (y2 - y1) < 12
        if self.stage == "zones":
            if not small and len(self.zones) < 6:
                self.zones.append((x1, y1, x2, y2))
        else:
            if small:
                snap = snap_to_detection((int(a[0]), int(a[1])), self.detections)
                bbox = snap or (int(a[0] - 30), int(a[1] - 17), 60, 34)
            else:
                bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            self.begin_run(bbox)

    # ----------------------------------------------------------- tracking
    def new_coordinator(self) -> Coordinator:
        return Coordinator(self.zones, Config(seed=self.seed, loss_prob=self.loss))

    def begin_run(self, bbox, events=None, replaying=False):
        """Start tracking from the current frame with a *fresh* coordinator, so the same
        scenario always produces the same run. Used for first start, restart and replay."""
        self.run_start, self.run_bbox = self.frame_no, tuple(int(v) for v in bbox)
        self.run_loss, self.run_zoom = self.loss, self.zoom
        self.replaying = replaying
        self.replay_events = list(events or [])
        self.pending = list(self.replay_events) if replaying else []
        self.events = []
        self.track_log = []
        self.report_done = False
        self.tracker.init(self.work, self.run_bbox)
        self.bbox, self.score = self.run_bbox, 1.0
        self.coord = self.new_coordinator()
        self.trail.clear()
        self.track_frames = 0
        self.zoom_c = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
        self.stage, self.paused, self.ended, self.msg = "track", False, False, ""
        self.coord.log(0.0, f"Target #1 selected at frame {self.frame_no}", "ok")
        self.step_sim()
        self.autosave()

    def restart_run(self):
        if self.run_bbox is None:
            return
        self.export_report()
        self.seek(self.run_start)
        self.loss, self.zoom = self.run_loss, self.run_zoom
        self.begin_run(self.run_bbox, events=self.replay_events, replaying=self.replaying)
        self.coord.log(0.0, "Run restarted" + (" (scripted events replayed)" if self.replaying else ""), "info")

    def apply_event(self, ev: dict) -> None:
        t = self.track_frames / self.fps
        if ev["type"] == "loss":
            self.loss = float(ev["value"])
            self.coord.set_loss(self.loss)
            self.coord.log(t, f"Message loss set to {int(self.loss * 100)}%", "warn")
        elif ev["type"] == "fault":
            if 1 <= ev["drone"] <= len(self.coord.drones):
                self.coord.set_silent(ev["drone"], bool(ev["silent"]), t)
        elif ev["type"] == "zoom":
            self.zoom = float(ev["value"])

    def do_event(self, ev: dict) -> None:
        """A live action: apply it now and record it so the scenario can replay it."""
        ev = {"frame": self.track_frames, **ev}
        target = self.replay_events if self.replaying else self.events
        target.append(ev)
        target.sort(key=lambda e: e["frame"])  # stable: same-frame events keep their order
        self.apply_event(ev)

    def step_sim(self):
        c = self.coord
        t = self.track_frames / self.fps
        center = None
        if self.bbox is not None and not self.tracker.lost:
            center = (self.bbox[0] + self.bbox[2] / 2, self.bbox[1] + self.bbox[3] / 2)
            self.trail.append(center)
        self.track_log.append(center)
        c.update(t, center)

    def export_report(self, force: bool = False) -> Path | None:
        """Write the run report (JSON + handovers CSV) once per run."""
        if not self.coord or self.run_bbox is None or len(self.track_log) < 2:
            return None
        if self.report_done and not force:
            return None
        self.report_done = True
        meta = {"video": self.video.name, "created": datetime.now().isoformat(timespec="seconds"),
                "fps": round(self.fps, 3), "start_frame": self.run_start, "target": list(self.run_bbox),
                "seed": self.seed, "initial_loss": self.run_loss, "replayed_scenario": self.replaying,
                "zones": self.norm_zones(),
                "events": self.replay_events if self.replaying else self.events}
        path = ROOT / "output" / "reports" / f"{self.video.stem}_{datetime.now():%Y%m%d_%H%M%S}.json"
        write_report(path, meta, self.coord, self.track_log, self.fps)
        self.msg = f"Report saved: {path.name}"
        print(self.msg)
        return path

    def advance_track(self) -> bool:
        while self.pending and self.pending[0]["frame"] <= self.track_frames:
            self.apply_event(self.pending.pop(0))
        if not self.read():
            return False
        self.track_frames += 1
        was_lost = self.tracker.lost
        ok, bbox, score = self.tracker.update(self.work)
        self.score = score
        if ok:
            self.bbox = bbox
            cx, cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
            self.zoom_c = (self.zoom_c[0] + 0.25 * (cx - self.zoom_c[0]), self.zoom_c[1] + 0.25 * (cy - self.zoom_c[1]))
        elif not was_lost:
            self.bbox = None
            self.coord.mark_target_lost(self.track_frames / self.fps)
        self.step_sim()
        return True

    # ------------------------------------------------------------- render
    def scene(self) -> Scene:
        if self.stage != "track" and (self.coord is None or len(self.coord.drones) != len(self.zones)):
            self.coord = self.new_coordinator()
        drag = None
        if self.drag0 and self.drag1:
            x1, x2 = sorted((self.drag0[0], self.drag1[0]))
            y1, y2 = sorted((self.drag0[1], self.drag1[1]))
            drag = (x1, y1, x2, y2)
        return Scene(stage=self.stage, work=self.work, full=self.full, zones=self.zones, coord=self.coord,
                     video_name=self.video.name, bbox=self.bbox, score=self.score,
                     lost=self.tracker.lost or self.stage != "track", trail=list(self.trail), zoom=self.zoom,
                     zoom_center=self.zoom_c, drag=drag, detections=self.detections, replaying=self.replaying,
                     paused=self.paused, ended=self.ended, frame_no=self.frame_no, nframes=self.nframes,
                     t=(self.track_frames / self.fps) if self.stage == "track" else 0.0, toast=self.msg)

    def render(self):
        return self.ui.render(self.scene())

    def fit_layout(self, w: int, h: int) -> None:
        if (w, h) != (self.layout.W, self.layout.H) and w > 200 and h > 120:
            self.layout = Layout.fit(w, h)
            self.ui = UI(self.layout)

    def write(self):
        """Record the window as a fixed 1600x900 video (independent of the live window size)."""
        if not self.args.save:
            return
        if self.rec_ui is None:
            self.rec_ui = UI(Layout.fit(R.DESIGN_W, R.DESIGN_H))
            self.args.save.parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(str(self.args.save), cv2.VideoWriter_fourcc(*"mp4v"),
                                          self.fps, (R.DESIGN_W, R.DESIGN_H))
        self.writer.write(self.rec_ui.render(self.scene()))

    # --------------------------------------------------------------- run
    def start_from_args(self) -> bool:
        """Start a run from --scenario/--replay/--target. Returns True if a run was started."""
        a = self.args
        if a.scenario or a.replay:
            path = self.scenario_path
            if not path.is_file():
                sys.exit(f"Scenario not found: {path}  (press K during a run to create it)")
            if not self.load_scenario(path):
                sys.exit(self.msg)
            return True
        if a.target:
            self.load_zones(a.zones)
            if not self.zones:
                self.zones = auto_zones(R.WORK_W, R.WORK_H, 3)
            self.seek(a.start_frame)
            self.begin_run(tuple(int(v) for v in a.target.split(",")))
            return True
        return False

    def run_headless(self):
        a = self.args
        if not (a.scenario or a.replay or (a.zones and a.target)):
            sys.exit("--headless needs --scenario/--replay, or --zones and --target")
        if not (a.scenario or a.replay):
            self.load_zones(a.zones)
        self.read()
        self.start_from_args()
        n = 0
        while True:
            if n in self.shots:
                cv2.imwrite(str(ROOT / "output" / f"shot_{n:04d}.png"), self.render())
            self.write()
            if not self.advance_track() or (a.max_frames and n >= a.max_frames):
                break
            n += 1
        self.finish()

    def finish(self):
        if self.writer:
            self.writer.release()
            self.writer = None
            print(f"Saved: {self.args.save}")
        if self.coord:
            print("Summary:", json.dumps(self.coord.summary(), indent=1))
            print(f"Tracked {self.track_frames} frames; target "
                  f"{'LOST' if self.tracker.lost else 'still tracked'} at end.")
            self.export_report()

    def run(self):
        if self.args.headless:
            return self.run_headless()
        self.load_zones(self.args.zones)
        self.read()
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        w, h = self.initial_window_size()
        cv2.resizeWindow(WINDOW, w, h)
        self.fit_layout(w, h)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        if self.start_from_args():
            pass
        elif self.scenario_path.is_file():
            self.msg = "Saved scenario found - press O to replay it"
        while True:
            t0 = time.time()
            if self.stage == "select" and not self.paused:
                if not self.read():
                    self.seek(0)
            elif self.stage == "track" and not self.paused and not self.ended:
                if not self.advance_track():
                    self.ended = True
                    self.coord.log(self.track_frames / self.fps, "End of video", "info")
                    self.finish()
            if self.stage == "select" and self.paused and self.detector and not self.detections:
                self.detections = self.detector.detect(self.work)
            try:  # render at the window's real pixel size so nothing is rescaled (crisp text)
                _, _, ww, wh = cv2.getWindowImageRect(WINDOW)
                self.fit_layout(ww, wh)
            except cv2.error:
                pass
            cv2.imshow(WINDOW, self.render())
            if self.stage == "track" and not self.ended and not self.paused:
                self.write()
            wait = max(1, int(1000 / self.fps - (time.time() - t0) * 1000))
            if self.paused or self.ended:
                wait = 30
            key = cv2.waitKey(wait) & 0xFF
            if key in (ord("q"), ord("Q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            self.handle_key(key)
        self.autosave()
        self.export_report()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()

    @staticmethod
    def initial_window_size() -> tuple[int, int]:
        """Largest 16:9-ish window (up to 1600x900) that fits the screen."""
        sw, sh = 1600, 900
        try:
            import tkinter as tk

            r = tk.Tk()
            r.withdraw()
            sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
            r.destroy()
        except Exception:
            pass
        f = min(1.0, (sw * 0.96) / R.DESIGN_W, (sh * 0.88) / R.DESIGN_H)
        return int(R.DESIGN_W * f), int(R.DESIGN_H * f)

    def handle_key(self, key):
        if key == 255:
            return
        if key in (ord("f"), ord("F")):
            self.fullscreen = not self.fullscreen
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
            return
        if key in (ord("o"), ord("O")):
            if self.scenario_path.is_file():
                self.load_scenario(self.scenario_path)
            else:
                self.msg = f"No scenario at {self.scenario_path.name} (press K during a run to save one)"
            return
        if self.stage == "zones":
            if key in (13, 10):
                if not self.zones:
                    self.msg = "draw at least one zone first"
                else:
                    self.save_zones()
                    self.stage, self.msg = "select", ""
                    self.coord = self.new_coordinator()
            elif key in (8, 127) and self.zones:
                self.zones.pop()
            elif key in (ord("c"), ord("C")):
                self.zones, self.msg = [], ""
            elif ord("2") <= key <= ord("6"):
                self.zones = auto_zones(R.WORK_W, R.WORK_H, key - ord("0"))
        elif self.stage == "select":
            if key == ord(" "):
                self.paused = not self.paused
                self.detections = []
            elif key in (ord(","), ord(".")):
                self.paused = True
                self.seek(self.frame_no + (1 if key == ord(".") else -1))
            elif key == 27:
                self.stage = "zones"
        elif self.stage == "track":
            t = self.track_frames / self.fps
            if key == ord(" "):
                self.paused = not self.paused
            elif key in (ord("t"), ord("T")):
                self.restart_run()
            elif key in (ord("k"), ord("K")):
                path = self.save_scenario()
                if path:
                    self.msg = f"Saved {path.name}"
                    self.coord.log(t, f"Scenario saved: {path.name} ({len(self.make_scenario().events)} events)", "ok")
            elif key in (ord("z"), ord("Z")):
                nxt = ZOOMS[(ZOOMS.index(self.zoom) + 1) % len(ZOOMS)] if self.zoom in ZOOMS else ZOOMS[0]
                self.do_event({"type": "zoom", "value": nxt})
            elif key in (ord("l"), ord("L")):
                i = min(range(len(LOSSES)), key=lambda k: abs(LOSSES[k] - self.loss))
                self.do_event({"type": "loss", "value": LOSSES[(i + 1) % len(LOSSES)]})
            elif ord("1") <= key <= ord("6") and key - ord("0") <= len(self.zones):
                d = self.coord.drone(key - ord("0"))
                self.do_event({"type": "fault", "drone": d.id, "silent": not d.silent})
            elif key in (ord("e"), ord("E")):
                self.export_report(force=True)
            elif key in (ord("r"), ord("R")):
                self.autosave()
                self.export_report()
                self.stage, self.paused, self.ended = "select", True, False
                self.tracker.lost = True
                self.bbox = None
                self.coord = self.new_coordinator()
            elif key in (ord("s"), ord("S")):
                path = ROOT / "output" / f"screenshot_{self.frame_no:05d}.png"
                cv2.imwrite(str(path), self.render())
                self.coord.log(t, f"Saved {path.name}", "info")


if __name__ == "__main__":
    App(parse_args()).run()
