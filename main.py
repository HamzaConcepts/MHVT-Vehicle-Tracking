"""MHVT demo: pick a video, draw one zone per virtual UAV, click a vehicle, and watch
it get tracked with a zoomed 'drone camera' feed and simulated multi-UAV telemetry.

    python main.py [video]             interactive
    python main.py video --zones auto:3 --target 640,330,52,28 --headless --save out.mp4

Keys (interactive)
  zones   : drag = add zone | 2-6 = auto bands | Backspace = undo | C = clear | Enter = continue
  select  : drag/click on a vehicle | Space = play/pause | , . = step frame | Esc = back to zones
  track   : Space pause | Z zoom | L message loss 0/10/30% | 1-6 toggle UAV fault | R reselect | S screenshot
  anywhere: Q quit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from mhvt import render as R
from mhvt.coordinator import Config, Coordinator, auto_zones
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
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--headless", action="store_true", help="no window (needs --zones and --target)")
    ap.add_argument("--save", type=Path, help="write the composite window to this mp4")
    ap.add_argument("--screenshots", default="", help="comma list of tracked-frame numbers to save as PNG (headless)")
    ap.add_argument("--model", help="optional YOLO weights (drone-view/VisDrone) for click-to-snap detections")
    ap.add_argument("--loss", type=float, default=0.0, help="coordinator message loss probability")
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
        self.loss_idx = 0
        self.stage = "zones"
        self.zones: list[tuple[float, float, float, float]] = []
        self.sidecar = self.video.with_suffix(".zones.json")
        self.coord: Coordinator | None = None
        self.paused = True
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
        self.msg = ""
        self.shots = {int(s) for s in args.screenshots.split(",") if s}
        (ROOT / "output").mkdir(exist_ok=True)

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

    def save_zones(self) -> None:
        norm = [[a / R.WORK_W, b / R.WORK_H, c / R.WORK_W, d / R.WORK_H] for a, b, c, d in self.zones]
        self.sidecar.write_text(json.dumps(norm))

    # ------------------------------------------------------------- mouse
    def to_work(self, x, y):
        if x >= R.MAIN_W or y >= R.MAIN_H:
            return None
        return x / R.SCALE, y / R.SCALE

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
            self.start_tracking(bbox)

    # ----------------------------------------------------------- tracking
    def new_coordinator(self) -> Coordinator:
        cfg = Config(seed=self.args.seed, loss_prob=max(self.args.loss, LOSSES[self.loss_idx]))
        return Coordinator(self.zones, cfg)

    def start_tracking(self, bbox):
        self.tracker.init(self.work, bbox)
        self.bbox, self.score = bbox, 1.0
        if self.coord is None or len(self.coord.drones) != len(self.zones):
            self.coord = self.new_coordinator()
        self.coord.reset_target()
        self.trail.clear()
        self.track_frames = 0
        self.zoom_c = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
        self.stage, self.paused = "track", False
        self.coord.log(0.0, f"Target #1 selected at frame {self.frame_no}", "ok")
        self.step_sim()

    def step_sim(self):
        c = self.coord
        t = self.track_frames / self.fps
        center = None
        if self.bbox is not None and not self.tracker.lost:
            center = (self.bbox[0] + self.bbox[2] / 2, self.bbox[1] + self.bbox[3] / 2)
            self.trail.append(center)
        c.update(t, center)

    def advance_track(self) -> bool:
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
    def render(self):
        t = (self.track_frames / self.fps) if self.coord else 0.0
        main = cv2.resize(self.work, (R.MAIN_W, R.MAIN_H), interpolation=cv2.INTER_AREA)
        pending = None
        if self.drag0 and self.drag1:
            x1, x2 = sorted((self.drag0[0], self.drag1[0]))
            y1, y2 = sorted((self.drag0[1], self.drag1[1]))
            pending = (x1, y1, x2, y2)
        coord = self.coord if self.stage == "track" else None
        R.draw_zones(main, coord, self.zones, pending=pending if self.stage == "zones" else None, t=t)
        for (x, y, w, h) in self.detections:
            cv2.rectangle(main, R.sp((x, y)), R.sp((x + w, y + h)), (200, 200, 200), 1)
        lost = self.tracker.lost
        if self.stage == "track":
            R.draw_target(main, self.bbox, self.score, lost, self.trail)
            R.draw_handover(main, self.coord, self.zones)
        elif pending is not None and self.stage == "select":
            cv2.rectangle(main, R.sp(pending[:2]), R.sp(pending[2:]), R.GREEN, 2)
        banners = {
            "zones": ("STEP 1/3  Draw one zone per UAV (drag). 2-6 = auto bands. Enter = continue. "
                      f"Zones: {len(self.zones)}  {self.msg}", R.CYAN),
            "select": ("STEP 2/3  Click or drag a box around the vehicle to track. Space = play/pause, , . = step", R.CYAN),
            "track": (("TARGET LOST - press R to reselect" if lost else "TRACKING") +
                      f"   frame {self.frame_no}/{self.nframes}", R.RED if lost else R.GREEN),
        }
        R.draw_status_banner(main, *banners[self.stage])

        owner = self.coord.owner if self.coord else None
        name = f"UAV-{owner}" if owner else "UAV"
        zoom = R.zoom_panel(self.full, self.zoom_c, lost or self.stage != "track", self.zoom, name)
        if self.stage != "track" and (self.coord is None or len(self.coord.drones) != len(self.zones)):
            self.coord = self.new_coordinator()
        status = "IDLE" if self.stage != "track" else ("LOST" if lost else "TRACKING")
        tel = R.telemetry_panel(self.coord, status, self.score, (R.SIDE_W, R.CANVAS_H - R.ZOOM_H), t)
        help_text = {"zones": "drag zone | 2-6 auto | Bksp undo | C clear | Enter next | Q quit",
                     "select": "click/drag vehicle | Space play | , . step | Esc zones | Q quit",
                     "track": "Space pause | Z zoom | L loss | 1-6 fault UAV | R reselect | S shot | Q quit"}[self.stage]
        logp = R.log_panel(self.coord, (R.MAIN_W, R.CANVAS_H - R.MAIN_H), help_text)
        return R.compose(main, zoom, tel, logp)

    def write(self, canvas):
        if self.args.save:
            if self.writer is None:
                self.args.save.parent.mkdir(parents=True, exist_ok=True)
                self.writer = cv2.VideoWriter(str(self.args.save), cv2.VideoWriter_fourcc(*"mp4v"),
                                              self.fps, (R.CANVAS_W, R.CANVAS_H))
            self.writer.write(canvas)

    # --------------------------------------------------------------- run
    def run_headless(self):
        if not (self.args.zones and self.args.target):
            sys.exit("--headless needs --zones and --target")
        self.load_zones(self.args.zones)
        self.seek(self.args.start_frame)
        self.coord = self.new_coordinator()
        self.start_tracking(tuple(int(v) for v in self.args.target.split(",")))
        n = 0
        while True:
            if n in self.shots:
                cv2.imwrite(str(ROOT / "output" / f"shot_{n:04d}.png"), self.render())
            self.write(self.render())
            if not self.advance_track() or (self.args.max_frames and n >= self.args.max_frames):
                break
            n += 1
        self.finish()

    def finish(self):
        if self.writer:
            self.writer.release()
            print(f"Saved: {self.args.save}")
        if self.coord:
            print("Summary:", json.dumps(self.coord.summary(), indent=1))
            print(f"Tracked {self.track_frames} frames; target "
                  f"{'LOST' if self.tracker.lost else 'still tracked'} at end.")

    def run(self):
        if self.args.headless:
            return self.run_headless()
        self.load_zones(self.args.zones)
        self.read()
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1360, 765)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        if self.args.target:
            self.stage = "select"
            self.seek(self.args.start_frame)
            if not self.zones:
                self.zones = auto_zones(R.WORK_W, R.WORK_H, 3)
            self.start_tracking(tuple(int(v) for v in self.args.target.split(",")))
        ended = False
        while True:
            t0 = time.time()
            if self.stage == "select" and not self.paused:
                if not self.read():
                    self.seek(0)
            elif self.stage == "track" and not self.paused and not ended:
                if not self.advance_track():
                    ended = True
                    self.coord.log(self.track_frames / self.fps, "End of video", "info")
                    self.finish()
            if self.stage == "select" and self.paused and self.detector and not self.detections:
                self.detections = self.detector.detect(self.work)
            canvas = self.render()
            if self.stage == "track" and not ended and not self.paused:
                self.write(canvas)
            cv2.imshow(WINDOW, canvas)
            wait = max(1, int(1000 / self.fps - (time.time() - t0) * 1000))
            if self.paused or ended:
                wait = 30
            key = cv2.waitKey(wait) & 0xFF
            if key in (ord("q"), ord("Q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            ended = self.handle_key(key, ended)
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()

    def handle_key(self, key, ended):
        if key == 255:
            return ended
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
            if key == ord(" "):
                self.paused = not self.paused
            elif key in (ord("z"), ord("Z")):
                self.zoom = ZOOMS[(ZOOMS.index(self.zoom) + 1) % len(ZOOMS)] if self.zoom in ZOOMS else ZOOMS[0]
            elif key in (ord("l"), ord("L")):
                self.loss_idx = (self.loss_idx + 1) % len(LOSSES)
                self.coord.set_loss(LOSSES[self.loss_idx])
                self.coord.log(self.track_frames / self.fps, f"Message loss set to {int(LOSSES[self.loss_idx] * 100)}%", "warn")
            elif ord("1") <= key <= ord("6") and key - ord("0") <= len(self.zones):
                d = self.coord.drone(key - ord("0"))
                self.coord.set_silent(d.id, not d.silent, self.track_frames / self.fps)
            elif key in (ord("r"), ord("R")):
                self.stage, self.paused, ended = "select", True, False
                self.coord.reset_target()
                self.tracker.lost = True
                self.bbox = None
            elif key in (ord("s"), ord("S")):
                path = ROOT / "output" / f"screenshot_{self.frame_no:05d}.png"
                cv2.imwrite(str(path), self.render())
                self.coord.log(self.track_frames / self.fps, f"Saved {path.name}", "info")
        return ended


if __name__ == "__main__":
    App(parse_args()).run()
