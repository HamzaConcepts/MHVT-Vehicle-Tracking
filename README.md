# MHVT - Multi-Drone Handover System for Continuous Vehicle Tracking (FYP demo)

An interactive prototype of the MHVT mission-control experience, driven by a real video:

1. **Load a video** (file dialog, or `python main.py my_video.mp4`).
2. **Draw one zone per virtual UAV** on the first frame (or press `2`-`6` for automatic overlapping bands).
3. **Click / drag a box on a vehicle.** It is tracked with a visual tracker (OpenCV VitTrack) and
   enclosed in a box with a trail and a confidence score.
4. A **zoomed "UAV camera" feed** follows the vehicle (cropped from the full-resolution frame).
5. **Mission-control telemetry** shows every UAV's state, battery, link freshness and HandoverScore, plus a
   coordinator log of the handover protocol (REQ -> ACK -> ACQUIRED -> COMMIT) as the vehicle moves between zones.

> **What is real and what is simulated.** Video, tracking box, zoom feed and the vehicle's position are
> real. UAV battery, link quality, message latency/loss and the coordinator are *simulated* (as in the report's
> FYP-1 scope). The zones are regions of one video frame standing in for different drones' coverage areas.

## Run
```bash
git lfs pull                      # traffic.mp4 is stored in Git LFS (a clone without LFS gets a 133-byte stub)
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py traffic.mp4
```
| Stage | Keys |
|---|---|
| Zones | drag = add zone, `2`-`6` = auto bands, `Backspace` undo, `C` clear, `Enter` continue (zones are remembered next to the video), `O` load saved scenario |
| Select | click or drag on a vehicle, `Space` play/pause to find a good frame, `,` `.` step a frame, `Esc` back to zones |
| Track | `Space` pause, **`T` restart the run**, **`K` save scenario**, `Z` zoom 2x/3x/4x/6x, `L` message loss 0/10/30 %, `1`-`6` cut/restore a UAV's telemetry (fault injection), `E` write run report, `R` reselect, `O` load scenario, `S` screenshot, `F` fullscreen |
| Any | `Q` quit |

Things worth showing in a defense: press `L` to inject 10 % packet loss, or `2` to silence UAV-2 and watch
the coordinator time it out (~3 s), fall back to another successor, or fail safely while the sending UAV keeps ownership.

## The window
The interface is drawn at the window's real pixel size (resize it or press `F` for fullscreen; text stays sharp),
with a header showing the three steps, the video with zones and target, a zoomed UAV camera, a telemetry panel
(target state, owner, handover counts, message loss, a coverage strip and one card per UAV) and a coordinator log
with on-screen key hints. Font: Inter (SIL OFL, in `assets/fonts/`).

## Repeatable runs (save / replay / restart)
A **scenario** captures everything needed to repeat a test: the zones, the start frame, the vehicle you picked,
the random seed, the message-loss setting, and every live action you took during the run (loss changes, UAV faults,
zoom) with the frame it happened at.

1. Set up a run the way you want it, inject whatever faults you want to show, then press **`K`**.
   This writes `<video>.scenario.json` next to the video (commit it; it is small).
2. Replay it identically any time: press **`O`** in the window, or start with
   ```bash
   python main.py clip.mp4 --replay                  # uses clip.scenario.json
   python main.py clip.mp4 --scenario my_case.json   # a specific file
   ```
3. **`T`** restarts the current run from its first frame with a fresh coordinator. A run you built live restarts
   clean (no injected events); a replayed scenario restarts with its scripted events re-applied.
4. The program also autosaves `<video>.last_run.json` when you pick a vehicle and when you quit, so a good run is
   never lost. Rename it to keep it.

Replays are deterministic: the tracker and the seeded coordinator give the same result every time on the same
video (checked in testing). Keep the video file name unchanged, because a scenario is tied to it.

## Run report and experiments
**Run report.** When a run ends (end of video, restart, reselect, quit, or `E`), the program writes
`output/reports/<video>_<time>.json` plus `.handovers.csv`: tracked time, tracking continuity, how long exactly one UAV
owned the target, every handover (reason, latency, success), messages sent/dropped (delivery ratio) and the full log.
A handover cut short because the *vision* tracker lost the target is counted as `aborted`, not as a protocol failure.

**Experiments (report NFR3).** Turn a saved scenario into numbers:
```bash
python scripts/run_experiments.py clip.mp4 --seeds 100            # loss 0/5/10/20/30 %
python scripts/run_experiments.py clip.mp4 --faults --losses 0,0.1 # also replay the scenario's UAV faults
```
The tracker runs once (cached), then the coordinator is replayed for every seed and loss level, which takes seconds.
`output/experiments/<clip>/` gets `summary.md` (table with 95 % Wilson intervals and a PASS/FAIL for NFR3: >= 90 %
handover success at 10 % loss), `summary.csv`, `runs.csv` and `success_vs_loss.png`. The offline replay is tested to
reproduce the live app exactly. Caveats: the link is simulated, and all seeds replay the *same* target path, so the
attempts are not fully independent - quote the interval, and use several clips for a convincing result.

Headless / scripted (also records a video of the whole window):
```bash
python main.py traffic.mp4 --headless --zones auto:4 --target 1061,311,52,28 --save output/demo.mp4
python main.py traffic.mp4 --headless --replay --save output/replay.mp4      # render a saved scenario
```

## Honest status
* **Tracker:** VitTrack reports a confidence score, so the status switches to **LOST** instead of silently
  following the wrong object; LOST is final until you reselect (`R`). Fast camera motion or leaving the frame
  ends a track - the sample drone clip moves quickly, so tracks last ~5-8 s.
* **Detector (optional, needs `pip install -r requirements-train.txt`):** stock COCO YOLOv8 finds only ~5 % of
  the vehicles in top-down footage. `weights/vehicle_visdrone_n_bootstrap.pt` is a *bootstrap* YOLOv8n fine-tuned
  on VisDrone (single `vehicle` class, 6 epochs on 40 % of the data at 640 px, trained on a laptop-class CPU):
  mAP50 0.64 on VisDrone val, ~25-30 vehicles found per frame of `traffic.mp4` instead of ~1. Use it with
  `python main.py traffic.mp4 --model weights/vehicle_visdrone_n_bootstrap.pt` (detections are drawn while you pick
  a frame and a click snaps to the vehicle). Retrain properly with `docs/TRAINING.md` for the final defense.
* **Not implemented yet:** SUMO/TraCI simulation, database logging, FastAPI/WebSocket dashboard, ROS 2,
  automatic cross-camera re-identification. See the report schedule.

## Repository layout
```
main.py                  interactive demo + headless mode
mhvt/coordinator.py      zones, HandoverScore ranking, handover protocol state machine (pure Python, tested)
mhvt/tracking.py         VitTrack wrapper with LOST logic; optional YOLO detector
mhvt/render.py           window drawing (video, zoom feed, telemetry, log)
mhvt/scenario.py         save/load of a repeatable run (zones, target, seed, events)
mhvt/metrics.py          run metrics + report writer; mhvt/experiment.py = offline replay/sweeps
models/                  VitTrack ONNX (Apache-2.0, from opencv_zoo)
weights/                bootstrap vehicle detector (VisDrone fine-tune)
scripts/                 run_experiments.py, prepare_visdrone.py, train_vehicle_detector.py, yolo_bytetrack_baseline.py
assets/fonts/            Inter (OFL) used by the UI
tests/                   pytest for the coordinator (single owner, 10 % loss, faults, reproducibility)
docs/                    DATASETS.md, TRAINING.md
notebooks/               Colab/Kaggle training notebook
```
Run the tests with `pip install pytest && pytest -q`.
