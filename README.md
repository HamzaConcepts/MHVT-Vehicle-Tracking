# MHVT - Multi-Drone Handover System for Continuous Vehicle Tracking (FYP demo)

An interactive prototype of the MHVT mission-control experience, driven by a real video:

1. **Load a video** (file dialog, or `python main.py my_video.mp4`).
2. **Make one zone per virtual UAV** - press `2`-`6` for equal-area zones (no drawing needed), or drag your own.
3. **Click / drag a box on a vehicle.** A YOLOv8 detector + ByteTrack (or BoT-SORT) give every vehicle an ID and
   follow the one you picked; if it disappears (a bridge, a building) its path is predicted and it is
   re-acquired when it comes out. Without the detector it falls back to OpenCV VitTrack.
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
pip install ultralytics lap       # YOLO + ByteTrack/BoT-SORT backend (optional: without it the demo uses VitTrack)
python main.py traffic.mp4
```
`--tracker auto|bytetrack|botsort|vit` picks the backend (`auto` = YOLO + ByteTrack when it can be loaded, else VitTrack;
the first lines printed say which one is running). `--imgsz 960` is faster, the default `1280` is more accurate.
| Stage | Keys |
|---|---|
| Zones | **`2`-`6` equal zones**, `V` vertical / `H` horizontal / `G` grid, `A` pick the layout from the video's motion, `[` `]` less/more overlap, or drag a custom zone; `Shift`+drag = simulated bridge, `Backspace` undo, `C` clear, `Enter` continue (zones are remembered next to the video), `O` load saved scenario |
| Select | click or drag on a vehicle, `Space` play/pause to find a good frame, `,` `.` step a frame, `Shift`+drag a bridge, `X` clear bridges, `Esc` back to zones |
| Track | `Space` pause, **`T` restart the run**, **`K` save scenario**, `Z` zoom 2x/3x/4x/6x, `L` message loss 0/10/30 %, `1`-`6` cut/restore a UAV's telemetry (fault injection), `E` write run report, `R` reselect, `O` load scenario, `S` screenshot, `F` fullscreen |
| Any | `Q` quit |

Things worth showing in a defense: press `L` to inject 10 % packet loss, or `2` to silence UAV-2 and watch
the coordinator time it out (~3 s), fall back to another successor, or fail safely while the sending UAV keeps ownership.

## Equal zones
`mhvt/zoning.py` makes zones of exactly equal area that overlap their neighbours by a chosen fraction (default 15 %):
`n` vertical bands, `n` horizontal bands, or a 2x2 / 2x3 grid. `A` looks at a few frame pairs of the video
(optical flow) to see whether traffic moves along x or y and picks bands across the flow. From the command line:
`--zone-count 4 --zone-layout auto --zone-overlap 0.15`.

## Path prediction, bridges and re-acquisition
When the followed vehicle's tracker ID vanishes (it drives under a bridge, behind a building, or ByteTrack just
changes its ID) `mhvt/prediction.py` keeps going instead of giving up:
* A Kalman motion model predicts the position (dead reckoning). The status turns **PREDICTING**: a dashed amber ghost
  box with an uncertainty circle, the zoom feed says DEAD RECKONING, and the handover logic keeps running on the
  predicted position.
* Motion is modelled in *scene* coordinates: `mhvt/motion.py` measures the camera's own movement (phase
  correlation), so the drone speeding up or slowing down does not throw the prediction off.
* A vehicle that appears is accepted as the target only if it is **new** (an already-tracked car is another car),
  lies inside a gate that is tight sideways (vehicles keep their lane) and loose along the road, has a similar size,
  moved in the same direction, looks the same (colour histogram), passes for 4 frames in a row and is clearly better than
  the runner-up. Two equally plausible vehicles => it refuses to guess.
* No candidate within `--max-occlusion` seconds (default 8), or the prediction leaves the picture => **LOST**.

No bridge in your footage? **Shift+drag** on the video (zones or select step) to add a simulated one. Its pixels are
blanked, so the detector truly cannot see there. Bridges are saved in scenarios; `--occluder 0.45,0,0.6,1` does the
same from the command line (fractions of the frame).

**How well does it work?** `python scripts/eval_occlusion.py clip.mp4` measures it without labels: for each vehicle it
compares a run with a bridge to a reference run without one. On the sample clip (bootstrap detector, 1280 px, 24 bridge
placements): **13 correct (54 %), 7 locked onto a wrong vehicle (29 %), 4 never recovered (17 %)**. It is reliable when
the prediction is good (error under ~30 px) and fails mostly when the vehicle has been followed for under a second before
disappearing (its speed is not known yet), in dense traffic of similar cars, and with long occlusions. A
stronger detector (see `docs/TRAINING.md`) steadies the tracker IDs and should help; the measurement is the
tool to prove it. Not handled: occlusion while the drone itself turns or zooms (camera motion is translation-only).

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
* **Trackers:** the main backend is YOLO + ByteTrack/BoT-SORT with path prediction (above). The VitTrack fallback
  reports a confidence score, so it says **LOST** instead of silently following the wrong object, but it has no
  prediction or re-acquisition. LOST is final until you reselect (`R`). Fast camera motion or leaving the frame
  ends a track - the sample drone clip moves quickly, so tracks last ~5-8 s. On this clip ByteTrack and BoT-SORT
  perform about the same (15 vs 11 fps on a 4-core CPU at 960 px); comparing them properly needs VisDrone-MOT labels.
* **Detector (needs `pip install ultralytics lap`):** stock COCO YOLOv8 finds only ~5 % of
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
mhvt/tracking.py         YOLO + ByteTrack/BoT-SORT follower, VitTrack fallback
mhvt/prediction.py       Kalman path prediction + re-acquisition after occlusions (pure, tested)
mhvt/motion.py           camera-motion estimate;  mhvt/zoning.py = equal-area automatic zones
mhvt/render.py           window drawing (video, zoom feed, telemetry, log)
mhvt/scenario.py         save/load of a repeatable run (zones, target, seed, events)
mhvt/metrics.py          run metrics + report writer; mhvt/experiment.py = offline replay/sweeps
models/                  VitTrack ONNX (Apache-2.0, from opencv_zoo)
weights/                bootstrap vehicle detector (VisDrone fine-tune)
scripts/                 eval_occlusion.py, run_experiments.py, prepare_visdrone.py, train_vehicle_detector.py, yolo_bytetrack_baseline.py
assets/fonts/            Inter (OFL) used by the UI
tests/                   pytest: coordinator, prediction, zoning, scenarios, experiments, rendering
docs/                    DATASETS.md, TRAINING.md
notebooks/               Colab/Kaggle training notebook
```
Run the tests with `pip install pytest && pytest -q`.
