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
| Zones | drag = add zone, `2`-`6` = auto bands, `Backspace` undo, `C` clear, `Enter` continue (zones are remembered next to the video) |
| Select | click or drag on a vehicle, `Space` play/pause to find a good frame, `,` `.` step a frame, `Esc` back to zones |
| Track | `Space` pause, `Z` zoom 2x/3x/4x/6x, `L` message loss 0/10/30 %, `1`-`6` cut/restore a UAV's telemetry (fault injection), `R` reselect, `S` screenshot |
| Any | `Q` quit |

Things worth showing in a defense: press `L` to inject 10 % packet loss, or `2` to silence UAV-2 and watch
the coordinator time it out (~3 s), fall back to another successor, or fail safely while the sending UAV keeps ownership.

Headless / scripted (also records a video of the whole window):
```bash
python main.py traffic.mp4 --headless --zones auto:4 --target 1061,311,52,28 --save output/demo.mp4
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
models/                  VitTrack ONNX (Apache-2.0, from opencv_zoo)
weights/                bootstrap vehicle detector (VisDrone fine-tune)
scripts/                 prepare_visdrone.py, train_vehicle_detector.py, yolo_bytetrack_baseline.py
tests/                   pytest for the coordinator (single owner, 10 % loss, faults, reproducibility)
docs/                    DATASETS.md, TRAINING.md
notebooks/               Colab/Kaggle training notebook
```
Run the tests with `pip install pytest && pytest -q`.
