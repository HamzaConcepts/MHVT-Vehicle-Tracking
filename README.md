# MHVT Automatic Car Tracking Demo

This project is a small computer-vision demonstration for the **Multi-Drone
Handover System for Continuous Vehicle Tracking (MHVT)** FYP. It detects and
tracks cars in a prerecorded road-traffic video. It does not simulate
drones and does not use ROS 2, SUMO, or a web dashboard.

## Technologies

- Python 3.10 or 3.11
- YOLOv8n for lightweight car detection
- ByteTrack through Ultralytics `model.track()` for automatic car IDs
- OpenCV for video input, display, annotation, and output

## Setup

1. Open this folder in VS Code.
2. Create and activate a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If `py -3.11` is unavailable, use an installed Python 3.10 or 3.11
   interpreter instead.

3. Install the dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

The first run downloads the pretrained `yolov8n.pt` model automatically if it
is not already present. CPU inference is selected by the program, so no GPU
package is required. The project uses
[`bytetrack_mhvt.yaml`](bytetrack_mhvt.yaml), which keeps lost tracks alive for
a short occlusion while avoiding immediate ID reuse.

## Input and output

Put the provided prerecorded video at:

```text
MHVT_Vehicle_Tracking/traffic.mp4
```

Run the demo with:

```powershell
python main.py
```

Tracking starts automatically when the video opens; no clicks or manual
vehicle selection are used. A detector confidence threshold of `0.25` rejects
very weak detections, while ByteTrack uses separate association thresholds.
New IDs are drawn only after two consecutive observations. The window
**MHVT - Automatic Vehicle Tracking**
shows all active car tracks with confidence scores and ByteTrack IDs. YOLO uses
a moderate `0.20` confidence threshold and is
restricted to COCO class 2 (`car`), so trucks, buses, motorcycles, and
non-car classes are not passed to the tracker. Green boxes are boxes returned
by the detector/tracker for the current frame. The program does not draw
guessed boxes or silently transfer a lost ID to another vehicle. The annotated
video is saved as
`output/tracked_traffic.mp4`.

ByteTrack attempts to keep an ID attached to the same vehicle across
consecutive frames. IDs can change when a vehicle is hidden, detections are
lost, or vehicles overlap; they are not guaranteed to remain unchanged in
every situation.

Press **Q** while the video window is focused to stop processing. Resources
are released automatically.

To run a lightweight headless check over consecutive frames, use:

```powershell
python main.py --test-frames 120
```

This prints the number of unique IDs, frames containing active IDs, temporary
missing-track observations, and possible identity-switch candidates. The last
metric is an IoU-based warning only; proving identity switches requires
ground-truth labels. To verify an ID visually, pause the output video and
follow the same car across consecutive frames. Its `ID N` label should remain
unchanged while it is visible. A short missed-detection gap may occur while
ByteTrack retains the track; a new ID after a genuine exit/re-entry is
expected.

Optional webcam mode is available without changing the default video mode:

```powershell
python main.py --camera
```

## Detection versus tracking

Object detection finds cars independently in each frame and returns bounding
boxes. Object tracking links detections across frames, allowing the
application to assign and maintain an ID for a car over time.

## Common errors

- **`traffic.mp4` not found:** Confirm the filename is exactly `traffic.mp4`
  and that it is in the same folder as `main.py`.
- **Video cannot be opened:** Try a standard MP4 encoded with H.264 or
  MPEG-4, and confirm the file is not corrupted.
- **`No module named ...`:** Activate `.venv` and run
  `python -m pip install -r requirements.txt` again.
- **Slow processing:** Close other applications, use a shorter video, and
  keep the lightweight 960x540 display and `imgsz=960` settings.
- **Model download failure:** Check the internet connection and run the
  program again; `yolov8n.pt` is downloaded only when it is not already
  available locally.
