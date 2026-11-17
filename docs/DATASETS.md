# Datasets for training and testing MHVT

Only the **VisDrone-DET** links were verified from this project's build environment (they download
fine and `scripts/prepare_visdrone.py` converts them). Check every other link/licence yourself.

## 1. Train the vehicle detector (needed - COCO weights fail on top-down footage)
| Dataset | What you get | Use here |
|---|---|---|
| **VisDrone2019-DET** (verified) | 6,471 train / 548 val / 1,610 test-dev drone images, 10 classes incl. car, van, truck, bus | Main training set. `python scripts/prepare_visdrone.py` merges car/van/truck/bus into one `vehicle` class |
| **UAVDT** | ~80k frames of drone traffic video (car/truck/bus), detection + tracking labels, weather/altitude/view attributes | Extra training data; closest to road-traffic UAV footage |
| **DroneVehicle** | ~56k RGB+thermal drone images, vehicle boxes (oriented) | Optional extra, night/thermal robustness |

## 2. Test tracking (needs identity labels, so you can report MOTA / IDF1 / ID switches - UC-04, NFR2)
| Dataset | Notes |
|---|---|
| **VisDrone-MOT** | Official multi-object tracking benchmark with track IDs (train/val/test-dev). Your report already cites it |
| **UAVDT-MOT** | Vehicle tracks with IDs on traffic videos |

Score with [TrackEval](https://github.com/JonathonLuiten/TrackEval) (HOTA, MOTA, IDF1) or `py-motmetrics`.
Run ByteTrack and BoT-SORT with the *same* detector weights so the comparison is fair.

## 3. Ground-truth trajectories for the handover simulation (no vision needed)
| Dataset | Notes |
|---|---|
| **highD** | ~110k vehicles on German highways recorded by a drone: per-vehicle position/speed/lane CSVs. Mostly trajectories, not raw video. Free for non-commercial research after registration. Ideal for replaying real highway traffic through the coordinator and for calibrating SUMO |

## 4. Demo footage
Free stock drone traffic clips (Pexels, Pixabay, Mixkit) - pick top-down, daylight, steady shots.
Your own `traffic.mp4` is good for the demo but the drone moves fast, so vehicles leave the frame in
~5-8 s; a hovering or slow-moving clip shows longer tracks and handovers.

## Not a good fit
UA-DETRAC (ground-level cameras, not drones), CARPK (static parking-lot counting), Stanford Drone
(mostly pedestrians on a campus).
