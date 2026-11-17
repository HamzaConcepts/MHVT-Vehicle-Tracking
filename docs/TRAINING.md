# Training the vehicle detector

Why: the stock COCO `yolov8n.pt` finds only ~5% of the vehicles in top-down drone video
(measured on `traffic.mp4`: 0.5 vehicles/frame vs ~20-40 visible). Fine-tuning on VisDrone fixes that.

## Can I train on an RX 6600 XT (8 GB)?
8 GB is enough memory for YOLOv8n/s at 640-960 px. The problem is software support, not VRAM
(this is from general knowledge - check AMD's current compatibility matrix):

| Route | Verdict |
|---|---|
| **Linux + ROCm PyTorch** | Works for many RX 6000 cards: the 6600 XT (gfx1032) is not officially supported but usually runs with `export HSA_OVERRIDE_GFX_VERSION=10.3.0`. Install the ROCm build of PyTorch from pytorch.org, then `torch.cuda.is_available()` should be `True` and `scripts/train_vehicle_detector.py` just works |
| **Windows (native / WSL2)** | ROCm for PyTorch on Windows targets newer cards (RX 7000/9000); RX 6000 is, as far as I know, not covered |
| **torch-directml (Windows)** | Runs on any DX12 GPU but Ultralytics does not support it; expect missing ops, crashes and slow training. Not recommended |
| **Free cloud GPU (recommended)** | Kaggle Notebooks or Google Colab (T4 class). Roughly 3-5 min/epoch for YOLOv8n at 960 px on VisDrone, so 40 epochs is a few hours. Use `notebooks/train_colab.ipynb` |
| **CPU** | Works (a 4-core CPU did ~17 min/epoch at 640 px) but only for a quick bootstrap |

The detector does **not** need your GPU at run time: the demo and a 640-1280 px YOLOv8n run on CPU.

## Steps
```bash
pip install -r requirements-train.txt
python scripts/prepare_visdrone.py --out datasets/visdrone-vehicle      # ~1.7 GB download
python scripts/train_vehicle_detector.py --data datasets/visdrone-vehicle/data.yaml \
       --model yolov8n.pt --imgsz 960 --epochs 40            # use yolov8s.pt if you have time
```
The script prints the best-weights path (`runs/vehicle_visdrone/weights/best.pt`). Use them in the demo with
`python main.py traffic.mp4 --model runs/vehicle_visdrone/weights/best.pt` (shows detections and snaps
your click to the vehicle you point at).

Then report `mAP50` on the validation split and the tracker numbers from VisDrone-MOT (see DATASETS.md).

## Bootstrap result already in the repo
`weights/vehicle_visdrone_n_bootstrap.pt`: YOLOv8n, 640 px, 6 epochs on 40 % of VisDrone-DET train, CPU only (~45 min).
VisDrone val (vehicle class): P 0.71 / R ~0.5 / **mAP50 0.64** / mAP50-95 0.36 (still rising at epoch 6).
A full GPU run at 960 px should do clearly better, especially on small/distant vehicles.
