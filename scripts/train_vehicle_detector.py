"""Fine-tune YOLOv8 on VisDrone (single 'vehicle' class) so the detector understands
top-down drone footage. Works on NVIDIA GPUs (Colab/Kaggle), AMD GPUs with a ROCm build of
PyTorch on Linux, or CPU (slow).

    python scripts/prepare_visdrone.py --out datasets/visdrone-vehicle
    python scripts/train_vehicle_detector.py --data datasets/visdrone-vehicle/data.yaml --model yolov8s.pt --imgsz 960 --epochs 60
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():  # also true for ROCm builds of PyTorch
        print("GPU:", torch.cuda.get_device_name(0))
        return "0"
    print("No GPU visible to PyTorch - training on CPU (slow).")
    return "cpu"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=-1, help="-1 = auto-fit GPU memory")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--name", default="vehicle_visdrone")
    a = ap.parse_args()
    model = YOLO(a.model)
    model.train(
        data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, fraction=a.fraction,
        device=a.device or pick_device(), project=str(Path("runs").resolve()), name=a.name, exist_ok=True,
        patience=15, cos_lr=True, close_mosaic=5, workers=4,
        flipud=0.5, degrees=15,  # aerial views have no canonical 'up'
    )
    print("Best weights:", model.trainer.best)
