"""Download VisDrone2019-DET (from the Ultralytics GitHub release mirror) and convert it to
YOLO format with a single 'vehicle' class (car, van, truck, bus) - or all 10 classes.

    python scripts/prepare_visdrone.py --out datasets/visdrone-vehicle
    python scripts/prepare_visdrone.py --out datasets/visdrone-10cls --all-classes

VisDrone annotation line: x,y,w,h,score,category,truncation,occlusion
categories: 1 pedestrian 2 people 3 bicycle 4 car 5 van 6 truck 7 tricycle
            8 awning-tricycle 9 bus 10 motor (0 = ignored region)
"""
import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-{}.zip"
SPLITS = {"train": "train", "val": "val", "test": "test-dev"}
NAMES10 = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
VEHICLE_IDS = {4, 5, 6, 9}


def fetch(split: str, cache: Path) -> Path:
    z = cache / f"VisDrone2019-DET-{SPLITS[split]}.zip"
    if not z.exists():
        print(f"Downloading {z.name} ...")
        urllib.request.urlretrieve(BASE.format(SPLITS[split]), z)
    d = cache / f"VisDrone2019-DET-{SPLITS[split]}"
    if not d.exists():
        with zipfile.ZipFile(z) as f:
            f.extractall(cache)
    return d


def convert(src: Path, dst: Path, all_classes: bool) -> int:
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    n = 0
    for img in sorted((src / "images").glob("*.jpg")):
        w, h = __import__("PIL.Image", fromlist=["Image"]).open(img).size
        lines = []
        ann = src / "annotations" / (img.stem + ".txt")
        for row in ann.read_text().strip().splitlines() if ann.exists() else []:
            x, y, bw, bh, score, cat = (int(v) for v in row.split(",")[:6])
            if score == 0 or cat == 0 or bw <= 0 or bh <= 0:
                continue
            if all_classes:
                cls = cat - 1
            elif cat in VEHICLE_IDS:
                cls = 0
            else:
                continue
            lines.append(f"{cls} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
        shutil.copy(img, dst / "images" / img.name)
        (dst / "labels" / (img.stem + ".txt")).write_text("\n".join(lines))
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("datasets/visdrone-vehicle"))
    ap.add_argument("--all-classes", action="store_true")
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=list(SPLITS))
    a = ap.parse_args()
    cache = a.out.parent / "_visdrone_raw"
    cache.mkdir(parents=True, exist_ok=True)
    for s in a.splits:
        print(s, convert(fetch(s, cache), a.out / s, a.all_classes), "images")
    names = NAMES10 if a.all_classes else ["vehicle"]
    yaml = [f"path: {a.out.resolve().as_posix()}", "train: train/images", "val: val/images",
            "names:"] + [f"  {i}: {n}" for i, n in enumerate(names)]
    (a.out / "data.yaml").write_text("\n".join(yaml) + "\n")
    print("Wrote", a.out / "data.yaml")


if __name__ == "__main__":
    main()
