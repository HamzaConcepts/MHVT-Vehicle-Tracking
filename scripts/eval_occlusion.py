"""Benchmark the bridge / occlusion handling on any video.

For every vehicle found in the first frame we (1) follow it with no bridge to get a reference path (only the
tracker's own direct observations of that vehicle), then (2) re-run with a simulated bridge placed across its path
at several moments, and score what the path predictor + re-acquisition did:

  correct        target found again and it is the SAME vehicle (median position error < 30 px vs the reference)
  WRONG_VEHICLE  target found again but it is a different vehicle
  no_recovery    the system kept predicting and finally gave up

    python scripts/eval_occlusion.py clip.mp4
    python scripts/eval_occlusion.py clip.mp4 --imgsz 960 --max-targets 8

No labels are needed, but the result depends on how stable the detector's tracker IDs are: a vehicle only
qualifies if the tracker followed it cleanly for 100+ frames. A better detector qualifies more vehicles.
"""
import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import main as M  # noqa: E402


def make_app(video, occluder, a):
    ns = argparse.Namespace(video=video, zones=None, target=None, scenario=None, replay=False, start_frame=0, max_frames=None,
                            headless=True, save=None, screenshots="", model=a.model, loss=0.0, seed=7, zoom=3.0,
                            tracker=a.tracker, imgsz=a.imgsz, conf=a.conf, max_occlusion=a.max_occlusion,
                            occluder=[occluder] if occluder else [], zone_count=None, zone_layout="vertical", zone_overlap=0.15)
    app = M.App(ns)
    app.read()
    app.set_equal_zones(3)
    return app


def run(app, bbox, frames):
    app.begin_run(bbox)
    tid0 = getattr(app.follower, "mgr", None) and app.follower.mgr.target_id
    rows = []
    for _ in range(frames):
        if not app.advance_track():
            break
        c = None if app.bbox is None or app.follower.lost else (app.bbox[0] + app.bbox[2] / 2, app.bbox[1] + app.bbox[3] / 2)
        raw = c if (c is not None and not app.predicted and app.follower.mgr.target_id == tid0) else None
        rows.append((c, bool(app.predicted), app.follower.lost, raw))
    return tid0, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "botsort"])
    ap.add_argument("--model")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--max-occlusion", type=float, default=8.0)
    ap.add_argument("--frames", type=int, default=130)
    ap.add_argument("--max-targets", type=int, default=16)
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "occlusion_eval.json")
    a = ap.parse_args()
    fps = make_app(a.video, None, a).fps
    app0 = make_app(a.video, None, a)
    boxes = app0.follower.preview(app0.work)
    W, H = M.R.WORK_W, M.R.WORK_H
    cands = sorted([b for b in boxes if 0.3 * H < b[1] + b[3] / 2 < 0.7 * H and 30 < b[2] < 90 and 15 < b[3] < 60
                    and 0.15 * W < b[0] + b[2] / 2 < 0.85 * W], key=lambda b: b[0])[:a.max_targets]
    print(f"{len(boxes)} vehicles in frame 0, testing up to {len(cands)}")
    results = []
    for n, b in enumerate(cands):
        tid0, base = run(make_app(a.video, None, a), b, a.frames)
        if tid0 is None:
            continue
        raw = [r[3] for r in base]
        seq = next((i for i, r in enumerate(raw) if r is None), len(raw))
        if seq < 100:
            print(f"target {n}: skipped (reference only {seq} clean frames)")
            continue
        for start in (14, 40, 66):
            win = [raw[i] for i in range(start, start + 12)]
            if any(w is None for w in win) or any(raw[i] is None for i in range(start - 6, min(seq, start + 40))):
                continue
            xs = [w[0] for w in win]
            x1, x2 = max(0, min(xs) - 20), min(W, max(xs) + 20)
            _, br = run(make_app(a.video, f"{x1 / W:.4f},0,{x2 / W:.4f},1", a), b, a.frames)
            first_pred = next((i for i, r in enumerate(br) if r[1] and i >= start - 4), None)
            rec = None
            if first_pred is not None:
                rec = next((i for i in range(first_pred, len(br)) if not br[i][1] and br[i][0] is not None and not br[i][2]), None)
            err = lambda i: math.dist(br[i][0], raw[i]) if (i < len(br) and br[i][0] and raw[i]) else None  # noqa: E731
            post = [err(i) for i in range(rec, min(len(raw), rec + 60)) if err(i) is not None] if rec is not None else []
            pred = [err(i) for i in range(len(raw)) if i < len(br) and br[i][1] and err(i) is not None]
            med = statistics.median(post) if post else None
            outcome = "no_recovery" if rec is None else ("correct" if med is not None and med < 30 else "WRONG_VEHICLE")
            r = dict(target=n, start_frame=start, hidden_s=None if (first_pred is None or rec is None) else round((rec - first_pred) / fps, 2),
                     prediction_error_px=round(statistics.mean(pred), 1) if pred else None,
                     error_after_recovery_px=round(med, 1) if med is not None else None, outcome=outcome)
            results.append(r)
            print(json.dumps(r))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=1))
    c = Counter(r["outcome"] for r in results)
    n = max(1, len(results))
    print(f"\n{len(results)} bridge placements: " + ", ".join(f"{k} {v} ({100 * v / n:.0f}%)" for k, v in c.items()))
    pe = [r["prediction_error_px"] for r in results if r["prediction_error_px"] is not None]
    if pe:
        print(f"median prediction error under the bridge: {statistics.median(pe):.0f} px")
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
