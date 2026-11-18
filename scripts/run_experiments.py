"""Handover robustness experiments on a saved scenario (report NFR3: >=90 % handovers at 10 % loss).

    python scripts/run_experiments.py clip.mp4                  # uses clip.scenario.json
    python scripts/run_experiments.py clip.mp4 --seeds 100 --losses 0,0.05,0.1,0.2,0.3 --faults

The vision tracker runs once (cached in output/cache); the coordinator is then replayed for every
seed and loss level. Outputs go to output/experiments/<clip>/ : runs.csv, summary.csv, summary.md, chart PNG.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from mhvt.experiment import NFR3_LOSS, NFR3_TARGET, aggregate, sweep  # noqa: E402
from mhvt.scenario import Scenario, default_path  # noqa: E402


def record_track(video: Path, scenario: Path, refresh: bool):
    sc = Scenario.load(scenario)
    digest = hashlib.sha1(scenario.read_bytes() + str(video.stat().st_size).encode()).hexdigest()[:8]
    cache = ROOT / "output" / "cache" / f"{video.stem}.{digest}.track.json"
    if cache.is_file() and not refresh:
        d = json.loads(cache.read_text())
        print(f"Using cached target path: {cache.name}")
        return sc, d["zones_px"], [tuple(c) if c else None for c in d["track_log"]], d["fps"]
    import main as M  # the demo app, run headless over the scenario to record the target path

    args = argparse.Namespace(video=video, zones=None, target=None, scenario=scenario, replay=False, start_frame=0,
                              max_frames=None, headless=True, save=None, screenshots="", model=None, loss=0.0,
                              seed=7, zoom=3.0)
    app = M.App(args)
    app.read()
    app.start_from_args()
    print("Tracking the target once through the clip ...")
    while app.advance_track():
        pass
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"zones_px": app.zones, "track_log": app.track_log, "fps": app.fps}))
    return sc, app.zones, app.track_log, app.fps


def chart(summary, path: Path, title: str):
    reg, semi = ROOT / "assets/fonts/Inter-Regular.ttf", ROOT / "assets/fonts/Inter-SemiBold.ttf"
    f_t, f_l, f_s = (ImageFont.truetype(str(semi), 26), ImageFont.truetype(str(reg), 18), ImageFont.truetype(str(reg), 15))
    W, H, L, R, T, B = 1100, 620, 90, 40, 90, 90
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((L, 28), title, font=f_t, fill="#14202b")
    px = lambda frac: H - B - frac * (H - T - B)  # noqa: E731
    for g in (0, .25, .5, .75, 1):
        d.line([(L, px(g)), (W - R, px(g))], fill="#dde3ea")
        d.text((L - 12, px(g)), f"{int(g * 100)}%", font=f_s, fill="#5b6773", anchor="rm")
    d.line([(L, px(NFR3_TARGET)), (W - R, px(NFR3_TARGET))], fill="#d9534f", width=2)
    d.text((W - R, px(NFR3_TARGET) - 6), "NFR3 target 90%", font=f_s, fill="#d9534f", anchor="rb")
    n = len(summary)
    slot = (W - L - R) / max(1, n)
    for i, r in enumerate(summary):
        x0, x1 = L + i * slot + slot * .2, L + (i + 1) * slot - slot * .2
        v = r["success_rate"]
        if v is not None:
            col = "#2e9e5b" if v >= NFR3_TARGET else "#e0a030" if v >= .7 else "#d9534f"
            d.rectangle([x0, px(v), x1, px(0)], fill=col)
            xm = (x0 + x1) / 2
            d.line([(xm, px(r["ci95_low"])), (xm, px(r["ci95_high"]))], fill="#14202b", width=3)
            d.text((xm, px(v) - 10), f"{v * 100:.0f}%", font=f_l, fill="#14202b", anchor="mb")
        d.text(((x0 + x1) / 2, H - B + 14), f"{int(r['loss'] * 100)}% loss", font=f_l, fill="#14202b", anchor="mt")
        d.text(((x0 + x1) / 2, H - B + 40), f"{r['ok']}/{r['handover_attempts']} handovers", font=f_s, fill="#5b6773", anchor="mt")
    d.text((L, H - 24), "Bars: handover success rate. Whiskers: 95% Wilson interval. Simulated link; real tracked target path.",
           font=f_s, fill="#5b6773")
    im.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--scenario", type=Path)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--losses", default="0,0.05,0.1,0.2,0.3")
    ap.add_argument("--faults", action="store_true", help="also replay the scenario's UAV fault events")
    ap.add_argument("--refresh", action="store_true", help="re-run the tracker instead of using the cache")
    a = ap.parse_args()
    scen = a.scenario or default_path(a.video)
    if not scen.is_file():
        sys.exit(f"No scenario at {scen}. Save one in the demo with K first.")
    sc, zones, track, fps = record_track(a.video, scen, a.refresh)
    tracked = sum(c is not None for c in track)
    print(f"Target tracked for {tracked / fps:.1f}s of {len(track) / fps:.1f}s")
    faults = [e for e in sc.events if e["type"] == "fault"] if a.faults else []
    losses = [float(x) for x in a.losses.split(",")]
    rows = sweep(zones, track, fps, a.seeds, losses, faults)
    summary = aggregate(rows)
    out = ROOT / "output" / "experiments" / a.video.stem
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("runs.csv", rows), ("summary.csv", summary)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
    f = lambda v, p=1: "-" if v is None else f"{v * 100:.{p}f}%"  # noqa: E731
    lines = [f"# Handover experiment: {a.video.name}", "",
             f"{a.seeds} seeds per loss level, {'with' if faults else 'without'} injected UAV faults. "
             f"Target tracked {tracked / fps:.1f}s of {len(track) / fps:.1f}s.", "",
             "| Message loss | Runs | Handover attempts | Success | 95% CI | Mean latency | Delivery ratio | NFR3 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in summary:
        nfr = "-" if r["meets_nfr3"] is None else ("PASS" if r["meets_nfr3"] else "FAIL")
        lat = "-" if r["latency_mean_s"] is None else f"{r['latency_mean_s']:.2f}s"
        ci = "-" if r["success_rate"] is None else f"{r['ci95_low'] * 100:.0f}-{r['ci95_high'] * 100:.0f}%"
        lines.append(f"| {int(r['loss'] * 100)}% | {r['runs']} | {r['handover_attempts']} | {f(r['success_rate'])} | "
                     f"{ci} | {lat} | {f(r['delivery_ratio'])} | {nfr} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    chart(summary, out / "success_vs_loss.png", f"Handover success vs message loss - {a.video.name}")
    print("\n".join(lines))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
