"""Drawing code for the MHVT mission-control style demo window (ASCII text only:
OpenCV's Hershey fonts cannot render Unicode)."""

from __future__ import annotations

import cv2
import numpy as np

from .coordinator import Coordinator, DState

CANVAS_W, CANVAS_H = 1600, 900
MAIN_W, MAIN_H = 1040, 585            # main video panel (16:9)
SIDE_X = MAIN_W                       # right column x offset
SIDE_W = CANVAS_W - MAIN_W            # 560
ZOOM_H = 315
WORK_W, WORK_H = 1280, 720            # working resolution used for tracking
SCALE = MAIN_W / WORK_W               # working px -> main panel px

BG = (28, 24, 22)
PANEL = (44, 38, 34)
TEXT = (235, 232, 228)
DIM = (150, 145, 140)
GREEN, AMBER, RED, CYAN, BLUE = (90, 220, 120), (40, 180, 255), (70, 80, 240), (230, 200, 60), (255, 160, 70)
ZONE_COLORS = [(255, 150, 60), (90, 200, 255), (130, 230, 130), (220, 120, 230), (80, 160, 255), (200, 200, 90)]
STATE_COLORS = {
    DState.TRACKING: GREEN, DState.HANDING_OVER: AMBER, DState.ACQUIRING: CYAN,
    DState.SEARCHING: RED, DState.OFFLINE: (110, 110, 110), DState.STANDBY: (190, 170, 150),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def text(img, s, org, scale=0.5, color=TEXT, thick=1):
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def zone_color(i: int):
    return ZONE_COLORS[i % len(ZONE_COLORS)]


def blend_rect(img, p1, p2, color, alpha):
    x1, y1, x2, y2 = (max(0, p1[0]), max(0, p1[1]), min(img.shape[1] - 1, p2[0]), min(img.shape[0] - 1, p2[1]))
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    roi[:] = (roi * (1 - alpha) + np.array(color, np.float32) * alpha).astype(np.uint8)


def sp(p):
    return int(p[0] * SCALE), int(p[1] * SCALE)


def draw_zones(main, coord: Coordinator | None, zones, active=None, pending=None, t=0.0):
    """zones: list of (x1,y1,x2,y2) in working px."""
    for i, z in enumerate(zones):
        c = zone_color(i)
        a, b = sp(z[:2]), sp(z[2:])
        blend_rect(main, a, b, c, 0.10)
        thick = 1
        label = f"UAV-{i + 1} ZONE"
        if coord is not None:
            d = coord.drone(i + 1)
            if coord.owner == d.id:
                thick, label = 3, f"UAV-{i + 1} PRIMARY"
            elif d.state == DState.ACQUIRING and int(t * 4) % 2 == 0:
                thick = 3
            elif d.state == DState.OFFLINE:
                c = (110, 110, 110)
        cv2.rectangle(main, a, b, c, thick, cv2.LINE_AA)
        text(main, label, (a[0] + 8, a[1] + 20 + 20 * (i % 2)), 0.5, c, 1)
    if pending is not None:
        cv2.rectangle(main, sp(pending[:2]), sp(pending[2:]), TEXT, 1, cv2.LINE_AA)


def draw_target(main, bbox, score, lost, trail, target_id=1):
    if bbox is not None and not lost:
        x, y, w, h = bbox
        a, b = sp((x, y)), sp((x + w, y + h))
        pad = 6
        a, b = (a[0] - pad, a[1] - pad), (b[0] + pad, b[1] + pad)
        cv2.rectangle(main, a, b, GREEN, 2, cv2.LINE_AA)
        for (px, py), (dx, dy) in (((a[0], a[1]), (1, 1)), ((b[0], a[1]), (-1, 1)),
                                   ((a[0], b[1]), (1, -1)), ((b[0], b[1]), (-1, -1))):
            cv2.line(main, (px, py), (px + 12 * dx, py), GREEN, 4, cv2.LINE_AA)
            cv2.line(main, (px, py), (px, py + 12 * dy), GREEN, 4, cv2.LINE_AA)
        label = f"TARGET #{target_id}  conf {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        cv2.rectangle(main, (a[0], a[1] - th - 10), (a[0] + tw + 8, a[1]), GREEN, -1)
        text(main, label, (a[0] + 4, a[1] - 5), 0.5, (20, 20, 20), 1)
    pts = [sp(p) for p in trail]
    for i in range(1, len(pts)):
        k = i / len(pts)
        cv2.line(main, pts[i - 1], pts[i], (int(60 + 30 * k), int(180 * k + 40), int(90 + 100 * k)), 2, cv2.LINE_AA)


def draw_handover(main, coord: Coordinator, zones):
    h = coord.handover
    if not h:
        return
    a = sp(((zones[h.source - 1][0] + zones[h.source - 1][2]) / 2, (zones[h.source - 1][1] + zones[h.source - 1][3]) / 2))
    b = sp(((zones[h.successor - 1][0] + zones[h.successor - 1][2]) / 2, (zones[h.successor - 1][1] + zones[h.successor - 1][3]) / 2))
    cv2.arrowedLine(main, a, b, AMBER, 3, cv2.LINE_AA, tipLength=0.08)
    mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2 - 12)
    label = f"HANDOVER #{h.hid}: UAV-{h.source} -> UAV-{h.successor}  [{h.phase}]"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 1)
    cv2.rectangle(main, (mid[0] - 6, mid[1] - th - 6), (mid[0] + tw + 6, mid[1] + 8), (20, 20, 20), -1)
    text(main, label, mid, 0.55, AMBER, 1)


def draw_status_banner(main, status: str, color):
    cv2.rectangle(main, (0, MAIN_H - 30), (MAIN_W, MAIN_H), (20, 20, 20), -1)
    text(main, status, (10, MAIN_H - 10), 0.55, color, 1)


def zoom_panel(full_frame, center_work, lost, zoom, drone_name, size=(SIDE_W, ZOOM_H)):
    """Crop around the (smoothed) target centre from the full-resolution frame."""
    pw, ph = size
    fh, fw = full_frame.shape[:2]
    k = fw / WORK_W                       # full-res px per working px
    cw, ch = pw / zoom * k, ph / zoom * k  # crop size in full-res px
    cx, cy = center_work[0] * k, center_work[1] * k
    x0 = int(min(max(cx - cw / 2, 0), max(0, fw - cw)))
    y0 = int(min(max(cy - ch / 2, 0), max(0, fh - ch)))
    crop = full_frame[y0:y0 + int(ch), x0:x0 + int(cw)]
    panel = cv2.resize(crop, size, interpolation=cv2.INTER_CUBIC)
    if lost:
        panel = (panel * 0.35).astype(np.uint8)
    col = RED if lost else GREEN
    m = 14
    for (px, py), (dx, dy) in (((m, m), (1, 1)), ((pw - m, m), (-1, 1)), ((m, ph - m), (1, -1)), ((pw - m, ph - m), (-1, -1))):
        cv2.line(panel, (px, py), (px + 26 * dx, py), col, 2, cv2.LINE_AA)
        cv2.line(panel, (px, py), (px, py + 26 * dy), col, 2, cv2.LINE_AA)
    # crosshair is drawn at the target centre only while not lost
    if not lost:
        tx = int((cx - x0) / cw * pw)
        ty = int((cy - y0) / ch * ph)
        cv2.drawMarker(panel, (tx, ty), GREEN, cv2.MARKER_CROSS, 18, 1, cv2.LINE_AA)
    title = f"{drone_name} CAMERA  {zoom:.0f}x  " + ("TARGET LOST" if lost else "LOCKED")
    cv2.rectangle(panel, (0, 0), (pw, 26), (20, 20, 20), -1)
    text(panel, title, (10, 18), 0.5, col, 1)
    return panel


def bar(img, x, y, w, h, frac, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), (70, 64, 60), -1)
    cv2.rectangle(img, (x, y), (x + int(w * max(0.0, min(1.0, frac))), y + h), color, -1)


def telemetry_panel(coord: Coordinator, tracking_status: str, score: float, size, t: float):
    pw, ph = size
    p = np.full((ph, pw, 3), PANEL, np.uint8)
    # header
    text(p, "MISSION CONTROL  (telemetry simulated)", (12, 22), 0.55, TEXT, 1)
    own = coord.drone(coord.owner).name if coord.owner else "none"
    col = {"TRACKING": GREEN, "LOST": RED, "IDLE": DIM}.get(tracking_status, TEXT)
    text(p, f"Target #1: {tracking_status}", (12, 48), 0.55, col, 1)
    text(p, f"Primary owner: {own}", (230, 48), 0.55, TEXT, 1)
    s = coord.summary()
    rate = "-" if s["success_rate"] is None else f"{100 * s['success_rate']:.0f}%"
    text(p, f"Handovers ok/failed: {s['handovers_ok']}/{s['handovers_failed']} ({rate})   "
            f"msg loss {int(100 * coord.cfg.loss_prob)}%  ({s['msgs_lost']}/{s['msgs_sent']} dropped)",
         (12, 72), 0.45, DIM, 1)
    cv2.line(p, (0, 82), (pw, 82), (70, 64, 60), 1)
    n = len(coord.drones)
    top, avail = 90, ph - 96
    ch = min(150, avail // max(1, n))
    for i, d in enumerate(coord.drones):
        y = top + i * ch
        c = zone_color(i)
        card = (y, y + ch - 8)
        primary = coord.owner == d.id
        cv2.rectangle(p, (8, card[0]), (pw - 8, card[1]), (58, 51, 46), -1)
        cv2.rectangle(p, (8, card[0]), (14, card[1]), c, -1)
        if primary:
            cv2.rectangle(p, (8, card[0]), (pw - 8, card[1]), GREEN, 2)
        text(p, d.name + ("  PRIMARY" if primary else ""), (24, card[0] + 22), 0.6, TEXT, 1)
        sc = STATE_COLORS.get(d.state, DIM)
        label = d.state.value
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        px = pw - 20 - tw - 12
        cv2.rectangle(p, (px, card[0] + 6), (px + tw + 12, card[0] + 28), sc, -1)
        text(p, label, (px + 6, card[0] + 22), 0.5, (20, 20, 20), 1)
        by = card[0] + 40
        text(p, "BAT", (24, by + 10), 0.4, DIM, 1)
        bat_col = GREEN if d.battery >= 50 else AMBER if d.battery >= coord.cfg.battery_min else RED
        bar(p, 62, by, 170, 10, d.battery / 100, bat_col)
        text(p, f"{d.battery:4.0f}%", (240, by + 10), 0.4, TEXT, 1)
        text(p, "LINK", (300, by + 10), 0.4, DIM, 1)
        fresh = 1 - min(1.0, d.link_age / coord.cfg.offline_timeout)
        bar(p, 345, by, 110, 10, fresh, GREEN if d.link_age <= coord.cfg.link_age_max else RED)
        text(p, f"{d.link_age:3.1f}s", (462, by + 10), 0.4, TEXT, 1)
        if ch >= 100:
            sc_txt = f"HandoverScore {d.score:.2f}" if (d.score is not None and any(r["id"] == d.id for r in coord.ranking_view)) else "HandoverScore  --"
            text(p, sc_txt, (24, by + 36), 0.45, DIM, 1)
            if d.silent:
                text(p, "FAULT: telemetry cut", (300, by + 36), 0.45, RED, 1)
            elif d.battery < coord.cfg.battery_min:
                text(p, "LOW BATTERY", (300, by + 36), 0.45, RED, 1)
    return p


LOG_COLORS = {"info": TEXT, "ok": GREEN, "warn": AMBER, "err": RED}


def log_panel(coord: Coordinator, size, help_text: str):
    pw, ph = size
    p = np.full((ph, pw, 3), BG, np.uint8)
    text(p, "COORDINATOR LOG", (12, 22), 0.55, TEXT, 1)
    lines = ph // 20 - 3
    for i, e in enumerate(coord.events[-lines:]):
        text(p, f"{e.t:6.2f}s  {e.text}"[:120], (12, 48 + i * 20), 0.45, LOG_COLORS.get(e.level, TEXT), 1)
    text(p, help_text, (12, ph - 12), 0.45, DIM, 1)
    return p


def compose(main, zoom, tel, log):
    canvas = np.full((CANVAS_H, CANVAS_W, 3), BG, np.uint8)
    canvas[0:MAIN_H, 0:MAIN_W] = main
    canvas[MAIN_H:CANVAS_H, 0:MAIN_W] = log
    canvas[0:ZOOM_H, SIDE_X:CANVAS_W] = zoom
    canvas[ZOOM_H:CANVAS_H, SIDE_X:CANVAS_W] = tel
    return canvas
