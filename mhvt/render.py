"""Mission-control UI. Everything is drawn at the window's real pixel size (no resampling of the
finished picture), text is anti-aliased TrueType (bundled Inter), shapes are anti-aliased.

main.py builds a `Scene` each frame and calls `UI.render(scene)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .coordinator import Coordinator, DState

WORK_W, WORK_H = 1280, 720            # working resolution used for tracking
DESIGN_W, DESIGN_H = 1600, 900        # layout is designed at this size and scaled uniformly
FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"


def hx(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)  # BGR


BG, PANEL, CARD, LINE = hx("#0D1117"), hx("#151B23"), hx("#1D2631"), hx("#2A3442")
TEXT, DIM, FAINT = hx("#E8EDF3"), hx("#8C98A8"), hx("#5B6777")
GREEN, AMBER, RED, CYAN, BLUE = hx("#3DDC84"), hx("#FFB020"), hx("#FF5C5C"), hx("#4CC9F0"), hx("#5B9DFF")
INK = hx("#0B0F14")
ZONE_COLORS = [hx(c) for c in ("#4C9AFF", "#FFB020", "#3DDC84", "#C77DFF", "#FF6B9A", "#2EC4B6")]
STATE_COLORS = {
    DState.TRACKING: GREEN, DState.HANDING_OVER: AMBER, DState.ACQUIRING: CYAN,
    DState.SEARCHING: RED, DState.OFFLINE: FAINT, DState.STANDBY: DIM,
}

HINTS = {
    "zones": [("Drag", "Add zone"), ("2-6", "Auto zones"), ("Bksp", "Undo"), ("C", "Clear"), ("Enter", "Continue"), ("O", "Load scenario"), ("Q", "Quit")],
    "select": [("Click", "Pick vehicle"), ("Space", "Play / pause"), (", .", "Step frame"), ("Esc", "Back to zones"), ("O", "Load scenario"), ("Q", "Quit")],
    "track": [("Space", "Pause"), ("T", "Restart"), ("K", "Save scenario"), ("E", "Report"), ("Z", "Zoom"), ("L", "Msg loss"),
              ("1-6", "UAV fault"), ("R", "Reselect"), ("F", "Fullscreen"), ("Q", "Quit")],
}
INSTRUCTION = {
    "zones": "Step 1 of 3 - drag to draw one zone per UAV, or press 2-6 for automatic zones",
    "select": "Step 2 of 3 - click or drag a box on the vehicle to track (Space plays the video)",
}


@dataclass
class Scene:
    stage: str                       # zones | select | track
    work: np.ndarray                 # 1280x720 BGR frame
    full: np.ndarray                 # full-resolution frame (for the zoom feed)
    zones: list                      # (x1,y1,x2,y2) in working px
    coord: Coordinator
    video_name: str = ""
    bbox: tuple | None = None
    score: float = 0.0
    lost: bool = True
    trail: list = field(default_factory=list)
    zoom: float = 3.0
    zoom_center: tuple = (WORK_W / 2, WORK_H / 2)
    drag: tuple | None = None        # (x1,y1,x2,y2) working px being dragged
    detections: list = field(default_factory=list)
    replaying: bool = False
    paused: bool = False
    ended: bool = False
    frame_no: int = 0
    nframes: int = 0
    t: float = 0.0
    toast: str = ""


@dataclass
class Layout:
    W: int
    H: int
    s: float
    ox: int
    oy: int

    @classmethod
    def fit(cls, w: int, h: int) -> "Layout":
        s = max(0.3, min(w / DESIGN_W, h / DESIGN_H))
        return cls(w, h, s, int((w - DESIGN_W * s) / 2), int((h - DESIGN_H * s) / 2))

    def u(self, v: float) -> int:
        return int(round(v * self.s))

    def rect(self, x, y, w, h) -> tuple[int, int, int, int]:
        return (self.ox + self.u(x), self.oy + self.u(y), self.ox + self.u(x + w), self.oy + self.u(y + h))

    # panel rectangles (design units)
    HEADER = (12, 8, 1576, 44)
    VIDEO = (12, 60, 1020, 574)
    LOG = (12, 646, 1020, 242)
    ZOOM = (1044, 60, 544, 306)
    TEL = (1044, 378, 544, 510)

    def video_px(self):
        return self.rect(*self.VIDEO)

    def video_scale(self) -> float:
        x1, _, x2, _ = self.video_px()
        return (x2 - x1) / WORK_W


@lru_cache(maxsize=512)
def _rr_mask(w: int, h: int, r: int):
    r = max(0, min(r, w // 2, h // 2))
    m = np.zeros((h, w), np.uint8)
    cv2.rectangle(m, (r, 0), (w - 1 - r, h - 1), 255, -1)
    cv2.rectangle(m, (0, r), (w - 1, h - 1 - r), 255, -1)
    for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
        cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return m.astype(np.float32)[:, :, None] / 255.0


_GLYPH_CACHE: dict = {}
_LUT_CACHE: dict = {}
_WIDTH_CACHE: dict = {}


class UI:
    def __init__(self, layout: Layout):
        self.L = layout
        self.canvas = np.empty((layout.H, layout.W, 3), np.uint8)
        self._fonts: dict = {}

    # ----------------------------------------------------------- primitives
    def font(self, size: float, bold: bool = False):
        px = max(8, int(round(size * self.L.s)))
        key = (px, bold)
        if key not in self._fonts:
            f = FONT_DIR / ("Inter-SemiBold.ttf" if bold else "Inter-Regular.ttf")
            self._fonts[key] = ImageFont.truetype(str(f), px)
        return self._fonts[key]

    def _glyphs(self, f, key, s, anc):
        """Anti-aliased coverage mask for a string, cached (most strings repeat between frames)."""
        k = (key, s, anc)
        hit = _GLYPH_CACHE.get(k)
        if hit is None:
            l, t, r, b = f.getbbox(s, anchor=anc)
            w, h = r - l + 4, b - t + 4
            img = Image.new("L", (max(1, w), max(1, h)), 0)
            ImageDraw.Draw(img).text((2 - l, 2 - t), s, font=f, fill=255, anchor=anc)
            hit = (np.asarray(img, np.float32)[:, :, None] / 255.0, l - 2, t - 2, r - l)
            if len(_GLYPH_CACHE) > 6000:
                _GLYPH_CACHE.clear()
            _GLYPH_CACHE[k] = hit
        return hit

    def text(self, x, y, s, size=13, color=TEXT, bold=False, anchor="l", maxw=None) -> int:
        """Draw anti-aliased text, vertically centred on y. x,y in canvas px. Returns the width."""
        f = self.font(size, bold)
        s = s.replace("->", "\u2192")
        if maxw is not None and f.getlength(s) > maxw:
            while len(s) > 1 and f.getlength(s + "\u2026") > maxw:
                s = s[:-1]
            s += "\u2026"
        anc = {"l": "lm", "r": "rm", "c": "mm"}[anchor]
        mask, ox, oy, width = self._glyphs(f, (round(size * self.L.s), bold), s, anc)
        x0, y0 = int(x) + ox, int(y) + oy
        h, w = mask.shape[:2]
        H, W = self.canvas.shape[:2]
        sx, sy = max(0, -x0), max(0, -y0)
        ex, ey = w - max(0, x0 + w - W), h - max(0, y0 + h - H)
        if ex <= sx or ey <= sy:
            return width
        reg = self.canvas[y0 + sy:y0 + ey, x0 + sx:x0 + ex]
        m = mask[sy:ey, sx:ex]
        reg[:] = (reg * (1 - m) + np.array(color, np.float32) * m).astype(np.uint8)
        return width

    def twidth(self, s, size=13, bold=False) -> int:
        k = (round(size * self.L.s), bold, s)
        w = _WIDTH_CACHE.get(k)
        if w is None:
            w = _WIDTH_CACHE[k] = int(self.font(size, bold).getlength(s.replace("->", "→")))
        return w

    def _blend(self, x1, y1, x2, y2, mask, color, alpha):
        H, W = self.canvas.shape[:2]
        sx, sy = max(0, -x1), max(0, -y1)
        ex, ey = x2 - x1 - max(0, x2 - W), y2 - y1 - max(0, y2 - H)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            return
        m = mask[sy:ey, sx:ex] * alpha
        reg = self.canvas[y1:y2, x1:x2]
        reg[:] = (reg * (1 - m) + np.array(color, np.float32) * m).astype(np.uint8)

    def rect_alpha(self, x1, y1, x2, y2, color, alpha):
        H, W = self.canvas.shape[:2]
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))
        if x2 <= x1 or y2 <= y1:
            return
        key = (color, round(alpha, 3))
        lut = _LUT_CACHE.get(key)
        if lut is None:
            ramp = np.arange(256, dtype=np.float32)
            lut = np.empty((256, 1, 3), np.uint8)
            for ch in range(3):
                lut[:, 0, ch] = (ramp * (1 - alpha) + color[ch] * alpha).astype(np.uint8)
            _LUT_CACHE[key] = lut
        roi = self.canvas[y1:y2, x1:x2]
        roi[:] = cv2.LUT(roi, lut)

    def rrect(self, x1, y1, x2, y2, r, fill=None, alpha=1.0, border=None, bw=1, balpha=1.0):
        x1, y1 = int(x1), int(y1)
        w, h = int(x2) - x1, int(y2) - y1
        if w < 2 or h < 2:
            return
        r = int(min(r, w // 2, h // 2))
        if fill is not None:
            if alpha >= 0.999 and r > 0 and w > 2 * r and h > 2 * r and x1 >= 0 and y1 >= 0 \
                    and x1 + w <= self.L.W and y1 + h <= self.L.H:
                c = self.canvas
                c[y1 + r:y1 + h - r, x1:x1 + w] = fill           # middle band, full width
                c[y1:y1 + r, x1 + r:x1 + w - r] = fill           # top strip
                c[y1 + h - r:y1 + h, x1 + r:x1 + w - r] = fill   # bottom strip
                m = _rr_mask(w, h, r)                            # only the four r x r corners need blending
                for (cx0, cy0, mx0, my0) in ((x1, y1, 0, 0), (x1 + w - r, y1, w - r, 0),
                                             (x1, y1 + h - r, 0, h - r), (x1 + w - r, y1 + h - r, w - r, h - r)):
                    reg = c[cy0:cy0 + r, cx0:cx0 + r]
                    mk = m[my0:my0 + r, mx0:mx0 + r]
                    reg[:] = (reg * (1 - mk) + np.array(fill, np.float32) * mk).astype(np.uint8)
            else:
                self._blend(x1, y1, x1 + w, y1 + h, _rr_mask(w, h, r), fill, alpha)
        if border is not None:
            t = max(1, int(round(bw)))
            c, col, ls = self.canvas, border, cv2.LINE_AA
            a, b = x1 + t // 2, y1 + t // 2
            e, f = x1 + w - 1 - t // 2, y1 + h - 1 - t // 2
            if r <= t:
                cv2.rectangle(c, (a, b), (e, f), col, t, ls)
            else:
                rr = r - t // 2
                cv2.line(c, (a + rr, b), (e - rr, b), col, t, ls)
                cv2.line(c, (a + rr, f), (e - rr, f), col, t, ls)
                cv2.line(c, (a, b + rr), (a, f - rr), col, t, ls)
                cv2.line(c, (e, b + rr), (e, f - rr), col, t, ls)
                for (cx, cy, st) in ((a + rr, b + rr, 180), (e - rr, b + rr, 270), (e - rr, f - rr, 0), (a + rr, f - rr, 90)):
                    cv2.ellipse(c, (cx, cy), (rr, rr), 0, st, st + 90, col, t, ls)

    def bar(self, x, y, w, h, frac, color, track=LINE):
        self.rrect(x, y, x + w, y + h, h / 2, fill=track)
        frac = max(0.0, min(1.0, frac))
        if frac > 0:
            self.rrect(x, y, x + max(h, w * frac), y + h, h / 2, fill=color)

    def chip(self, x, y, label, size=11, bg=LINE, fg=TEXT, bold=True, padx=9, h=22, align="l", border=None, alpha=1.0) -> int:
        u = self.L.u
        tw = self.twidth(label, size, bold)
        w = tw + 2 * u(padx)
        x0 = x if align == "l" else x - w
        self.rrect(x0, y, x0 + w, y + u(h), u(h) / 2, fill=bg, alpha=alpha, border=border, bw=1)
        self.text(x0 + w / 2, y + u(h) / 2, label, size, fg, bold, "c")
        return w

    def dot(self, x, y, r, color):
        cv2.circle(self.canvas, (int(x), int(y)), max(1, int(r)), color, -1, cv2.LINE_AA)

    def panel(self, rect, r=12, fill=PANEL, border=LINE):
        x, y, w, h = rect
        self.rrect(x, y, x + w, y + h, self.L.u(r), fill=fill, border=border, bw=1)

    # ---------------------------------------------------------------- render
    def render(self, sc: Scene) -> np.ndarray:
        L = self.L
        self.canvas[:] = BG
        self._header(sc)
        self._video(sc)
        self._zoom(sc)
        self._telemetry(sc)
        self._log(sc)
        return self.canvas

    # ----------------------------------------------------------------- header
    def _header(self, sc: Scene):
        L, u = self.L, self.L.u
        x1, y1, x2, y2 = L.rect(*L.HEADER)
        cy = (y1 + y2) // 2
        self.rrect(x1, y1 + u(4), x1 + u(36), y1 + u(40), u(9), fill=BLUE)
        self.text(x1 + u(18), cy, "M", 18, INK, True, "c")
        self.text(x1 + u(50), cy - u(8), "MHVT  Mission Control", 17, TEXT, True)
        self.text(x1 + u(50), cy + u(11), "Multi-Drone Handover System for Continuous Vehicle Tracking", 11.5, DIM)
        # stepper
        steps = [("zones", "Zones"), ("select", "Select"), ("track", "Track")]
        order = {"zones": 0, "select": 1, "track": 2}
        cur = order[sc.stage]
        sx = x1 + u(560)
        for i, (_, name) in enumerate(steps):
            active, done = i == cur, i < cur
            col = CYAN if active else GREEN if done else FAINT
            self.dot(sx + u(11), cy, u(11), col if (active or done) else LINE)
            self.text(sx + u(11), cy, str(i + 1), 12, INK if (active or done) else DIM, True, "c")
            w = self.text(sx + u(28), cy, name, 14, TEXT if active else DIM, active)
            sx += u(28) + w + u(14)
            if i < 2:
                cv2.line(self.canvas, (sx, cy), (sx + u(34), cy), LINE if not done else GREEN, max(1, u(2)), cv2.LINE_AA)
                sx += u(48)
        # right badges
        rx = x2
        mode = "REPLAY" if sc.replaying and sc.stage == "track" else "LIVE"
        rx -= self.chip(rx, cy - u(11), mode, 11, bg=BLUE if mode == "REPLAY" else LINE, fg=INK if mode == "REPLAY" else TEXT, align="r") + u(8)
        mm, ss = divmod(sc.t, 60)
        rx -= self.chip(rx, cy - u(11), f"{int(mm):02d}:{ss:04.1f}", 11, bg=CARD, fg=TEXT, align="r", border=LINE) + u(8)
        name = sc.video_name if len(sc.video_name) < 34 else sc.video_name[:31] + "…"
        self.chip(rx, cy - u(11), name, 11, bg=CARD, fg=DIM, bold=False, align="r", border=LINE)

    # ------------------------------------------------------------------ video
    def _video(self, sc: Scene):
        L, u = self.L, self.L.u
        x1, y1, x2, y2 = L.video_px()
        k = L.video_scale()
        self.canvas[y1:y2, x1:x2] = cv2.resize(sc.work, (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)
        P = lambda p: (int(x1 + p[0] * k), int(y1 + p[1] * k))  # noqa: E731
        for (x, y, w, h) in sc.detections:
            cv2.rectangle(self.canvas, P((x, y)), P((x + w, y + h)), (210, 210, 210), 1, cv2.LINE_AA)
        track = sc.stage == "track"
        coord = sc.coord if track else None
        # zones
        for i, z in enumerate(sc.zones):
            c = ZONE_COLORS[i % len(ZONE_COLORS)]
            a, b = P(z[:2]), P(z[2:])
            a = (max(a[0], x1 + 1), max(a[1], y1 + 1))
            b = (min(b[0], x2 - 2), min(b[1], y2 - 2))
            primary = coord is not None and coord.owner == i + 1
            acquiring = coord is not None and coord.drone(i + 1).state == DState.ACQUIRING
            offline = coord is not None and coord.drone(i + 1).state == DState.OFFLINE
            if offline:
                c = FAINT
            self.rect_alpha(a[0], a[1], b[0], b[1], c, 0.14 if primary else 0.06)
            thick = u(3.5) if primary else u(3) if (acquiring and int(sc.t * 4) % 2 == 0) else u(1.5)
            cv2.rectangle(self.canvas, a, b, c, max(1, thick), cv2.LINE_AA)
            label = f"UAV-{i + 1}" + ("  PRIMARY" if primary else "  OFFLINE" if offline else "")
            self.chip(a[0] + u(8), a[1] + u(48) + (i % 2) * u(28), label, 11, bg=c, fg=INK, alpha=0.95)
        if sc.drag and sc.stage == "zones":
            a, b = P(sc.drag[:2]), P(sc.drag[2:])
            cv2.rectangle(self.canvas, a, b, TEXT, max(1, u(1.5)), cv2.LINE_AA)
        elif sc.drag:
            a, b = P(sc.drag[:2]), P(sc.drag[2:])
            cv2.rectangle(self.canvas, a, b, GREEN, max(1, u(2)), cv2.LINE_AA)
        if track:
            self._target(sc, P)
        # status chip + instruction bar
        if track:
            if sc.ended:
                st, col = "END OF VIDEO", AMBER
            elif sc.lost:
                st, col = "TARGET LOST", RED
            elif sc.paused:
                st, col = "PAUSED", AMBER
            else:
                st, col = "TRACKING", GREEN
        else:
            st, col = ("SETUP" if sc.stage == "zones" else "SELECT TARGET"), CYAN
        w = self.twidth(st, 12, True) + u(34)
        self.rrect(x1 + u(10), y1 + u(10), x1 + u(10) + w, y1 + u(36), u(13), fill=INK, alpha=0.78)
        self.dot(x1 + u(24), y1 + u(23), u(5), col)
        self.text(x1 + u(35), y1 + u(23), st, 12, col, True)
        if track:
            own = f"UAV-{sc.coord.owner}" if sc.coord.owner else "no UAV"
            hint = {"TARGET LOST": "Tracker lost the vehicle - T restarts the run, R picks another",
                    "END OF VIDEO": "T restarts the run, K saves the scenario, E writes the report",
                    "PAUSED": "Paused - press Space to resume"}.get(st, f"Following target #1 - primary owner {own}")
        else:
            hint = INSTRUCTION[sc.stage]
        bar_h = u(34)
        self.rrect(x1, y2 - bar_h, x2, y2, 0, fill=INK, alpha=0.72)
        self.text(x1 + u(14), y2 - bar_h / 2, hint, 12.5, TEXT)
        self.text(x2 - u(14), y2 - bar_h / 2, f"frame {sc.frame_no} / {sc.nframes}", 11.5, DIM, anchor="r", )
        if sc.toast:
            tw = self.twidth(sc.toast, 12.5, True) + u(28)
            cx = (x1 + x2) // 2
            self.rrect(cx - tw // 2, y2 - bar_h - u(44), cx + tw // 2, y2 - bar_h - u(12), u(16), fill=CARD, alpha=0.95, border=CYAN)
            self.text(cx, y2 - bar_h - u(28), sc.toast, 12.5, TEXT, True, "c")
        self.rrect(x1, y1, x2, y2, u(12), border=LINE, bw=1)

    def _target(self, sc: Scene, P):
        u = self.L.u
        pts = [P(p) for p in sc.trail]
        n = len(pts)
        for i in range(1, n):
            f = i / n
            col = (int(40 + 20 * f), int(110 + 110 * f), int(50 + 70 * f))
            cv2.line(self.canvas, pts[i - 1], pts[i], col, max(1, int(u(1) + u(2.5) * f)), cv2.LINE_AA)
        if sc.bbox is None or sc.lost:
            return
        x, y, w, h = sc.bbox
        pad = 5
        a, b = P((x - pad, y - pad)), P((x + w + pad, y + h + pad))
        cv2.rectangle(self.canvas, a, b, GREEN, max(1, u(1.5)), cv2.LINE_AA)
        arm = max(u(10), min(u(18), (b[0] - a[0]) // 3))
        t = max(2, u(3.5))
        for (px, py), (dx, dy) in (((a[0], a[1]), (1, 1)), ((b[0], a[1]), (-1, 1)), ((a[0], b[1]), (1, -1)), ((b[0], b[1]), (-1, -1))):
            cv2.line(self.canvas, (px, py), (px + arm * dx, py), GREEN, t, cv2.LINE_AA)
            cv2.line(self.canvas, (px, py), (px, py + arm * dy), GREEN, t, cv2.LINE_AA)
        lab = f"TARGET #1   {sc.score:.2f}"
        wl = self.twidth(lab, 11.5, True) + u(18)
        ly = a[1] - u(26) if a[1] - u(26) > self.L.video_px()[1] + u(44) else b[1] + u(6)
        self.rrect(a[0], ly, a[0] + wl, ly + u(22), u(6), fill=GREEN)
        self.text(a[0] + wl / 2, ly + u(11), lab, 11.5, INK, True, "c")
        # handover arrow
        h_ = sc.coord.handover
        if h_ and len(sc.zones) >= max(h_.source, h_.successor):
            zc = lambda i: P(((sc.zones[i - 1][0] + sc.zones[i - 1][2]) / 2, (sc.zones[i - 1][1] + sc.zones[i - 1][3]) / 2))  # noqa: E731
            p1, p2 = zc(h_.source), zc(h_.successor)
            cv2.arrowedLine(self.canvas, p1, p2, AMBER, max(2, u(3)), cv2.LINE_AA, tipLength=0.07)
            lab = f"HANDOVER #{h_.hid}   UAV-{h_.source} -> UAV-{h_.successor}   {h_.phase.replace('_', ' ')}"
            wl = self.twidth(lab, 12, True) + u(24)
            mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - u(20)
            self.rrect(mx - wl // 2, my - u(13), mx + wl // 2, my + u(13), u(13), fill=INK, alpha=0.88, border=AMBER)
            self.text(mx, my, lab, 12, AMBER, True, "c")

    # ------------------------------------------------------------------- zoom
    def _zoom(self, sc: Scene):
        L, u = self.L, self.L.u
        x1, y1, x2, y2 = L.rect(*L.ZOOM)
        pw, ph = x2 - x1, y2 - y1
        k = sc.full.shape[1] / WORK_W                      # full-res px per working px
        zx, zy = Layout.ZOOM[2], Layout.ZOOM[3]            # panel size in design units
        cw, ch = zx / sc.zoom * k, zy / sc.zoom * k        # crop size (full-res px): zoom x the working frame
        cw, ch = min(cw, sc.full.shape[1]), min(ch, sc.full.shape[0])
        cx, cy = sc.zoom_center[0] * k, sc.zoom_center[1] * k
        fx0 = int(min(max(cx - cw / 2, 0), max(0, sc.full.shape[1] - cw)))
        fy0 = int(min(max(cy - ch / 2, 0), max(0, sc.full.shape[0] - ch)))
        crop = sc.full[fy0:fy0 + int(ch), fx0:fx0 + int(cw)]
        view = cv2.resize(crop, (pw, ph), interpolation=cv2.INTER_CUBIC)
        active = sc.stage == "track" and not sc.lost
        if not active:
            view = (view * 0.30).astype(np.uint8)
        self.canvas[y1:y2, x1:x2] = view
        col = GREEN if active else RED if sc.stage == "track" else FAINT
        m, arm, t = u(14), u(26), max(2, u(2.5))
        for (px, py), (dx, dy) in (((x1 + m, y1 + m), (1, 1)), ((x2 - m, y1 + m), (-1, 1)), ((x1 + m, y2 - m), (1, -1)), ((x2 - m, y2 - m), (-1, -1))):
            cv2.line(self.canvas, (px, py), (px + arm * dx, py), col, t, cv2.LINE_AA)
            cv2.line(self.canvas, (px, py), (px, py + arm * dy), col, t, cv2.LINE_AA)
        if active:
            tx = int(x1 + (cx - fx0) / cw * pw)
            ty = int(y1 + (cy - fy0) / ch * ph)
            cv2.drawMarker(self.canvas, (tx, ty), GREEN, cv2.MARKER_CROSS, u(22), max(1, u(1.5)), cv2.LINE_AA)
        owner = sc.coord.owner
        name = f"UAV-{owner} CAMERA" if owner else "UAV CAMERA"
        st = f"{sc.zoom:.0f}x   " + ("LOCKED" if active else "NO LOCK" if sc.stage == "track" else "STANDBY")
        w = self.twidth(f"{name}   {st}", 12, True) + u(30)
        self.rrect(x1 + u(10), y1 + u(10), x1 + u(10) + w, y1 + u(36), u(13), fill=INK, alpha=0.8)
        self.dot(x1 + u(24), y1 + u(23), u(5), col)
        self.text(x1 + u(35), y1 + u(23), f"{name}   {st}", 12, col if not active else TEXT, True)
        if not active:
            msg = "TARGET LOST" if sc.stage == "track" else "Pick a vehicle to start the camera feed"
            self.text((x1 + x2) // 2, (y1 + y2) // 2 - u(6), msg, 20 if sc.stage == "track" else 14, RED if sc.stage == "track" else DIM, True, "c")
            if sc.stage == "track":
                self.text((x1 + x2) // 2, (y1 + y2) // 2 + u(22), "T restart run   R reselect", 12.5, DIM, False, "c")
        self.rrect(x1, y1, x2, y2, u(12), border=LINE, bw=1)

    # -------------------------------------------------------------- telemetry
    def _telemetry(self, sc: Scene):
        L, u = self.L, self.L.u
        co = sc.coord
        x1, y1, x2, y2 = L.rect(*L.TEL)
        self.rrect(x1, y1, x2, y2, u(12), fill=PANEL, border=LINE)
        pad = u(14)
        ix1, ix2 = x1 + pad, x2 - pad
        self.text(ix1, y1 + u(22), "TELEMETRY", 12.5, DIM, True)
        self.chip(ix2, y1 + u(11), "SIMULATED", 10, bg=PANEL, fg=AMBER, align="r", h=20, border=AMBER)
        # stat tiles
        status = "IDLE" if sc.stage != "track" else ("LOST" if sc.lost else "TRACKING")
        s = co.summary()
        rate = "-" if s["success_rate"] is None else f"{100 * s['success_rate']:.0f}% success"
        tiles = [
            ("TARGET", status, {"TRACKING": GREEN, "LOST": RED}.get(status, DIM), f"score {sc.score:.2f}" if status == "TRACKING" else " "),
            ("PRIMARY OWNER", f"UAV-{co.owner}" if co.owner else "none", TEXT if co.owner else DIM, "single owner" if co.owner else " "),
            ("HANDOVERS", f"{s['handovers_ok']} ok / {s['handovers_failed']} failed", TEXT, rate),
        ]
        gap = u(8)
        tw = (ix2 - ix1 - 2 * gap) // 3
        ty = y1 + u(40)
        th = u(66)
        for i, (lab, val, col, sub) in enumerate(tiles):
            tx = ix1 + i * (tw + gap)
            self.rrect(tx, ty, tx + tw, ty + th, u(9), fill=CARD)
            self.text(tx + u(10), ty + u(14), lab, 9.5, DIM, True)
            self.text(tx + u(10), ty + u(35), val, 13.5 if len(val) < 12 else 11.5, col, True, maxw=tw - u(16))
            self.text(tx + u(10), ty + u(54), sub, 10.5, DIM, maxw=tw - u(16))
        # link row
        sent, lost = s["msgs_sent"], s["msgs_lost"]
        dr = "-" if not sent else f"{100 * (1 - lost / sent):.0f}%"
        ly = ty + th + u(16)
        self.text(ix1, ly, f"Message loss {int(100 * co.cfg.loss_prob)}%", 11.5, AMBER if co.cfg.loss_prob else DIM, bool(co.cfg.loss_prob))
        self.text(ix2, ly, f"delivered {dr}   dropped {lost}/{sent}   aborted {co.aborted}", 11.5, DIM, anchor="r")
        # coverage strip
        cy = ly + u(26)
        self.text(ix1, cy, "COVERAGE", 9.5, DIM, True)
        sy, sh = cy + u(10), u(20)
        self.rrect(ix1, sy, ix2, sy + sh, u(6), fill=CARD)
        sw = ix2 - ix1
        for i, z in enumerate(sc.zones):
            c = ZONE_COLORS[i % len(ZONE_COLORS)]
            zx1, zx2 = ix1 + int(sw * z[0] / WORK_W), ix1 + int(sw * z[2] / WORK_W)
            primary = co.owner == i + 1
            self.rrect(zx1 + 1, sy + u(3) + (i % 2) * u(7), max(zx1 + 3, zx2 - 1), sy + u(3) + (i % 2) * u(7) + u(6), u(3), fill=c, alpha=1.0 if primary else 0.45)
        if sc.stage == "track" and not sc.lost and sc.bbox:
            tx = ix1 + int(sw * (sc.bbox[0] + sc.bbox[2] / 2) / WORK_W)
            self.dot(tx, sy + sh / 2, u(5.5), GREEN)
            cv2.circle(self.canvas, (tx, int(sy + sh / 2)), u(7.5), INK, max(1, u(1.5)), cv2.LINE_AA)
        # drone cards
        top = sy + sh + u(14)
        n = max(1, len(co.drones))
        avail = y2 - u(12) - top
        g = u(8)
        ch = (avail - g * (n - 1)) // n
        ch = min(ch, u(92))
        rich = ch >= u(70)
        ranking = {r["id"]: r for r in co.ranking_view}
        for i, d in enumerate(co.drones):
            cy0 = top + i * (ch + g)
            c = ZONE_COLORS[i % len(ZONE_COLORS)]
            primary = co.owner == d.id
            self.rrect(ix1, cy0, ix2, cy0 + ch, u(10), fill=CARD, border=GREEN if primary else None, bw=max(1, u(1.6)))
            self.rrect(ix1 + u(4), cy0 + u(8), ix1 + u(8), cy0 + ch - u(8), u(2), fill=c)
            self.text(ix1 + u(20), cy0 + u(17), d.name, 13, TEXT, True)
            if primary:
                self.text(ix1 + u(20) + self.twidth(d.name, 13, True) + u(8), cy0 + u(17), "PRIMARY", 9.5, GREEN, True)
            sc_col = STATE_COLORS.get(d.state, DIM)
            self.chip(ix2 - u(10), cy0 + u(7), d.state.value, 10, bg=sc_col, fg=INK, align="r", h=20)
            by = cy0 + u(38) if ch > u(52) else cy0 + u(33)
            half = (ix2 - ix1 - u(34)) // 2
            bx = ix1 + u(20)
            bat_c = GREEN if d.battery >= 50 else AMBER if d.battery >= co.cfg.battery_min else RED
            self.text(bx, by, "BAT", 9.5, DIM, True)
            self.bar(bx + u(30), by - u(4), half - u(88), u(8), d.battery / 100, bat_c)
            self.text(bx + half - u(8), by, f"{d.battery:.0f}%", 11, TEXT, anchor="r")
            lx = bx + half + u(10)
            fresh = 1 - min(1.0, d.link_age / co.cfg.offline_timeout)
            lk_c = GREEN if d.link_age <= co.cfg.link_age_max else RED
            self.text(lx, by, "LINK", 9.5, DIM, True)
            self.bar(lx + u(34), by - u(4), half - u(92), u(8), fresh, lk_c)
            self.text(lx + half - u(8), by, f"{d.link_age:.1f}s", 11, TEXT, anchor="r")
            if rich:
                ry = cy0 + u(60)
                self.text(bx, ry, "SCORE", 9.5, DIM, True)
                r = ranking.get(d.id)
                self.bar(bx + u(46), ry - u(4), half - u(100), u(8), r["score"] if r else 0, BLUE)
                self.text(bx + half - u(8), ry, f"{r['score']:.2f}" if r else "-", 11, TEXT if r else FAINT, anchor="r")
                if d.silent:
                    self.text(lx, ry, "FAULT: telemetry cut", 10.5, RED, True)
                elif d.battery < co.cfg.battery_min:
                    self.text(lx, ry, "LOW BATTERY", 10.5, RED, True)
                elif r:
                    self.text(lx, ry, "covers target", 10.5, DIM)

    # -------------------------------------------------------------------- log
    def _log(self, sc: Scene):
        L, u = self.L, self.L.u
        x1, y1, x2, y2 = L.rect(*L.LOG)
        self.rrect(x1, y1, x2, y2, u(12), fill=PANEL, border=LINE)
        pad = u(16)
        self.text(x1 + pad, y1 + u(22), "COORDINATOR LOG", 12.5, DIM, True)
        events = sc.coord.events
        self.text(x2 - pad, y1 + u(22), f"{len(events)} events", 11, FAINT, anchor="r")
        foot = u(62)
        lh = u(19)
        top = y1 + u(44)
        lines = max(1, (y2 - foot - top) // lh)
        colors = {"info": TEXT, "ok": GREEN, "warn": AMBER, "err": RED}
        for i, e in enumerate(events[-lines:]):
            yy = top + i * lh + lh // 2
            self.text(x1 + pad + u(48), yy, f"{e.t:6.2f}s", 11, FAINT, anchor="r")
            ind = e.text.startswith("  ")
            self.dot(x1 + pad + u(64) + (u(10) if ind else 0), yy, u(3), colors.get(e.level, TEXT))
            self.text(x1 + pad + u(76) + (u(10) if ind else 0), yy, e.text.strip(), 11.5, colors.get(e.level, TEXT) if e.level != "info" else TEXT,
                      maxw=x2 - x1 - pad * 2 - u(86))
        # key hints
        cv2.line(self.canvas, (x1 + pad, y2 - foot + u(4)), (x2 - pad, y2 - foot + u(4)), LINE, 1, cv2.LINE_AA)
        hx0, hy = x1 + pad, y2 - foot + u(20)
        for key, label in HINTS[sc.stage]:
            kw = self.twidth(key, 10.5, True) + u(14)
            lw = self.twidth(label, 11, False)
            if hx0 + kw + lw + u(22) > x2 - pad:
                hx0, hy = x1 + pad, hy + u(26)
            self.rrect(hx0, hy - u(10), hx0 + kw, hy + u(10), u(5), fill=CARD, border=LINE)
            self.text(hx0 + kw / 2, hy, key, 10.5, TEXT, True, "c")
            self.text(hx0 + kw + u(6), hy, label, 11, DIM)
            hx0 += kw + lw + u(22)
