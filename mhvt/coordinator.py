"""Simulated multi-UAV coordinator: zones, successor ranking and handover protocol.

Pure Python (no OpenCV) so it can be unit-tested and reused by a SUMO-based
simulation later. Time is simulation time in seconds, supplied by the caller.

Implements the report's design:
  * HandoverScore = w1*(1-d) + w2*b + w3*(1-age)            (Fig. 3.5)
  * REQ -> ACK -> ACQUIRED -> COMMIT with timeouts/retry/fallback (Figs. 3.2, 3.10, 3.11)
  * exactly one primary owner at any time (FR4 / NFR4)
  * a drone silent for ~3 s is declared unavailable (NFR5)
Battery and link freshness are *simulated*; the target position comes from the
real video tracker.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum

Rect = tuple[float, float, float, float]  # x1, y1, x2, y2


class DState(str, Enum):
    STANDBY = "STANDBY"
    TRACKING = "TRACKING"
    HANDING_OVER = "HANDING OVER"
    ACQUIRING = "ACQUIRING"
    SEARCHING = "SEARCHING"
    OFFLINE = "OFFLINE"


@dataclass
class Config:
    w_distance: float = 0.5
    w_battery: float = 0.3
    w_link: float = 0.2
    battery_min: float = 20.0       # % below which a drone is not eligible
    link_age_max: float = 2.0       # s; older beacon => not eligible
    offline_timeout: float = 3.0    # s of silence => declared OFFLINE (NFR5)
    beacon_interval: float = 0.5
    edge_margin: float = 0.18       # fraction of zone width that counts as "near edge"
    ack_timeout: float = 0.8
    max_retries: int = 1
    acquire_time: float = 0.4       # target must be inside successor zone this long
    handover_timeout: float = 4.0
    cooldown: float = 1.0
    latency_mean: float = 0.12
    latency_jitter: float = 0.04
    loss_prob: float = 0.0
    drain_tracking: float = 0.30    # %/s (exaggerated so it is visible in a short demo)
    drain_idle: float = 0.05
    seed: int = 7


@dataclass
class Drone:
    id: int
    zone: Rect
    battery: float
    state: DState = DState.STANDBY
    link_age: float = 0.0
    silent: bool = False            # injected fault: stops sending beacons
    score: float | None = None      # last HandoverScore computed for it

    @property
    def name(self) -> str:
        return f"UAV-{self.id}"


@dataclass
class Event:
    t: float
    text: str
    level: str = "info"  # info | ok | warn | err


@dataclass
class Handover:
    hid: int
    source: int
    reason: str
    ranking: list[dict]
    idx: int
    t_start: float
    successor: int = 0
    phase: str = "REQ_SENT"          # REQ_SENT | ACKED | ACQ_SENT
    t_phase: float = 0.0
    retries: int = 0
    acquire_since: float | None = None


@dataclass
class _Msg:
    kind: str
    deliver_t: float
    hid: int
    to: int


def contains(zone: Rect, p: tuple[float, float]) -> bool:
    return zone[0] <= p[0] <= zone[2] and zone[1] <= p[1] <= zone[3]


def edge_distance(zone: Rect, p: tuple[float, float]) -> float:
    """Distance from p to the nearest zone edge; negative when outside."""
    return min(p[0] - zone[0], zone[2] - p[0], p[1] - zone[1], zone[3] - p[1])


class Coordinator:
    def __init__(self, zones: list[Rect], config: Config | None = None):
        self.cfg = config or Config()
        self.rng = random.Random(self.cfg.seed)
        self.drones = [
            Drone(i + 1, tuple(z), battery=self.rng.uniform(55, 95))
            for i, z in enumerate(zones)
        ]
        self.owner: int | None = None
        self.handover: Handover | None = None
        self.events: list[Event] = []
        self.target_lost = False
        self.ok = 0
        self.failed = 0
        self.latencies: list[float] = []
        self.msgs_sent = 0
        self.msgs_lost = 0
        self._queue: list[_Msg] = []
        self._hid = 0
        self._t: float | None = None
        self._next_beacon = 0.0
        self._cooldown_until = 0.0
        self._warned_gap_at = -99.0
        self._ranking_view: list[dict] = []

    # ------------------------------------------------------------ helpers
    def drone(self, did: int) -> Drone:
        return self.drones[did - 1]

    def log(self, t: float, text: str, level: str = "info") -> None:
        self.events.append(Event(t, text, level))

    def eligible(self, d: Drone) -> bool:
        return (
            d.state != DState.OFFLINE
            and d.battery >= self.cfg.battery_min
            and d.link_age <= self.cfg.link_age_max
        )

    def rank(self, p: tuple[float, float], exclude: int | None = None) -> list[dict]:
        """Eligible drones that cover p, best HandoverScore first."""
        c = self.cfg
        rows = []
        for d in self.drones:
            if d.id == exclude or not self.eligible(d) or not contains(d.zone, p):
                continue
            zx, zy = (d.zone[0] + d.zone[2]) / 2, (d.zone[1] + d.zone[3]) / 2
            diag = ((d.zone[2] - d.zone[0]) ** 2 + (d.zone[3] - d.zone[1]) ** 2) ** 0.5
            dist = min(1.0, ((p[0] - zx) ** 2 + (p[1] - zy) ** 2) ** 0.5 / (diag / 2))
            b = d.battery / 100.0
            age = min(1.0, d.link_age / c.link_age_max)
            score = c.w_distance * (1 - dist) + c.w_battery * b + c.w_link * (1 - age)
            d.score = score
            rows.append({"id": d.id, "score": score, "d": dist, "b": b, "age": age})
        rows.sort(key=lambda r: -r["score"])
        return rows

    @property
    def ranking_view(self) -> list[dict]:
        return self._ranking_view

    def set_silent(self, did: int, silent: bool, t: float) -> None:
        d = self.drone(did)
        d.silent = silent
        self.log(t, f"FAULT {'injected' if silent else 'cleared'}: {d.name} telemetry "
                    f"{'stopped' if silent else 'restored'}", "warn")

    def set_loss(self, p: float) -> None:
        self.cfg.loss_prob = p

    # ------------------------------------------------------------- target
    def reset_target(self) -> None:
        """Operator selected a (new) target: forget old ownership."""
        if self.owner is not None:
            self.drone(self.owner).state = DState.STANDBY
        if self.handover:
            self._abort_handover(self._t or 0.0, "target re-selected", count=False)
        self.owner = None
        self.target_lost = False
        self._queue.clear()

    def mark_target_lost(self, t: float) -> None:
        if self.target_lost:
            return
        self.target_lost = True
        if self.handover:
            self._abort_handover(t, "target lost", count=True)
        if self.owner is not None:
            self.drone(self.owner).state = DState.SEARCHING
            self.log(t, f"TARGET LOST - {self.drone(self.owner).name} searching", "err")

    # ------------------------------------------------------------- update
    def update(self, t: float, target: tuple[float, float] | None) -> None:
        dt = 0.0 if self._t is None else max(0.0, t - self._t)
        self._t = t
        self._tick_drones(t, dt)
        self._deliver(t, target)
        if target is None or self.target_lost:
            return
        if self.owner is None:
            self._initial_assign(t, target)
            return
        self._check_owner(t, target)
        if self.owner is None:
            return
        if self.handover:
            self._progress(t, target)
        else:
            self._maybe_trigger(t, target)
        self._ranking_view = self.rank(target)

    def _tick_drones(self, t: float, dt: float) -> None:
        c = self.cfg
        beacon = t >= self._next_beacon
        if beacon:
            self._next_beacon = t + c.beacon_interval
        for d in self.drones:
            busy = d.state in (DState.TRACKING, DState.HANDING_OVER, DState.ACQUIRING, DState.SEARCHING)
            d.battery = max(0.0, d.battery - dt * (c.drain_tracking if busy else c.drain_idle))
            d.link_age += dt
            if beacon and not d.silent and self.rng.random() >= c.loss_prob:
                d.link_age = 0.0
            if d.state != DState.OFFLINE and d.link_age > c.offline_timeout:
                d.state = DState.OFFLINE
                self.log(t, f"{d.name} silent {c.offline_timeout:.0f}s -> declared OFFLINE", "err")
            elif d.state == DState.OFFLINE and d.link_age <= c.link_age_max and not d.silent:
                d.state = DState.STANDBY
                self.log(t, f"{d.name} link restored -> STANDBY", "ok")

    def _check_owner(self, t: float, target) -> None:
        o = self.drone(self.owner)
        if o.state == DState.OFFLINE:
            self.log(t, f"Primary {o.name} lost - reassigning target", "err")
            if self.handover:
                self._abort_handover(t, "owner offline", count=True)
            self.owner = None
            self._initial_assign(t, target)

    # --------------------------------------------------------- assignment
    def _initial_assign(self, t: float, p) -> None:
        rows = self.rank(p)
        self._ranking_view = rows
        if not rows:
            if t - self._warned_gap_at > 2.0:
                self._warned_gap_at = t
                self.log(t, "NO ELIGIBLE UAV covers the target", "warn")
            return
        best = self.drone(rows[0]["id"])
        self.owner = best.id
        best.state = DState.TRACKING
        self.log(t, f"ASSIGN target -> {best.name} (score {rows[0]['score']:.2f})", "ok")

    # ------------------------------------------------------------ trigger
    def _maybe_trigger(self, t: float, p) -> None:
        if t < self._cooldown_until:
            return
        o = self.drone(self.owner)
        margin = self.cfg.edge_margin * (o.zone[2] - o.zone[0])
        ed = edge_distance(o.zone, p)
        reason = None
        if ed < 0:
            reason = "OWNER_OUT_OF_ZONE"
        elif ed < margin:
            reason = "ZONE_EDGE"
        elif o.battery < self.cfg.battery_min:
            reason = "LOW_BATTERY"
        if reason is None:
            return
        rows = self.rank(p, exclude=o.id)
        if not rows:
            if t - self._warned_gap_at > 2.0 and reason != "ZONE_EDGE":
                self._warned_gap_at = t
                self.log(t, f"{reason}: no successor covers target (coverage gap)", "warn")
            return
        self._hid += 1
        h = Handover(self._hid, o.id, reason, rows, 0, t_start=t)
        self.handover = h
        o.state = DState.HANDING_OVER
        order = ", ".join(f"UAV-{r['id']}={r['score']:.2f}" for r in rows)
        self.log(t, f"HANDOVER #{h.hid} start ({reason}) ranking: {order}", "info")
        self._request(t, h)

    def _request(self, t: float, h: Handover) -> None:
        h.successor = h.ranking[h.idx]["id"]
        h.phase, h.t_phase = "REQ_SENT", t
        self._send("HANDOVER_REQ", h, h.successor, t)

    def _send(self, kind: str, h: Handover, to: int, t: float) -> None:
        self.msgs_sent += 1
        if self.rng.random() < self.cfg.loss_prob:
            self.msgs_lost += 1
            self.log(t, f"  {kind} -> {self._who(to)} LOST in transit", "warn")
            return
        lat = max(0.02, self.rng.gauss(self.cfg.latency_mean, self.cfg.latency_jitter))
        self._queue.append(_Msg(kind, t + lat, h.hid, to))
        self.log(t, f"  {kind} -> {self._who(to)}")

    def _who(self, did: int) -> str:
        return "COORD" if did == 0 else self.drone(did).name

    # ----------------------------------------------------------- delivery
    def _deliver(self, t: float, target) -> None:
        due = [m for m in self._queue if m.deliver_t <= t]
        self._queue = [m for m in self._queue if m.deliver_t > t]
        h = self.handover
        for m in due:
            if not h or m.hid != h.hid:
                continue  # stale message from an aborted handover
            if m.kind == "HANDOVER_REQ":
                s = self.drone(h.successor)
                if s.silent or s.state == DState.OFFLINE:
                    continue  # no ACK will come
                s.state = DState.ACQUIRING
                self._send("ACK", h, 0, t)
            elif m.kind == "ACK" and h.phase == "REQ_SENT":
                h.phase, h.t_phase = "ACKED", t
                self.log(t, f"  ACK received from {self.drone(h.successor).name}", "info")
            elif m.kind == "ACQUIRED":
                self._commit(t, h)
                h = self.handover

    def _commit(self, t: float, h: Handover) -> None:
        src, dst = self.drone(h.source), self.drone(h.successor)
        src.state = DState.STANDBY
        dst.state = DState.TRACKING
        self.owner = dst.id
        self.ok += 1
        self.latencies.append(t - h.t_start)
        self.handover = None
        self._cooldown_until = t + self.cfg.cooldown
        self.log(t, f"COMMIT: {dst.name} is now the single owner "
                    f"(handover {t - h.t_start:.2f}s)", "ok")

    # ----------------------------------------------------------- progress
    def _progress(self, t: float, p) -> None:
        h, c = self.handover, self.cfg
        if t - h.t_start > c.handover_timeout:
            self._fail(t, "handover timeout")
            return
        s = self.drone(h.successor)
        if h.phase == "REQ_SENT" and t - h.t_phase > c.ack_timeout:
            if h.retries < c.max_retries:
                h.retries += 1
                self.log(t, f"  no ACK from {s.name} - retry {h.retries}", "warn")
                self._request(t, h)
            else:
                self._fallback(t)
            return
        if h.phase in ("ACKED", "ACQ_SENT"):
            if contains(s.zone, p):
                if h.acquire_since is None:
                    h.acquire_since = t
                if t - h.acquire_since >= c.acquire_time and (
                    h.phase == "ACKED" or t - h.t_phase > c.ack_timeout
                ):
                    if h.phase == "ACQ_SENT":
                        self.log(t, "  ACQUIRED not confirmed - resending", "warn")
                    h.phase, h.t_phase = "ACQ_SENT", t
                    self._send("ACQUIRED", h, 0, t)
            else:
                h.acquire_since = None
                if t - h.t_phase > c.handover_timeout / 2 and h.phase == "ACKED":
                    self._fallback(t)

    def _fallback(self, t: float) -> None:
        h = self.handover
        self.drone(h.successor).state = (
            DState.OFFLINE if self.drone(h.successor).state == DState.OFFLINE else DState.STANDBY
        )
        h.idx += 1
        h.retries, h.acquire_since = 0, None
        if h.idx < len(h.ranking):
            self.log(t, f"  fallback to next candidate UAV-{h.ranking[h.idx]['id']}", "warn")
            self._request(t, h)
        else:
            self._fail(t, "no candidate acknowledged")

    def _fail(self, t: float, why: str) -> None:
        self._abort_handover(t, why, count=True)

    def _abort_handover(self, t: float, why: str, count: bool) -> None:
        h = self.handover
        if h is None:
            return
        succ = self.drone(h.successor)
        if succ.state == DState.ACQUIRING:
            succ.state = DState.STANDBY
        src = self.drone(h.source)
        if src.state == DState.HANDING_OVER:
            src.state = DState.SEARCHING if self.target_lost else DState.TRACKING
        self.handover = None
        self._cooldown_until = t + self.cfg.cooldown
        if count:
            self.failed += 1
            self.log(t, f"HANDOVER #{h.hid} FAILED ({why}) - {src.name} keeps the target", "err")

    # ------------------------------------------------------------- stats
    def summary(self) -> dict:
        total = self.ok + self.failed
        return {
            "handovers_ok": self.ok,
            "handovers_failed": self.failed,
            "success_rate": (self.ok / total) if total else None,
            "mean_latency_s": (sum(self.latencies) / len(self.latencies)) if self.latencies else None,
            "msgs_sent": self.msgs_sent,
            "msgs_lost": self.msgs_lost,
        }


def auto_zones(width: float, height: float, n: int, overlap: float = 0.25) -> list[Rect]:
    """n full-height vertical bands that overlap their neighbours."""
    base = width / n
    pad = base * overlap / 2
    return [
        (max(0.0, i * base - pad), 0.0, min(width, (i + 1) * base + pad), height)
        for i in range(n)
    ]
