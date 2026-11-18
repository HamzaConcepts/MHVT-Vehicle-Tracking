"""Global camera motion between frames (translation only), so a hidden vehicle can be predicted in the
*scene* instead of in the picture: when the drone speeds up or slows down, a straight line in picture
coordinates is wrong, but a vehicle's motion over the ground stays smooth.

Uses phase correlation on a small grey copy of the frame (about 1-2 ms)."""

from __future__ import annotations

import cv2
import numpy as np

SMALL = (320, 180)


class CameraMotion:
    def __init__(self, work_size=(1280, 720)):
        self.scale = (work_size[0] / SMALL[0], work_size[1] / SMALL[1])
        self.window = cv2.createHanningWindow(SMALL, cv2.CV_32F)
        self.prev: np.ndarray | None = None
        self.cum = (0.0, 0.0)          # accumulated shift of the scene in the picture, working px
        self.last_response = 0.0

    @staticmethod
    def prepare(bgr: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(cv2.resize(bgr, SMALL, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        return g.astype(np.float32)

    def reset(self, small_gray: np.ndarray) -> None:
        self.prev = small_gray
        self.cum = (0.0, 0.0)
        self.last_response = 1.0

    def update(self, small_gray: np.ndarray) -> tuple[float, float]:
        """Returns the accumulated camera shift (working px). A scene point that was at p is now at p + cum."""
        if self.prev is None:
            self.reset(small_gray)
            return self.cum
        (dx, dy), resp = cv2.phaseCorrelate(self.prev, small_gray, self.window)
        self.last_response = float(resp)
        if resp > 0.05 and abs(dx) < SMALL[0] / 2 and abs(dy) < SMALL[1] / 2:  # ignore failed matches
            self.cum = (self.cum[0] + dx * self.scale[0], self.cum[1] + dy * self.scale[1])
        self.prev = small_gray
        return self.cum
