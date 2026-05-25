"""EMA smoothing and temporal stability filter for detection targets."""

import math


def distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


class EMAFilter:

    def __init__(self, alpha: float = 0.25):
        self.alpha = min(max(alpha, 0.0), 1.0)
        self._value: tuple[float, float, float] | None = None

    def update(self, raw: tuple[float, float, float]) -> tuple[float, float, float]:
        if self._value is None:
            self._value = raw
        else:
            a = self.alpha
            self._value = tuple(
                a * raw[i] + (1.0 - a) * self._value[i] for i in range(3)
            )
        return self._value

    def reset(self):
        self._value = None


class StabilityFilter:

    def __init__(self, radius_m: float = 0.02, required_frames: int = 3):
        self.radius_m = radius_m
        self.required_frames = max(required_frames, 1)
        self._candidate: tuple[float, float, float] | None = None
        self._count = 0

    def update(self, target: tuple[float, float, float]) -> tuple[float, float, float] | None:
        if self._candidate is None:
            self._candidate = target
            self._count = 1
        elif distance(target, self._candidate) <= self.radius_m:
            self._candidate = tuple(
                0.5 * self._candidate[i] + 0.5 * target[i] for i in range(3)
            )
            self._count += 1
        else:
            self._candidate = target
            self._count = 1
            return None

        if self._count >= self.required_frames:
            return self._candidate
        return None

    def reset(self):
        self._candidate = None
        self._count = 0
