"""
Sliding-window coordinate stabilizer.

Collects consecutive 3D positions in a sliding window. Only outputs the
window average once the window is full AND the variance is below threshold.
"""

from collections import deque

import numpy as np


class CoordinateStabilizer:
    """
    Sliding-window 3D coordinate stabilizer.

    Adds coordinates to a fixed-size window. When the window is full and
    the position variance is below max_variance_m2, returns the window
    average. Otherwise returns None (not stable).

    Usage:
        stabilizer = CoordinateStabilizer(window_size=5, max_variance_m2=0.0004)
        stable = stabilizer.update((0.1, 0.2, 0.5))
        if stable is not None:
            publish(stable)
    """

    def __init__(
        self,
        window_size: int = 7,
        max_variance_m2: float = 0.0002,
        output_mode: str = 'median',
    ):
        """
        Args:
            window_size: Number of consecutive positions to collect.
            max_variance_m2: Maximum acceptable position variance (m²).
                             Default 0.0004 ≈ 0.02m standard deviation.
        """
        self._window_size = max(window_size, 2)
        self._max_variance_m2 = float(max_variance_m2)
        self._output_mode = output_mode.strip().lower()
        if self._output_mode not in ('median', 'mean'):
            self._output_mode = 'median'
        self._buffer: deque[tuple[float, float, float]] = deque(maxlen=self._window_size)

    def update(
        self, pos: tuple[float, float, float]
    ) -> tuple[float, float, float] | None:
        """
        Add a new position and return stabilized value if ready.

        Returns:
            (x, y, z) averaged over the window if window is full and stable,
            otherwise None.
        """
        self._buffer.append(pos)
        if len(self._buffer) < self._window_size:
            return None

        arr = np.array(self._buffer, dtype=np.float64)
        # Per-axis temporal variance — pool X/Y/Z independently,
        # otherwise cross-axis spread dominates temporal jitter.
        variance = float(np.max(np.var(arr, axis=0)))
        if variance > self._max_variance_m2:
            return None

        if self._output_mode == 'mean':
            output = np.mean(arr, axis=0)
        else:
            output = np.median(arr, axis=0)
        return (float(output[0]), float(output[1]), float(output[2]))

    @property
    def is_ready(self) -> bool:
        """Whether the window has been filled at least once."""
        return len(self._buffer) >= self._window_size

    @property
    def window_fill(self) -> int:
        """Number of positions currently in the window."""
        return len(self._buffer)

    def reset(self) -> None:
        """Clear the buffer (e.g. when switching targets)."""
        self._buffer.clear()
