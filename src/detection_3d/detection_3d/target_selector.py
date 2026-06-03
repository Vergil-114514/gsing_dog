"""
Target selection and coordinate jump detection.

Selects the best detection based on confidence and depth quality,
and rejects detections that jump too far from the previous frame.
"""

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class TargetCandidate:
    """A scored detection candidate with 3D position and metadata."""

    x: float
    y: float
    z: float
    class_name: str
    confidence: float
    depth_quality: float
    composite_score: float
    source: Any = None  # reference to original Detection2D/Detection3D for downstream use


def compute_composite_score(confidence: float, depth_quality: float) -> float:
    """Weighted composite: 0.6 × confidence + 0.4 × depth_quality."""
    return 0.6 * confidence + 0.4 * depth_quality


def select_best_target(candidates: list[TargetCandidate]) -> TargetCandidate | None:
    """
    Select the single best target from scored candidates.

    Returns the candidate with highest composite_score, or None if list is empty.
    """
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.composite_score)


def detect_coordinate_jump(
    new_pos: tuple[float, float, float],
    prev_pos: tuple[float, float, float] | None,
    threshold_m: float,
) -> bool:
    """
    Check if new 3D position has jumped too far from previous.

    Args:
        new_pos: New (x, y, z) in meters.
        prev_pos: Previous (x, y, z) or None (first frame).
        threshold_m: Maximum allowed Euclidean distance between frames.

    Returns:
        True if the jump exceeds threshold (should be rejected).
    """
    if prev_pos is None:
        return False
    dist = math.sqrt(
        (new_pos[0] - prev_pos[0]) ** 2
        + (new_pos[1] - prev_pos[1]) ** 2
        + (new_pos[2] - prev_pos[2]) ** 2
    )
    return dist > threshold_m
