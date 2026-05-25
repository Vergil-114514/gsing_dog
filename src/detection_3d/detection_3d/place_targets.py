"""Validate and retrieve place targets from a flat coordinate array.

The array format is: [x1, y1, z1, x2, y2, z2, ...]  (unit: m).
"""


def validate_place_targets(arr: list[float]) -> list[tuple[float, float, float]]:
    if len(arr) % 3 != 0:
        raise ValueError(
            f"place_targets_m length must be a multiple of 3, got {len(arr)}"
        )
    targets = []
    for i in range(0, len(arr), 3):
        targets.append((float(arr[i]), float(arr[i + 1]), float(arr[i + 2])))
    return targets


def get_place_target(
    targets: list[tuple[float, float, float]], index: int
) -> tuple[float, float, float]:
    if not targets:
        raise IndexError("place_targets_m is empty")
    if index < 0 or index >= len(targets):
        raise IndexError(
            f"place_target_index {index} out of range [0, {len(targets) - 1}]"
        )
    return targets[index]
