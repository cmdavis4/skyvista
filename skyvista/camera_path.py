"""
Camera choreography for animations: keyframes in model time.

A :class:`CameraPath` holds :class:`CameraKeyframe` objects at chosen model
times and gives the camera's position, aim point and focal length at any
time in between. Keyframes come in two kinds, and one path uses one kind:

* **Absolute**: each keyframe gives a ``location`` and a ``look_at`` point.
* **Relative to a target**: each keyframe gives an azimuth, a horizontal
  distance and a height relative to a target that may move, and the height
  of the aim point above it. The target is a fixed (x, y) or a
  :class:`Track`, e.g. a tracked storm's position over time. The camera then
  follows the storm while its framing changes smoothly.

Interpolation
-------------
Every keyframed quantity is interpolated on its own with a monotone cubic
(Fritsch-Carlson "PCHIP") spline in time. It passes through every keyframe,
its first derivative is continuous (no jerks), and it never overshoots: a
distance that grows from one keyframe to the next grows monotonically in
between. A plain cubic spline would overshoot, e.g. swing the camera past its
end azimuth and back. Before or after the keyframes, values are held at the
first or last keyframe. Azimuths are unwrapped first, so 350 deg -> 10 deg
turns 20 deg, not 340.

A quantity left unset on some keyframes is interpolated through the
keyframes that set it, so e.g. the lens can be keyframed once and held.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

# Focal length used when no keyframe sets one (mm, 36 mm sensor width).
DEFAULT_LENS_MM = 50.0


# =============================================================================
# Monotone cubic interpolation
# =============================================================================


def _pchip_slopes(knot_times: np.ndarray, knot_values: np.ndarray) -> np.ndarray:
    """
    Fritsch-Carlson slopes at the knots of a monotone cubic Hermite spline.

    Interior slopes are the weighted harmonic mean of the neighbouring secant
    slopes when those have the same sign, and 0 at a local extremum (so the
    curve flattens there instead of overshooting). End slopes use the
    one-sided three-point formula, limited the same way.
    """
    n_knots = len(knot_times)
    if n_knots == 1:
        return np.zeros(1)
    intervals = np.diff(knot_times)
    secants = np.diff(knot_values) / intervals
    if n_knots == 2:
        return np.array([secants[0], secants[0]])

    slopes = np.zeros(n_knots)
    # Interior knots: weighted harmonic mean where the secants agree in sign.
    for knot_index in range(1, n_knots - 1):
        left_secant, right_secant = secants[knot_index - 1], secants[knot_index]
        if left_secant * right_secant <= 0.0:
            continue
        left_interval, right_interval = intervals[knot_index - 1], intervals[knot_index]
        weight_left = 2.0 * right_interval + left_interval
        weight_right = right_interval + 2.0 * left_interval
        slopes[knot_index] = (weight_left + weight_right) / (
            weight_left / left_secant + weight_right / right_secant
        )

    # End knots: one-sided estimate, kept from overshooting.
    def end_slope(first_interval, second_interval, first_secant, second_secant):
        slope = (
            (2.0 * first_interval + second_interval) * first_secant
            - first_interval * second_secant
        ) / (first_interval + second_interval)
        if np.sign(slope) != np.sign(first_secant):
            return 0.0
        if np.sign(first_secant) != np.sign(second_secant) and abs(slope) > abs(
            3.0 * first_secant
        ):
            return 3.0 * first_secant
        return slope

    slopes[0] = end_slope(intervals[0], intervals[1], secants[0], secants[1])
    slopes[-1] = end_slope(intervals[-1], intervals[-2], secants[-1], secants[-2])
    return slopes


def monotone_cubic(
    knot_times: Sequence[float], knot_values: Sequence[float], query_times
) -> np.ndarray:
    """
    Evaluate the monotone cubic through (knot_times, knot_values).

    Queries outside the knots take the nearest end value.
    """
    knot_times = np.asarray(knot_times, dtype=float)
    knot_values = np.asarray(knot_values, dtype=float)
    query_times = np.atleast_1d(np.asarray(query_times, dtype=float))
    if len(knot_times) == 1:
        return np.full(query_times.shape, knot_values[0])

    slopes = _pchip_slopes(knot_times, knot_values)
    # Interval index of every query, clamped to the knot range.
    clamped_times = np.clip(query_times, knot_times[0], knot_times[-1])
    interval_index = np.clip(
        np.searchsorted(knot_times, clamped_times, side="right") - 1,
        0,
        len(knot_times) - 2,
    )
    start_time = knot_times[interval_index]
    interval_length = knot_times[interval_index + 1] - start_time
    unit_time = (clamped_times - start_time) / interval_length

    # Cubic Hermite basis on [0, 1].
    h00 = 2 * unit_time**3 - 3 * unit_time**2 + 1
    h10 = unit_time**3 - 2 * unit_time**2 + unit_time
    h01 = -2 * unit_time**3 + 3 * unit_time**2
    h11 = unit_time**3 - unit_time**2
    return (
        h00 * knot_values[interval_index]
        + h10 * interval_length * slopes[interval_index]
        + h01 * knot_values[interval_index + 1]
        + h11 * interval_length * slopes[interval_index + 1]
    )


# =============================================================================
# Moving targets
# =============================================================================


@dataclass
class Track:
    """
    A target's horizontal position over time, optionally smoothed.

    Attributes:
        times: Sample times (s), increasing.
        x, y: Positions at those times (data units, e.g. meters).
        smoothing: Gaussian smoothing timescale (s; the kernel's standard
            deviation). Tracked feature positions jitter between frames as
            the detected core shifts, and a camera that follows them exactly
            shakes. 0 interpolates the samples linearly.

    Smoothing is a Gaussian-weighted mean of the samples around each query
    time (kernel regression). It needs no evenly spaced samples, so tracks
    with gaps or joined segments are fine. Near the ends it averages over the
    available samples only, which holds the position steady instead of
    extrapolating. Beyond the ends the position is held.
    """

    times: Sequence[float]
    x: Sequence[float]
    y: Sequence[float]
    smoothing: float = 0.0

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float)
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        order = np.argsort(self.times)
        self.times, self.x, self.y = self.times[order], self.x[order], self.y[order]

    def position(self, time: float) -> Tuple[float, float]:
        """The (x, y) position at ``time``."""
        clamped_time = float(np.clip(time, self.times[0], self.times[-1]))
        if self.smoothing <= 0.0:
            return (
                float(np.interp(clamped_time, self.times, self.x)),
                float(np.interp(clamped_time, self.times, self.y)),
            )
        weights = np.exp(-0.5 * ((self.times - clamped_time) / self.smoothing) ** 2)
        total_weight = weights.sum()
        return (
            float((weights * self.x).sum() / total_weight),
            float((weights * self.y).sum() / total_weight),
        )


# =============================================================================
# Keyframes and the path
# =============================================================================


@dataclass
class CameraKeyframe:
    """
    The camera at one model time.

    Give either ``location`` and ``look_at`` (absolute), or any of the
    target-relative quantities. Unset quantities are interpolated through
    the keyframes that set them.

    Attributes:
        time: Model time (s).
        location: Absolute camera position (x, y, z).
        look_at: Absolute aim point (x, y, z).
        azimuth_deg: Direction from the target to the camera,
            counterclockwise from +x.
        horizontal_distance: Camera's horizontal distance from the target.
        height: Camera height above z = 0.
        target_height: Height of the aim point, above the target position.
        lens_mm: Focal length (36 mm sensor width).
    """

    time: float
    location: Optional[Tuple[float, float, float]] = None
    look_at: Optional[Tuple[float, float, float]] = None
    azimuth_deg: Optional[float] = None
    horizontal_distance: Optional[float] = None
    height: Optional[float] = None
    target_height: Optional[float] = None
    lens_mm: Optional[float] = None

    @property
    def is_absolute(self) -> bool:
        return self.location is not None or self.look_at is not None


@dataclass
class CameraState:
    """The camera at one time: position, aim point, focal length."""

    location: Tuple[float, float, float]
    look_at: Tuple[float, float, float]
    lens_mm: float


# Target-relative quantities and their defaults when no keyframe sets them.
RELATIVE_QUANTITY_DEFAULTS = {
    "azimuth_deg": -60.0,
    "horizontal_distance": 50000.0,
    "height": 1500.0,
    "target_height": 5000.0,
}


class CameraPath:
    """
    A camera path through keyframes at model times.

    Args:
        keyframes: The keyframes, in any order; all absolute or all relative.
        target: For relative keyframes, a fixed (x, y) or a :class:`Track`.
    """

    def __init__(
        self,
        keyframes: List[CameraKeyframe],
        target: Union[Track, Tuple[float, float], None] = None,
    ):
        if not keyframes:
            raise ValueError("A CameraPath needs at least one keyframe.")
        self.keyframes = sorted(keyframes, key=lambda keyframe: keyframe.time)
        kinds = {keyframe.is_absolute for keyframe in self.keyframes}
        if len(kinds) > 1:
            raise ValueError(
                "Mixing absolute (location/look_at) and target-relative "
                "keyframes in one CameraPath is not supported."
            )
        self.absolute = kinds.pop()
        if self.absolute:
            for keyframe in self.keyframes:
                if keyframe.location is None or keyframe.look_at is None:
                    raise ValueError(
                        f"Absolute keyframe at t = {keyframe.time} needs both "
                        "location and look_at."
                    )
        elif target is None:
            raise ValueError("Target-relative keyframes need a target.")
        self.target = target

    def _interpolate(self, attribute: str, time: float, default: float) -> float:
        """One scalar quantity at ``time``, through the keyframes that set it."""
        keyed = [
            (keyframe.time, getattr(keyframe, attribute))
            for keyframe in self.keyframes
            if getattr(keyframe, attribute) is not None
        ]
        if not keyed:
            return default
        knot_times, knot_values = zip(*keyed, strict=True)
        knot_values = np.asarray(knot_values, dtype=float)
        # Azimuths turn the short way between keyframes.
        if attribute == "azimuth_deg":
            knot_values = np.degrees(np.unwrap(np.radians(knot_values)))
        return float(monotone_cubic(knot_times, knot_values, [time])[0])

    def _interpolate_point(self, attribute: str, time: float) -> np.ndarray:
        """An absolute (x, y, z) quantity, interpolated per component."""
        knot_times = [keyframe.time for keyframe in self.keyframes]
        points = np.array([getattr(keyframe, attribute) for keyframe in self.keyframes])
        return np.array(
            [
                monotone_cubic(knot_times, points[:, axis], [time])[0]
                for axis in range(3)
            ]
        )

    def target_position(self, time: float) -> Tuple[float, float]:
        """The target's (x, y) at ``time``."""
        if isinstance(self.target, Track):
            return self.target.position(time)
        return (float(self.target[0]), float(self.target[1]))

    def at(self, time: float) -> CameraState:
        """The camera at model time ``time`` (s)."""
        lens_mm = self._interpolate("lens_mm", time, DEFAULT_LENS_MM)
        if self.absolute:
            return CameraState(
                location=tuple(self._interpolate_point("location", time)),
                look_at=tuple(self._interpolate_point("look_at", time)),
                lens_mm=lens_mm,
            )

        # Relative: place the camera around the (possibly moving) target.
        values = {
            name: self._interpolate(name, time, default)
            for name, default in RELATIVE_QUANTITY_DEFAULTS.items()
        }
        target_x, target_y = self.target_position(time)
        azimuth = np.radians(values["azimuth_deg"])
        location = (
            target_x + values["horizontal_distance"] * float(np.cos(azimuth)),
            target_y + values["horizontal_distance"] * float(np.sin(azimuth)),
            values["height"],
        )
        look_at = (target_x, target_y, values["target_height"])
        return CameraState(location=location, look_at=look_at, lens_mm=lens_mm)

    def manifest_keyframes(
        self, times_seconds: Sequence[float], frame_start: int = 1
    ) -> List[dict]:
        """
        One manifest camera keyframe per frame, for the Blender build script.

        A keyframe on every frame leaves Blender nothing to interpolate, so
        the rendered motion is exactly this path.
        """
        keyframes = []
        for frame_offset, time in enumerate(times_seconds):
            state = self.at(float(time))
            keyframes.append(
                {
                    "frame": frame_start + frame_offset,
                    "location": [float(value) for value in state.location],
                    "look_at": [float(value) for value in state.look_at],
                    "up": [0.0, 0.0, 1.0],
                    "lens_mm": float(state.lens_mm),
                }
            )
        return keyframes
