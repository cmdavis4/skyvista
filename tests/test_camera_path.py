"""Tests for camera choreography (skyvista.camera_path)."""

import numpy as np
import pytest

from skyvista.camera_path import (
    CameraKeyframe,
    CameraPath,
    Track,
    monotone_cubic,
)


def test_monotone_cubic_passes_through_knots_without_overshoot():
    knot_times = [0.0, 1.0, 2.0, 3.0]
    knot_values = [0.0, 1.0, 1.0, 5.0]
    np.testing.assert_allclose(
        monotone_cubic(knot_times, knot_values, knot_times), knot_values
    )
    dense = monotone_cubic(knot_times, knot_values, np.linspace(0, 3, 301))
    # Monotone data stays monotone, and the flat segment stays flat.
    assert np.all(np.diff(dense) >= -1e-12)
    flat = monotone_cubic(knot_times, knot_values, np.linspace(1, 2, 11))
    np.testing.assert_allclose(flat, 1.0)


def test_monotone_cubic_has_continuous_velocity():
    knot_times = [0.0, 1.0, 3.0, 4.0]
    knot_values = [0.0, 2.0, 3.0, 7.0]
    step = 1e-5
    for knot_time in knot_times[1:-1]:
        left_slope = (
            monotone_cubic(knot_times, knot_values, [knot_time])[0]
            - monotone_cubic(knot_times, knot_values, [knot_time - step])[0]
        ) / step
        right_slope = (
            monotone_cubic(knot_times, knot_values, [knot_time + step])[0]
            - monotone_cubic(knot_times, knot_values, [knot_time])[0]
        ) / step
        assert left_slope == pytest.approx(right_slope, rel=1e-3)


def test_values_are_held_outside_the_keyframes():
    path = CameraPath(
        [
            CameraKeyframe(10.0, location=(0, 0, 0), look_at=(1, 0, 0)),
            CameraKeyframe(20.0, location=(10, 0, 0), look_at=(11, 0, 0)),
        ]
    )
    assert path.at(0.0).location == pytest.approx((0, 0, 0))
    assert path.at(30.0).location == pytest.approx((10, 0, 0))
    assert path.at(15.0).location[0] == pytest.approx(5.0)


def test_relative_camera_keeps_its_offset_from_a_moving_target():
    """A camera following a track keeps the keyframed distance and azimuth."""
    track = Track(times=[0, 100], x=[0, 1000], y=[0, 0])
    path = CameraPath(
        [
            CameraKeyframe(
                0.0,
                azimuth_deg=-90.0,
                horizontal_distance=500.0,
                height=100.0,
                target_height=300.0,
            )
        ],
        target=track,
    )
    state = path.at(50.0)
    assert state.look_at == pytest.approx((500.0, 0.0, 300.0))
    assert state.location == pytest.approx((500.0, -500.0, 100.0), abs=1e-9)


def test_azimuth_turns_the_short_way():
    path = CameraPath(
        [
            CameraKeyframe(0.0, azimuth_deg=350.0, horizontal_distance=1.0),
            CameraKeyframe(10.0, azimuth_deg=10.0, horizontal_distance=1.0),
        ],
        target=(0.0, 0.0),
    )
    location = path.at(5.0).location
    # Halfway is 360 = 0 deg, i.e. on +x; the long way would be 180 deg (-x).
    assert location[0] == pytest.approx(1.0, abs=1e-6)


def test_track_smoothing_removes_jitter():
    times = np.arange(0.0, 600.0, 10.0)
    jitter = np.where(np.arange(len(times)) % 2 == 0, 200.0, -200.0)
    track = Track(times=times, x=10.0 * times + jitter, y=0 * times, smoothing=30.0)
    # The smoothed position follows the 10 m/s trend within a few meters.
    assert track.position(300.0)[0] == pytest.approx(3000.0, abs=20.0)


def test_mixed_keyframe_kinds_are_rejected():
    with pytest.raises(ValueError, match="Mixing"):
        CameraPath(
            [
                CameraKeyframe(0.0, location=(0, 0, 0), look_at=(1, 0, 0)),
                CameraKeyframe(1.0, azimuth_deg=0.0),
            ],
            target=(0, 0),
        )


def test_manifest_has_a_keyframe_per_frame_with_lens():
    path = CameraPath(
        [
            CameraKeyframe(0.0, azimuth_deg=0.0, lens_mm=50.0),
            CameraKeyframe(10.0, azimuth_deg=0.0, lens_mm=28.0),
        ],
        target=(0.0, 0.0),
    )
    keyframes = path.manifest_keyframes([0.0, 5.0, 10.0], frame_start=1)
    assert [keyframe["frame"] for keyframe in keyframes] == [1, 2, 3]
    assert keyframes[0]["lens_mm"] == pytest.approx(50.0)
    assert keyframes[2]["lens_mm"] == pytest.approx(28.0)
