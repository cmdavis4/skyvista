"""
Tests for streamlines: tracing, downstream ordering, resampling, the frame
shift, and the Blender polyline export. (Tube and cone placement inside
Blender were checked on a rendered vortex.)
"""

import json

import numpy as np
import pytest
import xarray as xr

import skyvista as sv


def vortex_field(points):
    """Analytic test flow: counterclockwise swirl plus a central updraft."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    radius = np.hypot(x, y)
    swirl = 10 * np.exp(-((radius / 3000) ** 2))
    return np.stack(
        [
            -swirl * y / np.maximum(radius, 1),
            swirl * x / np.maximum(radius, 1),
            5 * np.exp(-((radius / 1500) ** 2)) * np.sin(np.pi * z / 6000),
        ],
        axis=1,
    )


@pytest.fixture
def vortex_dataset():
    x = np.linspace(-5000, 5000, 41)
    y = np.linspace(-5000, 5000, 41)
    z = np.linspace(0, 6000, 25)
    grid_points = np.stack(
        [axis.ravel() for axis in np.meshgrid(x, y, z, indexing="ij")], axis=1
    )
    components = vortex_field(grid_points).reshape(41, 41, 25, 3)
    return xr.Dataset(
        {
            "U": (("x", "y", "z"), components[..., 0]),
            "V": (("x", "y", "z"), components[..., 1]),
            "W": (("x", "y", "z"), components[..., 2]),
        },
        coords={"x": x, "y": y, "z": z},
    )


def make_vortex_streamlines(**kwargs):
    defaults = dict(u="U", v="V", w="W", seed_stride={"x": 8, "y": 8, "z": 12})
    return sv.make_streamlines(**(defaults | kwargs))


def test_polylines_run_downstream_and_are_evenly_resampled(vortex_dataset):
    spec = make_vortex_streamlines(min_length=500.0, point_spacing=100.0)
    lines = spec.polylines(vortex_dataset, None)
    assert len(lines) > 10
    for points, colors in lines:
        segments = np.diff(points, axis=0)
        # Downstream: steps follow the flow overall.
        assert np.sum(segments * vortex_field(points[:-1])) > 0
        # Even spacing, at most the requested distance.
        segment_lengths = np.linalg.norm(segments, axis=1)
        assert segment_lengths.max() <= 100.0 + 1e-6
        assert len(colors) == len(points)


def test_streamlines_follow_the_swirl(vortex_dataset):
    # Seeds near mid-height on the x axis; their streamlines circle the axis
    # at a nearly constant radius.
    seeds = np.array([[2000.0, 0.0, 3000.0], [-2500.0, 0.0, 3000.0]])
    spec = make_vortex_streamlines(seeds=seeds, max_length=5000.0)
    for points, _ in spec.polylines(vortex_dataset, None):
        radii = np.hypot(points[:, 0], points[:, 1])
        assert np.ptp(radii) < 0.1 * radii.mean()


def test_frame_velocity_is_subtracted(vortex_dataset):
    uniform = vortex_dataset.copy()
    uniform["U"] = xr.full_like(uniform["U"], 5.0)
    uniform["V"] = xr.zeros_like(uniform["V"])
    uniform["W"] = xr.zeros_like(uniform["W"])
    seeds = np.array([[0.0, 0.0, 3000.0]])
    ground_frame = make_vortex_streamlines(seeds=seeds, direction="forward")
    (points, _), *_ = ground_frame.polylines(uniform, None)
    assert points[-1, 0] > points[0, 0]
    # In a frame moving at 10 m/s along x, the air moves toward -x.
    moving_frame = make_vortex_streamlines(
        seeds=seeds, direction="forward", frame_velocity=(10.0, 0.0, 0.0)
    )
    (points, _), *_ = moving_frame.polylines(uniform, None)
    assert points[-1, 0] < points[0, 0]


def test_blender_export_writes_polylines(vortex_dataset, tmp_path):
    scene = sv.Scene().add_streamlines(
        vortex_dataset,
        u="U",
        v="V",
        w="W",
        seed_stride={"x": 8, "y": 8, "z": 12},
        color_by="w",
        cmap="RdBu_r",
        clim=(-5.0, 5.0),
        tube_radius=40.0,
        arrow_spacing=1500.0,
    )
    scene.to_blender(tmp_path / "bundle")
    manifest = json.loads((tmp_path / "bundle" / "scene.json").read_text())
    (entry,) = manifest["objects"]
    geometry = entry["geometry"]
    assert geometry["carrier"] == "polylines"
    assert geometry["tube_radius"] == 40.0
    assert geometry["arrow_spacing"] == 1500.0
    point_path = tmp_path / "bundle" / geometry["path_pattern"].replace("####", "0001")
    with np.load(point_path) as polyline_file:
        # Every point belongs to exactly one curve.
        assert polyline_file["curve_sizes"].sum() == len(polyline_file["positions"])
        assert len(polyline_file["colors"]) == len(polyline_file["positions"])
