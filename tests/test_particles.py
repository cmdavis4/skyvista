"""
Tests for trajectory particles exported as point clouds for Blender.

The export writes per-frame ``.npz`` point files (positions and baked colors);
Blender's build script turns them into native point clouds. Placement inside
Blender was checked by evaluating a built scene's point cloud.
"""

import json

import numpy as np
import pytest
import xarray as xr

import skyvista as sv


@pytest.fixture
def parcel_dataset():
    """Five parcels over three times; parcel 4 starts late (NaN at t=0)."""
    n_parcels, times = 5, np.array([0.0, 60.0, 120.0])
    x = np.tile(np.arange(n_parcels, dtype=float)[:, None] * 100.0, (1, 3))
    y = np.zeros((n_parcels, 3))
    # Each parcel rises 10 m per second.
    z = np.tile(times[None, :] * 10.0, (n_parcels, 1))
    x[4, 0] = np.nan
    return xr.Dataset(
        {
            "x": (("trajectory_ix", "time"), x),
            "y": (("trajectory_ix", "time"), y),
            "z": (("trajectory_ix", "time"), z),
            "tracer": (("trajectory_ix", "time"), z / 1000.0),
        },
        coords={"trajectory_ix": np.arange(n_parcels), "time": times},
    )


def test_particle_positions_follow_the_requested_time(parcel_dataset):
    spec = sv.make_trajectory(scalar="tracer", style="particle")
    positions, tracer = spec.particle_positions(parcel_dataset, 60.0)
    np.testing.assert_allclose(positions[:, 2], 600.0)
    np.testing.assert_allclose(tracer, 0.6)
    # The late starter has no position at t = 0, so it is dropped there.
    positions_at_start, _ = spec.particle_positions(parcel_dataset, 0.0)
    assert len(positions_at_start) == 4


def test_pyvista_particle_radius(parcel_dataset):
    spec = sv.make_trajectory(style="particle", particle_radius=30.0)
    mesh = spec.create_mesh(parcel_dataset, None)
    # Sphere glyphs span twice the radius around each parcel.
    z_extent = mesh.bounds[5] - mesh.bounds[4]
    assert z_extent == pytest.approx(60.0, rel=0.05)


def test_blender_export_writes_point_cloud_frames(parcel_dataset, tmp_path):
    scene = sv.Scene().add_trajectories(
        parcel_dataset,
        scalar="tracer",
        style="particle",
        particle_radius=25.0,
        cmap="viridis",
    )
    scene.to_blender(tmp_path / "bundle")
    manifest = json.loads((tmp_path / "bundle" / "scene.json").read_text())
    (entry,) = manifest["objects"]
    assert entry["geometry"]["carrier"] == "point_glyphs"
    assert entry["geometry"]["glyph_type"] == "sphere"
    assert entry["geometry"]["radius"] == 25.0
    assert entry["material"]["coloring"]["mode"] == "vertex_color"

    bundle = tmp_path / "bundle"
    frame_paths = [
        bundle / entry["geometry"]["path_pattern"].replace("####", f"{index:04d}")
        for index in (1, 2, 3)
    ]
    with np.load(frame_paths[0]) as first, np.load(frame_paths[2]) as last:
        # The late starter is missing only at the first time.
        assert len(first["positions"]) == 4
        assert len(last["positions"]) == 5
        np.testing.assert_allclose(last["positions"][:, 2], 1200.0)
        # Particles carry no vectors.
        assert "vectors" not in first
