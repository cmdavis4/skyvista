"""
Tests for streamlines: tracing, downstream ordering, resampling, the frame
shift, travel times, the Blender polyline export, and (when a Blender is
available) markers that move with the flow clock. Tube and static cone
placement inside Blender were checked on a rendered vortex.
"""

import json
import os
import shutil
import subprocess
import textwrap

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
    defaults = {"u": "U", "v": "V", "w": "W", "seed_stride": {"x": 8, "y": 8, "z": 12}}
    return sv.make_streamlines(**(defaults | kwargs))


def test_polylines_run_downstream_and_are_evenly_resampled(vortex_dataset):
    spec = make_vortex_streamlines(min_length=500.0, point_spacing=100.0)
    lines = spec.polylines(vortex_dataset, None)
    assert len(lines) > 10
    for points, colors, *_ in lines:
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
    for points, *_ in spec.polylines(vortex_dataset, None):
        radii = np.hypot(points[:, 0], points[:, 1])
        assert np.ptp(radii) < 0.1 * radii.mean()


def test_frame_velocity_is_subtracted(vortex_dataset):
    uniform = vortex_dataset.copy()
    uniform["U"] = xr.full_like(uniform["U"], 5.0)
    uniform["V"] = xr.zeros_like(uniform["V"])
    uniform["W"] = xr.zeros_like(uniform["W"])
    seeds = np.array([[0.0, 0.0, 3000.0]])
    ground_frame = make_vortex_streamlines(seeds=seeds, direction="forward")
    points = ground_frame.polylines(uniform, None)[0].points
    assert points[-1, 0] > points[0, 0]
    # In a frame moving at 10 m/s along x, the air moves toward -x.
    moving_frame = make_vortex_streamlines(
        seeds=seeds, direction="forward", frame_velocity=(10.0, 0.0, 0.0)
    )
    points = moving_frame.polylines(uniform, None)[0].points
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


def uniform_flow_dataset(u_speed: float) -> xr.Dataset:
    """A uniform wind of u_speed (m/s) along +x on a 10 km cube."""
    x = np.linspace(-5000, 5000, 41)
    y = np.linspace(-5000, 5000, 41)
    z = np.linspace(0, 6000, 25)
    shape = (len(x), len(y), len(z))
    return xr.Dataset(
        {
            "U": (("x", "y", "z"), np.full(shape, u_speed)),
            "V": (("x", "y", "z"), np.zeros(shape)),
            "W": (("x", "y", "z"), np.zeros(shape)),
        },
        coords={"x": x, "y": y, "z": z},
    )


def test_travel_times_are_distance_over_speed_from_the_seed():
    # In a uniform 5 m/s wind, air reaches x from the seed at x = 1000 m in
    # (x - 1000) / 5 s: negative upstream, positive downstream.
    seed_x = 1000.0
    spec = make_vortex_streamlines(
        seeds=np.array([[seed_x, 0.0, 3000.0]]), max_length=3000.0
    )
    # The two halves traced from the seed are joined into one line.
    (line,) = spec.polylines(uniform_flow_dataset(5.0), None)
    assert line.points[0, 0] < seed_x < line.points[-1, 0]
    expected = (line.points[:, 0] - seed_x) / 5.0
    np.testing.assert_allclose(line.travel_times, expected, atol=1e-3)
    assert np.all(np.diff(line.travel_times) > 0)
    np.testing.assert_allclose(line.speeds, 5.0, rtol=1e-6)


def test_minimum_speed_bounds_travel_time():
    # Nearly still air: travel times use the speed floor, not 1e-3 m/s.
    spec = make_vortex_streamlines(
        seeds=np.array([[0.0, 0.0, 3000.0]]), direction="forward", max_length=1000.0
    )
    spec.appearance.arrow_minimum_speed = 2.0
    (line,) = spec.polylines(uniform_flow_dataset(0.01), None)
    length = np.linalg.norm(line.points[-1] - line.points[0])
    assert line.travel_times[-1] == pytest.approx(length / 2.0, rel=1e-3)


def test_hold_writes_video_frames_and_a_flow_clock(vortex_dataset, tmp_path):
    scene = sv.Scene().add_streamlines(
        vortex_dataset,
        u="U",
        v="V",
        w="W",
        seed_stride={"x": 8, "y": 8, "z": 12},
        arrow_travel_time=60.0,
    )
    config = sv.BlenderExportConfig(
        hold=sv.BlenderHoldConfig(n_frames=5, clock_seconds_per_frame=12.0)
    )
    scene.to_blender(tmp_path / "bundle", config=config)
    manifest = json.loads((tmp_path / "bundle" / "scene.json").read_text())
    assert manifest["time"]["frame_end"] == 5
    clock = [keyframe["seconds"] for keyframe in manifest["flow_clock"]["keyframes"]]
    assert clock == [0.0, 12.0, 24.0, 36.0, 48.0]
    (entry,) = manifest["objects"]
    assert entry["geometry"]["arrow_travel_time"] == 60.0
    # One data file, shown on every video frame.
    assert entry["geometry"]["n_frames"] == 1
    point_path = (
        tmp_path / "bundle" / entry["geometry"]["path_pattern"].replace("####", "0001")
    )
    with np.load(point_path) as polyline_file:
        assert len(polyline_file["travel_times"]) == len(polyline_file["positions"])
        assert len(polyline_file["speeds"]) == len(polyline_file["positions"])


def test_hold_needs_a_single_time(vortex_dataset, tmp_path):
    two_times = xr.concat(
        [vortex_dataset, vortex_dataset], dim=xr.DataArray([0.0, 60.0], dims="time")
    )
    scene = sv.Scene().add_streamlines(two_times, u="U", v="V", w="W")
    config = sv.BlenderExportConfig(
        hold=sv.BlenderHoldConfig(n_frames=5, clock_seconds_per_frame=12.0)
    )
    with pytest.raises(ValueError, match="one data time"):
        scene.to_blender(tmp_path / "bundle", config=config)


def find_blender():
    """A Blender executable from $SKYVISTA_BLENDER or the PATH, or None."""
    return os.environ.get("SKYVISTA_BLENDER") or shutil.which("blender")


# Runs inside Blender: prints the x of every cone tip (the only vertices on
# the streamline's axis; tube vertices sit one tube radius off it) at each
# requested frame.
CONE_TIP_PROBE = textwrap.dedent("""
    import json, sys
    import bpy
    frames = json.loads(sys.argv[sys.argv.index("--") + 1])
    tips_by_frame = {}
    for frame in frames:
        bpy.context.scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        tips = []
        for blender_object in bpy.context.scene.objects:
            if blender_object.type != "CURVES":
                continue
            evaluated = blender_object.evaluated_get(depsgraph)
            # Keep the geometry set alive while its mesh is read.
            geometry = evaluated.evaluated_geometry()
            mesh = geometry.mesh
            # The axis is the middle of the (symmetric) tubes and cones.
            y_values = [v.co.y for v in mesh.vertices]
            z_values = [v.co.z for v in mesh.vertices]
            axis_y = (min(y_values) + max(y_values)) / 2
            axis_z = (min(z_values) + max(z_values)) / 2
            tips += [
                v.co.x
                for v in mesh.vertices
                if abs(v.co.y - axis_y) < 0.1 and abs(v.co.z - axis_z) < 0.1
            ]
        tips_by_frame[frame] = sorted(tips)
    print("TIPS" + json.dumps(tips_by_frame))
    """)


@pytest.mark.slow
@pytest.mark.skipif(find_blender() is None, reason="needs Blender")
def test_markers_move_with_the_air(tmp_path):
    # A straight streamline in a 10 m/s wind, markers every 100 s of travel
    # (1000 m apart), the clock advancing 20 s per frame: every cone moves
    # 200 m downstream per frame, and the spacing stays 1000 m.
    scene = sv.Scene().add_streamlines(
        uniform_flow_dataset(10.0),
        u="U",
        v="V",
        w="W",
        seeds=np.array([[0.0, 0.0, 3000.0]]),
        max_length=4000.0,
        tube_radius=40.0,
        arrow_travel_time=100.0,
        color="white",
    )
    config = sv.BlenderExportConfig(
        hold=sv.BlenderHoldConfig(n_frames=3, clock_seconds_per_frame=20.0)
    )
    bundle_dir = tmp_path / "bundle"
    scene.to_blender(
        bundle_dir, config=config, build=True, blender_executable=find_blender()
    )
    probe_path = tmp_path / "probe.py"
    probe_path.write_text(CONE_TIP_PROBE)
    completed = subprocess.run(
        [
            find_blender(),
            "--background",
            str(bundle_dir / "bundle.blend"),
            "--python",
            str(probe_path),
            "--",
            "[1, 2]",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    (tips_line,) = [
        line for line in completed.stdout.splitlines() if line.startswith("TIPS")
    ]
    tips_by_frame = json.loads(tips_line[len("TIPS") :])
    # Positions are in the Curves object's local frame (data meters).
    first_tips = np.array(tips_by_frame["1"])
    second_tips = np.array(tips_by_frame["2"])
    # Full-size markers, away from the line's ends (at x = -4000 and 4000).
    interior_first = np.unique(first_tips[np.abs(first_tips) < 3000.0].round(1))
    assert len(interior_first) >= 5
    # Markers sit on the seed's travel-time lattice: tips one head length
    # (8 tube radii) past x = 1000 k.
    head_length = 8 * 40.0
    np.testing.assert_allclose((interior_first - head_length) % 1000.0, 0.0, atol=1.0)
    for tip in interior_first:
        assert np.min(np.abs(second_tips - (tip + 200.0))) < 1.0
