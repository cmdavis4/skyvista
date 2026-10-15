"""
Tests for trajectory trails: the curve export of tube-style trajectories.

``TrajectorySpec.trails`` cuts every track to the trail behind the time shown
and flattens the trails end to end; the Blender export writes them as
polylines that the build script sweeps into tapered tubes with a head cone.
"""

import json
import subprocess
import textwrap

import numpy as np
import pytest
import xarray as xr

import skyvista as sv
from skyvista.blender import _resolve_blender_runner


def straight_tracks(times, n_tracks: int = 3, speed: float = 10.0) -> xr.Dataset:
    """
    Tracks that move along +x at ``speed`` m/s, one per y = 0, 100, 200, ...

    The track index is ``trajectory_ix``; a per-track label ``track_label``
    (one value per track) and a per-point ``height`` are included for
    coloring.
    """
    elapsed_seconds = np.arange(len(times)) * 10.0
    x = np.tile(speed * elapsed_seconds, (n_tracks, 1))
    y = np.repeat(np.arange(n_tracks) * 100.0, len(times)).reshape(n_tracks, -1)
    z = np.full_like(x, 1000.0)
    return xr.Dataset(
        {
            "x": (("trajectory_ix", "time"), x),
            "y": (("trajectory_ix", "time"), y),
            "z": (("trajectory_ix", "time"), z),
            "height": (("trajectory_ix", "time"), z + x / 100.0),
            "track_label": (("trajectory_ix",), np.arange(n_tracks) * 1.0),
        },
        coords={"time": times, "trajectory_ix": np.arange(n_tracks)},
    )


# Ten output times, 10 s apart, as plain seconds (as in CM1 output).
SECONDS_TIMES = np.arange(10) * 10.0


def make_trail_spec(**kwargs) -> "sv.TrajectorySpec":
    """A tube trajectory spec with no track limit, for the trail tests."""
    return sv.make_trajectory(limit=None, **kwargs)


def test_trails_are_flat_end_to_end_and_oldest_first():
    tracks = straight_tracks(SECONDS_TIMES)
    trails = make_trail_spec().trails(tracks, 50.0)
    # Every track has the six times up to t = 50 s.
    np.testing.assert_array_equal(trails.curve_sizes, [6, 6, 6])
    assert trails.positions.shape == (18, 3)
    # The first trail runs from x = 0 to its head at x = 500 m.
    np.testing.assert_allclose(trails.positions[:6, 0], np.arange(6) * 100.0)
    # Ages count back from the time shown.
    np.testing.assert_allclose(trails.ages[:6], [50, 40, 30, 20, 10, 0])


def test_trail_duration_keeps_only_the_recent_points():
    tracks = straight_tracks(SECONDS_TIMES)
    trails = make_trail_spec(trail_duration=20.0).trails(tracks, 90.0)
    # t = 70, 80, 90 s: three points per trail.
    np.testing.assert_array_equal(trails.curve_sizes, [3, 3, 3])
    np.testing.assert_allclose(trails.positions[:3, 0], [700.0, 800.0, 900.0])


def test_trail_duration_works_with_datetime_times():
    run_start = np.datetime64("2000-01-01T00:00:00")
    datetime_times = run_start + (SECONDS_TIMES * 1e9).astype("timedelta64[ns]")
    tracks = straight_tracks(datetime_times)
    trails = make_trail_spec(trail_duration=20.0).trails(tracks, datetime_times[-1])
    np.testing.assert_array_equal(trails.curve_sizes, [3, 3, 3])
    np.testing.assert_allclose(trails.ages[:3], [20.0, 10.0, 0.0])


def test_taper_falls_linearly_with_age_against_the_trail_duration():
    tracks = straight_tracks(SECONDS_TIMES)
    spec = make_trail_spec(trail_duration=40.0, trail_taper=0.2)
    trails = spec.trails(tracks, 90.0)
    # Ages 40, 30, 20, 10, 0 s -> scales 0.2, 0.4, 0.6, 0.8, 1.0.
    np.testing.assert_allclose(trails.radius_scales[:5], [0.2, 0.4, 0.6, 0.8, 1.0])
    # A young track shown at t = 20 s has filled only half its trail, so its
    # oldest point (20 s old) is only half tapered.
    young_trails = spec.trails(tracks, 20.0)
    np.testing.assert_allclose(young_trails.radius_scales[:3], [0.6, 0.8, 1.0])


def test_taper_without_a_duration_spans_each_trail():
    tracks = straight_tracks(SECONDS_TIMES)
    trails = make_trail_spec(trail_taper=0.0).trails(tracks, 40.0)
    np.testing.assert_allclose(trails.radius_scales[:5], [0.0, 0.25, 0.5, 0.75, 1.0])


def test_missing_positions_are_dropped_and_short_tracks_skipped():
    tracks = straight_tracks(SECONDS_TIMES)
    # Track 0 is missing at t = 20 s; track 1 has only one valid point.
    tracks["x"][0, 2] = np.nan
    tracks["x"][1, :] = np.nan
    tracks["x"][1, 0] = 0.0
    trails = make_trail_spec().trails(tracks, 40.0)
    np.testing.assert_array_equal(trails.curve_sizes, [4, 5])
    np.testing.assert_allclose(trails.positions[:4, 0], [0.0, 100.0, 300.0, 400.0])


def test_per_track_scalar_is_repeated_along_the_trail():
    tracks = straight_tracks(SECONDS_TIMES)
    trails = make_trail_spec(scalar="track_label").trails(tracks, 20.0)
    np.testing.assert_allclose(trails.color_values, [0, 0, 0, 1, 1, 1, 2, 2, 2])
    point_trails = make_trail_spec(scalar="height").trails(tracks, 20.0)
    np.testing.assert_allclose(point_trails.color_values[:3], [1000, 1001, 1002])


def test_pyvista_tubes_also_keep_only_the_trail():
    tracks = straight_tracks(SECONDS_TIMES)
    spec = make_trail_spec(trail_duration=20.0, tube_radius=5.0, head_length_frac=0)
    tube_mesh = spec.create_mesh(tracks, 90.0)
    # The trail runs from x = 700 m (t = 70 s) to 900 m (t = 90 s).
    x_min, x_max = tube_mesh.bounds[0], tube_mesh.bounds[1]
    np.testing.assert_allclose([x_min, x_max], [700.0, 900.0], atol=1.0)


def test_blender_export_writes_trails_as_polylines(tmp_path):
    tracks = straight_tracks(SECONDS_TIMES)
    scene = sv.Scene().add_trajectories(
        tracks,
        scalar="height",
        limit=None,
        tube_radius=20.0,
        trail_duration=30.0,
        trail_taper=0.5,
    )
    bundle_dir = tmp_path / "bundle"
    scene.to_blender(bundle_dir, times=[90.0])
    manifest = json.loads((bundle_dir / "scene.json").read_text())
    (object_entry,) = manifest["objects"]
    geometry = object_entry["geometry"]
    assert object_entry["spec_type"] == "trajectory"
    assert geometry["carrier"] == "polylines"
    # Head cone of the default head_length_frac (10) and head_radius_frac (2.5).
    assert geometry["head_cone"] == {"radius": 50.0, "length": 200.0}
    with np.load(bundle_dir / geometry["path_pattern"].replace("####", "0001")) as f:
        np.testing.assert_array_equal(f["curve_sizes"], [4, 4, 4])
        np.testing.assert_allclose(
            f["radius_scales"][:4], [0.5, 2 / 3, 5 / 6, 1.0], rtol=1e-6
        )


def test_head_length_zero_drops_the_head_cone(tmp_path):
    scene = sv.Scene().add_trajectories(
        straight_tracks(SECONDS_TIMES), limit=None, head_length_frac=0.0
    )
    scene.to_blender(tmp_path / "bundle", times=[90.0])
    manifest = json.loads((tmp_path / "bundle" / "scene.json").read_text())
    assert manifest["objects"][0]["geometry"]["head_cone"] is None


def find_blender_runner():
    """How skyvista would run Blender here (program or bpy module), or None."""
    try:
        return _resolve_blender_runner()
    except FileNotFoundError:
        return None


# Runs inside Blender: prints, for the evaluated trail mesh, the largest
# distance of a vertex from the track's axis (y = 0, z = 1000 m) in x bins,
# and the largest x of any vertex (the head cone's tip).
TUBE_PROFILE_PROBE = textwrap.dedent("""
    import json
    import bpy
    depsgraph = bpy.context.evaluated_depsgraph_get()
    vertices = []
    for blender_object in bpy.context.scene.objects:
        if blender_object.type != "CURVES":
            continue
        evaluated = blender_object.evaluated_get(depsgraph)
        # Keep the geometry set alive while its mesh is read.
        geometry = evaluated.evaluated_geometry()
        vertices += [tuple(v.co) for v in geometry.mesh.vertices]
    print("VERTICES" + json.dumps(vertices))
    """)


@pytest.mark.slow
@pytest.mark.skipif(find_blender_runner() is None, reason="needs Blender")
def test_blender_tube_tapers_toward_the_tail_and_has_a_head_cone(tmp_path):
    # One straight track along +x; the trail of the last 40 s runs from
    # x = 500 m to its head at x = 900 m, narrowing to 0.25 of the radius.
    tube_radius = 20.0
    scene = sv.Scene().add_trajectories(
        straight_tracks(SECONDS_TIMES, n_tracks=1),
        color="white",
        limit=None,
        tube_radius=tube_radius,
        tube_resolution=8,
        head_length_frac=5.0,
        trail_duration=40.0,
        trail_taper=0.25,
    )
    bundle_dir = tmp_path / "bundle"
    scene.to_blender(bundle_dir, times=[90.0], build=True)
    probe_path = tmp_path / "probe.py"
    probe_path.write_text(TUBE_PROFILE_PROBE)
    completed = subprocess.run(
        find_blender_runner().script_command(
            probe_path, [], blend_path=bundle_dir / "bundle.blend"
        ),
        capture_output=True,
        text=True,
        check=True,
    )
    (vertices_line,) = [
        line for line in completed.stdout.splitlines() if line.startswith("VERTICES")
    ]
    vertices = np.array(json.loads(vertices_line[len("VERTICES") :]))
    # Positions are in the Curves object's local frame (data meters).
    x = vertices[:, 0]
    distance_from_axis = np.hypot(vertices[:, 1], vertices[:, 2] - 1000.0)
    # The tube's radius near the tail end and just behind the head.
    tail_radius = distance_from_axis[np.abs(x - 500.0) < 1.0].max()
    head_radius = distance_from_axis[np.abs(x - 880.0) < 15.0].max()
    np.testing.assert_allclose(tail_radius, 0.25 * tube_radius, rtol=0.05)
    np.testing.assert_allclose(head_radius, tube_radius, rtol=0.1)
    # The head cone's tip leads the head by its length (5 tube radii).
    np.testing.assert_allclose(x.max(), 900.0 + 5 * tube_radius, atol=1.0)


def test_animation_frames_hold_the_trail_up_to_their_own_time(tmp_path):
    # Every frame's scene holds the whole track history; each frame's bundle
    # must still be one frame, with the trail ending at that frame's time.
    tracks = straight_tracks(SECONDS_TIMES)
    frame_times = [50.0, 90.0]

    def build_trail_frame(frame_time):
        return sv.Scene().add_trajectories(tracks, limit=None, trail_duration=20.0)

    animation = sv.BlenderAnimation(
        tmp_path / "animation",
        frame_times,
        build_trail_frame,
        config=sv.BlenderExportConfig(),
    )
    animation.export(workers=1)
    bundle_dir = animation.assemble()
    manifest = json.loads((bundle_dir / "scene.json").read_text())
    (trail_entry,) = manifest["objects"]
    assert trail_entry["geometry"]["n_frames"] == 2
    # The heads (each trail's last point) are where the tracks are at 50 s and
    # 90 s: x = 500 m and 900 m.
    head_x_by_frame = []
    for frame_number in (1, 2):
        trail_path = bundle_dir / trail_entry["geometry"]["path_pattern"].replace(
            "####", f"{frame_number:04d}"
        )
        with np.load(trail_path) as trail_file:
            head_indices = np.cumsum(trail_file["curve_sizes"]) - 1
            head_x_by_frame.append(trail_file["positions"][head_indices, 0])
    # Positions are relative to the fixed origin shift, so compare the step.
    np.testing.assert_allclose(head_x_by_frame[1] - head_x_by_frame[0], 400.0)
