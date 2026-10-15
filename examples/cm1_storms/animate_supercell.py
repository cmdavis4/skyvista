"""
Animate the dc_entrainment CM1 supercell from first cloud to maturity.

The camera follows the storm along its tobac track: the parent cell until the
storm splits (~59 min), then the right mover (the cell the tracking config
selects). It pulls back as the storm grows, from ~18 km at the first cumulus
to ~60 km at the mature supercell, and slowly swings around it. After the
split the left mover is faded out over FADE_DURATION_SECONDS with the same
tapered split the portrait uses (see ``tracked_cell_keep_weight`` in
``render_cm1_storm_portraits.py``), so the animation ends on the right mover
alone.

Example (from the skyvista repository root, in the pixi ``blender``
environment)::

    pixi run -e blender python examples/cm1_storms/animate_supercell.py \\
        supercell_animation \\
        --blender ~/programs/blender-5.0.1-linux-x64/blender \\
        --ffmpeg /path/to/ffmpeg

Every stage resumes from what is on disk, so the command can be rerun after
an interruption. ``--test-times 300 3560 7180`` renders just those times.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from render_cm1_storm_portraits import (
    CM1_CLOUD_VARIABLES,
    DC_ENTRAINMENT_DATA_ROOT,
    DROPLET_NUMBER_PER_CUBIC_METER,
    HAIL_DENSITY,
    STORM_WEIGHT_VARIABLE,
    output_path_at_time,
    tracked_cell_keep_weight,
)

import skyvista as sv
from skyvista.cloud_optics import cm1_morrison_hydrometeor_species

SUPERCELL_RUN_DIR = (
    DC_ENTRAINMENT_DATA_ROOT / "sc/higher_res/dx-250_nstar-1e-07-nx-1500_ny-1060"
)
# Reference time of the tobac tables (model time zero).
TRACK_TIME_ORIGIN = pd.Timestamp("2000-01-01")

# The storm's region: it starts near (125, 170) km and the right mover ends
# near (180, 120) km, with its anvil reaching x = 300 km. Small cumulus
# elsewhere in the domain are left out.
X_RANGE = (80.0e3, 320.0e3)
Y_RANGE = (50.0e3, 230.0e3)

# After the split, the left mover fades out over this long (s).
FADE_DURATION_SECONDS = 600.0
# Width of the fade between the right and left movers' sides (m).
ISOLATION_TAPER = 10.0e3
# Smoothing timescale of the camera's target track (s).
TRACK_SMOOTHING_SECONDS = 300.0
# Uniform vertical voxel size of the cloud VDBs (m).
VERTICAL_SPACING = 125.0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output_dir", type=Path, help="Directory for the outputs.")
    parser.add_argument("--blender", default=None, help="Blender executable.")
    parser.add_argument("--ffmpeg", default=None, help="ffmpeg executable.")
    parser.add_argument("--start", type=float, default=300.0, help="First time (s).")
    parser.add_argument("--end", type=float, default=7180.0, help="Last time (s).")
    parser.add_argument(
        "--cadence", type=float, default=20.0, help="Model time between frames (s)."
    )
    parser.add_argument(
        "--test-times",
        type=float,
        nargs="+",
        default=None,
        help="Render only these times (s), e.g. to check timing and framing.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Width (px).")
    parser.add_argument("--height", type=int, default=720, help="Height (px).")
    parser.add_argument("--samples", type=int, default=16, help="Cycles samples.")
    parser.add_argument("--fps", type=float, default=24.0, help="Video frame rate.")
    parser.add_argument(
        "--export-workers", type=int, default=12, help="Parallel frame exports."
    )
    parser.add_argument(
        "--render-processes", type=int, default=4, help="Parallel Blender renders."
    )
    return parser.parse_args()


# =============================================================================
# The storm's track
# =============================================================================
def track_seconds(times: pd.Series) -> np.ndarray:
    """Seconds since model time zero, for tobac table timestamps."""
    return ((times - TRACK_TIME_ORIGIN) / pd.Timedelta(seconds=1)).to_numpy()


def split_time_seconds() -> float:
    """When the selected cell (the right mover) first appears (s)."""
    selected = pd.read_csv(SUPERCELL_RUN_DIR / "derived" / "tracks.csv")
    return float(track_seconds(pd.to_datetime(selected["time"])).min())


def storm_track() -> sv.Track:
    """
    The storm's position over time, for the camera to follow.

    Before the split the storm is the parent cell. tobac re-labels it
    several times (cells 1 to 6, back to back), so at each time the largest
    feature is taken. From the split on, the storm is the right mover. The
    hand-off jumps by the distance between the two cores, so the track is
    smoothed over TRACK_SMOOTHING_SECONDS.
    """
    derived_dir = SUPERCELL_RUN_DIR / "derived"
    all_tracks = pd.read_csv(derived_dir / "_all_tracks.csv", parse_dates=["time"])
    selected_cell = int(pd.read_csv(derived_dir / "tracks.csv")["cell"].iloc[0])
    all_tracks["seconds"] = track_seconds(all_tracks["time"])
    split = split_time_seconds()

    # Largest feature at each time before the split; the right mover after.
    before_split = all_tracks[all_tracks["seconds"] < split]
    parent = before_split.loc[before_split.groupby("seconds")["num"].idxmax()]
    right_mover = all_tracks[
        (all_tracks["seconds"] >= split) & (all_tracks["cell"] == selected_cell)
    ]
    storm = pd.concat([parent, right_mover]).sort_values("seconds")

    # Fractional grid indices (hdim_1 = y, hdim_2 = x) to meters.
    with xr.open_dataset(SUPERCELL_RUN_DIR / "output" / "cm1out_000002.nc") as grid:
        full_x = grid["xh"].values
        full_y = grid["yh"].values
    x = np.interp(storm["hdim_2"], np.arange(len(full_x)), full_x)
    y = np.interp(storm["hdim_1"], np.arange(len(full_y)), full_y)
    return sv.Track(
        times=storm["seconds"].to_numpy(),
        x=x,
        y=y,
        smoothing=TRACK_SMOOTHING_SECONDS,
    )


# =============================================================================
# One frame
# =============================================================================
SPECIES = cm1_morrison_hydrometeor_species(
    droplet_number_concentration=DROPLET_NUMBER_PER_CUBIC_METER,
    graupel_density=HAIL_DENSITY,
)
SPLIT_SECONDS = split_time_seconds()


def load_storm_region(time_seconds: float) -> xr.Dataset:
    """The storm region's hydrometeors at one time, coordinates in meters."""
    data_path = output_path_at_time(SUPERCELL_RUN_DIR, time_seconds)
    with xr.open_dataset(data_path) as raw:
        clouds = (
            raw[list(CM1_CLOUD_VARIABLES)]
            .isel(time=0)
            .rename({"xh": "x", "yh": "y", "zh": "z"})
            .sel(x=slice(*X_RANGE), y=slice(*Y_RANGE))
            .load()
        )
    return clouds


def left_mover_fade(time_seconds: float) -> float:
    """0 before the split, rising linearly to 1 FADE_DURATION_SECONDS after."""
    return float(np.clip((time_seconds - SPLIT_SECONDS) / FADE_DURATION_SECONDS, 0, 1))


def build_frame(time_seconds: float) -> sv.Scene:
    """
    The scene at one time: the storm's cloud, with the left mover fading.

    After the split, the right mover's keep weight (1 on its side, 0 on the
    left mover's) is blended toward it: ``1 - fade * (1 - weight)``. At fade
    0 nothing is removed; at fade 1 the cut is the portrait's.
    """
    clouds = load_storm_region(time_seconds)
    fade = left_mover_fade(time_seconds)
    if fade == 0.0:
        return sv.Scene().add_clouds(
            clouds, species=SPECIES, vertical_spacing=VERTICAL_SPACING
        )
    keep_weight, _ = tracked_cell_keep_weight(
        clouds, SUPERCELL_RUN_DIR, ISOLATION_TAPER
    )
    clouds = clouds.assign({STORM_WEIGHT_VARIABLE: 1.0 - fade * (1.0 - keep_weight)})
    return sv.Scene().add_clouds(
        clouds,
        species=SPECIES,
        vertical_spacing=VERTICAL_SPACING,
        weight=STORM_WEIGHT_VARIABLE,
    )


# =============================================================================
# Camera, figure, animation
# =============================================================================
def camera_path(start: float, end: float) -> sv.CameraPath:
    """
    Follow the storm, pulling back and rising as it grows.

    The aim point climbs with the cloud top, and the camera swings 30 deg
    around the storm from south-south-east to south-east over the whole
    animation, so the view keeps some parallax.
    """
    lens_mm = 28.0
    keyframes = [
        # A few cumulus towers, ~3 km tall.
        sv.CameraKeyframe(
            start,
            azimuth_deg=-75.0,
            horizontal_distance=18.0e3,
            height=1000.0,
            target_height=2500.0,
            lens_mm=lens_mm,
        ),
        # Towering cumulus reaching 13 km.
        sv.CameraKeyframe(
            1200.0, horizontal_distance=32.0e3, height=1500.0, target_height=6000.0
        ),
        # The split: a 21 km cumulonimbus with a spreading anvil.
        sv.CameraKeyframe(3600.0, horizontal_distance=50.0e3, target_height=8000.0),
        # The mature right mover.
        sv.CameraKeyframe(
            end, azimuth_deg=-45.0, horizontal_distance=60.0e3, target_height=8000.0
        ),
    ]
    return sv.CameraPath(keyframes, target=storm_track())


def phase_note(frame: sv.compose.FrameContext) -> str:
    """A caption for what is happening in the storm at this frame."""
    if frame.time_seconds < SPLIT_SECONDS:
        return "Parent storm"
    if left_mover_fade(frame.time_seconds) < 1.0:
        return "The storm splits; the left mover is faded out"
    return "Right mover"


def make_figure() -> sv.compose.Figure:
    """The frame layout: render, title, phase caption, time stamp."""
    return sv.compose.Figure(
        panels=[
            sv.compose.Panel(
                lambda frame: frame.render_path,
                [
                    sv.compose.Text(
                        "Right-moving supercell  (CM1, 250 m grid)",
                        anchor="upper left",
                        size=0.04,
                    ),
                    sv.compose.Text(
                        phase_note, anchor="upper left", size=0.03, offset=(0.0, -0.06)
                    ),
                    sv.compose.TimeStamp(anchor="lower left", size=0.04),
                ],
            )
        ]
    )


def frame_times(arguments) -> List[float]:
    """Model times of the frames (s), on existing 10 s outputs."""
    if arguments.test_times:
        return list(arguments.test_times)
    return list(np.arange(arguments.start, arguments.end + 0.5, arguments.cadence))


def make_config(arguments) -> sv.BlenderExportConfig:
    """Physical sky, no haze, the portrait's sun; reduced render quality."""
    config = sv.BlenderExportConfig.for_clouds()
    config.render.resolution = (arguments.width, arguments.height)
    config.render.samples = arguments.samples
    config.world.sun_azimuth_deg = -150.0
    config.world.sun_elevation_deg = 30.0
    return config


def main() -> None:
    arguments = parse_arguments()
    times = frame_times(arguments)
    animation = sv.BlenderAnimation(
        arguments.output_dir,
        times,
        build_frame,
        config=make_config(arguments),
        camera_path=camera_path(times[0], times[-1]),
        fps=arguments.fps,
    )
    animation.run(
        figure=make_figure(),
        export_workers=arguments.export_workers,
        blender_executable=arguments.blender,
        render_processes=arguments.render_processes,
        ffmpeg=arguments.ffmpeg,
    )


def describe_track() -> List[Tuple[float, float, float]]:
    """(time, x, y) of the smoothed track every 10 min, for checking."""
    track = storm_track()
    return [(t, *track.position(t)) for t in np.arange(300.0, 7200.0, 600.0)]


if __name__ == "__main__":
    main()
