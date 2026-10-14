"""
Render one physically based "portrait" of each dc_entrainment CM1 storm.

Three storms, each at an informative time in its life cycle:

* ``ic`` -- isolated deep convection at 40 min, when the tower overshoots
  to its highest (16 km) and the anvil has begun to spread. The tobac
  updraft peak (23:50) comes earlier, while the tower is still growing.
* ``sl`` -- the squall line at 3 h, its tracking peak, seen from the
  ground ahead of the line: the row of convective towers under the forward
  anvil, lit by a low sun that shines in beneath the anvil.
* ``sc`` -- the right-moving supercell at its tracking peak (1:59:30). The
  domain also holds the left mover and smaller cells, and their anvils merge
  with the right mover's, so the right mover is cut out using the tobac
  tracks (see :func:`isolate_tracked_cell`).

Example (from the skyvista repository root, in the pixi ``blender``
environment)::

    pixi run -e blender python examples/cm1_storms/render_cm1_storm_portraits.py \\
        storm_portraits --storms ic sl sc \\
        --blender ~/programs/blender-5.0.1-linux-x64/blender

Every run uses CM1's Morrison scheme with hail (``ihail = 1``) and a fixed
droplet number of 300 cm^-3 (``ndcnst``), which the optics below match.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

import skyvista as sv
from skyvista.cloud_optics import cm1_morrison_hydrometeor_species

# Root of the dc_entrainment CM1 runs.
DC_ENTRAINMENT_DATA_ROOT = Path("/moonbow/cmdavis4/projects/dc_entrainment")

# Morrison settings shared by every run (namelist ihail = 1, ndcnst = 300).
DROPLET_NUMBER_PER_CUBIC_METER = 300.0e6
HAIL_DENSITY = 900.0

# Hydrometeor mixing ratios (kg/kg) and number concentrations (#/kg) that CM1
# writes for Morrison, plus the air density (kg m^-3).
CM1_CLOUD_VARIABLES = (
    "qc",
    "qr",
    "qi",
    "qs",
    "qg",
    "nci",
    "ncr",
    "ncs",
    "ncg",
    "rho",
)

# A competing tobac cell counts only if its feature is at least this fraction
# of the selected cell's size; smaller cells are fragments of the same storm.
COMPETING_CELL_MINIMUM_SIZE_FRACTION = 0.1


@dataclass
class StormView:
    """
    Which output to render and how to frame it.

    Attributes:
        run_dir: The CM1 run directory (holding ``output/`` and ``derived/``).
        time_seconds: Model time to render (s); the nearest output is used.
        title: Short description, printed with the result.
        x_range, y_range: Crop of the domain (m), or None for all of it.
        periodic_y_copies: Number of copies of the domain laid end to end in
            y. Valid only for a run that is periodic in y; a line that spans
            the domain then reaches that many times as far, so its ends can
            be kept out of the frame.
        isolate_tracked_cell: Keep only the tobac-selected cell's cloud.
        cutaway: ``(point, normal)``: remove the cloud on the side of this
            plane the normal points to, to show a cross-section.
        camera_azimuth_deg: Direction from the scene center (or the camera
            target) to the camera, counterclockwise from +x (east).
        camera_elevation_deg: Camera elevation above the scene center. Used
            only for bounding-box framing.
        camera_distance: Camera distance in bounding-box diagonals. Used only
            for bounding-box framing.
        camera_horizontal_distance: Frame explicitly instead (m): put the
            camera this far from the target, horizontally, at
            ``camera_azimuth_deg``. The target is the tracked cell with
            ``isolate_tracked_cell``, else ``camera_target_xy``, else the
            center of the crop. Bounding-box
            framing fits everything exported, which for a storm with a long
            anvil puts the camera far away and the storm on the horizon.
        camera_target_xy: (x, y) the explicit camera aims at (m).
        camera_height: Height of an explicit camera above the ground (m).
        camera_target_height: Height of the point an explicit camera aims at.
        camera_lens_mm: Focal length; 50 mm spans ~40 deg horizontally.
        sun_azimuth_deg: Direction toward the sun, counterclockwise from +x.
        sun_elevation_deg: Sun elevation (deg).
        vertical_spacing: Uniform vertical voxel size (m) of the cloud VDB;
            None lets skyvista choose (the finest cloudy spacing, floored at
            a quarter of the horizontal spacing).
    """

    run_dir: Path
    time_seconds: float
    title: str
    x_range: Optional[Tuple[float, float]] = None
    y_range: Optional[Tuple[float, float]] = None
    periodic_y_copies: int = 1
    isolate_tracked_cell: bool = False
    cutaway: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = (
        None
    )
    camera_azimuth_deg: float = -60.0
    camera_elevation_deg: float = -5.0
    camera_distance: float = 1.0
    camera_horizontal_distance: Optional[float] = None
    camera_target_xy: Optional[Tuple[float, float]] = None
    camera_height: float = 1500.0
    camera_target_height: float = 7000.0
    camera_lens_mm: float = 50.0
    sun_azimuth_deg: float = -150.0
    sun_elevation_deg: float = 30.0
    vertical_spacing: Optional[float] = None


STORM_VIEWS = {
    "ic": StormView(
        run_dir=DC_ENTRAINMENT_DATA_ROOT
        / "ic/higher_res/test_compression_tapfrq-10_prclfrq-10",
        time_seconds=2400.0,
        title="Isolated deep convection, 40 min",
        # The storm sits near (105, 75) km of a 250 x 150 km domain.
        x_range=(60.0e3, 160.0e3),
        y_range=(30.0e3, 120.0e3),
        # Far enough back that a 16:9 frame holds the whole anvil.
        camera_distance=1.25,
    ),
    "sl": StormView(
        run_dir=DC_ENTRAINMENT_DATA_ROOT / "sl/sl_mulholland21/bcs-periodic_nstar-1e-7",
        time_seconds=10800.0,
        title="Squall line, 3 h",
        # By 3 h the line moves east at ~16 m/s with its leading updrafts at
        # x = 265-300 km, a forward anvil to 348 km and trailing stratiform
        # cloud back to 202 km.
        x_range=(180.0e3, 380.0e3),
        # Three copies make a 300 km line, so its ends stay out of frame.
        periodic_y_copies=3,
        # From the ground, south-east of the line and ahead of it, looking
        # along it as it recedes north: the row of convective towers under
        # the forward anvil. The near end (y = 0) lies outside the frame.
        camera_azimuth_deg=-47.0,
        camera_horizontal_distance=110.0e3,
        camera_target_xy=(285.0e3, 110.0e3),
        camera_target_height=6000.0,
        camera_lens_mm=35.0,
        # A low sun behind the camera, ahead of the line. The forward anvil
        # (base ~8 km) overhangs the towers by ~50 km, so a sun higher than
        # ~9 deg leaves them in its shadow; at 8 deg the light passes under
        # the anvil onto the towers' camera-facing sides.
        sun_azimuth_deg=-40.0,
        sun_elevation_deg=8.0,
        vertical_spacing=125.0,
    ),
    "sc": StormView(
        run_dir=DC_ENTRAINMENT_DATA_ROOT
        / "sc/higher_res/dx-250_nstar-1e-07-nx-1500_ny-1060",
        time_seconds=7170.0,
        title="Right-moving supercell, 1:59:30",
        # The storm complex spans x = 120-300 km, y = 80-210 km.
        x_range=(100.0e3, 320.0e3),
        y_range=(60.0e3, 215.0e3),
        isolate_tracked_cell=True,
        # From the south-east, so the cut toward the left mover is behind,
        # aimed at the right mover's tracked position.
        camera_azimuth_deg=-60.0,
        camera_horizontal_distance=60.0e3,
        camera_target_height=8000.0,
        camera_lens_mm=28.0,
        vertical_spacing=125.0,
    ),
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output_dir", type=Path, help="Directory for the outputs.")
    parser.add_argument(
        "--storms",
        nargs="+",
        default=list(STORM_VIEWS),
        choices=list(STORM_VIEWS),
        help="Storms to render.",
    )
    parser.add_argument("--blender", default=None, help="Blender executable.")
    parser.add_argument("--samples", type=int, default=16, help="Cycles samples.")
    parser.add_argument("--width", type=int, default=1920, help="Image width (px).")
    parser.add_argument("--height", type=int, default=1080, help="Image height (px).")
    parser.add_argument(
        "--isolation-taper-km",
        type=float,
        default=10.0,
        help="Width of the fade between the tracked cell and its neighbors (km).",
    )
    return parser.parse_args()


def output_path_at_time(run_dir: Path, time_seconds: float) -> Path:
    """
    The gridded output file nearest a model time.

    CM1 numbers its files from 1 at t = 0 at a fixed interval, so the
    interval is read from the second file and the index follows from it.
    """
    output_dir = run_dir / "output"
    with xr.open_dataset(output_dir / "cm1out_000002.nc") as second_output:
        output_interval = float(second_output["time"].values[0])
    file_number = int(round(time_seconds / output_interval)) + 1
    return output_dir / f"cm1out_{file_number:06d}.nc"


def load_cm1_clouds(view: StormView) -> xr.Dataset:
    """
    Read the hydrometeor fields of one output time, cropped, in meters.

    CM1 names its scalar-point coordinates ``xh``, ``yh``, ``zh``; they are
    renamed to x, y, z, which skyvista recognises. Their values are already
    in meters in these runs.
    """
    data_path = output_path_at_time(view.run_dir, view.time_seconds)
    with xr.open_dataset(data_path) as raw:
        clouds = (
            raw[list(CM1_CLOUD_VARIABLES)]
            .isel(time=0)
            .rename({"xh": "x", "yh": "y", "zh": "z"})
        )
        # Crop to the storm, then read into memory.
        if view.x_range is not None:
            clouds = clouds.sel(x=slice(*view.x_range))
        if view.y_range is not None:
            clouds = clouds.sel(y=slice(*view.y_range))
        clouds = clouds.load()
    print(f"  {data_path.name}: t = {float(clouds['time']):.0f} s")
    return clouds


def tile_periodic_in_y(clouds: xr.Dataset, n_copies: int) -> xr.Dataset:
    """
    Lay ``n_copies`` of a y-periodic domain end to end.

    Each copy's y coordinates continue where the previous copy's end, one
    grid spacing on, so the tiled grid stays evenly spaced. This is exact only
    for a run with periodic north and south boundaries.
    """
    y_values = clouds["y"].values
    y_spacing = float(y_values[1] - y_values[0])
    domain_length = float(y_values[-1] - y_values[0]) + y_spacing
    shifted_copies = [
        clouds.assign_coords(y=y_values + copy_index * domain_length)
        for copy_index in range(n_copies)
    ]
    return xr.concat(shifted_copies, dim="y")


def isolate_tracked_cell(
    clouds: xr.Dataset, view: StormView, taper_width: float
) -> Tuple[xr.Dataset, Tuple[float, float]]:
    """
    Keep only the cloud of the tobac-selected cell (the right mover).

    The tracking products in ``derived/`` give every tracked cell's position
    (``_all_tracks.csv``) and which cell is the storm (``tracks.csv``, chosen
    by the tracking config's cell-selection rule). At the render time, each
    column is assigned to the nearer of the selected cell and its nearest
    competitor -- a Voronoi split of the plan view. Merged anvils have no
    other dividing line. The split is softened into a linear fade across
    ``taper_width``, so the cut anvil thins out instead of ending in a flat
    wall. Mixing ratios and number concentrations are scaled together, so
    the particle sizes, and with them the optics per unit mass, are kept.

    Competing cells smaller than COMPETING_CELL_MINIMUM_SIZE_FRACTION of the
    selected cell are ignored: they are fragments of the same storm.

    Returns:
        The weighted dataset, and the selected cell's (x, y) position (m).
    """
    derived_dir = view.run_dir / "derived"
    selected_tracks = pd.read_csv(derived_dir / "tracks.csv", parse_dates=["time"])
    all_tracks = pd.read_csv(derived_dir / "_all_tracks.csv", parse_dates=["time"])
    selected_cell = int(selected_tracks["cell"].iloc[0])

    # Features at the tracked time nearest the render time.
    render_time = pd.Timestamp("2000-01-01") + pd.Timedelta(
        seconds=float(clouds["time"])
    )
    nearest_tracked_time = all_tracks["time"].iloc[
        (all_tracks["time"] - render_time).abs().argmin()
    ]
    features_now = all_tracks[all_tracks["time"] == nearest_tracked_time]
    selected_feature = features_now[features_now["cell"] == selected_cell].iloc[0]
    competing_features = features_now[
        (features_now["cell"] != selected_cell)
        & (
            features_now["num"]
            >= COMPETING_CELL_MINIMUM_SIZE_FRACTION * selected_feature["num"]
        )
    ]

    # Feature positions are fractional grid indices (hdim_1 = y, hdim_2 = x)
    # into the full domain; convert them to meters with the uncropped grid.
    with xr.open_dataset(view.run_dir / "output" / "cm1out_000002.nc") as grid:
        full_x = grid["xh"].values
        full_y = grid["yh"].values

    def feature_position(feature) -> Tuple[float, float]:
        grid_indices = np.arange(len(full_x)), np.arange(len(full_y))
        return (
            float(np.interp(feature["hdim_2"], grid_indices[0], full_x)),
            float(np.interp(feature["hdim_1"], grid_indices[1], full_y)),
        )

    # Distance of every column to the selected cell and to the nearest rival.
    x_grid, y_grid = np.meshgrid(clouds["x"].values, clouds["y"].values)
    selected_x, selected_y = feature_position(selected_feature)
    distance_to_selected = np.hypot(x_grid - selected_x, y_grid - selected_y)
    distance_to_competitor = np.full_like(distance_to_selected, np.inf)
    for _, feature in competing_features.iterrows():
        competitor_x, competitor_y = feature_position(feature)
        distance_to_competitor = np.minimum(
            distance_to_competitor,
            np.hypot(x_grid - competitor_x, y_grid - competitor_y),
        )
    print(
        f"  tobac cell {selected_cell} at ({selected_x / 1e3:.0f},"
        f" {selected_y / 1e3:.0f}) km; split from cells"
        f" {competing_features['cell'].tolist()}"
    )

    # 1 on the selected cell's side, 0 on the competitor's, linear between.
    keep_weight = np.clip(
        0.5 + (distance_to_competitor - distance_to_selected) / taper_width, 0.0, 1.0
    )
    keep_weight = xr.DataArray(keep_weight, dims=("y", "x"))
    hydrometeor_variables = [name for name in clouds.data_vars if name != "rho"]
    isolated_clouds = clouds.assign(
        {name: clouds[name] * keep_weight for name in hydrometeor_variables}
    )
    return isolated_clouds, (selected_x, selected_y)


def make_export_config(
    view: StormView, arguments, target_xy: Tuple[float, float]
) -> sv.BlenderExportConfig:
    """
    Physical-sky export settings with this storm's framing and lighting.

    With ``camera_horizontal_distance`` the camera is placed explicitly,
    relative to ``target_xy``; otherwise skyvista frames the bounding box.
    """
    config = sv.BlenderExportConfig.for_clouds()
    config.render.resolution = (arguments.width, arguments.height)
    config.render.samples = arguments.samples
    config.world.sun_azimuth_deg = view.sun_azimuth_deg
    config.world.sun_elevation_deg = view.sun_elevation_deg
    config.camera.azimuth_deg = view.camera_azimuth_deg
    config.camera.elevation_deg = view.camera_elevation_deg
    config.camera.distance_factor = view.camera_distance
    config.camera.lens_mm = view.camera_lens_mm
    if view.camera_horizontal_distance is not None:
        # Explicit view: a camera near the ground, aimed at the target.
        azimuth = np.radians(view.camera_azimuth_deg)
        target_x, target_y = target_xy
        config.camera.location = (
            target_x + view.camera_horizontal_distance * np.cos(azimuth),
            target_y + view.camera_horizontal_distance * np.sin(azimuth),
            view.camera_height,
        )
        config.camera.look_at = (target_x, target_y, view.camera_target_height)
    return config


def main() -> None:
    arguments = parse_arguments()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    species = cm1_morrison_hydrometeor_species(
        droplet_number_concentration=DROPLET_NUMBER_PER_CUBIC_METER,
        graupel_density=HAIL_DENSITY,
    )

    for storm_name in arguments.storms:
        view = STORM_VIEWS[storm_name]
        print(f"{storm_name}: {view.title}")
        clouds = load_cm1_clouds(view)
        if view.periodic_y_copies > 1:
            clouds = tile_periodic_in_y(clouds, view.periodic_y_copies)
        # The camera target: the tracked cell, else the preset's target,
        # else the middle of the crop.
        target_xy = view.camera_target_xy or (
            float(clouds["x"].mean()),
            float(clouds["y"].mean()),
        )
        if view.isolate_tracked_cell:
            clouds, target_xy = isolate_tracked_cell(
                clouds, view, taper_width=arguments.isolation_taper_km * 1e3
            )

        bundle_dir = arguments.output_dir / storm_name
        sv.Scene().add_clouds(
            clouds,
            species=species,
            vertical_spacing=view.vertical_spacing,
            cutaway=view.cutaway,
        ).to_blender(
            bundle_dir,
            config=make_export_config(view, arguments, target_xy),
            render=True,
            blender_executable=arguments.blender,
        )
        print(f"  render: {bundle_dir / 'render' / 'frame_0001.png'}")


if __name__ == "__main__":
    main()
