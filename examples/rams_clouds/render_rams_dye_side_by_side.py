"""
Render a RAMS cloud side by side: the physically based cloud, and the same
view with the cloud thinned and the passive tracers shown as colored dye.

Example (from the repository root, in the pixi ``blender`` environment)::

    HDF5_PLUGIN_PATH=/path/to/hdf5plugin/plugins \\
    pixi run -e blender python examples/rams_clouds/render_rams_dye_side_by_side.py \\
        /path/to/output/a-L-2000-01-01-011000-g1.h5  dye_figure \\
        --blender ~/programs/blender-5.0.1-linux-x64/blender

The run must carry layered passive tracers (RAMS ``ITRACER`` > 0), each
marking the air that started in one horizontal layer. The script reads every
tracer's starting height from the run's first output file. It then dyes only
air that has moved more than ``--min-displacement`` vertically from where it
started, and colors that air by its mean starting height. The undisturbed
environment stays clear. What remains shows where the cloud's air came from,
and where air from aloft has been brought down.
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
from cloud_render_config import add_render_arguments, make_cloud_export_config
from rams_lite_loader import load_rams_lite_hydrometeors
from render_rams_streamlines_side_by_side import pinned_view
from side_by_side_figure import ColorbarSpec, compose_side_by_side

import skyvista as sv

# RAMS passive tracer variables: TRACERP001, TRACERP002, ...
TRACER_NAME_PATTERN = re.compile(r"^TRACERP\d{3}$")
# Tracer value of undiluted air from its layer (RAMS init_tracer).
TRACER_INITIAL_MIXING_RATIO = 1.0e5


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_path", type=Path, help="RAMS a-L/a-A ...-g1.h5 file.")
    parser.add_argument("output_dir", type=Path, help="Directory for the outputs.")
    parser.add_argument("--blender", default=None, help="Blender executable.")
    parser.add_argument(
        "--min-displacement",
        type=float,
        default=1000.0,
        help=(
            "Dye only air moved more than this (m) from its starting height."
            " Below about 1 km the turbulent boundary layer, which mixes air"
            " through its depth, fills with dye."
        ),
    )
    parser.add_argument(
        "--extinction-per-km",
        type=float,
        default=3.0,
        help="Optical depth of one kilometer of undiluted dye.",
    )
    parser.add_argument(
        "--self-illumination",
        type=float,
        default=0.3,
        help="Dye emission, as a fraction of display white.",
    )
    parser.add_argument(
        "--height-limits",
        type=float,
        nargs=2,
        default=(0.0, 3000.0),
        help="Starting heights (m) at the ends of the colormap.",
    )
    parser.add_argument(
        "--colormap", default="turbo", help="Colormap of the starting height."
    )
    parser.add_argument(
        "--cloud-thinning",
        type=float,
        default=0.03,
        help="Factor on cloud extinction in the dye panel.",
    )
    parser.add_argument(
        "--visibility-km",
        type=float,
        default=80.0,
        help="Ground visibility for the aerial-perspective haze (km).",
    )
    add_render_arguments(parser)
    # A tighter view than the storm scripts: this cloud is a narrow congestus.
    parser.set_defaults(
        samples=16, width=1440, height=1080, top_height=9000.0, camera_distance=0.8
    )
    return parser.parse_args()


def tracer_starting_heights(
    data_path: Path, top_height: float
) -> Tuple[List[str], List[float]]:
    """
    Names and starting heights (m) of the tracers that start below top_height.

    Each tracer's starting height is its tracer-weighted mean height in the
    run's first output file (the ``...-000000-g1.h5`` file next to
    ``data_path``). Air from above ``top_height`` cannot reach the loaded
    levels without having moved, so those tracers are left out.
    """
    import h5py

    initial_path = data_path.with_name(
        re.sub(r"-\d{6}-g1\.h5$", "-000000-g1.h5", data_path.name)
    )
    with h5py.File(initial_path, "r") as h5_file:
        tracer_names = sorted(
            name for name in h5_file if TRACER_NAME_PATTERN.match(name)
        )
    # Read the tracers on the full column, then weight heights by them.
    initial = load_rams_lite_hydrometeors(
        initial_path, top_height=None, variables=tracer_names
    )
    kept_names, starting_heights = [], []
    for tracer_name in tracer_names:
        # Horizontal total of the tracer on each level.
        level_totals = initial[tracer_name].sum(dim=("x", "y"))
        starting_height = float(
            (level_totals * initial["z"]).sum() / level_totals.sum()
        )
        if starting_height < top_height:
            kept_names.append(tracer_name)
            starting_heights.append(starting_height)
    return kept_names, starting_heights


def main() -> None:
    arguments = parse_arguments()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    # Hydrometeors and the tracers that can reach the loaded levels.
    clouds = load_rams_lite_hydrometeors(
        arguments.data_path, top_height=arguments.top_height
    )
    tracer_names, starting_heights = tracer_starting_heights(
        arguments.data_path, arguments.top_height
    )
    tracers = load_rams_lite_hydrometeors(
        arguments.data_path, top_height=arguments.top_height, variables=tracer_names
    )
    minutes = float(clouds["time"]) / 60.0

    # ---- Left panel: physical cloud with aerial perspective.
    realistic_config = make_cloud_export_config(arguments)
    realistic_config.world.aerial_perspective = True
    realistic_config.world.visibility_km = arguments.visibility_km
    realistic_dir = arguments.output_dir / "realistic"
    sv.Scene().add_clouds(clouds, microphysics="rams").to_blender(
        realistic_dir,
        config=realistic_config,
        render=True,
        blender_executable=arguments.blender,
    )

    # ---- Right panel: same view, thinned cloud, dye.
    camera_location, camera_look_at, origin_shift = pinned_view(realistic_dir)
    dye_config = make_cloud_export_config(arguments, origin_shift=origin_shift)
    dye_config.camera.location = camera_location
    dye_config.camera.look_at = camera_look_at
    dye_config.world.aerial_perspective = True
    dye_config.world.visibility_km = arguments.visibility_km
    dye_config.overlay_colorbars = False

    dye_dir = arguments.output_dir / "dye"
    dye_scene = sv.Scene().add_clouds(
        clouds, microphysics="rams", extinction_scale=arguments.cloud_thinning
    )
    dye_scene.add_dye(
        tracers,
        tracers=tracer_names,
        tracer_values=starting_heights,
        reference_value=TRACER_INITIAL_MIXING_RATIO,
        min_vertical_displacement=arguments.min_displacement,
        cmap=arguments.colormap,
        clim=tuple(arguments.height_limits),
        extinction_per_km=arguments.extinction_per_km,
        self_illumination=arguments.self_illumination,
    )
    dye_scene.to_blender(
        dye_dir,
        config=dye_config,
        render=True,
        blender_executable=arguments.blender,
    )

    # ---- Side by side.
    output_png = arguments.output_dir / f"dye_side_by_side_t{minutes:05.1f}min.png"
    height_minimum, height_maximum = arguments.height_limits
    compose_side_by_side(
        realistic_dir / "render" / "frame_0001.png",
        dye_dir / "render" / "frame_0001.png",
        {
            "left_title": "Cloud, physically based",
            "right_title": "Passive tracers as dye, colored by starting height",
            "time": f"t = {minutes:.1f} min",
            "right_note": (
                f"Dyed: air moved > {arguments.min_displacement / 1000:g} km "
                "vertically\n"
                f"Cloud extinction x {arguments.cloud_thinning:g}"
            ),
        },
        ColorbarSpec(
            cmap=arguments.colormap,
            clim=(height_minimum, height_maximum),
            tick_values=np.linspace(height_minimum, height_maximum, 4),
            title="Starting height of the air (km)",
            tick_scale=0.001,
        ),
        output_png,
    )
    print(f"Side-by-side figure: {output_png}")


if __name__ == "__main__":
    main()
