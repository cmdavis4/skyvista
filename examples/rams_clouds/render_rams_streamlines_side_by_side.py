"""
Render a RAMS storm side by side: the physically based cloud, and the same
view with the cloud thinned and storm-relative streamlines through the
updraft. Both panels include aerial perspective (physical haze).

Example (from the repository root, in the pixi ``blender`` environment)::

    pixi run -e blender python examples/rams_clouds/render_rams_streamlines_side_by_side.py \\
        /path/to/output/a-L-2000-01-01-013420-g1.h5  streamline_figure \\
        --blender ~/programs/blender-5.0.1-linux-x64/blender

Streamlines follow the instantaneous wind, so their shape depends on the
frame they are drawn in. In the ground frame, the environmental wind sweeps
every line downstream and hides the storm's own circulation. This script
subtracts an estimate of the storm motion -- the mean wind between the ground
and 6 km, a classic proxy -- so the inflow, the updraft and the anvil
outflow show as the storm itself sees them.

Seeds are lattice points inside the updraft (w above ``--seed-w``); each
streamline is traced both ways from its seed, so it shows where updraft air
comes from and where it goes. Cones along each tube point downstream.
"""

import argparse
from pathlib import Path

import numpy as np
from cloud_render_config import add_render_arguments, make_cloud_export_config
from rams_lite_loader import load_rams_lite_hydrometeors, load_rams_lite_winds
from side_by_side_figure import ColorbarSpec, compose_side_by_side

import skyvista as sv

# Layer whose mean wind estimates the storm motion (m).
STORM_MOTION_LAYER_TOP_M = 6000.0
# Streamline colormap: diverging, white at w = 0 (reads on haze and sky).
STREAMLINE_COLORMAP = "RdBu_r"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_path", type=Path, help="RAMS a-L/a-A ...-g1.h5 file.")
    parser.add_argument("output_dir", type=Path, help="Directory for the outputs.")
    parser.add_argument("--blender", default=None, help="Blender executable.")
    parser.add_argument(
        "--seed-stride",
        type=int,
        default=14,
        help="Lattice spacing of candidate seeds, in grid points.",
    )
    parser.add_argument(
        "--seed-w",
        type=float,
        default=5.0,
        help="Seed only where w exceeds this (m/s): inside the updraft.",
    )
    parser.add_argument(
        "--max-length",
        type=float,
        default=25000.0,
        help="Longest streamline (m) each way from its seed.",
    )
    parser.add_argument(
        "--tube-radius", type=float, default=45.0, help="Tube radius (m)."
    )
    parser.add_argument(
        "--arrow-spacing",
        type=float,
        default=2500.0,
        help="Distance (m) between direction cones along each tube.",
    )
    parser.add_argument(
        "--w-limit",
        type=float,
        default=12.0,
        help="w (m/s) at the ends of the colormap.",
    )
    parser.add_argument(
        "--cloud-thinning",
        type=float,
        default=0.03,
        help="Factor on cloud extinction in the streamline panel.",
    )
    parser.add_argument(
        "--visibility-km",
        type=float,
        default=80.0,
        help="Ground visibility for the aerial-perspective haze (km).",
    )
    add_render_arguments(parser)
    parser.set_defaults(samples=16, width=1440, height=1080)
    return parser.parse_args()


def storm_motion_estimate(winds) -> tuple:
    """
    Mean horizontal wind of the lowest STORM_MOTION_LAYER_TOP_M (m/s).

    Averaged over the whole domain and the layer: a simple, widely used
    proxy for the motion of deep convection. w is taken as zero.
    """
    layer = winds.sel(z=slice(0.0, STORM_MOTION_LAYER_TOP_M))
    return (float(layer["U"].mean()), float(layer["V"].mean()), 0.0)


def main() -> None:
    arguments = parse_arguments()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    # Hydrometeors and cell-centred winds of this output time.
    clouds = load_rams_lite_hydrometeors(
        arguments.data_path, top_height=arguments.top_height
    )
    winds = load_rams_lite_winds(arguments.data_path, top_height=arguments.top_height)
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

    # ---- Right panel: same view, thinned cloud, storm-relative streamlines.
    streamline_config = make_cloud_export_config(arguments).use_view(
        sv.BlenderView.from_bundle(realistic_dir)
    )
    streamline_config.world.aerial_perspective = True
    streamline_config.world.visibility_km = arguments.visibility_km
    streamline_config.overlay_colorbars = False

    storm_motion = storm_motion_estimate(winds)
    # Seeds: lattice points inside the updraft.
    winds["in_updraft"] = winds["W"] > arguments.seed_w
    streamlines = sv.make_streamlines(
        u="U",
        v="V",
        w="W",
        seed_stride=arguments.seed_stride,
        seed_mask_by="in_updraft",
        max_length=arguments.max_length,
        min_length=2000.0,
        # 100 m between exported points: twice the grid spacing, so every
        # resolved bend is kept while the tube geometry stays small.
        point_spacing=100.0,
        frame_velocity=storm_motion,
        color_by="w",
        cmap=STREAMLINE_COLORMAP,
        clim=(-arguments.w_limit, arguments.w_limit),
        tube_radius=arguments.tube_radius,
        arrow_spacing=arguments.arrow_spacing,
        shader="self_lit",
    )
    streamline_dir = arguments.output_dir / "streamlines"
    streamline_scene = sv.Scene().add_clouds(
        clouds, microphysics="rams", extinction_scale=arguments.cloud_thinning
    )
    streamline_scene.add(winds, streamlines)
    streamline_scene.to_blender(
        streamline_dir,
        config=streamline_config,
        render=True,
        blender_executable=arguments.blender,
    )

    # ---- Side by side.
    output_png = (
        arguments.output_dir / f"streamlines_side_by_side_t{minutes:05.1f}min.png"
    )
    compose_side_by_side(
        realistic_dir / "render" / "frame_0001.png",
        streamline_dir / "render" / "frame_0001.png",
        {
            "left_title": "Cloud, physically based",
            "right_title": "Storm-relative streamlines through the updraft",
            "time": f"t = {minutes:.1f} min",
            "right_note": (
                f"Storm motion ({storm_motion[0]:.1f}, {storm_motion[1]:.1f}) m/s "
                f"subtracted\nCloud extinction x {arguments.cloud_thinning:g}; "
                "cones point downstream"
            ),
        },
        ColorbarSpec(
            cmap=STREAMLINE_COLORMAP,
            clim=(-arguments.w_limit, arguments.w_limit),
            tick_values=np.linspace(-arguments.w_limit, arguments.w_limit, 5),
            title="Vertical velocity w (m/s)",
        ),
        output_png,
    )
    print(f"Side-by-side figure: {output_png}")


if __name__ == "__main__":
    main()
