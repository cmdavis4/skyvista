"""
Animate storm-relative streamlines through a RAMS updraft: the storm is held
at one output time while cones move along the streamlines with the air.

Example (from the repository root, in the pixi ``blender`` environment)::

    pixi run -e blender python examples/rams_clouds/animate_rams_streamlines.py \\
        /path/to/output/a-L-2000-01-01-013420-g1.h5  streamline_animation \\
        --blender ~/programs/blender-5.0.1-linux-x64/blender --ffmpeg /path/to/ffmpeg

The scene is the right panel of ``render_rams_streamlines_side_by_side.py``
(thinned cloud, storm-relative streamlines seeded in the updraft, aerial
perspective) in the same camera view. Only the direction cones move.

How the cones move
------------------
Each cone sits at a fixed *travel time* from its streamline's seed: the
time air moving with the storm-relative wind takes to get there. Cones are
``--arrow-travel-time`` apart, so they bunch up where the air is slow and
spread out where it is fast. A flow clock (``sv.BlenderHoldConfig``) advances
every video frame and every cone moves downstream by as far as the air moves
in that time, so the cones show the speed of the flow as well as its path.

Rendering one loop is enough
----------------------------
After the clock advances one marker travel time, every cone has moved to
where its downstream neighbour was, so the picture repeats exactly. The
script renders ``--frames-per-loop`` frames covering one travel time, then
plays that loop ``--loops`` times in the video. One loop lasts
``frames-per-loop / fps`` seconds of video, so the motion is sped up by
``arrow-travel-time * fps / frames-per-loop`` relative to the real flow.
"""

import argparse
from pathlib import Path

import numpy as np
from cloud_render_config import add_render_arguments, make_cloud_export_config
from rams_lite_loader import load_rams_lite_hydrometeors, load_rams_lite_winds
from render_rams_streamlines_side_by_side import (
    STREAMLINE_COLORMAP,
    storm_motion_estimate,
)

import skyvista as sv

# Text heights, as fractions of the frame height.
TITLE_SIZE, LABEL_SIZE = 0.031, 0.024


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_path", type=Path, help="RAMS a-L/a-A ...-g1.h5 file.")
    parser.add_argument("output_dir", type=Path, help="Directory for the outputs.")
    parser.add_argument("--blender", default=None, help="Blender executable.")
    parser.add_argument("--ffmpeg", default=None, help="ffmpeg executable.")
    parser.add_argument(
        "--seed-stride",
        type=int,
        default=24,
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
        "--arrow-travel-time",
        type=float,
        default=400.0,
        help=(
            "Travel time (s) of the air between neighbouring cones. Storm-"
            "relative speeds along these streamlines are mostly 3-5 m/s, so"
            " the default spaces cones about 2 km apart (6 km at 15 m/s)."
        ),
    )
    parser.add_argument(
        "--frames-per-loop",
        type=int,
        default=48,
        help="Video frames over which the clock advances one travel time.",
    )
    parser.add_argument("--fps", type=float, default=24.0, help="Video frame rate.")
    parser.add_argument(
        "--loops", type=int, default=5, help="Times the loop plays in the video."
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
        help="Factor on cloud extinction.",
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


def side_by_side_view(arguments, clouds) -> sv.BlenderView:
    """
    The camera and scene origin of the side-by-side figure.

    That figure frames the physically based cloud automatically and pins its
    streamline panel to the same view. The cloud alone, with the same
    settings, therefore gives the same view here.
    """
    cloud_scene = sv.Scene().add_clouds(clouds, microphysics="rams")
    return cloud_scene.blender_view(make_cloud_export_config(arguments))


def make_figure(arguments, minutes: float, storm_motion: tuple) -> sv.compose.Figure:
    """Each frame's labels and colorbar, as on the side-by-side figure."""
    speed_up = arguments.arrow_travel_time * arguments.fps / arguments.frames_per_loop
    return sv.compose.Figure(
        panels=[
            sv.compose.Panel(
                lambda frame: frame.render_path,
                [
                    sv.compose.Text(
                        "Storm-relative streamlines through the updraft",
                        size=TITLE_SIZE,
                    ),
                    sv.compose.Text(
                        f"t = {minutes:.1f} min (held)",
                        size=LABEL_SIZE,
                        offset=(0.0, -0.045),
                    ),
                    sv.compose.Text(
                        (
                            f"Storm motion ({storm_motion[0]:.1f}, "
                            f"{storm_motion[1]:.1f}) m/s subtracted\n"
                            f"Cones move with the air, {speed_up:.0f}x real time\n"
                            "Cone spacing: "
                            f"{arguments.arrow_travel_time:.0f} s of travel"
                        ),
                        anchor="lower left",
                        size=LABEL_SIZE,
                    ),
                    sv.compose.Colorbar(
                        cmap=STREAMLINE_COLORMAP,
                        clim=(-arguments.w_limit, arguments.w_limit),
                        label="Vertical velocity w (m/s)",
                        ticks=list(
                            np.linspace(-arguments.w_limit, arguments.w_limit, 5)
                        ),
                        anchor="lower right",
                        length=0.3,
                        thickness=0.025,
                        margin=0.07,
                        text_size=LABEL_SIZE,
                    ),
                ],
            )
        ]
    )


def main() -> None:
    arguments = parse_arguments()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    # Hydrometeors and cell-centred winds of the held output time.
    clouds = load_rams_lite_hydrometeors(
        arguments.data_path, top_height=arguments.top_height
    )
    winds = load_rams_lite_winds(arguments.data_path, top_height=arguments.top_height)
    minutes = float(clouds["time"]) / 60.0

    # ---- The side-by-side figure's view and look.
    config = make_cloud_export_config(arguments).use_view(
        side_by_side_view(arguments, clouds)
    )
    config.world.aerial_perspective = True
    config.world.visibility_km = arguments.visibility_km
    config.overlay_colorbars = False
    config.fps = arguments.fps
    # ---- The hold: one loop, the clock advancing one travel time over it.
    config.hold = sv.BlenderHoldConfig(
        n_frames=arguments.frames_per_loop,
        clock_seconds_per_frame=arguments.arrow_travel_time / arguments.frames_per_loop,
    )

    # ---- Thinned cloud and storm-relative streamlines seeded in the updraft.
    storm_motion = storm_motion_estimate(winds)
    winds["in_updraft"] = winds["W"] > arguments.seed_w
    streamlines = sv.make_streamlines(
        u="U",
        v="V",
        w="W",
        seed_stride=arguments.seed_stride,
        seed_mask_by="in_updraft",
        max_length=arguments.max_length,
        min_length=2000.0,
        point_spacing=100.0,
        frame_velocity=storm_motion,
        color_by="w",
        cmap=STREAMLINE_COLORMAP,
        clim=(-arguments.w_limit, arguments.w_limit),
        tube_radius=arguments.tube_radius,
        arrow_travel_time=arguments.arrow_travel_time,
        shader="self_lit",
    )
    scene = sv.Scene().add_clouds(
        clouds, microphysics="rams", extinction_scale=arguments.cloud_thinning
    )
    scene.add(winds, streamlines)
    bundle_dir = arguments.output_dir / "bundle"
    scene.to_blender(
        bundle_dir, config=config, render=True, blender_executable=arguments.blender
    )

    # ---- Labels on every frame, then the loop played --loops times.
    figure = make_figure(arguments, minutes, storm_motion)
    composed_dir = arguments.output_dir / "composed"
    composed_dir.mkdir(exist_ok=True)
    for frame_index in range(arguments.frames_per_loop):
        frame_number = config.frame_start + frame_index
        figure.render(
            composed_dir / f"frame_{frame_index + 1:04d}.png",
            sv.compose.FrameContext(
                index=frame_index,
                n_frames=arguments.frames_per_loop,
                render_path=bundle_dir / "render" / f"frame_{frame_number:04d}.png",
            ),
        )
    sv.encode_video(
        composed_dir / "frame_%04d.png",
        arguments.output_dir / f"{arguments.output_dir.name}.mp4",
        fps=arguments.fps,
        repeat=arguments.loops,
        crf=16,
        ffmpeg=arguments.ffmpeg,
    )


if __name__ == "__main__":
    main()
