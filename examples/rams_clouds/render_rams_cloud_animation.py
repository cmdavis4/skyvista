"""
Render a time-lapse of RAMS clouds with the skyvista Blender backend.

Example (from the repository root, in the pixi ``blender`` environment)::

    pixi run -e blender python examples/rams_clouds/render_rams_cloud_animation.py \\
        /path/to/case/output  storm_animation \\
        --start-minute 42 --end-minute 120 --cadence-seconds 40 \\
        --samples 64 --blender ~/programs/blender-5.0.1-linux-x64/blender

How it works
------------
The pipeline is :class:`skyvista.BlenderAnimation`; this script supplies the
RAMS-specific parts:

1. **Frames.** One frame per lite file at the requested cadence.
   ``build_frame`` reads one file and returns a scene with its clouds.
2. **Fixed coordinates.** Every frame maps the domain center (at the ground)
   to the Blender origin, so all frames share one coordinate system.
3. **Static camera.** The camera frames the largest cloud of the run, so the
   mature storm fits in view throughout (frames without cloud are ignored).
4. **Stamp and encode.** Each frame is stamped with its model time and the
   frames are encoded to ``<bundle>/<bundle name>.mp4`` if an ffmpeg is
   available (``--ffmpeg``, ``$SKYVISTA_FFMPEG``, PATH, or ``imageio-ffmpeg``).

Every stage skips work that is already on disk, so the command can simply be
rerun after an interruption.
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
from cloud_render_config import add_render_arguments, make_cloud_export_config
from rams_lite_loader import (
    load_rams_lite_hydrometeors,
    model_time_seconds,
    read_header_coordinates,
)

import skyvista as sv


def find_output_files(
    output_dir: Path, start_minute: float, end_minute: float, cadence_seconds: int
) -> List[Path]:
    """
    The lite files at the requested times, in time order.

    Lists every ``a-L-*-g1.h5`` file once, keys it by model time, then picks
    the times start, start + cadence, ... up to end. A requested time with no
    file is an error, since a gap would silently speed up the time-lapse.
    """
    files_by_time = {
        model_time_seconds(path): path for path in output_dir.glob("a-L-*-g1.h5")
    }
    requested_times = np.arange(
        start_minute * 60.0, end_minute * 60.0 + 0.5, cadence_seconds
    )
    missing_times = [t for t in requested_times if float(t) not in files_by_time]
    if missing_times:
        raise FileNotFoundError(
            f"No lite file at model times (s): {missing_times[:10]}"
            f"{' ...' if len(missing_times) > 10 else ''}"
        )
    return [files_by_time[float(t)] for t in requested_times]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output_dir", type=Path, help="RAMS case output directory.")
    parser.add_argument("bundle_dir", type=Path, help="Working/output directory.")
    parser.add_argument("--start-minute", type=float, required=True)
    parser.add_argument("--end-minute", type=float, required=True)
    parser.add_argument(
        "--cadence-seconds",
        type=int,
        default=60,
        help="Model time between frames (a multiple of the lite output interval).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help=(
            "Export processes. Each needs ~10-20 GB for a 400x400x330 grid."
            " More is not faster: with 24 workers on a 64-core node, kernel"
            " (system) CPU time reached ~37%% and each frame took ~25 min,"
            " versus ~45 s each with 8 workers."
        ),
    )
    parser.add_argument(
        "--render-processes",
        type=int,
        default=1,
        help="Blender processes rendering in parallel.",
    )
    parser.add_argument("--fps", type=float, default=15.0, help="Video frame rate.")
    parser.add_argument("--blender", default=None, help="Blender executable.")
    parser.add_argument("--ffmpeg", default=None, help="ffmpeg executable.")
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Stop after assembling the bundle.",
    )
    add_render_arguments(parser)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    data_paths = find_output_files(
        arguments.output_dir,
        arguments.start_minute,
        arguments.end_minute,
        arguments.cadence_seconds,
    )
    paths_by_time = {model_time_seconds(path): path for path in data_paths}
    times = list(paths_by_time)
    print(
        f"{len(times)} frames from {times[0] / 60:.1f} to {times[-1] / 60:.1f} min"
        f" every {arguments.cadence_seconds} s"
    )

    # Fixed transform for every frame: domain center horizontally, ground at 0.
    coordinates = read_header_coordinates(
        data_paths[0].with_name(data_paths[0].name.replace("-g1.h5", "-head.txt"))
    )
    origin_shift = (
        float(np.mean(coordinates["x"])),
        float(np.mean(coordinates["y"])),
        0.0,
    )

    def build_frame(time_seconds: float) -> sv.Scene:
        """The clouds of the lite file at one model time."""
        dataset = load_rams_lite_hydrometeors(
            paths_by_time[time_seconds], top_height=arguments.top_height
        )
        return sv.Scene().add_clouds(dataset, microphysics="rams")

    animation = sv.BlenderAnimation(
        arguments.bundle_dir,
        times,
        build_frame,
        config=make_cloud_export_config(arguments, origin_shift=origin_shift),
        fps=arguments.fps,
    )
    animation.export(workers=arguments.workers)
    animation.assemble()
    if arguments.skip_render:
        print("Bundle assembled at", animation.bundle_dir)
        return
    animation.render(
        blender_executable=arguments.blender, processes=arguments.render_processes
    )

    # Stamp each frame with its model time, then encode.
    figure = sv.compose.Figure(
        panels=[
            sv.compose.Panel(
                lambda frame: frame.render_path,
                [
                    sv.compose.TimeStamp(
                        formatter=lambda seconds: f"t = {seconds / 60.0:5.1f} min"
                    )
                ],
            )
        ]
    )
    animation.compose(figure)
    animation.encode(
        arguments.bundle_dir / f"{arguments.bundle_dir.name}.mp4",
        crf=16,
        ffmpeg=arguments.ffmpeg,
    )


if __name__ == "__main__":
    main()
