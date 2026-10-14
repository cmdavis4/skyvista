"""
Render one RAMS output time as a physically based cloud image in Blender.

Example (from the repository root, in the pixi ``blender`` environment)::

    pixi run -e blender python examples/rams_clouds/render_rams_cloud_still.py \\
        /path/to/output/a-L-2000-01-01-013000-g1.h5  storm_bundle \\
        --blender ~/programs/blender-5.0.1-linux-x64/blender

The bundle directory receives the VDB volumes, the ``.blend`` file and
``render/frame_0001.png``. A 1920x1080, 128-sample frame of a mature storm on
a 400 x 400 x 330 grid takes ~9 min on 64 CPU cores.
"""

import argparse
from pathlib import Path

from cloud_render_config import add_render_arguments, make_cloud_export_config
from rams_lite_loader import load_rams_lite_hydrometeors

import skyvista as sv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_path", type=Path, help="RAMS a-L/a-A ...-g1.h5 file.")
    parser.add_argument("bundle_dir", type=Path, help="Output bundle directory.")
    parser.add_argument(
        "--blender",
        default=None,
        help="Blender executable (default: $SKYVISTA_BLENDER, then PATH).",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Only write the bundle; do not run Blender.",
    )
    add_render_arguments(parser)
    arguments = parser.parse_args()

    # Hydrometeors of this output time, capped below the stratosphere.
    dataset = load_rams_lite_hydrometeors(
        arguments.data_path, top_height=arguments.top_height
    )

    # Export the clouds and, unless asked not to, build and render in Blender.
    config = make_cloud_export_config(arguments)
    sv.Scene().add_clouds(dataset, microphysics="rams").to_blender(
        arguments.bundle_dir,
        config=config,
        render=not arguments.no_render,
        blender_executable=arguments.blender,
    )
    print(f"Bundle written to {arguments.bundle_dir}")


if __name__ == "__main__":
    main()
