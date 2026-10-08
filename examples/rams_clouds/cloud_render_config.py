"""
Shared Blender export settings for the RAMS cloud examples.

Both the still and the animation scripts start from
``BlenderExportConfig.for_clouds()`` (physical sky, ground plane, 128 volume
bounces, auto exposure) and override only the framing and render quality, so
an animation frame looks exactly like the still rendered with the same
arguments.
"""

import argparse
from typing import Optional, Tuple

import skyvista as sv


def add_render_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the framing/quality command-line options shared by both scripts."""
    parser.add_argument("--width", type=int, default=1920, help="Image width (px).")
    parser.add_argument("--height", type=int, default=1080, help="Image height (px).")
    parser.add_argument(
        "--samples", type=int, default=128, help="Cycles samples per pixel."
    )
    parser.add_argument(
        "--sun-elevation", type=float, default=30.0, help="Sun elevation (deg)."
    )
    parser.add_argument(
        "--sun-azimuth",
        type=float,
        default=15.0,
        help="Direction toward the sun, CCW from +x (deg).",
    )
    parser.add_argument(
        "--camera-azimuth",
        type=float,
        default=-55.0,
        help="Direction from the scene center to the camera, CCW from +x (deg).",
    )
    parser.add_argument(
        "--camera-elevation",
        type=float,
        default=-8.0,
        help=(
            "Camera elevation above the scene center (deg). Negative values"
            " put the camera low, looking up at the storm."
        ),
    )
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=1.1,
        help="Camera distance in bounding-box diagonals.",
    )
    parser.add_argument(
        "--top-height",
        type=float,
        default=16500.0,
        help="Ignore model levels above this height (m).",
    )


def make_cloud_export_config(
    arguments: argparse.Namespace,
    origin_shift: Optional[Tuple[float, float, float]] = None,
) -> sv.BlenderExportConfig:
    """
    Build the export config from parsed command-line arguments.

    Args:
        arguments: Parsed options from :func:`add_render_arguments`.
        origin_shift: Fixed data point mapped to the Blender origin. An
            animation must pass one, so every frame shares the same scene
            transform; a still can leave it to the automatic choice.
    """
    config = sv.BlenderExportConfig.for_clouds()
    # Render quality.
    config.render.resolution = (arguments.width, arguments.height)
    config.render.samples = arguments.samples
    # Lighting: the physical sky's sun position.
    config.world.sun_elevation_deg = arguments.sun_elevation
    config.world.sun_azimuth_deg = arguments.sun_azimuth
    # Framing: a static camera around the data's bounding box.
    config.camera.azimuth_deg = arguments.camera_azimuth
    config.camera.elevation_deg = arguments.camera_elevation
    config.camera.distance_factor = arguments.camera_distance
    if origin_shift is not None:
        config.transform.origin_shift = origin_shift
    return config
