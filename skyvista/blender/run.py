from .animation import (
    calculate_frames_per_timestep,
    register_frame_change_handlers,
    update_for_new_frame,
)
from .camera import setup_camera
from .import_ import get_data_filepaths, import_assets_library
from .config import DEFAULT_BLENDER_CONFIG
from .core import (
    add_object_to_scene,
    create_collection,
    create_grass_field,
    deselect_all,
    get_object_by_name,
    position_text_for_camera,
    reset_scene,
    set_sky_background,
)
from .render import setup_render_settings
from .import_ import BPY_IMPORT_FUNCTIONS_BY_FILEPATH_SUFFIX
from ..types_ import ConfigDict, PathLike
from skyutils.utils import raise_if_not_evenly_spaced_, to_kv_pairs


from copy import copy
from pathlib import Path
from typing import Any, Dict, Tuple

import bpy
from fractions import Fraction


def run_atmospheric_animation(
    data_dir: PathLike,
    output_dir: PathLike,
    kwargs_data: Dict[str, Dict[str, Any]],
    config: ConfigDict = {},
    use_grass=True,
    simulation_minutes_per_second: float = 5,
    fps: int = 24,
    render: bool = False,
    assets_libraries: list[str] = [],
    global_scale: float | Tuple[float, float, float] = 0.001,
    limit: int | None = None,
    use_time_stretching: bool = False,
    output_format="PNG",
) -> None:
    """
    Main function to set up and run complete atmospheric visualization in .

    This is the primary entry point that orchestrates the entire atmospheric
    animation pipeline: importing meshes, setting up animations, creating grass
    fields with wind forces, configuring lighting and rendering.

    Args:
        ply_directory: Directory containing PLY mesh files organized by timestamp
        output_dir: Directory where rendered frames and wind data should be stored
        kwargs_data: Dictionary mapping category names to their properties (scale, location, etc.)
        config: Configuration dictionary for render settings (merged with defaults)
        simulation_minutes_per_second: Simulation time rate for animation timing
        fps: Frames per second for the animation
        render: Whether to automatically start rendering after setup

    Returns:
        None

    Side Effects:
        - Clears existing Blender scene
        - Imports PLY meshes and creates collections
        - Sets up mesh visibility animations
        - Creates grass field with wind force animations
        - Configures camera, lighting, and render settings
        - Optionally starts animation rendering

    Example:
        kwargs_data = {
            "Rcondensate": {"scale": (0.0002, 0.0002, 0.0002), "location": (0.0, 0.0, 5.0)},
            "thetadeficit": {"scale": (0.0002, 0.0002, 0.001), "location": (0.0, 0.0, 0.1)},
        }
        run_atmospheric_animation(
            ply_directory="/path/to/meshes",
            output_dir="/path/to/output",
            kwargs_data=kwargs_data,
            config={"resolution_x": 1920, "resolution_y": 1080},
            render=False  # Setup only, don't render
        )
    """
    # Make a global for our frame change callback so it doesn't get garbage collected

    accepted_output_formats = ["PNG", "FFMPEG"]
    if output_format not in accepted_output_formats:
        raise ValueError(
            f"{output_format} not one of accepted output format"
            f" {accepted_output_formats}"
        )
    # Reset the scene
    reset_scene()

    # Set the render output type
    bpy.context.scene.render.image_settings.file_format = output_format
    # Need to set the container type if it's ffmpeg
    if output_format == "FFMPEG":
        bpy.context.scene.render.ffmpeg.format = "MPEG4"

    if config:
        # Use our default settings except for what was passed
        passed_config = copy(config)
        config = copy(DEFAULT_BLENDER_CONFIG)
        config.update(passed_config)
    else:
        config = copy(DEFAULT_BLENDER_CONFIG)

    # Convert paths to Paths
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # Handle global scale
    if isinstance(global_scale, float):
        global_scale = (
            global_scale,
            global_scale,
            global_scale,
        )

    print("=== Atmospheric Model Animation in Blender ===")
    print(f"PLY Directory: {data_dir}")
    # print(f"Wind Data: {WIND_DATA_CSV}")
    print(f"Output directory: {output_dir}")

    # Load asset libraries
    for assets_library_name in assets_libraries:
        import_assets_library(assets_library_name=assets_library_name)

    # Set up camera and rendering
    print("Configuring camera and render settings...")
    output_dir.mkdir(exist_ok=True, parents=False)
    bpy.context.scene.render.engine = config["render_engine"]
    camera_obj = setup_camera()
    setup_render_settings(
        output_dir=output_dir,
        engine=config.get("render_engine", "BLENDER_EEVEE_NEXT"),
        resolution=(config.get("resolution_x", 1920), config.get("resolution_y", 1080)),
        quality=config.get("quality", "MEDIUM"),
    )

    # Convert animation_times from string to dt.datetime if needed
    animation_times = sorted(
        list(
            set([
                (to_kv_pairs(fpath, parse_datetimes=True)["dt"])
                for fpath in get_data_filepaths(data_dir)
            ])
        )
    )
    if not animation_times:
        raise ValueError("No data files found")
    # Make a flag for whether this is an animation
    animation = len(animation_times) > 1
    if animation:
        # Make sure these are evenly spaced
        raise_if_not_evenly_spaced_(animation_times)
        # Limit if we want that
        if limit:
            animation_times = animation_times[:limit]

        # Calculate frames per timestep based on simulation parameters
        animation_times_timestep = animation_times[1] - animation_times[0]
        # FIXME: As a workaround we set frames per timestep to 1 and change the FPS
        # to make the animation the right length
        fps = max(
            1,
            int(
                (simulation_minutes_per_second * 60)
                / animation_times_timestep.total_seconds()
            ),
        )
        bpy.context.scene.render.fps = fps
        frames_per_timestep = 1.0
        # frames_per_timestep = calculate_frames_per_timestep(
        #     simulation_minutes_per_second, fps, animation_times_timestep
        # )
    else:
        # Doesn't matter
        frames_per_timestep = 1.0

    # Since we're pretty much always in the realm of a timestep lasting more than
    # a single frame (at least for a simulation time per wall second of 5 minutes),
    # can optionally use time stretching to speed up the render, where each frame
    # is only rendered once but we use time stretching to get to the right simulation
    # time per wall second. This won't allow e.g. grass to move between updates
    # to the actual geometry of the data-based objects in the scene, but is much
    # faster
    frames_per_timestep_int = max(int(frames_per_timestep), 1)
    if animation and use_time_stretching:
        # In this case, we enforce that each timestep gets one frame (as far as
        # blender is concerned)
        # (Can still use this if frames_per_timestep is < 1, just to slow things
        # down without adjusting the fps)
        frames_per_timestep_int = 1
        # Set time stretching parameters to get to correct simulation_minutes_per_second
        # These both have to be integers, so we get to do some fun math
        # Calculate the smallest integers that maintain the ratio
        ratio = (
            Fraction(frames_per_timestep_int).limit_denominator()
            / Fraction(frames_per_timestep).limit_denominator()
        )
        # Get the reduced fraction
        reduced_ratio = ratio.limit_denominator()
        frame_map_old = reduced_ratio.numerator
        frame_map_new = reduced_ratio.denominator
        print(f"Simulation minutes per second: {simulation_minutes_per_second}")
        print(f"Calculated frames per timestep: {frames_per_timestep}")
        print(f"Integer frames per timestep: {frames_per_timestep_int}")
        print(f"Frame map old (numerator): {frame_map_old}")
        print(f"Frame map new (denominator): {frame_map_new}")
        bpy.context.scene.render.frame_map_old = frame_map_old
        # Still has to be an integer so we'll do the best we can
        bpy.context.scene.render.frame_map_new = frame_map_new

    # - Index = frame number (0-based, but Blender uses 1-based)
    # - Value = simulation time to show in that frame
    frame_to_time_mapping = []
    for animation_time in animation_times:
        # Each simulation time gets frames_per_timestep consecutive frames
        for _ in range(frames_per_timestep_int):
            frame_to_time_mapping.append(animation_time)
    print(
        f"Created frame mapping: {len(frame_to_time_mapping)} frames covering"
        f" {len(animation_times)} simulation times"
    )
    print(f"Each simulation timestep shown for {frames_per_timestep} frames")

    # Add text showing the datetime
    date_text_obj = get_object_by_name("date_text")
    add_object_to_scene(date_text_obj)
    # Position it in front of the camera
    position_text_for_camera(
        text_obj=date_text_obj,
        camera_obj=camera_obj,
        # For whatever reason these are the offsets that work
        camera_axis_distance=40,  # m
        horizontal_offset=-6,  # m
        vertical_offset=6,  # m
    )

    # Add a plys collection for the objects we import from plys
    data_collection = create_collection("data")
    for file_type in BPY_IMPORT_FUNCTIONS_BY_FILEPATH_SUFFIX:
        create_collection(file_type[1:] + "s", parent_collection=data_collection)

    @bpy.app.handlers.persistent
    def callback_function(scene, depsgraph=None):
        return update_for_new_frame(
            data_directory=data_dir,
            frame_to_time_mapping=frame_to_time_mapping,
            kwargs_data=kwargs_data,
            global_scale=global_scale,
            scene=scene,
            depsgraph=depsgraph,
        )

    register_frame_change_handlers(frame_change_callback=callback_function)
    # Initialize the scene
    bpy.context.scene.frame_set(1)

    # Setup grass wind animation
    if use_grass:
        create_grass_field()

    # Add background from existing asset
    set_sky_background("sun")

    # Deselect all
    deselect_all()

    print("=== Setup Complete ===")
    print(
        f"Animation: {bpy.context.scene.frame_start} to"
        f" {bpy.context.scene.frame_end} frames"
    )
    print(f"Render Engine: {config['render_engine']}")
    print(
        "Render Resolution:"
        f" {config.get('resolution_x', 1920)}x{config.get('resolution_y', 1080)}"
    )
    print(f"Render output: {output_dir}")
    print("\\nReady to render!")
    print("Use: bpy.ops.render.render(animation=True)")

    # Optionally start rendering
    if render:
        bpy.ops.render.render(animation=True)
