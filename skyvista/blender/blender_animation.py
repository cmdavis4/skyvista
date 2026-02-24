from copy import copy
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional
from skyutils.types_skyutils import PathLike
from skyutils.types_skyutils import BlenderObject
from skyutils.utils import HUMAN_DT_FORMAT, dt_to_str, to_kv_str, to_kv_pairs
from .blender_core import (
    select_object,
    get_object_by_name,
    setup_object,
    update_object_geometry,
    get_collection,
)
from .blender_import import import_data_for_time

import bpy


def register_frame_change_handlers(frame_change_callback: Any) -> None:
    """
    Register frame change handlers for animation.

    Args:
        frame_change_callback: Callback function to execute on frame changes
    """
    # Clear existing handlers
    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_post.clear()
    bpy.app.handlers.render_pre.clear()
    bpy.app.handlers.render_post.clear()
    bpy.app.handlers.render_init.clear()

    # Add handlers for both post, in both viewport and render
    bpy.app.handlers.frame_change_post.append(frame_change_callback)
    # bpy.app.handlers.render_pre.append(frame_change_callback)
    bpy.app.handlers.render_pre.append(frame_change_callback)


def set_f_curve_interpolation(
    obj: BlenderObject, parameter_name: str, interpolation_type: str
) -> None:
    """
    Set the interpolation type for animation curves of a specific parameter.

    Args:
        obj: The Blender object with animation data
        parameter_name: Name of the animated parameter (e.g., "rotation_euler", "field.strength")
        interpolation_type: Type of interpolation ("CONSTANT", "LINEAR", "BEZIER", etc.)

    Returns:
        None

    Note:
        If the object has no animation data or action, the function returns early.
    """
    if not obj.animation_data or not obj.animation_data.action:
        return
    fcurves = [
        fc for fc in obj.animation_data.action.fcurves if fc.data_path == parameter_name
    ]
    for fcurve in fcurves:
        for kf in fcurve.keyframe_points:
            kf.interpolation = interpolation_type


def _animate_object_visibility_helper(
    objects: List[BlenderObject], frame: int, visible: bool
) -> None:
    """
    Helper function to set visibility keyframes for a list of objects at a specific frame.

    Args:
        objects: List of Blender objects to animate
        frame: Frame number to set the keyframes at
        visible: Whether objects should be visible (True) or hidden (False)

    Returns:
        None
    """
    bpy.context.scene.frame_set(frame)
    for object in objects:
        object.hide_viewport = not visible
        object.hide_render = not visible
        object.keyframe_insert(data_path="hide_viewport", frame=frame)
        object.keyframe_insert(data_path="hide_render", frame=frame)


def _animate_show_objects(objects: List[BlenderObject], frame: int) -> None:
    """Show objects at the specified frame."""
    return _animate_object_visibility_helper(objects=objects, frame=frame, visible=True)


def _animate_hide_objects(objects: List[BlenderObject], frame: int) -> None:
    """Hide objects at the specified frame."""
    return _animate_object_visibility_helper(
        objects=objects, frame=frame, visible=False
    )


def calculate_frames_per_timestep(
    simulation_minutes_per_second: float, fps: int, timestep: dt.timedelta
) -> float:
    """
    Calculate how many animation frames each simulation timestep should span.

    Args:
        simulation_minutes_per_second: How many simulation minutes pass per real-time second
        fps: Frames per second of the animation
        timestep: Time interval between simulation timesteps

    Returns:
        Number of frames each timestep should span (minimum 1)

    Example:
        If simulation runs at 10 min/sec, animation is 24 fps, and timestep is 5 minutes:
        frames = 24 * (1/10) * (1/60) * (5*60) = 24 * 0.1 * 0.0167 * 300 = 12 frames
    """
    return (
        fps
        * (1 / simulation_minutes_per_second)
        * (1 / 60)
        * (timestep.total_seconds())
    )


def update_for_new_frame(
    data_directory: PathLike,
    frame_to_time_mapping: List[Any],
    kwargs_data: Dict[str, Any],
    global_scale: float,
    scene: Any,
    depsgraph: Optional[Any] = None,
) -> None:
    """
    Update objects for a new animation frame.

    Args:
        data_directory: Directory containing data files
        frame_to_time_mapping: Mapping from frame numbers to time values
        kwargs_data: Configuration data for object setup
        global_scale: Global scaling factor
        scene: Blender scene object
        depsgraph: Dependency graph for updates
    """
    # Get the originally selected object so we can switch back to it at the end,
    # if possible
    original_selection = bpy.context.active_object
    # Cast to Path
    data_directory = Path(data_directory)
    # Get current frame (Blender frames are 1-indexed)
    current_frame = copy(scene.frame_current)
    # Force set this to trigger a refresh
    # scene.frame_set(current_frame)
    # print(current_frame)

    # Check bounds - frame_to_time_mapping is 0-indexed Python list
    if current_frame < 1 or current_frame > len(frame_to_time_mapping):
        print(
            f"Warning: Frame {current_frame} is out of valid range"
            f" (1-{len(frame_to_time_mapping)})"
        )
        return

    # Convert 1-based frame to 0-based array index
    frame_index = current_frame - 1
    current_time = frame_to_time_mapping[frame_index]

    # Update the date text
    dt_str = dt_to_str(current_time, date_format=HUMAN_DT_FORMAT)
    print(f"Setting date string to {dt_str}")
    get_object_by_name("date_text").data.body = dt_str

    # Get the data files for this time
    # Don't want to apply any properties like materials or geometry nodes,
    # so pass a dict with the same keys as the real kwargs_data (to filter mesh
    # categories in the same way) but with empty dicts for values
    this_time_data = import_data_for_time(
        data_directory=data_directory,
        frame_time=current_time,
        kwargs_data={k: {} for k in kwargs_data.keys()},
        global_scale=global_scale,
    )

    # Update mesh geometries using the new meshes
    for new_obj_name, new_obj in this_time_data.items():
        # Get the name of the time-independent version of this object
        permanent_obj_name = to_kv_str({
            "category": to_kv_pairs(new_obj_name)["category"],
            "varname": to_kv_pairs(new_obj_name)["varname"],
        })
        try:
            # Get the existing permanent mesh if it exists
            permanent_obj = get_object_by_name(permanent_obj_name)
            # If this succeeds, we already have an object for this varname
        except ValueError:
            print(f"Adding {permanent_obj_name}")
            # Mesh doesn't exist, all we needed to do was import it
            permanent_obj = new_obj
            # Set this object up
            setup_object(new_obj, kwargs_obj=kwargs_data[permanent_obj["category"]])
            # Set the name, since it's already been added to the appropriate collections
            permanent_obj.name = permanent_obj_name
        else:
            # Update the data
            update_object_geometry(
                obj=permanent_obj, new_mesh=new_obj, depsgraph=depsgraph
            )
        print(f"Added permanent_obj.name")
    # Also clear any meshes that shouldn't currently be visible
    # for obj in get_collection("data").objects:
    #     if obj.name not in this_time_data:
    #         bpy.data.objects.remove(obj, do_unlink=True)
    # Restore original selection if possible
    try:
        if original_selection and original_selection.name in [
            obj.name for obj in bpy.data.objects
        ]:
            select_object(original_selection, _deselect_all=True)
    except ReferenceError:
        pass

    # Not sure why but the animation end_frame seems to get overwritten by something
    # here, so we just set the animation length here since it doesn't really
    # affect anything else
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = len(frame_to_time_mapping)


def animate_objects_visibility(
    objects: List[BlenderObject], start_frame: int, end_frame: int
) -> None:
    """
    Animate visibility for a group of objects during a specific frame range.

    Objects are hidden before the start frame, visible during the range,
    and hidden again after the end frame.

    Args:
        objects: List of Blender objects to animate
        start_frame: First frame where objects should be visible
        end_frame: Last frame where objects should be visible

    Returns:
        None
    """
    # Hide before start
    if start_frame > 1:
        _animate_hide_objects(objects=objects, frame=start_frame - 1)

    # Show during active period
    _animate_show_objects(objects=objects, frame=start_frame)

    # Keep visible until end
    _animate_show_objects(objects=objects, frame=end_frame)

    # Hide after end
    if end_frame < bpy.context.scene.frame_end:
        _animate_hide_objects(objects=objects, frame=end_frame + 1)


def setup_visibility_keyframes(
    all_timestep_objects: Dict[Any, Dict[str, BlenderObject]],
    frame_to_time_mapping: List[Any],
) -> None:
    """
    Set up visibility keyframes for all imported timestep objects.

    This function creates keyframes to show/hide objects based on their timestamp,
    enabling pure keyframe-based animation without frame change callbacks.

    Args:
        all_timestep_objects: Dictionary mapping timestamps to their imported objects
        frame_to_time_mapping: List mapping frame numbers (index+1) to timestamps

    Returns:
        None

    Note:
        - Objects are visible only during frames matching their timestamp
        - Uses CONSTANT interpolation for instant show/hide
        - All keyframes are created upfront during setup
    """
    print("Setting up visibility keyframes for all timesteps...")

    # Build reverse mapping: timestamp -> list of frame numbers
    time_to_frames: Dict[Any, List[int]] = {}
    for frame_idx, time in enumerate(frame_to_time_mapping):
        frame_num = frame_idx + 1  # Blender uses 1-indexed frames
        if time not in time_to_frames:
            time_to_frames[time] = []
        time_to_frames[time].append(frame_num)

    # Animate visibility for each timestep's objects
    total_objects = sum(len(objs) for objs in all_timestep_objects.values())
    processed = 0

    for time, objects_dict in all_timestep_objects.items():
        if time not in time_to_frames:
            print(f"Warning: Timestep {time} has no corresponding frames")
            continue

        frames = time_to_frames[time]
        start_frame = min(frames)
        end_frame = max(frames)

        # Get all objects for this timestep
        objects = list(objects_dict.values())

        # Animate their visibility
        animate_objects_visibility(
            objects=objects,
            start_frame=start_frame,
            end_frame=end_frame,
        )

        # Set interpolation to CONSTANT for instant visibility changes
        for obj in objects:
            set_f_curve_interpolation(obj, "hide_viewport", "CONSTANT")
            set_f_curve_interpolation(obj, "hide_render", "CONSTANT")

        processed += len(objects)
        print(f"  Set keyframes for timestep {time}: {len(objects)} objects (frames {start_frame}-{end_frame})")

    print(f"Successfully set up visibility keyframes for {processed}/{total_objects} objects")
