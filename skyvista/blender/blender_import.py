import csv
import datetime as dt
from typing import Any, Dict, List, Tuple
from skyutils.types_skyutils import BlenderObject
from pathlib import Path

from .blender_core import (
    uvw_to_magnitude_and_rotation,
)
from skyutils.utils import dt_to_str, to_kv_pairs
from skyutils.types_skyutils import PathLike
from .blender_core import DATA_FILE_SUFFIXES, resize_object

import bpy

# Disable the overwrite scene setting in SciBlend
try:
    bpy.context.scene.x3d_import_settings.overwrite_scene = False
except:
    print("Failed setting SciBlend overwrite scene setting to False")


def get_data_filepaths(directory):
    directory = Path(directory)
    return [
        x for x in directory.iterdir() if x.is_file() and x.suffix in DATA_FILE_SUFFIXES
    ]


def to_data_type_collection_name(fpath):
    fpath = Path(fpath)
    if fpath.suffix not in DATA_FILE_SUFFIXES:
        raise ValueError(
            f"Suffix {fpath.suffix} is not one of the known data                       "
            f"   file types {DATA_FILE_SUFFIXES}"
        )
    return fpath.suffix[1:] + "s"  # Drop the .


def import_assets_library(assets_library_name: str):
    this_lib_index = bpy.context.preferences.filepaths.asset_libraries.find(
        assets_library_name
    )
    if this_lib_index == -1:
        raise ValueError(f"Library {assets_library_name} not found")

    this_asset_lib = bpy.context.preferences.filepaths.asset_libraries[this_lib_index]
    this_asset_lib_dir = this_asset_lib.path
    assets_blend_path = Path(this_asset_lib_dir) / f"{assets_library_name}_assets.blend"
    if not assets_blend_path.exists():
        raise ValueError(f"Asset library blend file {assets_blend_path} not found")
    assets_blend_path = str(assets_blend_path)

    # Append everything.
    with bpy.data.libraries.load(assets_blend_path, assets_only=True, link=False) as (
        data_from,
        data_to,
    ):
        for attr in dir(data_to):
            setattr(data_to, attr, getattr(data_from, attr))
    return assets_blend_path


def _import_vtk_helper(filepath):
    filepath = Path(filepath)
    return bpy.ops.import_vtk.animation(
        directory=str(filepath.parent),
        files=[{
            "name": filepath.name,
        }],
    )


BPY_IMPORT_FUNCTIONS_BY_FILEPATH_SUFFIX = {
    ".ply": (
        bpy.ops.wm.ply_import
        if bpy.app.version >= (4, 0, 0)
        else bpy.ops.import_mesh.ply
    ),
    ".vdb": bpy.ops.object.volume_import,
    ".obj": bpy.ops.wm.obj_import,
    ".vtk": _import_vtk_helper,
    ".vtp": _import_vtk_helper,
}


def import_mesh(filepath: PathLike, **kwargs):
    filepath = Path(filepath)
    fn = BPY_IMPORT_FUNCTIONS_BY_FILEPATH_SUFFIX[filepath.suffix]
    return fn(filepath=str(filepath), **kwargs)


def import_single_data_file(
    filepath: PathLike,
) -> BlenderObject:
    """
    Import a single PLY file into Blender and configure it for animation.

    This function imports a PLY mesh file, renames it based on the filename,
    assigns it to the appropriate collection, applies properties (scale, location, etc.),
    and sets up initial visibility state for animation.

    Args:
        file_path: Absolute path to the PLY file to import
        mesh_properties: Dictionary of properties to apply to the mesh (scale, location, etc.)
        collections: Dictionary mapping category names to Blender collections

    Returns:
        The imported Blender object, or None if import failed

    Raises:
        ValueError: If multiple objects are imported (expected only one)
        Exception: Re-raises any import errors that occur
    """
    # Cast to Path
    filepath = Path(filepath)
    print(f"Importing {filepath}")

    # Get initially selected objects to track what's new after import
    initial_objects = set(bpy.context.scene.objects)

    try:
        BPY_IMPORT_FUNCTIONS_BY_FILEPATH_SUFFIX[filepath.suffix](filepath=str(filepath))

        # Get objects in the scene that are not in initially_selected_objects
        new_objects = [
            obj for obj in bpy.context.scene.objects if obj not in initial_objects
        ]
        if not new_objects:
            print(f"Warning: No object imported from {filepath}")
            return None
        if len(new_objects) > 1:
            raise ValueError("Multiple new objects imported, should only be one")
        this_object = new_objects[0]

        # Add custom metadata to see what kind of object it is
        this_object["data_file_suffix"] = filepath.suffix
        this_object["category"] = to_kv_pairs(filepath)["category"]
        this_object["varname"] = to_kv_pairs(filepath)["varname"]
        # Set object name
        this_object.name = filepath.stem
        # If it's a .vtk file we need to unrotate it, for some reason
        this_object.rotation_euler = (0, 0, 0)
        print(f"Imported {this_object.name}")
        return this_object

    except Exception as e:
        print(f"Error importing {filepath}: {e}")
        # raise (e)


def import_data_for_time(
    data_directory: PathLike,
    frame_time,
    kwargs_data: Dict[str, Dict[str, Any]],
    global_scale: Tuple[float, float, float],
) -> Dict[str, BlenderObject]:
    """
    Import multiple PLY mesh files organized by timestamp and category.

    Scans a directory for PLY files matching the pattern, imports them into Blender,
    and organizes them by timestamp and category. Each category gets its own collection.

    Args:
        base_directory: Directory containing PLY files to import
        kwargs_data: Dictionary mapping category names to their properties (scale, location, etc.)
        pattern: Glob pattern for matching files (default: "*.ply")

    Returns:
        Tuple containing:
            - Dictionary mapping datetime timestamps to lists of imported objects
            - Dictionary mapping category names to Blender collections

    Note:
        Only categories specified in kwargs_data will be imported.
        Files that don't match the expected naming schema will be skipped.
    """
    # Cast to Path
    data_directory = Path(data_directory)

    # Update the pattern using the current time
    glob_pattern = rf"dt-{dt_to_str(frame_time)}*"
    # Get the paths to the .ply files for this time
    this_time_data_filepaths = sorted(data_directory.glob(glob_pattern))

    if not this_time_data_filepaths:
        raise ValueError(f"No data files found matching pattern: {glob_pattern}")

    print(f"Found {len(this_time_data_filepaths)} data files to import")

    this_time_objects = {}
    # For some reason need to do the vtk files first, or they corrupt the others
    this_time_data_filepaths = sorted(
        this_time_data_filepaths,
        key=lambda p: (p.suffix not in [".vtk", ".vtp"], p.suffix, p.name),
    )
    for this_data_filepath in this_time_data_filepaths:
        if this_data_filepath.suffix not in DATA_FILE_SUFFIXES:
            continue
        # Get category and variable name
        this_category = to_kv_pairs(this_data_filepath)["category"]
        this_varname = to_kv_pairs(this_data_filepath)["varname"]
        if this_category not in kwargs_data:
            continue
        # Get the properties for this category
        category_properties = kwargs_data[this_category]
        # Check if we should include this varname
        if (
            "varnames" in category_properties
            and this_varname not in category_properties["varnames"]
        ):
            continue
        # Copy them so we can overwrite the scale
        category_properties = {k: v for k, v in category_properties.items()}
        # Get the object
        this_object = import_single_data_file(this_data_filepath)
        # If there was an issue, just continue
        if not this_object:
            continue
        # Scale by the global scale if it's not a volume
        print(this_object["data_file_suffix"])
        if this_object["data_file_suffix"] != ".vdb":
            resize_object(this_object, global_scale)

        # Store it for output
        this_time_objects[this_object.name] = this_object
        print(this_time_objects)

    # Try recreating this dict
    # this_time_objects = {k: get_object_by_name(k) for k in this_time_objects}
    # print(this_time_objects)
    return this_time_objects


def import_all_timesteps(
    data_directory: PathLike,
    animation_times: List[dt.datetime],
    kwargs_data: Dict[str, Dict[str, Any]],
    global_scale: Tuple[float, float, float],
) -> Dict[dt.datetime, Dict[str, BlenderObject]]:
    """
    Import data files for all timesteps at once for visibility keyframe animation.

    Unlike the callback-based approach that imports data on-demand, this imports
    all objects for all timesteps upfront. Objects retain their time-stamped names
    and will have visibility animated via keyframes.

    Args:
        data_directory: Directory containing data files organized by timestamp
        animation_times: List of all simulation times to import
        kwargs_data: Dictionary mapping category names to their properties
        global_scale: Global scaling factor (x, y, z)

    Returns:
        Dictionary mapping each timestamp to a dict of imported objects

    Note:
        - Objects keep time in their names (e.g., dt-20240101120000_category-clouds)
        - All setup (materials, properties) happens during import
        - Memory usage scales with number of timesteps
    """
    data_directory = Path(data_directory)
    all_timestep_objects = {}

    print(f"Importing data for {len(animation_times)} timesteps...")

    for i, frame_time in enumerate(animation_times):
        print(f"Importing timestep {i+1}/{len(animation_times)}: {frame_time}")

        # Import all objects for this timestep
        this_time_objects = import_data_for_time(
            data_directory=data_directory,
            frame_time=frame_time,
            kwargs_data=kwargs_data,
            global_scale=global_scale,
        )

        # Apply setup/properties to each object since we won't do it during animation
        from .blender_core import setup_object

        for obj_name, obj in this_time_objects.items():
            category = obj["category"]
            if category in kwargs_data:
                setup_object(obj, kwargs_obj=kwargs_data[category])

        all_timestep_objects[frame_time] = this_time_objects

    print(
        "Successfully imported"
        f" {sum(len(objs) for objs in all_timestep_objects.values())} total objects"
    )

    return all_timestep_objects


def load_wind_data(
    wind_output_dir: PathLike,
    scale: Tuple[float, float, float],
    force_scale: float = 0.3,
) -> Tuple[Dict[dt.datetime, List[Dict[str, Any]]], List[Tuple[float, float]]]:
    """
    Load wind vector data from CSV files organized by simulation timestep.

    Reads CSV files containing wind vectors (U, V, W components) at spatial points
    and converts them to magnitude and rotation data suitable for Blender force fields.

    Args:
        wind_output_dir: Directory containing timestamped CSV wind data files
        scale: Spatial scaling factor for coordinates (default: 0.0002)
        force_scale: Scaling factor for wind force magnitudes (default: 0.5)

    Returns:
        Tuple containing:
            - Dictionary mapping simulation times to lists of wind point data
            - List of [(x_min, x_max), (y_min, y_max)] bounds for the wind field

    Note:
        Each wind point data dict contains: 'magnitude', 'location', 'rotation' keys.
        CSV files are expected to have columns: x, y, z, UC, VC, WC.
    """
    wind_output_dir = Path(wind_output_dir)
    all_wind_data = {}

    for wind_file in wind_output_dir.glob("*.csv"):
        this_file_wind_data = []
        this_file_time = to_kv_pairs(wind_file)["dt"]
        with wind_file.open("r") as f:
            reader = csv.DictReader(f)
            for row in list(reader):
                # Convert the u, v, w to magnitude and rotation
                magnitude, rotation = uvw_to_magnitude_and_rotation(
                    float(row["UC"]), float(row["VC"]), float(row["WC"])
                )
                this_file_wind_data.append({
                    "magnitude": magnitude * force_scale,
                    "location": (
                        float(row["x"]) * scale[0],
                        float(row["y"]) * scale[1],
                        float(row["z"]) * scale[2],
                    ),
                    "rotation": rotation,
                })
        all_wind_data[this_file_time] = this_file_wind_data

    # Get the bounds of the plane as well
    xs = [d["location"][0] for points in all_wind_data.values() for d in points]
    ys = [d["location"][1] for points in all_wind_data.values() for d in points]
    # These are the bounds AFTER scaling is applied, i.e. in the scaled world,
    # not in RAMS data world
    bounds = [(min(xs), max(xs)), (min(ys), max(ys))]

    return all_wind_data, bounds
