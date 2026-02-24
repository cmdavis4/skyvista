import re
from pathlib import Path
from typing import Dict, Tuple

#!/usr/bin/env python3
"""
Blender Python script for importing and animating atmospheric model PLY files.

This script provides functions to:
1. Import PLY meshes organized by timestamp
2. Control mesh visibility based on animation frames
3. Set up materials for different atmospheric variables
4. Create grass animation driven by wind data

Usage:
- Run this script within Blender's Python environment
- Modify the paths and parameters in the main() function
- The script assumes PLY files are named with timestamp patterns
"""

import datetime as dt
import math
from typing import Any, Dict, List, Optional, Tuple, TypeAlias, Union

import bpy
import bmesh


from skyutils.types_skyutils import BlenderObject, PathLike

# Ignore some errors in this file
# pyright: reportReturnType=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportAttributeAccessIssue=false

# Type aliases for better readability
TimestampDict: TypeAlias = Dict[dt.datetime, List[BlenderObject]]


def deselect_all() -> None:
    """
    Deselect all objects in the current Blender scene.
    """
    bpy.ops.object.select_all(action="DESELECT")


def multiply_scales(
    scale1: Tuple[float, float, float], scale2: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    """
    Multiply two 3D scale tuples element-wise.

    Args:
        scale1: First scale tuple (x, y, z)
        scale2: Second scale tuple (x, y, z)

    Returns:
        Element-wise product of the two scale tuples
    """
    return tuple([scale1[i] * scale2[i] for i in range(3)])


def reset_scene():
    """Reset scene to completely empty state"""

    # Clear frame change handlers
    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_post.clear()

    # Delete all objects
    for obj in list(bpy.data.objects):
        if obj.type != "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)

    # Clear all collections except master collection
    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)

    # Clear materials, textures, etc. (optional)
    for material in list(bpy.data.materials):
        bpy.data.materials.remove(material, do_unlink=True)

    for texture in list(bpy.data.textures):
        bpy.data.textures.remove(texture, do_unlink=True)

    # Purge orphaned data
    bpy.ops.outliner.orphans_purge()

    print("Scene reset finished")


def get_object_by_name(
    object_name: str, raise_if_not_exists: bool = True
) -> Optional[BlenderObject]:
    """
    Retrieve a Blender object by name.

    Args:
        object_name: Name of the object to retrieve
        raise_if_not_exists: Whether to raise an exception if object not found

    Returns:
        Blender object if found, None if not found and raise_if_not_exists=False

    Raises:
        ValueError: If object not found and raise_if_not_exists=True
    """
    obj = bpy.data.objects.get(object_name)
    if not obj:
        if raise_if_not_exists:
            raise ValueError(f"Object {object_name} not found")
        else:
            return None
    else:
        return obj


def select_object(
    obj: BlenderObject, _deselect_all: bool = False, make_active: bool = True
) -> None:
    """
    Select a Blender object and optionally make it active.

    Args:
        obj: Blender object to select
        _deselect_all: Whether to deselect all objects first
        make_active: Whether to make the object the active object
    """
    if _deselect_all:
        deselect_all()
    obj.select_set(True)
    if make_active:
        bpy.context.view_layer.objects.active = obj


def translate_object(obj: BlenderObject, location: Tuple[float, float, float]) -> None:
    """
    Translate a Blender object to a new location.

    Args:
        obj: Blender object to translate
        location: New location as (x, y, z) tuple
    """
    select_object(obj, _deselect_all=True)
    bpy.ops.transform.translate(value=location)
    bpy.ops.object.transform_apply(location=True)


def resize_object(obj: BlenderObject, scale: Tuple[float, float, float]) -> None:
    """
    Resize a Blender object by applying a scale transformation.

    Args:
        obj: Blender object to resize
        scale: Scale factors as (x, y, z) tuple
    """
    # Select the object
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.transform.resize(value=scale)
    bpy.ops.object.transform_apply(scale=True)


def resize_with_constant_xy_position(
    obj: BlenderObject, scale: Tuple[float, float, float]
) -> None:
    """
    Resize an object while keeping its XY position constant.

    This function maintains the object's horizontal position while allowing
    scaling in all dimensions. Useful for scaling atmospheric data while
    preserving geographic positioning.

    Args:
        obj: Blender object to resize
        scale: Scale factors as (x, y, z) tuple
    """
    # Select the object
    select_object(obj, _deselect_all=True)
    # Set the origin to object geometry
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")
    # Set the cursor to the selected object's origin, except for in z
    bpy.context.scene.cursor.location = (obj.location[0], obj.location[1], 0)
    # Set the origin to the cursor position
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
    # Now resize the object
    resize_object(obj=obj, scale=scale)
    # Now set origin back to world origin
    bpy.context.scene.cursor.location = (0, 0, 0)
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")


def add_object_to_scene(obj: BlenderObject) -> None:
    """
    Add a Blender object to the current scene.

    Args:
        obj: Blender object to add to scene
    """
    bpy.context.scene.collection.objects.link(obj)


def get_object_name(filepath: str) -> str:
    """
    Extract object name from filepath by removing timestamp patterns.

    Args:
        filepath: Path to the file

    Returns:
        Object name with timestamp pattern removed
    """
    return re.sub(r"dt-[0-9]{14}_?", "", Path(filepath).stem)


def get_collection(collection_name: str) -> Optional[Any]:
    """
    Get a Blender collection by name.

    Args:
        collection_name: Name of the collection

    Returns:
        Collection object if found, None otherwise
    """
    return bpy.data.collections.get(collection_name)


def create_collection(
    collection_name: str, parent_collection: Optional[Union[str, Any]] = None
) -> Any:
    """
    Create a new Blender collection.

    Args:
        collection_name: Name for the new collection
        parent_collection: Parent collection (name or object), defaults to scene collection

    Returns:
        The newly created collection
    """
    parent_collection = parent_collection or bpy.context.scene.collection
    # Be nice and handle the case in which the parent collection is just a name
    if isinstance(parent_collection, str):
        parent_collection = get_collection(parent_collection)
    # Make a new collection
    new_collection = bpy.data.collections.new(collection_name)
    # Add it as a child to the parent collection
    parent_collection.children.link(new_collection)
    return new_collection


def assign_material(obj: BlenderObject, material_name: str) -> None:
    """
    Assign a material to a Blender object if the material exists.

    Args:
        obj: The Blender object to assign the material to
        material_name: Name of the material to assign

    Returns:
        None

    Note:
        If the material doesn't exist, prints a warning but continues execution.
    """
    materials = bpy.data.materials
    if material_name in materials.keys():
        material = materials[material_name]
        # Assign material to object
        obj.data.materials.append(material)
    else:
        # Just don't assign a material
        print(f"No material {material_name} found, not assigning")


def move_to_collection(obj: BlenderObject, collection) -> None:
    """
    Move a Blender object to the appropriate collection based on category.

    Args:
        obj: Blender object to move
        category: Category name that corresponds to a collection
        collections: Dictionary mapping category names to collection objects

    Returns:
        None

    Note:
        If the category doesn't exist in collections, the object remains unmoved.
    """
    if isinstance(collection, str):
        collection = get_collection(collection_name=collection)
    # Add to target collection
    collection.objects.link(obj)


def setup_object(obj: BlenderObject, kwargs_obj: Dict[str, Any]) -> None:
    """
    Set up a Blender object with various properties and configurations.

    Args:
        obj: Blender object to configure
        kwargs_obj: Dictionary of properties and values to apply
    """
    # Apply properties from mesh_kwargs
    # List kwargs we should ignore
    ignored_kwargs = ["varnames"]
    for mesh_kwarg_name, mesh_kwarg_value in kwargs_obj.items():
        # Handle some manually
        if mesh_kwarg_name == "location":
            translate_object(obj, mesh_kwarg_value)
        elif mesh_kwarg_name == "scale":
            # Need to keep the horizontal position constant, but scale z appropriately
            resize_object(obj, mesh_kwarg_value)
        # Handle geometry nodes
        elif mesh_kwarg_name == "geometry_nodes":
            for node_group_name in mesh_kwarg_value:
                # Get the node group from the name
                node_group = bpy.data.node_groups.get(node_group_name)
                # Add a new modifier
                modifier = obj.modifiers.new(name=node_group_name, type="NODES")
                # Assign the node group
                modifier.node_group = node_group
        elif mesh_kwarg_name == "material":
            print("Assigning material")
            assign_material(obj, kwargs_obj["material"])
        elif mesh_kwarg_name == "collections":
            for collection in mesh_kwarg_value:
                move_to_collection(
                    obj,
                    collection=get_collection(collection_name=collection)
                    or create_collection(collection_name=collection),
                )
        elif mesh_kwarg_name not in ignored_kwargs:
            setattr(obj, mesh_kwarg_name, mesh_kwarg_value)
        # Also move to a collection for the file type
        file_type_collection_name = obj["data_file_suffix"][1:] + "s"
        move_to_collection(
            obj,
            collection=get_collection(collection_name=file_type_collection_name)
            or create_collection(collection_name=file_type_collection_name),
        )
        # Remove from scene collection
        bpy.context.scene.collection.objects.unlink(obj)


def update_object_geometry(
    obj: BlenderObject, new_mesh: BlenderObject, depsgraph: Optional[Any] = None
) -> None:
    """Update only the geometry, preserve everything else"""
    print(
        f"Updating mesh geometry for object '{obj.name}' from new mesh"
        f" '{new_mesh.name}'"
    )

    # Store current settings
    obj_data = obj.data
    materials = obj_data.materials[:]

    # Make the target object active and selected for proper updates
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    try:
        if obj["data_file_suffix"] == ".vdb":
            # This one's easy
            obj.data.filepath = new_mesh.data.filepath
        else:
            # Transfer geometry using BMesh
            bm_new = bmesh.new()
            bm_new.from_mesh(new_mesh.data)

            # Clear existing geometry
            obj_data.clear_geometry()

            # Apply new geometry
            bm_new.to_mesh(obj_data)
            bm_new.free()

            # Clear and restore materials properly
            obj_data.materials.clear()
            for mat in materials:
                obj_data.materials.append(mat)
            # Force mesh data update
            obj_data.update()

        # Update dependency graph and viewport
        if depsgraph:
            depsgraph.update()
        bpy.context.view_layer.update()

    except Exception as e:
        print(f"Error updating geometry for '{obj.name}': {e}")
        raise

    finally:
        # Clean up the temporary object
        try:
            print(f"Cleaned up temporary object '{new_mesh.name}'")
            bpy.data.objects.remove(new_mesh, do_unlink=True)
        except Exception as e:
            raise (e)


def create_grass_field() -> Tuple[BlenderObject, BlenderObject]:
    """
    Create a procedural grass field using geometry nodes.

    Creates a ground plane sized to fit the wind field bounds and adds a
    geometry nodes setup to generate grass that deforms based on VDB wind data.

    Args:
        bounds: List of [(x_min, x_max), (y_min, y_max)] defining the field area

    Returns:
        The ground plane object with grass geometry nodes setup

    Note:
        The grass field is positioned and sized to match the wind data bounds
        to ensure proper interaction between wind VDB volumes and grass deformation.
    """
    # Add the plane from the asset library
    grass_plane = get_object_by_name("grass_plane")
    add_object_to_scene(grass_plane)
    # Also add the bigger plane without hair
    grass_plane_nohair = get_object_by_name("grass_plane_nohair")
    add_object_to_scene(grass_plane_nohair)

    return grass_plane, grass_plane_nohair


def position_text_for_camera(
    text_obj: BlenderObject,
    camera_obj: BlenderObject,
    camera_axis_distance: float = 1.0,
    vertical_offset: float = 0.0,
    horizontal_offset: float = 0.0,
) -> None:
    """
    Position object in front of camera AND orient it to face the camera.

    Args:
        text_obj: Text object to position
        camera_obj: Camera object to position relative to
        camera_axis_distance: Distance from camera along forward axis
        vertical_offset: Vertical offset from camera position
        horizontal_offset: Horizontal offset from camera position
    """
    """
    Position object in front of camera AND orient it to face the camera.
    """
    # Get camera's world matrix
    camera_matrix = camera_obj.matrix_world

    # Extract camera's local axes
    forward_vector = -camera_matrix.col[2].xyz.normalized()
    up_vector = camera_matrix.col[1].xyz.normalized()
    right_vector = camera_matrix.col[0].xyz.normalized()

    # Position the object with offsets
    new_position = (
        camera_obj.location
        + (forward_vector * camera_axis_distance)
        + (up_vector * vertical_offset)
        + (right_vector * horizontal_offset)
    )
    text_obj.location = new_position

    # Orient object to face camera
    direction = camera_obj.location - text_obj.location
    text_obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()


def set_sky_background(world_asset_name: str) -> None:
    """
    Set the world background to a specific sky asset.

    Args:
        world_asset_name: Name of the world asset to use as background
    """
    # Set as active world
    if world_asset_name in bpy.data.worlds:
        bpy.context.scene.world = bpy.data.worlds[world_asset_name]


DATA_FILE_SUFFIXES = [".ply", ".vdb", ".vtk"]


def uvw_to_magnitude_and_rotation(
    U: float, V: float, W: float
) -> Tuple[float, Tuple[float, float, float]]:
    """
    Convert wind vector components to magnitude and rotation angles.

    Uses the same mathematical approach as Blender's to_track_quat function to convert
    a 3D wind vector (U, V, W) into magnitude and Euler rotation angles.

    Args:
        U: Wind velocity component in X direction (m/s)
        V: Wind velocity component in Y direction (m/s)
        W: Wind velocity component in Z direction (m/s, forced positive for surface winds)

    Returns:
        Tuple containing:
            - magnitude: Wind speed magnitude (m/s)
            - rotation: Tuple of (roll, pitch, yaw) angles in degrees

    Note:
        This function has no Blender dependencies and can be used externally.
        W component is clamped to positive values for surface wind modeling.
    """

    # W has to be positive since we're at the surface
    W = max(W, 0)

    # Normalize vector
    magnitude = math.sqrt(U**2 + V**2 + W**2)
    if magnitude == 0:
        return 0.0, (0.0, 0.0, 0.0)

    u, v, w = U / magnitude, V / magnitude, W / magnitude

    # This mimics Blender's to_track_quat('Z', 'Y') calculation
    # Calculate rotation to align (0,0,1) with (u,v,w)

    # Angle from Z-axis to target vector
    dot_z = w  # dot product with (0,0,1)
    angle_from_z = math.acos(max(-1, min(1, dot_z)))

    # If vector is already aligned with Z, no rotation needed
    if abs(dot_z) > 0.9999:
        if dot_z > 0:
            return magnitude, (0.0, 0.0, 0.0)  # Same direction
        else:
            return magnitude, (180.0, 0.0, 0.0)  # Opposite direction

    # Axis of rotation (cross product of Z with target)
    axis_x = -v  # (0,0,1) × (u,v,w) = (-v, u, 0)
    axis_y = u
    axis_z = 0

    # Normalize rotation axis
    axis_len = math.sqrt(axis_x * axis_x + axis_y * axis_y)
    if axis_len > 0:
        axis_x /= axis_len
        axis_y /= axis_len

    # Convert axis-angle to quaternion
    half_angle = angle_from_z / 2
    sin_half = math.sin(half_angle)
    cos_half = math.cos(half_angle)

    qx = axis_x * sin_half
    qy = axis_y * sin_half
    qz = axis_z * sin_half
    qw = cos_half

    # Convert quaternion to Euler angles (XYZ order)
    # Standard quaternion to Euler conversion
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = math.asin(max(-1, min(1, 2 * (qw * qy - qz * qx))))
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))

    return magnitude, (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def add_map(map_image_fpath: PathLike, corners: List[Tuple]):

    mesh = bpy.data.meshes.new("GroundPlane")
    mesh.from_pydata(corners, [], [(0, 1, 2, 3)])
    mesh.update()

    obj = bpy.data.objects.new("GroundPlane", mesh)
    bpy.context.collection.objects.link(obj)

    # --- UVs ---
    uv_layer = mesh.uv_layers.new(name="UVMap")
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for loop, uv in zip(mesh.loops, uvs):
        uv_layer.data[loop.index].uv = uv

    # --- Material ---
    mat = bpy.data.materials.new("GroundMat")
    mat.use_nodes = True

    # Remove the default Principled Volume node
    nodes = mat.node_tree.nodes
    for node in nodes:
        if node.type == "VOLUME_PRINCIPLED":
            nodes.remove(node)

    # Get the Material Output node (created by default)
    output_node = nodes.get("Material Output")

    # Create Principled BSDF node
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")

    # Create Image Texture node
    tex_node = nodes.new("ShaderNodeTexImage")
    tex_node.image = bpy.data.images.load(str(map_image_fpath))

    # Connect texture to BSDF
    mat.node_tree.links.new(
        bsdf.inputs["Base Color"],
        tex_node.outputs["Color"],
    )

    # Connect BSDF to Material Output Surface input
    mat.node_tree.links.new(
        output_node.inputs["Surface"],
        bsdf.outputs["BSDF"],
    )

    obj.data.materials.append(mat)
