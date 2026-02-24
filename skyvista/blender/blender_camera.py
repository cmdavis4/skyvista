from skyutils.types_skyutils import BlenderObject
from .blender_core import get_object_by_name

import bpy


def setup_camera() -> BlenderObject:
    """
    Set up camera for atmospheric visualization with predefined positioning.

    Creates and positions a camera object optimized for viewing atmospheric
    simulations. The default position provides a good overview of the scene.

    Args:
        location: (x, y, z) position coordinates for the camera
        rotation: (x, y, z) Euler rotation angles in degrees

    Returns:
        The created camera object

    Side Effects:
        - Creates new camera object in the scene
        - Sets it as the active scene camera
    """
    # Do nothing if we already have a camera
    if not get_object_by_name("Camera", raise_if_not_exists=False):
        # Add camera
        bpy.ops.object.camera_add()
        camera_obj = bpy.context.active_object

        # Make it the active camera
        bpy.context.scene.camera = camera_obj
        camera_obj.name = "Camera"
    else:
        camera_obj = get_object_by_name("Camera")
    return camera_obj
