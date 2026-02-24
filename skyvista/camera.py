import numpy as np


def calculate_camera_offset(camera_position, storm_position_ds):
    """
    Calculate camera offset for storm-following animations.

    Args:
        camera_position (tuple): (position, focal_point) camera configuration.
        storm_position_ds (xr.Dataset): Dataset with storm_position_x and storm_position_y.

    Returns:
        np.ndarray: Camera offset array for storm following.
    """
    current_position = camera_position[0]
    current_fp = camera_position[1]
    return np.vstack([
        np.array([
            current_position[0] - storm_position_ds["storm_position_x"].values,
            current_position[1] - storm_position_ds["storm_position_y"].values,
            0,
        ]),
        np.array([
            current_fp[0] - storm_position_ds["storm_position_x"].values,
            current_fp[1] - storm_position_ds["storm_position_y"].values,
            0,
        ]),
        np.array([0, 0, 0]),
    ])


def get_trajectory_camera(plotter, position, direction, distance=2000):
    """
    Calculate camera position for trajectory following.

    Args:
        plotter: PyVista plotter object.
        position (array): Camera position.
        direction (array): Look direction vector.
        distance (float, optional): Distance to focal point. Defaults to 2000.

    Returns:
        np.ndarray: Camera configuration array.
    """
    # Normalize direction vector
    direction = direction / np.linalg.norm(direction)

    # Calculate focal point
    focal_point = position + distance * direction

    # Set up vector (default to z-axis)
    up_vector = np.array([0, 0, 1])
    return np.array([position * plotter.scale, focal_point * plotter.scale, up_vector])


def camera_follow_callback(camera_offsets):
    """
    Create callback for camera following storm motion.

    Args:
        camera_offsets (dict or array): Camera offset configuration.

    Returns:
        callable: Callback function for storm following.
    """
    # Handle single offset case
    if not isinstance(camera_offsets, dict):
        camera_offsets = {(0, 0): camera_offsets}

    def _camera_follow_callback_helper(plotter, current_time, simulation_ds, *args):
        # Update camera for each subplot with offsets
        for (row, col), camera_offset in camera_offsets.items():
            plotter.subplot(row, col)
            # Set camera to follow storm position
            plotter.camera_position = (
                np.array([
                    [
                        simulation_ds.sel({"time": current_time})[
                            "storm_position_x"
                        ].values,
                        simulation_ds.sel({"time": current_time})[
                            "storm_position_y"
                        ].values,
                        plotter.camera_position[0][2],
                    ],
                    [
                        simulation_ds.sel({"time": current_time})[
                            "storm_position_x"
                        ].values,
                        simulation_ds.sel({"time": current_time})[
                            "storm_position_y"
                        ].values,
                        plotter.camera_position[1][2],
                    ],
                    plotter.camera_position[2],
                ])
                + camera_offset
            )

    return _camera_follow_callback_helper


def right_pyvista_camera(plotter):
    """
    Reorient a PyVista camera so that its z-axis points up.

    Parameters
    ----------
    plotter : pyvista.Plotter
        The PyVista plotter object to modify
    """
    # Get current camera position components
    current_pos, current_focal, current_up = plotter.camera_position

    # Convert to numpy arrays for easier manipulation
    pos = np.array(current_pos)
    focal = np.array(current_focal)

    # Set the up vector to point in the positive z direction
    new_up = (0, 0, 1)

    # Keep the same position and focal point, just change the up vector
    plotter.camera_position = [tuple(pos), tuple(focal), new_up]


def reflect_camera(plotter):
    # Get the current camera position components
    current_pos, current_focal, current_up = plotter.camera_position

    # Calculate the vector from focal point to camera position
    cam_vector = np.array(current_pos) - np.array(current_focal)

    # Rotate 180 degrees around the focal point by negating x and y components
    new_cam_vector = np.array([-cam_vector[0], -cam_vector[1], cam_vector[2]])

    # Calculate new camera position
    new_pos = np.array(current_focal) + new_cam_vector

    # Set the new camera position (keeping the same focal point and up vector)
    plotter.camera_position = [tuple(new_pos), current_focal, current_up]
