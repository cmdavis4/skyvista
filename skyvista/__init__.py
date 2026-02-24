"""
skyvista: PyVista-based 3D visualization for atmospheric data.

This package provides comprehensive 3D visualization capabilities using PyVista
for atmospheric modeling data including trajectories, isosurfaces, and animations.
"""

__version__ = "0.1.0"

# Core plotting functions
from .plotter import initialize_plotter
from .core import (
    get_subplot_keys,
    sanitize_inputs,
    plot_rams_and_trajectories,
    plot_trajectory_frame,
    animate_trajectories,
    rectangle_mesh,
    screenshot_render,
    add_mesh_to_subplots,
)

# Type definitions
from .types_sv import (
    PVConfig,
    PVRamsData,
    PVTrajectoryData,
    PV2DSpec,
    PVContourSpec,
    PVVectorSpec,
    PVTrajectorySpec,
)

# Trajectory functions
from .trajectories import (
    create_trajectory_polydata,
    create_tetrahedron_head,
    create_trajectory_mesh,
    generate_trajectory_mesh,
)

# Camera functions
from .camera import (
    calculate_camera_offset,
    get_trajectory_camera,
    camera_follow_callback,
)

# Blender export functions
from .pv_to_blender import (
    export_meshes_to_blender,
)

# Convenience functions for simplified API
from .convenience import (
    quick_plot,
    plot_trajectories_only,
    plot_isosurfaces_only,
    make_contour,
    make_vector,
    make_trajectory,
)

# Main plotting function alias
plot_trajectories = plot_rams_and_trajectories

__all__ = [
    # Main functions
    "plot_rams_and_trajectories",
    "plot_trajectories",
    "animate_trajectories",
    # Convenience functions
    "quick_plot",
    "plot_trajectories_only",
    "plot_isosurfaces_only",
    "make_contour",
    "make_vector",
    "make_trajectory",
    # Core utilities
    "initialize_plotter",
    "get_subplot_keys",
    "sanitize_inputs",
    "plot_trajectory_frame",
    "rectangle_mesh",
    "screenshot_render",
    "add_mesh_to_subplots",
    # Types
    "PVConfig",
    "PVRamsData",
    "PVTrajectoryData",
    "PV2DSpec",
    "PVContourSpec",
    "PVVectorSpec",
    "PVTrajectorySpec",
    # Trajectories
    "create_trajectory_polydata",
    "create_tetrahedron_head",
    "create_trajectory_mesh",
    "generate_trajectory_mesh",
    # Camera
    "calculate_camera_offset",
    "get_trajectory_camera",
    "camera_follow_callback",
    # Blender
    "export_meshes_to_blender",
]
