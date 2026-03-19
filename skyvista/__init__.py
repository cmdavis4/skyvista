"""
skyvista: 3D atmospheric data visualization.

Primary API:
    Scene              - Container for building visualizations
    plot_gridded       - Create scene with gridded data visualizations
    plot_trajectories  - Create scene with trajectory visualization

Factory functions (create VarSpecs):
    make_contour       - Create contour/isosurface spec
    make_volume        - Create volume rendering spec
    make_vectors       - Create vector field spec
    make_slice         - Create 2D slice spec
    make_trajectory    - Create trajectory spec

Example:
    >>> import skyvista as sv
    >>> scene = sv.plot_gridded(sim_ds, contours={"THETA": [300, 310]})
    >>> scene.add_trajectories(traj_ds, scalar="altitude")
    >>> scene.show()
"""

__version__ = "0.2.0"

# =============================================================================
# PRIMARY API (new Scene-based API)
# =============================================================================

# Scene class
from .scene import Scene

# Convenience functions that return Scenes
from .convenience import (
    plot_gridded,
    plot_trajectories,
    # Factory functions
    make_contour,
    make_volume,
    make_vectors,
    make_vector,  # Alias for make_vectors
    make_slice,
    make_trajectory,
    # Deprecated (kept for backwards compatibility)
    quick_plot,
    plot_trajectories_only,
    plot_isosurfaces_only,
)

# =============================================================================
# ADVANCED API (for power users)
# =============================================================================

# VarSpec classes
from .varspec import (
    VarSpec,
    ContourSpec,
    VolumeSpec,
    VectorSpec,
    SliceSpec,
    TrajectorySpec,
)

# Geometry classes
from .geometry import (
    Geometry,
    ContourGeometry,
    VolumeGeometry,
    VectorGeometry,
    SliceGeometry,
    TrajectoryGeometry,
)

# Appearance classes
from .appearance import (
    Appearance,
    ContourAppearance,
    VolumeAppearance,
    VectorAppearance,
    TrajectoryAppearance,
)

# =============================================================================
# UTILITIES
# =============================================================================

from .plotter import initialize_plotter
from .examples import load_example_storm_data
from . import presets

# Grid builders and coordinate utilities
from .grids import (
    COORD_ALIASES,
    GridBuilder,
    RectilinearGridBuilder,
    CurvilinearGridBuilder,
    UnstructuredGridBuilder,
    detect_grid_type,
    get_grid_builder,
    resolve_coordinate,
    resolve_coordinates,
)

# =============================================================================
# DEPRECATED API (kept for backwards compatibility)
# =============================================================================

# Old core functions
from .core import (
    plot_gridded_and_trajectories,
    plot_rams_and_trajectories,
    get_subplot_keys,
    sanitize_inputs,
    plot_trajectory_frame,
    animate_trajectories,
    rectangle_mesh,
    screenshot_render,
    add_mesh_to_subplots,
)

# Old type definitions
from .types_sv import (
    PVConfig,
    PVGriddedData,
    PVRamsData,
    PVTrajectoryData,
    PVMesh,
    PV2DSpec,
    PVContourSpec,
    PVVolumeSpec,
    PVVectorSpec,
    PVTrajectorySpec,
    PVVarSpec,
    PVData,
)

# Trajectory utilities
from .trajectories import (
    create_trajectory_polydata,
    create_tetrahedron_head,
    create_trajectory_mesh,
    generate_trajectory_mesh,
)

# Camera utilities
from .camera import (
    calculate_camera_offset,
    get_trajectory_camera,
    camera_follow_callback,
)

# Blender export utilities (optional - requires bpy)
try:
    from .blender.pv_to_blender import export_meshes_to_blender
except ImportError:
    # bpy not available - provide stub that raises helpful error
    def export_meshes_to_blender(*args, **kwargs):
        raise ImportError(
            "export_meshes_to_blender requires Blender's bpy module. "
            "Run this code from within Blender or install bpy."
        )

# =============================================================================
# __all__ - Public API
# =============================================================================

__all__ = [
    # Primary API
    "Scene",
    "plot_gridded",
    "plot_trajectories",
    # Factory functions
    "make_contour",
    "make_volume",
    "make_vectors",
    "make_vector",  # Alias for make_vectors
    "make_slice",
    "make_trajectory",
    # Blender export
    "export_meshes_to_blender",
    # VarSpec classes (advanced)
    "VarSpec",
    "ContourSpec",
    "VolumeSpec",
    "VectorSpec",
    "SliceSpec",
    "TrajectorySpec",
    # Geometry classes (advanced)
    "Geometry",
    "ContourGeometry",
    "VolumeGeometry",
    "VectorGeometry",
    "SliceGeometry",
    "TrajectoryGeometry",
    # Appearance classes (advanced)
    "Appearance",
    "ContourAppearance",
    "VolumeAppearance",
    "VectorAppearance",
    "TrajectoryAppearance",
    # Utilities
    "initialize_plotter",
    "load_example_storm_data",
    "presets",
    # Grid builders
    "COORD_ALIASES",
    "GridBuilder",
    "RectilinearGridBuilder",
    "CurvilinearGridBuilder",
    "UnstructuredGridBuilder",
    "detect_grid_type",
    "get_grid_builder",
    "resolve_coordinate",
    "resolve_coordinates",
    # Deprecated (backwards compatibility)
    "quick_plot",
    "plot_trajectories_only",
    "plot_isosurfaces_only",
    "plot_gridded_and_trajectories",
    "plot_rams_and_trajectories",
    "PVConfig",
    "PVGriddedData",
    "PVRamsData",
    "PVTrajectoryData",
    "PVMesh",
    "PV2DSpec",
    "PVContourSpec",
    "PVVolumeSpec",
    "PVVectorSpec",
    "PVTrajectorySpec",
    "PVVarSpec",
    "PVData",
]
