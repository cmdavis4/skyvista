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

__version__ = "0.3.0"

# =============================================================================
# PRIMARY API
# =============================================================================

# Scene class - the main entry point
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
)

# =============================================================================
# VARSPEC CLASSES (for advanced users building custom specs)
# =============================================================================

from .varspec import (
    VarSpec,
    ContourSpec,
    VolumeSpec,
    VectorSpec,
    SliceSpec,
    TrajectorySpec,
)

# =============================================================================
# GEOMETRY CLASSES (for advanced users)
# =============================================================================

from .geometry import (
    Geometry,
    ContourGeometry,
    VolumeGeometry,
    VectorGeometry,
    SliceGeometry,
    TrajectoryGeometry,
)

# =============================================================================
# APPEARANCE CLASSES (for advanced users)
# =============================================================================

from .appearance import (
    Appearance,
    ContourAppearance,
    VolumeAppearance,
    VectorAppearance,
    TrajectoryAppearance,
)

# =============================================================================
# GRID BUILDERS AND COORDINATE UTILITIES
# =============================================================================

from .grids import (
    # Constants
    COORD_ALIASES,
    GEOGRAPHIC_COORD_NAMES,
    SPHERICAL_COORD_NAMES,
    EARTH_RADIUS_M,
    # Grid builders
    GridBuilder,
    RectilinearGridBuilder,
    CurvilinearGridBuilder,
    UnstructuredGridBuilder,
    GeographicGridBuilder,
    SphericalGridBuilder,
    # Detection and factory functions
    detect_grid_type,
    get_grid_builder,
    is_geographic_grid,
    is_spherical_grid,
    # Coordinate resolution
    resolve_coordinate,
    resolve_coordinates,
    resolve_spherical_coordinates,
)
from .grid_utils import normalize_dimension_order

# =============================================================================
# UTILITIES
# =============================================================================

from .plotter import initialize_plotter
from .mesh import PVMesh
from . import presets

# Example data loader
from .examples import load_example_storm_data

# Camera utilities
from .camera import (
    calculate_camera_offset,
    get_trajectory_camera,
    camera_follow_callback,
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
    "make_vector",
    "make_slice",
    "make_trajectory",
    # VarSpec classes
    "VarSpec",
    "ContourSpec",
    "VolumeSpec",
    "VectorSpec",
    "SliceSpec",
    "TrajectorySpec",
    # Geometry classes
    "Geometry",
    "ContourGeometry",
    "VolumeGeometry",
    "VectorGeometry",
    "SliceGeometry",
    "TrajectoryGeometry",
    # Appearance classes
    "Appearance",
    "ContourAppearance",
    "VolumeAppearance",
    "VectorAppearance",
    "TrajectoryAppearance",
    # Grid builders
    "COORD_ALIASES",
    "GEOGRAPHIC_COORD_NAMES",
    "SPHERICAL_COORD_NAMES",
    "EARTH_RADIUS_M",
    "GridBuilder",
    "RectilinearGridBuilder",
    "CurvilinearGridBuilder",
    "UnstructuredGridBuilder",
    "GeographicGridBuilder",
    "SphericalGridBuilder",
    "detect_grid_type",
    "get_grid_builder",
    "is_geographic_grid",
    "is_spherical_grid",
    "resolve_coordinate",
    "resolve_coordinates",
    "resolve_spherical_coordinates",
    "normalize_dimension_order",
    # Utilities
    "initialize_plotter",
    "PVMesh",
    "presets",
    "load_example_storm_data",
    # Camera utilities
    "calculate_camera_offset",
    "get_trajectory_camera",
    "camera_follow_callback",
]
