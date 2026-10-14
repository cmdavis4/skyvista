"""
skyvista: 3D atmospheric data visualization.

Primary API:
    Scene              - Container for building visualizations
    plot_gridded       - Create scene with gridded data visualizations
    plot_trajectories  - Create scene with trajectory visualization

Factory functions (create VarSpecs):
    make_contour       - Create contour/isosurface spec
    make_volume        - Create volume rendering spec
    make_clouds        - Create physically based cloud spec (hydrometeors)
    make_vectors       - Create vector field spec
    make_slice         - Create 2D slice spec
    make_trajectory    - Create trajectory spec

Example:
    >>> import skyvista as sv
    >>> scene = sv.plot_gridded(sim_ds, contours={"THETA": [300, 310]})
    >>> scene.add_trajectories(traj_ds, scalar="altitude")
    >>> scene.show()
"""

__version__ = "1.2.0"

from pathlib import Path

SKYVISTA_ROOT_DIR = Path(__file__).parent.parent

# =============================================================================
# HEADLESS RENDERING SETUP
# =============================================================================
# Auto-configure offscreen rendering BEFORE PyVista is first imported (below,
# via .scene), so PYVISTA_OFF_SCREEN is in place when PyVista reads it. On a
# headless server this makes rendering work with no Xvfb/DISPLAY setup; opt out
# with SKYVISTA_NO_AUTOCONFIG=1. See skyvista.configure() / skyvista.doctor().
from .headless import configure, doctor, has_working_display
from .headless import _autoconfigure_on_import as _sv_autoconfigure_on_import

_sv_autoconfigure_on_import()

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
    make_clouds,
    make_vectors,
    make_vector,  # Alias for make_vectors
    make_slice,
    make_dye,
    make_streamlines,
    make_trajectory,
)

# =============================================================================
# VARSPEC CLASSES (for advanced users building custom specs)
# =============================================================================

from .varspec import (
    VarSpec,
    ContourSpec,
    VolumeSpec,
    CloudSpec,
    DyeSpec,
    VectorSpec,
    SliceSpec,
    StreamlineSpec,
    TrajectorySpec,
)

# =============================================================================
# GEOMETRY CLASSES (for advanced users)
# =============================================================================

from .geometry import (
    Geometry,
    ContourGeometry,
    VolumeGeometry,
    CloudGeometry,
    DyeGeometry,
    VectorGeometry,
    SliceGeometry,
    StreamlineGeometry,
    TrajectoryGeometry,
)

# =============================================================================
# APPEARANCE CLASSES (for advanced users)
# =============================================================================

from .appearance import (
    Appearance,
    ContourAppearance,
    VolumeAppearance,
    CloudAppearance,
    DyeAppearance,
    VectorAppearance,
    StreamlineAppearance,
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
from . import animation
from . import compose
from .blender_animation import BlenderAnimation
from .camera_path import CameraKeyframe, CameraPath, Track

# Blender export configuration (the backend itself is imported lazily by
# Scene.to_blender so the optional 'alembic3d' dep is only needed on export).
from .blender import (
    BlenderCameraConfig,
    BlenderExportConfig,
    BlenderHoldConfig,
    BlenderRenderConfig,
    BlenderTransform,
)

# Cloud optics: hydrometeor species definitions and microphysics presets
from .cloud_optics import (
    MICROPHYSICS_PRESETS,
    HydrometeorSpecies,
)

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
    "make_clouds",
    "make_vectors",
    "make_dye",
    "make_streamlines",
    "make_vector",
    "make_slice",
    "make_trajectory",
    # VarSpec classes
    "VarSpec",
    "ContourSpec",
    "VolumeSpec",
    "CloudSpec",
    "DyeSpec",
    "VectorSpec",
    "StreamlineSpec",
    "SliceSpec",
    "TrajectorySpec",
    # Geometry classes
    "Geometry",
    "ContourGeometry",
    "VolumeGeometry",
    "CloudGeometry",
    "DyeGeometry",
    "VectorGeometry",
    "StreamlineGeometry",
    "SliceGeometry",
    "TrajectoryGeometry",
    # Appearance classes
    "Appearance",
    "ContourAppearance",
    "VolumeAppearance",
    "CloudAppearance",
    "DyeAppearance",
    "VectorAppearance",
    "StreamlineAppearance",
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
    "animation",
    "compose",
    "BlenderAnimation",
    "CameraKeyframe",
    "CameraPath",
    "Track",
    # Headless rendering setup / diagnostics
    "configure",
    "doctor",
    "has_working_display",
    # Blender export
    "BlenderCameraConfig",
    "BlenderExportConfig",
    "BlenderHoldConfig",
    "BlenderRenderConfig",
    "BlenderTransform",
    "load_example_storm_data",
    # Cloud optics
    "HydrometeorSpecies",
    "MICROPHYSICS_PRESETS",
    # Camera utilities
    "calculate_camera_offset",
    "get_trajectory_camera",
    "camera_follow_callback",
]
