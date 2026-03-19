"""
Grid builders for different data types.

This module provides the GridBuilder abstraction for handling different grid types
(rectilinear, curvilinear, unstructured) uniformly. It also provides auto-detection
of grid type using CF conventions and common coordinate name aliases.

Primary API:
    detect_grid_type(ds)    - Auto-detect grid type and return appropriate builder
    get_grid_builder(ds)    - Get builder, with helpful error messages on failure

Grid Builders:
    RectilinearGridBuilder  - Regular orthogonal grids (x, y, z coords)
    CurvilinearGridBuilder  - Non-orthogonal structured grids (2D/3D coord arrays)
    UnstructuredGridBuilder - Point clouds and unstructured meshes

Coordinate Detection:
    Coordinates are detected using:
    1. CF Conventions axis attributes (standard_name, axis)
    2. COORD_ALIASES fallback (common naming patterns)

Example:
    >>> builder = detect_grid_type(ds)
    >>> mesh = builder.build_mesh(ds, "temperature")
    >>> bounds = builder.create_bounds_mesh(ds)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pyvista as pv
import xarray as xr


# =============================================================================
# COORDINATE ALIASES
# =============================================================================

# Common coordinate name patterns for each axis.
# These are used as fallback when CF conventions are not present.
# Listed in priority order (most common first).
COORD_ALIASES: Dict[str, List[str]] = {
    "x": [
        "x", "X",
        "lon", "longitude", "Longitude", "LONGITUDE",
        "XLONG", "XLONG_M",  # WRF
        "xc", "XC",  # Some ocean models
        "ni", "xi_rho",  # ROMS
    ],
    "y": [
        "y", "Y",
        "lat", "latitude", "Latitude", "LATITUDE",
        "XLAT", "XLAT_M",  # WRF
        "yc", "YC",  # Some ocean models
        "nj", "eta_rho",  # ROMS
    ],
    "z": [
        "z", "Z",
        "lev", "level", "Level", "LEV",
        "height", "Height", "HEIGHT",
        "altitude", "Altitude", "ALTITUDE",
        "zc", "ZC",
        "depth", "Depth", "DEPTH",
        "pressure", "Pressure", "PRESSURE",
        "sigma", "s_rho",  # ROMS
    ],
    "time": [
        "time", "Time", "TIME", "t", "T",
        "XTIME",  # WRF
        "ocean_time",  # ROMS
    ],
}


# CF Convention standard names for coordinate detection
CF_STANDARD_NAMES: Dict[str, List[str]] = {
    "x": ["projection_x_coordinate", "grid_longitude", "longitude"],
    "y": ["projection_y_coordinate", "grid_latitude", "latitude"],
    "z": ["altitude", "height", "depth", "air_pressure", "atmosphere_sigma_coordinate"],
    "time": ["time"],
}

CF_AXIS_NAMES: Dict[str, str] = {
    "x": "X",
    "y": "Y",
    "z": "Z",
    "time": "T",
}


# =============================================================================
# COORDINATE RESOLUTION HELPERS
# =============================================================================


def _find_coord_by_cf(ds: xr.Dataset, axis: str) -> Optional[str]:
    """
    Find coordinate using CF conventions (axis attribute and standard_name).

    Args:
        ds: xarray Dataset
        axis: Axis to find ('x', 'y', 'z', 'time')

    Returns:
        Coordinate name if found, None otherwise
    """
    # Check axis attribute first (most explicit)
    cf_axis = CF_AXIS_NAMES.get(axis)
    if cf_axis:
        for name, coord in ds.coords.items():
            if coord.attrs.get("axis") == cf_axis:
                return str(name)

    # Check standard_name
    standard_names = CF_STANDARD_NAMES.get(axis, [])
    for name, coord in ds.coords.items():
        if coord.attrs.get("standard_name") in standard_names:
            return str(name)

    return None


def _find_coord_by_alias(ds: xr.Dataset, axis: str) -> Optional[str]:
    """
    Find coordinate using COORD_ALIASES fallback.

    Args:
        ds: xarray Dataset
        axis: Axis to find ('x', 'y', 'z', 'time')

    Returns:
        Coordinate name if found, None otherwise
    """
    aliases = COORD_ALIASES.get(axis, [])

    # Check both coordinates and dimensions
    available = set(ds.coords.keys()) | set(ds.dims.keys())

    for alias in aliases:
        if alias in available:
            return alias

    return None


def resolve_coordinate(ds: xr.Dataset, axis: str) -> str:
    """
    Resolve the coordinate name for a given axis.

    Uses CF conventions first, then falls back to COORD_ALIASES.
    Raises helpful error if coordinate cannot be found.

    Args:
        ds: xarray Dataset
        axis: Axis to find ('x', 'y', 'z', 'time')

    Returns:
        Coordinate name

    Raises:
        ValueError: If coordinate cannot be resolved, with helpful message
    """
    # Try CF conventions first
    coord = _find_coord_by_cf(ds, axis)
    if coord:
        return coord

    # Fall back to aliases
    coord = _find_coord_by_alias(ds, axis)
    if coord:
        return coord

    # Build helpful error message
    aliases = COORD_ALIASES.get(axis, [])
    available = sorted(set(ds.coords.keys()) | set(ds.dims.keys()))

    raise ValueError(
        f"Could not find '{axis}' coordinate in dataset.\n"
        f"  Looked for CF axis='{CF_AXIS_NAMES.get(axis, '?')}' or "
        f"standard_name in {CF_STANDARD_NAMES.get(axis, [])}\n"
        f"  Also checked aliases: {aliases[:5]}{'...' if len(aliases) > 5 else ''}\n"
        f"  Available coordinates/dimensions: {available}\n"
        f"  To fix: either add CF-compliant attributes to your coordinates,\n"
        f"  or rename coordinates to match common patterns."
    )


def resolve_coordinates(
    ds: xr.Dataset,
    axes: List[str] = None,
    required: List[str] = None,
) -> Dict[str, Optional[str]]:
    """
    Resolve multiple coordinate names.

    Args:
        ds: xarray Dataset
        axes: Axes to resolve (default: ['x', 'y', 'z'])
        required: Which axes are required (raises error if not found)

    Returns:
        Dict mapping axis name to coordinate name (None if not found and not required)
    """
    axes = axes or ["x", "y", "z"]
    required = required or axes

    result = {}
    for axis in axes:
        try:
            result[axis] = resolve_coordinate(ds, axis)
        except ValueError:
            if axis in required:
                raise
            result[axis] = None

    return result


def has_coordinate(ds: xr.Dataset, axis: str) -> bool:
    """Check if dataset has a coordinate for the given axis."""
    try:
        resolve_coordinate(ds, axis)
        return True
    except ValueError:
        return False


# =============================================================================
# GRID BUILDER BASE CLASS
# =============================================================================


class GridBuilder(ABC):
    """
    Abstract base class for grid builders.

    Grid builders know how to:
    1. Build a PyVista mesh from an xarray Dataset
    2. Create a bounds mesh (wireframe box) for the data domain

    Subclasses implement specific grid types (rectilinear, curvilinear, etc.)
    """

    @property
    @abstractmethod
    def grid_type(self) -> str:
        """Return human-readable grid type name."""
        ...

    @abstractmethod
    def build_mesh(
        self,
        ds: xr.Dataset,
        varname: Optional[str] = None,
    ) -> pv.DataSet:
        """
        Build PyVista mesh from dataset.

        Args:
            ds: xarray Dataset (should already be time-selected if needed)
            varname: Optional variable to add as scalar data

        Returns:
            PyVista DataSet (RectilinearGrid, StructuredGrid, or UnstructuredGrid)
        """
        ...

    @abstractmethod
    def create_bounds_mesh(self, ds: xr.Dataset) -> pv.PolyData:
        """
        Create a lightweight wireframe mesh showing the data bounds.

        This mesh is used to force the scene bounds to match the data domain.

        Args:
            ds: xarray Dataset

        Returns:
            PyVista PolyData representing the bounding box edges
        """
        ...

    def add_scalar(
        self,
        mesh: pv.DataSet,
        ds: xr.Dataset,
        varname: str,
    ) -> None:
        """
        Add scalar data from dataset to mesh (in-place).

        Args:
            mesh: PyVista mesh to add data to
            ds: xarray Dataset containing the variable
            varname: Name of variable to add
        """
        mesh[varname] = ds[varname].values.ravel(order="F")


# =============================================================================
# RECTILINEAR GRID BUILDER
# =============================================================================


class RectilinearGridBuilder(GridBuilder):
    """
    Builder for rectilinear (orthogonal) grids.

    Handles datasets with 1D coordinate arrays for x, y, z axes.
    This is the most common grid type for atmospheric model output.
    """

    def __init__(self, x_coord: str, y_coord: str, z_coord: str):
        """
        Initialize with resolved coordinate names.

        Args:
            x_coord: Name of x coordinate in dataset
            y_coord: Name of y coordinate in dataset
            z_coord: Name of z coordinate in dataset
        """
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.z_coord = z_coord

    @property
    def grid_type(self) -> str:
        return "rectilinear"

    def build_mesh(
        self,
        ds: xr.Dataset,
        varname: Optional[str] = None,
    ) -> pv.RectilinearGrid:
        """Build PyVista RectilinearGrid from dataset."""
        x = ds[self.x_coord].values
        y = ds[self.y_coord].values
        z = ds[self.z_coord].values

        grid = pv.RectilinearGrid(x, y, z)

        if varname:
            self.add_scalar(grid, ds, varname)

        return grid

    def create_bounds_mesh(self, ds: xr.Dataset) -> pv.PolyData:
        """Create wireframe box at data bounds."""
        x = ds[self.x_coord].values
        y = ds[self.y_coord].values
        z = ds[self.z_coord].values

        bounds = [
            x.min(), x.max(),
            y.min(), y.max(),
            z.min(), z.max(),
        ]

        return pv.Box(bounds=bounds).extract_feature_edges()

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> "RectilinearGridBuilder":
        """
        Create builder from dataset, auto-resolving coordinates.

        Args:
            ds: xarray Dataset with 1D x, y, z coordinates

        Returns:
            RectilinearGridBuilder instance
        """
        coords = resolve_coordinates(ds, ["x", "y", "z"])
        return cls(
            x_coord=coords["x"],
            y_coord=coords["y"],
            z_coord=coords["z"],
        )


# =============================================================================
# CURVILINEAR GRID BUILDER
# =============================================================================


class CurvilinearGridBuilder(GridBuilder):
    """
    Builder for curvilinear (non-orthogonal structured) grids.

    Handles datasets where coordinates are 2D or 3D arrays defining
    the grid point locations. Common for regional atmospheric models.
    """

    def __init__(
        self,
        x_coord: str,
        y_coord: str,
        z_coord: str,
        dims: Tuple[str, str, str],
    ):
        """
        Initialize with resolved coordinate names and dimension names.

        Args:
            x_coord: Name of x coordinate (may be 2D/3D array)
            y_coord: Name of y coordinate (may be 2D/3D array)
            z_coord: Name of z coordinate (may be 2D/3D array)
            dims: Tuple of dimension names (i, j, k order)
        """
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.z_coord = z_coord
        self.dims = dims

    @property
    def grid_type(self) -> str:
        return "curvilinear"

    def build_mesh(
        self,
        ds: xr.Dataset,
        varname: Optional[str] = None,
    ) -> pv.StructuredGrid:
        """Build PyVista StructuredGrid from dataset."""
        x = ds[self.x_coord].values
        y = ds[self.y_coord].values
        z = ds[self.z_coord].values

        # Broadcast coordinates to 3D if needed
        if x.ndim == 1:
            x, y, z = np.meshgrid(x, y, z, indexing="ij")
        elif x.ndim == 2:
            # 2D coordinates with 1D z - broadcast
            z_1d = z if z.ndim == 1 else z[0, 0, :]
            x = np.repeat(x[:, :, np.newaxis], len(z_1d), axis=2)
            y = np.repeat(y[:, :, np.newaxis], len(z_1d), axis=2)
            z = np.broadcast_to(z_1d, x.shape)

        grid = pv.StructuredGrid(x, y, z)

        if varname:
            self.add_scalar(grid, ds, varname)

        return grid

    def create_bounds_mesh(self, ds: xr.Dataset) -> pv.PolyData:
        """Create wireframe box at data bounds."""
        x = ds[self.x_coord].values
        y = ds[self.y_coord].values
        z = ds[self.z_coord].values

        bounds = [
            x.min(), x.max(),
            y.min(), y.max(),
            z.min(), z.max(),
        ]

        return pv.Box(bounds=bounds).extract_feature_edges()

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> "CurvilinearGridBuilder":
        """
        Create builder from dataset, auto-resolving coordinates.

        Args:
            ds: xarray Dataset with 2D/3D coordinate arrays

        Returns:
            CurvilinearGridBuilder instance
        """
        coords = resolve_coordinates(ds, ["x", "y", "z"])

        # Determine dimension names from coordinate shapes
        x_coord = ds[coords["x"]]
        dims = x_coord.dims if x_coord.ndim > 1 else tuple(ds.dims.keys())[:3]

        return cls(
            x_coord=coords["x"],
            y_coord=coords["y"],
            z_coord=coords["z"],
            dims=dims,
        )


# =============================================================================
# UNSTRUCTURED GRID BUILDER
# =============================================================================


class UnstructuredGridBuilder(GridBuilder):
    """
    Builder for unstructured grids and point clouds.

    Handles datasets where points are defined as arrays of coordinates
    without regular structure. Used for particle data, observational
    data, and finite element meshes.
    """

    def __init__(
        self,
        x_coord: str,
        y_coord: str,
        z_coord: str,
        cells: Optional[np.ndarray] = None,
        cell_types: Optional[np.ndarray] = None,
    ):
        """
        Initialize with resolved coordinate names.

        Args:
            x_coord: Name of x coordinate
            y_coord: Name of y coordinate
            z_coord: Name of z coordinate
            cells: Optional cell connectivity array
            cell_types: Optional cell type array
        """
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.z_coord = z_coord
        self.cells = cells
        self.cell_types = cell_types

    @property
    def grid_type(self) -> str:
        return "unstructured"

    def build_mesh(
        self,
        ds: xr.Dataset,
        varname: Optional[str] = None,
    ) -> pv.DataSet:
        """Build PyVista mesh from dataset."""
        x = ds[self.x_coord].values.ravel()
        y = ds[self.y_coord].values.ravel()
        z = ds[self.z_coord].values.ravel()

        points = np.column_stack([x, y, z])

        # Remove NaN points
        valid_mask = ~np.isnan(points).any(axis=1)
        points = points[valid_mask]

        if self.cells is not None and self.cell_types is not None:
            # Create unstructured grid with cells
            mesh = pv.UnstructuredGrid(self.cells, self.cell_types, points)
        else:
            # Create point cloud
            mesh = pv.PolyData(points)

        if varname and varname in ds:
            scalar_data = ds[varname].values.ravel()
            mesh[varname] = scalar_data[valid_mask]

        return mesh

    def create_bounds_mesh(self, ds: xr.Dataset) -> pv.PolyData:
        """Create wireframe box at data bounds."""
        x = ds[self.x_coord].values
        y = ds[self.y_coord].values
        z = ds[self.z_coord].values

        # Handle NaN values
        x_valid = x[~np.isnan(x)]
        y_valid = y[~np.isnan(y)]
        z_valid = z[~np.isnan(z)]

        bounds = [
            x_valid.min(), x_valid.max(),
            y_valid.min(), y_valid.max(),
            z_valid.min(), z_valid.max(),
        ]

        return pv.Box(bounds=bounds).extract_feature_edges()

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> "UnstructuredGridBuilder":
        """
        Create builder from dataset, auto-resolving coordinates.

        Args:
            ds: xarray Dataset with point data

        Returns:
            UnstructuredGridBuilder instance
        """
        coords = resolve_coordinates(ds, ["x", "y", "z"])
        return cls(
            x_coord=coords["x"],
            y_coord=coords["y"],
            z_coord=coords["z"],
        )


# =============================================================================
# GRID TYPE DETECTION
# =============================================================================


def _is_rectilinear(ds: xr.Dataset, coords: Dict[str, str]) -> bool:
    """Check if dataset has rectilinear grid structure."""
    x_coord = ds[coords["x"]]
    y_coord = ds[coords["y"]]
    z_coord = ds[coords["z"]]

    # Rectilinear: all coords are 1D
    return x_coord.ndim == 1 and y_coord.ndim == 1 and z_coord.ndim == 1


def _is_curvilinear(ds: xr.Dataset, coords: Dict[str, str]) -> bool:
    """Check if dataset has curvilinear grid structure."""
    x_coord = ds[coords["x"]]
    y_coord = ds[coords["y"]]

    # Curvilinear: x and y coords are 2D or 3D
    return x_coord.ndim >= 2 or y_coord.ndim >= 2


def detect_grid_type(ds: xr.Dataset) -> GridBuilder:
    """
    Auto-detect grid type and return appropriate builder.

    Detection order:
    1. Rectilinear (1D coordinates)
    2. Curvilinear (2D/3D structured coordinates)
    3. Unstructured (fallback for point data)

    Args:
        ds: xarray Dataset

    Returns:
        Appropriate GridBuilder instance

    Example:
        >>> builder = detect_grid_type(storm_ds)
        >>> mesh = builder.build_mesh(storm_ds, "THETA")
    """
    # First, resolve coordinates
    coords = resolve_coordinates(ds, ["x", "y", "z"])

    # Detect grid type
    if _is_rectilinear(ds, coords):
        return RectilinearGridBuilder(
            x_coord=coords["x"],
            y_coord=coords["y"],
            z_coord=coords["z"],
        )
    elif _is_curvilinear(ds, coords):
        x_coord = ds[coords["x"]]
        dims = x_coord.dims if x_coord.ndim > 1 else tuple(ds.dims.keys())[:3]
        return CurvilinearGridBuilder(
            x_coord=coords["x"],
            y_coord=coords["y"],
            z_coord=coords["z"],
            dims=dims,
        )
    else:
        return UnstructuredGridBuilder(
            x_coord=coords["x"],
            y_coord=coords["y"],
            z_coord=coords["z"],
        )


def get_grid_builder(
    ds: xr.Dataset,
    grid_type: Optional[str] = None,
) -> GridBuilder:
    """
    Get appropriate grid builder for dataset.

    If grid_type is specified, uses that type. Otherwise auto-detects.
    Provides helpful error messages on failure.

    Args:
        ds: xarray Dataset
        grid_type: Optional explicit grid type ("rectilinear", "curvilinear", "unstructured")

    Returns:
        GridBuilder instance

    Example:
        >>> builder = get_grid_builder(ds)
        >>> mesh = builder.build_mesh(ds, "temperature")
    """
    if grid_type is None:
        return detect_grid_type(ds)

    grid_type = grid_type.lower()
    coords = resolve_coordinates(ds, ["x", "y", "z"])

    if grid_type == "rectilinear":
        return RectilinearGridBuilder(
            x_coord=coords["x"],
            y_coord=coords["y"],
            z_coord=coords["z"],
        )
    elif grid_type == "curvilinear":
        x_coord = ds[coords["x"]]
        dims = x_coord.dims if x_coord.ndim > 1 else tuple(ds.dims.keys())[:3]
        return CurvilinearGridBuilder(
            x_coord=coords["x"],
            y_coord=coords["y"],
            z_coord=coords["z"],
            dims=dims,
        )
    elif grid_type == "unstructured":
        return UnstructuredGridBuilder(
            x_coord=coords["x"],
            y_coord=coords["y"],
            z_coord=coords["z"],
        )
    else:
        raise ValueError(
            f"Unknown grid type '{grid_type}'. "
            f"Valid types: 'rectilinear', 'curvilinear', 'unstructured'"
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def merge_bounds_meshes(meshes: List[pv.PolyData]) -> pv.PolyData:
    """
    Merge multiple bounds meshes into a single mesh.

    This is useful when a scene has multiple datasets and we want
    a single bounds mesh that encompasses all of them.

    Args:
        meshes: List of bounds PolyData meshes

    Returns:
        Merged PolyData mesh
    """
    if not meshes:
        return pv.PolyData()

    if len(meshes) == 1:
        return meshes[0]

    # Get overall bounds from all meshes
    all_bounds = [m.bounds for m in meshes]

    xmin = min(b[0] for b in all_bounds)
    xmax = max(b[1] for b in all_bounds)
    ymin = min(b[2] for b in all_bounds)
    ymax = max(b[3] for b in all_bounds)
    zmin = min(b[4] for b in all_bounds)
    zmax = max(b[5] for b in all_bounds)

    bounds = [xmin, xmax, ymin, ymax, zmin, zmax]

    return pv.Box(bounds=bounds).extract_feature_edges()
