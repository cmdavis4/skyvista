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
        "x",
        "X",
        "lon",
        "longitude",
        "Longitude",
        "LONGITUDE",
        "XLONG",
        "XLONG_M",  # WRF
        "xc",
        "XC",  # Some ocean models
        "ni",
        "xi_rho",  # ROMS
    ],
    "y": [
        "y",
        "Y",
        "lat",
        "latitude",
        "Latitude",
        "LATITUDE",
        "XLAT",
        "XLAT_M",  # WRF
        "yc",
        "YC",  # Some ocean models
        "nj",
        "eta_rho",  # ROMS
    ],
    "z": [
        "z",
        "Z",
        "lev",
        "level",
        "Level",
        "LEV",
        "height",
        "Height",
        "HEIGHT",
        "altitude",
        "Altitude",
        "ALTITUDE",
        "zc",
        "ZC",
        "depth",
        "Depth",
        "DEPTH",
        "pressure",
        "Pressure",
        "PRESSURE",
        "sigma",
        "s_rho",  # ROMS
    ],
    "time": [
        "time",
        "Time",
        "TIME",
        "t",
        "T",
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

# Coordinate names that indicate geographic (lat/lon) coordinates.
# These grids should be treated as curvilinear even if coordinates are 1D,
# because lat/lon grids are not orthogonal in Cartesian space.
GEOGRAPHIC_COORD_NAMES: Dict[str, List[str]] = {
    "x": [
        "lon",
        "longitude",
        "Longitude",
        "LONGITUDE",
        "XLONG",
        "XLONG_M",  # WRF
    ],
    "y": [
        "lat",
        "latitude",
        "Latitude",
        "LATITUDE",
        "XLAT",
        "XLAT_M",  # WRF
    ],
}

# CF standard names that indicate geographic coordinates
GEOGRAPHIC_CF_STANDARD_NAMES: List[str] = [
    "longitude",
    "latitude",
    "grid_longitude",
    "grid_latitude",
]

# Coordinate names for spherical/radar coordinates.
# These require transformation to Cartesian coordinates.
SPHERICAL_COORD_NAMES: Dict[str, List[str]] = {
    # Radial distance from origin
    "range": [
        "range",
        "Range",
        "RANGE",
        "r",
        "R",
        "radius",
        "Radius",
        "distance",
        "Distance",
        "slant_range",  # Radar
    ],
    # Azimuth angle (horizontal angle from reference direction)
    "azimuth": [
        "azimuth",
        "Azimuth",
        "AZIMUTH",
        "az",
        "AZ",
        "phi",
        "Phi",
        "PHI",
        "theta",  # Some conventions use theta for azimuth
        "heading",
        "bearing",
    ],
    # Elevation angle (vertical angle from horizontal plane)
    "elevation": [
        "elevation",
        "Elevation",
        "ELEVATION",
        "el",
        "EL",
        "elev",
        "Elev",
        "tilt",
        "Tilt",  # Radar terminology
        "altitude_angle",
    ],
}

# Default Earth radius in meters for geographic coordinate conversion
EARTH_RADIUS_M: float = 6_371_000.0


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
        "  To fix: either add CF-compliant attributes to your coordinates,\n"
        "  or rename coordinates to match common patterns."
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


def _is_geographic_coord(ds: xr.Dataset, coord_name: str, axis: str) -> bool:
    """
    Check if a coordinate represents geographic (lat/lon) data.

    Geographic coordinates need special handling because lat/lon grids
    are not orthogonal in Cartesian space, even when stored as 1D arrays.

    Args:
        ds: xarray Dataset
        coord_name: Name of the coordinate to check
        axis: Which axis this coordinate represents ('x' or 'y')

    Returns:
        True if the coordinate is geographic (lat/lon)
    """
    if axis not in ("x", "y"):
        return False

    coord = ds[coord_name]

    # Check CF standard_name attribute
    standard_name = coord.attrs.get("standard_name", "")
    if standard_name in GEOGRAPHIC_CF_STANDARD_NAMES:
        return True

    # Check units attribute (degrees indicate geographic)
    units = coord.attrs.get("units", "").lower()
    if "degree" in units:
        return True

    # Check coordinate name against known geographic names
    geographic_names = GEOGRAPHIC_COORD_NAMES.get(axis, [])
    if coord_name in geographic_names:
        return True

    return False


def is_geographic_grid(ds: xr.Dataset, coords: Dict[str, str]) -> bool:
    """
    Check if dataset uses geographic (lat/lon) coordinates.

    Args:
        ds: xarray Dataset
        coords: Dict mapping axis names to coordinate names

    Returns:
        True if x or y coordinates are geographic
    """
    x_is_geo = _is_geographic_coord(ds, coords["x"], "x")
    y_is_geo = _is_geographic_coord(ds, coords["y"], "y")

    # If either x or y is geographic, treat as geographic grid
    return x_is_geo or y_is_geo


def resolve_spherical_coordinate(ds: xr.Dataset, axis: str) -> Optional[str]:
    """
    Find coordinate for spherical axis (range, azimuth, elevation).

    Args:
        ds: xarray Dataset
        axis: Spherical axis name ("range", "azimuth", or "elevation")

    Returns:
        Coordinate name if found, None otherwise
    """
    if axis not in SPHERICAL_COORD_NAMES:
        return None

    aliases = SPHERICAL_COORD_NAMES[axis]

    # Check coordinates and data variables
    for name in aliases:
        if name in ds.coords or name in ds.data_vars:
            return name

    return None


def resolve_spherical_coordinates(
    ds: xr.Dataset,
) -> Optional[Dict[str, str]]:
    """
    Try to resolve all three spherical coordinates.

    Args:
        ds: xarray Dataset

    Returns:
        Dict with range, azimuth, elevation coordinate names, or None if not all found
    """
    range_coord = resolve_spherical_coordinate(ds, "range")
    azimuth_coord = resolve_spherical_coordinate(ds, "azimuth")
    elevation_coord = resolve_spherical_coordinate(ds, "elevation")

    if range_coord and azimuth_coord and elevation_coord:
        return {
            "range": range_coord,
            "azimuth": azimuth_coord,
            "elevation": elevation_coord,
        }

    return None


def is_spherical_grid(ds: xr.Dataset) -> bool:
    """
    Check if dataset uses spherical/radar coordinates.

    Args:
        ds: xarray Dataset

    Returns:
        True if range, azimuth, and elevation coordinates are found
    """
    return resolve_spherical_coordinates(ds) is not None


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
            x.min(),
            x.max(),
            y.min(),
            y.max(),
            z.min(),
            z.max(),
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
            x.min(),
            x.max(),
            y.min(),
            y.max(),
            z.min(),
            z.max(),
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
            x_valid.min(),
            x_valid.max(),
            y_valid.min(),
            y_valid.max(),
            z_valid.min(),
            z_valid.max(),
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
# GEOGRAPHIC GRID BUILDER
# =============================================================================


class GeographicGridBuilder(GridBuilder):
    """
    Builder for geographic (lat/lon) grids.

    Converts latitude/longitude/altitude coordinates to 3D Cartesian
    coordinates on a sphere. This properly handles the fact that lat/lon
    grids are not orthogonal in Cartesian space.

    The conversion formula:
        x = (R + alt) * cos(lat) * cos(lon)
        y = (R + alt) * cos(lat) * sin(lon)
        z = (R + alt) * sin(lat)

    Where R is Earth radius and alt is altitude above the surface.
    """

    def __init__(
        self,
        lon_coord: str,
        lat_coord: str,
        alt_coord: str,
        earth_radius: float = EARTH_RADIUS_M,
        altitude_scale: float = 1.0,
    ):
        """
        Initialize with resolved coordinate names.

        Args:
            lon_coord: Name of longitude coordinate (degrees)
            lat_coord: Name of latitude coordinate (degrees)
            alt_coord: Name of altitude/level coordinate
            earth_radius: Earth radius for conversion (default: 6,371,000 m)
            altitude_scale: Scale factor for altitude (for exaggeration)
        """
        self.lon_coord = lon_coord
        self.lat_coord = lat_coord
        self.alt_coord = alt_coord
        self.earth_radius = earth_radius
        self.altitude_scale = altitude_scale

    @property
    def grid_type(self) -> str:
        return "geographic"

    def _to_cartesian(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        alt: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert lon/lat/alt to Cartesian coordinates.

        Args:
            lon: Longitude in degrees
            lat: Latitude in degrees
            alt: Altitude (same units as earth_radius)

        Returns:
            Tuple of (x, y, z) Cartesian coordinates
        """
        # Convert to radians
        lon_rad = np.deg2rad(lon)
        lat_rad = np.deg2rad(lat)

        # Apply altitude scaling
        r = self.earth_radius + alt * self.altitude_scale

        # Convert to Cartesian
        x = r * np.cos(lat_rad) * np.cos(lon_rad)
        y = r * np.cos(lat_rad) * np.sin(lon_rad)
        z = r * np.sin(lat_rad)

        return x, y, z

    def build_mesh(
        self,
        ds: xr.Dataset,
        varname: Optional[str] = None,
    ) -> pv.StructuredGrid:
        """Build PyVista StructuredGrid from geographic dataset."""
        lon = ds[self.lon_coord].values
        lat = ds[self.lat_coord].values
        alt = ds[self.alt_coord].values

        # Broadcast coordinates to 3D if needed
        if lon.ndim == 1 and lat.ndim == 1 and alt.ndim == 1:
            lon, lat, alt = np.meshgrid(lon, lat, alt, indexing="ij")
        elif lon.ndim == 2 and alt.ndim == 1:
            # 2D lon/lat with 1D altitude - broadcast
            lon = np.repeat(lon[:, :, np.newaxis], len(alt), axis=2)
            lat = np.repeat(lat[:, :, np.newaxis], len(alt), axis=2)
            alt = np.broadcast_to(alt, lon.shape)

        # Convert to Cartesian
        x, y, z = self._to_cartesian(lon, lat, alt)

        grid = pv.StructuredGrid(x, y, z)

        if varname:
            self.add_scalar(grid, ds, varname)

        return grid

    def create_bounds_mesh(self, ds: xr.Dataset) -> pv.PolyData:
        """Create wireframe at data bounds in Cartesian space."""
        lon = ds[self.lon_coord].values
        lat = ds[self.lat_coord].values
        alt = ds[self.alt_coord].values

        # Get corner coordinates
        lon_min, lon_max = lon.min(), lon.max()
        lat_min, lat_max = lat.min(), lat.max()
        alt_min, alt_max = alt.min(), alt.max()

        # Create corner points
        corners_lon = [
            lon_min,
            lon_max,
            lon_min,
            lon_max,
            lon_min,
            lon_max,
            lon_min,
            lon_max,
        ]
        corners_lat = [
            lat_min,
            lat_min,
            lat_max,
            lat_max,
            lat_min,
            lat_min,
            lat_max,
            lat_max,
        ]
        corners_alt = [
            alt_min,
            alt_min,
            alt_min,
            alt_min,
            alt_max,
            alt_max,
            alt_max,
            alt_max,
        ]

        x, y, z = self._to_cartesian(
            np.array(corners_lon),
            np.array(corners_lat),
            np.array(corners_alt),
        )

        bounds = [x.min(), x.max(), y.min(), y.max(), z.min(), z.max()]
        return pv.Box(bounds=bounds).extract_feature_edges()

    @classmethod
    def from_dataset(
        cls,
        ds: xr.Dataset,
        earth_radius: float = EARTH_RADIUS_M,
        altitude_scale: float = 1.0,
    ) -> "GeographicGridBuilder":
        """
        Create builder from dataset, auto-resolving coordinates.

        Args:
            ds: xarray Dataset with lon/lat/alt coordinates
            earth_radius: Earth radius for conversion
            altitude_scale: Scale factor for altitude

        Returns:
            GeographicGridBuilder instance
        """
        coords = resolve_coordinates(ds, ["x", "y", "z"])
        return cls(
            lon_coord=coords["x"],
            lat_coord=coords["y"],
            alt_coord=coords["z"],
            earth_radius=earth_radius,
            altitude_scale=altitude_scale,
        )


# =============================================================================
# SPHERICAL GRID BUILDER
# =============================================================================


class SphericalGridBuilder(GridBuilder):
    """
    Builder for spherical/radar coordinate grids.

    Converts range/azimuth/elevation coordinates to 3D Cartesian
    coordinates centered at the instrument location.

    The conversion formula (radar convention):
        x = range * cos(elevation) * sin(azimuth)
        y = range * cos(elevation) * cos(azimuth)
        z = range * sin(elevation)

    Where azimuth is measured clockwise from north (y-axis) and
    elevation is measured up from the horizontal plane.
    """

    def __init__(
        self,
        range_coord: str,
        azimuth_coord: str,
        elevation_coord: str,
        azimuth_offset: float = 0.0,
    ):
        """
        Initialize with resolved coordinate names.

        Args:
            range_coord: Name of range/radial distance coordinate
            azimuth_coord: Name of azimuth angle coordinate (degrees)
            elevation_coord: Name of elevation angle coordinate (degrees)
            azimuth_offset: Offset to add to azimuth (degrees), for adjusting
                           reference direction
        """
        self.range_coord = range_coord
        self.azimuth_coord = azimuth_coord
        self.elevation_coord = elevation_coord
        self.azimuth_offset = azimuth_offset

    @property
    def grid_type(self) -> str:
        return "spherical"

    def _to_cartesian(
        self,
        range_vals: np.ndarray,
        azimuth: np.ndarray,
        elevation: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert spherical to Cartesian coordinates.

        Uses radar convention where azimuth is clockwise from north (y-axis).

        Args:
            range_vals: Radial distance from origin
            azimuth: Azimuth angle in degrees (clockwise from north)
            elevation: Elevation angle in degrees (up from horizontal)

        Returns:
            Tuple of (x, y, z) Cartesian coordinates
        """
        # Convert to radians
        az_rad = np.deg2rad(azimuth + self.azimuth_offset)
        el_rad = np.deg2rad(elevation)

        # Convert to Cartesian (radar convention: azimuth from north/y-axis)
        x = range_vals * np.cos(el_rad) * np.sin(az_rad)
        y = range_vals * np.cos(el_rad) * np.cos(az_rad)
        z = range_vals * np.sin(el_rad)

        return x, y, z

    def build_mesh(
        self,
        ds: xr.Dataset,
        varname: Optional[str] = None,
    ) -> pv.StructuredGrid:
        """Build PyVista StructuredGrid from spherical dataset."""
        range_vals = ds[self.range_coord].values
        azimuth = ds[self.azimuth_coord].values
        elevation = ds[self.elevation_coord].values

        # Broadcast coordinates to match data shape
        # Common radar data layout: (azimuth, range) per elevation sweep
        if range_vals.ndim == 1 and azimuth.ndim <= 2 and elevation.ndim <= 2:
            # Determine target shape from the data
            if azimuth.ndim == 1:
                # Simple case: 1D azimuth, broadcast to 3D
                range_vals, azimuth, elevation = np.meshgrid(
                    range_vals, azimuth, elevation, indexing="ij"
                )
            else:
                # azimuth/elevation are 2D (volume_scan, time) - need to handle carefully
                # This is the radar case mentioned by the user
                # We need to broadcast range against the azimuth/elevation grid
                n_range = len(range_vals)
                az_shape = azimuth.shape

                # Create output arrays
                out_shape = (n_range,) + az_shape
                range_3d = np.broadcast_to(
                    range_vals[:, np.newaxis, np.newaxis], out_shape
                )
                azimuth_3d = np.broadcast_to(azimuth[np.newaxis, :, :], out_shape)
                elevation_3d = np.broadcast_to(elevation[np.newaxis, :, :], out_shape)

                range_vals = range_3d
                azimuth = azimuth_3d
                elevation = elevation_3d

        # Convert to Cartesian
        x, y, z = self._to_cartesian(range_vals, azimuth, elevation)

        grid = pv.StructuredGrid(x, y, z)

        if varname:
            self.add_scalar(grid, ds, varname)

        return grid

    def create_bounds_mesh(self, ds: xr.Dataset) -> pv.PolyData:
        """Create wireframe at data bounds in Cartesian space."""
        range_vals = ds[self.range_coord].values
        azimuth = ds[self.azimuth_coord].values
        elevation = ds[self.elevation_coord].values

        # Get coordinate bounds
        r_min, r_max = range_vals.min(), range_vals.max()
        az_min, az_max = azimuth.min(), azimuth.max()
        el_min, el_max = elevation.min(), elevation.max()

        # Sample points around the bounds to find Cartesian extent
        n_samples = 20
        az_samples = np.linspace(az_min, az_max, n_samples)
        el_samples = np.linspace(el_min, el_max, n_samples)

        # Generate all corner/edge combinations
        all_x, all_y, all_z = [], [], []
        for r in [r_min, r_max]:
            for az in az_samples:
                for el in [el_min, el_max]:
                    x, y, z = self._to_cartesian(
                        np.array([r]), np.array([az]), np.array([el])
                    )
                    all_x.append(x[0])
                    all_y.append(y[0])
                    all_z.append(z[0])

        bounds = [
            min(all_x),
            max(all_x),
            min(all_y),
            max(all_y),
            min(all_z),
            max(all_z),
        ]

        return pv.Box(bounds=bounds).extract_feature_edges()

    @classmethod
    def from_dataset(
        cls,
        ds: xr.Dataset,
        azimuth_offset: float = 0.0,
    ) -> "SphericalGridBuilder":
        """
        Create builder from dataset, auto-resolving coordinates.

        Args:
            ds: xarray Dataset with range/azimuth/elevation coordinates
            azimuth_offset: Offset to add to azimuth angles

        Returns:
            SphericalGridBuilder instance
        """
        coords = resolve_spherical_coordinates(ds)
        if coords is None:
            raise ValueError(
                "Could not resolve spherical coordinates. "
                "Dataset must have range, azimuth, and elevation coordinates."
            )

        return cls(
            range_coord=coords["range"],
            azimuth_coord=coords["azimuth"],
            elevation_coord=coords["elevation"],
            azimuth_offset=azimuth_offset,
        )


# =============================================================================
# GRID TYPE DETECTION
# =============================================================================


def _is_rectilinear(ds: xr.Dataset, coords: Dict[str, str]) -> bool:
    """
    Check if dataset has rectilinear grid structure.

    A grid is rectilinear if:
    1. All coordinates are 1D arrays
    2. The coordinates are NOT geographic (lat/lon)

    Geographic grids with 1D lat/lon are treated as curvilinear because
    lat/lon grids are not orthogonal in Cartesian space.
    """
    x_coord = ds[coords["x"]]
    y_coord = ds[coords["y"]]
    z_coord = ds[coords["z"]]

    # Must have 1D coordinates
    if not (x_coord.ndim == 1 and y_coord.ndim == 1 and z_coord.ndim == 1):
        return False

    # Geographic coordinates should be treated as curvilinear
    if is_geographic_grid(ds, coords):
        return False

    return True


def _is_curvilinear(ds: xr.Dataset, coords: Dict[str, str]) -> bool:
    """
    Check if dataset has curvilinear grid structure (non-geographic).

    A grid is curvilinear if x or y coordinates are 2D or 3D arrays.
    Geographic grids are handled separately by GeographicGridBuilder.
    """
    x_coord = ds[coords["x"]]
    y_coord = ds[coords["y"]]

    # Explicitly 2D/3D coordinates (and NOT geographic - those use GeographicGridBuilder)
    if x_coord.ndim >= 2 or y_coord.ndim >= 2:
        # Geographic curvilinear grids should use GeographicGridBuilder
        if is_geographic_grid(ds, coords):
            return False
        return True

    return False


def detect_grid_type(ds: xr.Dataset) -> GridBuilder:
    """
    Auto-detect grid type and return appropriate builder.

    Detection order:
    1. Spherical (radar: range/azimuth/elevation coordinates)
    2. Geographic (lat/lon coordinates - converted to sphere)
    3. Rectilinear (1D Cartesian coordinates)
    4. Curvilinear (2D/3D non-geographic structured coordinates)
    5. Unstructured (fallback for point data)

    Args:
        ds: xarray Dataset

    Returns:
        Appropriate GridBuilder instance

    Example:
        >>> builder = detect_grid_type(storm_ds)
        >>> mesh = builder.build_mesh(storm_ds, "THETA")
    """
    # First, check for spherical/radar coordinates
    spherical_coords = resolve_spherical_coordinates(ds)
    if spherical_coords is not None:
        return SphericalGridBuilder(
            range_coord=spherical_coords["range"],
            azimuth_coord=spherical_coords["azimuth"],
            elevation_coord=spherical_coords["elevation"],
        )

    # Try to resolve Cartesian/geographic coordinates
    try:
        coords = resolve_coordinates(ds, ["x", "y", "z"])
    except ValueError:
        raise ValueError(
            "Could not detect grid type. Dataset must have either:\n"
            "  - Cartesian coordinates (x, y, z or similar)\n"
            "  - Geographic coordinates (lon, lat, altitude)\n"
            "  - Spherical coordinates (range, azimuth, elevation)"
        )

    # Check for geographic grid (lat/lon) - use GeographicGridBuilder
    if is_geographic_grid(ds, coords):
        return GeographicGridBuilder(
            lon_coord=coords["x"],
            lat_coord=coords["y"],
            alt_coord=coords["z"],
        )

    # Detect Cartesian grid type
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
        grid_type: Optional explicit grid type:
            - "rectilinear": Regular orthogonal Cartesian grid
            - "curvilinear": Non-orthogonal structured grid (2D/3D coords)
            - "geographic": Lat/lon coordinates on a sphere
            - "spherical": Radar-style range/azimuth/elevation
            - "unstructured": Point cloud or unstructured mesh

    Returns:
        GridBuilder instance

    Example:
        >>> builder = get_grid_builder(ds)
        >>> mesh = builder.build_mesh(ds, "temperature")
    """
    if grid_type is None:
        return detect_grid_type(ds)

    grid_type = grid_type.lower()

    # Handle spherical coordinates
    if grid_type == "spherical":
        spherical_coords = resolve_spherical_coordinates(ds)
        if spherical_coords is None:
            raise ValueError(
                "Cannot use 'spherical' grid type: dataset does not have "
                "range, azimuth, and elevation coordinates."
            )
        return SphericalGridBuilder(
            range_coord=spherical_coords["range"],
            azimuth_coord=spherical_coords["azimuth"],
            elevation_coord=spherical_coords["elevation"],
        )

    # For other types, we need x/y/z coordinates
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
    elif grid_type == "geographic":
        return GeographicGridBuilder(
            lon_coord=coords["x"],
            lat_coord=coords["y"],
            alt_coord=coords["z"],
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
            "Valid types: 'rectilinear', 'curvilinear', 'geographic', "
            "'spherical', 'unstructured'"
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
