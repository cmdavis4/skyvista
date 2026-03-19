"""
Shared utilities for grid and data manipulation.

These utilities are used by multiple VarSpec classes to avoid code duplication.
For advanced grid handling, see the grids module.
"""

from typing import Any, List, Optional

import pyvista as pv
import xarray as xr

from .grids import (
    GridBuilder,
    detect_grid_type,
    get_grid_builder,
    resolve_coordinate,
    resolve_coordinates,
)


def normalize_dimension_order(
    ds: xr.Dataset,
    required_axes: Optional[List[str]] = None,
) -> xr.Dataset:
    """
    Normalize dimension order for consistent data access.

    Uses resolved coordinate names to determine the correct dimension order,
    handling datasets with different naming conventions (x/y/z, lon/lat/lev, etc.).

    The target order is: (time, x-axis, y-axis, z-axis) but using the actual
    dimension names present in the dataset.

    Args:
        ds: xarray Dataset
        required_axes: Which axes to include in ordering (default: ["x", "y", "z"])

    Returns:
        Dataset with dimensions transposed to canonical order
    """
    required_axes = required_axes or ["x", "y", "z"]

    # Resolve coordinate names for each axis
    resolved = {}
    for axis in required_axes:
        try:
            resolved[axis] = resolve_coordinate(ds, axis)
        except ValueError:
            # Axis not found - skip it
            pass

    # Build target dimension order
    # Start with time if present
    target_dims = []

    # Check for time dimension
    try:
        time_coord = resolve_coordinate(ds, "time")
        if time_coord in ds.dims:
            target_dims.append(time_coord)
    except ValueError:
        # No time dimension
        if "time" in ds.dims:
            target_dims.append("time")

    # Add spatial dimensions in x, y, z order
    for axis in ["x", "y", "z"]:
        if axis in resolved:
            coord_name = resolved[axis]
            # The dimension might be named differently from the coordinate
            # For 1D coords, the dimension name is usually the coord name
            if coord_name in ds.dims:
                target_dims.append(coord_name)
            else:
                # For multi-dimensional coords, we need to find the actual dimension
                coord = ds[coord_name]
                for dim in coord.dims:
                    if dim not in target_dims:
                        target_dims.append(dim)

    # Only include dimensions that actually exist in the dataset
    target_dims = [d for d in target_dims if d in ds.dims]

    # Add any remaining dimensions not yet included (preserve their relative order)
    for dim in ds.dims:
        if dim not in target_dims:
            target_dims.append(dim)

    # Only transpose if order is different
    current_dims = list(ds.sizes.keys())
    if target_dims != current_dims:
        return ds.transpose(*target_dims)

    return ds


def select_time(ds: xr.Dataset, time: Any) -> xr.Dataset:
    """
    Select a single time from dataset, or return as-is if no time dimension.

    Args:
        ds: xarray Dataset
        time: Time value to select, or None for no selection

    Returns:
        Dataset with single time (or original if no time dimension)
    """
    if time is None:
        return ds

    # Try to resolve time coordinate name
    try:
        time_coord = resolve_coordinate(ds, "time")
        if time_coord in ds.dims:
            return ds.sel({time_coord: time})
    except ValueError:
        pass

    # Fallback to literal "time" dimension
    if "time" in ds.dims:
        return ds.sel(time=time)

    return ds


def build_rectilinear_grid(ds: xr.Dataset) -> pv.RectilinearGrid:
    """
    Build PyVista RectilinearGrid from xarray Dataset.

    Uses the grids module for coordinate resolution.

    Args:
        ds: xarray Dataset with x, y, z coordinates

    Returns:
        PyVista RectilinearGrid
    """
    coords = resolve_coordinates(ds, ["x", "y", "z"])
    return pv.RectilinearGrid(
        ds[coords["x"]].values,
        ds[coords["y"]].values,
        ds[coords["z"]].values,
    )


def build_grid(
    ds: xr.Dataset,
    varname: Optional[str] = None,
    grid_type: Optional[str] = None,
) -> pv.DataSet:
    """
    Build PyVista mesh from xarray Dataset.

    Auto-detects grid type (rectilinear, curvilinear, unstructured)
    and uses the appropriate builder.

    Args:
        ds: xarray Dataset
        varname: Optional variable to add as scalar data
        grid_type: Optional explicit grid type

    Returns:
        PyVista mesh (RectilinearGrid, StructuredGrid, or PolyData)
    """
    # Normalize dimension order before building grid
    ds = normalize_dimension_order(ds)
    builder = get_grid_builder(ds, grid_type)
    return builder.build_mesh(ds, varname)


def add_scalar_to_grid(
    grid: pv.DataSet,
    ds: xr.Dataset,
    varname: str,
) -> None:
    """
    Add a scalar field from dataset to grid (in-place).

    Args:
        grid: PyVista mesh to add scalar to
        ds: xarray Dataset containing the variable
        varname: Name of the variable to add
    """
    # Normalize dimension order for consistent raveling
    ds = normalize_dimension_order(ds)
    grid[varname] = ds[varname].values.ravel(order="F")


def create_bounds_mesh(
    ds: xr.Dataset,
    grid_type: Optional[str] = None,
) -> pv.PolyData:
    """
    Create a lightweight bounds mesh for the dataset.

    This mesh can be used to force scene bounds to match data domain.

    Args:
        ds: xarray Dataset
        grid_type: Optional explicit grid type

    Returns:
        PyVista PolyData wireframe box
    """
    builder = get_grid_builder(ds, grid_type)
    return builder.create_bounds_mesh(ds)
