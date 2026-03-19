"""
Shared utilities for grid and data manipulation.

These utilities are used by multiple VarSpec classes to avoid code duplication.
For advanced grid handling, see the grids module.
"""

from typing import Any, Optional

import pyvista as pv
import xarray as xr

from .grids import (
    GridBuilder,
    detect_grid_type,
    get_grid_builder,
    resolve_coordinate,
    resolve_coordinates,
)

DIMENSION_ORDER = ("time", "x", "y", "z")


def transpose_if_xr_type(maybe_xr):
    if isinstance(maybe_xr, (xr.Dataset, xr.DataArray)):
        maybe_xr = maybe_xr.transpose(
            *[x for x in DIMENSION_ORDER if x in maybe_xr.dims]
        )
    return maybe_xr


def enforce_dimension_order(func):
    def wrapper(*args, **kwargs):
        wrapped_args = [transpose_if_xr_type(x) for x in args]
        wrapped_kwargs = {k: transpose_if_xr_type(v) for k, v in kwargs.items()}
        return func(*wrapped_args, **wrapped_kwargs)

    return wrapper


@enforce_dimension_order
def select_time(ds: xr.Dataset, time: Any) -> xr.Dataset:
    """
    Select a single time from dataset, or return as-is if no time dimension.

    Args:
        ds: xarray Dataset
        time: Time value to select, or None for no selection

    Returns:
        Dataset with single time (or original if no time dimension)
    """
    if time is not None and "time" in ds.dims:
        return ds.sel(time=time)
    return ds


@enforce_dimension_order
def build_rectilinear_grid(ds: xr.Dataset) -> pv.RectilinearGrid:
    """
    Build PyVista RectilinearGrid from xarray Dataset.

    Now uses the grids module for coordinate resolution.

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


@enforce_dimension_order
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
    builder = get_grid_builder(ds, grid_type)
    return builder.build_mesh(ds, varname)


@enforce_dimension_order
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
