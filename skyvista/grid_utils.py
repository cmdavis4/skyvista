"""
Shared utilities for grid and data manipulation.

These utilities are used by multiple VarSpec classes to avoid code duplication.
"""

from typing import Any

import pyvista as pv
import xarray as xr

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

    Expects the dataset to have 'x', 'y', 'z' coordinates.

    Args:
        ds: xarray Dataset with x, y, z coordinates

    Returns:
        PyVista RectilinearGrid
    """
    return pv.RectilinearGrid(
        ds["x"].values,
        ds["y"].values,
        ds["z"].values,
    )


@enforce_dimension_order
def add_scalar_to_grid(
    grid: pv.RectilinearGrid,
    ds: xr.Dataset,
    varname: str,
) -> None:
    """
    Add a scalar field from dataset to grid (in-place).

    Args:
        grid: PyVista RectilinearGrid to add scalar to
        ds: xarray Dataset containing the variable
        varname: Name of the variable to add
    """
    grid[varname] = ds[varname].values.ravel(order="F")
