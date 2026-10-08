"""
Load the hydrometeor fields of a RAMS analysis/lite file into xarray.

RAMS writes each output time as an HDF5 file (``a-L-...-g1.h5`` for lite
output, ``a-A-...-g1.h5`` for full analysis output) plus a ``-head.txt``
header. The HDF5 file holds bare 3-D arrays in (z, y, x) order. The header
holds the grid coordinates as ``__xtn01``, ``__ytn01`` and ``__ztn01``
records. This module joins the two into a ``Dataset`` that
``skyvista.Scene.add_clouds`` understands: RAMS mixing ratios (``R?P``),
number concentrations (``C?P``), and the base-state air density ``DN0``.

ZFP-compressed output
---------------------
Lite files are often written with the ZFP lossy compressor (HDF5 filter
32013), which stock HDF5 cannot decode. Either:

* install the ``hdf5plugin`` package in the running environment (importing
  it registers the filter), or
* point ``HDF5_PLUGIN_PATH`` at a directory containing the H5Z-ZFP plugin,
  e.g. the ``plugins`` directory of an ``hdf5plugin`` installed elsewhere.
  This works with the pixi ``blender`` environment, whose conda HDF5 cannot
  take ``hdf5plugin`` itself.
"""

import os
import re
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import xarray as xr

# Mixing ratios (kg/kg) and number concentrations (#/kg) of the RAMS
# hydrometeor categories, plus the base-state air density (kg m^-3). Drizzle
# (RDP/CDP) is only present when IDRIZ > 0; absent variables are skipped.
RAMS_CLOUD_VARIABLES = (
    "RCP",
    "RDP",
    "RRP",
    "RPP",
    "RSP",
    "RAP",
    "RGP",
    "RHP",
    "CCP",
    "CDP",
    "CRP",
    "CPP",
    "CSP",
    "CAP",
    "CGP",
    "CHP",
    "DN0",
)

# Pattern of the model time in RAMS output names: a-L-YYYY-MM-DD-HHMMSS-g1.h5
RAMS_TIME_PATTERN = re.compile(
    r"a-[AL]-(\d{4}-\d{2}-\d{2})-(\d{2})(\d{2})(\d{2})-g\d+\.h5$"
)


def register_zfp_filter() -> None:
    """
    Make the ZFP HDF5 filter available, or explain how to.

    Importing ``hdf5plugin`` registers every filter it bundles. If it is not
    installed we fall back on ``HDF5_PLUGIN_PATH``; if that is unset too, the
    first read of a ZFP dataset will fail, so we warn early with the fix.
    """
    try:
        import hdf5plugin  # noqa: F401  (import registers the filters)
    except ImportError:
        if not os.environ.get("HDF5_PLUGIN_PATH"):
            import warnings

            warnings.warn(
                "Neither hdf5plugin nor HDF5_PLUGIN_PATH is available; ZFP-"
                "compressed RAMS lite files will fail to read. Install "
                "hdf5plugin, or set HDF5_PLUGIN_PATH to its 'plugins' "
                "directory.",
                stacklevel=2,
            )


def read_header_coordinates(header_path: Path) -> Dict[str, np.ndarray]:
    """
    Read the x, y, z grid coordinates (m) from a RAMS ``-head.txt`` file.

    Each coordinate is stored as ``__<name>01  <count>  <value> <value> ...``
    in a whitespace-separated stream, so we locate the record name and read
    the ``count`` numbers that follow.
    """
    header_tokens = Path(header_path).read_text().split()

    def header_record(record_name: str) -> np.ndarray:
        record_index = header_tokens.index(record_name)
        value_count = int(header_tokens[record_index + 1])
        first_value_index = record_index + 2
        return np.array(
            header_tokens[first_value_index : first_value_index + value_count],
            dtype=float,
        )

    return {
        "x": header_record("__xtn01"),
        "y": header_record("__ytn01"),
        "z": header_record("__ztn01"),
    }


def model_time_seconds(data_path: Path) -> float:
    """Seconds since the start of the run, parsed from the file name."""
    match = RAMS_TIME_PATTERN.search(Path(data_path).name)
    if match is None:
        raise ValueError(f"Not a RAMS output file name: {Path(data_path).name}")
    hours, minutes, seconds = (int(group) for group in match.groups()[1:])
    return float(3600 * hours + 60 * minutes + seconds)


def load_rams_lite_hydrometeors(
    data_path: Path,
    top_height: Optional[float] = None,
    bottom_height: Optional[float] = 0.0,
    variables: Sequence[str] = RAMS_CLOUD_VARIABLES,
) -> xr.Dataset:
    """
    Read the hydrometeor variables of one RAMS output time into a Dataset.

    Args:
        data_path: The ``a-L-...-g1.h5`` (or ``a-A-...``) file.
        top_height: Drop levels above this height (m). Clouds rarely reach
            the model top, and skipping the stratosphere saves memory and I/O.
        bottom_height: Drop levels below this height (m). RAMS's lowest
            thermodynamic level lies below the surface (a boundary level, at
            -dz/2), so the default 0 removes it. Rendered, it would poke
            through the ground plane as flat patches. None keeps every level.
        variables: Variables to read; ones absent from the file are skipped.

    Returns:
        Dataset with dims (z, y, x), coordinates in meters, and a scalar
        ``time`` coordinate in seconds since the start of the run.
    """
    import h5py

    register_zfp_filter()
    data_path = Path(data_path)
    header_path = data_path.with_name(data_path.name.replace("-g1.h5", "-head.txt"))
    coordinates = read_header_coordinates(header_path)

    # Range of levels to keep: [first_level, last_level_exclusive).
    first_level = 0
    if bottom_height is not None:
        first_level = int(np.searchsorted(coordinates["z"], bottom_height, side="left"))
    last_level_exclusive = len(coordinates["z"])
    if top_height is not None:
        last_level_exclusive = int(
            np.searchsorted(coordinates["z"], top_height, side="right")
        )
    level_slice = slice(first_level, last_level_exclusive)

    # Read each present variable's lowest n_levels levels.
    data_variables = {}
    with h5py.File(data_path, "r") as h5_file:
        for variable_name in variables:
            if variable_name not in h5_file:
                continue
            values = h5_file[variable_name][level_slice].astype(np.float32)
            data_variables[variable_name] = (("z", "y", "x"), values)

    return xr.Dataset(
        data_variables,
        coords={
            "z": coordinates["z"][level_slice],
            "y": coordinates["y"],
            "x": coordinates["x"],
            "time": model_time_seconds(data_path),
        },
    )
