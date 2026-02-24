"""VDB (VolumeDB) conversion utilities for atmospheric data.

This module provides functions for converting atmospheric model data
to VDB format for use in Blender and other 3D applications.
"""

import numpy as np
from tqdm import tqdm
from pathlib import Path
import subprocess
from jinja2 import Template
from typing import Any, Dict, List, Optional, Union
import xarray as xr

from skyutils.utils import (
    NUMERICAL_DT_FORMAT,
    raise_if_not_evenly_spaced_,
    dt_to_str,
    to_kv_str,
)
from skyutils.types_skyutils import PathLike


def rams_ds_to_npy(
    ds: Any,
    _vars: List[str],
    output_dir: PathLike,
    dz: float = 500,
    global_scale: float = 0.001,
) -> Dict[str, Any]:
    """
    Convert RAMS dataset to NPY files for VDB conversion.

    Args:
        ds: xarray Dataset containing RAMS data
        _vars: List of variable names to process
        output_dir: Directory to save NPY files
        dz: Vertical grid spacing for interpolation
        global_scale: Global scaling factor for coordinates

    Returns:
        Dictionary containing conversion metadata
    """
    data = {}
    for time_val in tqdm(ds.time.values):
        # Extract data for this time step
        this_time_data = ds.sel({"time": time_val})
        for this_var in _vars:
            # Drop the fictitious z level
            this_var_data = this_time_data[this_var].sel({"z": slice(1, None)})
            # Need to interpolate this onto a regular grid, because
            # vdb requires that

            this_var_data = this_var_data.interp(
                {
                    coord_var: np.arange(
                        this_var_data[coord_var].values[0],
                        this_var_data[coord_var].values[-1],
                        step=dz,
                    )
                    for coord_var in ["x", "y", "z"]
                }
                # {
                #     "z": np.arange(
                #         this_var_data["z"].values[0],
                #         this_var_data["z"].values[-1],
                #         step=dz,
                #     )
                # }
            )
            # Create filename
            this_filename = to_kv_str({
                "dt": dt_to_str(time_val, date_format=NUMERICAL_DT_FORMAT),
                "category": this_var,
                "varname": this_var,
            })
            # Save as numpy binary file
            this_output_path = (Path(output_dir) / this_filename).with_suffix(".npy")
            np.save(str(this_output_path), this_var_data)
            data[this_output_path] = this_var_data

    # Also need to write out a transform based on the grid dimensions
    # Get the grid spacing in all dimensions
    # grid_spacings = {}
    # for this_var in ["x", "y", "z"]:
    #     # Just handle the case of dx and dy being different, bc why not
    #     # Use whatever the last value of this_var_data is, since we interpolated
    #     # to constant z there
    #     this_coord_var_values = this_var_data[this_var].values
    #     raise_if_not_evenly_spaced(this_coord_var_values)
    #     grid_spacings[this_var] = this_coord_var_values[1] - this_coord_var_values[0]
    scaled_grid_step = dz * global_scale
    transform = np.array(
        [
            [scaled_grid_step, 0.0, 0.0, 0.0],
            [0.0, scaled_grid_step, 0.0, 0.0],
            [0.0, 0.0, scaled_grid_step, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    np.save(Path(output_dir) / "transform.npy", transform)

    return data


def call_npy_to_vdb(
    output_dir: PathLike,
    tolerance: float,
    wind_vectors: bool = True,
    verbose: bool = False,
    dry_run: bool = False,
) -> Optional[str]:
    """Call numpy to VDB conversion using conda environment.

    Args:
        output_dir: Directory containing numpy files to convert
        tolerance: Tolerance for VDB compression
        wind_vectors: Whether to process wind vectors
        verbose: Whether to enable verbose output
        dry_run: If True, return command without executing

    Returns:
        Command string if dry_run=True, None otherwise
    """
    command_template = Template(
        """/home/cmdavis4/programs/miniforge3/condabin/conda run -n vdb python -c \"from cloudy.vdb.npy_to_vdb import npy_to_vdb; npy_to_vdb('{{ output_dir }}', {{ tolerance }}, wind_vectors={{ wind_vectors }}, verbose={{ verbose }})\"""".strip()
    )

    command = command_template.render(
        output_dir=output_dir,
        wind_vectors=wind_vectors,
        tolerance=tolerance,
        verbose=verbose,
    )
    print(command)
    if dry_run:
        return command
    sp = subprocess.Popen(command, shell=True, start_new_session=True)
    sp.wait()
    return


def export_to_openvdb(
    ds: xr.Dataset,
    vars: List[str],
    output_dir: PathLike,
    tolerance: float = 0.0001,
    wind_vectors: bool = True,
    cleanup: bool = True,
    verbose: bool = False,
) -> Dict[Path, Any]:
    """Export xarray Dataset to OpenVDB format for 3D visualization.

    Args:
        ds: xarray Dataset containing atmospheric data
        vars: List of variable names to export
        output_dir: Directory for output VDB files
        tolerance: Compression tolerance for VDB format
        wind_vectors: Whether to process wind vectors specially
        cleanup: Whether to delete intermediate numpy files
        verbose: Whether to enable verbose output

    Returns:
        Dictionary mapping VDB file paths to original data arrays
    """
    # Export them to numpy
    print("Exporting to numpy...")
    npy_data = rams_ds_to_npy(ds=ds, _vars=vars, output_dir=Path(output_dir))
    # Convert these to openVDB using a subprocess
    print("Converting to OpenVDB...")
    call_npy_to_vdb(
        output_dir=output_dir,
        tolerance=tolerance,
        wind_vectors=wind_vectors,
        verbose=verbose,
    )
    # Delete the intermediate numpy files
    if cleanup:
        print("Cleaning up...")
        for npy_path in npy_data.keys():
            npy_path.unlink()
        (Path(output_dir) / "transform.npy").unlink()
    return {Path(k).with_suffix(".vdb"): v for k, v in npy_data.items()}
