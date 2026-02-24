import pyopenvdb as vdb
import numpy as np

from pathlib import Path
from tqdm import tqdm

import cloudy.utils as cmu

WIND_VARS = ["UC", "VC", "WC"]


def preprocess_np_data(fpath):
    this_data = np.load(fpath)
    # Transpose to (x, y, z) which is what openVDB expects
    this_data = np.transpose(this_data, (2, 1, 0))
    # Need this next line too
    this_data = np.ascontiguousarray(this_data) * 10000
    return this_data


def npy_to_vdb(npy_dir, threshold, wind_vectors=True, verbose=False):
    npy_dir = Path(npy_dir)

    def print_if_verbose(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    # Get a list of all of the files
    npy_files = list(npy_dir.glob("*.npy"))
    # Exclude the transform
    npy_files = [x for x in npy_files if x.stem != "transform"]

    # Read the transform file
    transform_path = npy_dir / "transform.npy"
    if not transform_path.exists():
        raise FileNotFoundError(f"Transform file not found: {transform_path}")
    transform = vdb.createLinearTransform(matrix=np.load(transform_path))

    # Get the datetime from all of these files
    all_file_kv_pairs = [
        cmu.to_kv_pairs(fpath.stem, parse_floats=False) for fpath in npy_files
    ]
    # Extract unique datetime values
    all_dts = list(set([kv.get("dt") for kv in all_file_kv_pairs if "dt" in kv]))

    print_if_verbose(f"Found {len(all_dts)} unique datetimes to process")

    # Iterate over each datetime
    for dt in tqdm(all_dts, desc="Processing datetimes"):
        # Find all files for this datetime
        dt_files = [
            f
            for f in npy_files
            if cmu.to_kv_pairs(f.stem, parse_floats=False).get("dt") == dt
        ]

        # Extract variable names for this datetime
        var_names = [cmu.to_kv_pairs(f.stem).get("category") for f in dt_files]

        # Check for wind components
        wind_components = ["UC", "VC", "WC"]
        has_all_wind = all(wind_var in var_names for wind_var in wind_components)

        # Process each variable individually
        for npy_file in dt_files:
            _npy_to_vdb_helper(npy_file, transform, threshold, verbose=verbose)

        # If all wind components are present and wind_vectors is True, create vector field
        if has_all_wind and wind_vectors:
            print_if_verbose(f"Creating wind vector field for datetime {dt}")

            # Load the three wind components
            wind_arrays = {}
            for wind_var in wind_components:
                wind_file = next(
                    f
                    for f in dt_files
                    if cmu.to_kv_pairs(f.stem).get("category") == wind_var
                )
                wind_arrays[wind_var] = preprocess_np_data(wind_file)

            # Construct 4D array: (x, y, z, 3) where the last dimension is [UC, VC, WC]
            wind_vector_array = np.stack(
                [wind_arrays["UC"], wind_arrays["VC"], wind_arrays["WC"]], axis=-1
            )

            # Create VDB grid for wind vectors
            wind_grid = to_vdb_grid(arr=wind_vector_array, transform=transform)
            wind_grid.prune((threshold, threshold, threshold))

            # Generate output filename for wind vectors
            dt_kv = {
                "dt": cmu.dt_to_str(dt),
                "category": "windvector",
                "varname": "windvector",
            }
            wind_output_name = cmu.to_kv_str(dt_kv)
            wind_output_path = (npy_dir / wind_output_name).with_suffix(".vdb")

            vdb.write(str(wind_output_path), grids=[wind_grid])
            print_if_verbose(f"Wind vector field saved to {wind_output_path}")


def to_vdb_grid(arr, transform):
    # Create the right grid for whether this is a scalar or vector field
    grid = vdb.FloatGrid() if arr.ndim == 3 else vdb.Vec3SGrid()
    # Set the transform
    grid.transform = transform
    # Set numpy array as data on VDB grid
    grid.copyFromArray(arr.astype(np.float32))
    return grid


def _npy_to_vdb_helper(npy_path, transform, threshold, verbose=False):
    npy_path = Path(npy_path)

    def print_if_verbose(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    print_if_verbose(f"Processing {npy_path.name}")

    # Read in and preprocess numpy array file
    this_data = preprocess_np_data(npy_path)
    # Create openVDB grid+
    this_var_grid = to_vdb_grid(arr=this_data, transform=transform)
    this_var_grid.name = "density"
    # Prune for sparsity
    this_var_grid.prune(threshold)
    # Output to the same directory as the input file
    output_fpath = npy_path.with_suffix(".vdb")
    vdb.write(str(output_fpath), grids=[this_var_grid])
    print_if_verbose(f"Saved VDB file: {output_fpath}")


if __name__ == "__main__":

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Convert numpy arrays to VDB format")
    parser.add_argument("directory", help="Directory containing .npy files")
    parser.add_argument(
        "--tolerance", type=float, required=True, help="Pruning tolerance threshold"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    cli_args = parser.parse_args()

    npy_dir = Path(cli_args.directory)
    threshold = cli_args.tolerance
    verbose = cli_args.verbose

    npy_to_vdb(
        npy_dir=npy_dir,
        threshold=threshold,
        verbose=verbose,
    )
