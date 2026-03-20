from pathlib import Path

import xarray as xr


def load_example_storm_data():
    return xr.open_dataset(
        Path(__file__).parent.parent / "assets" / "data" / "example_ic_storm.nc"
    )
