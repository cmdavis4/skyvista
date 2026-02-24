# skyvista

**3D atmospheric data visualization in Python**

`skyvista` provides scientifically accurate 3D data of atmospheric science data (but likely applicable in many other disciplines). Skyvista's visualizations are primarily built on top of the [pyvista](https://pyvista.org/#) library, and with the appropriate setup can be rendered directly in jupyter notebooks or IDEs, or written to disk. Pyvista is capable of creating interactive visualizations in pure HTML, making these visualizations conveniently portable. Skyvista also contains simplified functionality for creating animations of 3D data using pyvista, in addition to single visualizations.

Skyvista also contains functionality for exporting meshes and volumes created with pyvista to a variety of file formats. One intended use of this functionality is to create visualizations in Blender, using the optional `[blender]` addon functionality. Skyvista can handle things like setting up animations from a set of single-timestep output files, assigning shaders, as well as more complicated logic.

**Further documentation is forthcoming!**

## Features

- **Gridded data visualization**: Create sets of isosurfaces, volumes, vectors, or planes (for things like land/ocean surfaces or cross-sections) from xarray datasets
- **Trajectory visualization**: Visualize Lagrangian data, with options to show trajectories as continuous arrows or as particles at their instantaneous positions (among other visualization customizations)
- **Animation support**: Generate animations of time-evolving atmospheric data
- **Camera control**: Advanced camera positioning and following for dynamic views
- **Blender functionality**: Export meshes to Blender and setup scenes and animations
- **Convenience API**: Simple functions for quick visualizations

## Installation

First install Skyvista, which will install pyvista as a dependency:

```bash
pip install git@github.com:cmdavis4/skyvista.git
```
From here, pyvista configuration may be complicated depending on your setup. We recommend following the [pyvista installation documentation](https://docs.pyvista.org/getting-started/installation.html) and ensuring that you can successfully create an interactive bunny visualization, following the documentation's example:
```
from pyvista import examples
dataset = examples.download_bunny()
dataset.plot(cpos='xy')
```

## Quick Start


### Gridded model or observation data

```python
import skyvista as sv
import xarray as xr

# Create a plot of vertical velocity and condensate loading isosurfaces

# Load gridded data
storm_ds = xr.open_dataset("model_output.nc")
quick_plot(
    simulation_ds=storm_ds,
    contours={
        # Vertical velocity
        "W": {
            "isosurfaces": [1, 3, 5, 10],
            "cmap": "Greens",
            "opacity": 0.8,
        },
        # Condensate loading
        "RC": {
            # Isosurfaces will be calculated automatically if not specified
            "cmap": "Blues",
            "opacity": 0.4,
        }
    },
```

## Data Format

### Gridded data

Gridded data should be xarray `Dataset`s with dimensions `x`, `y`, `z`, and optionally `time` if creating an animation:

```python
import xarray as xr

# Example DataArray; this should be a data variable in an xr.Dataset,
# not passed directly into skyvista
cloud_field = xr.DataArray(
    data=3d_array,
    dims=['x', 'y', 'z'],
    coords={
        'x': x_coordinates,
        'y': y_coordinates,
        'z': z_coordinates,
    }
)
```

## License

MIT License

## Citation

If you use skyvista in your research, please cite:

```bibtex
@software{skyvista,
  author = {Davis, Charles},
  title = {skyvista: 3D atmospheric data visualization in Python},
  year = {2026},
  url = {https://github.com/cmdavis4/skyvista}
}
```
