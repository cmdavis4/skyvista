# skyvista

**PyVista-based 3D visualization for atmospheric data**

`skyvista` provides powerful, publication-quality 3D visualizations for atmospheric modeling data using PyVista. Built for atmospheric scientists working with numerical weather models, trajectory data, and volumetric atmospheric fields.

## Features

- **3D Trajectory Visualization**: Render particle trajectories with customizable appearance
- **Isosurface Rendering**: Create contours and isosurfaces from 3D atmospheric fields
- **Vector Field Plotting**: Visualize wind fields and other vector quantities
- **Animation Support**: Generate smooth animations of time-evolving atmospheric data
- **Camera Control**: Advanced camera positioning and following for dynamic views
- **Blender Export**: Export meshes to Blender for high-quality rendering
- **Subplot Support**: Create multi-panel 3D visualizations
- **Convenience API**: Simple functions for common visualization tasks

## Installation

```bash
pip install skyvista
```

For local development:
```bash
pip install -e /path/to/skyvista
```

## Quick Start

### Simple Trajectory Plot

```python
import skyvista as sv
import numpy as np

# Create sample trajectory data
time = np.linspace(0, 10, 100)
trajectory_data = {
    'x': np.sin(time) * 10,
    'y': np.cos(time) * 10,
    'z': time * 2,
}

# Quick plot
sv.quick_plot(trajectory_data)
```

### Plot with Atmospheric Data

```python
import skyvista as sv
import xarray as xr

# Load your atmospheric model data
data = xr.open_dataset("model_output.nc")

# Create isosurface specification
contour_spec = sv.make_contour(
    data_array=data['cloud_water'],
    isovalues=[0.001, 0.01],
    opacity=0.3,
    color='white'
)

# Plot with trajectories
sv.plot_rams_and_trajectories(
    rams_data={'contours': [contour_spec]},
    trajectory_data=trajectory_data,
    show=True
)
```

## Core Concepts

### Data Specifications

`skyvista` uses specification objects to define how data should be rendered:

- **`PVContourSpec`**: Isosurface/contour specifications
- **`PVVectorSpec`**: Vector field specifications
- **`PVTrajectorySpec`**: Trajectory rendering specifications
- **`PV2DSpec`**: 2D slice specifications

### Convenience Functions

For quick visualizations:

- **`quick_plot()`**: Instantly visualize data with defaults
- **`plot_trajectories_only()`**: Focus on trajectories
- **`plot_isosurfaces_only()`**: Focus on isosurfaces
- **`make_contour()`**: Easy contour specification
- **`make_vector()`**: Easy vector field specification
- **`make_trajectory()`**: Easy trajectory specification

### Advanced Usage

```python
import skyvista as sv

# Initialize custom plotter with subplots
plotter = sv.initialize_plotter(
    shape=(1, 2),  # 1 row, 2 columns
    window_size=(1600, 800)
)

# Add data to specific subplots
sv.add_mesh_to_subplots(
    plotter,
    mesh,
    subplot_keys=[(0, 0)],  # First subplot
    name="data"
)

# Control camera
camera = sv.get_trajectory_camera(
    trajectory_data,
    offset=(50, 50, 20),
    focal_point_offset=(0, 0, 0)
)

plotter.camera = camera
plotter.show()
```

### Animation

```python
import skyvista as sv

# Create animation from time series data
sv.animate_trajectories(
    trajectory_data=time_series_trajectories,
    rams_data=atmospheric_fields,
    output_path="animation.mp4",
    fps=30,
    quality=5
)
```

### Blender Export

```python
import skyvista as sv

# Export meshes for high-quality rendering in Blender
sv.export_meshes_to_blender(
    meshes={'cloud': cloud_mesh, 'trajectory': traj_mesh},
    output_file="scene.blend"
)
```

## Data Format

### Trajectory Data

Trajectory data should be provided as dictionaries or DataFrames with spatial coordinates:

```python
trajectory_data = {
    'x': array_of_x_coordinates,
    'y': array_of_y_coordinates,
    'z': array_of_z_coordinates,
    'time': array_of_times  # optional
}
```

### Atmospheric Fields

Atmospheric data should be xarray DataArrays with named dimensions:

```python
import xarray as xr

# Example atmospheric field
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

## Examples

See the documentation for comprehensive examples including:

- Basic trajectory visualization
- Isosurface rendering
- Multi-panel plots
- Animation workflows
- Custom camera paths
- Blender integration

## API Reference

### Main Functions

- **`plot_rams_and_trajectories()`**: Main plotting function for combined visualization
- **`animate_trajectories()`**: Create animations
- **`initialize_plotter()`**: Create custom PyVista plotter

### Convenience API

- **`quick_plot()`**: Simplified plotting interface
- **`plot_trajectories_only()`**: Trajectory-focused visualization
- **`plot_isosurfaces_only()`**: Isosurface-focused visualization
- **`make_contour()`**: Create contour specifications
- **`make_vector()`**: Create vector field specifications
- **`make_trajectory()`**: Create trajectory specifications

### Utilities

- **`get_trajectory_camera()`**: Position camera based on trajectories
- **`calculate_camera_offset()`**: Calculate optimal camera positions
- **`export_meshes_to_blender()`**: Export to Blender
- **`rectangle_mesh()`**: Create 2D rectangular meshes
- **`screenshot_render()`**: Capture high-resolution screenshots

## Requirements

- Python ≥ 3.8
- PyVista ≥ 0.38
- NumPy
- xarray

Optional:
- bpy (Blender as a Python module) for Blender export

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License

## Citation

If you use skyvista in your research, please cite:

```bibtex
@software{skyvista,
  author = {Davis, Charles},
  title = {skyvista: PyVista-based 3D visualization for atmospheric data},
  year = {2025},
  url = {https://github.com/cmdavis4/skyvista}
}
```

---

Built for atmospheric science visualization 🌤️📊
