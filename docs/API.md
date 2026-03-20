# Skyvista API Reference

Skyvista provides a clean, composable API for 3D atmospheric data visualization.
The API is designed around the `Scene` class which acts as a container for
visualization specifications.

## Quick Start

```python
import skyvista as sv
import xarray as xr

# Load data
ds = xr.open_dataset("simulation.nc")

# Create a scene with isosurfaces
scene = sv.plot_gridded(ds, contours={"THETA": [300, 310, 320]})
scene.show()

# Or build programmatically
scene = (
    sv.Scene()
    .add_contour(ds, "THETA", isosurfaces=[300], opacity=0.7, color="blue")
    .add_contour(ds, "THETA", isosurfaces=[320], opacity=0.5, color="red")
)
scene.show()
```

## Core Concepts

### Scene

The `Scene` class is the primary entry point. It accumulates visualization
specifications (VarSpecs) and handles rendering.

```python
scene = sv.Scene(
    background="#f8f6f1",  # Background color
    title="Storm Simulation",
    show_grid=True,  # Show coordinate grid
    force_bounds=False,  # Force bounds to data extent
)
```

### VarSpecs

Each visualization type has a corresponding VarSpec class that defines
what to visualize and how:

- `ContourSpec` - Isosurfaces
- `VolumeSpec` - Volume rendering
- `VectorSpec` - Vector glyphs
- `SliceSpec` - 2D slices
- `TrajectorySpec` - Lagrangian trajectories

VarSpecs have two components:
- **Geometry** - What data to extract and how (varname, thresholds, etc.)
- **Appearance** - How to render it (opacity, color, cmap, etc.)

## Primary API

### Convenience Functions

These are the simplest way to create visualizations:

```python
# Plot gridded data
scene = sv.plot_gridded(
    ds,
    contours={"THETA": [300, 310]},  # Variable: isosurface values
    volumes={"QC": {"threshold": (0.001, 0.01)}},
    show=False,  # Don't show yet
)

# Plot trajectories
scene = sv.plot_trajectories(
    traj_ds,
    scalar="altitude",  # Color by this field
    cmap="viridis",
    show=False,
)
```

### Factory Functions

For more control, use factory functions to create VarSpecs:

```python
# Create individual specs
contour_spec = sv.make_contour(
    "THETA",
    isosurfaces=[300, 310, 320],
    opacity=0.7,
    color="blue",
)

volume_spec = sv.make_volume(
    "QC",
    threshold=(0.001, 0.01),
    cmap="Greys_r",
    opacity_unit_distance=500,
)

vector_spec = sv.make_vectors(
    "wind",
    u="UC", v="VC", w="WC",
    factor=0.3,
    cmap="coolwarm",
)

slice_spec = sv.make_slice(
    "THETA",
    dim="z",
    value=1000,
    cmap="RdBu_r",
)

trajectory_spec = sv.make_trajectory(
    scalar="altitude",
    style="tube",
    tube_radius=50,
)
```

### Scene Methods

Add specs to a scene:

```python
scene = sv.Scene()

# Add using spec objects
scene.add(ds, contour_spec)
scene.add(ds, volume_spec)

# Or use shorthand methods
scene.add_contour(ds, "THETA", isosurfaces=[300])
scene.add_volume(ds, "QC", threshold=(0.001, 0.01))
scene.add_vectors(ds, "wind", u="UC", v="VC", w="WC")
scene.add_slice(ds, "THETA", dim="z", value=1000)
scene.add_trajectories(traj_ds, scalar="altitude")

# Multiple contours from dict
scene.add_contours(ds, {
    "THETA": [300, 310],
    "W": {"isosurfaces": [5, 10], "color": "red"},
})
```

### Rendering

```python
# Interactive display
scene.show()

# Save screenshot
scene.screenshot("output.png")

# Animate (for time-varying data)
scene.animate("output.gif", fps=10)

# Export interactive HTML
scene.export_html("output.html")

# Export to Blender
scene.export_blender("output_dir/")
```

## VarSpec Classes

### ContourSpec

Creates isosurface meshes from scalar fields.

**Geometry parameters:**
- `varname` (str): Variable name in dataset
- `isosurfaces` (list): Isosurface values

**Appearance parameters:**
- `opacity` (float): 0.0-1.0
- `color` (str): Named color or hex
- `cmap` (str): Colormap name
- `show_scalar_bar` (bool): Show colorbar
- `style` (str): "surface" or "wireframe"

```python
spec = sv.make_contour(
    "THETA",
    isosurfaces=[300, 310],
    opacity=0.7,
    color="blue",
    style="surface",
)
```

### VolumeSpec

Volume rendering of scalar fields.

**Geometry parameters:**
- `varname` (str): Variable name
- `threshold` (tuple): (min, max) for opacity mapping

**Appearance parameters:**
- `opacity` (float): Base opacity
- `cmap` (str): Colormap
- `opacity_unit_distance` (float): Distance scaling

```python
spec = sv.make_volume(
    "QC",
    threshold=(0.001, 0.01),
    cmap="Greys_r",
    opacity=0.5,
)
```

### VectorSpec

Vector field visualization using glyphs.

**Geometry parameters:**
- `varname` (str): Name for vector field
- `u_varname`, `v_varname`, `w_varname` (str): Component names
- `factor` (float): Scale factor
- `every_nth` (int): Subsample rate

**Appearance parameters:**
- `cmap` (str): Colormap
- `color` (str): Uniform color

```python
spec = sv.make_vectors(
    "wind",
    u="UC", v="VC", w="WC",
    factor=0.3,
    every_nth=2,
    cmap="coolwarm",
)
```

### SliceSpec

2D cross-sections through 3D data.

**Geometry parameters:**
- `varname` (str): Variable name
- `slice_dim` (str): Dimension to slice ("x", "y", or "z")
- `slice_value` (float): Value along slice dimension

**Appearance parameters:**
- `cmap` (str): Colormap
- `clim` (tuple): Color limits (min, max)

```python
spec = sv.make_slice(
    "THETA",
    dim="z",
    value=1000,
    cmap="RdBu_r",
    clim=(290, 310),
)
```

### TrajectorySpec

Lagrangian trajectory visualization.

**Geometry parameters:**
- `scalar` (str): Variable for coloring
- `tube_radius` (float): Tube thickness

**Appearance parameters:**
- `style` (str): "tube" or "particle"
- `color` (str): Uniform color (if no scalar)
- `cmap` (str): Colormap for scalar
- `silhouettes` (bool): Edge highlighting

```python
spec = sv.make_trajectory(
    scalar="altitude",
    style="tube",
    tube_radius=70,
    cmap="viridis",
)
```

## Grid Builders

Skyvista automatically detects grid types and builds appropriate PyVista meshes:

- `RectilinearGridBuilder` - Regular x/y/z grids
- `CurvilinearGridBuilder` - 2D/3D coordinate arrays
- `GeographicGridBuilder` - Lat/lon data on a sphere
- `SphericalGridBuilder` - Radar data (range/azimuth/elevation)

```python
# Automatic detection
grid_type = sv.detect_grid_type(ds)
builder = sv.get_grid_builder(ds)
pv_grid = builder.build()

# Explicit builder
builder = sv.SphericalGridBuilder(ds, azimuth_offset=-90)
pv_grid = builder.build()
```

## Data Requirements

### Gridded Data

Datasets should have dimensions `x`, `y`, `z` (or CF-convention equivalents):

```python
ds = xr.Dataset(
    {"THETA": (["x", "y", "z"], data)},
    coords={"x": x_vals, "y": y_vals, "z": z_vals},
)
```

For animated data, add a `time` dimension:

```python
ds = xr.Dataset(
    {"THETA": (["x", "y", "z", "time"], data)},
    coords={"x": x_vals, "y": y_vals, "z": z_vals, "time": times},
)
```

### Trajectory Data

Trajectory datasets need `trajectory_ix` and `time` dimensions with
`x`, `y`, `z` variables:

```python
ds = xr.Dataset(
    {
        "x": (["trajectory_ix", "time"], x_positions),
        "y": (["trajectory_ix", "time"], y_positions),
        "z": (["trajectory_ix", "time"], z_positions),
        "altitude": (["trajectory_ix", "time"], scalar_values),
    },
    coords={"trajectory_ix": ids, "time": times},
)
```

## Advanced Usage

### Method Chaining

All Scene methods return `self` for chaining:

```python
scene = (
    sv.Scene(background="black")
    .add_contour(ds, "THETA", isosurfaces=[300], color="blue")
    .add_contour(ds, "THETA", isosurfaces=[320], color="red")
    .add_trajectories(traj_ds, scalar="altitude")
)
scene.show()
```

### Custom VarSpecs

For maximum control, construct VarSpecs directly:

```python
from skyvista import ContourSpec
from skyvista.geometry import ContourGeometry
from skyvista.appearance import ContourAppearance

spec = ContourSpec(
    geometry=ContourGeometry(
        varname="THETA",
        isosurfaces=[300, 310],
    ),
    appearance=ContourAppearance(
        opacity=0.7,
        color="blue",
        style="surface",
    ),
    name="theta_contours",
    material_preset="cloud",
)
```

### Blender Export

Export scenes for high-quality Blender rendering:

```python
scene = sv.Scene()
scene.add_contour(ds, "THETA", isosurfaces=[300], material_preset="cloud")
scene.export_blender("blender_output/")
```

The export creates:
- VTK mesh files
- JSON metadata with material presets
- Scene configuration

## Best Practices

1. **Start simple**: Use `plot_gridded()` or `plot_trajectories()` first
2. **Use factory functions**: `make_contour()` etc. for custom visualization
3. **Chain methods**: Build complex scenes fluently
4. **Name your specs**: Helps debugging and Blender export
5. **Set material_preset**: For quality Blender renders
