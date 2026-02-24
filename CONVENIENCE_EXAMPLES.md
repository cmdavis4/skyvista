# Convenience API Examples

This document shows how the new convenience API simplifies common plotting tasks.

## Before vs After

### Example 1: Simple trajectory plot with temperature isosurfaces

**Before (verbose):**
```python
from cloudy.pvplotting import (
    PVConfig, PVRamsData, PVTrajectoryData,
    PVContourSpec, PVTrajectorySpec,
    plot_rams_and_trajectories, initialize_plotter
)

# Create plotter
plotter = initialize_plotter()

# Create specs
contour_spec = PVContourSpec(
    varname='THETA',
    isosurfaces=[300, 305, 310],
    add_mesh_kwargs={'opacity': 0.5}
)

trajectory_spec = PVTrajectorySpec(
    varname='trajectories',
    scalar='THETA',
    cmap='RdBu_r',
    scalar_bar=True
)

# Create data objects
rams_data = PVRamsData(
    simulation_ds=sim_ds,
    varspecs=(contour_spec,)
)

trajectory_data = PVTrajectoryData(
    trajectory_ds=traj_ds,
    varspecs=(trajectory_spec,),
    n_parcel_limit=1000
)

# Create config
pv_config = PVConfig(
    plotter=plotter,
    show=True
)

# Plot
meshes = plot_rams_and_trajectories(pv_config, [rams_data, trajectory_data])
```

**After (simple):**
```python
from cloudy.pvplotting import quick_plot

meshes = quick_plot(
    simulation_ds=sim_ds,
    trajectory_ds=traj_ds,
    contours={'THETA': [300, 305, 310]},
    trajectory_scalar='THETA',
    trajectory_cmap='RdBu_r',
    trajectory_scalar_bar=True
)
```

---

### Example 2: Animation with multiple variables

**Before:**
```python
from cloudy.pvplotting import (
    PVConfig, PVRamsData, PVTrajectoryData,
    PVContourSpec, PVTrajectorySpec,
    plot_rams_and_trajectories, initialize_plotter
)

plotter = initialize_plotter()

theta_spec = PVContourSpec(
    varname='THETA',
    isosurfaces=[300, 310],
    add_mesh_kwargs={'opacity': 0.6}
)

rv_spec = PVContourSpec(
    varname='RV',
    isosurfaces=[0.01, 0.02],
    add_mesh_kwargs={'opacity': 0.4, 'color': 'blue'}
)

trajectory_spec = PVTrajectorySpec(
    varname='trajectories',
    particles=True,
    scalar='height'
)

rams_data = PVRamsData(
    simulation_ds=sim_ds,
    varspecs=(theta_spec, rv_spec)
)

trajectory_data = PVTrajectoryData(
    trajectory_ds=traj_ds,
    varspecs=(trajectory_spec,)
)

pv_config = PVConfig(
    plotter=plotter,
    animation=True,
    gif_path='output.gif',
    fps=15
)

meshes = plot_rams_and_trajectories(pv_config, [rams_data, trajectory_data])
```

**After:**
```python
from cloudy.pvplotting import quick_plot

meshes = quick_plot(
    simulation_ds=sim_ds,
    trajectory_ds=traj_ds,
    contours={
        'THETA': {'isosurfaces': [300, 310], 'opacity': 0.6},
        'RV': {'isosurfaces': [0.01, 0.02], 'opacity': 0.4, 'color': 'blue'}
    },
    trajectory_particles=True,
    trajectory_scalar='height',
    animate=True,
    gif_path='output.gif',
    fps=15
)
```

---

### Example 3: Just trajectories (no simulation data)

**Before:**
```python
from cloudy.pvplotting import (
    PVConfig, PVTrajectoryData, PVTrajectorySpec,
    plot_rams_and_trajectories, initialize_plotter
)

plotter = initialize_plotter()

trajectory_spec = PVTrajectorySpec(
    varname='trajectories',
    color='red'
)

trajectory_data = PVTrajectoryData(
    trajectory_ds=traj_ds,
    varspecs=(trajectory_spec,)
)

pv_config = PVConfig(plotter=plotter)

meshes = plot_rams_and_trajectories(pv_config, [trajectory_data])
```

**After:**
```python
from cloudy.pvplotting import plot_trajectories_only

meshes = plot_trajectories_only(traj_ds, color='red')
```

---

### Example 4: Vector fields with trajectories

**Before:**
```python
from cloudy.pvplotting import (
    PVConfig, PVRamsData, PVTrajectoryData,
    PVVectorSpec, PVTrajectorySpec,
    plot_rams_and_trajectories, initialize_plotter
)

plotter = initialize_plotter()

vector_spec = PVVectorSpec(
    varname='wind',
    u_varname='UC',
    v_varname='VC',
    w_varname='WC',
    create_mesh_kwargs={'scale': 'speed'},
    add_mesh_kwargs={'opacity': 0.7}
)

trajectory_spec = PVTrajectorySpec(
    varname='trajectories',
    scalar='temperature'
)

rams_data = PVRamsData(
    simulation_ds=sim_ds,
    varspecs=(vector_spec,)
)

trajectory_data = PVTrajectoryData(
    trajectory_ds=traj_ds,
    varspecs=(trajectory_spec,)
)

pv_config = PVConfig(plotter=plotter)

meshes = plot_rams_and_trajectories(pv_config, [rams_data, trajectory_data])
```

**After:**
```python
from cloudy.pvplotting import quick_plot

meshes = quick_plot(
    simulation_ds=sim_ds,
    trajectory_ds=traj_ds,
    vectors={'wind': {'u': 'UC', 'v': 'VC', 'w': 'WC', 'scale': 'speed', 'opacity': 0.7}},
    trajectory_scalar='temperature'
)
```

---

## Hybrid Approach: Using Factory Functions

For users who need more control but still want less verbosity:

```python
from cloudy.pvplotting import (
    plot_rams_and_trajectories,
    PVConfig, PVRamsData, PVTrajectoryData,
    make_contour, make_trajectory  # Factory functions
)

# Use factory functions to create specs more easily
theta_spec = make_contour('THETA', isosurfaces=[300, 310], opacity=0.5)
rv_spec = make_contour('RV', isosurfaces=[0.01], color='blue')
traj_spec = make_trajectory(scalar='temperature', cmap='viridis', scalar_bar=True)

# Still use the full API for control
rams_data = PVRamsData(simulation_ds=sim_ds, varspecs=(theta_spec, rv_spec))
traj_data = PVTrajectoryData(trajectory_ds=traj_ds, varspecs=(traj_spec,))
config = PVConfig(animation=True, gif_path='out.gif')

meshes = plot_rams_and_trajectories(config, [rams_data, traj_data])
```

## API Levels Summary

The module now supports three levels of API complexity:

1. **Simple** (`quick_plot`, `plot_trajectories_only`, etc.): Best for quick visualization and common use cases
2. **Hybrid** (factory functions + dataclasses): Good balance of simplicity and control
3. **Full** (raw dataclasses): Maximum control and flexibility for complex visualizations

Choose the level that best matches your needs!
