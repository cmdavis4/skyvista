# Migration Guide: Simplified API

## Overview

The pvplotting module now includes a simplified convenience API that makes common visualization tasks much less verbose while maintaining the full power of the underlying dataclass-based architecture.

## What Changed?

### New Functions Added

1. **`quick_plot()`** - Main convenience function for most use cases
2. **`plot_trajectories_only()`** - Simplified trajectory-only plotting
3. **`plot_isosurfaces_only()`** - Simplified isosurface-only plotting
4. **`make_contour()`** - Factory function for PVContourSpec
5. **`make_vector()`** - Factory function for PVVectorSpec
6. **`make_trajectory()`** - Factory function for PVTrajectorySpec

### Backward Compatibility

**All existing code continues to work!** The original API using `plot_rams_and_trajectories()` with dataclasses is unchanged. This is purely additive.

## Quick Migration Examples

### Migrate: Basic trajectory + contour plot

```python
# Old way (still works!)
pv_config = PVConfig(plotter=initialize_plotter())
rams_data = PVRamsData(
    simulation_ds=sim_ds,
    varspecs=(PVContourSpec(varname='THETA', isosurfaces=[300, 310]),)
)
traj_data = PVTrajectoryData(
    trajectory_ds=traj_ds,
    varspecs=(PVTrajectorySpec(scalar='temperature'),)
)
meshes = plot_rams_and_trajectories(pv_config, [rams_data, traj_data])

# New way
meshes = quick_plot(
    simulation_ds=sim_ds,
    trajectory_ds=traj_ds,
    contours={'THETA': [300, 310]},
    trajectory_scalar='temperature'
)
```

### Migrate: Animation

```python
# Old way
pv_config = PVConfig(
    plotter=initialize_plotter(),
    animation=True,
    gif_path='output.gif',
    fps=15
)
# ... create data objects ...
meshes = plot_rams_and_trajectories(pv_config, pv_datas)

# New way
meshes = quick_plot(
    simulation_ds=sim_ds,
    trajectory_ds=traj_ds,
    contours={'THETA': [300, 310]},
    animate=True,
    gif_path='output.gif',
    fps=15
)
```

### Migrate: Custom styling

```python
# Old way
contour_spec = PVContourSpec(
    varname='THETA',
    isosurfaces=[300, 310],
    add_mesh_kwargs={'opacity': 0.5, 'color': 'red'}
)

# New way (using quick_plot)
quick_plot(
    simulation_ds=sim_ds,
    contours={'THETA': {'isosurfaces': [300, 310], 'opacity': 0.5, 'color': 'red'}}
)

# Or using factory function
contour_spec = make_contour('THETA', isosurfaces=[300, 310], opacity=0.5, color='red')
```

## When to Use Each API Level

### Use `quick_plot()` when:
- ✓ Making quick visualizations
- ✓ Exploring data interactively
- ✓ Creating simple animations
- ✓ You want minimal boilerplate
- ✓ Standard styling is acceptable

### Use factory functions (`make_contour()`, etc.) when:
- ✓ You need some customization
- ✓ You're building reusable spec objects
- ✓ You want to avoid dataclass verbosity but still need control
- ✓ You're combining simple and complex visualizations

### Use full dataclass API when:
- ✓ Maximum control is required
- ✓ Complex multi-subplot layouts
- ✓ Custom callbacks and advanced features
- ✓ Building a library on top of pvplotting
- ✓ You need to programmatically manipulate specs

## Common Patterns

### Pattern 1: Quick exploration
```python
# Fastest way to visualize your data
quick_plot(trajectory_ds=traj_ds, trajectory_color='red')
```

### Pattern 2: Publication-ready figures
```python
# More control over styling
quick_plot(
    simulation_ds=sim_ds,
    trajectory_ds=traj_ds,
    contours={
        'THETA': {'isosurfaces': [300, 305, 310], 'opacity': 0.6},
        'RV': {'isosurfaces': [0.01], 'color': 'blue', 'opacity': 0.4}
    },
    trajectory_scalar='THETA',
    trajectory_cmap='RdBu_r',
    trajectory_scalar_bar=True,
    screenshot_path='figure1.png',
    export_html=True,
    background='white'
)
```

### Pattern 3: Hybrid approach for complex needs
```python
# Use factory functions for most specs, but still use full API for special cases
from cloudy.pvplotting import (
    plot_rams_and_trajectories,
    PVConfig, PVRamsData, PVTrajectoryData,
    make_contour, make_trajectory, PVVectorSpec  # Mix factory and dataclass
)

# Simple specs with factory functions
theta = make_contour('THETA', [300, 310], opacity=0.5)
traj = make_trajectory(scalar='temp', cmap='viridis')

# Complex spec needs full dataclass
vector = PVVectorSpec(
    varname='wind',
    u_varname='UC', v_varname='VC', w_varname='WC',
    create_mesh_kwargs={'scale': 'custom_scale_var', 'factor': 0.5},
    add_mesh_kwargs={'opacity': 0.3}
)

rams_data = PVRamsData(simulation_ds=sim_ds, varspecs=(theta, vector))
traj_data = PVTrajectoryData(trajectory_ds=traj_ds, varspecs=(traj,))
config = PVConfig(animation=True, gif_path='out.gif')

meshes = plot_rams_and_trajectories(config, [rams_data, traj_data])
```

## Tips

1. **Start simple**: Begin with `quick_plot()` and only move to more complex APIs if needed
2. **Mix and match**: You can use factory functions with the full API for a good balance
3. **Gradual migration**: No need to update all your code at once - both APIs coexist
4. **Check examples**: See `CONVENIENCE_EXAMPLES.md` for side-by-side comparisons

## Getting Help

- Check `CONVENIENCE_EXAMPLES.md` for comprehensive examples
- See docstrings: `help(quick_plot)`, `help(make_contour)`, etc.
- For complex cases, the full dataclass API documentation still applies
