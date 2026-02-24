"""
Convenience functions for simplified pvplotting API.

This module provides high-level wrapper functions that hide the complexity
of the dataclass-based API for common visualization tasks.
"""

from typing import Optional, Dict, Any, Union, List
import xarray as xr
from pathlib import Path

from .types_pvplotting import (
    PVConfig,
    PVRamsData,
    PVTrajectoryData,
    PVContourSpec,
    PVVolumeSpec,
    PV2DSpec,
    PVVectorSpec,
    PVTrajectorySpec,
)
from .core_pvplotting import plot_rams_and_trajectories
from .plotter import initialize_plotter

# ============================================================================
# FACTORY FUNCTIONS FOR SPECS
# ============================================================================


def make_contour(
    varname: str,
    isosurfaces: Optional[Union[List[float], float]] = None,
    opacity: Optional[float] = None,
    color: Optional[str] = None,
    individual_meshes: bool = False,
    scalar_bar: bool = False,
    **kwargs,
) -> PVContourSpec:
    """
    Create a contour specification with simplified arguments.

    Args:
        varname: Variable name for contouring
        isosurfaces: Isosurface values (single value or list)
        opacity: Mesh opacity (0-1)
        color: Mesh color name
        individual_meshes: Create separate mesh for each isosurface
        scalar_bar: Show scalar bar
        **kwargs: Additional arguments for create_mesh_kwargs or add_mesh_kwargs

    Returns:
        PVContourSpec instance

    Example:
        >>> spec = make_contour('THETA', isosurfaces=[300, 310], opacity=0.5)
    """
    # Convert single isosurface to list
    if isosurfaces is not None and not isinstance(isosurfaces, (list, tuple)):
        isosurfaces = [isosurfaces]

    # Separate kwargs into create_mesh and add_mesh kwargs
    add_mesh_kwargs = {}
    if opacity is not None:
        add_mesh_kwargs["opacity"] = opacity
    if color is not None:
        add_mesh_kwargs["color"] = color

    # Any remaining kwargs go to add_mesh_kwargs by default
    # Users can override with create_mesh_kwargs_ prefix
    create_mesh_kwargs = {}
    for key, value in kwargs.items():
        if key.startswith("create_mesh_"):
            create_mesh_kwargs[key.replace("create_mesh_", "")] = value
        else:
            add_mesh_kwargs[key] = value

    return PVContourSpec(
        varname=varname,
        isosurfaces=isosurfaces,
        individual_meshes=individual_meshes,
        scalar_bar=scalar_bar,
        create_mesh_kwargs=create_mesh_kwargs,
        add_mesh_kwargs=add_mesh_kwargs,
    )


def make_volume(
    varname: str,
    opacity: Optional[float] = None,
    scalar_bar: bool = False,
    **kwargs,
) -> PVVolumeSpec:
    """
    Create a contour specification with simplified arguments.

    Args:
        varname: Variable name for contouring
        isosurfaces: Isosurface values (single value or list)
        opacity: Mesh opacity (0-1)
        color: Mesh color name
        individual_meshes: Create separate mesh for each isosurface
        scalar_bar: Show scalar bar
        **kwargs: Additional arguments for create_mesh_kwargs or add_mesh_kwargs

    Returns:
        PVContourSpec instance

    Example:
        >>> spec = make_contour('THETA', isosurfaces=[300, 310], opacity=0.5)
    """

    # Separate kwargs into create_mesh and add_mesh kwargs
    add_mesh_kwargs = {}
    if opacity is not None:
        add_mesh_kwargs["opacity"] = opacity

    # Any remaining kwargs go to add_mesh_kwargs by default
    # Users can override with create_mesh_kwargs_ prefix
    create_mesh_kwargs = {}
    for key, value in kwargs.items():
        if key.startswith("create_mesh_"):
            create_mesh_kwargs[key.replace("create_mesh_", "")] = value
        else:
            add_mesh_kwargs[key] = value

    return PVVolumeSpec(
        varname=varname,
        scalar_bar=scalar_bar,
        create_mesh_kwargs=create_mesh_kwargs,
        add_mesh_kwargs=add_mesh_kwargs,
    )


def make_vector(
    varname: str,
    u_varname: str = "UC",
    v_varname: str = "VC",
    w_varname: str = "WC",
    scale: Optional[str] = None,
    opacity: Optional[float] = None,
    color: Optional[str] = None,
    **kwargs,
) -> PVVectorSpec:
    """
    Create a vector field specification with simplified arguments.

    Args:
        varname: Name for the vector field
        u_varname: Variable name for u component
        v_varname: Variable name for v component
        w_varname: Variable name for w component
        scale: Variable name or value for arrow scaling
        opacity: Mesh opacity (0-1)
        color: Arrow color name
        **kwargs: Additional arguments for glyph creation

    Returns:
        PVVectorSpec instance

    Example:
        >>> spec = make_vector('wind', scale='speed', opacity=0.7)
    """
    add_mesh_kwargs = {}
    if opacity is not None:
        add_mesh_kwargs["opacity"] = opacity
    if color is not None:
        add_mesh_kwargs["color"] = color

    create_mesh_kwargs = {}
    if scale is not None:
        create_mesh_kwargs["scale"] = scale

    # Add any extra kwargs
    for key, value in kwargs.items():
        if key.startswith("create_mesh_"):
            create_mesh_kwargs[key.replace("create_mesh_", "")] = value
        else:
            add_mesh_kwargs[key] = value

    return PVVectorSpec(
        varname=varname,
        u_varname=u_varname,
        v_varname=v_varname,
        w_varname=w_varname,
        create_mesh_kwargs=create_mesh_kwargs,
        add_mesh_kwargs=add_mesh_kwargs,
    )


def make_trajectory(
    varname: str = "trajectories",
    color: Optional[str] = None,
    scalar: Optional[str] = None,
    cmap: str = "viridis",
    scalar_bar: bool = False,
    particles: bool = False,
    silhouettes: bool = False,
    opacity: Optional[float] = None,
    **kwargs,
) -> PVTrajectorySpec:
    """
    Create a trajectory specification with simplified arguments.

    Args:
        varname: Name for the trajectory mesh
        color: Solid color name (mutually exclusive with scalar)
        scalar: Variable name for scalar coloring (mutually exclusive with color)
        cmap: Colormap name for scalar coloring
        scalar_bar: Show scalar bar for scalar coloring
        particles: Render as particles instead of arrows
        silhouettes: Add silhouette effect
        opacity: Mesh opacity (0-1)
        **kwargs: Additional arguments (body_radius, head_length_frac, etc.)

    Returns:
        PVTrajectorySpec instance

    Example:
        >>> spec = make_trajectory(scalar='temperature', cmap='RdBu_r', scalar_bar=True)
    """
    add_mesh_kwargs = {}
    if opacity is not None:
        add_mesh_kwargs["opacity"] = opacity

    # Add any extra kwargs to add_mesh_kwargs
    for key, value in kwargs.items():
        if key not in [
            "body_radius",
            "head_length_frac",
            "head_radius_frac",
            "head_radial_resolution",
            "tube_resolution",
            "label",
        ]:
            add_mesh_kwargs[key] = value

    # Extract trajectory-specific kwargs
    traj_kwargs = {
        k: kwargs[k]
        for k in [
            "body_radius",
            "head_length_frac",
            "head_radius_frac",
            "head_radial_resolution",
            "tube_resolution",
            "label",
        ]
        if k in kwargs
    }

    return PVTrajectorySpec(
        varname=varname,
        color=color,
        scalar=scalar,
        cmap=cmap,
        scalar_bar=scalar_bar,
        particles=particles,
        silhouettes=silhouettes,
        add_mesh_kwargs=add_mesh_kwargs,
        **traj_kwargs,
    )


# ============================================================================
# HIGH-LEVEL CONVENIENCE FUNCTION
# ============================================================================


def quick_plot(
    simulation_ds: Optional[xr.Dataset] = None,
    trajectory_ds: Optional[xr.Dataset] = None,
    # Simplified contour specs
    contours: Optional[Dict[str, Any]] = None,
    volumes: Optional[Dict[str, Any]] = None,
    vectors: Optional[Dict[str, Any]] = None,
    slices_2d: Optional[Dict[str, Any]] = None,
    # Simplified trajectory specs
    trajectory_color: Optional[str] = None,
    trajectory_scalar: Optional[str] = None,
    trajectory_cmap: str = "viridis",
    trajectory_scalar_bar: bool = False,
    trajectory_particles: bool = False,
    trajectory_limit: Optional[int] = 1000,
    # Animation settings
    animate: bool = False,
    gif_path: Optional[str] = None,
    fps: float = 10,
    interactive: bool = False,
    gif_scrubber: bool = False,
    # Display settings
    show: bool = True,
    screenshot_path: Optional[str] = None,
    export_html: bool = False,
    title: Optional[str] = None,
    # Plotter config
    plotter=None,
    background: str = "#f8f6f1",
    # Advanced passthrough
    pv_config_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[Any, Any]:
    """
    Simplified interface for creating 3D visualizations of atmospheric data.

    This convenience function wraps the full dataclass-based API, making it
    easier to create common visualizations without manually instantiating
    PVConfig, PVData, and PVVarSpec objects.

    Args:
        simulation_ds: xarray Dataset with simulation data (e.g., RAMS output)
        trajectory_ds: xarray Dataset with trajectory/parcel data
        contours: Dictionary mapping variable names to isosurface specifications.
            Simple form: {'THETA': [300, 310]}
            Full form: {'THETA': {'isosurfaces': [300, 310], 'opacity': 0.5, ...}}
        vectors: Dictionary mapping names to vector field specifications.
            Example: {'wind': {'u': 'UC', 'v': 'VC', 'w': 'WC', 'scale': 'speed'}}
        slices_2d: Dictionary for 2D slice specifications.
            Example: {'DBZ': {'slice_dim': 'x', 'slice_value': -50000, 'opacity': 0.8}}
            Supported keys:
                - slice_dim: 'x', 'y', or 'z' (default: 'z') - dimension to slice
                - slice_value: float or None (default: None) - value at which to slice
                - slice_method: 'nearest' or 'interp' (default: 'nearest') - selection method
                - opacity, cmap, scalar_bar: standard mesh styling options
        trajectory_color: Solid color name for trajectories (e.g., 'red')
        trajectory_scalar: Scalar variable name for coloring trajectories
        trajectory_cmap: Colormap for trajectory scalar coloring
        trajectory_scalar_bar: Show scalar bar for trajectories
        trajectory_particles: Render trajectories as particles instead of arrows
        trajectory_limit: Maximum number of trajectories to plot (default: 1000)
        animate: Create animation instead of still image
        gif_path: Output path for animation (required if animate=True and not interactive)
        fps: Frames per second for animation
        interactive: Use interactive slider instead of saving GIF
        gif_scrubber: Use gif scrubber widget for animations
        show: Display the plot
        screenshot_path: Path to save screenshot
        export_html: Export HTML alongside screenshot/animation
        title: Custom title text for the plot
        plotter: PyVista plotter instance (created automatically if None)
        background: Background color (hex or name)
        pv_config_kwargs: Additional keyword arguments for PVConfig
        **kwargs: Additional keyword arguments (reserved for future use)

    Returns:
        Dictionary mapping times to dictionaries of PVMesh objects

    Raises:
        ValueError: If animate=True but gif_path not provided (when not interactive)

    Examples:
        Simple trajectory plot:
        >>> quick_plot(trajectory_ds=traj_ds, trajectory_color='red')

        Trajectories with temperature isosurfaces:
        >>> quick_plot(
        ...     simulation_ds=sim_ds,
        ...     trajectory_ds=traj_ds,
        ...     contours={'THETA': [300, 305, 310]},
        ...     trajectory_scalar='THETA',
        ...     trajectory_scalar_bar=True
        ... )

        Create animation with custom styling:
        >>> quick_plot(
        ...     simulation_ds=sim_ds,
        ...     trajectory_ds=traj_ds,
        ...     contours={'RV': {'isosurfaces': [0.01, 0.02], 'opacity': 0.6}},
        ...     trajectory_particles=True,
        ...     animate=True,
        ...     gif_path='output.gif',
        ...     fps=15
        ... )

        Vector field with trajectories:
        >>> quick_plot(
        ...     simulation_ds=sim_ds,
        ...     trajectory_ds=traj_ds,
        ...     vectors={'wind': {'u': 'UC', 'v': 'VC', 'w': 'WC'}},
        ...     trajectory_scalar='height'
        ... )
    """
    # Validate inputs
    if not simulation_ds and not trajectory_ds:
        raise ValueError("Must provide at least one of simulation_ds or trajectory_ds")

    if animate and not interactive and not gif_path:
        raise ValueError(
            "Must provide gif_path when animate=True (unless interactive=True)"
        )

    # Build PVData objects
    pv_datas = []

    # Handle simulation data
    if simulation_ds is not None:
        varspecs = []

        # Process contours
        if contours:
            for varname, spec in contours.items():
                if isinstance(spec, (list, tuple)):
                    # Simple form: just isosurface values
                    varspecs.append(make_contour(varname, isosurfaces=spec))
                elif isinstance(spec, dict):
                    # Full form: dictionary of parameters
                    varspecs.append(make_contour(varname, **spec))
                else:
                    raise ValueError(
                        f"Contour spec for {varname} must be list of isosurfaces or"
                        " dict of parameters"
                    )

        # Process volumes
        if volumes:
            for varname, spec in volumes.items():
                if isinstance(spec, dict):
                    # Full form: dictionary of parameters
                    varspecs.append(make_volume(varname, **spec))
                else:
                    raise ValueError(
                        f"Volume spec for {varname} must be list of isosurfaces or"
                        " dict of parameters"
                    )

        # Process vectors
        if vectors:
            for varname, spec in vectors.items():
                if isinstance(spec, dict):
                    varspecs.append(make_vector(varname, **spec))
                else:
                    raise ValueError(f"Vector spec for {varname} must be a dictionary")

        # Process 2D slices
        if slices_2d:
            for varname, spec in slices_2d.items():
                spec_dict = spec if isinstance(spec, dict) else {}
                # Extract PV2DSpec-specific parameters
                slice_params = {}
                for key in ["slice_dim", "slice_value", "slice_method"]:
                    if key in spec_dict:
                        slice_params[key] = spec_dict.pop(key)

                # Extract PVVarSpec base parameters
                base_params = {}
                for key in ["scalar_bar", "empty_ok", "name"]:
                    if key in spec_dict:
                        base_params[key] = spec_dict.pop(key)

                # Separate remaining kwargs into create_mesh and add_mesh kwargs
                create_mesh_kwargs = {}
                add_mesh_kwargs = {}
                for key, value in spec_dict.items():
                    if key.startswith("create_mesh_"):
                        create_mesh_kwargs[key.replace("create_mesh_", "")] = value
                    else:
                        add_mesh_kwargs[key] = value

                varspecs.append(
                    PV2DSpec(
                        varname=varname,
                        create_mesh_kwargs=create_mesh_kwargs,
                        add_mesh_kwargs=add_mesh_kwargs,
                        **base_params,
                        **slice_params,
                    )
                )

        if varspecs:
            pv_datas.append(
                PVRamsData(simulation_ds=simulation_ds, varspecs=tuple(varspecs))
            )

    # Handle trajectory data
    if trajectory_ds is not None:
        # Build trajectory spec
        traj_spec = make_trajectory(
            color=trajectory_color,
            scalar=trajectory_scalar,
            cmap=trajectory_cmap,
            scalar_bar=trajectory_scalar_bar,
            particles=trajectory_particles,
        )

        pv_datas.append(
            PVTrajectoryData(
                trajectory_ds=trajectory_ds,
                varspecs=(traj_spec,),
                n_parcel_limit=trajectory_limit,
            )
        )

    # Build PVConfig
    if plotter is None:
        plotter = initialize_plotter(background=background)

    config_kwargs = pv_config_kwargs or {}
    pv_config = PVConfig(
        plotter=plotter,
        animation=animate,
        gif_path=gif_path,
        gif_scrubber=gif_scrubber,
        screenshot_path=screenshot_path,
        interactive=interactive,
        export_html=export_html,
        fps=fps,
        show=show,
        title=title,
        **config_kwargs,
    )

    # Call the main plotting function
    return plot_rams_and_trajectories(pv_config=pv_config, pv_datas=pv_datas)


# ============================================================================
# SPECIALIZED CONVENIENCE FUNCTIONS
# ============================================================================


def plot_trajectories_only(
    trajectory_ds: xr.Dataset,
    color: Optional[str] = None,
    scalar: Optional[str] = None,
    cmap: str = "viridis",
    animate: bool = False,
    gif_path: Optional[str] = None,
    **kwargs,
) -> Dict[Any, Any]:
    """
    Quick function for plotting only trajectories without simulation data.

    Args:
        trajectory_ds: Trajectory dataset
        color: Solid color for trajectories
        scalar: Variable for scalar coloring
        cmap: Colormap for scalar coloring
        animate: Create animation
        gif_path: Output path for animation
        **kwargs: Additional arguments passed to quick_plot

    Returns:
        Dictionary of meshes by time

    Example:
        >>> plot_trajectories_only(traj_ds, scalar='temperature', cmap='RdBu_r')
    """
    return quick_plot(
        trajectory_ds=trajectory_ds,
        trajectory_color=color,
        trajectory_scalar=scalar,
        trajectory_cmap=cmap,
        animate=animate,
        gif_path=gif_path,
        **kwargs,
    )


def plot_isosurfaces_only(
    simulation_ds: xr.Dataset,
    contours: Dict[str, Any],
    animate: bool = False,
    gif_path: Optional[str] = None,
    **kwargs,
) -> Dict[Any, Any]:
    """
    Quick function for plotting only isosurfaces without trajectories.

    Args:
        simulation_ds: Simulation dataset
        contours: Dictionary of contour specifications
        animate: Create animation
        gif_path: Output path for animation
        **kwargs: Additional arguments passed to quick_plot

    Returns:
        Dictionary of meshes by time

    Example:
        >>> plot_isosurfaces_only(
        ...     sim_ds,
        ...     contours={'THETA': [300, 305, 310], 'RV': [0.01, 0.02]}
        ... )
    """
    return quick_plot(
        simulation_ds=simulation_ds,
        contours=contours,
        animate=animate,
        gif_path=gif_path,
        **kwargs,
    )
