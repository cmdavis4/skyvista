"""
Convenience API for skyvista.

This module provides the primary user-facing API:

Scene creation functions:
    plot_gridded    - Create scene with gridded data visualizations
    plot_trajectories - Create scene with trajectory visualization

Factory functions (create VarSpecs):
    make_contour    - Create contour/isosurface spec
    make_volume     - Create volume rendering spec
    make_vectors    - Create vector field spec
    make_slice      - Create 2D slice spec
    make_trajectory - Create trajectory spec

Example:
    >>> import skyvista as sv
    >>> scene = sv.plot_gridded(sim_ds, contours={"THETA": [300, 310]})
    >>> scene.add_trajectories(traj_ds, scalar="altitude")
    >>> scene.show()
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import xarray as xr

from .appearance import (
    Appearance,
    ContourAppearance,
    TrajectoryAppearance,
    VectorAppearance,
    VolumeAppearance,
)
from .geometry import (
    ContourGeometry,
    SliceGeometry,
    TrajectoryGeometry,
    VectorGeometry,
    VolumeGeometry,
)
from .scene import Scene
from .varspec import (
    ContourSpec,
    SliceSpec,
    TrajectorySpec,
    VarSpec,
    VectorSpec,
    VolumeSpec,
)

# =============================================================================
# FACTORY FUNCTIONS (create VarSpecs)
# =============================================================================


def make_contour(
    varname: str,
    isosurfaces: Optional[List[float]] = None,
    scalar: Optional[str] = None,
    individual_meshes: bool = False,
    # Appearance
    color: Optional[str] = None,
    opacity: float = 1.0,
    cmap: Optional[str] = None,
    clim: Optional[Tuple[float, float]] = None,
    show_scalar_bar: bool = False,
    scalar_bar_title: Optional[str] = None,
    style: str = "surface",
    material_preset: Optional[str] = None,
    # VarSpec base
    name: Optional[str] = None,
    empty_ok: bool = False,
    **kwargs,
) -> ContourSpec:
    """
    Create a contour (isosurface) visualization spec.

    Args:
        varname: Variable to contour
        isosurfaces: List of isosurface values (None = auto)
        scalar: Variable to sample onto surface for coloring
        individual_meshes: Create separate mesh per isosurface
        color: Solid color (mutually exclusive with cmap)
        opacity: Opacity from 0.0 to 1.0
        cmap: Colormap for scalar coloring
        clim: Color limits (min, max)
        show_scalar_bar: Show scalar bar
        scalar_bar_title: Title for scalar bar
        style: "surface", "wireframe", or "points"
        material_preset: Blender material preset
        name: Unique identifier
        empty_ok: Don't skip when mesh is empty
        **kwargs: Additional kwargs (pyvista_create_kwargs, pyvista_add_kwargs)

    Returns:
        ContourSpec instance

    Example:
        >>> spec = make_contour("THETA", isosurfaces=[300, 310], opacity=0.7)
    """
    geometry = ContourGeometry(
        varname=varname,
        isosurfaces=isosurfaces,
        scalar=scalar,
        individual_meshes=individual_meshes,
    )
    appearance = ContourAppearance(
        color=color,
        opacity=opacity,
        cmap=cmap,
        clim=clim,
        show_scalar_bar=show_scalar_bar,
        scalar_bar_title=scalar_bar_title or varname,
        style=style,
        material_preset=material_preset,
    )
    return ContourSpec(
        _geometry=geometry,
        _appearance=appearance,
        name=name,
        empty_ok=empty_ok,
        **kwargs,
    )


def make_volume(
    varname: str,
    threshold: Optional[Tuple[Optional[float], Optional[float]]] = None,
    # Appearance
    opacity: float = 1.0,
    opacity_transfer: Optional[List[float]] = None,
    cmap: Optional[str] = None,
    clim: Optional[Tuple[float, float]] = None,
    show_scalar_bar: bool = False,
    scalar_bar_title: Optional[str] = None,
    mapper: str = "smart",
    opacity_unit_distance: Optional[float] = None,
    material_preset: Optional[str] = None,
    # VarSpec base
    name: Optional[str] = None,
    empty_ok: bool = False,
    **kwargs,
) -> VolumeSpec:
    """
    Create a volume rendering spec.

    Args:
        varname: Variable to render
        threshold: (min, max) to clip values (None for unbounded)
        opacity: Base opacity (0.0 to 1.0)
        opacity_transfer: Custom opacity transfer function as list
        cmap: Colormap name
        clim: Color limits (min, max)
        show_scalar_bar: Show scalar bar
        scalar_bar_title: Title for scalar bar
        mapper: PyVista volume mapper ("smart", "fixed_point", "gpu")
        opacity_unit_distance: Controls opacity accumulation
        material_preset: Blender material preset
        name: Unique identifier
        empty_ok: Don't skip when mesh is empty
        **kwargs: Additional kwargs

    Returns:
        VolumeSpec instance

    Example:
        >>> spec = make_volume("QC", clim=(0.001, 0.01), cmap="Greys_r")
    """
    geometry = VolumeGeometry(
        varname=varname,
        threshold=threshold,
    )
    appearance = VolumeAppearance(
        opacity=opacity,
        opacity_transfer=opacity_transfer,
        cmap=cmap,
        clim=clim,
        show_scalar_bar=show_scalar_bar,
        scalar_bar_title=scalar_bar_title or varname,
        mapper=mapper,
        opacity_unit_distance=opacity_unit_distance,
        material_preset=material_preset,
    )
    return VolumeSpec(
        _geometry=geometry,
        _appearance=appearance,
        name=name,
        empty_ok=empty_ok,
        **kwargs,
    )


def make_vectors(
    varname: str,
    u: str = "UC",
    v: str = "VC",
    w: str = "WC",
    scale_by: Optional[str] = None,
    factor: Optional[float] = None,
    tolerance: Optional[float] = None,
    # Appearance
    color: Optional[str] = None,
    opacity: float = 1.0,
    cmap: Optional[str] = None,
    clim: Optional[Tuple[float, float]] = None,
    show_scalar_bar: bool = False,
    glyph_type: str = "arrow",
    # VarSpec base
    name: Optional[str] = None,
    empty_ok: bool = False,
    **kwargs,
) -> VectorSpec:
    """
    Create a vector field glyph spec.

    Args:
        varname: Name for the vector field
        u: Variable for u (x) component
        v: Variable for v (y) component
        w: Variable for w (z) component
        scale_by: Variable to scale arrows by
        factor: Scale factor for glyph size (None = auto)
        tolerance: Point merging tolerance (0 = no merging)
        color: Solid color
        opacity: Opacity (0.0 to 1.0)
        cmap: Colormap for scalar coloring
        clim: Color limits
        show_scalar_bar: Show scalar bar
        glyph_type: "arrow", "cone", or "sphere"
        name: Unique identifier
        empty_ok: Don't skip when mesh is empty
        **kwargs: Additional kwargs

    Returns:
        VectorSpec instance

    Example:
        >>> spec = make_vectors("wind", u="UC", v="VC", w="WC", factor=0.001)
    """
    geometry = VectorGeometry(
        varname=varname,
        u_varname=u,
        v_varname=v,
        w_varname=w,
        scale_by=scale_by,
        factor=factor,
        tolerance=tolerance,
    )
    appearance = VectorAppearance(
        color=color,
        opacity=opacity,
        cmap=cmap,
        clim=clim,
        show_scalar_bar=show_scalar_bar,
        glyph_type=glyph_type,
    )
    return VectorSpec(
        _geometry=geometry,
        _appearance=appearance,
        name=name,
        empty_ok=empty_ok,
        **kwargs,
    )


def make_slice(
    varname: str,
    dim: str = "z",
    value: Optional[float] = None,
    method: str = "nearest",
    # Appearance
    color: Optional[str] = None,
    opacity: float = 1.0,
    cmap: Optional[str] = None,
    clim: Optional[Tuple[float, float]] = None,
    show_scalar_bar: bool = False,
    scalar_bar_title: Optional[str] = None,
    # VarSpec base
    name: Optional[str] = None,
    empty_ok: bool = False,
    **kwargs,
) -> SliceSpec:
    """
    Create a 2D slice spec.

    Args:
        varname: Variable to slice
        dim: Dimension to slice along ('x', 'y', or 'z')
        value: Value at which to slice (None = first index)
        method: Selection method ('nearest' or 'interp')
        color: Solid color
        opacity: Opacity (0.0 to 1.0)
        cmap: Colormap for scalar coloring
        clim: Color limits
        show_scalar_bar: Show scalar bar
        scalar_bar_title: Title for scalar bar
        name: Unique identifier
        empty_ok: Don't skip when mesh is empty
        **kwargs: Additional kwargs

    Returns:
        SliceSpec instance

    Example:
        >>> spec = make_slice("DBZ", dim="z", value=1000, cmap="pyart_NWSRef")
    """
    geometry = SliceGeometry(
        varname=varname,
        slice_dim=dim,
        slice_value=value,
        slice_method=method,
    )
    appearance = Appearance(
        color=color,
        opacity=opacity,
        cmap=cmap,
        clim=clim,
        show_scalar_bar=show_scalar_bar,
        scalar_bar_title=scalar_bar_title or varname,
    )
    return SliceSpec(
        _geometry=geometry,
        _appearance=appearance,
        name=name,
        empty_ok=empty_ok,
        **kwargs,
    )


def make_trajectory(
    scalar: Optional[str] = None,
    color: Optional[str] = None,
    cmap: str = "viridis",
    style: str = "tube",
    limit: Optional[int] = 1000,
    # Geometry
    tube_radius: float = 70,
    head_length_frac: float = 10,
    head_radius_frac: float = 2.5,
    tube_resolution: int = 4,
    head_radial_resolution: int = 30,
    # Appearance
    opacity: float = 1.0,
    clim: Optional[Tuple[float, float]] = None,
    show_scalar_bar: bool = False,
    scalar_bar_title: Optional[str] = None,
    silhouettes: bool = False,
    material_preset: Optional[str] = None,
    # VarSpec base
    name: Optional[str] = None,
    empty_ok: bool = False,
    **kwargs,
) -> TrajectorySpec:
    """
    Create a trajectory visualization spec.

    Args:
        scalar: Variable for scalar coloring
        color: Solid color (alternative to scalar)
        cmap: Colormap for scalar coloring
        style: "tube" or "particle"
        limit: Maximum number of trajectories
        tube_radius: Radius of trajectory tubes
        head_length_frac: Arrow head length as fraction of tube radius
        head_radius_frac: Arrow head radius as fraction of tube radius
        tube_resolution: Number of sides on tube
        head_radial_resolution: Resolution of arrow head
        opacity: Opacity (0.0 to 1.0)
        clim: Color limits
        show_scalar_bar: Show scalar bar
        scalar_bar_title: Title for scalar bar
        silhouettes: Add silhouette effect
        material_preset: Blender material preset
        name: Unique identifier
        empty_ok: Don't skip when mesh is empty
        **kwargs: Additional kwargs

    Returns:
        TrajectorySpec instance

    Example:
        >>> spec = make_trajectory(scalar="altitude", cmap="viridis")
    """
    geometry = TrajectoryGeometry(
        scalar=scalar,
        tube_radius=tube_radius,
        head_length_frac=head_length_frac,
        head_radius_frac=head_radius_frac,
        tube_resolution=tube_resolution,
        head_radial_resolution=head_radial_resolution,
    )
    appearance = TrajectoryAppearance(
        color=color,
        opacity=opacity,
        cmap=cmap,
        clim=clim,
        show_scalar_bar=show_scalar_bar,
        scalar_bar_title=scalar_bar_title or scalar,
        style=style,
        silhouettes=silhouettes,
        material_preset=material_preset,
    )
    return TrajectorySpec(
        _geometry=geometry,
        _appearance=appearance,
        limit=limit,
        name=name,
        empty_ok=empty_ok,
        **kwargs,
    )


# =============================================================================
# SCENE CREATION FUNCTIONS
# =============================================================================


def plot_gridded(
    ds: xr.Dataset,
    contours: Optional[Dict[str, Any]] = None,
    volumes: Optional[Dict[str, Any]] = None,
    vectors: Optional[Dict[str, Any]] = None,
    slices: Optional[Dict[str, Any]] = None,
    scene: Optional[Scene] = None,
    show: bool = True,
    **scene_kwargs,
) -> Scene:
    """
    Create a Scene with gridded data visualizations.

    Args:
        ds: xarray Dataset with gridded data
        contours: Dict mapping varname to isosurface list or spec dict
        volumes: Dict mapping varname to volume spec dict
        vectors: Dict mapping varname to vector spec dict
        slices: Dict mapping varname to slice spec dict
        scene: Existing Scene to add to (creates new if None)
        **scene_kwargs: Passed to Scene constructor (background, title, etc.)

    Returns:
        Scene with visualizations added

    Example:
        >>> scene = plot_gridded(
        ...     ds,
        ...     contours={"THETA": [300, 310], "W": {"isosurfaces": [5], "opacity": 0.5}},
        ...     volumes={"QC": {"cmap": "Greys_r"}},
        ... )
        >>> scene.show()
    """
    scene = scene or Scene(**scene_kwargs)

    if contours:
        scene.add_contours(ds, contours)

    if volumes:
        for varname, spec in volumes.items():
            spec = spec if isinstance(spec, dict) else {}
            scene.add_volume(ds, varname, **spec)

    if vectors:
        for varname, spec in vectors.items():
            spec = spec if isinstance(spec, dict) else {}
            scene.add_vectors(ds, varname, **spec)

    if slices:
        for varname, spec in slices.items():
            spec = spec if isinstance(spec, dict) else {}
            scene.add_slice(ds, varname, **spec)

    if show:
        scene.show()

    return scene


def plot_trajectories(
    ds: xr.Dataset,
    scalar: Optional[str] = None,
    color: Optional[str] = None,
    cmap: str = "viridis",
    style: str = "tube",
    limit: Optional[int] = 1000,
    scene: Optional[Scene] = None,
    show: bool = True,
    **kwargs,
) -> Scene:
    """
    Create a Scene with trajectory visualization.

    Args:
        ds: xarray Dataset with trajectory data
        scalar: Variable for scalar coloring
        color: Solid color (alternative to scalar)
        cmap: Colormap for scalar coloring
        style: "tube" or "particle"
        limit: Maximum number of trajectories
        scene: Existing Scene to add to (creates new if None)
        **kwargs: Additional trajectory kwargs

    Returns:
        Scene with trajectory visualization added

    Example:
        >>> scene = plot_trajectories(traj_ds, scalar="altitude", cmap="viridis")
        >>> scene.show()
    """
    # Extract scene kwargs vs trajectory kwargs
    scene_kwargs = {}
    traj_kwargs = {}
    scene_keys = {"background", "title", "show_grid"}
    for key, value in kwargs.items():
        if key in scene_keys:
            scene_kwargs[key] = value
        else:
            traj_kwargs[key] = value

    scene = scene or Scene(**scene_kwargs)
    scene.add_trajectories(
        ds,
        scalar=scalar,
        color=color,
        cmap=cmap,
        style=style,
        limit=limit,
        **traj_kwargs,
    )

    if show:
        scene.show()

    return scene


# =============================================================================
# DEPRECATED OLD API (for backwards compatibility)
# =============================================================================


def quick_plot(
    simulation_ds: Optional[xr.Dataset] = None,
    trajectory_ds: Optional[xr.Dataset] = None,
    contours: Optional[Dict[str, Any]] = None,
    volumes: Optional[Dict[str, Any]] = None,
    vectors: Optional[Dict[str, Any]] = None,
    slices_2d: Optional[Dict[str, Any]] = None,
    trajectory_color: Optional[str] = None,
    trajectory_scalar: Optional[str] = None,
    trajectory_cmap: str = "viridis",
    trajectory_scalar_bar: bool = False,
    trajectory_particles: bool = False,
    trajectory_limit: Optional[int] = 1000,
    animate: bool = False,
    gif_path: Optional[str] = None,
    fps: float = 10,
    interactive: bool = False,
    show: bool = True,
    show_grid: bool = True,
    screenshot_path: Optional[str] = None,
    export_html: bool = False,
    title: Optional[str] = None,
    plotter=None,
    background: str = "#f8f6f1",
    pv_config_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[Any, Any]:
    """
    DEPRECATED: Use plot_gridded() and plot_trajectories() instead.

    This function is maintained for backwards compatibility but will be
    removed in a future version.
    """
    import warnings

    warnings.warn(
        "quick_plot() is deprecated. Use plot_gridded() and plot_trajectories() "
        "with Scene.show() or Scene.animate() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Build scene using new API
    scene = Scene(background=background, title=title, show_grid=show_grid)

    # Add gridded data
    if simulation_ds is not None:
        if contours:
            scene.add_contours(simulation_ds, contours)
        if volumes:
            for varname, spec in volumes.items():
                spec = spec if isinstance(spec, dict) else {}
                scene.add_volume(simulation_ds, varname, **spec)
        if vectors:
            for varname, spec in vectors.items():
                spec = spec if isinstance(spec, dict) else {}
                scene.add_vectors(simulation_ds, varname, **spec)
        if slices_2d:
            for varname, spec in slices_2d.items():
                spec = spec if isinstance(spec, dict) else {}
                scene.add_slice(simulation_ds, varname, **spec)

    # Add trajectories
    if trajectory_ds is not None:
        scene.add_trajectories(
            trajectory_ds,
            scalar=trajectory_scalar,
            color=trajectory_color,
            cmap=trajectory_cmap,
            style="particle" if trajectory_particles else "tube",
            show_scalar_bar=trajectory_scalar_bar,
            limit=trajectory_limit,
        )

    # Render
    if animate:
        if gif_path:
            scene.animate(gif_path, fps=fps)
        elif interactive:
            scene.interactive_slider()
    elif screenshot_path:
        scene.screenshot(screenshot_path)
        if export_html:
            from pathlib import Path

            html_path = Path(screenshot_path).with_suffix(".html")
            scene.export_html(html_path)
    elif show:
        scene.show()

    # Return empty dict for backwards compatibility (old API returned meshes)
    return {}


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
    DEPRECATED: Use plot_trajectories() instead.
    """
    import warnings

    warnings.warn(
        "plot_trajectories_only() is deprecated. Use plot_trajectories() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    scene = plot_trajectories(
        trajectory_ds,
        scalar=scalar,
        color=color,
        cmap=cmap,
        **kwargs,
    )

    if animate and gif_path:
        scene.animate(gif_path)
    else:
        scene.show()

    return {}


def plot_isosurfaces_only(
    simulation_ds: xr.Dataset,
    contours: Dict[str, Any],
    animate: bool = False,
    gif_path: Optional[str] = None,
    **kwargs,
) -> Dict[Any, Any]:
    """
    DEPRECATED: Use plot_gridded() instead.
    """
    import warnings

    warnings.warn(
        "plot_isosurfaces_only() is deprecated. Use plot_gridded() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    scene = plot_gridded(simulation_ds, contours=contours, **kwargs)

    if animate and gif_path:
        scene.animate(gif_path)
    else:
        scene.show()

    return {}


# =============================================================================
# ALIASES (backwards compatibility)
# =============================================================================

# Alias for consistent singular naming
make_vector = make_vectors
