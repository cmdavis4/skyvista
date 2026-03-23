"""
Scene class for skyvista.

The Scene is the central object that accumulates visualization specs and
renders them to various targets (PyVista, HTML).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pyvista as pv
import xarray as xr

from carlee_tools import NumpyNumeric, PathLike, warn_if_not_evenly_spaced

from .grids import get_grid_builder, merge_bounds_meshes
from .varspec import (
    ContourSpec,
    SliceSpec,
    TrajectorySpec,
    VarSpec,
    VectorSpec,
    VolumeSpec,
)
from .mesh import PVMesh
from .animation import FPS


@dataclass
class Scene:
    """
    Container for visualization specs and scene-level configuration.

    The Scene accumulates (dataset, varspec) pairs and renders them to
    various targets (PyVista, HTML). The Scene is renderer-agnostic
    until a render method is called.

    Attributes:
        background: Background color (hex or name)
        title: Title text to display
        show_grid: Whether to show coordinate grid
        force_bounds: Force scene bounds to match data domain

    Example:
        >>> scene = Scene()
        >>> scene.add_contour(sim_ds, "THETA", isosurfaces=[300, 310])
        >>> scene.add_trajectories(traj_ds, scalar="altitude")
        >>> scene.show()

        Or with chaining:
        >>> Scene().add_contour(ds, "THETA", isosurfaces=[300]).show()
    """

    # Scene-level configuration
    background: str = "#f8f6f1"
    title: Optional[str] = None
    show_grid: bool = True
    force_bounds: bool = False

    # Accumulated specs: list of (dataset, varspec) tuples
    _specs: List[Tuple[xr.Dataset, VarSpec]] = field(default_factory=list)

    # Cached bounds meshes keyed by dataset id
    _bounds_meshes: Dict[int, pv.PolyData] = field(default_factory=dict, repr=False)

    # Cached plotter for interactive use
    _plotter: Optional[pv.Plotter] = field(default=None, repr=False)

    @property
    def plotter(self) -> pv.Plotter:
        """
        Get the PyVista Plotter for this scene.

        Lazily creates the plotter on first access. The plotter can be used
        to set camera position, scale, and other properties before calling
        show().

        Returns:
            PyVista Plotter instance
        """
        if self._plotter is None:
            self._plotter = self._build_plotter()
        return self._plotter

    # -------------------------------------------------------------------------
    # Adding specs
    # -------------------------------------------------------------------------

    def add(self, ds: xr.Dataset, spec: VarSpec) -> "Scene":
        """
        Add a visualization spec to the scene.

        Args:
            ds: xarray Dataset containing the data
            spec: VarSpec defining how to visualize it

        Returns:
            self (for method chaining)
        """
        self._specs.append((ds, spec))

        # Track bounds mesh for this dataset if force_bounds is enabled
        # Use dataset id to avoid duplicating bounds for same dataset
        if self.force_bounds:
            ds_id = id(ds)
            if ds_id not in self._bounds_meshes:
                try:
                    builder = get_grid_builder(ds)
                    self._bounds_meshes[ds_id] = builder.create_bounds_mesh(ds)
                except ValueError as e:
                    import warnings

                    warnings.warn(
                        f"Could not create bounds mesh for dataset: {e}\n"
                        "Scene bounds may not fully enclose the data. "
                        "Set force_bounds=False to suppress this warning.",
                        stacklevel=2,
                    )

        return self

    def add_contour(
        self,
        ds: xr.Dataset,
        varname: str,
        isosurfaces: Optional[List[float]] = None,
        scalar: Optional[str] = None,
        individual_meshes: bool = False,
        color: Optional[str] = None,
        opacity: float = 1.0,
        cmap: Optional[str] = None,
        clim: Optional[Tuple[float, float]] = None,
        show_scalar_bar: bool = False,
        style: str = "surface",
        **kwargs,
    ) -> "Scene":
        """
        Add contour (isosurface) visualization.

        Args:
            ds: xarray Dataset with gridded data
            varname: Variable to contour
            isosurfaces: List of isosurface values
            scalar: Variable to color by (if different from varname)
            individual_meshes: Create separate mesh per isosurface
            color: Solid color
            opacity: Opacity (0-1)
            cmap: Colormap name
            clim: Color limits
            show_scalar_bar: Show scalar bar
            style: "surface", "wireframe", or "points"
            **kwargs: Additional VarSpec kwargs

        Returns:
            self (for method chaining)
        """
        from .convenience import make_contour

        spec = make_contour(
            varname=varname,
            isosurfaces=isosurfaces,
            scalar=scalar,
            individual_meshes=individual_meshes,
            color=color,
            opacity=opacity,
            cmap=cmap,
            clim=clim,
            show_scalar_bar=show_scalar_bar,
            style=style,
            **kwargs,
        )
        return self.add(ds, spec)

    def add_contours(self, ds: xr.Dataset, specs: Dict[str, Any]) -> "Scene":
        """
        Add multiple contours from a dictionary.

        Args:
            ds: xarray Dataset with gridded data
            specs: Dict mapping varname to isosurface list or spec dict

        Returns:
            self (for method chaining)

        Example:
            >>> scene.add_contours(ds, {
            ...     "THETA": [300, 310],  # Simple form
            ...     "W": {"isosurfaces": [5], "opacity": 0.5},  # Full form
            ... })
        """
        for varname, spec in specs.items():
            if isinstance(spec, list):
                # Simple form: just isosurfaces
                self.add_contour(ds, varname, isosurfaces=spec)
            elif isinstance(spec, dict):
                # Full form: dict of params
                self.add_contour(ds, varname, **spec)
            else:
                raise ValueError(
                    f"Contour spec for {varname} must be list or dict, got {type(spec)}"
                )
        return self

    def add_volume(
        self,
        ds: xr.Dataset,
        varname: str,
        threshold: Optional[Tuple[Optional[float], Optional[float]]] = None,
        opacity: float = 1.0,
        opacity_transfer: Optional[List[float]] = None,
        cmap: Optional[str] = None,
        clim: Optional[Tuple[float, float]] = None,
        show_scalar_bar: bool = False,
        mapper: str = "smart",
        **kwargs,
    ) -> "Scene":
        """
        Add volume rendering visualization.

        Args:
            ds: xarray Dataset with gridded data
            varname: Variable to render
            threshold: (min, max) to clip values
            opacity: Base opacity
            opacity_transfer: Custom opacity transfer function
            cmap: Colormap name
            clim: Color limits
            show_scalar_bar: Show scalar bar
            mapper: PyVista volume mapper
            **kwargs: Additional VarSpec kwargs

        Returns:
            self (for method chaining)
        """
        from .convenience import make_volume

        spec = make_volume(
            varname=varname,
            threshold=threshold,
            opacity=opacity,
            opacity_transfer=opacity_transfer,
            cmap=cmap,
            clim=clim,
            show_scalar_bar=show_scalar_bar,
            mapper=mapper,
            **kwargs,
        )
        return self.add(ds, spec)

    def add_vectors(
        self,
        ds: xr.Dataset,
        varname: str,
        u: str = "UC",
        v: str = "VC",
        w: str = "WC",
        scale_by: Optional[str] = None,
        factor: Optional[float] = None,
        color: Optional[str] = None,
        opacity: float = 1.0,
        cmap: Optional[str] = None,
        **kwargs,
    ) -> "Scene":
        """
        Add vector field glyph visualization.

        Args:
            ds: xarray Dataset with gridded data
            varname: Name for the vector field
            u: Variable for u component
            v: Variable for v component
            w: Variable for w component
            scale_by: Variable to scale arrows by
            factor: Scale factor for glyph size
            color: Solid color
            opacity: Opacity (0-1)
            cmap: Colormap name
            **kwargs: Additional VarSpec kwargs

        Returns:
            self (for method chaining)
        """
        from .convenience import make_vectors

        spec = make_vectors(
            varname=varname,
            u=u,
            v=v,
            w=w,
            scale_by=scale_by,
            factor=factor,
            color=color,
            opacity=opacity,
            cmap=cmap,
            **kwargs,
        )
        return self.add(ds, spec)

    def add_slice(
        self,
        ds: xr.Dataset,
        varname: str,
        dim: str = "z",
        value: Optional[float] = None,
        method: str = "nearest",
        cmap: Optional[str] = None,
        clim: Optional[Tuple[float, float]] = None,
        opacity: float = 1.0,
        show_scalar_bar: bool = False,
        **kwargs,
    ) -> "Scene":
        """
        Add 2D slice visualization.

        Args:
            ds: xarray Dataset with gridded data
            varname: Variable to slice
            dim: Dimension to slice along ('x', 'y', or 'z')
            value: Value at which to slice
            method: Selection method ('nearest' or 'interp')
            cmap: Colormap name
            clim: Color limits
            opacity: Opacity (0-1)
            show_scalar_bar: Show scalar bar
            **kwargs: Additional VarSpec kwargs

        Returns:
            self (for method chaining)
        """
        from .convenience import make_slice

        spec = make_slice(
            varname=varname,
            dim=dim,
            value=value,
            method=method,
            cmap=cmap,
            clim=clim,
            opacity=opacity,
            show_scalar_bar=show_scalar_bar,
            **kwargs,
        )
        return self.add(ds, spec)

    def add_trajectories(
        self,
        ds: xr.Dataset,
        scalar: Optional[str] = None,
        color: Optional[str] = None,
        cmap: str = "viridis",
        style: str = "tube",
        limit: Optional[int] = 1000,
        tube_radius: float = 70,
        head_length_frac: float = 10,
        opacity: float = 1.0,
        show_scalar_bar: bool = False,
        silhouettes: bool = False,
        **kwargs,
    ) -> "Scene":
        """
        Add trajectory visualization.

        Args:
            ds: xarray Dataset with trajectory data
            scalar: Variable for scalar coloring
            color: Solid color (alternative to scalar)
            cmap: Colormap for scalar coloring
            style: "tube" or "particle"
            limit: Maximum number of trajectories
            tube_radius: Radius of trajectory tubes
            head_length_frac: Arrow head length fraction
            opacity: Opacity (0-1)
            show_scalar_bar: Show scalar bar
            silhouettes: Add silhouette effect
            **kwargs: Additional VarSpec kwargs

        Returns:
            self (for method chaining)
        """
        from .convenience import make_trajectory

        spec = make_trajectory(
            scalar=scalar,
            color=color,
            cmap=cmap,
            style=style,
            limit=limit,
            tube_radius=tube_radius,
            head_length_frac=head_length_frac,
            opacity=opacity,
            show_scalar_bar=show_scalar_bar,
            silhouettes=silhouettes,
            **kwargs,
        )
        return self.add(ds, spec)

    # -------------------------------------------------------------------------
    # Time utilities
    # -------------------------------------------------------------------------

    def _get_all_times(self) -> List[Any]:
        """Get sorted union of all time values across datasets."""
        all_times = set()
        for ds, _ in self._specs:
            if "time" in ds.dims:
                all_times.update(ds["time"].values)
        return sorted(all_times) if all_times else [None]

    def _get_last_time(self) -> Any:
        """Get the last time value, or None if no time dimension."""
        times = self._get_all_times()
        return times[-1] if times and times[0] is not None else None

    # -------------------------------------------------------------------------
    # Bounds handling
    # -------------------------------------------------------------------------

    def _get_merged_bounds(self) -> Optional[pv.PolyData]:
        """Get merged bounds mesh from all datasets."""
        if not self._bounds_meshes:
            return None

        meshes = list(self._bounds_meshes.values())
        return merge_bounds_meshes(meshes)

    def _add_bounds_to_plotter(self, plotter: pv.Plotter) -> None:
        """Add bounds mesh to plotter if force_bounds is enabled."""
        if not self.force_bounds:
            return

        bounds_mesh = self._get_merged_bounds()
        if bounds_mesh is not None:
            # Add as very transparent wireframe so it forces bounds without being visible
            plotter.add_mesh(
                bounds_mesh,
                name="_bounds",
                color="gray",
                opacity=0.05,
                line_width=0.5,
            )

    # -------------------------------------------------------------------------
    # Rendering (PyVista)
    # -------------------------------------------------------------------------

    def _build_plotter(self) -> pv.Plotter:
        """Create configured PyVista plotter."""
        from .plotter import initialize_plotter

        return initialize_plotter(background=self.background)

    def _render_frame(self, plotter: pv.Plotter, time: Any) -> List[PVMesh]:
        """Render all specs for a single time point."""
        meshes = []
        for ds, spec in self._specs:
            result = spec.render_to_plotter(plotter, ds, time)
            if result is not None:
                meshes.append(result)
        return meshes

    def _add_timestamp(self, plotter: pv.Plotter, time: Any) -> None:
        """Add timestamp text to plotter if time is available."""
        if time is not None:
            # Try to get t_minutes if available
            for ds, _ in self._specs:
                if "t_minutes" in ds.coords:
                    try:
                        t_minutes = ds.sel(time=time)["t_minutes"].values
                        plotter.add_text(
                            f"t={t_minutes:.0f} minutes",
                            position="upper_edge",
                            name="timestamp",
                        )
                        return
                    except (KeyError, ValueError):
                        pass
            # Fall back to raw time value
            plotter.add_text(str(time), position="upper_edge", name="timestamp")

    def show(
        self,
        time: Optional[Any] = None,
        interactive: bool = True,
    ) -> "Scene":
        """
        Display as still image in PyVista.

        Args:
            time: Specific time to render (default: last time)
            interactive: Whether to enable interactive mode

        Returns:
            self (for method chaining)
        """
        # Use cached plotter if already created, otherwise build new one
        plotter = self.plotter
        self._render_frame(plotter, time or self._get_last_time())
        self._add_bounds_to_plotter(plotter)

        if self.show_grid:
            plotter.show_grid()
        if self.title:
            plotter.add_text(self.title, position="upper_edge", name="title")

        plotter.show()
        return self

    def screenshot(
        self,
        path: PathLike,
        time: Optional[Any] = None,
        scale: int = 3,
    ) -> "Scene":
        """
        Save still image to file.

        Args:
            path: Output file path
            time: Specific time to render (default: last time)
            scale: Resolution scale factor

        Returns:
            self (for method chaining)
        """
        plotter = self._build_plotter()
        self._render_frame(plotter, time or self._get_last_time())
        self._add_bounds_to_plotter(plotter)

        if self.show_grid:
            plotter.show_grid()
        if self.title:
            plotter.add_text(self.title, position="upper_edge", name="title")

        plotter.show(auto_close=False)
        plotter.screenshot(str(path), scale=scale)
        plotter.close()
        return self

    def animate(
        self,
        path: PathLike,
        fps: float | FPS = 10,
        times: Optional[List[Any]] = None,
    ) -> "Scene":
        """
        Render animation to GIF.

        Args:
            path: Output file path
            fps: Frames per second
            times: Specific times to render (default: all times)

        Returns:
            self (for method chaining)
        """
        from tqdm.notebook import tqdm

        plotter = self._build_plotter()
        times = times or self._get_all_times()
        # Rather than adding another parameter, assume we should warn if times
        # aren't evenly spaced
        warn_if_not_evenly_spaced(times)

        if self.show_grid:
            plotter.show_grid()

        # Add bounds once at the beginning (static throughout animation)
        self._add_bounds_to_plotter(plotter)

        # Convert our fps object to a number if needed
        if isinstance(fps, FPS):
            fps = fps.to_fps(times)
        plotter.open_gif(str(path), fps=fps)

        for i, t in enumerate(tqdm(times, desc="Rendering frames")):
            self._render_frame(plotter, t)
            self._add_timestamp(plotter, t)

            if i == 0:
                plotter.show(auto_close=False)

            plotter.write_frame()

        plotter.close()
        return self

    def interactive_slider(self) -> "Scene":
        """
        Display with interactive time slider.

        Returns:
            self (for method chaining)
        """
        from tqdm.notebook import tqdm

        plotter = self._build_plotter()
        times = self._get_all_times()

        if not times or times[0] is None:
            raise ValueError("No time dimension found in datasets")

        # Pre-render all frames
        meshes_by_time: Dict[Any, List[PVMesh]] = {}
        for t in tqdm(times, desc="Pre-rendering frames"):
            meshes_by_time[t] = self._render_frame(plotter, t)

        # Add bounds once (static)
        self._add_bounds_to_plotter(plotter)

        if self.show_grid:
            plotter.show_grid()

        # Set up slider callback
        def update_time(value):
            # Find nearest time
            time_idx = min(int(value), len(times) - 1)
            selected_time = times[time_idx]

            # Re-render for this time (meshes update in place via name)
            self._render_frame(plotter, selected_time)
            self._add_timestamp(plotter, selected_time)

        plotter.add_slider_widget(
            callback=update_time,
            rng=[0, len(times) - 1],
            value=len(times) - 1,
            title="Time Index",
            pointa=(0.1, 0.05),
            pointb=(0.9, 0.05),
            style="modern",
        )

        plotter.show()
        return self

    def export_html(
        self,
        path: PathLike,
        time: Optional[Any] = None,
    ) -> "Scene":
        """
        Export to interactive HTML.

        Args:
            path: Output file path
            time: Specific time to render (default: last time)

        Returns:
            self (for method chaining)
        """
        plotter = self._build_plotter()
        self._render_frame(plotter, time or self._get_last_time())
        self._add_bounds_to_plotter(plotter)

        if self.show_grid:
            plotter.show_grid()

        plotter.export_html(str(path))
        return self
