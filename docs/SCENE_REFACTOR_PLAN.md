# Scene Refactor Plan

This document outlines the implementation plan for refactoring skyvista to a Scene-centric, Renderable-based architecture.

## Goal

Replace the current `PVConfig` + `PVData` + `PVVarSpec` architecture with:
- **`Scene`**: Container for visualization specs + scene-level config (lighting, background, camera)
- **`VarSpec`**: Self-rendering specs that know how to create meshes and provide render kwargs
- **Convenience API**: Thin wrappers that create Scenes and VarSpecs

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          User Code                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Convenience API                                                    │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │ plot_gridded()  │  │ plot_trajectories│  │ make_contour()    │  │
│  │ → returns Scene │  │ → returns Scene  │  │ → returns VarSpec │  │
│  └────────┬────────┘  └────────┬─────────┘  └───────────────────┘  │
│           │                    │                                    │
│           ▼                    ▼                                    │
├─────────────────────────────────────────────────────────────────────┤
│  Scene                                                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ - _specs: List[Tuple[xr.Dataset, VarSpec]]                  │   │
│  │ - background, lighting, camera (scene-level config)         │   │
│  │                                                              │   │
│  │ + add(ds, spec) → Scene                                     │   │
│  │ + add_contours(), add_volume(), ... (convenience wrappers)  │   │
│  │                                                              │   │
│  │ + show() → renders to PyVista                               │   │
│  │ + animate() → renders animation                             │   │
│  │ + export_blender() → exports to Blender                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│           │                                                         │
│           │ iterates specs, calls spec.render_to_plotter()         │
│           ▼                                                         │
├─────────────────────────────────────────────────────────────────────┤
│  VarSpec (Renderable)                                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ + create_mesh(ds, time) → pv.DataSet      [abstract]        │   │
│  │ + get_pyvista_kwargs() → Dict                               │   │
│  │ + get_blender_config() → Dict                               │   │
│  │ + render_to_plotter(plotter, ds, time) → PVMesh             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│           │                                                         │
│           │ composed of                                             │
│           ▼                                                         │
│  ┌──────────────────┐    ┌──────────────────┐                      │
│  │ Geometry         │    │ Appearance       │                      │
│  │ (what to extract)│    │ (how it looks)   │                      │
│  └──────────────────┘    └──────────────────┘                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Create New Type System (Parallel to Old)

Create the new types without removing the old ones. This allows incremental migration and testing.

#### 1.1 Create Geometry Classes

**File: `skyvista/geometry.py`** (new file)

```python
@dataclass
class Geometry(ABC):
    """Base class for geometry extraction specifications."""
    varname: str

@dataclass
class ContourGeometry(Geometry):
    isosurfaces: Optional[List[float]] = None
    scalar: Optional[str] = None  # Scalar to sample onto surface
    individual_meshes: bool = False

@dataclass
class VolumeGeometry(Geometry):
    threshold: Optional[Tuple[float, float]] = None

@dataclass
class VectorGeometry(Geometry):
    u_varname: str = "UC"
    v_varname: str = "VC"
    w_varname: str = "WC"
    scale_by: Optional[str] = None
    factor: Optional[float] = None
    tolerance: Optional[float] = None

@dataclass
class SliceGeometry(Geometry):
    slice_dim: str = "z"
    slice_value: Optional[float] = None
    slice_method: str = "nearest"

@dataclass
class TrajectoryGeometry:
    """Trajectories don't have a varname in the same sense."""
    scalar: Optional[str] = None
    tube_radius: float = 70
    head_length_frac: float = 10
    head_radius_frac: float = 2.5
    tube_resolution: int = 4
    head_radial_resolution: int = 30
```

**Tasks:**
- [ ] Create `skyvista/geometry.py`
- [ ] Implement all Geometry dataclasses
- [ ] Add unit tests for Geometry construction

#### 1.2 Create Appearance Classes

**File: `skyvista/appearance.py`** (new file)

```python
@dataclass
class Appearance:
    """Renderer-agnostic visual properties."""
    color: Optional[str] = None
    opacity: float = 1.0
    cmap: Optional[str] = None
    clim: Optional[Tuple[float, float]] = None
    show_scalar_bar: bool = False
    scalar_bar_title: Optional[str] = None
    material_preset: Optional[str] = None  # For Blender

    def to_pyvista_kwargs(self) -> Dict[str, Any]:
        """Convert to PyVista add_mesh kwargs."""
        kwargs = {}
        if self.color is not None:
            kwargs["color"] = self.color
        if self.opacity != 1.0:
            kwargs["opacity"] = self.opacity
        if self.cmap is not None:
            kwargs["cmap"] = self.cmap
        if self.clim is not None:
            kwargs["clim"] = self.clim
        if self.show_scalar_bar:
            kwargs["scalar_bar_args"] = {"title": self.scalar_bar_title or ""}
        return kwargs

    def to_blender_config(self) -> Dict[str, Any]:
        """Convert to Blender material/object config."""
        config = {}
        if self.material_preset:
            config["material"] = self.material_preset
        if self.color:
            config["base_color"] = self.color
        if self.opacity < 1.0:
            config["alpha"] = self.opacity
        return config

@dataclass
class ContourAppearance(Appearance):
    style: str = "surface"  # "surface", "wireframe", "points"

@dataclass
class VolumeAppearance(Appearance):
    opacity_transfer: Optional[List[float]] = None
    mapper: str = "smart"
    opacity_unit_distance: Optional[float] = None

@dataclass
class VectorAppearance(Appearance):
    glyph_type: str = "arrow"

@dataclass
class TrajectoryAppearance(Appearance):
    style: str = "tube"  # "tube", "particle"
    silhouettes: bool = False
```

**Tasks:**
- [ ] Create `skyvista/appearance.py`
- [ ] Implement all Appearance dataclasses
- [ ] Implement `to_pyvista_kwargs()` for each
- [ ] Add stub `to_blender_config()` for each
- [ ] Add unit tests

#### 1.3 Create VarSpec Classes

**File: `skyvista/varspec.py`** (new file)

```python
@dataclass
class VarSpec(ABC):
    """Base class for self-rendering visualization specs."""
    name: Optional[str] = None
    empty_ok: bool = False

    # Escape hatches for edge cases
    pyvista_create_kwargs: Dict = field(default_factory=dict)
    pyvista_add_kwargs: Dict = field(default_factory=dict)

    @property
    @abstractmethod
    def geometry(self) -> Geometry:
        """Return the geometry spec."""
        ...

    @property
    @abstractmethod
    def appearance(self) -> Appearance:
        """Return the appearance spec."""
        ...

    @abstractmethod
    def create_mesh(self, ds: xr.Dataset, time: Any) -> Optional[pv.DataSet]:
        """Create PyVista mesh from dataset at given time."""
        ...

    def get_pyvista_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for plotter.add_mesh()."""
        kwargs = self.appearance.to_pyvista_kwargs()
        kwargs.update(self.pyvista_add_kwargs)  # Escape hatch overrides
        return kwargs

    def get_blender_config(self) -> Dict[str, Any]:
        """Get config for Blender export."""
        return self.appearance.to_blender_config()

    def is_volume(self) -> bool:
        """Whether this spec renders as a volume (vs surface mesh)."""
        return False

    def render_to_plotter(
        self,
        plotter: pv.Plotter,
        ds: xr.Dataset,
        time: Any,
    ) -> Optional[PVMesh]:
        """Create mesh and add to plotter."""
        mesh = self.create_mesh(ds, time)

        if mesh is None:
            return None
        if len(mesh.points) == 0 and not self.empty_ok:
            return None

        kwargs = self.get_pyvista_kwargs()

        if self.is_volume():
            actor = plotter.add_volume(mesh, name=self.name, **kwargs)
        else:
            actor = plotter.add_mesh(
                mesh,
                name=self.name,
                show_scalar_bar=self.appearance.show_scalar_bar,
                **kwargs
            )

        return PVMesh(varspec=self, mesh=mesh, actor=actor, time=time)
```

Then implement concrete specs: `ContourSpec`, `VolumeSpec`, `VectorSpec`, `SliceSpec`, `TrajectorySpec`.

**Tasks:**
- [ ] Create `skyvista/varspec.py`
- [ ] Implement `VarSpec` base class with `render_to_plotter()`
- [ ] Implement `ContourSpec` with `create_mesh()`
- [ ] Implement `VolumeSpec` with `create_mesh()` and `is_volume() -> True`
- [ ] Implement `VectorSpec` with `create_mesh()`
- [ ] Implement `SliceSpec` with `create_mesh()`
- [ ] Implement `TrajectorySpec` with `create_mesh()`
- [ ] Add unit tests for each spec's `create_mesh()`

#### 1.4 Create Shared Utilities

**File: `skyvista/grid_utils.py`** (new file)

```python
def select_time(ds: xr.Dataset, time: Any) -> xr.Dataset:
    """Select a single time from dataset, or return as-is if no time dim."""
    if time is not None and "time" in ds.dims:
        return ds.sel(time=time)
    return ds

def build_rectilinear_grid(ds: xr.Dataset) -> pv.RectilinearGrid:
    """Build PyVista RectilinearGrid from xarray Dataset."""
    return pv.RectilinearGrid(
        ds["x"].values,
        ds["y"].values,
        ds["z"].values,
    )

def add_scalar_to_grid(
    grid: pv.RectilinearGrid,
    ds: xr.Dataset,
    varname: str
) -> None:
    """Add a scalar field from dataset to grid."""
    grid[varname] = ds[varname].values.ravel(order="F")
```

**Tasks:**
- [ ] Create `skyvista/grid_utils.py`
- [ ] Implement shared utilities
- [ ] Use in VarSpec implementations to reduce duplication

---

### Phase 2: Create Scene Class

**File: `skyvista/scene.py`** (new file)

```python
@dataclass
class Scene:
    """
    Container for visualization specs and scene-level configuration.

    The Scene accumulates (dataset, varspec) pairs and renders them
    to various targets (PyVista, Blender, HTML).
    """

    # Scene-level configuration
    background: str = "#f8f6f1"
    title: Optional[str] = None
    show_grid: bool = True

    # Future: camera, lighting
    # camera: Optional[CameraConfig] = None
    # lighting: Optional[LightingConfig] = None

    # Accumulated specs
    _specs: List[Tuple[xr.Dataset, VarSpec]] = field(default_factory=list)

    # -------------------------------------------------------------------------
    # Adding specs
    # -------------------------------------------------------------------------

    def add(self, ds: xr.Dataset, spec: VarSpec) -> "Scene":
        """Add a visualization spec to the scene."""
        self._specs.append((ds, spec))
        return self

    # Convenience methods (delegate to factory functions)
    def add_contour(self, ds: xr.Dataset, varname: str, **kwargs) -> "Scene":
        return self.add(ds, make_contour(varname, **kwargs))

    def add_contours(self, ds: xr.Dataset, specs: Dict[str, Any]) -> "Scene":
        """Add multiple contours from a dict."""
        for varname, spec in specs.items():
            if isinstance(spec, list):
                # Simple form: just isosurfaces
                self.add_contour(ds, varname, isosurfaces=spec)
            elif isinstance(spec, dict):
                self.add_contour(ds, varname, **spec)
            else:
                raise ValueError(f"Invalid contour spec for {varname}")
        return self

    def add_volume(self, ds: xr.Dataset, varname: str, **kwargs) -> "Scene":
        return self.add(ds, make_volume(varname, **kwargs))

    def add_vectors(self, ds: xr.Dataset, varname: str, **kwargs) -> "Scene":
        return self.add(ds, make_vectors(varname, **kwargs))

    def add_slice(self, ds: xr.Dataset, varname: str, **kwargs) -> "Scene":
        return self.add(ds, make_slice(varname, **kwargs))

    def add_trajectories(self, ds: xr.Dataset, **kwargs) -> "Scene":
        return self.add(ds, make_trajectory(**kwargs))

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

    def show(
        self,
        time: Optional[Any] = None,
        interactive: bool = True,
    ) -> "Scene":
        """Display as still image in PyVista."""
        plotter = self._build_plotter()
        self._render_frame(plotter, time or self._get_last_time())

        if self.show_grid:
            plotter.show_grid()
        if self.title:
            plotter.add_text(self.title, position="upper_edge")

        plotter.show()
        return self

    def screenshot(
        self,
        path: PathLike,
        time: Optional[Any] = None,
        scale: int = 3,
    ) -> "Scene":
        """Save still image to file."""
        plotter = self._build_plotter()
        self._render_frame(plotter, time or self._get_last_time())

        if self.show_grid:
            plotter.show_grid()
        if self.title:
            plotter.add_text(self.title, position="upper_edge")

        plotter.show(auto_close=False)
        plotter.screenshot(str(path), scale=scale)
        plotter.close()
        return self

    def animate(
        self,
        path: PathLike,
        fps: float = 10,
        times: Optional[List[Any]] = None,
    ) -> "Scene":
        """Render animation to GIF."""
        from tqdm.notebook import tqdm

        plotter = self._build_plotter()
        times = times or self._get_all_times()

        if self.show_grid:
            plotter.show_grid()

        plotter.open_gif(str(path), fps=fps)

        for i, t in enumerate(tqdm(times, desc="Rendering frames")):
            self._render_frame(plotter, t)

            # Add timestamp if available
            if t is not None:
                plotter.add_text(f"t={t}", position="upper_edge", name="timestamp")

            if i == 0:
                plotter.show(auto_close=False)

            plotter.write_frame()

        plotter.close()
        return self

    def interactive_slider(self) -> "Scene":
        """Display with interactive time slider."""
        # Implementation similar to current _create_interactive_time_slider
        ...
        return self

    def export_html(
        self,
        path: PathLike,
        time: Optional[Any] = None,
    ) -> "Scene":
        """Export to interactive HTML."""
        plotter = self._build_plotter()
        self._render_frame(plotter, time or self._get_last_time())
        plotter.export_html(str(path))
        return self

    # -------------------------------------------------------------------------
    # Blender export
    # -------------------------------------------------------------------------

    def export_blender(
        self,
        output_dir: PathLike,
        preset: str = "quick_preview",
        times: Optional[List[Any]] = None,
    ) -> "Scene":
        """Export scene to Blender-ready format."""
        from pathlib import Path

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        times = times or self._get_all_times()

        # Export meshes with their configs
        for t in times:
            for ds, spec in self._specs:
                mesh = spec.create_mesh(ds, t)
                if mesh is not None and len(mesh.points) > 0:
                    config = spec.get_blender_config()
                    self._export_mesh_for_blender(mesh, spec, config, t, output_dir)

        # Export scene config (background, lighting, camera)
        self._export_scene_config(output_dir, preset)

        return self

    def _export_mesh_for_blender(
        self,
        mesh: pv.DataSet,
        spec: VarSpec,
        config: Dict,
        time: Any,
        output_dir: Path,
    ) -> None:
        """Export a single mesh for Blender import."""
        # Generate filename
        time_str = str(time).replace(" ", "_").replace(":", "-") if time else "static"
        filename = f"{time_str}_{spec.name}.vtk"

        # Save mesh
        mesh.save(output_dir / filename)

        # Save config as sidecar JSON
        config_file = (output_dir / filename).with_suffix(".json")
        import json
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

    def _export_scene_config(self, output_dir: Path, preset: str) -> None:
        """Export scene-level configuration."""
        config = {
            "background": self.background,
            "preset": preset,
            # Future: camera, lighting
        }
        import json
        with open(output_dir / "scene_config.json", "w") as f:
            json.dump(config, f, indent=2)
```

**Tasks:**
- [ ] Create `skyvista/scene.py`
- [ ] Implement `Scene` class with `add()` method
- [ ] Implement convenience `add_*` methods
- [ ] Implement `_get_all_times()` and `_get_last_time()`
- [ ] Implement `show()`
- [ ] Implement `screenshot()`
- [ ] Implement `animate()`
- [ ] Implement `interactive_slider()`
- [ ] Implement `export_html()`
- [ ] Implement `export_blender()` (basic version)
- [ ] Add integration tests

---

### Phase 3: Create New Convenience API

**File: `skyvista/convenience.py`** (replace existing)

```python
"""
Convenience API for skyvista.

These functions create and populate Scene objects, providing a simple
interface for common visualization tasks.
"""

from .scene import Scene
from .varspec import (
    ContourSpec, VolumeSpec, VectorSpec, SliceSpec, TrajectorySpec
)
from .geometry import (
    ContourGeometry, VolumeGeometry, VectorGeometry,
    SliceGeometry, TrajectoryGeometry
)
from .appearance import (
    ContourAppearance, VolumeAppearance, VectorAppearance,
    Appearance, TrajectoryAppearance
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
    style: str = "surface",
    material_preset: Optional[str] = None,
    **kwargs,
) -> ContourSpec:
    """Create a contour (isosurface) visualization spec."""
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
        style=style,
        material_preset=material_preset,
    )
    return ContourSpec(geometry=geometry, appearance=appearance, **kwargs)


def make_volume(
    varname: str,
    threshold: Optional[Tuple[float, float]] = None,
    # Appearance
    opacity: float = 1.0,
    opacity_transfer: Optional[List[float]] = None,
    cmap: Optional[str] = None,
    clim: Optional[Tuple[float, float]] = None,
    show_scalar_bar: bool = False,
    mapper: str = "smart",
    material_preset: Optional[str] = None,
    **kwargs,
) -> VolumeSpec:
    """Create a volume rendering spec."""
    geometry = VolumeGeometry(varname=varname, threshold=threshold)
    appearance = VolumeAppearance(
        opacity=opacity,
        opacity_transfer=opacity_transfer,
        cmap=cmap,
        clim=clim,
        show_scalar_bar=show_scalar_bar,
        mapper=mapper,
        material_preset=material_preset,
    )
    return VolumeSpec(geometry=geometry, appearance=appearance, **kwargs)


def make_vectors(
    varname: str,
    u: str = "UC",
    v: str = "VC",
    w: str = "WC",
    scale_by: Optional[str] = None,
    factor: Optional[float] = None,
    # Appearance
    color: Optional[str] = None,
    opacity: float = 1.0,
    cmap: Optional[str] = None,
    **kwargs,
) -> VectorSpec:
    """Create a vector field glyph spec."""
    geometry = VectorGeometry(
        varname=varname,
        u_varname=u,
        v_varname=v,
        w_varname=w,
        scale_by=scale_by,
        factor=factor,
    )
    appearance = VectorAppearance(
        color=color,
        opacity=opacity,
        cmap=cmap,
    )
    return VectorSpec(geometry=geometry, appearance=appearance, **kwargs)


def make_slice(
    varname: str,
    dim: str = "z",
    value: Optional[float] = None,
    method: str = "nearest",
    # Appearance
    cmap: Optional[str] = None,
    clim: Optional[Tuple[float, float]] = None,
    opacity: float = 1.0,
    show_scalar_bar: bool = False,
    **kwargs,
) -> SliceSpec:
    """Create a 2D slice spec."""
    geometry = SliceGeometry(
        varname=varname,
        slice_dim=dim,
        slice_value=value,
        slice_method=method,
    )
    appearance = Appearance(
        cmap=cmap,
        clim=clim,
        opacity=opacity,
        show_scalar_bar=show_scalar_bar,
    )
    return SliceSpec(geometry=geometry, appearance=appearance, **kwargs)


def make_trajectory(
    scalar: Optional[str] = None,
    color: Optional[str] = None,
    cmap: str = "viridis",
    style: str = "tube",
    limit: Optional[int] = 1000,
    # Geometry
    tube_radius: float = 70,
    head_length_frac: float = 10,
    # Appearance
    opacity: float = 1.0,
    show_scalar_bar: bool = False,
    silhouettes: bool = False,
    **kwargs,
) -> TrajectorySpec:
    """Create a trajectory visualization spec."""
    geometry = TrajectoryGeometry(
        scalar=scalar,
        tube_radius=tube_radius,
        head_length_frac=head_length_frac,
    )
    appearance = TrajectoryAppearance(
        color=color,
        cmap=cmap,
        opacity=opacity,
        show_scalar_bar=show_scalar_bar,
        style=style,
        silhouettes=silhouettes,
    )
    return TrajectorySpec(
        geometry=geometry,
        appearance=appearance,
        limit=limit,
        **kwargs
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
        **scene_kwargs: Passed to Scene constructor

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

    return scene


def plot_trajectories(
    ds: xr.Dataset,
    scalar: Optional[str] = None,
    color: Optional[str] = None,
    cmap: str = "viridis",
    style: str = "tube",
    limit: Optional[int] = 1000,
    scene: Optional[Scene] = None,
    **kwargs,
) -> Scene:
    """
    Create a Scene with trajectory visualization.

    Args:
        ds: xarray Dataset with trajectory data
        scalar: Variable name for scalar coloring
        color: Solid color (alternative to scalar)
        cmap: Colormap for scalar coloring
        style: "tube" or "particle"
        limit: Maximum number of trajectories
        scene: Existing Scene to add to (creates new if None)
        **kwargs: Additional arguments

    Returns:
        Scene with trajectory visualization added

    Example:
        >>> scene = plot_trajectories(traj_ds, scalar="altitude", cmap="viridis")
        >>> scene.show()
    """
    scene = scene or Scene()
    scene.add_trajectories(
        ds,
        scalar=scalar,
        color=color,
        cmap=cmap,
        style=style,
        limit=limit,
        **kwargs,
    )
    return scene
```

**Tasks:**
- [ ] Rewrite `convenience.py` with new factory functions
- [ ] Implement `plot_gridded()`
- [ ] Implement `plot_trajectories()`
- [ ] Add docstrings with examples
- [ ] Add tests

---

### Phase 4: Update Package Exports

**File: `skyvista/__init__.py`** (update)

```python
"""
Skyvista: 3D atmospheric data visualization.

Primary API:
    Scene          - Container for building visualizations
    plot_gridded   - Quick function for gridded data
    plot_trajectories - Quick function for trajectory data

Factory functions:
    make_contour, make_volume, make_vectors, make_slice, make_trajectory
"""

from .scene import Scene
from .convenience import (
    plot_gridded,
    plot_trajectories,
    make_contour,
    make_volume,
    make_vectors,
    make_slice,
    make_trajectory,
)

# For advanced users
from .varspec import (
    VarSpec,
    ContourSpec,
    VolumeSpec,
    VectorSpec,
    SliceSpec,
    TrajectorySpec,
)
from .geometry import (
    Geometry,
    ContourGeometry,
    VolumeGeometry,
    VectorGeometry,
    SliceGeometry,
    TrajectoryGeometry,
)
from .appearance import (
    Appearance,
    ContourAppearance,
    VolumeAppearance,
    VectorAppearance,
    TrajectoryAppearance,
)

__all__ = [
    # Primary API
    "Scene",
    "plot_gridded",
    "plot_trajectories",
    # Factory functions
    "make_contour",
    "make_volume",
    "make_vectors",
    "make_slice",
    "make_trajectory",
    # Advanced: VarSpecs
    "VarSpec",
    "ContourSpec",
    "VolumeSpec",
    "VectorSpec",
    "SliceSpec",
    "TrajectorySpec",
    # Advanced: Geometry
    "Geometry",
    "ContourGeometry",
    "VolumeGeometry",
    "VectorGeometry",
    "SliceGeometry",
    "TrajectoryGeometry",
    # Advanced: Appearance
    "Appearance",
    "ContourAppearance",
    "VolumeAppearance",
    "VectorAppearance",
    "TrajectoryAppearance",
]
```

**Tasks:**
- [ ] Update `__init__.py` with new exports
- [ ] Ensure old exports still work (deprecation period)

---

### Phase 5: Deprecate Old API

**Tasks:**
- [ ] Add deprecation warnings to old `types_sv.py` classes
- [ ] Add deprecation warnings to old `convenience.py` functions
- [ ] Update `core.py` to use new types internally
- [ ] Document migration path
- [ ] Update all examples and notebooks

```python
# In types_sv.py
import warnings

@dataclass
class PVContourSpec(PVVarSpec):
    def __post_init__(self):
        warnings.warn(
            "PVContourSpec is deprecated. Use skyvista.ContourSpec instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__post_init__()
```

---

### Phase 6: Remove Old API

After one release cycle with deprecation warnings:

**Tasks:**
- [ ] Remove `types_sv.py` (or keep only `PVMesh` if still needed)
- [ ] Remove old `core.py` functions
- [ ] Remove old convenience functions
- [ ] Final cleanup

---

## File Structure After Refactor

```
skyvista/
├── __init__.py          # Public API exports
├── scene.py             # Scene class (NEW)
├── varspec.py           # VarSpec classes (NEW)
├── geometry.py          # Geometry classes (NEW)
├── appearance.py        # Appearance classes (NEW)
├── grid_utils.py        # Shared utilities (NEW)
├── convenience.py       # Factory functions + plot_* (REWRITTEN)
├── plotter.py           # PyVista plotter initialization (KEEP)
├── trajectories.py      # Trajectory mesh generation (KEEP, maybe refactor)
├── camera.py            # Camera utilities (KEEP)
├── presets.py           # Presets (EXPAND in Phase 4 of roadmap)
├── types_sv.py          # DEPRECATED, then REMOVE
├── core.py              # DEPRECATED, then REMOVE
└── blender/
    ├── __init__.py
    ├── blender_import.py    # KEEP
    ├── blender_core.py      # KEEP
    ├── blender_run.py       # REFACTOR to use Scene
    └── ...
```

---

## Testing Strategy

1. **Unit tests for new types:**
   - Geometry construction and validation
   - Appearance `to_pyvista_kwargs()` conversion
   - VarSpec `create_mesh()` with mock data

2. **Integration tests:**
   - `Scene.show()` with various spec combinations
   - `Scene.animate()` produces valid GIF
   - `Scene.export_blender()` produces expected files

3. **Regression tests:**
   - Ensure old convenience API (`quick_plot`) still works during deprecation
   - Compare rendered output between old and new APIs

---

## Migration Example

### Before (Old API)

```python
from skyvista import quick_plot

meshes = quick_plot(
    simulation_ds=sim_ds,
    trajectory_ds=traj_ds,
    contours={"THETA": [300, 310]},
    trajectory_scalar="altitude",
    trajectory_cmap="viridis",
    show=True,
    animate=False,
)
```

### After (New API)

```python
import skyvista as sv

scene = sv.plot_gridded(sim_ds, contours={"THETA": [300, 310]})
scene.add_trajectories(traj_ds, scalar="altitude", cmap="viridis")
scene.show()
```

Or chained:

```python
import skyvista as sv

sv.Scene()\
    .add_contours(sim_ds, {"THETA": [300, 310]})\
    .add_trajectories(traj_ds, scalar="altitude", cmap="viridis")\
    .show()
```

---

## Summary Checklist

### Phase 1: New Type System
- [ ] `geometry.py` with all Geometry classes
- [ ] `appearance.py` with all Appearance classes
- [ ] `varspec.py` with VarSpec base and all concrete specs
- [ ] `grid_utils.py` with shared utilities
- [ ] Unit tests for all new types

### Phase 2: Scene Class
- [ ] `scene.py` with full Scene implementation
- [ ] Integration tests for Scene

### Phase 3: Convenience API
- [ ] Rewrite `convenience.py`
- [ ] `plot_gridded()` and `plot_trajectories()`
- [ ] All `make_*` factory functions

### Phase 4: Package Exports
- [ ] Update `__init__.py`

### Phase 5: Deprecation
- [ ] Add warnings to old types
- [ ] Migration documentation

### Phase 6: Removal
- [ ] Remove deprecated code
- [ ] Final cleanup
