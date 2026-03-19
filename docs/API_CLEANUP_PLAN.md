# API Cleanup Plan

This document outlines a plan to unify and clean up the skyvista OO API, addressing pain points and preparing for the Scene abstraction.

---

## Problem Analysis

### 1. The `create_mesh_kwargs` / `add_mesh_kwargs` Split

**Current situation:**
```python
@dataclass
class PVVarSpec(ABC):
    create_mesh_kwargs: Dict = field(default_factory=dict)  # → pv.contour(), pv.glyph(), etc.
    add_mesh_kwargs: Dict = field(default_factory=dict)     # → plotter.add_mesh()
```

**Why it exists:** PyVista separates mesh creation (e.g., `grid.contour(isosurfaces=[...])`) from mesh rendering (e.g., `plotter.add_mesh(mesh, opacity=0.5)`). The current API mirrors this.

**Why it's painful:**
1. Users must know which PyVista function each kwarg belongs to
2. The `convenience.py` factory functions paper over this with prefix hacks (`create_mesh_` prefix)
3. Some kwargs are ambiguous (e.g., `scalars` can go to either depending on context)
4. The split leaks PyVista implementation details into the domain model
5. For Blender export, we need to understand the *intent* (e.g., "make this semi-transparent blue"), not which PyVista function receives the kwarg

**Evidence of the problem in `convenience.py`:**
```python
# Any remaining kwargs go to add_mesh_kwargs by default
# Users can override with create_mesh_kwargs_ prefix
for key, value in kwargs.items():
    if key.startswith("create_mesh_"):
        create_mesh_kwargs[key.replace("create_mesh_", "")] = value
    else:
        add_mesh_kwargs[key] = value
```

### 2. Inconsistent Parameter Promotion

Some visualization-specific parameters are promoted to first-class dataclass fields, others are buried in kwargs:

| Spec Type | First-class fields | Buried in kwargs |
|-----------|-------------------|------------------|
| `PVContourSpec` | `isosurfaces`, `scalars`, `individual_meshes` | `opacity`, `color`, `cmap` |
| `PVVolumeSpec` | `individual_meshes` | `opacity`, `clim`, `cmap`, `mapper` |
| `PVVectorSpec` | `u/v/w_varname` | `scale`, `factor`, `tolerance` |
| `PVTrajectorySpec` | `color`, `scalar`, `cmap`, `particles`, geometry params | `opacity` |

This inconsistency makes it hard to know what parameters are available without reading the code.

### 3. `PVConfig` Mixes Concerns

`PVConfig` combines:
- Plotter instance (renderer state)
- Output settings (gif_path, screenshot_path, export_html)
- Display settings (show, show_grid, interactive)
- Animation settings (animation, fps)
- Subplot configuration
- User callback

This violates single-responsibility and makes it hard to reuse settings across different render targets.

### 4. Naming Inconsistencies

- `PVRamsData` vs `PVGriddedData` (alias exists but migration incomplete)
- `simulation_ds` vs `trajectory_ds` (inconsistent with generic `ds` property)
- `varname` on specs vs `varnames` concept in Blender kwargs

---

## Proposed Solution

### Core Principle: Semantic Properties Over Implementation Details

Instead of asking "which PyVista function receives this kwarg?", ask "what visual property am I specifying?"

### New Type Hierarchy

```python
# =============================================================================
# APPEARANCE (renderer-agnostic visual properties)
# =============================================================================

@dataclass
class Appearance:
    """Renderer-agnostic visual properties."""
    color: Optional[str] = None           # Solid color (name or hex)
    opacity: float = 1.0                   # 0.0 = transparent, 1.0 = opaque
    cmap: Optional[str] = None             # Colormap name
    clim: Optional[Tuple[float, float]] = None  # Color limits
    show_scalar_bar: bool = False
    scalar_bar_title: Optional[str] = None

    # For Blender export
    material_preset: Optional[str] = None  # e.g., "cloud", "rain", "trajectory"


@dataclass
class ContourAppearance(Appearance):
    """Appearance specific to isosurface rendering."""
    style: str = "surface"  # "surface", "wireframe", "points"


@dataclass
class VolumeAppearance(Appearance):
    """Appearance specific to volume rendering."""
    opacity_transfer: Optional[List[float]] = None  # Custom opacity curve
    mapper: str = "smart"  # PyVista volume mapper


@dataclass
class VectorAppearance(Appearance):
    """Appearance specific to vector glyphs."""
    glyph_type: str = "arrow"  # "arrow", "cone", "sphere"


@dataclass
class TrajectoryAppearance(Appearance):
    """Appearance specific to trajectory rendering."""
    style: str = "tube"  # "tube", "particle", "ribbon"
    tube_radius: float = 70
    head_style: str = "cone"  # "cone", "sphere", "none"
    silhouettes: bool = False


# =============================================================================
# GEOMETRY SPECS (what to extract from data)
# =============================================================================

@dataclass
class ContourGeometry:
    """Specifies how to extract isosurfaces from data."""
    varname: str
    isosurfaces: Optional[List[float]] = None
    color_by: Optional[str] = None  # Variable to color by (if different from varname)
    individual_meshes: bool = False
    # PyVista-specific extraction params (rarely needed)
    method: str = "contour"  # Could support "marching_cubes" etc.


@dataclass
class VolumeGeometry:
    """Specifies how to render a volume."""
    varname: str
    threshold: Optional[Tuple[float, float]] = None  # Clip to range


@dataclass
class VectorGeometry:
    """Specifies how to create vector glyphs."""
    varname: str  # Name for the combined vector field
    u_varname: str = "UC"
    v_varname: str = "VC"
    w_varname: str = "WC"
    scale_by: Optional[str] = None  # Variable to scale arrows by
    factor: Optional[float] = None  # Scale factor
    tolerance: Optional[float] = None  # Point merging tolerance


@dataclass
class SliceGeometry:
    """Specifies how to extract a 2D slice."""
    varname: str
    slice_dim: str = "z"
    slice_value: Optional[float] = None
    slice_method: str = "nearest"


@dataclass
class TrajectoryGeometry:
    """Specifies trajectory rendering geometry."""
    color_by: Optional[str] = None  # Scalar variable for coloring
    tube_radius: float = 70
    head_length_frac: float = 10
    head_radius_frac: float = 2.5
    tube_resolution: int = 4


# =============================================================================
# UNIFIED VARSPEC (combines geometry + appearance)
# =============================================================================

@dataclass
class VarSpec:
    """
    Complete specification for visualizing a variable.

    Combines geometry extraction settings with appearance settings.
    """
    geometry: Union[ContourGeometry, VolumeGeometry, VectorGeometry,
                    SliceGeometry, TrajectoryGeometry]
    appearance: Appearance = field(default_factory=Appearance)
    name: Optional[str] = None  # Unique identifier
    empty_ok: bool = False

    # Escape hatch for PyVista-specific overrides (documented as advanced)
    pyvista_create_kwargs: Dict = field(default_factory=dict)
    pyvista_add_kwargs: Dict = field(default_factory=dict)
```

### Benefits of This Approach

1. **Clear separation of concerns:**
   - Geometry = *what* to extract from data
   - Appearance = *how* it should look
   - PyVista kwargs = escape hatch for edge cases

2. **Renderer-agnostic appearance:**
   - `Appearance` can be translated to PyVista `add_mesh` kwargs
   - `Appearance` can be translated to Blender materials
   - `material_preset` provides a hook for Blender-specific materials

3. **Discoverable API:**
   - All common parameters are explicit dataclass fields
   - IDE autocomplete works
   - No need to know PyVista internals

4. **Backwards compatibility via factory functions:**
   ```python
   def make_contour(varname, isosurfaces=None, opacity=None, color=None, **kwargs):
       geometry = ContourGeometry(varname=varname, isosurfaces=isosurfaces)
       appearance = ContourAppearance(opacity=opacity or 1.0, color=color)
       return VarSpec(geometry=geometry, appearance=appearance, **kwargs)
   ```

### Splitting `PVConfig`

```python
@dataclass
class OutputConfig:
    """Where and how to save output."""
    gif_path: Optional[PathLike] = None
    screenshot_path: Optional[PathLike] = None
    export_html: bool = False
    fps: float = 10.0


@dataclass
class DisplayConfig:
    """How to display the visualization."""
    show: bool = True
    show_grid: bool = False
    interactive: bool = False
    title: Optional[str] = None
    background: str = "#f8f6f1"


@dataclass
class AnimationConfig:
    """Animation settings."""
    enabled: bool = False
    fps: float = 10.0
    # Future: interpolation settings, camera paths, etc.


@dataclass
class RenderConfig:
    """
    Complete configuration for a PyVista render.

    This is the PyVista-specific render target. A parallel BlenderRenderConfig
    would exist for Blender rendering.
    """
    plotter: Optional[pv.Plotter] = None  # Created on demand if None
    output: OutputConfig = field(default_factory=OutputConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    animation: AnimationConfig = field(default_factory=AnimationConfig)
    subplot_config: Dict = field(default_factory=dict)
    callback: Optional[Callable] = None
```

---

## Migration Strategy

### Phase 1: Add New Types Alongside Old

1. Create new `Appearance` and `*Geometry` dataclasses
2. Create new `VarSpec` that composes them
3. Add `VarSpec.to_legacy()` method that returns old-style spec
4. Update `core.py` to accept both old and new specs (check type, convert if needed)

### Phase 2: Update Factory Functions

1. Update `make_contour`, `make_volume`, etc. to return new `VarSpec`
2. These remain the primary user-facing API
3. Old dataclasses still work but emit deprecation warnings

### Phase 3: Update Core Processing

1. Refactor `_create_meshes_for_frame` to work with new types directly
2. Use `appearance.to_pyvista_kwargs()` instead of raw dict access
3. Add `appearance.to_blender_material()` for future Blender export

### Phase 4: Deprecate and Remove

1. Mark old `PVContourSpec`, etc. as deprecated
2. Remove after one release cycle

---

## Example: Before and After

### Before (current API)

```python
# User must know which kwargs go where
spec = PVContourSpec(
    varname="THETA",
    isosurfaces=[300, 310],
    scalars="W",  # First-class field
    create_mesh_kwargs={},  # What goes here?
    add_mesh_kwargs={
        "opacity": 0.7,
        "cmap": "coolwarm",
        "show_scalar_bar": True,  # Wait, is this a kwarg or field?
    }
)
```

### After (new API)

```python
# Clear, discoverable, IDE-friendly
spec = VarSpec(
    geometry=ContourGeometry(
        varname="THETA",
        isosurfaces=[300, 310],
        color_by="W",
    ),
    appearance=ContourAppearance(
        opacity=0.7,
        cmap="coolwarm",
        show_scalar_bar=True,
    ),
)

# Or use the convenience function (unchanged interface!)
spec = make_contour("THETA", isosurfaces=[300, 310], color_by="W",
                    opacity=0.7, cmap="coolwarm", show_scalar_bar=True)
```

---

## Open Questions

1. **Should `Appearance` be a protocol/ABC?** This would allow completely custom appearance types for specialized use cases.

2. **How to handle trajectory-specific geometry?** Trajectories don't extract from gridded data the same way. The current `TrajectoryGeometry` might need a different parent class.

3. **Should we support style presets at the VarSpec level?** E.g., `VarSpec.from_preset("cloud_isosurface", varname="QC")` that sets both geometry and appearance defaults.

4. **What about the `PVMesh` class?** It currently holds both the mesh and the actor. Should it be split into `MeshResult` (geometry) and `RenderResult` (actor)?

---

## Implementation Checklist

- [ ] Design and implement `Appearance` hierarchy
- [ ] Design and implement `*Geometry` classes
- [ ] Create unified `VarSpec` class
- [ ] Add conversion methods (`to_pyvista_kwargs`, `to_legacy`)
- [ ] Split `PVConfig` into focused config classes
- [ ] Update factory functions in `convenience.py`
- [ ] Update `core.py` to handle both old and new types
- [ ] Add deprecation warnings to old types
- [ ] Write migration guide
- [ ] Update tests
- [ ] Update documentation
