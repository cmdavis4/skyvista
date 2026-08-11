"""
Blender export backend for skyvista.

This module turns a :class:`~skyvista.scene.Scene` into a self-contained
*bundle* on disk -- data files plus a ``scene.json`` manifest -- that a Blender
build script (or the sciblend fork) can assemble into a ``.blend`` WITHOUT
Blender or ``bpy`` ever being importable in skyvista's own environment.

The design boundary is deliberate: skyvista runs in your normal (pixi/pip)
environment and only ever *writes* geometry + a declarative description;
everything that touches ``bpy`` lives on the Blender side and reads the bundle.

Carriers
--------
* Time-varying *meshes* (contours, slices, glyph meshes, trajectory tubes) are
  written as **Alembic** (``.abc``) with per-frame changing topology, via the
  ``alembic3d`` bindings. Blender ingests these through a Mesh Sequence Cache
  modifier, which supports the changing vertex/face counts that isosurfaces
  produce every timestep.
* Volumes are written as **VDB** sequences (handled elsewhere / later).

Coloring
--------
Scientific colormaps are baked on the skyvista side into per-vertex **C3f
vertex colors**. This is the only per-vertex attribute that survives Blender's
Alembic importer (raw float ``arbGeomParams`` do not import); it also keeps the
figure's colors deterministic across frames instead of letting Blender
renormalise per frame.

Optional dependency
-------------------
Requires the ``blender`` extra::

    pip install skyvista[blender]

which pulls in ``alembic3d`` -- the 3D Alembic bindings. NOTE: this is *not*
the PyPI package ``alembic`` (that is the unrelated SQLAlchemy database
migration tool); mixing them up is a classic and confusing failure mode.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from carlee_tools import PathLike

if TYPE_CHECKING:
    import pyvista as pv

    from .scene import Scene
    from .varspec import VarSpec


# The manifest schema version. Bump when the on-disk contract changes in a way
# the Blender-side reader must care about.
SKYVISTA_MANIFEST_VERSION = "0.1"

# Blender's Alembic importer only reads recognised typed params (colors, UVs,
# velocities). We carry the baked colormap under this name; it becomes a
# Blender Color Attribute of the same name.
VERTEX_COLOR_ATTRIBUTE_NAME = "color"

# Default colormap when an appearance requests scalar coloring but names none.
DEFAULT_COLORMAP_NAME = "viridis"


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class BlenderTransform:
    """
    The single spatial transform applied identically to every object, the
    camera, and the lights, so that all spatial relationships are preserved and
    the mapping back to physical units stays recoverable.

    The geometry written to disk stays in *physical* units (meters, etc.); this
    transform is recorded in the manifest and applied on the Blender side (as a
    parent-empty / object transform), so a Blender user still sees real
    coordinates and can inspect or override the transform with standard tools.

    The mapping is::

        blender_xyz = (data_xyz - origin_shift) * scale
        blender_z  *= z_exaggeration        # extra vertical factor, on top

    Attributes:
        scale: Uniform data-units -> Blender-units factor. Default 1/1000
            (e.g. meters -> "kilometer-sized" Blender units), which keeps
            atmospheric domains at a sane magnitude for Blender's single-
            precision math and default clip planes.
        origin_shift: Point in data units mapped to the Blender origin. When
            None, defaults to the center of the merged data bounds so the
            figure sits centered on the world origin (nice for orbiting and
            camera framing). Pass (0, 0, 0) to keep absolute data positions.
        z_exaggeration: Extra multiplier applied to the vertical axis only.
            1.0 = physically faithful; >1 exaggerates relief. Always recorded.
    """

    scale: float = 1.0e-3
    origin_shift: Optional[Tuple[float, float, float]] = None
    z_exaggeration: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin_shift": list(self.origin_shift)
            if self.origin_shift is not None
            else None,
            "scale": self.scale,
            "z_exaggeration": self.z_exaggeration,
        }


@dataclass
class BlenderRenderConfig:
    """
    Render-level settings recorded in the manifest for the build script.

    Attributes:
        engine: "CYCLES" (paper-quality, needed for volumes) or "EEVEE" (fast
            preview).
        samples: Render samples per pixel.
        resolution: (width, height) in pixels.
        view_transform: Color-management view transform. "Standard" keeps
            colors data-faithful; Blender's default AgX/Filmic would silently
            alter a scientific figure's colors, so "Standard" is the default.
        film_transparent: Whether the render background is transparent.
    """

    engine: str = "CYCLES"
    samples: int = 128
    resolution: Tuple[int, int] = (1920, 1080)
    view_transform: str = "Standard"
    film_transparent: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "samples": self.samples,
            "resolution": list(self.resolution),
            "view_transform": self.view_transform,
            "film_transparent": self.film_transparent,
        }


@dataclass
class BlenderCameraConfig:
    """
    How to place and animate the camera.

    Camera keyframes are written to the manifest in *data* (physical) space;
    the build script applies the same scene transform to them as to the
    geometry, so the camera stays consistent with the data.

    Attributes:
        mode: "static" (one fixed 3/4 view), "orbit" (circle the scene over the
            animation), or "follow" (track a moving feature via position vars).
        lens_mm: Camera focal length in millimeters.
        distance_factor: Camera distance as a multiple of the data bounding-box
            diagonal (larger = further away / more zoomed out).
        azimuth_deg: Horizontal viewing angle (0 = +x, 90 = +y).
        elevation_deg: Vertical angle above the horizon.
        n_orbit_keyframes: Number of keyframes for "orbit" mode.
        orbit_revolutions: How many full turns "orbit" makes over the animation.
        follow_position_vars: (x_var, y_var) giving the tracked feature's
            horizontal position over time, used by "follow" mode. Defaults to
            the storm-tracking convention used elsewhere in skyvista.
    """

    mode: str = "static"
    lens_mm: float = 50.0
    distance_factor: float = 2.0
    azimuth_deg: float = -55.0
    elevation_deg: float = 22.0
    n_orbit_keyframes: int = 24
    orbit_revolutions: float = 1.0
    follow_position_vars: Tuple[str, str] = (
        "storm_position_x",
        "storm_position_y",
    )


@dataclass
class BlenderExportConfig:
    """
    Top-level configuration for a Blender export.

    Attributes:
        transform: The single spatial transform (see :class:`BlenderTransform`).
        render: Render-level settings (see :class:`BlenderRenderConfig`).
        camera: Camera placement/animation (see :class:`BlenderCameraConfig`).
        fps: Frames per second the animation advertises; also the rate at which
            Alembic samples are spaced in time.
        frame_start: First Blender frame number.
        world_preset: Named lighting/world preset for a decent out-of-the-box
            look without Blender knowledge (interpreted on the Blender side).
    """

    transform: BlenderTransform = field(default_factory=BlenderTransform)
    render: BlenderRenderConfig = field(default_factory=BlenderRenderConfig)
    camera: BlenderCameraConfig = field(default_factory=BlenderCameraConfig)
    fps: float = 24.0
    frame_start: int = 1
    world_preset: str = "studio"


# =============================================================================
# Colormap baking (scalar field -> per-vertex RGB)
# =============================================================================
def bake_scalar_to_rgb(
    scalar_values: np.ndarray,
    colormap_name: str,
    color_limits: Tuple[float, float],
) -> np.ndarray:
    """
    Reproduce a scientific colormap as explicit per-vertex RGB.

    We bake colors on the skyvista side (rather than storing the raw scalar and
    letting Blender do it) because (a) raw float attributes do not survive
    Blender's Alembic import, and (b) baking with a fixed global color range
    guarantees the colors are identical across frames instead of drifting.

    Args:
        scalar_values: (n_points,) scalar field to color by.
        colormap_name: Any matplotlib colormap name (e.g. "viridis").
        color_limits: (vmin, vmax) mapped to the colormap ends. Held fixed
            across all frames so animated colors stay consistent.

    Returns:
        (n_points, 3) float array of RGB in [0, 1].
    """
    import matplotlib
    from matplotlib.colors import Normalize

    colormap = matplotlib.colormaps[colormap_name]
    normalize_to_unit_interval = Normalize(
        vmin=color_limits[0], vmax=color_limits[1]
    )
    rgba_values = colormap(normalize_to_unit_interval(np.asarray(scalar_values)))
    return np.ascontiguousarray(rgba_values[:, :3], dtype=np.float32)


def sample_colormap_stops(
    colormap_name: str, n_stops: int = 32
) -> List[List[float]]:
    """
    Sample a matplotlib colormap into ``[position, r, g, b]`` stops.

    Volumes store the *raw* scalar in the VDB (not baked colors), so the color
    ramp is rebuilt on the Blender side. Blender's bundled Python has no
    matplotlib, so we bake the colormap into explicit stops here and ship them
    in the bundle for the build script to load into a ColorRamp node.
    """
    import matplotlib

    colormap = matplotlib.colormaps[colormap_name]
    stop_positions = np.linspace(0.0, 1.0, n_stops)
    stops: List[List[float]] = []
    for position in stop_positions:
        r, g, b, _ = colormap(float(position))
        stops.append([float(position), float(r), float(g), float(b)])
    return stops


def write_colormap_lut(
    bundle_dir: Path, colormap_name: str, n_stops: int = 32
) -> str:
    """
    Write a colormap's stops to ``assets/colormaps/<cmap>.json`` in the bundle.

    Returns the path relative to the bundle, for the manifest to reference.
    """
    colormap_assets_dir = bundle_dir / "assets" / "colormaps"
    colormap_assets_dir.mkdir(parents=True, exist_ok=True)
    lut_path = colormap_assets_dir / f"{colormap_name}.json"
    with open(lut_path, "w") as lut_file:
        json.dump(
            {
                "cmap": colormap_name,
                "stops": sample_colormap_stops(colormap_name, n_stops),
            },
            lut_file,
        )
    return str(lut_path.relative_to(bundle_dir))


def write_colorbar_png(
    bundle_dir: Path,
    name: str,
    colormap_name: str,
    color_limits: Tuple[float, float],
    label: str = "",
) -> str:
    """
    Render a standalone scientific colorbar to ``assets/colorbars/<name>.png``.

    Blender cannot draw a data colorbar, so we render one with matplotlib (using
    the exact colormap + limits the figure was colored with) for compositing
    over the render or dropping into a paper. Uses the Figure API directly to
    avoid touching pyplot's global state / backend.

    Returns the path relative to the bundle, for the manifest to reference.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure

    colorbar_dir = bundle_dir / "assets" / "colorbars"
    colorbar_dir.mkdir(parents=True, exist_ok=True)
    colorbar_path = colorbar_dir / f"{name}.png"

    figure = Figure(figsize=(1.3, 4.5), dpi=200)
    # A narrow colorbar axis on the left, leaving room for ticks + label.
    colorbar_axis = figure.add_axes((0.06, 0.05, 0.22, 0.9))
    scalar_mappable = ScalarMappable(
        norm=Normalize(vmin=color_limits[0], vmax=color_limits[1]),
        cmap=colormap_name,
    )
    colorbar = figure.colorbar(scalar_mappable, cax=colorbar_axis)
    if label:
        colorbar.set_label(label)
    figure.savefig(str(colorbar_path), transparent=True, bbox_inches="tight")
    return str(colorbar_path.relative_to(bundle_dir))


# =============================================================================
# PyVista mesh -> plain arrays
# =============================================================================
@dataclass
class MeshFrame:
    """One timestep of a mesh: transposed into plain numpy arrays for Alembic."""

    points_xyz: np.ndarray  # (n_points, 3) float32, physical units
    triangle_vertex_indices: np.ndarray  # (n_triangles, 3) int32
    scalar_values: Optional[np.ndarray]  # (n_points,) float, or None
    rgb_values: Optional[np.ndarray] = None  # (n_points, 3) float, filled later


def pyvista_mesh_to_frame(
    pv_mesh: "pv.DataSet",
    scalar_name: Optional[str],
) -> MeshFrame:
    """
    Extract points, triangles, and (optionally) a scalar from a PyVista mesh.

    The mesh is triangulated first so Alembic receives a uniform triangle
    stream regardless of what polygon types the source produced.

    Args:
        pv_mesh: The mesh returned by ``VarSpec.create_mesh(ds, time)``.
        scalar_name: Name of the point scalar to carry for coloring, or None.

    Returns:
        A :class:`MeshFrame` with plain numpy arrays.
    """
    import warnings

    import pyvista as pv

    # Reduce to a PolyData surface first. Contours/tubes/glyphs are already
    # PolyData, but slices come back as a StructuredGrid, whose triangulate()
    # yields an UnstructuredGrid (no .faces). extract_surface() gives PolyData
    # for any dataset type; then triangulate() makes every face a triangle.
    if isinstance(pv_mesh, pv.PolyData):
        surface = pv_mesh
    else:
        # extract_surface() warns about a future default we aren't affected by
        # (we want the current 'dataset_surface' behavior); silence it.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            surface = pv_mesh.extract_surface()
    triangulated = surface.triangulate()

    points_xyz = np.ascontiguousarray(triangulated.points, dtype=np.float32)

    # PyVista stores faces as a flat [n, i0, i1, ..., n, j0, ...] stream. After
    # triangulation every face has n == 3, so reshape to (n_tri, 4) and drop
    # the leading count column.
    faces_flat = np.asarray(triangulated.faces)
    if faces_flat.size == 0:
        triangle_vertex_indices = np.zeros((0, 3), dtype=np.int32)
    else:
        triangle_vertex_indices = faces_flat.reshape(-1, 4)[:, 1:4].astype(np.int32)

    scalar_values: Optional[np.ndarray] = None
    if scalar_name is not None and scalar_name in triangulated.point_data:
        scalar_values = np.ascontiguousarray(
            triangulated.point_data[scalar_name], dtype=np.float64
        )
    elif triangulated.active_scalars is not None:
        # Fall back to whatever scalar create_mesh left active on the mesh.
        scalar_values = np.ascontiguousarray(
            triangulated.active_scalars, dtype=np.float64
        )

    return MeshFrame(
        points_xyz=points_xyz,
        triangle_vertex_indices=triangle_vertex_indices,
        scalar_values=scalar_values,
    )


# =============================================================================
# Alembic writing (the validated alembic3d path)
# =============================================================================
def _require_alembic():
    """Import alembic3d + imath, or raise a helpful install message."""
    try:
        import imath  # noqa: F401
        import alembic3d  # noqa: F401
    except ImportError as import_error:  # pragma: no cover - env dependent
        raise ImportError(
            "The Blender export backend requires the 'blender' extra:\n"
            "    pip install skyvista[blender]\n"
            "This installs 'alembic3d' (the 3D Alembic bindings). Do NOT install "
            "'alembic' -- that is the unrelated SQLAlchemy database tool."
        ) from import_error
    return alembic3d, imath


def _numpy_points_to_v3f(imath, points_xyz: np.ndarray):
    """Convert (n, 3) float32 -> imath.V3fArray using the fast buffer path."""
    contiguous = np.ascontiguousarray(points_xyz, dtype=np.float32)
    try:
        return imath.V3fArrayFromBuffer(contiguous)
    except Exception:
        # Fallback: element-wise (slower, but always works).
        v3f_array = imath.V3fArray(len(contiguous))
        for i, (px, py, pz) in enumerate(contiguous):
            v3f_array[i] = imath.V3f(float(px), float(py), float(pz))
        return v3f_array


def _numpy_ints_to_intarray(imath, values: np.ndarray):
    """Convert a 1-D int array -> imath.IntArray using the fast buffer path."""
    contiguous = np.ascontiguousarray(values, dtype=np.int32)
    try:
        return imath.IntArrayFromBuffer(contiguous)
    except Exception:
        int_array = imath.IntArray(len(contiguous))
        for i, value in enumerate(contiguous):
            int_array[i] = int(value)
        return int_array


def _numpy_rgb_to_c3f(imath, rgb_values: np.ndarray):
    """Convert (n, 3) RGB -> imath.C3fArray (element-wise; no buffer ctor exists)."""
    color_array = imath.C3fArray(len(rgb_values))
    for i, (r, g, b) in enumerate(rgb_values):
        color_array[i] = imath.Color3f(float(r), float(g), float(b))
    return color_array


def write_mesh_sequence_alembic(
    output_path: PathLike,
    frames: List[MeshFrame],
    fps: float,
    object_name: str,
) -> None:
    """
    Write a sequence of mesh frames (with changing topology) to one ``.abc``.

    Each frame may have a different vertex/face count; Alembic records this as
    heterogeneous topology, which tells Blender to swap meshes per frame rather
    than interpolate vertex positions.

    Args:
        output_path: Destination ``.abc`` path.
        frames: Per-timestep meshes; each frame's ``rgb_values`` (if present) is
            written as a per-vertex C3f color attribute named "color".
        fps: Frame rate; Alembic samples are spaced 1/fps seconds apart.
        object_name: Name of the poly-mesh object inside the archive.
    """
    alembic3d, imath = _require_alembic()
    from alembic3d.Abc import OArchive
    from alembic3d.AbcCoreAbstract import TimeSampling
    from alembic3d.AbcGeom import (
        GeometryScope,
        OC3fGeomParam,
        OC3fGeomParamSample,
        OPolyMesh,
        OPolyMeshSchemaSample,
    )

    archive = OArchive(str(output_path))

    # Uniform time sampling: one sample every 1/fps seconds, starting at t=0.
    time_sampling = TimeSampling(1.0 / fps, 0.0)
    time_sampling_index = archive.addTimeSampling(time_sampling)

    poly_mesh = OPolyMesh(archive.getTop(), object_name, time_sampling_index)
    mesh_schema = poly_mesh.getSchema()

    # Create the color param once (if any frame carries colors), then set it
    # every frame to match that frame's vertex count.
    any_frame_has_color = any(frame.rgb_values is not None for frame in frames)
    color_param = None
    if any_frame_has_color:
        color_param = OC3fGeomParam(
            mesh_schema.getArbGeomParams(),
            VERTEX_COLOR_ATTRIBUTE_NAME,
            False,  # not indexed
            GeometryScope.kVertexScope,
            1,  # extent
            time_sampling_index,
        )

    for frame in frames:
        imath_points = _numpy_points_to_v3f(imath, frame.points_xyz)

        # Flatten triangle indices and build the per-face vertex-count stream
        # (all 3, since we triangulated).
        flat_indices = frame.triangle_vertex_indices.ravel()
        imath_face_indices = _numpy_ints_to_intarray(imath, flat_indices)
        n_triangles = len(frame.triangle_vertex_indices)
        imath_face_counts = _numpy_ints_to_intarray(
            imath, np.full(n_triangles, 3, dtype=np.int32)
        )

        mesh_schema.set(
            OPolyMeshSchemaSample(
                imath_points, imath_face_indices, imath_face_counts
            )
        )

        if color_param is not None:
            # If this frame lacks colors, fall back to mid-gray to keep the
            # attribute's vertex count aligned with the geometry.
            if frame.rgb_values is not None:
                rgb = frame.rgb_values
            else:
                rgb = np.full((len(frame.points_xyz), 3), 0.5, dtype=np.float32)
            color_param.set(
                OC3fGeomParamSample(
                    _numpy_rgb_to_c3f(imath, rgb), GeometryScope.kVertexScope
                )
            )

    # Finalise the archive by dropping all references to it.
    del mesh_schema, poly_mesh, color_param, archive


# =============================================================================
# Bounds / transform helpers
# =============================================================================
def _accumulate_bounds(
    running_bounds: Optional[np.ndarray], points_xyz: np.ndarray
) -> np.ndarray:
    """Update a running [[xmin,ymin,zmin],[xmax,ymax,zmax]] with new points."""
    if len(points_xyz) == 0:
        return running_bounds
    frame_min = points_xyz.min(axis=0)
    frame_max = points_xyz.max(axis=0)
    if running_bounds is None:
        return np.vstack([frame_min, frame_max])
    running_bounds[0] = np.minimum(running_bounds[0], frame_min)
    running_bounds[1] = np.maximum(running_bounds[1], frame_max)
    return running_bounds


# =============================================================================
# Per-spec export
# =============================================================================
def _coloring_scalar_name(spec: "VarSpec") -> Optional[str]:
    """
    Which point scalar (if any) a mesh-type spec colors by.

    Differs per spec type because the geometry classes name their scalar
    differently. Returning None means "no explicit scalar" -- the exporter
    then falls back to whatever scalar ``create_mesh`` left active, or to a
    solid color when an appearance color is set.
    """
    from .varspec import ContourSpec, SliceSpec, TrajectorySpec, VectorSpec

    geometry = spec.geometry
    if isinstance(spec, ContourSpec):
        # geometry.scalar overrides varname; else color by the contoured var.
        return geometry.scalar or geometry.varname
    if isinstance(spec, SliceSpec):
        return geometry.varname
    if isinstance(spec, TrajectorySpec):
        return geometry.scalar  # may be None -> solid color
    if isinstance(spec, VectorSpec):
        return geometry.scale_by  # may be None -> fall back to active scalar
    return getattr(geometry, "scalar", None) or getattr(geometry, "varname", None)


def _build_object_entry(
    spec: "VarSpec",
    dataset,
    times: List[Any],
    data_subdir: Path,
    fps: float,
    running_bounds: Optional[np.ndarray],
) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    """
    Dispatch one spec to the appropriate carrier and return its manifest object
    entry plus the updated running spatial bounds.

    Surface-mesh specs (contour, slice, vector glyphs, trajectory tubes) go to
    Alembic; volume specs go to a VDB sequence.
    """
    from .varspec import VolumeSpec

    if isinstance(spec, VolumeSpec):
        return _build_volume_entry(spec, dataset, times, data_subdir, running_bounds)
    return _build_mesh_entry(
        spec, dataset, times, data_subdir, fps, running_bounds
    )


def _build_mesh_entry(
    spec: "VarSpec",
    dataset,
    times: List[Any],
    data_subdir: Path,
    fps: float,
    running_bounds: Optional[np.ndarray],
) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    """
    Export a surface-mesh spec to an Alembic sequence (one frame per time),
    baking scalar coloring into per-vertex colors, and return its manifest
    object entry plus the updated running spatial bounds.
    """
    scalar_name = _coloring_scalar_name(spec)
    appearance = spec.appearance

    # ---- Pass 1: build every frame's geometry + gather the global scalar range
    frames: List[MeshFrame] = []
    global_scalar_min = np.inf
    global_scalar_max = -np.inf
    for time in times:
        pv_mesh = spec.create_mesh(dataset, time)
        if pv_mesh is None or len(pv_mesh.points) == 0:
            # Keep an empty frame so frame indices stay aligned with time.
            frames.append(
                MeshFrame(
                    points_xyz=np.zeros((0, 3), dtype=np.float32),
                    triangle_vertex_indices=np.zeros((0, 3), dtype=np.int32),
                    scalar_values=None,
                )
            )
            continue
        frame = pyvista_mesh_to_frame(pv_mesh, scalar_name)
        frames.append(frame)
        running_bounds = _accumulate_bounds(running_bounds, frame.points_xyz)
        if frame.scalar_values is not None and len(frame.scalar_values) > 0:
            global_scalar_min = min(global_scalar_min, float(frame.scalar_values.min()))
            global_scalar_max = max(global_scalar_max, float(frame.scalar_values.max()))

    # ---- Decide coloring: solid color, or baked colormap from the scalar
    uses_scalar_coloring = (
        appearance.color is None
        and np.isfinite(global_scalar_min)
        and global_scalar_max > global_scalar_min
    )

    color_limits: Optional[Tuple[float, float]] = None
    colormap_name: Optional[str] = None
    if uses_scalar_coloring:
        color_limits = appearance.clim or (global_scalar_min, global_scalar_max)
        colormap_name = appearance.cmap or DEFAULT_COLORMAP_NAME
        # ---- Pass 2: bake per-vertex colors with the fixed global range
        for frame in frames:
            if frame.scalar_values is not None and len(frame.scalar_values) > 0:
                frame.rgb_values = bake_scalar_to_rgb(
                    frame.scalar_values, colormap_name, color_limits
                )

    # ---- Write the Alembic sequence
    object_subdir = data_subdir / spec.name
    object_subdir.mkdir(parents=True, exist_ok=True)
    alembic_path = object_subdir / "sequence.abc"
    write_mesh_sequence_alembic(
        alembic_path, frames, fps=fps, object_name=spec.name
    )

    # ---- Build the manifest material
    material = appearance.to_blender_material()
    if uses_scalar_coloring:
        # Label the colorbar by the scalar actually mapped to color. make_contour
        # auto-fills scalar_bar_title with the *contour* varname, which would
        # mislabel a surface colored by a different scalar; only honor a title
        # the user deliberately set (one that differs from the varname).
        contour_varname = getattr(spec.geometry, "varname", None)
        deliberate_title = (
            appearance.scalar_bar_title
            if appearance.scalar_bar_title not in (None, contour_varname)
            else None
        )
        colorbar_label = deliberate_title or scalar_name or ""
        material["coloring"] = {
            "mode": "vertex_color",
            "attribute": VERTEX_COLOR_ATTRIBUTE_NAME,
            "cmap": colormap_name,
            "clim": list(color_limits),
            "label": colorbar_label,
        }
    else:
        material["coloring"] = {
            "mode": "solid",
            "color": appearance.color or "#cccccc",
        }

    # Label the topology honestly: "heterogeneous" only if vertex/face counts
    # actually vary across frames (they usually do for isosurfaces, but not
    # always -- a rigidly translating feature can keep a constant count).
    distinct_frame_shapes = {
        (len(frame.points_xyz), len(frame.triangle_vertex_indices))
        for frame in frames
    }
    topology_label = (
        "heterogeneous" if len(distinct_frame_shapes) > 1 else "homogeneous"
    )

    object_entry = {
        "name": spec.name,
        "spec_type": type(spec).__name__.replace("Spec", "").lower(),
        "geometry": {
            "carrier": "alembic",
            "path": str(alembic_path.relative_to(data_subdir.parent)),
            "object_path": f"/{spec.name}",
            "topology": topology_label,
        },
        "material": material,
    }
    return object_entry, running_bounds


# =============================================================================
# Volume export (VDB)
# =============================================================================
def _require_openvdb():
    """
    Import the OpenVDB Python bindings, or raise a helpful message.

    Unlike the mesh carrier (``alembic3d`` on PyPI), OpenVDB has no usable
    Python wheel for recent CPython -- the only PyPI build is CPython 3.7.
    The practical source is conda-forge's ``openvdb`` package (natural for a
    pixi/conda environment). The import name varies by build, so try both.
    """
    try:
        import openvdb  # conda-forge exposes the module as 'openvdb'

        return openvdb
    except ImportError:
        pass
    try:
        import pyopenvdb  # some builds expose it as 'pyopenvdb'

        return pyopenvdb
    except ImportError as import_error:
        raise ImportError(
            "Volume (VDB) export requires the OpenVDB Python bindings, which "
            "have no PyPI wheel for recent Python. Install via conda-forge, "
            "e.g. in pixi add 'openvdb' to a 'blender' feature, or:\n"
            "    conda install -c conda-forge openvdb\n"
            "Mesh export (Alembic) does not need this."
        ) from import_error


def _axis_voxel_size(coordinate_values: np.ndarray) -> float:
    """
    Uniform voxel size for one axis (VDB grids are regular voxel lattices).

    Uses the mean spacing and warns if the coordinate is not evenly spaced,
    since a single voxel size cannot faithfully represent a stretched grid.
    """
    coordinate_values = np.asarray(coordinate_values, dtype=float)
    spacings = np.diff(coordinate_values)
    mean_spacing = float(np.mean(spacings))
    if spacings.size > 1 and np.ptp(spacings) > 1e-6 * abs(mean_spacing):
        import warnings

        warnings.warn(
            "VDB volume export assumes a uniform voxel size per axis, but a "
            "coordinate is not evenly spaced; using the mean spacing. A "
            "stretched vertical grid will be slightly distorted -- resample to "
            "a uniform grid for full fidelity.",
            stacklevel=2,
        )
    return mean_spacing


def write_volume_sequence_vdb(
    output_dir: Path,
    frame_arrays: List[np.ndarray],
    grid_name: str,
    transform_matrix: List[List[float]],
) -> List[Path]:
    """
    Write a sequence of dense scalar volumes to numbered ``.vdb`` files.

    The numbering (``name_0001.vdb`` ...) is what Blender auto-detects as an
    animated volume sequence. Each grid carries the same index->world linear
    transform so the volume sits in physical coordinates, consistent with the
    Alembic mesh objects.

    Args:
        output_dir: Directory to write the numbered files into.
        frame_arrays: Per-timestep dense (nx, ny, nz) float arrays.
        grid_name: Name of the grid inside each file (referenced by the shader).
        transform_matrix: 4x4 index->world matrix (OpenVDB row-vector
            convention: translation in the last row).

    Returns:
        The list of written file paths.
    """
    openvdb = _require_openvdb()
    written_paths: List[Path] = []
    for frame_index, dense_array in enumerate(frame_arrays):
        grid = openvdb.FloatGrid()
        grid.copyFromArray(np.ascontiguousarray(dense_array, dtype=np.float32))
        grid.name = grid_name
        grid.transform = openvdb.createLinearTransform(matrix=transform_matrix)
        file_path = output_dir / f"{grid_name}_{frame_index + 1:04d}.vdb"
        openvdb.write(str(file_path), grids=[grid])
        written_paths.append(file_path)
    return written_paths


def _build_volume_entry(
    spec: "VarSpec",
    dataset,
    times: List[Any],
    data_subdir: Path,
    running_bounds: Optional[np.ndarray],
) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    """
    Export a volume spec to a VDB sequence and return its manifest object entry.

    Requires a rectilinear grid (1-D x/y/z coordinates); VDB is a regular voxel
    lattice, so curvilinear/geographic/spherical grids would need resampling
    first and are rejected here.
    """
    from .grid_utils import select_time
    from .grids import resolve_coordinates

    varname = spec.geometry.varname
    appearance = spec.appearance

    # ---- Require a rectilinear grid (1-D coordinate axes)
    coordinate_names = resolve_coordinates(dataset, ["x", "y", "z"])
    x_name = coordinate_names["x"]
    y_name = coordinate_names["y"]
    z_name = coordinate_names["z"]
    for coordinate_name in (x_name, y_name, z_name):
        if coordinate_name not in dataset.coords or dataset[coordinate_name].ndim != 1:
            raise NotImplementedError(
                "VDB volume export requires a rectilinear grid with 1-D x/y/z "
                f"coordinates; '{coordinate_name}' is missing or multi-dimensional. "
                "Resample to a uniform grid first."
            )

    x_coordinates = dataset[x_name].values
    y_coordinates = dataset[y_name].values
    z_coordinates = dataset[z_name].values

    # ---- Build the index->world transform (voxel sizes + origin)
    voxel_size_x = _axis_voxel_size(x_coordinates)
    voxel_size_y = _axis_voxel_size(y_coordinates)
    voxel_size_z = _axis_voxel_size(z_coordinates)
    origin_x = float(x_coordinates[0])
    origin_y = float(y_coordinates[0])
    origin_z = float(z_coordinates[0])
    # OpenVDB row-vector convention: world = index_homogeneous @ M, so the
    # per-axis scales sit on the diagonal and the origin is the last ROW.
    transform_matrix = [
        [voxel_size_x, 0.0, 0.0, 0.0],
        [0.0, voxel_size_y, 0.0, 0.0],
        [0.0, 0.0, voxel_size_z, 0.0],
        [origin_x, origin_y, origin_z, 1.0],
    ]

    # ---- Per-time dense arrays (nx, ny, nz), thresholded, NaN-cleaned
    frame_arrays: List[np.ndarray] = []
    global_min = np.inf
    global_max = -np.inf
    for time in times:
        dataset_at_time = select_time(dataset, time)
        # Transpose to a consistent (x, y, z) index order for copyFromArray.
        data_array = dataset_at_time[varname].transpose(x_name, y_name, z_name)
        dense_array = np.asarray(data_array.values, dtype=np.float32)

        # Apply the spec's threshold by zeroing out-of-range voxels (0 is empty
        # space to the volume renderer).
        if spec.geometry.threshold:
            low_threshold, high_threshold = spec.geometry.threshold
            if low_threshold is not None:
                dense_array = np.where(dense_array < low_threshold, 0.0, dense_array)
            if high_threshold is not None:
                dense_array = np.where(dense_array > high_threshold, 0.0, dense_array)
        dense_array = np.nan_to_num(dense_array, nan=0.0)
        frame_arrays.append(dense_array)

        finite_values = dense_array[np.isfinite(dense_array)]
        if finite_values.size:
            global_min = min(global_min, float(finite_values.min()))
            global_max = max(global_max, float(finite_values.max()))

    # ---- Fold the volume's spatial extent into the running scene bounds
    volume_corner_points = np.array(
        [
            [x_coordinates.min(), y_coordinates.min(), z_coordinates.min()],
            [x_coordinates.max(), y_coordinates.max(), z_coordinates.max()],
        ],
        dtype=float,
    )
    running_bounds = _accumulate_bounds(running_bounds, volume_corner_points)

    # ---- Write the numbered VDB sequence
    object_subdir = data_subdir / spec.name
    object_subdir.mkdir(parents=True, exist_ok=True)
    write_volume_sequence_vdb(
        object_subdir, frame_arrays, grid_name=varname, transform_matrix=transform_matrix
    )
    # Blender picks the sequence up from the '####' numbered pattern.
    path_pattern = f"data/{spec.name}/{varname}_####.vdb"

    # ---- Material: a volume shader driven by the scalar through a color ramp
    color_limits = appearance.clim or (
        (global_min, global_max) if np.isfinite(global_min) else (0.0, 1.0)
    )
    colormap_name = appearance.cmap or DEFAULT_COLORMAP_NAME
    # Ship the colormap stops so the build script can rebuild the ramp without
    # matplotlib (Blender's Python lacks it).
    colormap_lut_path = write_colormap_lut(data_subdir.parent, colormap_name)
    material = {
        "type": "volume",
        "coloring": {
            "mode": "scalar_ramp",
            "attribute": varname,
            "cmap": colormap_name,
            "clim": list(color_limits),
            "colormap_lut": colormap_lut_path,
            "label": appearance.scalar_bar_title or varname,
        },
        # Density is driven by the same grid, normalised through clim on the
        # Blender side; the build script/user tunes the overall strength.
        "density": {"grid": varname, "clim": list(color_limits), "scale": 1.0},
    }

    object_entry = {
        "name": spec.name,
        "spec_type": "volume",
        "geometry": {
            "carrier": "vdb_sequence",
            "path_pattern": path_pattern,
            "grid_name": varname,
        },
        "material": material,
    }
    return object_entry, running_bounds


# =============================================================================
# Camera
# =============================================================================
def _follow_positions(
    scene: "Scene",
    render_times: List[Any],
    position_vars: Tuple[str, str],
) -> Optional[List[Tuple[float, float]]]:
    """
    Read a tracked feature's (x, y) position at each render time.

    Looks across the scene's datasets for the first one carrying both position
    variables (the storm-tracking convention). Returns one (x, y) per time, or
    None if no dataset has them.
    """
    x_var, y_var = position_vars
    for dataset, _ in scene._specs:
        if x_var in dataset and y_var in dataset:
            track: List[Tuple[float, float]] = []
            for time in render_times:
                if "time" in dataset.dims and time is not None:
                    dataset_at_time = dataset.sel(time=time)
                else:
                    dataset_at_time = dataset
                x_value = float(np.ravel(dataset_at_time[x_var].values)[0])
                y_value = float(np.ravel(dataset_at_time[y_var].values)[0])
                track.append((x_value, y_value))
            return track
    return None


def _compute_camera_manifest(
    camera_config: BlenderCameraConfig,
    running_bounds: Optional[np.ndarray],
    render_times: List[Any],
    frame_start: int,
    scene: "Scene",
) -> Optional[Dict[str, Any]]:
    """
    Build the manifest ``camera`` block in data (physical) space.

    Supports a fixed 3/4 "static" view, an "orbit" that circles the scene over
    the animation, and "follow" that tracks a moving feature at a constant
    offset. Keyframes are in data units; the build script applies the scene
    transform to them so the camera stays registered to the geometry.
    """
    import math
    import warnings

    if running_bounds is None:
        return None

    bounds_min = running_bounds[0]
    bounds_max = running_bounds[1]
    center = (bounds_min + bounds_max) / 2.0
    diagonal = float(np.linalg.norm(bounds_max - bounds_min)) or 1.0
    distance = diagonal * camera_config.distance_factor

    # Unit view direction from azimuth/elevation, then the camera offset vector.
    azimuth = math.radians(camera_config.azimuth_deg)
    elevation = math.radians(camera_config.elevation_deg)
    offset = np.array(
        [
            distance * math.cos(elevation) * math.cos(azimuth),
            distance * math.cos(elevation) * math.sin(azimuth),
            distance * math.sin(elevation),
        ]
    )

    n_times = len(render_times)
    frame_end = frame_start + max(n_times - 1, 0)
    up_vector = [0.0, 0.0, 1.0]
    center_point = [float(c) for c in center]
    keyframes: List[Dict[str, Any]] = []

    mode = camera_config.mode
    if mode == "follow":
        track = _follow_positions(
            scene, render_times, camera_config.follow_position_vars
        )
        if track is None:
            warnings.warn(
                "Camera mode 'follow' requested but no dataset has "
                f"{camera_config.follow_position_vars}; using a static camera.",
                stacklevel=2,
            )
            mode = "static"
        else:
            # Hold a constant offset from the tracked feature (at the data's
            # vertical center), so the feature stays framed as it moves.
            for time_index, (feature_x, feature_y) in enumerate(track):
                look_at = [feature_x, feature_y, float(center[2])]
                location = [look_at[i] + float(offset[i]) for i in range(3)]
                keyframes.append(
                    {
                        "frame": frame_start + time_index,
                        "location": location,
                        "look_at": look_at,
                        "up": up_vector,
                    }
                )

    if mode == "orbit" and n_times > 1:
        n_keyframes = max(2, camera_config.n_orbit_keyframes)
        for keyframe_index in range(n_keyframes):
            fraction = keyframe_index / (n_keyframes - 1)
            frame = int(round(frame_start + fraction * (frame_end - frame_start)))
            orbit_azimuth = (
                azimuth + 2 * math.pi * camera_config.orbit_revolutions * fraction
            )
            location = [
                float(center[0] + distance * math.cos(elevation) * math.cos(orbit_azimuth)),
                float(center[1] + distance * math.cos(elevation) * math.sin(orbit_azimuth)),
                float(center[2] + distance * math.sin(elevation)),
            ]
            keyframes.append(
                {
                    "frame": frame,
                    "location": location,
                    "look_at": center_point,
                    "up": up_vector,
                }
            )

    if not keyframes:
        # Static (also the fallback when follow/orbit produced nothing).
        location = [float(center[i] + offset[i]) for i in range(3)]
        keyframes = [
            {
                "frame": frame_start,
                "location": location,
                "look_at": center_point,
                "up": up_vector,
            }
        ]

    return {
        "type": "perspective",
        "lens_mm": camera_config.lens_mm,
        "keyframes": keyframes,
    }


# =============================================================================
# Orchestration
# =============================================================================
def export_scene_to_blender(
    scene: "Scene",
    path: PathLike,
    config: Optional[BlenderExportConfig] = None,
    times: Optional[List[Any]] = None,
    build: bool = False,
    blender_executable: str = "blender",
) -> Path:
    """
    Export a whole Scene to a self-contained Blender bundle directory.

    Writes ``scene.json`` (the manifest), ``provenance.json``, per-object
    geometry caches under ``data/``, and a copy of the ``build_scene.py`` build
    script so the bundle is runnable on its own.

    Args:
        scene: The Scene to export.
        path: Destination bundle directory (created if needed).
        config: Export configuration; a default is used when None.
        times: Times to render; defaults to the union of all dataset times.
        build: If True, invoke Blender headlessly to build the ``.blend`` from
            the bundle (requires ``blender_executable`` on PATH).
        blender_executable: Blender command used when ``build`` is True.

    Returns:
        The bundle directory path.
    """
    config = config or BlenderExportConfig()
    bundle_dir = Path(path)
    data_dir = bundle_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    render_times = times if times is not None else scene._get_all_times()
    n_frames = len(render_times)

    object_entries: List[Dict[str, Any]] = []
    running_bounds: Optional[np.ndarray] = None
    for dataset, spec in scene._specs:
        object_entry, running_bounds = _build_object_entry(
            spec, dataset, render_times, data_dir, config.fps, running_bounds
        )
        object_entries.append(object_entry)

    # ---- Resolve the transform's origin_shift (default: center of data bounds)
    transform = config.transform
    if transform.origin_shift is None and running_bounds is not None:
        center = (running_bounds[0] + running_bounds[1]) / 2.0
        resolved_origin_shift: Optional[Tuple[float, float, float]] = tuple(
            float(c) for c in center
        )
    else:
        resolved_origin_shift = transform.origin_shift

    # ---- Render a colorbar PNG for every scalar-colored object
    colorbar_annotations: List[Dict[str, Any]] = []
    for object_entry in object_entries:
        coloring = object_entry["material"].get("coloring", {})
        if coloring.get("cmap") and coloring.get("clim"):
            colorbar_image = write_colorbar_png(
                bundle_dir,
                object_entry["name"],
                coloring["cmap"],
                tuple(coloring["clim"]),
                coloring.get("label", ""),
            )
            colorbar_annotations.append(
                {
                    "for": object_entry["name"],
                    "cmap": coloring["cmap"],
                    "clim": coloring["clim"],
                    "label": coloring.get("label", ""),
                    "image": colorbar_image,
                }
            )

    # ---- Assemble the manifest
    manifest = {
        "skyvista_manifest_version": SKYVISTA_MANIFEST_VERSION,
        "generated_by": {
            "skyvista_version": _skyvista_version(),
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
        "transform": {
            "origin_shift": list(resolved_origin_shift)
            if resolved_origin_shift is not None
            else [0.0, 0.0, 0.0],
            "scale": transform.scale,
            "z_exaggeration": transform.z_exaggeration,
        },
        "time": {
            "fps": config.fps,
            "frame_start": config.frame_start,
            "frame_end": config.frame_start + max(n_frames - 1, 0),
            "data_times": [_json_safe_time(t) for t in render_times],
        },
        "render": config.render.to_dict(),
        "world": {"preset": config.world_preset},
        "camera": _compute_camera_manifest(
            config.camera, running_bounds, render_times, config.frame_start, scene
        ),
        "objects": object_entries,
        "annotations": {
            "title": scene.title,
            "show_grid": scene.show_grid,
            "background": scene.background,
            "colorbars": colorbar_annotations,
        },
    }

    manifest_path = bundle_dir / "scene.json"
    with open(manifest_path, "w") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)

    # ---- Minimal provenance record
    provenance = {
        "skyvista_version": _skyvista_version(),
        "n_specs": len(scene._specs),
        "n_frames": n_frames,
    }
    with open(bundle_dir / "provenance.json", "w") as provenance_file:
        json.dump(provenance, provenance_file, indent=2)

    # ---- Copy the build script so the bundle is self-contained and runnable
    build_script_path = _copy_build_script(bundle_dir)

    # ---- Optionally invoke Blender to assemble the .blend right now
    if build:
        _launch_blender_build(blender_executable, build_script_path, bundle_dir)

    return bundle_dir


def _copy_build_script(bundle_dir: Path) -> Path:
    """Copy the packaged Blender build script into the bundle."""
    import shutil

    source = Path(__file__).parent / "blender_assets" / "build_scene.py"
    destination = bundle_dir / "build_scene.py"
    shutil.copyfile(source, destination)
    return destination


def _launch_blender_build(
    blender_executable: str, build_script_path: Path, bundle_dir: Path
) -> None:
    """Run ``blender --background --python build_scene.py -- <bundle>`` headlessly."""
    import subprocess

    command = [
        blender_executable,
        "--background",
        "--python",
        str(build_script_path),
        "--",
        str(bundle_dir),
    ]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def _skyvista_version() -> str:
    try:
        from . import __version__

        return __version__
    except Exception:
        return "unknown"


def _json_safe_time(time: Any) -> Any:
    """Render a time value as something JSON can serialise."""
    if time is None:
        return None
    if isinstance(time, (np.datetime64,)):
        return str(time)
    if isinstance(time, dt.datetime):
        return time.isoformat()
    if isinstance(time, (np.integer, np.floating)):
        return time.item()
    return str(time)
