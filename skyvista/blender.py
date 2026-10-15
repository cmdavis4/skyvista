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
* Time-varying *meshes* (contours, slices, and glyph or trajectory meshes
  baked with ``blender_instancing=False``) are
  written as **Alembic** (``.abc``) with per-frame changing topology, via the
  ``alembic3d`` bindings. Blender ingests these through a Mesh Sequence Cache
  modifier, which supports the changing vertex/face counts that isosurfaces
  produce every timestep.
* Volumes are written as **VDB** sequences: one scalar grid per frame for a
  plain volume, or one extinction grid per hydrometeor species for a
  physically rendered cloud (see :func:`_build_cloud_entry` and
  :mod:`skyvista.cloud_optics`).

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

The extra also installs ``bpy``, Blender built as a Python module, so building
and rendering need no separate Blender install. skyvista still never imports
``bpy`` itself: it runs the Blender scripts in a fresh interpreter (see
:class:`BlenderRunner`). A standalone Blender program can be used instead via
``blender_executable=`` or ``$SKYVISTA_BLENDER``.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from carlee_tools import PathLike

if TYPE_CHECKING:
    import pyvista as pv
    from matplotlib.colors import Colormap

    from .scene import Scene
    from .varspec import VarSpec


# The manifest schema version. Bump when the on-disk contract changes in a way
# the Blender-side reader must care about.
SKYVISTA_MANIFEST_VERSION = "0.1"

# Blender's Alembic importer only reads recognised typed params (colors, UVs,
# velocities). We carry the baked colormap under this name; it becomes a
# Blender Color Attribute of the same name.
#
# NOT "color": that name collides with a reserved/implicit attribute in Cycles,
# which then renders the mesh uncoloured (grey) while EEVEE reads it fine. A
# distinct name ("col_data") is read correctly by both engines.
VERTEX_COLOR_ATTRIBUTE_NAME = "col_data"

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
            None, defaults to the horizontal center of the merged data bounds
            for x/y (so the figure sits centered on the world origin, nice for
            orbiting and camera framing) but the data's *floor* (z_min) for z,
            so the figure rests on the ground plane (Blender z=0) instead of
            straddling it. Pass (0, 0, 0) to keep absolute data positions.
        z_exaggeration: Extra multiplier applied to the vertical axis only.
            1.0 = physically faithful; >1 exaggerates relief. Always recorded.
    """

    scale: float = 1.0e-3
    origin_shift: Optional[Tuple[float, float, float]] = None
    z_exaggeration: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin_shift": (
                list(self.origin_shift) if self.origin_shift is not None else None
            ),
            "scale": self.scale,
            "z_exaggeration": self.z_exaggeration,
        }


# Named render qualities: (samples per pixel, (width, height) in pixels).
# "preview" renders a mature storm in seconds to a few minutes and is meant
# for choosing a camera, sun and colors. "final" is the default render setting.
RENDER_QUALITY_PRESETS: Dict[str, Tuple[int, Tuple[int, int]]] = {
    "preview": (16, (960, 540)),
    "final": (128, (1920, 1080)),
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
        volume_bounces: Maximum number of volume scattering events per light
            path. None means 128 when the scene contains clouds (enough for
            optically thick convection to converge to within ~1%) and
            Blender's default otherwise. Blender's own default is 0, i.e.
            single scattering only, which renders clouds far too dark.
        exposure: Film exposure in EV (stops). None means automatic metering
            under the "physical_sky" world (a white Lambertian surface in full
            sun renders at 80% of display white), and 0 otherwise.
        denoise: Run Cycles' denoiser. It also removes the rare negative
            pixels produced by Cycles' unbiased volume transmittance estimator
            in thick, sunlit clouds.
    """

    engine: str = "CYCLES"
    samples: int = 128
    resolution: Tuple[int, int] = (1920, 1080)
    view_transform: str = "Standard"
    film_transparent: bool = False
    volume_bounces: Optional[int] = None
    exposure: Optional[float] = None
    denoise: bool = True

    def set_quality(self, quality: str) -> "BlenderRenderConfig":
        """
        Set the samples and resolution from a named preset.

        The presets are listed in ``RENDER_QUALITY_PRESETS``: "preview" (16
        samples, 960 x 540) and "final" (128 samples, 1920 x 1080). Set
        ``resolution`` afterwards to keep a preset's samples at another size.

        Returns:
            This config, changed in place, for chaining.
        """
        if quality not in RENDER_QUALITY_PRESETS:
            raise ValueError(
                f"Unknown render quality {quality!r}; choose one of "
                f"{sorted(RENDER_QUALITY_PRESETS)}."
            )
        self.samples, self.resolution = RENDER_QUALITY_PRESETS[quality]
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "samples": self.samples,
            "resolution": list(self.resolution),
            "view_transform": self.view_transform,
            "film_transparent": self.film_transparent,
            "volume_bounces": self.volume_bounces,
            "exposure": self.exposure,
            "denoise": self.denoise,
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
        location: Explicit camera position in data (physical) coordinates.
            Used together with ``look_at`` for a "static" camera, in place of
            the automatic placement from the data bounds. Setting both pins
            the view, so different exports of the same storm (e.g. the panels
            of a side-by-side figure) are framed identically even when their
            objects' bounds differ.
        look_at: The data-space point an explicit camera aims at.
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
    location: Optional[Tuple[float, float, float]] = None
    look_at: Optional[Tuple[float, float, float]] = None


@dataclass(frozen=True)
class BlenderView:
    """
    A fixed camera and scene origin, shared by several exports.

    A side-by-side figure needs every panel framed the same way. The automatic
    camera cannot do that by itself, because it frames each export's own data
    bounds, and a panel with a thinned cloud plus arrows has other bounds than
    a plain cloud. The fix is to take the camera of one export and give it to
    the others. ``BlenderExportConfig.use_view`` applies a view to a config.

    The origin shift is part of the view: it is the data point that Blender
    puts at its origin, and the camera position is stored in data units, but
    the ground plane and the haze box are placed relative to the origin.

    Get a view in one of two ways:

    - :meth:`from_bundle` reads it from an export that already exists.
    - :meth:`skyvista.Scene.blender_view` computes the automatic camera of a
      scene, without rendering it.

    Attributes:
        location: Camera position in data (physical) coordinates.
        look_at: The data point the camera aims at.
        lens_mm: Camera focal length in millimeters.
        origin_shift: Data point mapped to the Blender origin.
    """

    location: Tuple[float, float, float]
    look_at: Tuple[float, float, float]
    lens_mm: float
    origin_shift: Tuple[float, float, float]

    @classmethod
    def from_bundle(cls, bundle_dir: PathLike) -> "BlenderView":
        """
        Read the camera and origin shift of an exported bundle.

        Only a static camera has a single view, so a bundle whose camera
        moves (several keyframes) is refused.
        """
        manifest = json.loads((Path(bundle_dir) / "scene.json").read_text())
        camera = manifest["camera"]
        if camera is None:
            raise ValueError(f"The bundle {bundle_dir} has no camera.")
        keyframes = camera["keyframes"]
        if len(keyframes) != 1:
            raise ValueError(
                f"The bundle {bundle_dir} has a moving camera with "
                f"{len(keyframes)} keyframes; a view needs a static camera."
            )
        (keyframe,) = keyframes
        return cls(
            location=tuple(float(value) for value in keyframe["location"]),
            look_at=tuple(float(value) for value in keyframe["look_at"]),
            lens_mm=float(camera["lens_mm"]),
            origin_shift=tuple(
                float(value) for value in manifest["transform"]["origin_shift"]
            ),
        )


# Named world/lighting looks the build script knows how to construct. Kept here
# (rather than only on the Blender side) so callers get validated names and so
# the set is documented in one place.
WORLD_PRESETS = ("sky", "physical_sky", "studio", "dark", "white")


@dataclass
class BlenderWorldConfig:
    """
    World background and key-light configuration.

    A "preset" picks the overall look; the sun_* / background_strength knobs
    then tune it. Every preset except "physical_sky" places a single key Sun
    lamp aimed from (sun_elevation_deg, sun_azimuth_deg) so shadows are
    consistent regardless of the background. The "sky" preset additionally
    points Blender's Nishita physical-sky sun in that same direction, and
    "physical_sky" uses that sky sun as the only sun.

    Attributes:
        preset: One of :data:`WORLD_PRESETS`:
            - "sky": Nishita physical daytime sky, with the key sun aligned to
              the sky's sun. Best default for outdoor atmospheric scenes.
            - "physical_sky": the same physical sky, but radiometrically
              faithful: the sun is the sky texture's own sun disc (so the
              direct/diffuse ratio is the model's, ~84/16 at 35 deg), no
              separate lamp, no damping of sky light, and automatic exposure.
              The preset for physically rendered clouds (``add_clouds``).
              ``sun_strength`` is ignored; ``background_strength`` scales sun
              and sky together.
            - "studio": neutral mid-gray environment + soft key sun. Isolates the
              figure with no distracting background (product-shot look).
            - "dark": near-black background + sun. Flatters glowing volumes and
              emissive isosurfaces.
            - "white": pure white background + sun. Clean look for print figures.
        sun_elevation_deg: Sun height above the horizon in degrees (0 = on the
            horizon, 90 = straight overhead). Drives both the key Sun lamp and,
            for the "sky" preset, the Nishita sun.
        sun_azimuth_deg: Direction of the sun in degrees, measured CCW from +x
            about +z (0 = sun toward +x, so light travels toward -x; 90 = sun
            toward +y).
        sun_strength: Irradiance of the key Sun lamp (Blender "Strength"/energy).
        background_strength: Brightness of the *camera-visible* background (the
            sky or backdrop). The background's contribution to *lighting* is
            damped separately per preset so a bright sky/backdrop can't wash the
            baked scientific colors to white; see the build script's setup_world.
        ground_albedo: Albedo of a matte, spectrally neutral ground plane, or
            None for no ground. Ground-reflected light brightens cloud bases
            and the ground receives the clouds' shadows; 0.15 is typical of
            vegetated land in the visible.
        ground_height: Height of the ground plane in data units (default 0,
            the model surface).
        aerial_perspective: Fill the clear air around the scene with a
            physical haze, so distant objects fade toward the sky color as in
            a real view across tens of kilometres. Two scatterers, both
            decaying exponentially with height above the ground: air molecules
            (Rayleigh phase, scattering coefficients for 680/550/440 nm, scale
            height 8 km) and aerosol (Henyey-Greenstein phase). The haze fills
            a box around the data and the camera only. Meant for the
            "physical_sky" preset; requires data in meters.
        visibility_km: Meteorological visibility at the ground, which sets the
            aerosol extinction by Koschmieder's relation, beta = 3.912 / V
            (minus the molecular part). 50 km is a clean continental day.
        aerosol_scale_height_m: E-folding height of the aerosol (roughly the
            boundary-layer depth).
        aerosol_asymmetry: Henyey-Greenstein g of the aerosol (~0.7 for
            continental aerosol in the visible).
        aerosol_single_scattering_albedo: Scattered fraction of the aerosol
            extinction; the rest is absorbed (~0.9 continental).
    """

    preset: str = "sky"
    sun_elevation_deg: float = 35.0
    sun_azimuth_deg: float = 40.0
    sun_strength: float = 2.0
    background_strength: float = 1.0
    ground_albedo: Optional[float] = None
    ground_height: float = 0.0
    aerial_perspective: bool = False
    visibility_km: float = 50.0
    aerosol_scale_height_m: float = 1500.0
    aerosol_asymmetry: float = 0.7
    aerosol_single_scattering_albedo: float = 0.9

    def __post_init__(self):
        if self.preset not in WORLD_PRESETS:
            raise ValueError(
                f"Unknown world preset {self.preset!r}; choose from {WORLD_PRESETS}."
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "preset": self.preset,
            "sun_elevation_deg": self.sun_elevation_deg,
            "sun_azimuth_deg": self.sun_azimuth_deg,
            "sun_strength": self.sun_strength,
            "background_strength": self.background_strength,
            "ground_albedo": self.ground_albedo,
            "ground_height": self.ground_height,
            "aerial_perspective": self.aerial_perspective,
            "visibility_km": self.visibility_km,
            "aerosol_scale_height_m": self.aerosol_scale_height_m,
            "aerosol_asymmetry": self.aerosol_asymmetry,
            "aerosol_single_scattering_albedo": self.aerosol_single_scattering_albedo,
        }


@dataclass
class BlenderHoldConfig:
    """
    Render a scene at one model time over many video frames.

    The data stay frozen; only what the *flow clock* drives moves. Today that
    is streamline markers with ``arrow_travel_time``, which sit at fixed
    travel times along each streamline and so move downstream as the clock
    advances. The clock is model time in seconds: it starts at
    ``clock_start`` and advances ``clock_seconds_per_frame`` every video
    frame, so ``clock_seconds_per_frame * fps`` is the playback speed-up.

    The motion loops seamlessly when the total clock advance,
    ``n_frames * clock_seconds_per_frame``, is a whole number of marker
    travel times: the last frame is then one step short of the first.

    Attributes:
        n_frames: Video frames to render.
        clock_seconds_per_frame: Model seconds the flow clock advances per
            video frame.
        clock_start: Flow clock at the first frame (s).
    """

    n_frames: int
    clock_seconds_per_frame: float
    clock_start: float = 0.0

    def clock_keyframes(self, frame_start: int) -> List[Dict[str, float]]:
        """The flow clock at every video frame, as manifest keyframes."""
        return [
            {
                "frame": frame_start + frame_offset,
                "seconds": self.clock_start
                + frame_offset * self.clock_seconds_per_frame,
            }
            for frame_offset in range(self.n_frames)
        ]


@dataclass
class BlenderExportConfig:
    """
    Top-level configuration for a Blender export.

    Attributes:
        transform: The single spatial transform (see :class:`BlenderTransform`).
        render: Render-level settings (see :class:`BlenderRenderConfig`).
        camera: Camera placement/animation (see :class:`BlenderCameraConfig`).
        world: World background + key-light look (see :class:`BlenderWorldConfig`).
        fps: Frames per second the animation advertises; also the rate at which
            Alembic samples are spaced in time.
        frame_start: First Blender frame number.
        overlay_colorbars: Composite each colormapped object's colorbar onto
            the render. False leaves the render clean (the colorbar PNGs are
            still written to ``assets/colorbars/``), for figures that draw
            their own legend.
        hold: Render a single-time scene over many video frames with a
            running flow clock (see :class:`BlenderHoldConfig`). None renders
            one frame per data time.
    """

    transform: BlenderTransform = field(default_factory=BlenderTransform)
    render: BlenderRenderConfig = field(default_factory=BlenderRenderConfig)
    camera: BlenderCameraConfig = field(default_factory=BlenderCameraConfig)
    world: BlenderWorldConfig = field(default_factory=BlenderWorldConfig)
    fps: float = 24.0
    frame_start: int = 1
    overlay_colorbars: bool = True
    hold: Optional[BlenderHoldConfig] = None

    @classmethod
    def for_clouds(
        cls, quality: Optional[str] = None, **overrides
    ) -> "BlenderExportConfig":
        """
        Defaults for a physically rendered cloud scene.

        Uses the "physical_sky" world (radiometrically consistent sun and sky,
        automatic exposure) over a ground of albedo 0.15. The render keeps 128
        volume bounces and denoising. The camera sits lower and closer than the
        general default (10 deg elevation, 1.2 bounding-box diagonals, 35 mm),
        which is how clouds are usually seen. Pass any field as a keyword to
        override it, e.g. ``for_clouds(fps=12)``.

        Args:
            quality: A render quality preset ("preview" or "final", see
                :meth:`BlenderRenderConfig.set_quality`). None keeps the
                "final" settings.
            **overrides: Fields of this config to replace.
        """
        config = cls(
            world=BlenderWorldConfig(preset="physical_sky", ground_albedo=0.15),
            camera=BlenderCameraConfig(
                elevation_deg=10.0, distance_factor=1.2, lens_mm=35.0
            ),
        )
        if quality is not None:
            config.render.set_quality(quality)
        for field_name, value in overrides.items():
            setattr(config, field_name, value)
        return config

    def use_view(self, view: "BlenderView") -> "BlenderExportConfig":
        """
        Pin this config to a fixed camera and scene origin (see BlenderView).

        Sets ``camera.location``, ``camera.look_at``, ``camera.lens_mm`` and
        ``transform.origin_shift``. Change ``camera.lens_mm`` afterwards to
        zoom one panel of a figure without moving its camera.

        Returns:
            This config, changed in place, for chaining.
        """
        self.camera.location = view.location
        self.camera.look_at = view.look_at
        self.camera.lens_mm = view.lens_mm
        self.transform.origin_shift = view.origin_shift
        return self


# =============================================================================
# Colormap baking (scalar field -> per-vertex RGB)
# =============================================================================
def _safe_colormap_name(name: Optional[str]) -> str:
    """
    Return a filename- and registry-safe colormap name not already in use.

    Custom colormap names can contain spaces or punctuation (the matplotlib
    default is even literally "custom colormap"), which make poor asset
    filenames and registry keys. Reduce to ``[0-9A-Za-z_-]`` and, if that
    collides with an already-registered colormap, append a numeric suffix so we
    never clobber a different colormap.
    """
    import matplotlib

    base = str(name) if name else "skyvista_cmap"
    safe = re.sub(r"[^0-9A-Za-z_-]+", "_", base).strip("_") or "skyvista_cmap"
    candidate = safe
    suffix = 1
    while candidate in matplotlib.colormaps:
        candidate = f"{safe}_{suffix}"
        suffix += 1
    return candidate


def resolve_colormap(cmap: Union[str, "Colormap"]) -> Tuple["Colormap", str]:
    """
    Normalize a ``cmap`` argument to a ``(Colormap, name)`` pair.

    ``cmap`` may be a matplotlib colormap *name* or an actual ``Colormap``
    object. A Colormap object is not JSON-serializable, so it cannot go into the
    manifest directly; and the steps that rebuild the colormap on the Blender
    side (the colorbar PNG and the volume color-ramp LUT) look it up by *name*.
    We therefore return both:

    - the ``Colormap`` object, used to bake per-vertex colors here; and
    - a name string that is JSON-safe and always resolvable via
      ``matplotlib.colormaps[name]``.

    A custom, unregistered Colormap is registered under a safe unique name so
    that by-name lookups downstream find exactly this colormap.
    """
    import matplotlib

    # A plain name: look up the object, pass the name straight through.
    if isinstance(cmap, str):
        return matplotlib.colormaps[cmap], cmap

    # A Colormap-like object whose name already round-trips through the registry
    # (e.g. "viridis", or a reversed "viridis_r"): use that name as-is.
    name = getattr(cmap, "name", None)
    if name and name in matplotlib.colormaps:
        return cmap, name

    # Unregistered/custom colormap: register it under a safe name so the colorbar
    # and LUT steps -- which resolve by name -- find exactly this colormap.
    safe_name = _safe_colormap_name(name)
    matplotlib.colormaps.register(cmap, name=safe_name)
    return cmap, safe_name


def bake_scalar_to_rgb(
    scalar_values: np.ndarray,
    colormap: Union[str, "Colormap"],
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
        colormap: A matplotlib colormap name (e.g. "viridis") or Colormap object.
        color_limits: (vmin, vmax) mapped to the colormap ends. Held fixed
            across all frames so animated colors stay consistent.

    Returns:
        (n_points, 3) float array of RGB in [0, 1].
    """
    import matplotlib
    from matplotlib.colors import Normalize

    colormap = matplotlib.colormaps[colormap] if isinstance(colormap, str) else colormap
    normalize_to_unit_interval = Normalize(vmin=color_limits[0], vmax=color_limits[1])
    rgba_values = colormap(normalize_to_unit_interval(np.asarray(scalar_values)))
    return np.ascontiguousarray(rgba_values[:, :3], dtype=np.float32)


def sample_colormap_stops(colormap_name: str, n_stops: int = 32) -> List[List[float]]:
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


def write_colormap_lut(bundle_dir: Path, colormap_name: str, n_stops: int = 32) -> str:
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
            "The Blender export backend requires the 'blender' extra:\n    pip"
            " install skyvista[blender]\nThis installs 'alembic3d' (the 3D"
            " Alembic bindings). Do NOT install 'alembic' -- that is the"
            " unrelated SQLAlchemy database tool."
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


def _zup_to_yup(points_xyz: np.ndarray) -> np.ndarray:
    """
    Rotate Z-up data coordinates into the Y-up convention Alembic expects.

    Blender's Alembic importer assumes the file is Y-up and rotates it to its
    own Z-up on import: file (x, y, z) -> Blender (x, -z, y). Our data is
    already Z-up, so writing it verbatim comes out mis-rotated (the vertical
    axis lands on -Y and, once the parent-empty centering is applied, the
    geometry is flung far off-origin). We pre-rotate Z-up -> Y-up here,
    file = (x, z, -y), so Blender's import rotation cancels it and the geometry
    lands back at true Z-up. This is a proper rotation (determinant +1, so face
    winding/normals are preserved) and is exactly what Blender itself does when
    it *exports* Alembic -- the resulting .abc is a standard Y-up file that any
    tool reads correctly.
    """
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]
    return np.column_stack([x, z, -y]).astype(points_xyz.dtype, copy=False)


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

    # Uniform time sampling: one sample every 1/fps seconds. Start at t=1/fps
    # (not 0) so sample i lands on Blender frame i+1 -- i.e. the first timestep
    # plays at frame 1. This makes the Mesh Sequence Cache 1-indexed, matching
    # the VDB volume sequence (frame_start=1) and the manifest's frame_start=1;
    # a t=0 start would put meshes on frames 0..N-1 while volumes sit on 1..N,
    # so the two carriers would show different timesteps on the same frame.
    time_sampling = TimeSampling(1.0 / fps, 1.0 / fps)
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
        # Write Y-up so Blender's Alembic import rotation lands it at true Z-up.
        imath_points = _numpy_points_to_v3f(imath, _zup_to_yup(frame.points_xyz))

        # Flatten triangle indices and build the per-face vertex-count stream
        # (all 3, since we triangulated).
        flat_indices = frame.triangle_vertex_indices.ravel()
        imath_face_indices = _numpy_ints_to_intarray(imath, flat_indices)
        n_triangles = len(frame.triangle_vertex_indices)
        imath_face_counts = _numpy_ints_to_intarray(
            imath, np.full(n_triangles, 3, dtype=np.int32)
        )

        mesh_schema.set(
            OPolyMeshSchemaSample(imath_points, imath_face_indices, imath_face_counts)
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
        # color_by, else scale_by; None -> fall back to the active scalar
        return spec.color_scalar_name()
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

    Surface-mesh specs (contour, slice, and trajectories or vector glyphs
    with ``blender_instancing=False``) go to Alembic; volume and cloud specs go
    to a VDB sequence; vector glyphs and particles by default go to per-frame
    point files that Blender instances on; streamlines and trajectory tubes by
    default go to per-frame polylines that Blender sweeps into tubes.
    """
    from .varspec import (
        CloudSpec,
        DyeSpec,
        StreamlineSpec,
        TrajectorySpec,
        VectorSpec,
        VolumeSpec,
    )

    if isinstance(spec, StreamlineSpec):
        return _build_streamline_entry(
            spec, dataset, times, data_subdir, running_bounds
        )

    if (
        isinstance(spec, TrajectorySpec)
        and spec.appearance.style == "particle"
        and spec.appearance.blender_instancing
    ):
        return _build_particle_points_entry(
            spec, dataset, times, data_subdir, running_bounds
        )
    if (
        isinstance(spec, TrajectorySpec)
        and spec.appearance.style == "tube"
        and spec.appearance.blender_instancing
    ):
        return _build_trajectory_curves_entry(
            spec, dataset, times, data_subdir, running_bounds
        )
    if isinstance(spec, VectorSpec) and spec.appearance.blender_instancing:
        return _build_vector_points_entry(
            spec, dataset, times, data_subdir, running_bounds
        )
    if isinstance(spec, CloudSpec):
        return _build_cloud_entry(spec, dataset, times, data_subdir, running_bounds)
    if isinstance(spec, DyeSpec):
        return _build_dye_entry(spec, dataset, times, data_subdir, running_bounds)
    if isinstance(spec, VolumeSpec):
        return _build_volume_entry(spec, dataset, times, data_subdir, running_bounds)
    return _build_mesh_entry(spec, dataset, times, data_subdir, fps, running_bounds)


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
        # cmap may be a name or an actual Colormap object; resolve to both the
        # object (for baking) and a JSON-safe name (for the manifest/colorbar).
        colormap, colormap_name = resolve_colormap(
            appearance.cmap or DEFAULT_COLORMAP_NAME
        )
        # ---- Pass 2: bake per-vertex colors with the fixed global range
        for frame in frames:
            if frame.scalar_values is not None and len(frame.scalar_values) > 0:
                frame.rgb_values = bake_scalar_to_rgb(
                    frame.scalar_values, colormap, color_limits
                )

    # ---- Write the Alembic sequence
    object_subdir = data_subdir / spec.name
    object_subdir.mkdir(parents=True, exist_ok=True)
    alembic_path = object_subdir / "sequence.abc"
    write_mesh_sequence_alembic(alembic_path, frames, fps=fps, object_name=spec.name)

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
        (len(frame.points_xyz), len(frame.triangle_vertex_indices)) for frame in frames
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
# Vector glyph export (point files, instanced in Blender)
# =============================================================================
# Glyph shapes the Blender build script can instance.
INSTANCED_GLYPH_TYPES = ("arrow", "cone")

# Arrow proportions, as fractions of the arrow's length. The arrow is scaled
# uniformly with the vector magnitude (like PyVista's glyphs), so these ratios
# hold for every arrow.
ARROW_SHAFT_RADIUS_FRACTION = 0.03
ARROW_HEAD_RADIUS_FRACTION = 0.08
ARROW_HEAD_LENGTH_FRACTION = 0.28

# Constant-width arrows (VectorAppearance.glyph_width): head size as multiples
# of the shaft diameter.
CONSTANT_WIDTH_HEAD_DIAMETER_PER_SHAFT_DIAMETER = 2.6
CONSTANT_WIDTH_HEAD_LENGTH_PER_SHAFT_DIAMETER = 3.5


def _rectilinear_glyph_points(
    spec, dataset_at_time
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """
    Glyph positions, vector components and color values for one time.

    The glyph lattice is the (strided) model grid itself, so positions are the
    Cartesian product of the x, y and z coordinates, flattened in the same
    Fortran order as the component arrays. Points where any component is
    non-finite (masked, below ground, outside the domain) are dropped, as are
    points that ``geometry.mask_by`` excludes.

    Returns:
        (positions (n, 3), [u, v, w] each (n,), color values (n,)).
    """
    from .grids import resolve_coordinates

    coordinate_names = resolve_coordinates(dataset_at_time, ["x", "y", "z"])
    u_variable = dataset_at_time[spec.geometry.u_varname]
    # The component's own dimension order fixes how the flattening runs, so
    # build the position grid in exactly that order.
    axis_by_dimension = {coordinate_names[axis]: axis for axis in ("x", "y", "z")}
    if set(u_variable.dims) != set(axis_by_dimension):
        raise NotImplementedError(
            "Instanced Blender glyphs need vector components on a rectilinear "
            f"(x, y, z) grid; got dims {u_variable.dims}. Select a single time "
            "first, or pass blender_instancing=False to bake PyVista glyphs."
        )
    coordinate_arrays = [
        np.asarray(dataset_at_time[dimension].values, dtype=float)
        for dimension in u_variable.dims
    ]
    meshed_coordinates = np.meshgrid(*coordinate_arrays, indexing="ij")
    positions_by_axis = {
        axis_by_dimension[dimension]: meshed.ravel(order="F")
        for dimension, meshed in zip(u_variable.dims, meshed_coordinates, strict=True)
    }
    positions = np.stack(
        [positions_by_axis["x"], positions_by_axis["y"], positions_by_axis["z"]],
        axis=1,
    )

    # Components and the coloring scalar, flattened the same way.
    components = spec.vector_components(dataset_at_time)
    color_values = np.asarray(
        spec.color_values(dataset_at_time, components), dtype=float
    )

    # Drop glyphs with no defined vector, and any the mask excludes.
    finite = np.isfinite(positions).all(axis=1)
    for component in components:
        finite &= np.isfinite(component)
    glyph_mask = spec.glyph_mask(dataset_at_time)
    if glyph_mask is not None:
        finite &= glyph_mask
    return (
        positions[finite],
        [np.asarray(component[finite], dtype=float) for component in components],
        color_values[finite],
    )


def _write_point_frames(
    object_name: str,
    appearance,
    frame_points: List[Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]],
    data_subdir: Path,
    coloring_mode: str,
    default_label: str,
    extra_arrays_per_frame: Optional[List[Dict[str, np.ndarray]]] = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Bake colors and write one ``.npz`` point file per frame.

    Shared by every point carrier (vector glyphs, particles). Colors are baked
    here with one global range, exactly as for meshes, so they are identical
    across frames and match the colorbar. The point data travels as ``.npz``
    (numpy) files, not Alembic: Blender's Alembic importer keeps only vertex
    colors, so other per-point data (vectors) would be lost.

    Files: ``data/<name>/points_0001.npz`` ... holding ``positions`` (data
    units), ``colors`` (sRGB in [0, 1], as matplotlib returns them; the build
    script linearizes them) and, when given, ``vectors``; all float32, (n, 3).

    Args:
        object_name: Object name; also the data subdirectory.
        appearance: The spec's appearance (color, cmap, clim, title, shader).
        frame_points: Per frame, (positions, vectors or None, color values or
            None).
        data_subdir: The bundle's data directory.
        coloring_mode: Manifest coloring mode for a colormapped object:
            "instance_color" (colors on instances, e.g. arrows) or
            "vertex_color" (colors on the points themselves, e.g. spheres).
        default_label: Colorbar label when the appearance sets no title.
        extra_arrays_per_frame: Additional arrays saved in each frame's file
            (e.g. ``curve_sizes`` for streamlines), one dict per frame.

    Returns:
        (manifest material block, path pattern relative to the bundle).
    """
    # ---- The global range of the coloring scalar over all frames.
    all_color_values = [
        np.asarray(color_values, dtype=float)
        for _, _, color_values in frame_points
        if color_values is not None and np.size(color_values)
    ]
    global_color_min = min((np.nanmin(v) for v in all_color_values), default=np.inf)
    global_color_max = max((np.nanmax(v) for v in all_color_values), default=-np.inf)

    # ---- Coloring: solid, or a baked colormap over the global range. An
    # explicit clim is enough by itself: a field that happens to be uniform
    # still has a meaningful color on a user-chosen scale.
    uses_scalar_coloring = (
        appearance.color is None
        and np.isfinite(global_color_min)
        and (appearance.clim is not None or global_color_max > global_color_min)
    )
    color_limits: Optional[Tuple[float, float]] = None
    colormap = colormap_name = None
    if uses_scalar_coloring:
        color_limits = appearance.clim or (global_color_min, global_color_max)
        colormap, colormap_name = resolve_colormap(
            appearance.cmap or DEFAULT_COLORMAP_NAME
        )
    solid_rgb = None
    if not uses_scalar_coloring:
        from matplotlib.colors import to_rgb

        solid_rgb = np.array(to_rgb(appearance.color or "#cccccc"))

    # ---- One point file per frame.
    object_subdir = data_subdir / object_name
    object_subdir.mkdir(parents=True, exist_ok=True)
    for frame_index, (positions, vectors, color_values) in enumerate(frame_points):
        if uses_scalar_coloring:
            colors = bake_scalar_to_rgb(color_values, colormap, color_limits)
        else:
            colors = np.tile(solid_rgb, (len(positions), 1))
        arrays = {
            "positions": np.asarray(positions, dtype=np.float32).reshape(-1, 3),
            "colors": np.asarray(colors, dtype=np.float32).reshape(-1, 3),
        }
        if vectors is not None:
            arrays["vectors"] = np.asarray(vectors, dtype=np.float32).reshape(-1, 3)
        if extra_arrays_per_frame is not None:
            arrays.update(extra_arrays_per_frame[frame_index])
        np.savez(object_subdir / f"points_{frame_index + 1:04d}.npz", **arrays)

    # ---- Manifest material.
    material = appearance.to_blender_material()
    if uses_scalar_coloring:
        material["coloring"] = {
            "mode": coloring_mode,
            "attribute": VERTEX_COLOR_ATTRIBUTE_NAME,
            "cmap": colormap_name,
            "clim": list(color_limits),
            "label": appearance.scalar_bar_title or default_label,
        }
    else:
        material["coloring"] = {"mode": "solid", "color": appearance.color or "#cccccc"}
    path_pattern = str(
        (object_subdir / "points_####.npz").relative_to(data_subdir.parent)
    )
    return material, path_pattern


def _build_streamline_entry(
    spec: "VarSpec",
    dataset,
    times: List[Any],
    data_subdir: Path,
    running_bounds: Optional[np.ndarray],
) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    """
    Export streamlines as per-frame polylines for Blender to sweep into tubes.

    Each frame's file holds every streamline's points end to end
    (``positions``), their baked colors, and ``curve_sizes`` (points per
    streamline, in order). The build script rebuilds them as a Curves object;
    geometry nodes sweep a round tube of ``tube_radius`` along each curve and,
    if ``arrow_spacing`` is set, place downstream-pointing cones along it. The
    points are ordered downstream (see ``StreamlineSpec.polylines``), so the
    curve tangent is the flow direction.

    Each file also holds ``travel_times`` (s from the seed) and ``speeds``
    per point. With ``arrow_travel_time`` set, the build script places the
    cones at fixed travel times offset by the flow clock instead, so they move
    with the air, and shrinks them where slow air packs them close together.
    """
    frame_points = []
    extra_arrays_per_frame = []
    for time in times:
        lines = spec.polylines(dataset, time)
        if lines:
            positions = np.concatenate([line.points for line in lines])
            color_values = np.concatenate([line.color_values for line in lines])
            travel_times = np.concatenate([line.travel_times for line in lines])
            speeds = np.concatenate([line.speeds for line in lines])
            curve_sizes = np.array([len(line.points) for line in lines], dtype=np.int32)
        else:
            positions = np.zeros((0, 3))
            color_values = np.zeros(0)
            travel_times = np.zeros(0)
            speeds = np.zeros(0)
            curve_sizes = np.zeros(0, dtype=np.int32)
        frame_points.append((positions, None, color_values))
        extra_arrays_per_frame.append(
            {
                "curve_sizes": curve_sizes,
                "travel_times": travel_times.astype(np.float32),
                "speeds": speeds.astype(np.float32),
            }
        )
        running_bounds = _accumulate_bounds(running_bounds, positions)

    appearance = spec.appearance
    material, path_pattern = _write_point_frames(
        spec.name,
        appearance,
        frame_points,
        data_subdir,
        coloring_mode="vertex_color",
        default_label=spec.geometry.color_by,
        extra_arrays_per_frame=extra_arrays_per_frame,
    )
    tube_radius = float(appearance.tube_radius)
    object_entry = {
        "name": spec.name,
        "spec_type": "streamline",
        "geometry": {
            "carrier": "polylines",
            "path_pattern": path_pattern,
            "n_frames": len(frame_points),
            "tube_radius": tube_radius,
            "arrow_spacing": (
                float(appearance.arrow_spacing)
                if appearance.arrow_spacing is not None
                else None
            ),
            "arrow_travel_time": (
                float(appearance.arrow_travel_time)
                if appearance.arrow_travel_time is not None
                else None
            ),
            "arrow_head_radius": tube_radius * appearance.arrow_head_radius_factor,
            "arrow_head_length": tube_radius * appearance.arrow_head_length_factor,
        },
        "material": material,
    }
    return object_entry, running_bounds


# Evaluated points per segment of a trajectory trail's curve in Blender.
TRAJECTORY_CURVE_RESOLUTION = 3


def _build_trajectory_curves_entry(
    spec: "VarSpec",
    dataset,
    times: List[Any],
    data_subdir: Path,
    running_bounds: Optional[np.ndarray],
) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    """
    Export trajectory tubes as per-frame trails for Blender to sweep into tubes.

    This uses the same ``polylines`` carrier as streamlines. Each frame's file
    holds every trail's points end to end (``positions``, oldest point first,
    so the curve tangent is the direction of motion), their baked colors,
    ``curve_sizes`` (points per trail), and ``radius_scales`` (the tube radius
    at each point as a fraction of ``tube_radius``; see ``trail_taper``). The
    build script rebuilds the trails as a Curves object, stores the radius
    scales as the curves' ``radius`` attribute (which Curve to Mesh multiplies
    into the profile), and places one cone on the last point of each trail.

    Compared with baking PyVista's tubes, the files hold three numbers per
    track point instead of a tube's ring of vertices, and the tube's look
    (radius, cone size, smoothness) stays editable in Blender.
    """
    geometry = spec.geometry
    frame_points = []
    extra_arrays_per_frame = []
    for time in times:
        trails = spec.trails(dataset, time)
        frame_points.append((trails.positions, None, trails.color_values))
        extra_arrays_per_frame.append(
            {
                "curve_sizes": trails.curve_sizes.astype(np.int32),
                "radius_scales": trails.radius_scales.astype(np.float32),
            }
        )
        running_bounds = _accumulate_bounds(running_bounds, trails.positions)

    material, path_pattern = _write_point_frames(
        spec.name,
        spec.appearance,
        frame_points,
        data_subdir,
        coloring_mode="vertex_color",
        default_label=geometry.scalar or "",
        extra_arrays_per_frame=extra_arrays_per_frame,
    )
    tube_radius = float(geometry.tube_radius)
    # A head cone only when it has a length; head_length_frac = 0 drops it.
    head_length = tube_radius * float(geometry.head_length_frac)
    object_entry = {
        "name": spec.name,
        "spec_type": "trajectory",
        "geometry": {
            "carrier": "polylines",
            "path_pattern": path_pattern,
            "n_frames": len(frame_points),
            "tube_radius": tube_radius,
            # Vertices around the tube: the trajectory's tube_resolution.
            "profile_vertices": int(geometry.tube_resolution),
            # Model tracks are sampled at output times; Catmull-Rom curves
            # with a few points per segment round the corners between them.
            "curve_resolution": TRAJECTORY_CURVE_RESOLUTION,
            "head_cone": (
                {
                    "radius": tube_radius * float(geometry.head_radius_frac),
                    "length": head_length,
                }
                if head_length > 0
                else None
            ),
        },
        "material": material,
    }
    return object_entry, running_bounds


def _build_particle_points_entry(
    spec: "VarSpec",
    dataset,
    times: List[Any],
    data_subdir: Path,
    running_bounds: Optional[np.ndarray],
) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    """
    Export trajectory particles as per-frame point files.

    Each frame holds every particle's position at that time and its baked
    color. The build script turns each frame into a native Blender point cloud
    (geometry nodes' Mesh to Points), which Cycles ray-traces as exact spheres
    of ``particle_radius`` -- no sphere meshes are generated or stored, so the
    cost stays small even for 10^5-10^6 particles (e.g. model parcels).
    """
    frame_points = []
    for time in times:
        positions, scalar_values = spec.particle_positions(dataset, time)
        frame_points.append((positions, None, scalar_values))
        running_bounds = _accumulate_bounds(running_bounds, positions)

    material, path_pattern = _write_point_frames(
        spec.name,
        spec.appearance,
        frame_points,
        data_subdir,
        coloring_mode="vertex_color",
        default_label=spec.geometry.scalar or "",
    )
    object_entry = {
        "name": spec.name,
        "spec_type": "trajectory",
        "geometry": {
            "carrier": "point_glyphs",
            "path_pattern": path_pattern,
            "n_frames": len(frame_points),
            "glyph_type": "sphere",
            "radius": float(spec.geometry.particle_radius),
        },
        "material": material,
    }
    return object_entry, running_bounds


def _build_vector_points_entry(
    spec: "VarSpec",
    dataset,
    times: List[Any],
    data_subdir: Path,
    running_bounds: Optional[np.ndarray],
) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    """
    Export vector glyphs as per-frame point files for instancing in Blender.

    Instead of baking every arrow's triangles (PyVista's glyph filter), each
    frame stores only what defines a glyph: its position, its vector, and its
    color. Blender's build script turns each file into a point cloud and a
    geometry-nodes tree places one arrow per point, aligned with the vector and
    scaled by ``factor * |vector|``. This keeps files a few numbers per arrow,
    renders the arrows as cheap instances, and lets the arrow shape, size and
    density be edited in Blender without re-exporting.

    The point data travels as ``.npz`` (numpy) files, not Alembic: Blender's
    Alembic importer keeps only vertex colors, so the vectors themselves would
    be lost. Colors are baked here with one global range, exactly as for
    meshes, so they are identical across frames and match the colorbar.

    Files: ``data/<name>/points_0001.npz`` ... with arrays ``positions``
    (data units), ``vectors`` (the field's units) and ``colors`` (sRGB in
    [0, 1], as matplotlib returns them; the build script linearizes them), all
    float32 with shape (n, 3).
    """
    from .grid_utils import select_time

    appearance = spec.appearance
    glyph_type = (
        appearance.glyph_type
        if appearance.glyph_type in INSTANCED_GLYPH_TYPES
        else "arrow"
    )

    # ---- Glyph points of every frame
    frame_points = []
    length_per_unit = spec.geometry.factor
    for time in times:
        dataset_at_time = spec.subsample(select_time(dataset, time))
        positions, components, color_values = _rectilinear_glyph_points(
            spec, dataset_at_time
        )
        frame_points.append((positions, components, color_values))
        running_bounds = _accumulate_bounds(running_bounds, positions)
        # Auto arrow scale from the first frame that has any vector at all, so
        # one scale (hence comparable lengths) applies to the whole animation.
        if length_per_unit is None and positions.size:
            length_per_unit = spec.auto_factor(dataset_at_time, components)
    if length_per_unit is None:
        length_per_unit = 1.0

    # ---- Colors + one point file per frame
    material, path_pattern = _write_point_frames(
        spec.name,
        appearance,
        [
            (
                positions,
                np.stack(components, axis=1) if positions.size else None,
                color_values,
            )
            for positions, components, color_values in frame_points
        ],
        data_subdir,
        coloring_mode="instance_color",
        default_label=spec.color_scalar_name() or "speed",
    )

    object_entry = {
        "name": spec.name,
        "spec_type": "vector",
        "geometry": {
            "carrier": "point_glyphs",
            "path_pattern": path_pattern,
            "n_frames": len(frame_points),
            "glyph_type": glyph_type,
            "length_per_unit": float(length_per_unit),
            "shaft_radius_fraction": ARROW_SHAFT_RADIUS_FRACTION,
            "head_radius_fraction": ARROW_HEAD_RADIUS_FRACTION,
            "head_length_fraction": ARROW_HEAD_LENGTH_FRACTION,
        },
        "material": material,
    }
    if appearance.glyph_width is not None:
        # Fixed shaft and head size in data units; only the length varies.
        shaft_diameter = float(appearance.glyph_width)
        object_entry["geometry"].update(
            {
                "width_mode": "constant",
                "shaft_radius": shaft_diameter / 2,
                "head_radius": shaft_diameter
                * CONSTANT_WIDTH_HEAD_DIAMETER_PER_SHAFT_DIAMETER
                / 2,
                "head_length": shaft_diameter
                * CONSTANT_WIDTH_HEAD_LENGTH_PER_SHAFT_DIAMETER,
            }
        )
    else:
        object_entry["geometry"]["width_mode"] = "proportional"
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
                "VDB volume export requires a rectilinear grid with 1-D x/y/z"
                f" coordinates; '{coordinate_name}' is missing or"
                " multi-dimensional. Resample to a uniform grid first."
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
        object_subdir,
        frame_arrays,
        grid_name=varname,
        transform_matrix=transform_matrix,
    )
    # Blender picks the sequence up from the '####' numbered pattern.
    path_pattern = f"data/{spec.name}/{varname}_####.vdb"

    # ---- Material: a volume shader driven by the scalar through a color ramp
    color_limits = appearance.clim or (
        (global_min, global_max) if np.isfinite(global_min) else (0.0, 1.0)
    )
    # cmap may be a name or an actual Colormap object; resolve to a JSON-safe
    # name that also always resolves via matplotlib.colormaps[...] downstream.
    _, colormap_name = resolve_colormap(appearance.cmap or DEFAULT_COLORMAP_NAME)
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
            "n_frames": len(times),
            "path_pattern": path_pattern,
            "grid_name": varname,
        },
        "material": material,
    }
    return object_entry, running_bounds


# =============================================================================
# Cloud export (multi-species VDB)
# =============================================================================
# Extinction grids are stored in km^-1 rather than m^-1. Blender loads VDB
# grids at half precision by default, whose normal range starts at 6e-5. Thin
# cloud edges (1e-6 to 1e-4 m^-1) would land in its subnormal range and lose
# precision; in km^-1 they are 1e-3 to 0.1, comfortably representable. The
# build script converts back to m^-1 in the shader.
CLOUD_EXTINCTION_GRID_UNITS = "1/km"
METERS_PER_KILOMETER = 1000.0

# Prefixes of the per-species grids inside each cloud VDB file.
CLOUD_EXTINCTION_GRID_PREFIX = "extinction_"
CLOUD_MIE_DIAMETER_GRID_PREFIX = "mie_diameter_"

# Above this many uniform vertical levels the requested spacing is coarsened,
# so a finely stretched grid cannot silently produce a multi-gigabyte VDB.
MAX_UNIFORM_VERTICAL_LEVELS = 2000


# The default uniform vertical spacing is never finer than this fraction of
# the horizontal grid spacing. Near-surface model levels can be ~20 m apart
# under a ~500 m horizontal grid; resolving them would multiply the voxel
# count ~5x while adding no visible structure, since every cloud feature is
# at least one horizontal cell wide.
MIN_VERTICAL_SPACING_FRACTION_OF_HORIZONTAL = 0.25


def choose_uniform_vertical_spacing(
    level_heights: np.ndarray,
    cloudy_level_mask: np.ndarray,
    horizontal_spacing: Optional[float] = None,
) -> float:
    """
    Pick a uniform vertical voxel size that preserves the cloudy layers.

    VDB voxels are uniform, but model levels are usually stretched (finer near
    the surface). Stretching the voxels to the mean spacing would misplace and
    distort clouds. Instead we resample onto uniform levels, at the finest
    native spacing found among levels that hold condensate (and their
    immediate neighbours). The cloudy part of the column therefore keeps its
    full native resolution, while clear layers far above or below do not force
    a needlessly fine grid. The result is floored at
    ``MIN_VERTICAL_SPACING_FRACTION_OF_HORIZONTAL`` of the horizontal spacing.

    Args:
        level_heights: Native level heights (m), increasing.
        cloudy_level_mask: True for levels containing any condensate.
        horizontal_spacing: Smaller of the x and y grid spacings (m), or None
            to skip the floor.

    Returns:
        The uniform spacing in meters.
    """
    level_spacings = np.diff(level_heights)
    # A spacing touches a cloudy layer if either of its two levels is cloudy.
    spacing_touches_cloud = cloudy_level_mask[:-1] | cloudy_level_mask[1:]
    if spacing_touches_cloud.any():
        finest_cloudy_spacing = float(level_spacings[spacing_touches_cloud].min())
    else:
        finest_cloudy_spacing = float(level_spacings.min())
    # Do not resolve vertically far beyond what the horizontal grid resolves.
    if horizontal_spacing is None:
        return finest_cloudy_spacing
    return max(
        finest_cloudy_spacing,
        MIN_VERTICAL_SPACING_FRACTION_OF_HORIZONTAL * float(horizontal_spacing),
    )


def interpolate_to_levels(
    data_array, vertical_dimension: str, native_heights: np.ndarray, target_heights
):
    """
    Linearly interpolate a DataArray along its vertical dimension.

    Implemented with numpy index arithmetic (no scipy): for every target level
    find the bracketing native levels and blend them with linear weights.
    Targets outside the native range are clamped to the end levels. Variables
    without the vertical dimension are returned unchanged.
    """
    import xarray as xr

    if vertical_dimension not in data_array.dims:
        return data_array

    # Index of the native level at or below each target (clamped into range).
    upper_index = np.searchsorted(native_heights, target_heights, side="right")
    upper_index = np.clip(upper_index, 1, len(native_heights) - 1)
    lower_index = upper_index - 1

    # Linear weight of the upper level, clamped so targets outside the native
    # range take the nearest end value instead of extrapolating.
    lower_heights = native_heights[lower_index]
    upper_heights = native_heights[upper_index]
    upper_weight = np.clip(
        (target_heights - lower_heights) / (upper_heights - lower_heights), 0.0, 1.0
    )

    # Blend the two bracketing levels. Coordinates along the vertical are
    # dropped first: indexing with repeated level indices duplicates them,
    # which would stop xarray from aligning the two halves of the blend.
    vertical_coordinates = [
        name
        for name, coordinate in data_array.coords.items()
        if vertical_dimension in coordinate.dims
    ]
    lower_values = data_array.isel({vertical_dimension: lower_index}).drop_vars(
        vertical_coordinates
    )
    upper_values = data_array.isel({vertical_dimension: upper_index}).drop_vars(
        vertical_coordinates
    )
    weight_array = xr.DataArray(upper_weight, dims=[vertical_dimension])
    blended = lower_values * (1.0 - weight_array) + upper_values * weight_array
    return blended.assign_coords({vertical_dimension: target_heights})


def write_multigrid_vdb(
    file_path: Path,
    grids: Dict[str, Tuple[np.ndarray, float]],
    transform_matrix: List[List[float]],
) -> None:
    """
    Write several named float grids, sharing one transform, to a single VDB.

    Each grid is copied from a dense (nx, ny, nz) array. Voxels equal to the
    grid's *background* value become inactive and are pruned into tiles, so
    clear air costs almost nothing on disk or in Blender's memory. The
    background is 0 for extinction grids. For Mie diameter grids it is a
    typical cloud value, so that interpolation at a cloud's edge blends toward
    a realistic diameter instead of toward 0 um.

    Args:
        file_path: Destination ``.vdb`` file.
        grids: Grid name -> (dense array in x, y, z order, background value).
        transform_matrix: 4x4 index->world matrix (OpenVDB row-vector form).
    """
    openvdb = _require_openvdb()
    vdb_grids = []
    for grid_name, (dense_array, background_value) in grids.items():
        grid = openvdb.FloatGrid(float(background_value))
        grid.copyFromArray(np.ascontiguousarray(dense_array, dtype=np.float32))
        # Collapse uniform (background) regions into tiles.
        grid.prune()
        grid.name = grid_name
        grid.transform = openvdb.createLinearTransform(matrix=transform_matrix)
        vdb_grids.append(grid)
    openvdb.write(str(file_path), grids=vdb_grids)


# Mixing ratio (kg/kg) above which a cell counts as cloudy when cropping the
# export. Not 0: lossily compressed output (e.g. RAMS ZFP-compressed lite
# files) and advection undershoots leave |q| ~ 1e-10 noise across the whole
# domain, which would defeat the crop. 1e-8 kg/kg of 10 um droplets gives an
# extinction of ~3e-6 m^-1, the same order as the default extinction floor, and
# precipitation at 1e-8 kg/kg is orders of magnitude fainter still.
FOOTPRINT_MIXING_RATIO_THRESHOLD = 1.0e-8


def _cloud_footprint(
    spec, dataset, times: List[Any], coordinate_names: Dict[str, str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Where hydrometeors exist at any render time, as 1-D masks per axis.

    A cell counts when its total mixing ratio exceeds
    ``FOOTPRINT_MIXING_RATIO_THRESHOLD``. With a cloud ``weight``, the total
    is multiplied by the weight first, so a faded-out neighbouring storm does
    not widen the crop or the automatic camera framing.

    Used to crop the exported volume to the cloudy region (plus a margin), so a
    large domain with one storm does not produce mostly-empty VDBs.

    Returns:
        (x_mask, y_mask, z_mask): booleans along each native axis.
    """
    from .grid_utils import select_time

    x_name = coordinate_names["x"]
    y_name = coordinate_names["y"]
    z_name = coordinate_names["z"]
    species = spec.resolve_species(dataset)

    x_mask = np.zeros(dataset.sizes[x_name], dtype=bool)
    y_mask = np.zeros(dataset.sizes[y_name], dtype=bool)
    z_mask = np.zeros(dataset.sizes[z_name], dtype=bool)
    for time in times:
        dataset_at_time = select_time(dataset, time)
        # Total condensate at this time; any positive value marks the cell.
        total_condensate = None
        for one_species in species:
            mixing_ratio = (
                dataset_at_time[one_species.mixing_ratio_variable]
                .clip(min=0.0)
                .fillna(0.0)
            )
            total_condensate = (
                mixing_ratio
                if total_condensate is None
                else total_condensate + mixing_ratio
            )
        # Weighted-out cloud does not count as present.
        weight_variable = spec.geometry.weight
        if weight_variable is not None:
            total_condensate = total_condensate * dataset_at_time[weight_variable].clip(
                0.0, 1.0
            ).fillna(0.0)
        has_condensate = (
            total_condensate > FOOTPRINT_MIXING_RATIO_THRESHOLD
        ).transpose(x_name, y_name, z_name)
        has_condensate = np.asarray(has_condensate.values)
        # Project the 3-D occupancy onto each axis.
        x_mask |= has_condensate.any(axis=(1, 2))
        y_mask |= has_condensate.any(axis=(0, 2))
        z_mask |= has_condensate.any(axis=(0, 1))
    return x_mask, y_mask, z_mask


def _mask_to_slice(axis_mask: np.ndarray, margin: int = 1) -> slice:
    """Index slice covering the True entries of a mask, padded by ``margin``."""
    true_indices = np.nonzero(axis_mask)[0]
    if true_indices.size == 0:
        return slice(0, len(axis_mask))
    first_index = max(int(true_indices[0]) - margin, 0)
    last_index = min(int(true_indices[-1]) + margin, len(axis_mask) - 1)
    return slice(first_index, last_index + 1)


def _build_cloud_entry(
    spec: "VarSpec",
    dataset,
    times: List[Any],
    data_subdir: Path,
    running_bounds: Optional[np.ndarray],
) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    """
    Export a cloud spec to a multi-grid VDB sequence and its manifest entry.

    Steps:

    1. Find the cloudy footprint over all times and crop to it (one-cell
       margin), so the VDB transform is identical for every frame.
    2. Resample the cropped column onto uniform vertical levels
       (:func:`choose_uniform_vertical_spacing`), interpolating the *inputs*
       (mixing ratios, numbers, air density) so the optics are computed
       consistently on the final grid.
    3. Compute every species' optics (:mod:`skyvista.cloud_optics`) and write
       one VDB per frame holding ``extinction_<species>`` grids in km^-1,
       plus ``mie_diameter_<species>`` (um) for species using the Mie phase.
    4. Describe each species' grid and phase function in the manifest; the
       build script turns that into one Volume Scatter closure per species,
       summed in a single shader.
    """
    import warnings

    from .grid_utils import select_time
    from .grids import resolve_coordinates

    geometry = spec.geometry

    # ---- Require a rectilinear grid (1-D coordinate axes)
    coordinate_names = resolve_coordinates(dataset, ["x", "y", "z"])
    x_name = coordinate_names["x"]
    y_name = coordinate_names["y"]
    z_name = coordinate_names["z"]
    for coordinate_name in (x_name, y_name, z_name):
        if coordinate_name not in dataset.coords or dataset[coordinate_name].ndim != 1:
            raise NotImplementedError(
                "Cloud VDB export requires a rectilinear grid with 1-D x/y/z"
                f" coordinates; '{coordinate_name}' is missing or"
                " multi-dimensional. Resample to a rectilinear grid first."
            )

    # ---- Variables the optics need: mixing ratios, numbers, air density
    species = spec.resolve_species(dataset)
    needed_variables = []
    for one_species in species:
        needed_variables.append(one_species.mixing_ratio_variable)
        if one_species.number_variable is not None:
            needed_variables.append(one_species.number_variable)
    air_density_setting = geometry.air_density
    if air_density_setting is None:
        from .cloud_optics import AIR_DENSITY_VARIABLE_CANDIDATES, _find_variable

        for candidate in AIR_DENSITY_VARIABLE_CANDIDATES:
            found_name = _find_variable(dataset, candidate)
            if found_name is not None:
                air_density_setting = found_name
                break
    if isinstance(air_density_setting, str):
        needed_variables.append(air_density_setting)
    # The weight is cropped and resampled along with the optics inputs.
    if geometry.weight is not None:
        needed_variables.append(geometry.weight)
    cloud_dataset = dataset[sorted(set(needed_variables))]

    # ---- 1. Crop to the cloudy footprint over all times
    x_mask, y_mask, z_mask = _cloud_footprint(
        spec, cloud_dataset, times, coordinate_names
    )
    if not z_mask.any():
        warnings.warn(
            f"Cloud spec {spec.name!r}: no hydrometeors at any render time; "
            "exporting empty volumes.",
            stacklevel=2,
        )
    x_slice = _mask_to_slice(x_mask)
    y_slice = _mask_to_slice(y_mask)
    z_slice = _mask_to_slice(z_mask)
    cropped_dataset = cloud_dataset.isel(
        {x_name: x_slice, y_name: y_slice, z_name: z_slice}
    )
    x_coordinates = cropped_dataset[x_name].values.astype(float)
    y_coordinates = cropped_dataset[y_name].values.astype(float)
    native_heights = cropped_dataset[z_name].values.astype(float)

    # ---- 2. Uniform vertical levels covering the cropped column
    if geometry.vertical_spacing is not None:
        vertical_spacing = float(geometry.vertical_spacing)
    else:
        vertical_spacing = choose_uniform_vertical_spacing(
            native_heights,
            z_mask[z_slice],
            horizontal_spacing=min(
                float(np.mean(np.diff(x_coordinates))),
                float(np.mean(np.diff(y_coordinates))),
            ),
        )
    column_depth = native_heights[-1] - native_heights[0]
    if column_depth / vertical_spacing + 1 > MAX_UNIFORM_VERTICAL_LEVELS:
        coarsened_spacing = column_depth / (MAX_UNIFORM_VERTICAL_LEVELS - 1)
        warnings.warn(
            f"Cloud spec {spec.name!r}: a vertical spacing of "
            f"{vertical_spacing:.1f} m needs more than "
            f"{MAX_UNIFORM_VERTICAL_LEVELS} levels; using {coarsened_spacing:.1f} m."
            " Pass vertical_spacing=... to choose explicitly.",
            stacklevel=2,
        )
        vertical_spacing = coarsened_spacing
    n_uniform_levels = int(np.floor(column_depth / vertical_spacing)) + 1
    uniform_heights = native_heights[0] + vertical_spacing * np.arange(n_uniform_levels)

    # ---- Index->world transform: voxel sizes on the diagonal, origin last row
    voxel_size_x = _axis_voxel_size(x_coordinates)
    voxel_size_y = _axis_voxel_size(y_coordinates)
    transform_matrix = [
        [voxel_size_x, 0.0, 0.0, 0.0],
        [0.0, voxel_size_y, 0.0, 0.0],
        [0.0, 0.0, vertical_spacing, 0.0],
        [
            float(x_coordinates[0]),
            float(y_coordinates[0]),
            float(uniform_heights[0]),
            1.0,
        ],
    ]

    # ---- 3. Optics per frame -> one multi-grid VDB per frame
    object_subdir = data_subdir / spec.name
    object_subdir.mkdir(parents=True, exist_ok=True)
    extinction_floor_per_km = geometry.extinction_floor * METERS_PER_KILOMETER
    max_extinction_per_species: Dict[str, float] = {s.name: 0.0 for s in species}
    mie_background_per_species: Dict[str, float] = {}
    for frame_index, time in enumerate(times):
        dataset_at_time = select_time(cropped_dataset, time)
        # Resample every input variable onto the uniform levels.
        resampled_dataset = dataset_at_time.map(
            lambda data_array: interpolate_to_levels(
                data_array, z_name, native_heights, uniform_heights
            )
        ).assign_coords({z_name: uniform_heights})
        # Physical extinction: the shader applies any extinction_scale, once.
        optics_per_species = spec.compute_optics(
            resampled_dataset, z_name, apply_extinction_scale=False
        )

        frame_grids: Dict[str, Tuple[np.ndarray, float]] = {}
        for optics in optics_per_species:
            species_name = optics.species.name
            # Render extinction in km^-1, (x, y, z) order, clear air set to 0.
            extinction_per_km = (
                optics.render_extinction.transpose(x_name, y_name, z_name).values
                * METERS_PER_KILOMETER
            )
            extinction_per_km = np.where(
                extinction_per_km >= extinction_floor_per_km, extinction_per_km, 0.0
            )
            frame_grids[CLOUD_EXTINCTION_GRID_PREFIX + species_name] = (
                extinction_per_km,
                0.0,
            )
            max_extinction_per_species[species_name] = max(
                max_extinction_per_species[species_name],
                float(extinction_per_km.max()),
            )

            # Mie species also carry the per-voxel droplet diameter (um).
            if optics.mie_fit_diameter_microns is not None:
                mie_diameter = optics.mie_fit_diameter_microns.transpose(
                    x_name, y_name, z_name
                ).values
                is_cloudy = extinction_per_km > 0.0
                # Background: the median cloudy diameter of the first frame
                # with cloud, held fixed so every frame shares it.
                if species_name not in mie_background_per_species and is_cloudy.any():
                    mie_background_per_species[species_name] = float(
                        np.median(mie_diameter[is_cloudy])
                    )
                background_diameter = mie_background_per_species.get(species_name, 15.0)
                frame_grids[CLOUD_MIE_DIAMETER_GRID_PREFIX + species_name] = (
                    np.where(is_cloudy, mie_diameter, background_diameter),
                    background_diameter,
                )

        frame_path = object_subdir / f"{spec.name}_{frame_index + 1:04d}.vdb"
        write_multigrid_vdb(frame_path, frame_grids, transform_matrix)

    # ---- Fold the volume into the scene bounds. Vertically we span from the
    # model's lowest level, so the default origin shift floors the figure at
    # the model surface rather than at the cloud base.
    volume_corner_points = np.array(
        [
            [x_coordinates.min(), y_coordinates.min(), float(dataset[z_name].min())],
            [x_coordinates.max(), y_coordinates.max(), float(uniform_heights.max())],
        ],
        dtype=float,
    )
    running_bounds = _accumulate_bounds(running_bounds, volume_corner_points)

    # ---- 4. Manifest: one entry per species that is ever present
    from .cloud_optics import species_summary

    species_entries = []
    for optics in optics_per_species:
        one_species = optics.species
        if max_extinction_per_species[one_species.name] <= 0.0:
            continue
        species_entry: Dict[str, Any] = {
            "name": one_species.name,
            "extinction_grid": CLOUD_EXTINCTION_GRID_PREFIX + one_species.name,
            "max_extinction": max_extinction_per_species[one_species.name],
            "phase_function": one_species.phase_function,
            "anisotropy": optics.render_anisotropy,
            "delta_scaled": one_species.delta_scale_diffraction,
            "microphysics": species_summary(one_species),
        }
        if one_species.phase_function == "mie":
            species_entry["mie_diameter_grid"] = (
                CLOUD_MIE_DIAMETER_GRID_PREFIX + one_species.name
            )
        species_entries.append(species_entry)

    material = spec.appearance.to_blender_material()
    material.update(
        {
            "extinction_grid_units": CLOUD_EXTINCTION_GRID_UNITS,
            # Grid values below this (in grid units) are clear air.
            "clipping": extinction_floor_per_km,
            "species": species_entries,
        }
    )
    object_entry = {
        "name": spec.name,
        "spec_type": "cloud",
        "geometry": {
            "carrier": "vdb_sequence",
            "n_frames": len(times),
            "path_pattern": f"data/{spec.name}/{spec.name}_####.vdb",
            "grid_name": (
                species_entries[0]["extinction_grid"] if species_entries else None
            ),
            "vertical_spacing": vertical_spacing,
        },
        "material": material,
    }
    return object_entry, running_bounds


# =============================================================================
# Dye export (passive tracers as a colored volume)
# =============================================================================
def _build_dye_entry(
    spec: "VarSpec",
    dataset,
    times: List[Any],
    data_subdir: Path,
    running_bounds: Optional[np.ndarray],
) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    """
    Export a dye spec to a two-grid VDB sequence and its manifest entry.

    Steps:

    1. Compute the dyed fraction at every render time and crop the export to
       where it exceeds ``fraction_floor`` at any time (one-cell margin), so
       the VDB transform is identical for every frame.
    2. Resample the cropped column onto uniform vertical levels. The fraction
       and the fraction-weighted label are interpolated, and the label is
       their ratio, so the label stays a proper dye-weighted mean.
    3. Write one VDB per frame holding ``dye_fraction`` (0 to 1) and
       ``dye_label`` (in the units of the tracer values). The label's
       background is the frame-independent dye-weighted mean label, so a
       sample at the dye's edge blends toward a typical color, not toward 0.
    4. Describe the material: the build script turns the fraction into
       extinction and the label into the dye's color.
    """
    from .grid_utils import select_time
    from .grids import resolve_coordinates
    from .varspec import DYE_FRACTION_VARIABLE, DYE_LABEL_VARIABLE

    geometry = spec.geometry
    appearance = spec.appearance

    # ---- Require a rectilinear grid (1-D coordinate axes)
    coordinate_names = resolve_coordinates(dataset, ["x", "y", "z"])
    x_name = coordinate_names["x"]
    y_name = coordinate_names["y"]
    z_name = coordinate_names["z"]
    for coordinate_name in (x_name, y_name, z_name):
        if coordinate_name not in dataset.coords or dataset[coordinate_name].ndim != 1:
            raise NotImplementedError(
                "Dye VDB export requires a rectilinear grid with 1-D x/y/z"
                f" coordinates; '{coordinate_name}' is missing or"
                " multi-dimensional. Resample to a rectilinear grid first."
            )
    tracer_dataset = dataset[list(geometry.tracer_varnames)]

    # ---- 1. Crop to where there is dye at any time
    x_mask = np.zeros(tracer_dataset.sizes[x_name], dtype=bool)
    y_mask = np.zeros(tracer_dataset.sizes[y_name], dtype=bool)
    z_mask = np.zeros(tracer_dataset.sizes[z_name], dtype=bool)
    for time in times:
        fraction, _ = spec.dye_fields(select_time(tracer_dataset, time))
        has_dye = np.asarray(
            (fraction > geometry.fraction_floor)
            .transpose(x_name, y_name, z_name)
            .values
        )
        # Project the 3-D occupancy onto each axis.
        x_mask |= has_dye.any(axis=(1, 2))
        y_mask |= has_dye.any(axis=(0, 2))
        z_mask |= has_dye.any(axis=(0, 1))
    cropped_dataset = tracer_dataset.isel(
        {
            x_name: _mask_to_slice(x_mask),
            y_name: _mask_to_slice(y_mask),
            z_name: _mask_to_slice(z_mask),
        }
    )
    x_coordinates = cropped_dataset[x_name].values.astype(float)
    y_coordinates = cropped_dataset[y_name].values.astype(float)
    native_heights = cropped_dataset[z_name].values.astype(float)

    # ---- 2. Uniform vertical levels, at the finest spacing among dyed levels
    vertical_spacing = choose_uniform_vertical_spacing(
        native_heights,
        z_mask[_mask_to_slice(z_mask)],
        horizontal_spacing=min(
            float(np.mean(np.diff(x_coordinates))),
            float(np.mean(np.diff(y_coordinates))),
        ),
    )
    column_depth = native_heights[-1] - native_heights[0]
    n_uniform_levels = int(np.floor(column_depth / vertical_spacing)) + 1
    uniform_heights = native_heights[0] + vertical_spacing * np.arange(n_uniform_levels)

    # ---- Index->world transform: voxel sizes on the diagonal, origin last row
    transform_matrix = [
        [_axis_voxel_size(x_coordinates), 0.0, 0.0, 0.0],
        [0.0, _axis_voxel_size(y_coordinates), 0.0, 0.0],
        [0.0, 0.0, vertical_spacing, 0.0],
        [
            float(x_coordinates[0]),
            float(y_coordinates[0]),
            float(uniform_heights[0]),
            1.0,
        ],
    ]

    # ---- 3. Dye fields per frame, resampled, then written
    frame_fields: List[Tuple[np.ndarray, np.ndarray]] = []
    total_dye = 0.0
    total_weighted_label = 0.0
    for time in times:
        fraction, label = spec.dye_fields(select_time(cropped_dataset, time))
        # Interpolate the fraction and the fraction-weighted label.
        weighted_label = (fraction * label).fillna(0.0)
        fraction_uniform = interpolate_to_levels(
            fraction, z_name, native_heights, uniform_heights
        )
        weighted_label_uniform = interpolate_to_levels(
            weighted_label, z_name, native_heights, uniform_heights
        )
        fraction_values = np.asarray(
            fraction_uniform.transpose(x_name, y_name, z_name).values, dtype=float
        )
        weighted_label_values = np.asarray(
            weighted_label_uniform.transpose(x_name, y_name, z_name).values,
            dtype=float,
        )
        # Below the floor the voxel is clear air.
        is_dyed = fraction_values > geometry.fraction_floor
        fraction_values = np.where(is_dyed, fraction_values, 0.0)
        frame_fields.append((fraction_values, weighted_label_values))
        # Accumulate the dye-weighted mean label for the background value.
        total_dye += float(fraction_values.sum())
        total_weighted_label += float(weighted_label_values[is_dyed].sum())

    label_values = spec.tracer_label_values()
    background_label = (
        total_weighted_label / total_dye
        if total_dye > 0
        else 0.5 * (min(label_values) + max(label_values))
    )
    object_subdir = data_subdir / spec.name
    object_subdir.mkdir(parents=True, exist_ok=True)
    for frame_index, (fraction_values, weighted_label_values) in enumerate(
        frame_fields
    ):
        # Mean label of the dyed air; the background elsewhere.
        is_dyed = fraction_values > 0.0
        label_values_on_grid = np.where(
            is_dyed,
            weighted_label_values / np.where(is_dyed, fraction_values, 1.0),
            background_label,
        )
        write_multigrid_vdb(
            object_subdir / f"{spec.name}_{frame_index + 1:04d}.vdb",
            {
                DYE_FRACTION_VARIABLE: (fraction_values, 0.0),
                DYE_LABEL_VARIABLE: (label_values_on_grid, background_label),
            },
            transform_matrix,
        )

    # ---- Fold the volume into the scene bounds
    volume_corner_points = np.array(
        [
            [x_coordinates.min(), y_coordinates.min(), uniform_heights.min()],
            [x_coordinates.max(), y_coordinates.max(), uniform_heights.max()],
        ],
        dtype=float,
    )
    running_bounds = _accumulate_bounds(running_bounds, volume_corner_points)

    # ---- 4. Material: extinction from the fraction, color from the label
    _, colormap_name = resolve_colormap(appearance.cmap or DEFAULT_COLORMAP_NAME)
    colormap_lut_path = write_colormap_lut(data_subdir.parent, colormap_name)
    color_limits = appearance.clim or (min(label_values), max(label_values))
    material = {
        "type": "dye",
        "fraction_grid": DYE_FRACTION_VARIABLE,
        "label_grid": DYE_LABEL_VARIABLE,
        # Extinction (m^-1) of undiluted dye; the shader multiplies it by the
        # fraction.
        "extinction_at_full_strength": appearance.extinction_per_km
        / METERS_PER_KILOMETER,
        "self_illumination": appearance.self_illumination,
        # Grid values below this are clear air.
        "clipping": geometry.fraction_floor,
        "coloring": {
            "mode": "scalar_ramp",
            "attribute": DYE_LABEL_VARIABLE,
            "cmap": colormap_name,
            "clim": [float(limit) for limit in color_limits],
            "colormap_lut": colormap_lut_path,
            "label": appearance.scalar_bar_title or "dye",
        },
    }
    object_entry = {
        "name": spec.name,
        "spec_type": "dye",
        "geometry": {
            "carrier": "vdb_sequence",
            "n_frames": len(times),
            "path_pattern": f"data/{spec.name}/{spec.name}_####.vdb",
            "grid_name": DYE_FRACTION_VARIABLE,
            "vertical_spacing": vertical_spacing,
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
                float(
                    center[0] + distance * math.cos(elevation) * math.cos(orbit_azimuth)
                ),
                float(
                    center[1] + distance * math.cos(elevation) * math.sin(orbit_azimuth)
                ),
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

    explicit_view = (
        camera_config.location is not None and camera_config.look_at is not None
    )
    if not keyframes and explicit_view:
        # A pinned static view: use the given position and aim point verbatim.
        keyframes = [
            {
                "frame": frame_start,
                "location": [float(value) for value in camera_config.location],
                "look_at": [float(value) for value in camera_config.look_at],
                "up": up_vector,
            }
        ]

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

    # A camera below the data almost always means it is under the ground
    # plane. The elevation angle is measured from the scene's center, so a
    # large scene puts a slightly negative angle kilometers below it.
    lowest_camera_height = min(keyframe["location"][2] for keyframe in keyframes)
    if lowest_camera_height < float(bounds_min[2]):
        warnings.warn(
            f"The camera is at z = {lowest_camera_height:.0f}, below the lowest "
            f"exported data (z = {float(bounds_min[2]):.0f}), so it is probably "
            "under the ground plane and sees nothing. Raise "
            "camera.elevation_deg or reduce camera.distance_factor; the "
            "elevation is measured from the center of the scene.",
            stacklevel=3,
        )

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
    render: bool = False,
    blender_executable: Optional[PathLike] = None,
    quiet: bool = False,
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
        render: If True, also render the animation to ``<bundle>/render/`` in the
            same headless Blender run. Implies ``build``.
        blender_executable: Blender command used when ``build``/``render`` is
            True.
        quiet: Write Blender's console output to ``<bundle>/blender.log``
            instead of the terminal.

    Returns:
        The bundle directory path.
    """
    if config is None:
        from .varspec import CloudSpec

        # Physically rendered clouds need the radiometric world and many volume
        # bounces; every other scene keeps the general-purpose defaults.
        scene_has_clouds = any(isinstance(spec, CloudSpec) for _, spec in scene._specs)
        config = (
            BlenderExportConfig.for_clouds()
            if scene_has_clouds
            else BlenderExportConfig()
        )
    bundle_dir = Path(path)
    data_dir = bundle_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Optical depth cannot stay physical under an anisotropic stretch: Cycles
    # compensates volume density for one uniform scale factor only.
    if config.transform.z_exaggeration != 1.0:
        from .varspec import CloudSpec

        if any(isinstance(spec, CloudSpec) for _, spec in scene._specs):
            import warnings

            warnings.warn(
                "z_exaggeration != 1 makes physically rendered clouds optically "
                "thicker along the vertical than the data imply (Cycles scales "
                "volume density by a single factor). Use z_exaggeration=1 for "
                "radiometrically faithful clouds.",
                stacklevel=2,
            )

    render_times = times if times is not None else scene._get_all_times()
    n_frames = len(render_times)
    # A hold renders its one data time over many video frames.
    hold = config.hold
    if hold is not None and n_frames > 1:
        raise ValueError(
            f"config.hold renders one data time, but the scene has {n_frames}; "
            "pass times=[t] to choose one."
        )
    n_video_frames = hold.n_frames if hold is not None else n_frames

    object_entries: List[Dict[str, Any]] = []
    running_bounds: Optional[np.ndarray] = None
    for dataset, spec in scene._specs:
        object_entry, running_bounds = _build_object_entry(
            spec, dataset, render_times, data_dir, config.fps, running_bounds
        )
        object_entries.append(object_entry)

    # ---- Resolve the transform's origin_shift (default: center x/y, floor z)
    # We center the *horizontal* axes on the data's midpoint (nice for orbiting
    # and camera framing) but anchor the *vertical* axis to the data's floor
    # (z_min), NOT its center. Subtracting the z-center would put the middle of
    # the domain at Blender z=0, dropping the entire lower half of the data below
    # the ground plane (trajectories/isosurfaces below z=0). For atmospheric data
    # z=0 is the surface, so the figure should rest *on* the ground: mapping the
    # data floor to Blender z=0 guarantees nothing sinks underground regardless
    # of where the vertical coordinate starts (0 m, 500 m, ...).
    transform = config.transform
    if transform.origin_shift is None and running_bounds is not None:
        bounds_min = running_bounds[0]
        bounds_max = running_bounds[1]
        horizontal_center = (bounds_min + bounds_max) / 2.0
        resolved_origin_shift: Optional[Tuple[float, float, float]] = (
            float(horizontal_center[0]),  # x -> centered
            float(horizontal_center[1]),  # y -> centered
            float(bounds_min[2]),  # z -> floored at data z_min
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
            "origin_shift": (
                list(resolved_origin_shift)
                if resolved_origin_shift is not None
                else [0.0, 0.0, 0.0]
            ),
            "scale": transform.scale,
            "z_exaggeration": transform.z_exaggeration,
        },
        "time": {
            "fps": config.fps,
            "frame_start": config.frame_start,
            "frame_end": config.frame_start + max(n_video_frames - 1, 0),
            "data_times": [_json_safe_time(t) for t in render_times],
        },
        # Model time (s) at each video frame, for clock-driven motion such as
        # streamline markers. Absent: the clock stands at zero.
        "flow_clock": (
            {"keyframes": hold.clock_keyframes(config.frame_start)}
            if hold is not None
            else None
        ),
        "render": config.render.to_dict(),
        "world": config.world.to_dict(),
        "camera": _compute_camera_manifest(
            config.camera,
            running_bounds,
            render_times,
            config.frame_start,
            scene,
        ),
        "objects": object_entries,
        # Data-space bounding box of everything exported, [[min], [max]];
        # the build script sizes scene-wide elements (the haze box) from it.
        "bounds": (
            [[float(value) for value in corner] for corner in running_bounds]
            if running_bounds is not None
            else None
        ),
        "annotations": {
            "title": scene.title,
            "show_grid": scene.show_grid,
            "background": scene.background,
            "colorbars": colorbar_annotations,
            "overlay_colorbars": config.overlay_colorbars,
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

    # ---- Optionally invoke Blender to assemble the .blend (and render) now.
    #      render implies build -- both happen in the one headless Blender run.
    if build or render:
        _launch_blender_build(
            blender_executable,
            build_script_path,
            bundle_dir,
            render=render,
            log_path=bundle_dir / "blender.log" if quiet else None,
        )

    return bundle_dir


def _copy_build_script(bundle_dir: Path) -> Path:
    """Copy the packaged Blender build script into the bundle."""
    import shutil

    source = Path(__file__).parent / "blender_assets" / "build_scene.py"
    destination = bundle_dir / "build_scene.py"
    shutil.copyfile(source, destination)
    return destination


@dataclass(frozen=True)
class BlenderRunner:
    """
    How skyvista starts Blender to run a script: as the standalone program, or
    as the ``bpy`` module in a fresh Python process.

    Both modes run the same scripts with the same arguments, so callers build
    every command through :meth:`script_command` and never care which mode is
    in use:

    * **Program mode** runs ``blender --background [file.blend] --python
      script.py -- <arguments>``. ``--python-exit-code 1`` makes an exception
      in the script exit non-zero, as it does in module mode.
    * **Module mode** runs ``python run_in_bpy.py <file.blend or ""> script.py
      -- <arguments>`` with the interpreter that has ``bpy`` installed (the
      ``blender`` extra). ``run_in_bpy.py`` opens the file and gives the
      script the same ``sys.argv`` the program would.

    Module mode always uses a separate process, never ``import bpy`` here:
    ``bpy`` bundles its own copies of libraries (OpenVDB, OpenImageIO, ...)
    that can clash with the ones VTK and the export writers load.

    Attributes:
        executable: Absolute path of the Blender program, or of the Python
            interpreter that imports ``bpy``.
        uses_bpy_module: True for module mode.
    """

    executable: str
    uses_bpy_module: bool

    def script_command(
        self,
        script_path: PathLike,
        script_arguments: Sequence[str] = (),
        blend_path: Optional[PathLike] = None,
    ) -> List[str]:
        """
        The command that runs ``script_path`` inside Blender.

        Args:
            script_path: A Blender script that reads its arguments after
                ``--`` in ``sys.argv``.
            script_arguments: Arguments the script sees after ``--``.
            blend_path: A ``.blend`` to open before the script runs; None
                starts from Blender's default scene.
        """
        script_arguments = [str(argument) for argument in script_arguments]
        if self.uses_bpy_module:
            # The launcher takes the file and script positionally, before "--".
            launcher_path = Path(__file__).parent / "blender_assets" / "run_in_bpy.py"
            return [
                self.executable,
                str(launcher_path),
                "" if blend_path is None else str(blend_path),
                str(script_path),
                "--",
                *script_arguments,
            ]
        command = [self.executable, "--background"]
        # The program opens a .blend given before --python.
        if blend_path is not None:
            command.append(str(blend_path))
        return command + [
            "--python-exit-code",
            "1",
            "--python",
            str(script_path),
            "--",
            *script_arguments,
        ]


def _bpy_module_is_installed() -> bool:
    """
    Whether this interpreter can import ``bpy``, checked without importing it.

    ``find_spec`` only locates the package. Importing it would load all of
    Blender into this process, which is slow and is what module mode avoids.
    """
    import importlib.util

    return importlib.util.find_spec("bpy") is not None


def _resolve_blender_runner(explicit: Optional[PathLike] = None) -> BlenderRunner:
    """
    Decide how to run Blender, in priority order:

    1. an explicit Blender program passed by the caller
       (``blender_executable=...``);
    2. the Blender program named by the ``SKYVISTA_BLENDER`` environment
       variable;
    3. the ``bpy`` module, if this interpreter has it (``pip install
       skyvista[blender]``), run by this same interpreter
       (``sys.executable``);
    4. ``blender`` on ``PATH``.

    The two explicit choices come first, so a user can always pin a specific
    Blender version over the installed module.

    Program paths are returned **absolute**, so the subprocess does not depend
    on inheriting ``PATH``. That inheritance is what usually breaks: an IDE or
    a Jupyter kernel is often launched with a minimal ``PATH`` that does not
    include wherever Blender lives (e.g. ``~/programs``), even when the shell
    profile adds it.

    Raises ``FileNotFoundError`` with actionable guidance if nothing resolves.
    """
    import os
    import shutil
    import sys

    # Explicit choices of the Blender program, highest priority first.
    explicit_candidates = [
        str(explicit) if explicit else None,
        os.environ.get("SKYVISTA_BLENDER"),
    ]
    for candidate in explicit_candidates:
        if not candidate:
            continue
        # shutil.which handles both a bare name (searched on PATH) and an
        # absolute/relative path (returned if it exists and is executable).
        resolved = shutil.which(candidate)
        if resolved:
            return BlenderRunner(executable=resolved, uses_bpy_module=False)

    # The bpy module from the blender extra, run by this same interpreter.
    if _bpy_module_is_installed():
        return BlenderRunner(executable=sys.executable, uses_bpy_module=True)

    # Last resort: a Blender program on PATH.
    resolved = shutil.which("blender")
    if resolved:
        return BlenderRunner(executable=resolved, uses_bpy_module=False)

    tried = [candidate for candidate in explicit_candidates if candidate]
    raise FileNotFoundError(
        "Could not find Blender. Provide it in any of these ways:\n"
        "  - pip install skyvista[blender], which installs Blender as the\n"
        "    'bpy' module (Python 3.13; Linux x86-64, Apple-silicon macOS,\n"
        "    Windows)\n"
        "  - pass blender_executable='/path/to/blender' to"
        " Scene.to_blender(...)\n  - export SKYVISTA_BLENDER=/path/to/blender"
        " in the environment that\n    launches Python (note: an IDE/Jupyter"
        " kernel may not read your\n    shell profile, so setting it there is"
        " more reliable than PATH)\n  - put 'blender' on PATH\nTried:"
        f" {tried + ['bpy module', 'blender']}"
    )


def _launch_blender_build(
    blender_executable: Union[None, PathLike, BlenderRunner],
    build_script_path: Path,
    bundle_dir: Path,
    render: bool = False,
    log_path: Optional[Path] = None,
) -> None:
    """
    Run the build script in Blender headlessly.

    The script runs with ``-- <bundle> [--render]``; with ``render`` the build
    also renders the animation to ``<bundle>/render/``. Blender is the
    program or the ``bpy`` module, as :func:`_resolve_blender_runner`
    decides from ``blender_executable``. A caller that has already resolved
    a :class:`BlenderRunner` may pass it instead.

    Blender's console output goes to the terminal, or with ``log_path`` to
    that file (overwritten). A failed run then raises a ``RuntimeError``
    whose message ends with the log's last lines, so the cause is visible
    without opening the file.
    """
    import shlex
    import subprocess

    runner = (
        blender_executable
        if isinstance(blender_executable, BlenderRunner)
        else _resolve_blender_runner(blender_executable)
    )
    build_arguments = [str(bundle_dir)]
    if render:
        build_arguments.append("--render")
    command = runner.script_command(build_script_path, build_arguments)
    if log_path is None:
        print("Running:", shlex.join(command))
        subprocess.run(command, check=True)
        return

    with open(log_path, "w") as log_file:
        completed = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        # The end of the log usually holds the error.
        last_lines = Path(log_path).read_text(errors="replace").splitlines()[-30:]
        raise RuntimeError(
            f"Blender exited with code {completed.returncode}; log: {log_path}\n"
            + "\n".join(last_lines)
        )


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
