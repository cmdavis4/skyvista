"""
Appearance classes for skyvista.

Appearance classes specify *how* visualizations look - colors, opacity,
colormaps, etc. They are renderer-agnostic and can be converted to
PyVista kwargs.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Appearance:
    """
    Renderer-agnostic visual properties.

    This is the base appearance class with properties common to all
    visualization types.

    Attributes:
        color: Solid color (name or hex). Mutually exclusive with cmap.
        opacity: Opacity from 0.0 (transparent) to 1.0 (opaque)
        cmap: Colormap name for scalar coloring
        clim: Color limits as (min, max) tuple
        show_scalar_bar: Whether to show a scalar bar
        scalar_bar_title: Title for the scalar bar
        shader: Named surface-material preset for the Blender export backend
            (e.g. "matte", "glossy", "metal", "glow", "emissive"). ``None`` uses
            the default matte look. See :mod:`skyvista.shaders` for the full set;
            "glow" is the emissive, bloom-haloed NCAR "fountain" look. Ignored by
            the PyVista renderer, which has no equivalent node-based shading.
        shader_overrides: Values that replace fields of the resolved shader
            preset, e.g. ``{"emission_strength": 1.0}`` to make a "self_lit"
            surface fully self-illuminated. Keys must be fields of the preset
            (see :class:`skyvista.shaders.ShaderPreset`).
    """

    color: Optional[str] = None
    opacity: float = 1.0
    cmap: Optional[str] = None
    clim: Optional[Tuple[float, float]] = None
    show_scalar_bar: bool = True
    scalar_bar_title: Optional[str] = None
    shader: Optional[str] = None
    shader_overrides: Optional[Dict[str, Any]] = None

    def to_pyvista_kwargs(self) -> Dict[str, Any]:
        """Convert appearance to PyVista add_mesh kwargs."""
        kwargs: Dict[str, Any] = {}

        if self.color is not None:
            kwargs["color"] = self.color

        if self.opacity != 1.0:
            kwargs["opacity"] = self.opacity

        if self.cmap is not None:
            kwargs["cmap"] = self.cmap

        if self.clim is not None:
            kwargs["clim"] = self.clim

        if self.show_scalar_bar and self.scalar_bar_title:
            kwargs["scalar_bar_args"] = {"title": self.scalar_bar_title}

        return kwargs

    def to_blender_material(self) -> Dict[str, Any]:
        """
        Convert appearance to a renderer-agnostic Blender material description.

        This is the Blender analog of :meth:`to_pyvista_kwargs`: it returns the
        appearance-derived parts of the manifest ``material`` block (shader,
        opacity, and a solid color if one is set). The coloring *mode*
        (baked vertex colors vs. solid) is decided by the exporter, which has
        the extra context of whether a scalar field is present.

        The ``shader`` block is resolved from the named preset in ``self.shader``
        (default: a clean matte surface); a Blender user can still override any
        of it on the imported material.
        """
        from .shaders import resolve_shader

        shader_block = resolve_shader(self.shader)
        for key, value in (self.shader_overrides or {}).items():
            if key not in shader_block:
                raise ValueError(
                    f"Unknown shader field {key!r}; valid fields are "
                    f"{', '.join(sorted(shader_block))}."
                )
            shader_block[key] = value
        material: Dict[str, Any] = {
            "type": "surface",
            "opacity": self.opacity,
            "shader": shader_block,
        }
        return material


@dataclass
class ContourAppearance(Appearance):
    """
    Appearance specific to isosurface rendering.

    Attributes:
        style: Rendering style - "surface", "wireframe", or "points"
    """

    style: str = "surface"

    def to_pyvista_kwargs(self) -> Dict[str, Any]:
        kwargs = super().to_pyvista_kwargs()
        if self.style != "surface":
            kwargs["style"] = self.style
        return kwargs


@dataclass
class VolumeAppearance(Appearance):
    """
    Appearance specific to volume rendering.

    Attributes:
        opacity_transfer: Custom opacity transfer function as list of values
        mapper: PyVista volume mapper ("smart", "fixed_point", "gpu", etc.)
        opacity_unit_distance: Controls opacity accumulation through volume
    """

    opacity_transfer: Optional[List[float]] = None
    mapper: str = "smart"
    opacity_unit_distance: Optional[float] = None

    def to_pyvista_kwargs(self) -> Dict[str, Any]:
        kwargs = super().to_pyvista_kwargs()

        # For volumes, opacity can be a transfer function
        if self.opacity_transfer is not None:
            kwargs["opacity"] = self.opacity_transfer
        elif self.opacity != 1.0:
            kwargs["opacity"] = self.opacity

        kwargs["mapper"] = self.mapper

        if self.opacity_unit_distance is not None:
            kwargs["opacity_unit_distance"] = self.opacity_unit_distance

        return kwargs


@dataclass
class VectorAppearance(Appearance):
    """
    Appearance specific to vector field glyphs.

    Attributes:
        glyph_type: Type of glyph - "arrow", "cone", "sphere". The Blender
            export draws "arrow" and "cone"; "sphere" falls back to "arrow".
        blender_instancing: How the Blender export carries the glyphs. True
            (default) writes only each glyph's position, vector and color, and
            Blender instances one arrow per point with geometry nodes, so files
            stay small and arrow size/shape can be changed in Blender without
            re-exporting. False bakes PyVista's full glyph meshes into Alembic.
        glyph_width: Shaft diameter of instanced Blender arrows, in data
            length units (e.g. meters). None scales each arrow uniformly with
            its length (PyVista's behavior), so weak vectors get thin, hard to
            see arrows. A width keeps every shaft that wide and every head the
            same size, and only the length follows the magnitude (matplotlib
            quiver's convention), which reads better in a distant 3-D view.
    """

    glyph_type: str = "arrow"
    blender_instancing: bool = True
    glyph_width: Optional[float] = None


@dataclass
class StreamlineAppearance(Appearance):
    """
    Appearance of streamlines drawn as tubes.

    Attributes:
        tube_radius: Tube radius in data length units (e.g. meters).
        arrow_spacing: Distance (data units) between direction markers -- cone
            heads along each tube pointing downstream. None draws plain tubes.
            Markers matter in a still: a bare streamline does not say which
            way the air moves. Blender only (PyVista draws plain tubes).
        arrow_head_radius_factor: Cone base radius / tube radius.
        arrow_head_length_factor: Cone length / tube radius.
        arrow_travel_time: Travel time (s) between neighbouring markers. When
            set, markers are spaced by the time air takes to move between
            them (not by ``arrow_spacing``) and move along the streamline at
            the local wind speed as the Blender flow clock advances, so they
            show the flow in motion: closely spaced and slow where the air is
            slow, widely spaced and fast where it is fast. The flow clock is
            the model time of each frame of a ``BlenderAnimation``, or the
            hold clock of a still scene (``BlenderExportConfig.hold``). Keep
            it several times the clock advance per frame, or the motion
            aliases (markers seem to stand still or run backwards). Where the
            air is slow, markers come closer than two cone lengths apart; they
            are shrunk in proportion to the gap there, so they never merge
            into a solid tube and their size also shows the speed. Blender
            only.
        arrow_minimum_speed: Speed floor (field units, e.g. m/s) used for the
            travel time, so stagnant air does not stall markers forever.
    """

    tube_radius: float = 50.0
    arrow_spacing: Optional[float] = None
    arrow_head_radius_factor: float = 3.0
    arrow_head_length_factor: float = 8.0
    arrow_travel_time: Optional[float] = None
    arrow_minimum_speed: float = 0.5


@dataclass
class TrajectoryAppearance(Appearance):
    """
    Appearance specific to trajectory rendering.

    Attributes:
        style: Rendering style - "tube" or "particle"
        silhouettes: Whether to add silhouette effect
        blender_instancing: How the Blender export carries the trajectories.
            True (default) writes only points: for the "particle" style each
            particle's position and color per frame, which Blender turns into
            a native point cloud that Cycles renders as exact spheres; for
            the "tube" style each trail's points, which Blender rebuilds as
            curves and sweeps into tubes (tapered by ``trail_taper``) with a
            cone at each head. No meshes are stored either way, so 10^4-10^6
            tracks stay cheap. False bakes PyVista's sphere or tube meshes
            into Alembic instead.
    """

    style: str = "tube"
    silhouettes: bool = False
    blender_instancing: bool = True

    def to_pyvista_kwargs(self) -> Dict[str, Any]:
        kwargs = super().to_pyvista_kwargs()
        # Silhouettes would be handled separately in render_to_plotter
        return kwargs


@dataclass
class CloudAppearance(Appearance):
    """
    Appearance of a physically rendered cloud.

    A cloud has no colormap: its color and brightness come from the optics of
    the hydrometeors and from the lighting. This class therefore only holds a
    deliberate departure from physics, for exploration. Leave it at the
    default for a scientifically faithful render.

    Attributes:
        extinction_scale: Multiplier on every species' extinction. 1.0 is the
            physical value; less than 1 makes clouds more transparent.
        cutaway: Remove the cloud on one side of a plane, to show what lies
            inside it, as ``(point, normal)`` in data coordinates. Cloud is
            removed wherever ``(position - point) . normal > 0``, i.e. on the
            side the normal points to. For example ``((0, 0, 0), (0, -1, 0))``
            removes the ``y < 0`` half. The remaining half is rendered with
            unchanged optics, and light reaches it through the cut face as it
            would in a real cut. None keeps the whole cloud.
        show_scalar_bar: Off by default: a physically rendered cloud has no
            colormap, so a scalar bar would describe nothing on screen.
    """

    extinction_scale: float = 1.0
    cutaway: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = (
        None
    )
    show_scalar_bar: bool = False

    def to_blender_material(self) -> Dict[str, Any]:
        """
        Describe the cloud material for the Blender manifest.

        The per-species grids and phase functions are filled in by the
        exporter, which has computed them; this contributes only the
        appearance-level part.
        """
        material: Dict[str, Any] = {
            "type": "cloud",
            "extinction_scale": self.extinction_scale,
        }
        if self.cutaway is not None:
            cut_point, cut_normal = self.cutaway
            material["cutaway"] = {
                "point": [float(value) for value in cut_point],
                "normal": [float(value) for value in cut_normal],
            }
        return material


@dataclass
class DyeAppearance(Appearance):
    """
    Appearance of a passive-tracer dye volume.

    The dye is a colored participating medium. At full strength (dyed
    fraction 1) its extinction is ``extinction_per_km`` per kilometer of
    path, and it scales linearly with the dyed fraction. Its single-scattering
    albedo in each color channel is the colormap color of the cell's dye
    label. So the dye scatters its own color and absorbs the rest, like ink
    in water: it looks colored in reflected light, and it darkens and tints
    what lies behind it.

    Attributes:
        extinction_per_km: Optical depth of one kilometer of undiluted dye.
        self_illumination: Emission that keeps the dye colored in shadow, as
            a fraction of the display's white level. Each voxel emits in
            proportion to its extinction, so an optically thick region of dye
            shows its color at this brightness however it is lit, and a thin
            wisp stays faint. 0 is a purely physical (lit) dye.
        show_scalar_bar: Off by default. The Blender figure scripts draw their
            own colorbars.
    """

    extinction_per_km: float = 2.0
    self_illumination: float = 0.0
    show_scalar_bar: bool = False
