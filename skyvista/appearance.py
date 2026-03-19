"""
Appearance classes for skyvista.

Appearance classes specify *how* visualizations look - colors, opacity,
colormaps, etc. They are renderer-agnostic and can be converted to
PyVista kwargs or Blender material configurations.
"""

from dataclasses import dataclass, field
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
        material_preset: Preset name for Blender export (e.g., "cloud", "rain")
    """

    color: Optional[str] = None
    opacity: float = 1.0
    cmap: Optional[str] = None
    clim: Optional[Tuple[float, float]] = None
    show_scalar_bar: bool = False
    scalar_bar_title: Optional[str] = None
    material_preset: Optional[str] = None

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

    def to_blender_config(self) -> Dict[str, Any]:
        """Convert appearance to Blender material/object configuration."""
        config: Dict[str, Any] = {}

        if self.material_preset:
            config["material"] = self.material_preset

        if self.color:
            config["base_color"] = self.color

        if self.opacity < 1.0:
            config["alpha"] = self.opacity

        if self.cmap:
            config["colormap"] = self.cmap

        if self.clim:
            config["clim"] = self.clim

        return config


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
        glyph_type: Type of glyph - "arrow", "cone", "sphere"
    """

    glyph_type: str = "arrow"


@dataclass
class TrajectoryAppearance(Appearance):
    """
    Appearance specific to trajectory rendering.

    Attributes:
        style: Rendering style - "tube" or "particle"
        silhouettes: Whether to add silhouette effect
    """

    style: str = "tube"
    silhouettes: bool = False

    def to_pyvista_kwargs(self) -> Dict[str, Any]:
        kwargs = super().to_pyvista_kwargs()
        # Silhouettes would be handled separately in render_to_plotter
        return kwargs

    def to_blender_config(self) -> Dict[str, Any]:
        config = super().to_blender_config()
        config["style"] = self.style
        if self.silhouettes:
            config["silhouettes"] = True
        return config
