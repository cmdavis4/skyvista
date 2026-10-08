"""
Geometry classes for skyvista.

Geometry classes specify *what* to extract from data - the shape and structure
of the visualization, independent of how it looks (appearance).
"""

from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    from .cloud_optics import HydrometeorSpecies


@dataclass
class Geometry(ABC):
    """Base class for geometry extraction specifications."""

    pass


@dataclass
class ContourGeometry(Geometry):
    """
    Specifies how to extract isosurfaces from gridded data.

    Attributes:
        varname: Variable name to contour
        isosurfaces: List of isosurface values (None = auto)
        scalar: Variable to sample onto surface for coloring (None = use varname)
        individual_meshes: If True, create separate mesh for each isosurface
    """

    varname: str
    isosurfaces: Optional[List[float]] = None
    scalar: Optional[str] = None
    individual_meshes: bool = False


@dataclass
class VolumeGeometry(Geometry):
    """
    Specifies how to render a volume.

    Attributes:
        varname: Variable name for volume rendering
        threshold: Optional (min, max) tuple to clip values (None = no threshold)
    """

    varname: str
    threshold: Optional[Tuple[Optional[float], Optional[float]]] = None


@dataclass
class VectorGeometry(Geometry):
    """
    Specifies how to create vector field glyphs.

    Attributes:
        varname: Name for the combined vector field
        u_varname: Variable name for u (x) component
        v_varname: Variable name for v (y) component
        w_varname: Variable name for w (z) component
        scale_by: Variable to scale arrows by (None = use vector magnitude)
        factor: Scale factor for glyph size (None = auto-calculate)
        tolerance: Point merging tolerance (None = default, 0 = no merging)
    """

    varname: str
    u_varname: str = "UC"
    v_varname: str = "VC"
    w_varname: str = "WC"
    scale_by: Optional[str] = None
    factor: Optional[float] = None
    tolerance: Optional[float] = None


@dataclass
class SliceGeometry(Geometry):
    """
    Specifies how to extract a 2D slice from gridded data.

    Attributes:
        varname: Variable name to slice
        slice_dim: Dimension to slice along ('x', 'y', or 'z')
        slice_value: Value at which to slice (None = first index)
        slice_method: Selection method ('nearest' or 'interp')
    """

    varname: str
    slice_dim: str = "z"
    slice_value: Optional[float] = 0
    slice_method: str = "nearest"


@dataclass
class TrajectoryGeometry(Geometry):
    """
    Specifies trajectory rendering geometry.

    Note: Trajectories don't extract from gridded data the same way as other
    geometry types - they use Lagrangian particle positions over time.

    Attributes:
        scalar: Variable to sample onto trajectories for coloring
        tube_radius: Radius of trajectory tubes in data units
        head_length_frac: Arrow head length as fraction of tube radius
        head_radius_frac: Arrow head radius as fraction of tube radius
        tube_resolution: Number of sides on the tube
        head_radial_resolution: Resolution of arrow head cone
    """

    scalar: Optional[str] = None
    tube_radius: float = 70
    head_length_frac: float = 10
    head_radius_frac: float = 2.5
    tube_resolution: int = 4
    head_radial_resolution: int = 30
    max_points: Optional[int] = None


@dataclass
class CloudGeometry(Geometry):
    """
    Specifies which hydrometeor fields make up a physically rendered cloud.

    Unlike the other geometries, a cloud is built from several variables at
    once: one mass mixing ratio per hydrometeor species (and its number
    concentration, when the scheme predicts one), plus the air density. The
    optics derived from them are documented in :mod:`skyvista.cloud_optics`.

    Attributes:
        species: Explicit species list. None selects a preset (see
            ``microphysics``).
        microphysics: Preset name ("rams", "cm1_morrison", "generic"), or
            None to detect it from the variable names. Ignored when
            ``species`` is given.
        air_density: Air-density variable name, a constant in kg m^-3, or
            None to search common names and fall back to the 1976 standard
            atmosphere.
        vertical_spacing: Uniform vertical voxel size in meters for the
            Blender (VDB) export. VDB voxels are uniform, so stretched model
            levels are linearly interpolated onto this spacing. None uses the
            finest native spacing among the levels that contain condensate,
            but no finer than a quarter of the horizontal grid spacing.
        extinction_floor: Render extinction (m^-1) below which a voxel is
            treated as clear air and dropped from the VDB. The default 1e-6
            m^-1 is an optical depth of 0.001 per kilometre.
    """

    species: Optional[Sequence["HydrometeorSpecies"]] = None
    microphysics: Optional[str] = None
    air_density: Union[str, float, None] = None
    vertical_spacing: Optional[float] = None
    extinction_floor: float = 1.0e-6
