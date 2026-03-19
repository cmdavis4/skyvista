"""
Geometry classes for skyvista.

Geometry classes specify *what* to extract from data - the shape and structure
of the visualization, independent of how it looks (appearance).
"""

from abc import ABC
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


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
    slice_value: Optional[float] = None
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
