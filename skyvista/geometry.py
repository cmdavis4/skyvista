"""
Geometry classes for skyvista.

Geometry classes specify *what* to extract from data - the shape and structure
of the visualization, independent of how it looks (appearance).
"""

from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

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
        factor: Glyph length per unit of vector magnitude, in data length
            units (e.g. meters of arrow per m/s of wind). None = auto: the
            longest arrow spans half the spacing between glyphs.
        tolerance: Point merging tolerance (None = default, 0 = no merging)
        stride: Keep every ``stride``-th grid point for glyphs. One int
            applies to every spatial dimension; a dict maps dimension names
            (``{"x": 8, "z": 8}``) to per-dimension strides, with unlisted
            dimensions kept whole. None keeps every point, which on a model
            grid is usually far too dense to read.
        color_by: Scalar to color glyphs by: "speed" (the vector magnitude),
            "u", "v" or "w" (one component), or the name of a dataset
            variable. None = color by ``scale_by`` if set, else by speed.
        mask_by: Name of a boolean (or 0/1) dataset variable on the same grid
            as the components. Glyphs are drawn only where it is true, e.g. to
            drop arrows that carry no information (weak wind far from cloud).
            Applied after ``stride``. None draws every (strided) glyph.
    """

    varname: str
    u_varname: str = "UC"
    v_varname: str = "VC"
    w_varname: str = "WC"
    scale_by: Optional[str] = None
    factor: Optional[float] = None
    tolerance: Optional[float] = None
    stride: Optional[Union[int, Dict[str, int]]] = None
    color_by: Optional[str] = None
    mask_by: Optional[str] = None


@dataclass
class StreamlineGeometry(Geometry):
    """
    Specifies streamlines traced through a 3-D vector field.

    A streamline is everywhere tangent to the *instantaneous* field, so it
    shows the flow pattern at one time (not the path a parcel follows over
    time, which is a trajectory). Its shape depends on the reference frame:
    a storm's inflow and outflow branches show clearly only in the storm's
    own frame, which ``frame_velocity`` provides.

    Attributes:
        u_varname: Variable name for u (x) component
        v_varname: Variable name for v (y) component
        w_varname: Variable name for w (z) component
        seeds: Explicit seed points, an (n, 3) array of data coordinates
            (x, y, z). None seeds on a lattice of grid points instead.
        seed_stride: Lattice spacing in grid points when ``seeds`` is None:
            one int for every axis, or per-axis, e.g. ``{"x": 10, "y": 10,
            "z": 6}``.
        seed_mask_by: Boolean (or 0/1) dataset variable; lattice seeds are
            kept only where it is true (e.g. inside the updraft, or in the
            boundary-layer inflow). Ignored with explicit ``seeds``.
        direction: Trace "forward", "backward", or "both" ways from each seed.
        max_length: Longest streamline (data length units) in each direction.
        min_length: Drop streamlines shorter than this (data length units),
            e.g. ones seeded in nearly still air.
        frame_velocity: (u, v, w) subtracted from the field before tracing,
            in the field's units: the velocity of the frame the streamlines
            are drawn in, e.g. the storm motion. None = the grid's frame.
        color_by: Scalar to color by: "speed" (in the chosen frame), "u", "v"
            or "w" (ground-relative components), or a dataset variable.
        point_spacing: Spacing (data units) of the points kept along each
            streamline for export. VTK's adaptive steps leave thousands of
            points per line; resampling at about the grid scale keeps every
            resolved bend. None = half the finest grid spacing.
    """

    u_varname: str = "UC"
    v_varname: str = "VC"
    w_varname: str = "WC"
    seeds: Optional[Any] = None
    seed_stride: Union[int, Dict[str, int]] = 10
    seed_mask_by: Optional[str] = None
    direction: str = "both"
    max_length: float = 30000.0
    min_length: float = 0.0
    frame_velocity: Optional[Tuple[float, float, float]] = None
    color_by: str = "speed"
    point_spacing: Optional[float] = None


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
        max_points: Keep only the last ``max_points`` times of each track
        particle_radius: Sphere radius (data units) of each particle in the
            "particle" style
        trail_duration: Keep only the part of each track within this many
            seconds before the time shown, so every track becomes a trail
            behind its current position. None keeps the whole history. Applies
            together with ``max_points``; the shorter one wins.
        trail_taper: Tube radius at the oldest end of a trail, as a fraction
            of ``tube_radius`` (the radius at the head). 1 keeps a uniform
            tube; 0 narrows the trail to a point, like a comet's tail. The
            radius shrinks linearly with age, measured against
            ``trail_duration`` when it is set (so a young track that has not
            yet filled its trail is only partly tapered), else against each
            trail's own age span. Used by the Blender export of the "tube"
            style; the PyVista tubes stay uniform.
    """

    scalar: Optional[str] = None
    tube_radius: float = 70
    head_length_frac: float = 10
    head_radius_frac: float = 2.5
    tube_resolution: int = 4
    head_radial_resolution: int = 30
    max_points: Optional[int] = None
    particle_radius: float = 250.0
    trail_duration: Optional[float] = None
    trail_taper: float = 1.0


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
        weight: Name of a dataset variable between 0 and 1 that multiplies
            every species' extinction point by point. 0 removes the cloud, and
            values between fade it. The particle sizes, and so the phase
            functions, are unchanged. A typical use fades out a neighbouring
            storm. The variable may lack dimensions, e.g. be (y, x) only.
    """

    species: Optional[Sequence["HydrometeorSpecies"]] = None
    microphysics: Optional[str] = None
    air_density: Union[str, float, None] = None
    vertical_spacing: Optional[float] = None
    extinction_floor: float = 1.0e-6
    weight: Optional[str] = None


@dataclass
class DyeGeometry(Geometry):
    """
    Specifies which passive tracers make up a colored "dye" volume.

    A model's passive tracers label where air came from. Each tracer holds
    the share of a cell's air that started in one labelled region, scaled by
    ``reference_value``, e.g. one 200 m layer of starting height. The dye
    combines them into two fields per cell:

    * the **dyed fraction** ``phi = sum_i tracer_i / reference_value``, which
      sets how much dye the cell holds (its optical density);
    * the **dye label** ``sum_i value_i * tracer_i / sum_i tracer_i``, the
      tracer-weighted mean of the per-tracer values, which sets its color.
      With starting heights as values, the label is the mean starting height
      of the cell's dyed air.

    Attributes:
        tracer_varnames: Tracer variables, one per labelled region.
        tracer_values: One value per tracer, e.g. its starting height (m).
            None labels tracer ``i`` with ``i``, so a categorical colormap
            gives each tracer its own color.
        reference_value: Tracer value of undiluted air from the region, so
            that ``tracer / reference_value`` is a fraction in [0, 1].
        min_vertical_displacement: Treat ``tracer_values`` as starting
            heights (m), and hide each tracer within this vertical distance
            of its starting height. Layered tracers fill the whole domain,
            so undyed environment would otherwise bury everything else in
            opaque horizontal sheets. With this set, only air that has been
            lifted or lowered by more than the distance carries dye. None
            dyes every tracer everywhere.
        fraction_floor: Dyed fraction below which a voxel counts as clear and
            is dropped from the Blender (VDB) export.
        preview_fraction: Dyed fraction of the isosurface that the PyVista
            preview draws (PyVista cannot color a volume by one field and
            shade its opacity by another).
    """

    tracer_varnames: Sequence[str] = field(default_factory=list)
    tracer_values: Optional[Sequence[float]] = None
    reference_value: float = 1.0
    min_vertical_displacement: Optional[float] = None
    fraction_floor: float = 1.0e-3
    preview_fraction: float = 0.3
