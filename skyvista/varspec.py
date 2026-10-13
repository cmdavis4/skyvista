"""
VarSpec classes for skyvista.

VarSpec classes are self-rendering visualization specifications that know how to:
1. Extract/create a mesh from data (create_mesh)
2. Provide rendering parameters for PyVista (get_pyvista_kwargs)
3. Render themselves to a plotter (render_to_plotter)

This design follows the Open/Closed Principle - adding new visualization types
requires creating a new VarSpec subclass, not modifying the Scene.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import warnings

import numpy as np
import pyvista as pv
import xarray as xr

from .appearance import (
    Appearance,
    CloudAppearance,
    ContourAppearance,
    DyeAppearance,
    StreamlineAppearance,
    TrajectoryAppearance,
    VectorAppearance,
    VolumeAppearance,
)
from .geometry import (
    CloudGeometry,
    ContourGeometry,
    DyeGeometry,
    Geometry,
    SliceGeometry,
    StreamlineGeometry,
    TrajectoryGeometry,
    VectorGeometry,
    VolumeGeometry,
)
from .grid_utils import (
    add_scalar_to_grid,
    build_grid,
    build_rectilinear_grid,
    select_time,
)
from .grids import get_grid_builder

if TYPE_CHECKING:
    from .cloud_optics import HydrometeorSpecies, SpeciesOptics
    from .mesh import PVMesh


@dataclass
class VarSpec(ABC):
    """
    Base class for self-rendering visualization specifications.

    Each VarSpec subclass knows how to create a mesh from data and how to
    render it. This allows the Scene to be agnostic about visualization types.

    Attributes:
        name: Unique identifier for this spec (auto-generated if None)
        empty_ok: If True, don't skip this spec when mesh is empty
        grid_type: Explicit grid type ("rectilinear", "curvilinear", "unstructured")
                   or None for auto-detection
        pyvista_create_kwargs: Escape hatch for PyVista mesh creation kwargs
        pyvista_add_kwargs: Escape hatch for PyVista add_mesh kwargs
    """

    name: Optional[str] = None
    empty_ok: bool = False
    grid_type: Optional[str] = None

    # Escape hatches for edge cases
    pyvista_create_kwargs: Dict[str, Any] = field(default_factory=dict)
    pyvista_add_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Subclasses must define geometry and appearance fields
    geometry: Geometry = field(init=False)
    appearance: Appearance = field(init=False)

    @abstractmethod
    def create_mesh(self, ds: xr.Dataset, time: Any) -> Optional[pv.DataSet]:
        """
        Create PyVista mesh from dataset at given time.

        Args:
            ds: xarray Dataset containing the data
            time: Time value to select (None for no time selection)

        Returns:
            PyVista mesh, or None if no mesh could be created
        """
        ...

    def get_pyvista_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for plotter.add_mesh()."""
        kwargs = self.appearance.to_pyvista_kwargs()
        kwargs.update(self.pyvista_add_kwargs)  # Escape hatch overrides
        return kwargs

    def is_volume(self) -> bool:
        """Whether this spec renders as a volume (vs surface mesh)."""
        return False

    def render_to_plotter(
        self,
        plotter: pv.Plotter,
        ds: xr.Dataset,
        time: Any,
    ) -> Optional["PVMesh"]:
        """
        Create mesh and add to plotter.

        Args:
            plotter: PyVista plotter to add mesh to
            ds: xarray Dataset containing the data
            time: Time value to select

        Returns:
            PVMesh object with mesh and actor, or None if skipped
        """
        from .mesh import PVMesh

        mesh = self.create_mesh(ds, time)

        if mesh is None:
            return None
        if len(mesh.points) == 0 and not self.empty_ok:
            return None

        kwargs = self.get_pyvista_kwargs()

        if self.is_volume():
            actor = plotter.add_volume(
                mesh,
                name=self.name,
                show_scalar_bar=self.appearance.show_scalar_bar,
                **kwargs,
            )
        else:
            actor = plotter.add_mesh(
                mesh,
                name=self.name,
                show_scalar_bar=self.appearance.show_scalar_bar,
                **kwargs,
            )

        return PVMesh(varspec=self, mesh=mesh, actor=actor, time=time)


@dataclass
class ContourSpec(VarSpec):
    """
    Isosurface visualization specification.

    Creates isosurfaces (3D contours) from scalar field data.
    """

    geometry: ContourGeometry = field(
        default_factory=lambda: ContourGeometry(varname="")
    )
    appearance: ContourAppearance = field(default_factory=ContourAppearance)

    def __post_init__(self):
        if self.name is None:
            iso_str = ""
            if (self.geometry.isosurfaces is not None) and (
                len(self.geometry.isosurfaces) > 0
            ):
                iso_str = f"_iso{self.geometry.isosurfaces[0]}"
            self.name = f"contour_{self.geometry.varname}{iso_str}"

    def create_mesh(self, ds: xr.Dataset, time: Any) -> Optional[pv.DataSet]:
        ds = select_time(ds, time)

        # Build grid (auto-detects type or uses explicit grid_type)
        varname = self.geometry.varname
        grid = build_grid(ds, varname=varname, grid_type=self.grid_type)

        # Create contour
        contour_kwargs = {"scalars": varname}
        if self.geometry.isosurfaces is not None:
            contour_kwargs["isosurfaces"] = self.geometry.isosurfaces
        contour_kwargs.update(self.pyvista_create_kwargs)

        mesh = grid.contour(**contour_kwargs)

        # Sample scalar field if different from contour variable
        if self.geometry.scalar and self.geometry.scalar != varname:
            add_scalar_to_grid(grid, ds, self.geometry.scalar)
            mesh = mesh.sample(grid, pass_point_data=False, pass_cell_data=False)
            mesh.set_active_scalars(self.geometry.scalar)

        return mesh


@dataclass
class VolumeSpec(VarSpec):
    """
    Volume rendering specification.

    Renders scalar field data as a 3D volume with opacity transfer function.
    """

    geometry: VolumeGeometry = field(default_factory=lambda: VolumeGeometry(varname=""))
    appearance: VolumeAppearance = field(default_factory=VolumeAppearance)

    def __post_init__(self):
        if self.name is None:
            self.name = f"volume_{self.geometry.varname}"

    def is_volume(self) -> bool:
        return True

    def create_mesh(self, ds: xr.Dataset, time: Any) -> Optional[pv.DataSet]:
        ds = select_time(ds, time)

        # Build grid (auto-detects type or uses explicit grid_type)
        varname = self.geometry.varname
        grid = build_grid(ds, varname=varname, grid_type=self.grid_type)

        # Apply threshold if specified
        if self.geometry.threshold:
            low, high = self.geometry.threshold
            # Handle None values in threshold
            if low is None and high is not None:
                grid = grid.threshold(value=high, scalars=varname, invert=True)
            elif high is None and low is not None:
                grid = grid.threshold(value=low, scalars=varname)
            elif low is not None and high is not None:
                grid = grid.threshold(value=(low, high), scalars=varname)

        return grid

    def get_pyvista_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_pyvista_kwargs()

        # Set scalars explicitly for volumes
        kwargs["scalars"] = self.geometry.varname

        return kwargs


@dataclass
class VectorSpec(VarSpec):
    """
    Vector field glyph specification.

    Creates arrow glyphs from vector field data.
    """

    geometry: VectorGeometry = field(default_factory=lambda: VectorGeometry(varname=""))
    appearance: VectorAppearance = field(default_factory=VectorAppearance)

    def __post_init__(self):
        if self.name is None:
            self.name = f"vector_{self.geometry.varname}"

    def subsample(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Thin the dataset to the glyph lattice set by ``geometry.stride``.

        A model grid has far more points than arrows a viewer can read, so
        glyphs are placed on every ``stride``-th grid point. Striding (rather
        than averaging) keeps each glyph equal to the model value at its own
        point, so an arrow never shows a wind that no grid point had.
        """
        from .grids import resolve_coordinates

        stride = self.geometry.stride
        if stride is None:
            return ds
        coordinate_names = resolve_coordinates(ds, ["x", "y", "z"])
        # One stride for all axes, or explicit per-axis strides.
        if isinstance(stride, int):
            strides_by_axis = dict.fromkeys(("x", "y", "z"), stride)
        else:
            strides_by_axis = dict(stride)
        index_slices = {}
        for axis, axis_stride in strides_by_axis.items():
            # Accept either the generic axis name or the dataset's own name.
            dimension_name = coordinate_names.get(axis, axis)
            if dimension_name in ds.dims and int(axis_stride) > 1:
                index_slices[dimension_name] = slice(None, None, int(axis_stride))
        return ds.isel(index_slices) if index_slices else ds

    def vector_components(self, ds: xr.Dataset) -> List[np.ndarray]:
        """The (u, v, w) component arrays, flattened in grid (Fortran) order."""
        return [
            ds[self.geometry.u_varname].values.ravel(order="F"),
            ds[self.geometry.v_varname].values.ravel(order="F"),
            ds[self.geometry.w_varname].values.ravel(order="F"),
        ]

    def glyph_mask(self, ds: xr.Dataset) -> Optional[np.ndarray]:
        """
        Which grid points get a glyph, flattened like the components.

        Reads ``geometry.mask_by``, transposed to the u component's dimension
        order first, so the flattened mask lines up point for point with the
        flattened components. None when no mask is set.
        """
        mask_name = self.geometry.mask_by
        if mask_name is None:
            return None
        component_dims = ds[self.geometry.u_varname].dims
        mask = ds[mask_name].transpose(*component_dims).values
        return np.asarray(mask, dtype=bool).ravel(order="F")

    def color_scalar_name(self) -> Optional[str]:
        """Name of the scalar the glyphs are colored by (see ``color_by``)."""
        return self.geometry.color_by or self.geometry.scale_by

    def color_values(self, ds: xr.Dataset, components: List[np.ndarray]) -> np.ndarray:
        """
        Per-glyph values of the coloring scalar, flattened like the components.

        "speed", "u", "v" and "w" are derived from the vector itself; any other
        name is read from the dataset (e.g. color arrows by temperature).
        """
        scalar_name = self.color_scalar_name() or "speed"
        derived_from_components = {
            "speed": lambda: np.sqrt(sum(component**2 for component in components)),
            "u": lambda: components[0],
            "v": lambda: components[1],
            "w": lambda: components[2],
        }
        if scalar_name in derived_from_components:
            return derived_from_components[scalar_name]()
        return ds[scalar_name].values.ravel(order="F")

    def auto_factor(
        self, ds: xr.Dataset, components: List[np.ndarray]
    ) -> Optional[float]:
        """
        Default glyph length per unit magnitude: longest arrow = half a spacing.

        Uses the spacing of the (already subsampled) glyph lattice, so arrows
        are sized against the distance to their neighbours rather than to the
        much finer model grid. Returns None for a field that is zero everywhere.
        """
        from carlee_tools import spacing

        from .grids import resolve_coordinates

        coordinate_names = resolve_coordinates(ds, ["x", "y", "z"])
        lattice_spacings = []
        for axis in ("x", "y", "z"):
            dimension_name = coordinate_names.get(axis)
            if dimension_name in ds.coords and len(ds[dimension_name]) > 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    lattice_spacings.append(
                        spacing(
                            ds[dimension_name].values,
                            raise_if_not_evenly_spaced=False,
                        )
                    )
        maximum_magnitude = float(
            np.nanmax(np.sqrt(sum(component**2 for component in components)))
        )
        if not lattice_spacings or not maximum_magnitude > 0:
            return None
        return 0.5 * min(lattice_spacings) / maximum_magnitude

    def create_mesh(self, ds: xr.Dataset, time: Any) -> Optional[pv.DataSet]:
        ds = self.subsample(select_time(ds, time))

        # Build grid (auto-detects type or uses explicit grid_type)
        grid = build_grid(ds, grid_type=self.grid_type)

        # Stack vector components
        components = self.vector_components(ds)
        vectors = np.vstack(components).T
        vector_name = self.geometry.varname
        grid[vector_name] = vectors
        grid.set_active_vectors(vector_name)

        # Add individual components for potential scaling
        grid["u"], grid["v"], grid["w"] = components

        # Carry the coloring scalar onto the glyphs when one was requested.
        color_name = self.geometry.color_by
        if color_name is not None and color_name not in ("u", "v", "w"):
            grid[color_name] = self.color_values(ds, components)

        # Keep only the points the mask selects (vertices only, no cells).
        mask = self.glyph_mask(ds)
        if mask is not None:
            grid = grid.extract_points(mask, adjacent_cells=False, include_cells=False)

        # Prepare glyph kwargs
        glyph_kwargs = {"orient": vector_name}

        # Auto-calculate factor if not provided
        factor = self.geometry.factor
        if factor is None:
            factor = self.auto_factor(ds, components)
        if factor is not None:
            glyph_kwargs["factor"] = factor

        # Handle scaling
        if self.geometry.scale_by:
            add_scalar_to_grid(grid, ds, self.geometry.scale_by)
            glyph_kwargs["scale"] = self.geometry.scale_by
        else:
            glyph_kwargs["scale"] = vector_name

        if self.geometry.tolerance is not None:
            glyph_kwargs["tolerance"] = self.geometry.tolerance

        glyph_kwargs.update(self.pyvista_create_kwargs)

        return grid.glyph(**glyph_kwargs)


# Point-data names used while tracing streamlines (internal to StreamlineSpec).
STREAMLINE_VECTOR_ARRAY = "_streamline_vectors"
STREAMLINE_COLOR_ARRAY = "_streamline_color"


@dataclass
class StreamlineSpec(VarSpec):
    """
    Streamlines of a 3-D vector field, drawn as tubes.

    Streamlines are traced here, in Python, with VTK's stream tracer (a
    Runge-Kutta 4-5 integrator with adaptive steps) on the model grid. Each
    resulting polyline is reoriented to run downstream, so anything drawn
    along it (Blender's direction markers) points the way the air moves. The
    PyVista renderer draws tubes; the Blender export sends the polylines
    themselves, and Blender sweeps the tubes and places the markers.
    """

    geometry: StreamlineGeometry = field(default_factory=StreamlineGeometry)
    appearance: StreamlineAppearance = field(default_factory=StreamlineAppearance)

    def __post_init__(self):
        if self.name is None:
            self.name = f"streamlines_{self.geometry.color_by}"

    def seed_points(self, ds: xr.Dataset) -> np.ndarray:
        """
        Seed points (n, 3) in data coordinates.

        Explicit ``geometry.seeds`` are used as given. Otherwise seeds sit on
        every ``seed_stride``-th grid point, optionally only where
        ``seed_mask_by`` is true.
        """
        from .grids import resolve_coordinates

        geometry = self.geometry
        if geometry.seeds is not None:
            return np.asarray(geometry.seeds, dtype=float).reshape(-1, 3)
        coordinate_names = resolve_coordinates(ds, ["x", "y", "z"])
        dimension_names = [coordinate_names[axis] for axis in ("x", "y", "z")]
        # Strided lattice of grid points.
        stride = geometry.seed_stride
        strides_by_axis = (
            dict.fromkeys(("x", "y", "z"), stride)
            if isinstance(stride, int)
            else dict(stride)
        )
        lattice = ds.isel(
            {
                coordinate_names[axis]: slice(None, None, int(axis_stride))
                for axis, axis_stride in strides_by_axis.items()
                if coordinate_names.get(axis) in ds.dims
            }
        )
        meshed = np.meshgrid(
            *[lattice[name].values for name in dimension_names], indexing="ij"
        )
        seeds = np.stack([coordinate.ravel() for coordinate in meshed], axis=1)
        # Optionally keep only seeds where the mask is true.
        if geometry.seed_mask_by is not None:
            mask = lattice[geometry.seed_mask_by].transpose(*dimension_names).values
            seeds = seeds[np.asarray(mask, dtype=bool).ravel()]
        return seeds

    def trace(self, ds: xr.Dataset, time: Any) -> pv.PolyData:
        """
        Trace the streamlines at one time; returns VTK's polyline output.

        The field is first moved into the requested frame (``frame_velocity``
        subtracted). The coloring scalar travels along as point data, so it
        is interpolated onto the streamline points by VTK.
        """
        geometry = self.geometry
        ds = select_time(ds, time)
        grid = build_grid(ds, grid_type=self.grid_type)

        # Components in the requested frame.
        frame_velocity = geometry.frame_velocity or (0.0, 0.0, 0.0)
        component_names = (geometry.u_varname, geometry.v_varname, geometry.w_varname)
        for component_name in component_names:
            add_scalar_to_grid(grid, ds, component_name)
        frame_components = [
            np.asarray(grid[name], dtype=float) - offset
            for name, offset in zip(component_names, frame_velocity, strict=True)
        ]
        grid[STREAMLINE_VECTOR_ARRAY] = np.stack(frame_components, axis=1)

        # The coloring scalar, as point data.
        color_by = geometry.color_by
        if color_by == "speed":
            color_values = np.sqrt(sum(c**2 for c in frame_components))
        elif color_by in ("u", "v", "w"):
            color_values = np.asarray(grid[component_names["uvw".index(color_by)]])
        else:
            add_scalar_to_grid(grid, ds, color_by)
            color_values = np.asarray(grid[color_by])
        grid[STREAMLINE_COLOR_ARRAY] = color_values

        seeds = self.seed_points(ds)
        if len(seeds) == 0:
            return pv.PolyData()
        return grid.streamlines_from_source(
            pv.PolyData(seeds),
            vectors=STREAMLINE_VECTOR_ARRAY,
            integration_direction=geometry.direction,
            max_length=geometry.max_length,
            initial_step_length=0.2,
            max_steps=100000,
            terminal_speed=1e-3,
            compute_vorticity=False,
            **self.pyvista_create_kwargs,
        )

    def polylines(
        self, ds: xr.Dataset, time: Any
    ) -> List["tuple[np.ndarray, np.ndarray]"]:
        """
        The streamlines at one time as (points (k, 3), color values (k,)).

        Each polyline is ordered downstream: VTK writes a backward-traced line
        from the seed upstream, so a line whose point order runs against its
        own vectors is reversed. Lines shorter than ``min_length`` (or with
        fewer than two points) are dropped. Each line is then resampled to
        evenly spaced points (``point_spacing``) by linear interpolation in
        arc length, with both ends kept.
        """
        from carlee_tools import spacing

        from .grids import resolve_coordinates

        # Resampling distance along the lines.
        point_spacing = self.geometry.point_spacing
        if point_spacing is None:
            coordinate_names = resolve_coordinates(ds, ["x", "y", "z"])
            grid_spacings = [
                spacing(
                    ds[coordinate_names[axis]].values, raise_if_not_evenly_spaced=False
                )
                for axis in ("x", "y", "z")
                if ds.sizes.get(coordinate_names.get(axis), 0) > 1
            ]
            point_spacing = 0.5 * min(grid_spacings) if grid_spacings else 1.0

        traced = self.trace(ds, time)
        if traced.n_points == 0:
            return []
        points = np.asarray(traced.points)
        vectors = np.asarray(traced.point_data[STREAMLINE_VECTOR_ARRAY])
        colors = np.asarray(traced.point_data[STREAMLINE_COLOR_ARRAY])
        # VTK's line connectivity: [n, i0 ... i(n-1), n, ...].
        connectivity = np.asarray(traced.lines)
        result = []
        cursor = 0
        while cursor < len(connectivity):
            n_line_points = int(connectivity[cursor])
            point_ids = connectivity[cursor + 1 : cursor + 1 + n_line_points]
            cursor += n_line_points + 1
            if n_line_points < 2:
                continue
            line_points = points[point_ids]
            segments = np.diff(line_points, axis=0)
            if np.linalg.norm(segments, axis=1).sum() < self.geometry.min_length:
                continue
            # Downstream order: segments should align with the flow.
            alignment = np.sum(segments * vectors[point_ids][:-1])
            if alignment < 0:
                point_ids = point_ids[::-1]
            line_points = points[point_ids]
            line_colors = colors[point_ids]
            # Even resampling in arc length.
            arc_length = np.concatenate(
                [[0.0], np.cumsum(np.linalg.norm(np.diff(line_points, axis=0), axis=1))]
            )
            n_samples = max(int(np.ceil(arc_length[-1] / point_spacing)) + 1, 2)
            sample_arc_length = np.linspace(0.0, arc_length[-1], n_samples)
            resampled_points = np.stack(
                [
                    np.interp(sample_arc_length, arc_length, line_points[:, axis])
                    for axis in range(3)
                ],
                axis=1,
            )
            resampled_colors = np.interp(sample_arc_length, arc_length, line_colors)
            result.append((resampled_points, resampled_colors))
        return result

    def create_mesh(self, ds: xr.Dataset, time: Any) -> Optional[pv.DataSet]:
        """Tubes around the traced streamlines, carrying the color scalar."""
        traced = self.trace(ds, time)
        if traced.n_points == 0:
            return pv.PolyData()
        traced[self.geometry.color_by] = traced.point_data[STREAMLINE_COLOR_ARRAY]
        return traced.tube(radius=self.appearance.tube_radius)

    def get_pyvista_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_pyvista_kwargs()
        if self.appearance.color is None:
            kwargs["scalars"] = self.geometry.color_by
        return kwargs


@dataclass
class SliceSpec(VarSpec):
    """
    2D slice visualization specification.

    Extracts a 2D slice from 3D scalar field data.
    """

    geometry: SliceGeometry = field(default_factory=lambda: SliceGeometry(varname=""))
    appearance: Appearance = field(default_factory=Appearance)

    def __post_init__(self):
        if self.name is None:
            self.name = f"slice_{self.geometry.varname}_{self.geometry.slice_dim}"

    def create_mesh(self, ds: xr.Dataset, time: Any) -> Optional[pv.DataSet]:
        from .grids import resolve_coordinates

        ds = select_time(ds, time)

        # Resolve coordinate names
        coord_names = resolve_coordinates(ds, ["x", "y", "z"])

        # Get slice parameters
        slice_dim = self.geometry.slice_dim
        slice_value = self.geometry.slice_value
        slice_method = self.geometry.slice_method

        # Map slice_dim to actual coordinate name if needed
        slice_coord = coord_names.get(slice_dim, slice_dim)

        # Select the slice from the data
        if slice_value is None:
            slice_sel = {slice_coord: ds[slice_coord].values[0]}
        else:
            slice_sel = {slice_coord: slice_value}

        sliced_data = ds.sel(slice_sel, method=slice_method)

        # Determine remaining dimensions
        remaining_dims = [d for d in ["x", "y", "z"] if d != slice_dim]
        dim1, dim2 = remaining_dims
        coord1_name = coord_names[dim1]
        coord2_name = coord_names[dim2]

        # Create meshgrid
        coord1 = sliced_data[coord1_name].values
        coord2 = sliced_data[coord2_name].values
        grid1, grid2 = np.meshgrid(coord1, coord2, indexing="ij")

        # Create constant coordinate for sliced dimension
        if slice_value is None:
            slice_coord_value = sliced_data[slice_coord].values.item()
        else:
            slice_coord_value = slice_value
        grid_sliced = np.ones_like(grid1) * slice_coord_value

        # Map coordinates to x, y, z positions for StructuredGrid
        grids = {slice_dim: grid_sliced, dim1: grid1, dim2: grid2}

        create_kwargs = dict(self.pyvista_create_kwargs)
        mesh = pv.StructuredGrid(grids["x"], grids["y"], grids["z"], **create_kwargs)

        # Add variable data
        varname = self.geometry.varname
        var_data = sliced_data[varname].values
        mesh[varname] = var_data.ravel(order="F")

        return mesh


@dataclass
class TrajectorySpec(VarSpec):
    """
    Lagrangian trajectory visualization specification.

    Creates tube or particle visualizations from trajectory data.
    All trajectory mesh creation logic is encapsulated within this class.
    """

    geometry: TrajectoryGeometry = field(default_factory=TrajectoryGeometry)
    appearance: TrajectoryAppearance = field(default_factory=TrajectoryAppearance)
    limit: Optional[int] = 1000

    def __post_init__(self):
        if self.name is None:
            style = self.appearance.style
            scalar_part = f"_{self.geometry.scalar}" if self.geometry.scalar else ""
            self.name = f"trajectory_{style}{scalar_part}"

    def create_mesh(self, ds: xr.Dataset, time: Any) -> Optional[pv.DataSet]:
        """Create trajectory mesh from dataset."""
        # Trajectories use all times up to current time
        if time is not None and "time" in ds.dims:
            ds = ds.sel(time=slice(None, time))

        # Handle trajectory_ix dimension (with backwards compatibility for parcel_ix)
        if "trajectory_ix" in ds.dims:
            trajectory_dim = "trajectory_ix"
        elif "parcel_ix" in ds.dims:
            trajectory_dim = "parcel_ix"
        else:
            raise ValueError(
                "Trajectory dataset must have 'trajectory_ix' or 'parcel_ix'"
                " dimension"
            )

        # Apply limit
        if self.limit and len(ds[trajectory_dim]) > self.limit:
            ds = ds.isel({trajectory_dim: slice(self.limit)})

        # Apply max points on the tail, if passed
        if self.geometry.max_points:
            ds = ds.isel(time=slice(-self.geometry.max_points, None))

        # Particles need only the current position, so one time is enough.
        if self.appearance.style == "particle":
            return self._create_particle_mesh(ds, trajectory_dim)

        # Need at least 2 time points for trajectory rendering
        if "time" in ds.dims and len(ds["time"]) < 2:
            return pv.PolyData()

        # Extract trajectory data for tube rendering
        trajectories_points_data = self._extract_trajectory_data(ds, trajectory_dim)

        if not trajectories_points_data:
            return pv.PolyData()

        # Create tube mesh with arrow heads
        mesh = self._create_tube_mesh(trajectories_points_data)

        # Store scalar info
        if self.geometry.scalar:
            mesh.field_data["arrow_color_scalar"] = self.geometry.scalar

        return mesh

    def _extract_trajectory_data(
        self, ds: xr.Dataset, trajectory_dim: str
    ) -> List[Dict[str, Any]]:
        """Extract trajectory point data from dataset."""
        from carlee_tools import maybe_cast_to_float

        trajectories_points_data = []
        scalar = self.geometry.scalar

        ds = ds.transpose(trajectory_dim, "time", ...)

        x_data = ds["x"].values
        y_data = ds["y"].values
        z_data = ds["z"].values
        trajectory_indices = ds[trajectory_dim].values

        # Pre-extract scalar data if needed
        scalar_data = None
        if scalar and scalar in ds:
            scalar_data = maybe_cast_to_float(ds[scalar].values)

        for i, trajectory_ix in enumerate(trajectory_indices):
            xyz_points = np.column_stack([x_data[i], y_data[i], z_data[i]])
            valid_mask = ~(np.isnan(xyz_points).any(axis=1))

            if valid_mask.sum() < 2:
                continue

            valid_points = xyz_points[valid_mask]

            trajectory_points_data = {
                trajectory_dim: trajectory_ix,
                "points": valid_points,
            }

            # Add scalar data
            if scalar and scalar_data is not None:
                if len(scalar_data.shape) > 1:
                    trajectory_points_data[scalar] = scalar_data[i][valid_mask]
                else:
                    trajectory_points_data[scalar] = np.full(
                        len(valid_points), scalar_data[i]
                    )

            trajectories_points_data.append(trajectory_points_data)

        return trajectories_points_data

    def particle_positions(
        self, ds: xr.Dataset, time: Any
    ) -> "tuple[np.ndarray, Optional[np.ndarray]]":
        """
        Particle positions (n, 3) at ``time``, and their scalar values if any.

        A particle sits at its track's position at the latest time not after
        ``time`` (``time`` None = the last time). Tracks with no valid
        position then (not yet started, or already ended) are dropped. The
        ``limit`` on the number of tracks applies, as for tubes.
        """
        from carlee_tools import maybe_cast_to_float

        # Tracks may be indexed by trajectory_ix or (older data) parcel_ix.
        trajectory_dim = "trajectory_ix" if "trajectory_ix" in ds.dims else "parcel_ix"
        if (
            self.limit
            and trajectory_dim in ds.dims
            and ds.sizes[trajectory_dim] > self.limit
        ):
            ds = ds.isel({trajectory_dim: slice(self.limit)})
        # Positions at the requested time.
        if "time" in ds.dims:
            if time is not None:
                ds = ds.sel(time=slice(None, time))
            ds = ds.isel(time=-1)
        points = np.column_stack([ds["x"].values, ds["y"].values, ds["z"].values])
        valid_mask = ~np.isnan(points).any(axis=1)
        scalar_values = None
        if self.geometry.scalar and self.geometry.scalar in ds:
            scalar_values = maybe_cast_to_float(ds[self.geometry.scalar].values)[
                valid_mask
            ]
        return points[valid_mask], scalar_values

    def _create_particle_mesh(self, ds: xr.Dataset, trajectory_dim: str) -> pv.DataSet:
        """Create particle-style mesh (spheres at final positions)."""
        points, scalar_values = self.particle_positions(ds, None)
        point_mesh = pv.PolyData(points)

        # Add scalar data
        if scalar_values is not None:
            point_mesh[self.geometry.scalar] = scalar_values

        # Convert to sphere glyphs
        radius = self.geometry.particle_radius
        glyph_kwargs = {
            "orient": False,
            "scale": False,
            "geom": pv.Sphere(radius=radius, phi_resolution=8, theta_resolution=8),
        }
        glyph_kwargs.update(self.pyvista_create_kwargs)

        return point_mesh.glyph(**glyph_kwargs)

    def _create_tube_mesh(
        self, trajectory_data_list: List[Dict[str, Any]]
    ) -> pv.PolyData:
        """
        Create tube mesh with arrow heads from trajectory point data.

        Args:
            trajectory_data_list: List of dicts with 'points' and optional scalar data.

        Returns:
            Single merged mesh with tubes and cone arrow heads.
        """
        if not trajectory_data_list:
            return pv.PolyData()

        scalar = self.geometry.scalar
        body_radius = self.geometry.tube_radius
        head_length_frac = self.geometry.head_length_frac
        head_radius_frac = self.geometry.head_radius_frac
        head_radial_res = self.geometry.head_radial_resolution
        tube_resolution = self.geometry.tube_resolution

        # Create polydata with all trajectory lines
        polydata = self._create_trajectory_polydata(trajectory_data_list, scalar)

        # Convert to tubes
        tube_mesh = polydata.tube(
            radius=body_radius, n_sides=tube_resolution, capping=False
        )

        # Create arrow heads
        head_meshes = []
        head_length = body_radius * head_length_frac

        for traj_data in trajectory_data_list:
            points = traj_data["points"]
            if len(points) < 2:
                continue

            # Calculate head direction and position
            dhead = points[-1] - points[-2]
            dhead_norm = np.linalg.norm(dhead)
            if dhead_norm > 0:
                dhead = dhead / dhead_norm
                head_center = points[-1] + dhead * (head_length / 2)

                head = pv.Cone(
                    center=head_center,
                    direction=dhead,
                    height=head_length,
                    radius=head_radius_frac * body_radius,
                    resolution=head_radial_res,
                )

                # Add scalar data to head if present
                if scalar and scalar in traj_data:
                    head[scalar] = np.full(head.n_cells, traj_data[scalar][-1])

                head_meshes.append(head)

        # Merge tubes and heads
        meshes_to_merge = [tube_mesh]
        if head_meshes:
            if len(head_meshes) == 1:
                heads_mesh = head_meshes[0]
            else:
                heads_mesh = head_meshes[0].merge(head_meshes[1:], merge_points=False)
            meshes_to_merge.append(heads_mesh)

        if len(meshes_to_merge) == 1:
            final_mesh = meshes_to_merge[0]
        else:
            final_mesh = meshes_to_merge[0].merge(
                meshes_to_merge[1:], merge_points=False
            )

        # Ensure scalar is in point_data
        if scalar and scalar in final_mesh.array_names:
            final_mesh.point_data[scalar] = final_mesh[scalar]

        return final_mesh

    @staticmethod
    def _create_trajectory_polydata(
        trajectory_data_list: List[Dict[str, Any]],
        scalar: Optional[str] = None,
    ) -> pv.PolyData:
        """
        Create PolyData with multiple trajectories as line cells.

        Args:
            trajectory_data_list: List of dicts with 'points' arrays.
            scalar: Optional scalar field name for coloring.

        Returns:
            PolyData with all trajectories as separate line cells.
        """
        if not trajectory_data_list:
            return pv.PolyData()

        total_points = sum(len(t["points"]) for t in trajectory_data_list)
        all_points = np.zeros((total_points, 3))
        all_lines = []
        all_scalars = [] if scalar else None

        point_offset = 0
        for traj_data in trajectory_data_list:
            points = traj_data["points"]
            n_points = len(points)

            all_points[point_offset : point_offset + n_points] = points

            # Line connectivity: [n_points, idx_0, idx_1, ..., idx_n-1]
            line_connectivity = [n_points] + list(
                range(point_offset, point_offset + n_points)
            )
            all_lines.extend(line_connectivity)

            if scalar and scalar in traj_data:
                all_scalars.extend(traj_data[scalar])

            point_offset += n_points

        mesh = pv.PolyData(all_points, lines=np.array(all_lines, dtype=np.int_))
        if scalar and all_scalars:
            mesh[scalar] = np.array(all_scalars)

        return mesh

    @staticmethod
    def _create_tetrahedron_head(
        base_center: np.ndarray, tip: np.ndarray, radius: float
    ) -> pv.PolyData:
        """
        Create tetrahedron arrow head (alternative to cone).

        Args:
            base_center: Center point of tetrahedron base.
            tip: Tip point of arrow head.
            radius: Radius of base.

        Returns:
            Tetrahedron mesh.
        """
        angles = np.linspace(0, 2 * np.pi, 7)[:-1]
        base_points = []

        for angle in angles:
            x = base_center[0] + radius * np.cos(angle)
            y = base_center[1] + radius * np.sin(angle)
            z = base_center[2]
            base_points.append([x, y, z])

        points = np.array(base_points + [list(tip)])

        head = pv.PolyData()
        head.points = points
        faces = np.array(
            [
                3,
                0,
                1,
                2,  # Base
                3,
                0,
                3,
                1,  # Side 1
                3,
                1,
                3,
                2,  # Side 2
                3,
                2,
                3,
                0,  # Side 3
            ]
        )
        head.faces = faces
        return head

    def get_pyvista_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_pyvista_kwargs()

        # Handle silhouettes
        if self.appearance.silhouettes:
            kwargs["silhouette"] = True

        return kwargs


# Point-data name of the total cloud extinction on the PyVista preview grid.
CLOUD_EXTINCTION_VARIABLE = "cloud_extinction"

# Number of samples in the Beer-Lambert opacity transfer function of the
# PyVista cloud preview. Deliberately NOT 256: PyVista copies an opacity array
# whose length equals the lookup table's size (256) straight into the 8-bit
# alpha channel without scaling 0-1 to 0-255, which renders the volume
# invisible. Any other length is interpolated and scaled correctly.
CLOUD_PREVIEW_OPACITY_SAMPLES = 128


@dataclass
class CloudSpec(VarSpec):
    """
    Physically based cloud rendering from hydrometeor fields.

    The spec turns each species' mixing ratio (and number concentration) into
    visible-wavelength optics with :mod:`skyvista.cloud_optics`. The Blender
    backend writes one extinction grid per species to a VDB and renders the
    mixture with Cycles: one scattering closure per species, summed in a
    single volume shader. Summing is exact, because extinction and
    scattering-weighted phase functions are additive for a mixture of
    independent scatterers.

    The PyVista renderer has no multiple scattering, so ``show()`` draws a
    preview: a white emission-absorption volume whose per-cell opacity is the
    Beer-Lambert value ``1 - exp(-beta * cell_size)``. Its transparency
    matches the physical optical depth, but its shading does not.
    """

    geometry: CloudGeometry = field(default_factory=CloudGeometry)
    appearance: CloudAppearance = field(default_factory=CloudAppearance)

    def __post_init__(self):
        if self.name is None:
            self.name = "clouds"
        # Filled by create_mesh() for the PyVista preview's transfer function.
        self._preview_extinction_max: float = 0.0
        self._preview_unit_distance: Optional[float] = None

    def is_volume(self) -> bool:
        return True

    def resolve_species(self, ds: xr.Dataset) -> List["HydrometeorSpecies"]:
        """Species to render, matched against this dataset's variable names."""
        from .cloud_optics import resolve_cloud_species

        return resolve_cloud_species(
            ds, species=self.geometry.species, microphysics=self.geometry.microphysics
        )

    def compute_optics(
        self,
        dataset_at_time: xr.Dataset,
        vertical_coordinate_name: str,
        apply_extinction_scale: bool = True,
    ) -> List["SpeciesOptics"]:
        """
        Optics of every species at one time, by default with ``extinction_scale``.

        Args:
            dataset_at_time: Model output at a single time.
            vertical_coordinate_name: Height coordinate (meters), used only
                when the air density falls back to the standard atmosphere.
            apply_extinction_scale: Multiply the extinctions by the
                appearance's ``extinction_scale``. The PyVista preview wants
                this. The Blender export does not: it writes physical
                extinction to the VDB and lets the shader apply the scale, so
                applying it here too would scale the cloud twice.
        """
        from .cloud_optics import compute_cloud_optics, resolve_air_density

        species = self.resolve_species(dataset_at_time)
        air_density = resolve_air_density(
            dataset_at_time, self.geometry.air_density, vertical_coordinate_name
        )
        optics_per_species = compute_cloud_optics(dataset_at_time, species, air_density)

        # A deliberate, user-requested departure from physical extinction.
        extinction_scale = self.appearance.extinction_scale
        if apply_extinction_scale and extinction_scale != 1.0:
            for optics in optics_per_species:
                optics.extinction = optics.extinction * extinction_scale
                optics.render_extinction = optics.render_extinction * extinction_scale
        return optics_per_species

    def apply_cutaway(
        self, extinction: xr.DataArray, dataset_at_time: xr.Dataset
    ) -> xr.DataArray:
        """
        Zero the extinction on the removed side of the cutaway plane, if any.

        The signed distance ``(position - point) . normal`` is built by
        broadcasting the 1-D x, y, z coordinates, so no 3-D position array is
        stored. Used by the PyVista preview; the Blender export instead passes
        the plane to the shader, which applies the same rule per sample.
        """
        from .grids import resolve_coordinates

        cutaway = self.appearance.cutaway
        if cutaway is None:
            return extinction
        cut_point, cut_normal = cutaway
        coordinate_names = resolve_coordinates(dataset_at_time, ["x", "y", "z"])
        signed_distance = sum(
            (dataset_at_time[coordinate_names[axis]] - cut_point[axis_index])
            * cut_normal[axis_index]
            for axis_index, axis in enumerate(("x", "y", "z"))
        )
        return extinction.where(signed_distance <= 0, 0.0)

    def create_mesh(self, ds: xr.Dataset, time: Any) -> Optional[pv.DataSet]:
        """Build the preview grid carrying the total render extinction (m^-1)."""
        from .cloud_optics import total_render_extinction
        from .grids import resolve_coordinate

        dataset_at_time = select_time(ds, time)
        vertical_coordinate_name = resolve_coordinate(dataset_at_time, "z")

        # Total extinction of the mixture: the sum over species.
        optics_per_species = self.compute_optics(
            dataset_at_time, vertical_coordinate_name
        )
        total_extinction = total_render_extinction(optics_per_species)
        total_extinction = self.apply_cutaway(total_extinction, dataset_at_time)
        dataset_with_extinction = dataset_at_time.assign(
            {CLOUD_EXTINCTION_VARIABLE: total_extinction}
        )
        grid = build_grid(
            dataset_with_extinction,
            varname=CLOUD_EXTINCTION_VARIABLE,
            grid_type=self.grid_type,
        )

        # Remember the range and a typical cell size for the transfer function.
        self._preview_extinction_max = float(np.nanmax(total_extinction.values))
        x_min, x_max, y_min, y_max, z_min, z_max = grid.bounds
        domain_volume = max((x_max - x_min) * (y_max - y_min) * (z_max - z_min), 0.0)
        self._preview_unit_distance = (
            (domain_volume / max(grid.n_points, 1)) ** (1.0 / 3.0)
            if domain_volume > 0
            else None
        )
        return grid

    def get_pyvista_kwargs(self) -> Dict[str, Any]:
        """
        White volume with Beer-Lambert opacity over one grid cell.

        VTK interprets each opacity as the opacity across
        ``opacity_unit_distance`` and compounds it along the ray as
        ``1 - (1 - a)**(step / unit)``. Setting
        ``a(beta) = 1 - exp(-beta * unit)`` makes that compounding exactly
        Beer-Lambert attenuation for every step length.
        """
        extinction_max = self._preview_extinction_max
        if not np.isfinite(extinction_max) or extinction_max <= 0.0:
            extinction_max = 1.0
        unit_distance = self._preview_unit_distance or 1.0

        # Opacity per unit distance at evenly spaced extinctions in [0, max].
        sampled_extinction = np.linspace(
            0.0, extinction_max, CLOUD_PREVIEW_OPACITY_SAMPLES
        )
        beer_lambert_opacity = 1.0 - np.exp(-sampled_extinction * unit_distance)

        kwargs: Dict[str, Any] = {
            "scalars": CLOUD_EXTINCTION_VARIABLE,
            "cmap": ["white", "white"],
            "clim": (0.0, extinction_max),
            "opacity": beer_lambert_opacity,
            "opacity_unit_distance": unit_distance,
        }
        kwargs.update(self.pyvista_add_kwargs)
        return kwargs


# Names of the dye fields on the preview grid and in the Blender export.
DYE_FRACTION_VARIABLE = "dye_fraction"
DYE_LABEL_VARIABLE = "dye_label"


@dataclass
class DyeSpec(VarSpec):
    """
    Passive tracers rendered as a colored dye.

    Each cell's dyed fraction sets how much dye it holds, and the
    tracer-weighted mean of the per-tracer values (its "label", e.g. the mean
    starting height of its air) sets the dye's color through the colormap.
    See :class:`~skyvista.geometry.DyeGeometry` for both definitions and
    :class:`~skyvista.appearance.DyeAppearance` for the optics.

    The Blender backend renders the dye as a scattering and absorbing volume
    in Cycles. ``Scene.show()`` draws a preview: the isosurface at
    ``preview_fraction``, colored by the label.
    """

    geometry: DyeGeometry = field(default_factory=DyeGeometry)
    appearance: DyeAppearance = field(default_factory=DyeAppearance)

    def __post_init__(self):
        if self.name is None:
            self.name = "dye"
        n_tracers = len(self.geometry.tracer_varnames)
        if n_tracers == 0:
            raise ValueError("DyeGeometry needs at least one tracer variable.")
        if (
            self.geometry.tracer_values is not None
            and len(self.geometry.tracer_values) != n_tracers
        ):
            raise ValueError(
                f"Got {len(self.geometry.tracer_values)} tracer_values for "
                f"{n_tracers} tracer variables; give one value per tracer."
            )
        if (
            self.geometry.min_vertical_displacement is not None
            and self.geometry.tracer_values is None
        ):
            raise ValueError(
                "min_vertical_displacement needs tracer_values: the starting "
                "height (m) of each tracer."
            )

    def tracer_label_values(self) -> List[float]:
        """The value each tracer contributes to the label (default: its index)."""
        if self.geometry.tracer_values is None:
            return [float(index) for index in range(len(self.geometry.tracer_varnames))]
        return [float(value) for value in self.geometry.tracer_values]

    def dye_fields(
        self, dataset_at_time: xr.Dataset
    ) -> "tuple[xr.DataArray, xr.DataArray]":
        """
        The dyed fraction and the dye label at one time.

        Tracers are accumulated one at a time, so memory holds only the two
        running sums, never all tracers at once. Each tracer is first
        clipped at 0 (advection undershoots leave small negative values)
        and, with ``min_vertical_displacement``, zeroed within that distance
        of its starting height. Where no dye is present the label is NaN.

        Returns:
            (fraction, label), both on the dataset's grid.
        """
        from .grids import resolve_coordinate

        geometry = self.geometry
        heights = None
        if geometry.min_vertical_displacement is not None:
            heights = dataset_at_time[resolve_coordinate(dataset_at_time, "z")]

        fraction_sum = None
        weighted_label_sum = None
        for tracer_name, label_value in zip(
            geometry.tracer_varnames, self.tracer_label_values(), strict=True
        ):
            # Share of each cell's air that came from this tracer's region.
            tracer_fraction = (
                dataset_at_time[tracer_name].clip(min=0.0).fillna(0.0)
                / geometry.reference_value
            )
            # Hide air still near its starting height (undisturbed layers).
            if heights is not None:
                vertical_displacement = abs(heights - label_value)
                tracer_fraction = tracer_fraction.where(
                    vertical_displacement > geometry.min_vertical_displacement, 0.0
                )
            # Running sums of the fraction and the fraction-weighted label.
            if fraction_sum is None:
                fraction_sum = tracer_fraction
                weighted_label_sum = tracer_fraction * label_value
            else:
                fraction_sum = fraction_sum + tracer_fraction
                weighted_label_sum = weighted_label_sum + tracer_fraction * label_value

        # Mean label of the dyed air, undefined where there is no dye.
        label = (weighted_label_sum / fraction_sum).where(fraction_sum > 0)
        return fraction_sum, label

    def create_mesh(self, ds: xr.Dataset, time: Any) -> Optional[pv.DataSet]:
        """Isosurface of the dyed fraction, carrying the label for coloring."""
        dataset_at_time = select_time(ds, time)
        fraction, label = self.dye_fields(dataset_at_time)
        dataset_with_dye = dataset_at_time.assign(
            {DYE_FRACTION_VARIABLE: fraction, DYE_LABEL_VARIABLE: label.fillna(0.0)}
        )
        grid = build_grid(
            dataset_with_dye, varname=DYE_FRACTION_VARIABLE, grid_type=self.grid_type
        )
        add_scalar_to_grid(grid, dataset_with_dye, DYE_LABEL_VARIABLE)
        surface = grid.contour(
            isosurfaces=[self.geometry.preview_fraction], scalars=DYE_FRACTION_VARIABLE
        )
        # Color the surface by the label interpolated onto it.
        surface = surface.sample(grid, pass_point_data=False, pass_cell_data=False)
        if surface.n_points:
            surface.set_active_scalars(DYE_LABEL_VARIABLE)
        return surface

    def get_pyvista_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_pyvista_kwargs()
        if self.appearance.color is None:
            kwargs["scalars"] = DYE_LABEL_VARIABLE
        return kwargs
