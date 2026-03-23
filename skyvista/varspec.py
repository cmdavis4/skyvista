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
    ContourAppearance,
    TrajectoryAppearance,
    VectorAppearance,
    VolumeAppearance,
)
from .geometry import (
    ContourGeometry,
    Geometry,
    SliceGeometry,
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
            mesh = mesh.sample(grid, pass_point_arrays=False, pass_cell_arrays=False)
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

    def create_mesh(self, ds: xr.Dataset, time: Any) -> Optional[pv.DataSet]:
        from carlee_tools import spacing

        from .grids import resolve_coordinates

        ds = select_time(ds, time)

        # Build grid (auto-detects type or uses explicit grid_type)
        grid = build_grid(ds, grid_type=self.grid_type)

        # Stack vector components
        u = ds[self.geometry.u_varname].values.ravel(order="F")
        v = ds[self.geometry.v_varname].values.ravel(order="F")
        w = ds[self.geometry.w_varname].values.ravel(order="F")

        vectors = np.vstack([u, v, w]).T
        vector_name = self.geometry.varname
        grid[vector_name] = vectors
        grid.set_active_vectors(vector_name)

        # Add individual components for potential scaling
        grid["u"] = u
        grid["v"] = v
        grid["w"] = w

        # Prepare glyph kwargs
        glyph_kwargs = {"orient": vector_name}

        # Auto-calculate factor if not provided
        factor = self.geometry.factor
        if factor is None:
            # Use resolved coordinate names
            coords = resolve_coordinates(ds, ["x", "y", "z"])
            component_to_dim = {"u": coords["x"], "v": coords["y"], "w": coords["z"]}
            ratios = []
            for component, dim in component_to_dim.items():
                if dim in ds.coords and len(ds[dim]) > 1:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        dim_spacing = spacing(
                            ds[dim].values, raise_if_not_evenly_spaced=False
                        )
                    max_component = np.abs(grid[component]).max()
                    if max_component > 0:
                        ratios.append(max_component / dim_spacing)
            if ratios:
                max_ratio = max(ratios)
                factor = 0.5 / max_ratio

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
                "Trajectory dataset must have 'trajectory_ix' or 'parcel_ix' dimension"
            )

        # Apply limit
        if self.limit and len(ds[trajectory_dim]) > self.limit:
            ds = ds.isel({trajectory_dim: slice(self.limit)})

        # Need at least 2 time points for trajectory rendering
        if "time" in ds.dims and len(ds["time"]) < 2:
            return pv.PolyData()

        # Handle particle style
        if self.appearance.style == "particle":
            return self._create_particle_mesh(ds, trajectory_dim)

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

    def _create_particle_mesh(self, ds: xr.Dataset, trajectory_dim: str) -> pv.DataSet:
        """Create particle-style mesh (spheres at final positions)."""
        from carlee_tools import maybe_cast_to_float

        # Get final time positions
        if "time" in ds.dims:
            ds = ds.isel(time=-1)

        points = np.column_stack(
            [
                ds["x"].values,
                ds["y"].values,
                ds["z"].values,
            ]
        )

        valid_mask = ~np.isnan(points).any(axis=1)
        points = points[valid_mask]

        point_mesh = pv.PolyData(points)

        # Add scalar data
        if self.geometry.scalar and self.geometry.scalar in ds:
            scalar_values = maybe_cast_to_float(ds[self.geometry.scalar].values)
            point_mesh[self.geometry.scalar] = scalar_values[valid_mask]

        # Convert to sphere glyphs
        glyph_kwargs = {
            "orient": False,
            "scale": False,
            "geom": pv.Sphere(radius=250, phi_resolution=8, theta_resolution=8),
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
