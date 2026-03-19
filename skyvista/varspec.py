"""
VarSpec classes for skyvista.

VarSpec classes are self-rendering visualization specifications that know how to:
1. Extract/create a mesh from data (create_mesh)
2. Provide rendering parameters for PyVista (get_pyvista_kwargs)
3. Provide export configuration for Blender (get_blender_config)
4. Render themselves to a plotter (render_to_plotter)

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
from .grid_utils import add_scalar_to_grid, build_rectilinear_grid, select_time

if TYPE_CHECKING:
    from .types_sv import PVMesh


@dataclass
class VarSpec(ABC):
    """
    Base class for self-rendering visualization specifications.

    Each VarSpec subclass knows how to create a mesh from data and how to
    render it. This allows the Scene to be agnostic about visualization types.

    Attributes:
        name: Unique identifier for this spec (auto-generated if None)
        empty_ok: If True, don't skip this spec when mesh is empty
        pyvista_create_kwargs: Escape hatch for PyVista mesh creation kwargs
        pyvista_add_kwargs: Escape hatch for PyVista add_mesh kwargs
    """

    name: Optional[str] = None
    empty_ok: bool = False

    # Escape hatches for edge cases
    pyvista_create_kwargs: Dict[str, Any] = field(default_factory=dict)
    pyvista_add_kwargs: Dict[str, Any] = field(default_factory=dict)

    @property
    @abstractmethod
    def geometry(self) -> Geometry:
        """Return the geometry specification."""
        ...

    @property
    @abstractmethod
    def appearance(self) -> Appearance:
        """Return the appearance specification."""
        ...

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

    def get_blender_config(self) -> Dict[str, Any]:
        """Get configuration for Blender export."""
        config = self.appearance.to_blender_config()
        # Add geometry info that Blender might need
        config["spec_type"] = self.__class__.__name__
        if self.name:
            config["name"] = self.name
        return config

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
        from .types_sv import PVMesh

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

    _geometry: ContourGeometry = field(
        default_factory=lambda: ContourGeometry(varname="")
    )
    _appearance: ContourAppearance = field(default_factory=ContourAppearance)

    @property
    def geometry(self) -> ContourGeometry:
        return self._geometry

    @property
    def appearance(self) -> ContourAppearance:
        return self._appearance

    def __post_init__(self):
        if self.name is None:
            iso_str = ""
            if (self._geometry.isosurfaces is not None) and (
                len(self._geometry.isosurfaces) > 0
            ):
                iso_str = f"_iso{self._geometry.isosurfaces[0]}"
            self.name = f"contour_{self._geometry.varname}{iso_str}"

    def create_mesh(self, ds: xr.Dataset, time: Any) -> Optional[pv.DataSet]:
        ds = select_time(ds, time)

        # Build rectilinear grid
        grid = build_rectilinear_grid(ds)

        # Add scalar data
        varname = self._geometry.varname
        add_scalar_to_grid(grid, ds, varname)

        # Create contour
        contour_kwargs = {"scalars": varname}
        if self._geometry.isosurfaces is not None:
            contour_kwargs["isosurfaces"] = self._geometry.isosurfaces
        contour_kwargs.update(self.pyvista_create_kwargs)

        mesh = grid.contour(**contour_kwargs)

        # Sample scalar field if different from contour variable
        if self._geometry.scalar and self._geometry.scalar != varname:
            add_scalar_to_grid(grid, ds, self._geometry.scalar)
            mesh = mesh.sample(grid, pass_point_arrays=False, pass_cell_arrays=False)
            mesh.set_active_scalars(self._geometry.scalar)

        return mesh


@dataclass
class VolumeSpec(VarSpec):
    """
    Volume rendering specification.

    Renders scalar field data as a 3D volume with opacity transfer function.
    """

    _geometry: VolumeGeometry = field(
        default_factory=lambda: VolumeGeometry(varname="")
    )
    _appearance: VolumeAppearance = field(default_factory=VolumeAppearance)

    @property
    def geometry(self) -> VolumeGeometry:
        return self._geometry

    @property
    def appearance(self) -> VolumeAppearance:
        return self._appearance

    def __post_init__(self):
        if self.name is None:
            self.name = f"volume_{self._geometry.varname}"

    def is_volume(self) -> bool:
        return True

    def create_mesh(self, ds: xr.Dataset, time: Any) -> Optional[pv.DataSet]:
        ds = select_time(ds, time)

        # Build rectilinear grid
        grid = build_rectilinear_grid(ds)

        # Add scalar data
        varname = self._geometry.varname
        add_scalar_to_grid(grid, ds, varname)

        # Apply threshold if specified
        if self._geometry.threshold:
            low, high = self._geometry.threshold
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
        kwargs["scalars"] = self._geometry.varname

        return kwargs


@dataclass
class VectorSpec(VarSpec):
    """
    Vector field glyph specification.

    Creates arrow glyphs from vector field data.
    """

    _geometry: VectorGeometry = field(
        default_factory=lambda: VectorGeometry(varname="")
    )
    _appearance: VectorAppearance = field(default_factory=VectorAppearance)

    @property
    def geometry(self) -> VectorGeometry:
        return self._geometry

    @property
    def appearance(self) -> VectorAppearance:
        return self._appearance

    def __post_init__(self):
        if self.name is None:
            self.name = f"vector_{self._geometry.varname}"

    def create_mesh(self, ds: xr.Dataset, time: Any) -> Optional[pv.DataSet]:
        from carlee_tools import spacing

        ds = select_time(ds, time)

        # Build rectilinear grid
        grid = build_rectilinear_grid(ds)

        # Stack vector components
        u = ds[self._geometry.u_varname].values.ravel(order="F")
        v = ds[self._geometry.v_varname].values.ravel(order="F")
        w = ds[self._geometry.w_varname].values.ravel(order="F")

        vectors = np.vstack([u, v, w]).T
        vector_name = self._geometry.varname
        grid[vector_name] = vectors
        grid.set_active_vectors(vector_name)

        # Add individual components for potential scaling
        grid["u"] = u
        grid["v"] = v
        grid["w"] = w

        # Prepare glyph kwargs
        glyph_kwargs = {"orient": vector_name}

        # Auto-calculate factor if not provided
        factor = self._geometry.factor
        if factor is None:
            component_to_dim = {"u": "x", "v": "y", "w": "z"}
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
        if self._geometry.scale_by:
            add_scalar_to_grid(grid, ds, self._geometry.scale_by)
            glyph_kwargs["scale"] = self._geometry.scale_by
        else:
            glyph_kwargs["scale"] = vector_name

        if self._geometry.tolerance is not None:
            glyph_kwargs["tolerance"] = self._geometry.tolerance

        glyph_kwargs.update(self.pyvista_create_kwargs)

        return grid.glyph(**glyph_kwargs)


@dataclass
class SliceSpec(VarSpec):
    """
    2D slice visualization specification.

    Extracts a 2D slice from 3D scalar field data.
    """

    _geometry: SliceGeometry = field(default_factory=lambda: SliceGeometry(varname=""))
    _appearance: Appearance = field(default_factory=Appearance)

    @property
    def geometry(self) -> SliceGeometry:
        return self._geometry

    @property
    def appearance(self) -> Appearance:
        return self._appearance

    def __post_init__(self):
        if self.name is None:
            self.name = f"slice_{self._geometry.varname}_{self._geometry.slice_dim}"

    def create_mesh(self, ds: xr.Dataset, time: Any) -> Optional[pv.DataSet]:
        ds = select_time(ds, time)

        # Get slice parameters
        slice_dim = self._geometry.slice_dim
        slice_value = self._geometry.slice_value
        slice_method = self._geometry.slice_method

        # Select the slice from the data
        if slice_value is None:
            slice_sel = {slice_dim: ds[slice_dim].values[0]}
        else:
            slice_sel = {slice_dim: slice_value}

        sliced_data = ds.sel(slice_sel, method=slice_method)

        # Determine remaining dimensions
        remaining_dims = [d for d in ["x", "y", "z"] if d != slice_dim]
        dim1, dim2 = remaining_dims

        # Create meshgrid
        coord1 = sliced_data[dim1].values
        coord2 = sliced_data[dim2].values
        grid1, grid2 = np.meshgrid(coord1, coord2, indexing="ij")

        # Create constant coordinate for sliced dimension
        if slice_value is None:
            slice_coord_value = sliced_data[slice_dim].values.item()
        else:
            slice_coord_value = slice_value
        grid_sliced = np.ones_like(grid1) * slice_coord_value

        # Map coordinates to x, y, z
        coords = {slice_dim: grid_sliced, dim1: grid1, dim2: grid2}

        create_kwargs = dict(self.pyvista_create_kwargs)
        mesh = pv.StructuredGrid(coords["x"], coords["y"], coords["z"], **create_kwargs)

        # Add variable data
        varname = self._geometry.varname
        var_data = sliced_data[varname].values
        mesh[varname] = var_data.ravel(order="F")

        return mesh


@dataclass
class TrajectorySpec(VarSpec):
    """
    Lagrangian trajectory visualization specification.

    Creates tube or particle visualizations from trajectory data.
    """

    _geometry: TrajectoryGeometry = field(default_factory=TrajectoryGeometry)
    _appearance: TrajectoryAppearance = field(default_factory=TrajectoryAppearance)
    limit: Optional[int] = 1000

    @property
    def geometry(self) -> TrajectoryGeometry:
        return self._geometry

    @property
    def appearance(self) -> TrajectoryAppearance:
        return self._appearance

    def __post_init__(self):
        if self.name is None:
            style = self._appearance.style
            scalar_part = f"_{self._geometry.scalar}" if self._geometry.scalar else ""
            self.name = f"trajectory_{style}{scalar_part}"

    def create_mesh(self, ds: xr.Dataset, time: Any) -> Optional[pv.DataSet]:
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

        # Import and use existing trajectory mesh generation
        from .trajectories import create_trajectory_mesh

        # Extract trajectory data
        trajectories_points_data = self._extract_trajectory_data(ds, trajectory_dim)

        if not trajectories_points_data:
            return pv.PolyData()

        # Handle particle style
        if self._appearance.style == "particle":
            return self._create_particle_mesh(ds, trajectory_dim)

        # Create tube mesh
        arrow_color_scalar = self._geometry.scalar
        trajectory_arrow_kwargs = {
            "body_radius": self._geometry.tube_radius,
            "head_length_frac": self._geometry.head_length_frac,
            "head_radius_frac": self._geometry.head_radius_frac,
            "head_radial_res": self._geometry.head_radial_resolution,
            "tube_resolution": self._geometry.tube_resolution,
        }

        mesh = create_trajectory_mesh(
            trajectories_points_data,
            arrow_color_scalar=arrow_color_scalar,
            create_mesh_kwargs=self.pyvista_create_kwargs,
            **trajectory_arrow_kwargs,
        )

        # Store scalar info
        if arrow_color_scalar:
            mesh.field_data["arrow_color_scalar"] = arrow_color_scalar

        return mesh

    def _extract_trajectory_data(
        self, ds: xr.Dataset, trajectory_dim: str
    ) -> List[Dict[str, Any]]:
        """Extract trajectory point data from dataset."""
        from carlee_tools import maybe_cast_to_float

        trajectories_points_data = []
        arrow_color_scalar = self._geometry.scalar

        x_data = ds["x"].values
        y_data = ds["y"].values
        z_data = ds["z"].values
        trajectory_indices = ds[trajectory_dim].values

        # Pre-extract scalar data if needed
        arrow_color_scalar_data = None
        if arrow_color_scalar and arrow_color_scalar in ds:
            arrow_color_scalar_data = maybe_cast_to_float(ds[arrow_color_scalar].values)

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
            if arrow_color_scalar and arrow_color_scalar_data is not None:
                if len(arrow_color_scalar_data.shape) > 1:
                    trajectory_points_data[arrow_color_scalar] = (
                        arrow_color_scalar_data[i][valid_mask]
                    )
                else:
                    trajectory_points_data[arrow_color_scalar] = np.full(
                        len(valid_points), arrow_color_scalar_data[i]
                    )

            trajectories_points_data.append(trajectory_points_data)

        return trajectories_points_data

    def _create_particle_mesh(self, ds: xr.Dataset, trajectory_dim: str) -> pv.DataSet:
        """Create particle-style mesh (spheres at final positions)."""
        from carlee_tools import maybe_cast_to_float

        # Get final time positions
        if "time" in ds.dims:
            ds = ds.isel(time=-1)

        points = np.column_stack([
            ds["x"].values,
            ds["y"].values,
            ds["z"].values,
        ])

        valid_mask = ~np.isnan(points).any(axis=1)
        points = points[valid_mask]

        point_mesh = pv.PolyData(points)

        # Add scalar data
        if self._geometry.scalar and self._geometry.scalar in ds:
            scalar_values = maybe_cast_to_float(ds[self._geometry.scalar].values)
            point_mesh[self._geometry.scalar] = scalar_values[valid_mask]

        # Convert to sphere glyphs
        glyph_kwargs = {
            "orient": False,
            "scale": False,
            "geom": pv.Sphere(radius=250, phi_resolution=8, theta_resolution=8),
        }
        glyph_kwargs.update(self.pyvista_create_kwargs)

        return point_mesh.glyph(**glyph_kwargs)

    def get_pyvista_kwargs(self) -> Dict[str, Any]:
        kwargs = super().get_pyvista_kwargs()

        # Handle silhouettes
        if self._appearance.silhouettes:
            kwargs["silhouette"] = True

        return kwargs
