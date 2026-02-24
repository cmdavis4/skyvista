import datetime as dt
from warnings import warn
import matplotlib.colors as mcolors
import pyvista as pv
import xarray as xr
import numpy as np
from abc import abstractmethod, ABC
from copy import copy

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple, Optional, Union
from ..types_core import PathLike

from ..utils import to_kv_str, dt_to_str, NUMERICAL_DT_FORMAT


@dataclass
class PVConfig:
    from .plotter import initialize_plotter

    plotter: pv.Plotter = field(default_factory=initialize_plotter)
    animation: bool = False
    gif_path: Optional[PathLike] = None
    gif_scrubber: bool = False
    screenshot_path: Optional[PathLike] = None
    subplot_included_meshes: dict = field(default_factory=dict)
    title: Optional[str] = None
    interactive: bool = False
    export_html: bool = False
    fps: Optional[float] = 10.0
    show: bool = True
    callback: Optional[
        Callable[
            [
                pv.Plotter,
                Any,
                Optional[xr.Dataset],
                Optional[xr.Dataset],
                Dict[str, Any],
            ],
            None,
        ]
    ] = None

    def __post_init__(self):
        self.sanitize()

    def sanitize(self):
        if self.animation and not self.interactive and not self.gif_path:
            raise ValueError("Need to pass gif_path if creating an animation")


@dataclass(kw_only=True)
class PVVarSpec(ABC):
    varname: str
    name: Optional[str] = None
    scalar_bar: bool = False
    create_mesh_kwargs: Dict = field(default_factory=dict)
    add_mesh_kwargs: Dict = field(default_factory=dict)
    empty_ok: bool = False

    def make_varspec_name(self) -> str:
        # Determine spec type for unique naming
        if isinstance(self, PVContourSpec):
            spec_type = "contour"
        elif isinstance(self, PV2DSpec):
            spec_type = "2d"
        elif isinstance(self, PVVectorSpec):
            spec_type = "vector"
        elif isinstance(self, PVTrajectorySpec):
            spec_type = "trajectory"
        else:
            spec_type = "unknown"

        return to_kv_str({
            "type": spec_type,
            "category": self.varname,
            "varname": (
                self.varname
                if not (isinstance(self, PVContourSpec) and self.individual_meshes)
                else f"i{self.isosurfaces[0]}"
            ),
        })

    def __post_init__(self):
        self.name = self.name or self.make_varspec_name()


@dataclass
class PVContourSpec(PVVarSpec):
    isosurfaces: Optional[Union[List[float], Tuple[float, ...], np.ndarray]] = None
    individual_meshes: bool = False
    scalars: Optional[str] = (
        None  # If set, color isosurface by this variable instead of varname
    )


@dataclass
class PVVolumeSpec(PVVarSpec):
    individual_meshes: bool = False


@dataclass
class PV2DSpec(PVVarSpec):
    slice_dim: str = "z"  # Which dimension to slice: 'x', 'y', or 'z'
    slice_value: Optional[float] = (
        None  # Value at which to slice (None = use first index)
    )
    slice_method: str = "nearest"  # How to select: 'nearest' or 'interp'


@dataclass
class PVVectorSpec(PVVarSpec):
    u_varname: str = "UC"
    v_varname: str = "VC"
    w_varname: str = "WC"


@dataclass
class PVTrajectorySpec(PVVarSpec):
    color: Optional[str] = None
    scalar: Optional[str] = None
    scalar_bar: bool = False
    scalar_bar_args: dict = field(default_factory=dict)
    silhouettes: bool = False
    cmap: Optional[Union[str, mcolors.Colormap]] = None
    particles: bool = False
    body_radius: float = 70
    head_length_frac: float = 10
    head_radius_frac: float = 2.5
    head_radial_resolution: int = 30
    tube_resolution: int = 4
    label: Optional[str] = None


@dataclass
class PVData(ABC):
    varspecs: Tuple[PVVarSpec, ...]

    @abstractmethod
    def ds(self):
        pass

    @abstractmethod
    def _check_varspec(self, varspec):
        pass

    @abstractmethod
    def sanitize(self):
        pass

    def __setattr__(self, name, value):
        # Hacky way to be sure we call _check_varspec
        if name == "varspecs":
            for varspec in value:
                self._check_varspec(varspec)
        super().__setattr__(name, value)

    def __post_init__(self):
        self.sanitize()


@dataclass
class PVTrajectoryData(PVData):
    trajectory_ds: xr.Dataset
    n_parcel_limit: Optional[int] = 1000

    @property
    def ds(self):
        """Alias for trajectory_ds attribute."""
        return self.trajectory_ds

    def _check_varspec(self, varspec):
        if not isinstance(varspec, PVTrajectorySpec):
            raise ValueError(
                "PVTrajectoryData object only accepts PVTRajectorySpec varspecs"
            )

    def sanitize(self):
        if (
            "time" not in self.trajectory_ds.dims
            or len(self.trajectory_ds["time"]) == 0
        ):
            raise ValueError(
                "`trajectory_ds` was passed with a `time` dimension of length 0"
            )
        self.trajectory_ds = self.trajectory_ds.sortby(
            "time", ascending=True
        ).transpose("parcel_ix", "time")
        # Limit and warn if we're supposed to
        # This is so that you don't accidentally try to make it plot tens of thousands
        # of trajectories, which will make it hang or crash, and which I often do
        if self.n_parcel_limit and (
            len(self.trajectory_ds["parcel_ix"]) > self.n_parcel_limit
        ):
            warn(
                f"Limiting to {self.n_parcel_limit} trajectories from"
                f" {len(self.trajectory_ds['parcel_ix'])} total; pass"
                " trajectory_limit=None to remove this limit"
            )
            self.trajectory_ds = self.trajectory_ds.isel(
                parcel_ix=slice(self.n_parcel_limit)
            )


@dataclass
class PVRamsData(PVData):
    simulation_ds: xr.Dataset

    @property
    def ds(self):
        """Alias for trajectory_ds attribute."""
        return self.simulation_ds

    def _check_varspec(self, varspec):
        if isinstance(varspec, PVTrajectorySpec):
            raise ValueError(
                "PVRamsData object does not accept PVTRajectorySpec varspecs"
            )

    def _split_contour_isosurface_varspecs(self):
        new_varspecs: List[PVVarSpec] = []
        for varspec in self.varspecs:
            if isinstance(varspec, PVContourSpec) and varspec.individual_meshes:
                # Need to have passed isosurfaces in this case
                if varspec.isosurfaces is None or len(varspec.isosurfaces) == 0:
                    raise ValueError(
                        "Need to explicitly pass isosurfaces to create individual"
                        " meshes for each"
                    )
                # Split this into individual varspecs with one isosurface each
                for isosurface in varspec.isosurfaces:
                    new_varspec = copy(varspec)
                    # Override the name and isosurfaces
                    new_varspec.isosurfaces = [
                        isosurface,
                    ]
                    new_varspec.name = new_varspec.make_varspec_name()
                    new_varspecs.append(new_varspec)
            else:
                new_varspecs.append(varspec)
        self.varspecs = tuple(new_varspecs)

    def sanitize(self):
        # Transpose data to correct order
        self.simulation_ds = self.simulation_ds.transpose("x", "y", "z", ...)

        # Sort by time
        if "time" in self.simulation_ds.dims:
            self.simulation_ds = self.simulation_ds.sortby("time", ascending=True)

        # Check that our varspecs are not trajectories
        for varspec in self.varspecs:
            self._check_varspec(varspec)

        # Split out single isosurface varspecs if needed
        self._split_contour_isosurface_varspecs()


@dataclass(kw_only=True)
class PVMesh:
    varspec: PVVarSpec
    name: Optional[str] = None
    time: Optional[dt.datetime] = None
    mesh: Optional[pv.DataObject] = None
    actor: Optional[pv.Actor] = None

    def make_mesh_name(self) -> str:
        return (
            to_kv_str({
                "dt": dt_to_str(self.time, date_format=NUMERICAL_DT_FORMAT),
            })
            + "_"
            + self.varspec.name
        )

    def __post_init__(self):
        self.name = self.name or self.make_mesh_name()

    @property
    def mesh_empty(self):
        return len(self.mesh.points) == 0
