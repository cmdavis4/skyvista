"""
Tests for skyvista grid builders and coordinate resolution.
"""

import numpy as np
import pytest
import pyvista as pv
import xarray as xr

from skyvista.grids import (
    RectilinearGridBuilder,
    CurvilinearGridBuilder,
    GeographicGridBuilder,
    SphericalGridBuilder,
    UnstructuredGridBuilder,
    detect_grid_type,
    get_grid_builder,
    resolve_coordinate,
    resolve_coordinates,
    resolve_spherical_coordinates,
    is_spherical_grid,
    is_geographic_grid,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def rectilinear_ds():
    """Dataset with simple 1D x/y/z coordinates."""
    x = np.linspace(0, 100, 10)
    y = np.linspace(0, 200, 15)
    z = np.linspace(0, 5000, 8)
    data = np.random.randn(10, 15, 8)
    return xr.Dataset(
        {"temperature": (["x", "y", "z"], data)},
        coords={"x": x, "y": y, "z": z},
    )


@pytest.fixture
def geographic_ds():
    """Dataset with lon/lat/altitude coordinates."""
    lon = np.linspace(-105, -100, 12)
    lat = np.linspace(38, 42, 10)
    alt = np.linspace(0, 15000, 6)
    data = np.random.randn(12, 10, 6)
    return xr.Dataset(
        {"temperature": (["lon", "lat", "altitude"], data)},
        coords={"lon": lon, "lat": lat, "altitude": alt},
    )


@pytest.fixture
def spherical_ds():
    """Dataset with range/azimuth/elevation coordinates (1D)."""
    ranges = np.linspace(1000, 100000, 20)
    azimuths = np.linspace(0, 350, 36)
    elevations = np.array([0.5, 1.5, 2.4, 3.4, 5.3])
    data = np.random.randn(20, 36, 5) * 10 + 20
    return xr.Dataset(
        {"reflectivity": (["range", "azimuth", "elevation"], data)},
        coords={
            "range": ranges,
            "azimuth": azimuths,
            "elevation": elevations,
        },
    )


@pytest.fixture
def spherical_2d_ds():
    """Dataset with range (1D) and azimuth/elevation as 2D coords."""
    n_range = 20
    n_vscan = 5
    n_time = 3
    ranges = np.linspace(1000, 100000, n_range)
    az_base = np.linspace(0, 350, n_vscan)
    time_offset = np.arange(n_time) * 36
    azimuth_2d = (az_base[:, np.newaxis] + time_offset[np.newaxis, :]) % 360
    el_base = np.array([0.5, 1.5, 2.4, 3.4, 5.3])
    elevation_2d = np.broadcast_to(
        el_base[:, np.newaxis], (n_vscan, n_time)
    ).copy()
    data = np.random.randn(n_range, n_vscan, n_time) * 10 + 20
    return xr.Dataset(
        {"reflectivity": (["range", "volume_scan", "time"], data)},
        coords={
            "range": ranges,
            "azimuth": (["volume_scan", "time"], azimuth_2d),
            "elevation": (["volume_scan", "time"], elevation_2d),
            "volume_scan": np.arange(n_vscan),
            "time": np.arange(n_time),
        },
    )


@pytest.fixture
def curvilinear_ds():
    """Dataset with 2D coordinate arrays (like WRF output)."""
    ni, nj, nk = 8, 10, 5
    # 2D lat/lon-like coordinates that aren't geographic names
    xc = np.random.randn(ni, nj) * 1000
    yc = np.random.randn(ni, nj) * 1000
    zc = np.linspace(0, 5000, nk)
    data = np.random.randn(ni, nj, nk)
    return xr.Dataset(
        {"wind": (["i", "j", "k"], data)},
        coords={
            "x": (["i", "j"], xc),
            "y": (["i", "j"], yc),
            "z": ("k", zc),
        },
    )


# =============================================================================
# COORDINATE RESOLUTION TESTS
# =============================================================================


class TestCoordinateResolution:
    def test_resolve_by_name(self, rectilinear_ds):
        assert resolve_coordinate(rectilinear_ds, "x") == "x"
        assert resolve_coordinate(rectilinear_ds, "y") == "y"
        assert resolve_coordinate(rectilinear_ds, "z") == "z"

    def test_resolve_by_cf_axis_attribute(self):
        ds = xr.Dataset(
            {"temp": (["a", "b", "c"], np.zeros((3, 4, 5)))},
            coords={
                "a": xr.DataArray(range(3), dims="a", attrs={"axis": "X"}),
                "b": xr.DataArray(range(4), dims="b", attrs={"axis": "Y"}),
                "c": xr.DataArray(range(5), dims="c", attrs={"axis": "Z"}),
            },
        )
        assert resolve_coordinate(ds, "x") == "a"
        assert resolve_coordinate(ds, "y") == "b"
        assert resolve_coordinate(ds, "z") == "c"

    def test_resolve_geographic_aliases(self, geographic_ds):
        assert resolve_coordinate(geographic_ds, "x") == "lon"
        assert resolve_coordinate(geographic_ds, "y") == "lat"
        assert resolve_coordinate(geographic_ds, "z") == "altitude"

    def test_resolve_missing_coordinate_error(self):
        ds = xr.Dataset(
            {"temp": (["a", "b", "c"], np.zeros((3, 4, 5)))},
            coords={"a": range(3), "b": range(4), "c": range(5)},
        )
        with pytest.raises(ValueError, match="x-axis"):
            resolve_coordinate(ds, "x")

    def test_resolve_multiple(self, rectilinear_ds):
        coords = resolve_coordinates(rectilinear_ds, ["x", "y", "z"])
        assert coords == {"x": "x", "y": "y", "z": "z"}

    def test_resolve_spherical(self, spherical_ds):
        coords = resolve_spherical_coordinates(spherical_ds)
        assert coords == {
            "range": "range",
            "azimuth": "azimuth",
            "elevation": "elevation",
        }

    def test_resolve_spherical_missing(self, rectilinear_ds):
        assert resolve_spherical_coordinates(rectilinear_ds) is None


# =============================================================================
# GRID TYPE DETECTION TESTS
# =============================================================================


class TestGridTypeDetection:
    def test_detect_rectilinear(self, rectilinear_ds):
        builder = detect_grid_type(rectilinear_ds)
        assert isinstance(builder, RectilinearGridBuilder)
        assert builder.grid_type == "rectilinear"

    def test_detect_geographic(self, geographic_ds):
        builder = detect_grid_type(geographic_ds)
        assert isinstance(builder, GeographicGridBuilder)
        assert builder.grid_type == "geographic"

    def test_detect_spherical(self, spherical_ds):
        builder = detect_grid_type(spherical_ds)
        assert isinstance(builder, SphericalGridBuilder)
        assert builder.grid_type == "spherical"

    def test_detect_spherical_2d(self, spherical_2d_ds):
        builder = detect_grid_type(spherical_2d_ds)
        assert isinstance(builder, SphericalGridBuilder)

    def test_detect_curvilinear(self, curvilinear_ds):
        builder = detect_grid_type(curvilinear_ds)
        assert isinstance(builder, CurvilinearGridBuilder)
        assert builder.grid_type == "curvilinear"

    def test_detect_unknown_raises(self):
        ds = xr.Dataset(
            {"temp": (["a", "b", "c"], np.zeros((3, 4, 5)))},
            coords={"a": range(3), "b": range(4), "c": range(5)},
        )
        with pytest.raises(ValueError, match="auto-detect"):
            detect_grid_type(ds)

    def test_get_grid_builder_explicit(self, rectilinear_ds):
        builder = get_grid_builder(rectilinear_ds, grid_type="rectilinear")
        assert isinstance(builder, RectilinearGridBuilder)

    def test_get_grid_builder_invalid_type(self, rectilinear_ds):
        with pytest.raises(ValueError, match="Unknown grid type"):
            get_grid_builder(rectilinear_ds, grid_type="magic")

    def test_is_spherical_grid(self, spherical_ds, rectilinear_ds):
        assert is_spherical_grid(spherical_ds) is True
        assert is_spherical_grid(rectilinear_ds) is False


# =============================================================================
# MESH BUILDING TESTS
# =============================================================================


class TestMeshBuilding:
    def test_rectilinear_build(self, rectilinear_ds):
        builder = detect_grid_type(rectilinear_ds)
        mesh = builder.build_mesh(rectilinear_ds, varname="temperature")
        assert isinstance(mesh, pv.RectilinearGrid)
        assert "temperature" in mesh.point_data
        assert mesh.n_points == 10 * 15 * 8

    def test_rectilinear_build_no_varname(self, rectilinear_ds):
        builder = detect_grid_type(rectilinear_ds)
        mesh = builder.build_mesh(rectilinear_ds)
        assert isinstance(mesh, pv.RectilinearGrid)
        assert len(mesh.point_data) == 0

    def test_geographic_build(self, geographic_ds):
        builder = detect_grid_type(geographic_ds)
        mesh = builder.build_mesh(geographic_ds, varname="temperature")
        assert isinstance(mesh, pv.StructuredGrid)
        assert "temperature" in mesh.point_data
        assert mesh.n_points == 12 * 10 * 6

    def test_spherical_build(self, spherical_ds):
        builder = detect_grid_type(spherical_ds)
        mesh = builder.build_mesh(spherical_ds, varname="reflectivity")
        assert isinstance(mesh, pv.StructuredGrid)
        assert "reflectivity" in mesh.point_data
        assert mesh.n_points == 20 * 36 * 5

    def test_spherical_2d_build(self, spherical_2d_ds):
        """Regression test: 2D azimuth/elevation should not raise."""
        builder = detect_grid_type(spherical_2d_ds)
        mesh = builder.build_mesh(spherical_2d_ds, varname="reflectivity")
        assert isinstance(mesh, pv.StructuredGrid)
        assert "reflectivity" in mesh.point_data
        assert mesh.n_points == 20 * 5 * 3

    def test_curvilinear_build(self, curvilinear_ds):
        builder = detect_grid_type(curvilinear_ds)
        mesh = builder.build_mesh(curvilinear_ds, varname="wind")
        assert isinstance(mesh, pv.StructuredGrid)
        assert "wind" in mesh.point_data


# =============================================================================
# ADD SCALAR TESTS
# =============================================================================


class TestAddScalar:
    def test_missing_varname_error(self, rectilinear_ds):
        builder = detect_grid_type(rectilinear_ds)
        mesh = builder.build_mesh(rectilinear_ds)
        with pytest.raises(KeyError, match="nonexistent"):
            builder.add_scalar(mesh, rectilinear_ds, "nonexistent")

    def test_no_dim_overlap_error(self, rectilinear_ds):
        """Variable with completely unrelated dimensions."""
        builder = detect_grid_type(rectilinear_ds)
        mesh = builder.build_mesh(rectilinear_ds)
        ds_extra = rectilinear_ds.assign(
            unrelated=xr.DataArray(
                np.zeros((3, 4)), dims=["a", "b"]
            )
        )
        with pytest.raises(ValueError, match="No dimensions overlap"):
            builder.add_scalar(mesh, ds_extra, "unrelated")

    def test_extra_dims_error(self, rectilinear_ds):
        """Variable with extra time dimension that wasn't selected."""
        builder = detect_grid_type(rectilinear_ds)
        mesh = builder.build_mesh(rectilinear_ds)
        ds_extra = rectilinear_ds.assign(
            timed=xr.DataArray(
                np.zeros((10, 15, 8, 3)), dims=["x", "y", "z", "time"]
            )
        )
        with pytest.raises(ValueError, match="extra dimensions"):
            builder.add_scalar(mesh, ds_extra, "timed")


# =============================================================================
# BOUNDS MESH TESTS
# =============================================================================


class TestBoundsMesh:
    def test_rectilinear_bounds(self, rectilinear_ds):
        builder = detect_grid_type(rectilinear_ds)
        bounds = builder.create_bounds_mesh(rectilinear_ds)
        assert isinstance(bounds, pv.PolyData)
        assert bounds.n_points > 0

    def test_geographic_bounds(self, geographic_ds):
        builder = detect_grid_type(geographic_ds)
        bounds = builder.create_bounds_mesh(geographic_ds)
        assert isinstance(bounds, pv.PolyData)

    def test_spherical_bounds(self, spherical_ds):
        builder = detect_grid_type(spherical_ds)
        bounds = builder.create_bounds_mesh(spherical_ds)
        assert isinstance(bounds, pv.PolyData)
