"""
Basic tests for the pvplotting module.

These tests cover the core functionality of the pvplotting module including
data classes, utility functions, and basic mesh creation logic.
"""

from cloudy_experimental.pvplotting.types_pvplotting import (
    PV2DSpec,
    PVConfig,
    PVContourSpec,
    PVMesh,
    PVRamsData,
    PVTrajectoryData,
    PVTrajectorySpec,
    PVVarSpec,
    PVVectorSpec,
)
import pytest
import numpy as np
import xarray as xr
import datetime as dt
from pathlib import Path
from unittest.mock import Mock, patch

# Import the module components to test
from cloudy_experimental.pvplotting.core_pvplotting import (
    sanitize_inputs,
    rectangle_mesh,
    _create_meshes_for_frame,
)


class TestDataClasses:
    """Test the core data classes."""

    def test_pv_config_defaults(self):
        """Test PVConfig initialization with defaults."""
        config = PVConfig()
        assert config.animation is False
        assert config.gif_path is None
        assert config.interactive is False
        assert config.fps == 10.0
        assert config.show is True
        assert config.plotter is not None  # Just verify plotter exists
        # Verify plotter is a PyVista Plotter object
        import pyvista as pv

        assert isinstance(config.plotter, pv.Plotter)

    def test_pv_config_animation_validation(self):
        """Test PVConfig validation for animation settings."""
        with patch("cloudy_experimental.pvplotting.plotter.initialize_plotter"):
            # Should raise error if animation=True but no gif_path and not interactive
            with pytest.raises(
                ValueError, match="Need to pass gif_path if creating an animation"
            ):
                PVConfig(animation=True, interactive=False, gif_path=None)

    def test_pv_varspec_basic(self):
        """Test basic PVVarSpec creation."""
        spec = PVVarSpec(varname="temperature")
        assert spec.varname == "temperature"
        assert spec.scalar_bar is False
        assert spec.add_mesh_kwargs == {}

    def test_pv_contour_spec(self):
        """Test PVContourSpec creation and defaults."""
        spec = PVContourSpec(varname="temperature", isosurfaces=[15, 20, 25])
        assert spec.varname == "temperature"
        assert spec.isosurfaces == [15, 20, 25]
        assert spec.individual_meshes is False

    def test_pv_trajectory_spec_defaults(self):
        """Test PVTrajectorySpec default values."""
        spec = PVTrajectorySpec(varname="trajectories")
        assert spec.varname == "trajectories"
        assert spec.color is None
        assert spec.particles is False
        assert spec.body_radius == 70
        assert spec.head_length_frac == 10
        assert spec.head_radius_frac == 2.5


class TestTrajectoryData:
    """Test PVTrajectoryData functionality."""

    def create_sample_trajectory_ds(self, n_parcels=10, n_times=5):
        """Create a sample trajectory dataset for testing."""
        times = np.arange(n_times)
        parcel_ix = np.arange(n_parcels)

        # Create random trajectory data
        x = np.random.rand(n_parcels, n_times) * 1000
        y = np.random.rand(n_parcels, n_times) * 1000
        z = np.random.rand(n_parcels, n_times) * 1000

        ds = xr.Dataset(
            {
                "x": (["parcel_ix", "time"], x),
                "y": (["parcel_ix", "time"], y),
                "z": (["parcel_ix", "time"], z),
            },
            coords={"parcel_ix": parcel_ix, "time": times},
        )
        return ds

    def test_trajectory_data_basic(self):
        """Test basic PVTrajectoryData creation."""
        ds = self.create_sample_trajectory_ds()
        traj_data = PVTrajectoryData(trajectory_ds=ds, varspecs=())
        # After sanitization, ds may be modified, so check structure rather than identity
        assert len(traj_data.ds.dims) == 2
        assert "parcel_ix" in traj_data.ds.dims
        assert "time" in traj_data.ds.dims
        assert traj_data.n_parcel_limit == 1000

    def test_trajectory_data_sanitize_sorting(self):
        """Test that sanitize properly sorts by time."""
        # Create dataset with times out of order
        times = np.array([2, 0, 1])
        ds = self.create_sample_trajectory_ds(n_times=3)
        ds = ds.assign_coords(time=times)

        traj_data = PVTrajectoryData(trajectory_ds=ds, varspecs=())
        traj_data.sanitize()

        # Check that times are now sorted
        expected_times = np.array([0, 1, 2])
        np.testing.assert_array_equal(traj_data.ds.time.values, expected_times)

    def test_trajectory_data_parcel_limit(self):
        """Test parcel limiting functionality."""
        ds = self.create_sample_trajectory_ds(n_parcels=2000)  # More than default limit

        # Warning is emitted during initialization
        with pytest.warns(UserWarning, match="Limiting to 100 trajectories"):
            traj_data = PVTrajectoryData(
                trajectory_ds=ds, n_parcel_limit=100, varspecs=()
            )

        assert len(traj_data.ds.parcel_ix) == 100

    def test_trajectory_data_empty_time_error(self):
        """Test error handling for empty time dimension."""
        # Create dataset with empty time dimension
        ds = xr.Dataset(
            {"x": (["parcel_ix", "time"], np.array([]).reshape(0, 0))},
            coords={"parcel_ix": [], "time": []},
        )

        # Error should be raised during initialization due to sanitize in __post_init__
        with pytest.raises(ValueError, match="time.*dimension of length 0"):
            PVTrajectoryData(trajectory_ds=ds, varspecs=())


class TestRamsData:
    """Test PVRamsData functionality."""

    def create_sample_rams_ds(self, nx=10, ny=10, nz=5, n_times=3):
        """Create a sample RAMS dataset for testing."""
        x = np.linspace(0, 1000, nx)
        y = np.linspace(0, 1000, ny)
        z = np.linspace(0, 5000, nz)
        times = np.arange(n_times)

        # Create sample temperature field
        temp = np.random.rand(nx, ny, nz, n_times) * 30 + 273.15  # Kelvin

        ds = xr.Dataset(
            {"temperature": (["x", "y", "z", "time"], temp)},
            coords={"x": x, "y": y, "z": z, "time": times},
        )
        return ds

    def test_rams_data_basic(self):
        """Test basic PVRamsData creation."""
        ds = self.create_sample_rams_ds()
        rams_data = PVRamsData(simulation_ds=ds, varspecs=())
        # After sanitization, ds may be modified, so check structure rather than identity
        assert "temperature" in rams_data.ds.data_vars
        assert "x" in rams_data.ds.dims
        assert "y" in rams_data.ds.dims
        assert "z" in rams_data.ds.dims

    def test_rams_data_sanitize_transpose(self):
        """Test that sanitize properly transposes dimensions."""
        # Create dataset with dimensions in wrong order
        ds = self.create_sample_rams_ds()
        ds = ds.transpose("time", "z", "y", "x")  # Wrong order

        rams_data = PVRamsData(simulation_ds=ds, varspecs=())
        rams_data.sanitize()

        # Check that dimensions are in correct order
        expected_dims = ("x", "y", "z", "time")
        assert rams_data.ds.temperature.dims == expected_dims


class TestUtilityFunctions:
    """Test utility functions."""

    def test_rectangle_mesh_xy_plane(self):
        """Test rectangle_mesh with z as singleton dimension."""
        x = np.linspace(0, 10, 5)
        y = np.linspace(0, 5, 3)
        z = np.array([1.0])  # Singleton

        mesh = rectangle_mesh(x, y, z)

        # Check that mesh has correct shape
        assert mesh.n_points == len(x) * len(y)
        # Check z coordinates are constant
        assert np.allclose(mesh.points[:, 2], 1.0)

    def test_rectangle_mesh_invalid_input(self):
        """Test rectangle_mesh error handling."""
        x = np.linspace(0, 10, 5)
        y = np.linspace(0, 5, 3)
        z = np.linspace(0, 2, 4)  # No singleton dimension

        with pytest.raises(
            ValueError, match="Exactly one of x, y, or z must be of length 1"
        ):
            rectangle_mesh(x, y, z)

    def test_sanitize_inputs_both_none(self):
        """Test sanitize_inputs error when both inputs are None."""
        with pytest.raises(ValueError, match="Must pass at least one"):
            sanitize_inputs(rams_data=None, trajectory_data=None)


class TestMeshCreation:
    """Test basic mesh creation functionality."""

    def create_simple_rams_data(self, varspecs=()):
        """Create simple RAMS data for testing."""
        x = np.linspace(0, 1000, 3)
        y = np.linspace(0, 1000, 3)
        z = np.linspace(0, 1000, 3)

        # Simple temperature field
        temp = np.ones((3, 3, 3)) * 300
        temp[1, 1, 1] = 320  # Hot spot

        ds = xr.Dataset(
            {"temperature": (["x", "y", "z"], temp)}, coords={"x": x, "y": y, "z": z}
        )
        return PVRamsData(simulation_ds=ds, varspecs=varspecs)

    def create_simple_trajectory_data(self, varspecs=()):
        """Create simple trajectory data for testing."""
        times = np.array([0, 1, 2])
        parcel_ix = np.array([0, 1])

        # Simple straight-line trajectories
        x = np.array([[0, 100, 200], [0, 150, 300]])  # 2 parcels, 3 times
        y = np.array([[0, 100, 200], [50, 150, 250]])
        z = np.array([[100, 150, 200], [200, 250, 300]])

        ds = xr.Dataset(
            {
                "x": (["parcel_ix", "time"], x),
                "y": (["parcel_ix", "time"], y),
                "z": (["parcel_ix", "time"], z),
            },
            coords={"parcel_ix": parcel_ix, "time": times},
        )
        return PVTrajectoryData(trajectory_ds=ds, varspecs=varspecs)

    @patch(
        "cloudy_experimental.pvplotting.core_pvplotting.generate_trajectory_meshes_single_mesh"
    )
    def test_create_meshes_for_frame_trajectories_only(self, mock_generate_traj):
        """Test mesh creation with only trajectory data."""
        # Mock trajectory mesh generation
        mock_mesh = Mock()
        mock_pv_mesh = Mock()
        mock_pv_mesh.time = None
        mock_generate_traj.return_value = mock_pv_mesh

        traj_spec = PVTrajectorySpec(varname="trajectories")
        traj_data = self.create_simple_trajectory_data(varspecs=(traj_spec,))

        meshes = _create_meshes_for_frame(
            rams_data=None,
            trajectory_data=traj_data,
            current_time=dt.datetime.now(),
        )

        assert len(meshes) == 1
        assert mock_generate_traj.called

    @patch("pyvista.RectilinearGrid")
    def test_create_meshes_for_frame_contour_only(self, mock_rectilinear):
        """Test mesh creation with only contour data."""
        # Mock PyVista grid
        mock_grid = Mock()
        mock_contour_mesh = Mock()
        mock_grid.contour.return_value = mock_contour_mesh
        # Allow item assignment for the mock grid
        mock_grid.__setitem__ = Mock()
        mock_rectilinear.return_value = mock_grid

        contour_spec = PVContourSpec(varname="temperature", isosurfaces=[310])
        rams_data = self.create_simple_rams_data(varspecs=(contour_spec,))

        meshes = _create_meshes_for_frame(
            rams_data=rams_data,
            trajectory_data=None,
            current_time=None,
        )

        assert len(meshes) == 1
        assert mock_rectilinear.called
        assert mock_grid.contour.called

    def test_create_meshes_for_frame_empty_varspecs(self):
        """Test mesh creation with empty varspecs."""
        rams_data = self.create_simple_rams_data(varspecs=())  # Empty varspecs

        meshes = _create_meshes_for_frame(
            rams_data=rams_data, trajectory_data=None, current_time=None
        )

        assert len(meshes) == 0

    def test_create_meshes_for_frame_time_dimension_error(self):
        """Test error when simulation_ds has time dimension."""
        # Create dataset with time dimension (not allowed for single frame)
        x = np.linspace(0, 1000, 3)
        y = np.linspace(0, 1000, 3)
        z = np.linspace(0, 1000, 3)
        times = np.array([0, 1])

        temp = np.ones((3, 3, 3, 2)) * 300
        ds = xr.Dataset(
            {"temperature": (["x", "y", "z", "time"], temp)},
            coords={"x": x, "y": y, "z": z, "time": times},
        )

        contour_spec = PVContourSpec(varname="temperature", isosurfaces=[310])
        rams_data = PVRamsData(simulation_ds=ds, varspecs=(contour_spec,))

        with pytest.raises(
            ValueError, match="simulation_ds must not have a time dimension"
        ):
            _create_meshes_for_frame(
                rams_data=rams_data,
                trajectory_data=None,
                current_time=None,
            )


if __name__ == "__main__":
    pytest.main([__file__])
