"""
Tests for the skyvista Scene-based API.

These tests cover the core functionality of the Scene class and VarSpec types.
"""

import pytest
import numpy as np
import xarray as xr
from unittest.mock import Mock, patch

# Import the new API
import skyvista as sv
from skyvista import (
    Scene,
    ContourSpec,
    VolumeSpec,
    VectorSpec,
    SliceSpec,
    TrajectorySpec,
    make_contour,
    make_volume,
    make_vectors,
    make_slice,
    make_trajectory,
    plot_gridded,
    plot_trajectories,
)
from skyvista.geometry import (
    ContourGeometry,
    VolumeGeometry,
    VectorGeometry,
    SliceGeometry,
    TrajectoryGeometry,
)
from skyvista.appearance import (
    Appearance,
    ContourAppearance,
    VolumeAppearance,
    VectorAppearance,
    TrajectoryAppearance,
)
from skyvista.mesh import PVMesh

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_gridded_ds():
    """Create a sample gridded dataset for testing."""
    x = np.linspace(0, 1000, 10)
    y = np.linspace(0, 1000, 10)
    z = np.linspace(0, 5000, 5)

    # Create sample temperature field with a gradient
    temp = np.zeros((10, 10, 5))
    for i, zi in enumerate(z):
        temp[:, :, i] = 300 + zi / 100  # Temperature increases with height

    ds = xr.Dataset(
        {"THETA": (["x", "y", "z"], temp)},
        coords={"x": x, "y": y, "z": z},
    )
    return ds


@pytest.fixture
def sample_gridded_ds_with_time():
    """Create a sample gridded dataset with time dimension."""
    x = np.linspace(0, 1000, 5)
    y = np.linspace(0, 1000, 5)
    z = np.linspace(0, 2000, 3)
    times = np.arange(3)

    temp = np.random.rand(5, 5, 3, 3) * 30 + 290
    ds = xr.Dataset(
        {"THETA": (["x", "y", "z", "time"], temp)},
        coords={"x": x, "y": y, "z": z, "time": times},
    )
    return ds


@pytest.fixture
def sample_vector_ds():
    """Create a sample dataset with vector field."""
    x = np.linspace(0, 1000, 5)
    y = np.linspace(0, 1000, 5)
    z = np.linspace(0, 2000, 3)

    # Create simple vector field
    u = np.ones((5, 5, 3)) * 10
    v = np.zeros((5, 5, 3))
    w = np.ones((5, 5, 3)) * 2

    ds = xr.Dataset(
        {
            "UC": (["x", "y", "z"], u),
            "VC": (["x", "y", "z"], v),
            "WC": (["x", "y", "z"], w),
        },
        coords={"x": x, "y": y, "z": z},
    )
    return ds


@pytest.fixture
def sample_trajectory_ds():
    """Create a sample trajectory dataset."""
    n_trajectories = 5
    n_times = 10
    trajectory_ix = np.arange(n_trajectories)
    times = np.arange(n_times)

    # Create simple straight-line trajectories
    x = np.zeros((n_trajectories, n_times))
    y = np.zeros((n_trajectories, n_times))
    z = np.zeros((n_trajectories, n_times))

    for i in range(n_trajectories):
        x[i, :] = np.linspace(0, 500, n_times) + i * 100
        y[i, :] = np.linspace(0, 500, n_times) + i * 50
        z[i, :] = np.linspace(100, 2000, n_times)

    # Add a scalar field for coloring
    altitude = z.copy()

    ds = xr.Dataset(
        {
            "x": (["trajectory_ix", "time"], x),
            "y": (["trajectory_ix", "time"], y),
            "z": (["trajectory_ix", "time"], z),
            "altitude": (["trajectory_ix", "time"], altitude),
        },
        coords={"trajectory_ix": trajectory_ix, "time": times},
    )
    return ds


# =============================================================================
# SCENE TESTS
# =============================================================================


class TestScene:
    """Test the Scene class."""

    def test_scene_creation(self):
        """Test basic scene creation."""
        scene = Scene()
        assert scene.background == "#f8f6f1"
        assert scene.show_grid is True
        assert scene.force_bounds is False

    def test_scene_with_custom_settings(self):
        """Test scene creation with custom settings."""
        scene = Scene(
            background="black",
            title="Test Scene",
            show_grid=False,
            force_bounds=True,
        )
        assert scene.background == "black"
        assert scene.title == "Test Scene"
        assert scene.show_grid is False
        assert scene.force_bounds is True

    def test_scene_add_returns_self(self, sample_gridded_ds):
        """Test that add methods return self for chaining."""
        scene = Scene()
        spec = make_contour("THETA", isosurfaces=[300])
        result = scene.add(sample_gridded_ds, spec)
        assert result is scene

    def test_scene_method_chaining(self, sample_gridded_ds):
        """Test that methods can be chained."""
        scene = (
            Scene()
            .add_contour(sample_gridded_ds, "THETA", isosurfaces=[300])
            .add_contour(sample_gridded_ds, "THETA", isosurfaces=[310])
        )
        assert len(scene._specs) == 2


class TestAddContour:
    """Test Scene.add_contour method."""

    def test_add_contour_basic(self, sample_gridded_ds):
        """Test basic contour addition."""
        scene = Scene()
        scene.add_contour(sample_gridded_ds, "THETA", isosurfaces=[300, 310])
        assert len(scene._specs) == 1
        ds, spec = scene._specs[0]
        assert isinstance(spec, ContourSpec)

    def test_add_contour_with_appearance(self, sample_gridded_ds):
        """Test contour with appearance options."""
        scene = Scene()
        scene.add_contour(
            sample_gridded_ds,
            "THETA",
            isosurfaces=[300],
            opacity=0.5,
            color="red",
        )
        _, spec = scene._specs[0]
        assert spec.appearance.opacity == 0.5
        assert spec.appearance.color == "red"

    def test_add_contours_dict(self, sample_gridded_ds):
        """Test add_contours with dictionary input."""
        scene = Scene()
        scene.add_contours(
            sample_gridded_ds,
            {
                "THETA": [300, 310],  # Simple form
            },
        )
        assert len(scene._specs) == 1


class TestAddVolume:
    """Test Scene.add_volume method."""

    def test_add_volume_basic(self, sample_gridded_ds):
        """Test basic volume addition."""
        scene = Scene()
        scene.add_volume(sample_gridded_ds, "THETA")
        assert len(scene._specs) == 1
        _, spec = scene._specs[0]
        assert isinstance(spec, VolumeSpec)
        assert spec.is_volume() is True


class TestAddVectors:
    """Test Scene.add_vectors method."""

    def test_add_vectors_basic(self, sample_vector_ds):
        """Test basic vector addition."""
        scene = Scene()
        scene.add_vectors(
            sample_vector_ds,
            "wind",
            u="UC",
            v="VC",
            w="WC",
        )
        assert len(scene._specs) == 1
        _, spec = scene._specs[0]
        assert isinstance(spec, VectorSpec)


class TestAddSlice:
    """Test Scene.add_slice method."""

    def test_add_slice_basic(self, sample_gridded_ds):
        """Test basic slice addition."""
        scene = Scene()
        scene.add_slice(sample_gridded_ds, "THETA", dim="z", value=2000)
        assert len(scene._specs) == 1
        _, spec = scene._specs[0]
        assert isinstance(spec, SliceSpec)


class TestAddTrajectories:
    """Test Scene.add_trajectories method."""

    def test_add_trajectories_basic(self, sample_trajectory_ds):
        """Test basic trajectory addition."""
        scene = Scene()
        scene.add_trajectories(sample_trajectory_ds)
        assert len(scene._specs) == 1
        _, spec = scene._specs[0]
        assert isinstance(spec, TrajectorySpec)

    def test_add_trajectories_with_scalar(self, sample_trajectory_ds):
        """Test trajectories with scalar coloring."""
        scene = Scene()
        scene.add_trajectories(sample_trajectory_ds, scalar="altitude")
        _, spec = scene._specs[0]
        assert spec.geometry.scalar == "altitude"

    def test_add_trajectories_particle_style(self, sample_trajectory_ds):
        """Test trajectories with particle style."""
        scene = Scene()
        scene.add_trajectories(sample_trajectory_ds, style="particle")
        _, spec = scene._specs[0]
        assert spec.appearance.style == "particle"


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestFactoryFunctions:
    """Test the factory functions."""

    def test_make_contour(self):
        """Test make_contour factory function."""
        spec = make_contour("THETA", isosurfaces=[300, 310], opacity=0.7)
        assert isinstance(spec, ContourSpec)
        assert spec.geometry.varname == "THETA"
        assert spec.geometry.isosurfaces == [300, 310]
        assert spec.appearance.opacity == 0.7

    def test_make_volume(self):
        """Test make_volume factory function."""
        spec = make_volume("QC", threshold=(0.001, 0.01), cmap="Greys_r")
        assert isinstance(spec, VolumeSpec)
        assert spec.geometry.varname == "QC"
        assert spec.geometry.threshold == (0.001, 0.01)
        assert spec.appearance.cmap == "Greys_r"

    def test_make_vectors(self):
        """Test make_vectors factory function."""
        spec = make_vectors("wind", u="UC", v="VC", w="WC", factor=0.5)
        assert isinstance(spec, VectorSpec)
        assert spec.geometry.varname == "wind"
        assert spec.geometry.u_varname == "UC"
        assert spec.geometry.factor == 0.5

    def test_make_slice(self):
        """Test make_slice factory function."""
        spec = make_slice("THETA", dim="z", value=1000)
        assert isinstance(spec, SliceSpec)
        assert spec.geometry.varname == "THETA"
        assert spec.geometry.slice_dim == "z"
        assert spec.geometry.slice_value == 1000

    def test_make_trajectory(self):
        """Test make_trajectory factory function."""
        spec = make_trajectory(scalar="altitude", style="tube", limit=500)
        assert isinstance(spec, TrajectorySpec)
        assert spec.geometry.scalar == "altitude"
        assert spec.appearance.style == "tube"
        assert spec.limit == 500


# =============================================================================
# VARSPEC TESTS
# =============================================================================


class TestVarSpecs:
    """Test VarSpec classes."""

    def test_contour_spec_auto_name(self):
        """Test ContourSpec auto-generates name."""
        spec = make_contour("THETA", isosurfaces=[300])
        assert "contour" in spec.name
        assert "THETA" in spec.name

    def test_volume_spec_auto_name(self):
        """Test VolumeSpec auto-generates name."""
        spec = make_volume("QC")
        assert "volume" in spec.name
        assert "QC" in spec.name

    def test_varspec_pyvista_kwargs(self):
        """Test VarSpec generates PyVista kwargs."""
        spec = make_contour("THETA", opacity=0.5, color="red")
        kwargs = spec.get_pyvista_kwargs()
        assert kwargs["opacity"] == 0.5
        assert kwargs["color"] == "red"


# =============================================================================
# GEOMETRY TESTS
# =============================================================================


class TestGeometry:
    """Test Geometry classes."""

    def test_contour_geometry(self):
        """Test ContourGeometry creation."""
        geom = ContourGeometry(varname="THETA", isosurfaces=[300, 310])
        assert geom.varname == "THETA"
        assert geom.isosurfaces == [300, 310]

    def test_volume_geometry(self):
        """Test VolumeGeometry creation."""
        geom = VolumeGeometry(varname="QC", threshold=(0.001, 0.01))
        assert geom.varname == "QC"
        assert geom.threshold == (0.001, 0.01)

    def test_vector_geometry(self):
        """Test VectorGeometry creation."""
        geom = VectorGeometry(
            varname="wind", u_varname="UC", v_varname="VC", w_varname="WC"
        )
        assert geom.varname == "wind"
        assert geom.u_varname == "UC"

    def test_trajectory_geometry(self):
        """Test TrajectoryGeometry creation."""
        geom = TrajectoryGeometry(scalar="altitude", tube_radius=100)
        assert geom.scalar == "altitude"
        assert geom.tube_radius == 100


# =============================================================================
# APPEARANCE TESTS
# =============================================================================


class TestAppearance:
    """Test Appearance classes."""

    def test_appearance_defaults(self):
        """Test default appearance values."""
        app = Appearance()
        assert app.opacity == 1.0
        assert app.color is None
        assert app.show_scalar_bar is False

    def test_contour_appearance(self):
        """Test ContourAppearance creation."""
        app = ContourAppearance(opacity=0.5, color="blue", style="wireframe")
        assert app.opacity == 0.5
        assert app.color == "blue"
        assert app.style == "wireframe"

    def test_trajectory_appearance(self):
        """Test TrajectoryAppearance creation."""
        app = TrajectoryAppearance(style="particle", silhouettes=True)
        assert app.style == "particle"
        assert app.silhouettes is True

    def test_appearance_to_pyvista_kwargs(self):
        """Test conversion to PyVista kwargs."""
        app = ContourAppearance(opacity=0.5, color="red", cmap="viridis")
        kwargs = app.to_pyvista_kwargs()
        assert kwargs["opacity"] == 0.5
        assert kwargs["color"] == "red"
        assert kwargs["cmap"] == "viridis"


# =============================================================================
# PVMESH TESTS
# =============================================================================


class TestPVMesh:
    """Test PVMesh class."""

    def test_pvmesh_creation(self):
        """Test PVMesh creation."""
        spec = make_contour("THETA")
        mesh = PVMesh(varspec=spec)
        assert mesh.varspec is spec
        assert mesh.mesh is None
        assert mesh.actor is None

    def test_pvmesh_auto_name(self):
        """Test PVMesh auto-generates name."""
        spec = make_contour("THETA")
        mesh = PVMesh(varspec=spec)
        assert mesh.name is not None
        assert "contour" in mesh.name

    def test_pvmesh_empty_check(self):
        """Test mesh_empty property."""
        spec = make_contour("THETA")
        mesh = PVMesh(varspec=spec, mesh=None)
        assert mesh.mesh_empty is True


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_plot_gridded_creates_scene(self, sample_gridded_ds):
        """Test plot_gridded creates a scene."""
        with patch.object(Scene, "show"):
            scene = plot_gridded(
                sample_gridded_ds, contours={"THETA": [300, 310]}, show=True
            )
        assert isinstance(scene, Scene)
        assert len(scene._specs) == 1

    def test_plot_trajectories_creates_scene(self, sample_trajectory_ds):
        """Test plot_trajectories creates a scene."""
        with patch.object(Scene, "show"):
            scene = plot_trajectories(
                sample_trajectory_ds, scalar="altitude", show=True
            )
        assert isinstance(scene, Scene)
        assert len(scene._specs) == 1


# =============================================================================
# MESH CREATION TESTS
# =============================================================================


class TestMeshCreation:
    """Test mesh creation from VarSpecs."""

    def test_contour_creates_mesh(self, sample_gridded_ds):
        """Test ContourSpec creates a mesh."""
        spec = make_contour("THETA", isosurfaces=[310])
        mesh = spec.create_mesh(sample_gridded_ds, time=None)
        assert mesh is not None
        # May be empty if no points at that isosurface

    def test_slice_creates_mesh(self, sample_gridded_ds):
        """Test SliceSpec creates a mesh."""
        spec = make_slice("THETA", dim="z", value=2500)
        mesh = spec.create_mesh(sample_gridded_ds, time=None)
        assert mesh is not None
        assert mesh.n_points > 0

    def test_vector_creates_mesh(self, sample_vector_ds):
        """Test VectorSpec creates a mesh."""
        spec = make_vectors("wind", u="UC", v="VC", w="WC")
        mesh = spec.create_mesh(sample_vector_ds, time=None)
        assert mesh is not None
        assert mesh.n_points > 0


# =============================================================================
# ANIMATION TESTS
# =============================================================================


class TestAnimation:
    """Test animation-related functionality."""

    def test_get_all_times(self, sample_gridded_ds_with_time):
        """Test _get_all_times returns all time values."""
        scene = Scene()
        scene.add_contour(sample_gridded_ds_with_time, "THETA", isosurfaces=[300])

        times = scene._get_all_times()
        assert len(times) == 3
        assert list(times) == [0, 1, 2]

    def test_get_all_times_no_time_dimension(self, sample_gridded_ds):
        """Test _get_all_times with no time dimension returns [None]."""
        scene = Scene()
        scene.add_contour(sample_gridded_ds, "THETA", isosurfaces=[300])

        times = scene._get_all_times()
        assert times == [None]

    def test_get_last_time(self, sample_gridded_ds_with_time):
        """Test _get_last_time returns last time value."""
        scene = Scene()
        scene.add_contour(sample_gridded_ds_with_time, "THETA", isosurfaces=[300])

        last_time = scene._get_last_time()
        assert last_time == 2

    def test_get_last_time_no_time_dimension(self, sample_gridded_ds):
        """Test _get_last_time with no time dimension returns None."""
        scene = Scene()
        scene.add_contour(sample_gridded_ds, "THETA", isosurfaces=[300])

        last_time = scene._get_last_time()
        assert last_time is None

    def test_contour_mesh_at_different_times(self, sample_gridded_ds_with_time):
        """Test ContourSpec creates mesh at different time steps."""
        spec = make_contour("THETA", isosurfaces=[300])

        # Create mesh at time 0
        mesh_t0 = spec.create_mesh(sample_gridded_ds_with_time, time=0)
        assert mesh_t0 is not None

        # Create mesh at time 2
        mesh_t2 = spec.create_mesh(sample_gridded_ds_with_time, time=2)
        assert mesh_t2 is not None

    def test_animate_creates_gif(self, sample_gridded_ds_with_time, tmp_path):
        """Test animate method creates a GIF file."""
        scene = Scene()
        scene.add_slice(sample_gridded_ds_with_time, "THETA", dim="z", value=1000)

        gif_path = tmp_path / "test_animation.gif"

        # Mock tqdm.notebook.tqdm to avoid progress bar issues in tests
        with patch("tqdm.notebook.tqdm", side_effect=lambda x, **kw: x):
            scene.animate(gif_path, fps=5)

        assert gif_path.exists()
        assert gif_path.stat().st_size > 0

    def test_animate_with_specific_times(self, sample_gridded_ds_with_time, tmp_path):
        """Test animate with specific time subset."""
        scene = Scene()
        scene.add_slice(sample_gridded_ds_with_time, "THETA", dim="z", value=1000)

        gif_path = tmp_path / "test_animation_subset.gif"

        with patch("tqdm.notebook.tqdm", side_effect=lambda x, **kw: x):
            scene.animate(gif_path, fps=5, times=[0, 2])  # Only 2 frames

        assert gif_path.exists()

    def test_screenshot_at_specific_time(self, sample_gridded_ds_with_time, tmp_path):
        """Test screenshot at a specific time."""
        scene = Scene()
        scene.add_slice(sample_gridded_ds_with_time, "THETA", dim="z", value=1000)

        png_path = tmp_path / "test_screenshot.png"
        scene.screenshot(png_path, time=1)

        assert png_path.exists()
        assert png_path.stat().st_size > 0


class TestTimeHandling:
    """Test time selection and handling in VarSpecs."""

    def test_select_time_with_time_dim(self, sample_gridded_ds_with_time):
        """Test time selection when time dimension exists."""
        from skyvista.grid_utils import select_time

        ds_at_time = select_time(sample_gridded_ds_with_time, time=1)
        assert "time" not in ds_at_time.dims

    def test_select_time_without_time_dim(self, sample_gridded_ds):
        """Test time selection when no time dimension exists."""
        from skyvista.grid_utils import select_time

        ds_result = select_time(sample_gridded_ds, time=None)
        # Should return unchanged dataset
        assert ds_result.dims == sample_gridded_ds.dims

    def test_trajectory_time_slicing(self, sample_trajectory_ds):
        """Test trajectory spec handles time slicing correctly."""
        spec = make_trajectory(scalar="altitude")

        # Create mesh at intermediate time (should include all times up to that point)
        mesh = spec.create_mesh(sample_trajectory_ds, time=5)
        assert mesh is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
