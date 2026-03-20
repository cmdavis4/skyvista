"""Test basic imports and package structure"""

import pytest


def test_package_import():
    """Test that the package can be imported"""
    import skyvista
    assert skyvista.__version__ is not None


def test_primary_api_available():
    """Test that primary API functions are accessible"""
    import skyvista as sv

    # Scene class
    assert hasattr(sv, 'Scene')

    # Main convenience functions
    assert hasattr(sv, 'plot_gridded')
    assert hasattr(sv, 'plot_trajectories')

    # Factory functions
    assert hasattr(sv, 'make_contour')
    assert hasattr(sv, 'make_volume')
    assert hasattr(sv, 'make_vectors')
    assert hasattr(sv, 'make_vector')  # Alias
    assert hasattr(sv, 'make_slice')
    assert hasattr(sv, 'make_trajectory')


def test_varspec_classes_available():
    """Test that VarSpec classes are accessible"""
    import skyvista as sv

    assert hasattr(sv, 'VarSpec')
    assert hasattr(sv, 'ContourSpec')
    assert hasattr(sv, 'VolumeSpec')
    assert hasattr(sv, 'VectorSpec')
    assert hasattr(sv, 'SliceSpec')
    assert hasattr(sv, 'TrajectorySpec')


def test_geometry_classes_available():
    """Test that Geometry classes are accessible"""
    import skyvista as sv

    assert hasattr(sv, 'Geometry')
    assert hasattr(sv, 'ContourGeometry')
    assert hasattr(sv, 'VolumeGeometry')
    assert hasattr(sv, 'VectorGeometry')
    assert hasattr(sv, 'SliceGeometry')
    assert hasattr(sv, 'TrajectoryGeometry')


def test_appearance_classes_available():
    """Test that Appearance classes are accessible"""
    import skyvista as sv

    assert hasattr(sv, 'Appearance')
    assert hasattr(sv, 'ContourAppearance')
    assert hasattr(sv, 'VolumeAppearance')
    assert hasattr(sv, 'VectorAppearance')
    assert hasattr(sv, 'TrajectoryAppearance')


def test_grid_builders_available():
    """Test that grid builders are accessible"""
    import skyvista as sv

    assert hasattr(sv, 'GridBuilder')
    assert hasattr(sv, 'RectilinearGridBuilder')
    assert hasattr(sv, 'CurvilinearGridBuilder')
    assert hasattr(sv, 'UnstructuredGridBuilder')
    assert hasattr(sv, 'GeographicGridBuilder')
    assert hasattr(sv, 'SphericalGridBuilder')
    assert hasattr(sv, 'detect_grid_type')
    assert hasattr(sv, 'get_grid_builder')


def test_utilities_available():
    """Test that utility functions are accessible"""
    import skyvista as sv

    assert hasattr(sv, 'initialize_plotter')
    assert hasattr(sv, 'PVMesh')
    assert hasattr(sv, 'presets')


def test_camera_functions_available():
    """Test that camera functions are accessible"""
    import skyvista as sv

    assert hasattr(sv, 'calculate_camera_offset')
    assert hasattr(sv, 'get_trajectory_camera')
    assert hasattr(sv, 'camera_follow_callback')


def test_trajectory_functions_available():
    """Test that trajectory functions are accessible"""
    import skyvista as sv

    assert hasattr(sv, 'create_trajectory_polydata')
    assert hasattr(sv, 'create_tetrahedron_head')
    assert hasattr(sv, 'create_trajectory_mesh')
    assert hasattr(sv, 'generate_trajectory_mesh')


def test_blender_functions_available():
    """Test that Blender export functions are accessible"""
    import skyvista as sv

    assert hasattr(sv, 'export_meshes_to_blender')
