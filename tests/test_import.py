"""Test basic imports and package structure"""

import pytest


def test_package_import():
    """Test that the package can be imported"""
    import skyvista
    assert skyvista.__version__ is not None


def test_main_functions_available():
    """Test that main functions are accessible"""
    import skyvista as sv

    # Check main functions exist
    assert hasattr(sv, 'plot_rams_and_trajectories')
    assert hasattr(sv, 'plot_trajectories')
    assert hasattr(sv, 'animate_trajectories')
    assert hasattr(sv, 'initialize_plotter')


def test_convenience_functions_available():
    """Test that convenience functions are accessible"""
    import skyvista as sv

    # Check convenience functions
    assert hasattr(sv, 'quick_plot')
    assert hasattr(sv, 'plot_trajectories_only')
    assert hasattr(sv, 'plot_isosurfaces_only')
    assert hasattr(sv, 'make_contour')
    assert hasattr(sv, 'make_vector')
    assert hasattr(sv, 'make_trajectory')


def test_types_available():
    """Test that type definitions are accessible"""
    import skyvista as sv

    # Check types exist
    assert hasattr(sv, 'PVConfig')
    assert hasattr(sv, 'PVRamsData')
    assert hasattr(sv, 'PVTrajectoryData')
    assert hasattr(sv, 'PVContourSpec')
    assert hasattr(sv, 'PVVectorSpec')
    assert hasattr(sv, 'PVTrajectorySpec')


def test_camera_functions_available():
    """Test that camera functions are accessible"""
    import skyvista as sv

    # Check camera functions
    assert hasattr(sv, 'calculate_camera_offset')
    assert hasattr(sv, 'get_trajectory_camera')
    assert hasattr(sv, 'camera_follow_callback')


def test_trajectory_functions_available():
    """Test that trajectory functions are accessible"""
    import skyvista as sv

    # Check trajectory functions
    assert hasattr(sv, 'create_trajectory_polydata')
    assert hasattr(sv, 'create_tetrahedron_head')
    assert hasattr(sv, 'create_trajectory_mesh')
    assert hasattr(sv, 'generate_trajectory_mesh')


def test_blender_functions_available():
    """Test that Blender export functions are accessible"""
    import skyvista as sv

    # Check Blender export
    assert hasattr(sv, 'export_meshes_to_blender')
