"""Test skyvista.blender subpackage with mocked bpy."""

from unittest.mock import Mock, MagicMock, patch
import sys
import pytest


@pytest.fixture(autouse=True)
def mock_bpy():
    """Mock bpy module for testing without Blender."""
    sys.modules['bpy'] = MagicMock()
    yield
    if 'bpy' in sys.modules:
        del sys.modules['bpy']


def test_blender_subpackage_import(mock_bpy):
    """Test that blender subpackage can be imported with mocked bpy."""
    from skyvista.blender import blender_core
    assert hasattr(blender_core, 'deselect_all')
    assert hasattr(blender_core, 'reset_scene')


def test_blender_functions_importable(mock_bpy):
    """Test that key blender functions are importable."""
    from skyvista.blender import (
        run_atmospheric_animation,
        deselect_all,
        reset_scene,
        setup_camera,
        setup_render_settings,
    )
    assert callable(run_atmospheric_animation)
    assert callable(deselect_all)
    assert callable(reset_scene)
    assert callable(setup_camera)
    assert callable(setup_render_settings)


def test_blender_types_importable(mock_bpy):
    """Test that Blender types are re-exported from blender subpackage."""
    from skyvista.blender import BlenderObject, BlenderCollection
    # These should be importable (they're from skyutils.types_skyutils)
    assert BlenderObject is not None
    assert BlenderCollection is not None
