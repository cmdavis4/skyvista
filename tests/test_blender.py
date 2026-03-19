"""Test skyvista.blender subpackage with mocked bpy."""

from unittest.mock import Mock, MagicMock, patch
import sys
import pytest


@pytest.fixture(autouse=True)
def mock_bpy():
    """Mock bpy and bmesh modules for testing without Blender."""
    # Mock both bpy and bmesh which are required by blender modules
    mock_bpy_module = MagicMock()
    # bpy.app.version is checked at module level, needs to return a tuple
    mock_bpy_module.app.version = (4, 0, 0)
    sys.modules['bpy'] = mock_bpy_module
    sys.modules['bmesh'] = MagicMock()

    # Clear any cached imports of skyvista.blender submodules
    modules_to_remove = [key for key in sys.modules if key.startswith('skyvista.blender')]
    for mod in modules_to_remove:
        del sys.modules[mod]

    yield

    # Cleanup
    if 'bpy' in sys.modules:
        del sys.modules['bpy']
    if 'bmesh' in sys.modules:
        del sys.modules['bmesh']
    # Also clean up any blender module imports so tests are isolated
    modules_to_remove = [key for key in sys.modules if key.startswith('skyvista.blender')]
    for mod in modules_to_remove:
        del sys.modules[mod]


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
    # These should be importable (they're from carlee_tools.types_carlee_tools)
    assert BlenderObject is not None
    assert BlenderCollection is not None
