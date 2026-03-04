"""
Type definitions for Blender integration.

This module provides type aliases and custom types used for Blender operations.
It includes conditional imports for Blender-specific types to maintain compatibility
when running outside of the Blender environment.
"""

import sys
from typing import Any, Dict, Union

# pyright: reportRedeclaration=false

# Handle TypeAlias for Python < 3.10 compatibility
if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    try:
        from typing_extensions import TypeAlias
    except ImportError:
        # Fallback for when typing_extensions is not available
        TypeAlias = None

# Configuration dictionary type for storing arbitrary configuration data
if TypeAlias is not None:
    ConfigDict: TypeAlias = Dict[str, Any]
else:
    ConfigDict = Dict[str, Any]

# Blender-specific types (conditionally imported)
try:
    import bpy

    if TypeAlias is not None:
        BlenderObject: TypeAlias = bpy.types.Object
        BlenderCollection: TypeAlias = bpy.types.Collection
    else:
        BlenderObject = bpy.types.Object
        BlenderCollection = bpy.types.Collection
except ImportError:
    # When bpy is not available, use Any as placeholder
    # This allows the code to import successfully outside of Blender
    if TypeAlias is not None:
        BlenderObject: TypeAlias = Any
        BlenderCollection: TypeAlias = Any
    else:
        BlenderObject = Any
        BlenderCollection = Any
