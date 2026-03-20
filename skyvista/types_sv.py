"""
Core type definitions for skyvista.

This module contains the PVMesh dataclass used throughout skyvista
for mesh-varspec pairing.
"""

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Optional

import pyvista as pv

from carlee_tools import NUMERICAL_DT_FORMAT, dt_to_str


@dataclass(kw_only=True)
class PVMesh:
    """
    Container for a mesh and its associated visualization spec.

    This class pairs a PyVista mesh with its VarSpec, along with metadata
    like time and the resulting actor from plotter.add_mesh().

    Attributes:
        varspec: The VarSpec that created this mesh
        name: Unique name for this mesh (auto-generated if None)
        time: Time value this mesh represents
        mesh: The PyVista mesh object
        actor: The actor returned by plotter.add_mesh()
    """

    varspec: Any  # VarSpec from varspec.py
    name: Optional[str] = None
    time: Optional[dt.datetime] = None
    mesh: Optional[pv.DataObject] = None
    actor: Optional[Any] = None  # Can be dict of actors for subplots

    def make_mesh_name(self) -> str:
        time_str = (
            dt_to_str(self.time, date_format=NUMERICAL_DT_FORMAT)
            if self.time
            else "none"
        )
        varspec_name = getattr(self.varspec, "name", "unknown")
        return f"dt={time_str}_{varspec_name}"

    def __post_init__(self):
        self.name = self.name or self.make_mesh_name()

    @property
    def mesh_empty(self) -> bool:
        """Check if the mesh is empty (None or no points)."""
        if self.mesh is None:
            return True
        return len(self.mesh.points) == 0
