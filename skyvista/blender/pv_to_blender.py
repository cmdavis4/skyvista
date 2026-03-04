from tqdm.notebook import tqdm
from pathlib import Path

from ..types_sv import PVMesh


def export_meshes_to_blender(meshes: list[PVMesh], out_dir):
    """
    Export meshes to PLY format for Blender import.

    Args:
        meshes_dict (dict): Dictionary of mesh data with structure
            {frame_time: {name: {"mesh": mesh_obj, ...}}}.
        out_dir (str or Path): Output directory for PLY files.
        coordinate_transform (str, optional): Coordinate transformation to apply.
            Options:
            - "auto": Apply cloudy PyVista->Blender coordinate fixes
            - "swap_yz": Swap Y and Z coordinates (x,y,z) -> (x,z,y)
            - "swap_xy": Swap X and Y coordinates (x,y,z) -> (y,x,z)
            - "swap_xz": Swap X and Z coordinates (x,y,z) -> (z,y,x)
            - "none": No transformation applied
            Defaults to "auto".

    Example:
        >>> export_to_blender(meshes, "blender_assets/", coordinate_transform="swap_yz")
        # Creates PLY files with Y/Z coordinates swapped for Blender
    """
    for pv_mesh in tqdm(meshes, desc="Exporting meshes to .vtk"):
        pv_mesh.mesh.save(
            (Path(out_dir) / pv_mesh.name).with_suffix(".vtk"),
        )
