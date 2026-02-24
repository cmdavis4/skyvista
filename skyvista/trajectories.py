# ============================================================================
# TRAJECTORY MESH GENERATION
# ============================================================================

import numpy as np
from .types_pvplotting import PVTrajectoryData, PVTrajectorySpec
import pyvista as pv
import xarray as xr
from typing import Any, Dict, List, Optional, Union

from ..utils import maybe_cast_to_float, to_kv_str
from .types_pvplotting import PVMesh


def create_trajectory_polydata(
    trajectory_data_list: List[Dict[str, Any]],
    scalar: Optional[str] = None,
    create_mesh_kwargs: Optional[Dict] = {},
) -> pv.PolyData:
    """
    Create single PolyData mesh containing multiple trajectories as separate lines.

    Combines multiple trajectory paths into a single mesh for efficient rendering.
    Each trajectory becomes a separate line cell within the mesh, allowing for
    bulk operations while maintaining individual trajectory identity.

    Args:
        trajectory_data_list (list): List of trajectory data dictionaries, each
            containing 'points' (Nx3 coordinates) and 'parcel_ix' keys.
        scalar (str, optional): Name of scalar field if present in trajectory data.
            Defaults to None.

    Returns:
        pyvista.PolyData: Single mesh containing all trajectories as line cells.

    Example:
        >>> traj_data = [
        ...     {'points': np.array([[0,0,0], [1,1,1]]), 'parcel_ix': 0},
        ...     {'points': np.array([[2,0,0], [3,1,1]]), 'parcel_ix': 1}
        ... ]
        >>> mesh = create_trajectory_polydata(traj_data)
    """
    if not trajectory_data_list:
        return pv.PolyData()

    # Calculate total points and prepare data structures
    total_points = sum(len(traj["points"]) for traj in trajectory_data_list)
    all_points = np.zeros((total_points, 3))
    all_lines = []
    all_scalars = [] if scalar else None

    point_offset = 0
    for traj_data in trajectory_data_list:
        points = traj_data["points"]
        n_points = len(points)

        # Add points to master array
        all_points[point_offset : point_offset + n_points] = points

        # Create line connectivity: [n_points, point_idx_0, ..., point_idx_n-1]
        line_connectivity = [n_points] + list(
            range(point_offset, point_offset + n_points)
        )
        all_lines.extend(line_connectivity)

        # Add scalar data if present
        if scalar and scalar in traj_data:
            all_scalars.extend(traj_data[scalar])

        point_offset += n_points

    # Create the PolyData mesh
    mesh = pv.PolyData(
        all_points, lines=np.array(all_lines, dtype=np.int_), **create_mesh_kwargs
    )
    if scalar and all_scalars:
        mesh[scalar] = np.array(all_scalars)

    return mesh


def create_tetrahedron_head(
    base_center: np.ndarray, tip: np.ndarray, radius: float
) -> pv.PolyData:
    """
    Create simple tetrahedron arrow head (alternative to cone head).

    Creates a 4-sided arrow head with triangular base. This is a simpler
    alternative to the default cone head for trajectory arrows.

    Args:
        base_center (array): Center point of tetrahedron base.
        tip (array): Tip point of arrow head.
        radius (float): Radius of base circle.

    Returns:
        pyvista.PolyData: Tetrahedron mesh.
    """
    # Create 6 points around base in hexagon (simplified to 3 triangle points)
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    base_points = []

    for angle in angles:
        x = base_center[0] + radius * np.cos(angle)
        y = base_center[1] + radius * np.sin(angle)
        z = base_center[2]
        base_points.append([x, y, z])

    # Add tip point
    points = np.array(base_points + [tip])

    head = pv.PolyData()
    head.points = points
    # Define tetrahedron faces (4 triangular faces)
    faces = np.array([
        3,
        0,
        1,
        2,  # Base triangle
        3,
        0,
        3,
        1,  # Side 1
        3,
        1,
        3,
        2,  # Side 2
        3,
        2,
        3,
        0,  # Side 3
    ])
    head.faces = faces
    return head


def create_trajectory_mesh(
    trajectory_data_list: List[Dict[str, Any]],
    body_radius: float = 70,
    head_length_frac: float = 10,
    head_radius_frac: float = 2.5,
    head_radial_res: int = 30,
    arrow_color_scalar: Optional[str] = None,
    decimation_target: Optional[float] = None,
    tube_resolution: int = 4,
    label_scalar: Union[bool, str] = False,
    create_mesh_kwargs: Optional[Dict] = {},
) -> pv.PolyData:
    """
    Create optimized trajectory visualization with tubes and arrow heads as single mesh.

    High-performance approach that creates all trajectory tubes in a single operation,
    then adds arrow heads, resulting in one merged mesh. This is much faster than
    creating individual meshes for large numbers of trajectories.

    Args:
        trajectory_data_list (list): List of trajectory data dictionaries.
        body_radius (int, optional): Tube radius. Defaults to 70.
        head_length_frac (int, optional): Head length fraction. Defaults to 10.
        head_radius_frac (float, optional): Head radius fraction. Defaults to 2.5.
        head_radial_res (int, optional): Head resolution. Defaults to 30.
        arrow_color_scalar (str, optional): Scalar field name for coloring arrows. Defaults to None.
        decimation_target (float, optional): Target fraction for mesh decimation.
            Defaults to None (no decimation).
        tube_resolution (int, optional): Radial resolution of tubes. Defaults to 4.
        label_scalar (bool or str, optional): Labeling mode. If True, labels with parcel_ix.
            If string matching scalar name, labels with scalar value at trajectory end.
            Defaults to False.

    Returns:
        pyvista.PolyData: Single optimized mesh containing all trajectories.

    Example:
        >>> mesh = create_trajectory_mesh(
        ...     trajectory_data_list,
        ...     decimation_target=0.5  # Reduce mesh complexity by 50%
        ... )
    """
    if not trajectory_data_list:
        return pv.PolyData()

    # Create single mesh with all trajectory tubes
    trajectory_body_polydata_mesh = create_trajectory_polydata(
        trajectory_data_list, arrow_color_scalar, create_mesh_kwargs
    )

    # Convert to tubes (single operation for all trajectories)
    tube_mesh = trajectory_body_polydata_mesh.tube(
        radius=body_radius, n_sides=tube_resolution, capping=False
    )

    # Create arrow heads and labels
    head_meshes = []
    label_meshes = []
    head_length = body_radius * head_length_frac

    for traj_data in trajectory_data_list:
        points = traj_data["points"]
        if len(points) < 2:
            continue

        # Calculate head direction and position
        dhead = points[-1] - points[-2]
        dhead_norm = np.linalg.norm(dhead)
        if dhead_norm > 0:
            dhead = dhead / dhead_norm
            head_center = points[-1] + dhead * (head_length / 2)

            head = pv.Cone(
                center=head_center,
                direction=dhead,
                height=head_length,
                radius=head_radius_frac * body_radius,
                resolution=head_radial_res,
            )

            # Add scalar data to head if present
            if arrow_color_scalar and arrow_color_scalar in traj_data:
                head[arrow_color_scalar] = np.full(
                    head.n_cells, traj_data[arrow_color_scalar][-1]
                )

            head_meshes.append(head)

            # Create label if requested
            if label_scalar:
                # Determine label value
                label_value = None
                if isinstance(label_scalar, str) and label_scalar in traj_data:
                    label_value = "{:.2f}".format(traj_data[label_scalar][-1])
                elif label_scalar is True:
                    label_value = traj_data["parcel_ix"]

                if label_value is not None:
                    label_center = np.copy(head_center)
                    label_center[2] = label_center[2] + 300  # Offset in z
                    label_mesh = pv.Text3D(
                        string=str(label_value),
                        center=label_center,
                        height=200,
                        normal=(0.0, -1.0, 0.0),
                    )

                    # Add scalar data to label if present
                    if arrow_color_scalar and arrow_color_scalar in traj_data:
                        label_mesh.point_data[arrow_color_scalar] = [
                            traj_data[arrow_color_scalar][-1]
                        ] * label_mesh.n_points

                    label_meshes.append(label_mesh)

    # Merge all heads and labels into single mesh
    meshes_to_merge = [tube_mesh]

    if head_meshes:
        if len(head_meshes) == 1:
            heads_mesh = head_meshes[0]
        else:
            heads_mesh = head_meshes[0].merge(head_meshes[1:], merge_points=False)
        meshes_to_merge.append(heads_mesh)

    if label_meshes:
        if len(label_meshes) == 1:
            labels_mesh = label_meshes[0]
        else:
            labels_mesh = label_meshes[0].merge(label_meshes[1:], merge_points=False)
        meshes_to_merge.append(labels_mesh)

    # Merge all components
    if len(meshes_to_merge) == 1:
        final_mesh = meshes_to_merge[0]
    else:
        final_mesh = meshes_to_merge[0].merge(meshes_to_merge[1:], merge_points=False)

    # Apply mesh decimation if requested (for performance)
    if decimation_target and final_mesh.n_cells > 1000:
        print(
            f"Original mesh: {final_mesh.n_cells} cells, {final_mesh.n_points} points"
        )

        # Convert to triangles first
        triangulated_mesh = final_mesh.triangulate()
        print(
            f"After triangulation: {triangulated_mesh.n_cells} cells,"
            f" {triangulated_mesh.n_points} points"
        )

        try:
            decimated_mesh = triangulated_mesh.decimate(decimation_target)
            print(
                f"Decimated mesh: {decimated_mesh.n_cells} cells,"
                f" {decimated_mesh.n_points} points"
            )
            return decimated_mesh
        except:
            print("Quadric decimation failed, using simple decimation")
            decimated_mesh = triangulated_mesh.decimate_pro(
                reduction=1.0 - decimation_target, preserve_topology=True
            )
            print(
                f"Pro decimated mesh: {decimated_mesh.n_cells} cells,"
                f" {decimated_mesh.n_points} points"
            )
            return decimated_mesh

    # Store the color scalar as point data if present
    if arrow_color_scalar:
        final_mesh.point_data[arrow_color_scalar] = final_mesh[arrow_color_scalar]
    # Save the names of the color and label scalars
    return final_mesh


def generate_trajectory_mesh(
    trajectory_ds: xr.Dataset,
    trajectory_spec: PVTrajectorySpec,
) -> pv.PolyData:
    """
    Generate trajectories as single optimized mesh for maximum performance.

    Modern high-performance approach that creates all trajectories in a single
    mesh operation using new class-based specifications.

    Args:
        trajectory_data: PVTrajectoryData object containing the trajectory dataset.
        trajectory_spec: PVTrajectorySpec object containing rendering specifications.

    Returns:
        PVMesh: Single mesh object containing all trajectories with specifications.

    Example:
        >>> traj_data = PVTrajectoryData(parcel_ds)
        >>> traj_spec = PVTrajectorySpec(varname='trajectories', scalar='temperature')
        >>> pv_mesh = generate_trajectory_meshes_single_mesh(traj_data, traj_spec)
    """

    arrow_color_scalar = trajectory_spec.scalar
    particles = trajectory_spec.particles
    label_scalar = False  # Can be extended to support labels from trajectory_spec

    # Extract trajectory arrow kwargs from trajectory_spec
    trajectory_arrow_kwargs = {
        "body_radius": trajectory_spec.body_radius,
        "head_length_frac": trajectory_spec.head_length_frac,
        "head_radius_frac": trajectory_spec.head_radius_frac,
        "head_radial_res": trajectory_spec.head_radial_resolution,
        "tube_resolution": trajectory_spec.tube_resolution,
    }
    if particles:
        # Particle case - create points and convert to spheres
        if "time" in trajectory_ds.dims:
            trajectory_ds = trajectory_ds.isel({"time": -1})

        points = np.column_stack([
            trajectory_ds["x"].values,
            trajectory_ds["y"].values,
            trajectory_ds["z"].values,
        ])

        valid_mask = ~np.isnan(points).any(axis=1)
        points = points[valid_mask]

        # Create point set first
        point_mesh = pv.PolyData(points)
        if arrow_color_scalar:
            scalar_values = maybe_cast_to_float(
                trajectory_ds[arrow_color_scalar].values
            )
            point_mesh[arrow_color_scalar] = scalar_values[valid_mask]

        # Convert to sphere glyphs for particle rendering
        # Set some defaults for the create_mesh_kwargs
        create_mesh_kwargs = {
            "orient": False,
            "scale": False,
            "geom": pv.Sphere(radius=250, phi_resolution=8, theta_resolution=8),
        }
        create_mesh_kwargs.update(trajectory_spec.create_mesh_kwargs)
        particle_mesh = point_mesh.glyph(**create_mesh_kwargs)

        return particle_mesh

    # Extract trajectory data efficiently
    trajectories_points_data = []

    # Vectorized data extraction
    x_data = trajectory_ds["x"].values
    y_data = trajectory_ds["y"].values
    z_data = trajectory_ds["z"].values
    parcel_indices = trajectory_ds["parcel_ix"].values

    # Pre-extract scalar data if needed
    arrow_color_scalar_data = None
    if arrow_color_scalar:
        arrow_color_scalar_data = maybe_cast_to_float(
            trajectory_ds[arrow_color_scalar].values
        )

    label_scalar_data = None
    if isinstance(label_scalar, str):
        label_scalar_data = maybe_cast_to_float(trajectory_ds[label_scalar].values)

    for i, parcel_ix in enumerate(parcel_indices):
        xyz_points = np.column_stack([x_data[i], y_data[i], z_data[i]])
        valid_mask = ~(np.isnan(xyz_points).any(axis=1))

        if valid_mask.sum() < 2:
            continue

        valid_points = xyz_points[valid_mask]

        trajectory_points_data = {
            "parcel_ix": parcel_ix,
            "points": valid_points,
        }

        # Add arrow color scalar data if present
        if arrow_color_scalar and arrow_color_scalar_data is not None:
            if "time" in trajectory_ds[arrow_color_scalar].dims:
                trajectory_points_data[arrow_color_scalar] = arrow_color_scalar_data[i][
                    valid_mask
                ]
            else:
                trajectory_points_data[arrow_color_scalar] = np.full(
                    len(valid_points), arrow_color_scalar_data[i]
                )

        # Add label scalar data if present (only last value needed for labeling)
        if isinstance(label_scalar, str) and label_scalar_data is not None:
            # Only store the last value for labeling
            last_valid_idx = np.where(valid_mask)[0][-1]  # Last valid time index
            trajectory_points_data[label_scalar] = [
                label_scalar_data[i][last_valid_idx]
            ]

        trajectories_points_data.append(trajectory_points_data)

    # Create single optimized mesh containing all trajectories
    single_mesh = create_trajectory_mesh(
        trajectories_points_data,
        arrow_color_scalar=arrow_color_scalar,
        label_scalar=label_scalar,
        create_mesh_kwargs=trajectory_spec.create_mesh_kwargs,
        **trajectory_arrow_kwargs,
    )
    # Save the names of the scalars if we want to see later
    if arrow_color_scalar:
        single_mesh.field_data["arrow_color_scalar"] = arrow_color_scalar
    if label_scalar:
        single_mesh.field_data["label_scalar"] = label_scalar

    # Return PVMesh object
    return single_mesh
