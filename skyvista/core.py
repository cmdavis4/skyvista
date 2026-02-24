"""
PyVista-based 3D visualization tools for atmospheric modeling data.

This module provides comprehensive 3D visualization capabilities for atmospheric
model output and trajectory data using PyVista. It supports both single-frame
visualization and animation generation, with specialized functions for RAMS
(Regional Atmospheric Modeling System) output.

Key Features:
- 3D trajectory visualization with arrows and particle modes
- Isosurface generation for scalar fields
- Animation support with time-series data
- Camera control and scene management
- Export capabilities (HTML, images, Blender-compatible formats)

Main Functions:
- plot_trajectories: Unified function for both still images and animations
- initialize_plotter: Set up PyVista plotter with atmospheric modeling defaults
- generate_trajectory_meshes: Create 3D trajectory representations

Example Usage:
    >>> import pvplotting as pvp
    >>> plotter = pvp.initialize_plotter()
    >>> meshes = pvp.plot_trajectories(
    ...     simulation_ds=sim_data,
    ...     parcel_ds=trajectory_data,
    ...     kwargs_contour={'temperature': {'isosurfaces': [15, 20, 25]}},
    ...     animate=True,
    ...     gif_path='animation.gif'
    ... )
"""

# Set PyVista to off-screen mode before importing to avoid kernel crashes on interrupt
from itertools import product

from .plotter import initialize_plotter

from .types_pvplotting import (
    PV2DSpec,
    PVConfig,
    PVContourSpec,
    PVMesh,
    PVData,
    PVRamsData,
    PVTrajectoryData,
    PVTrajectorySpec,
    PVVolumeSpec,
    PVVectorSpec,
)
from .trajectories_pvplotting import generate_trajectory_mesh
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm  # type: ignore
from IPython.display import HTML, display, Image  # type: ignore
from pathlib import Path
from typing import Iterator, Tuple, overload, Optional, Dict, Any, List
from copy import copy
import ipywidgets as widgets

import xarray as xr
import numpy as np
from skimage.measure import marching_cubes
import meshio

from ..utils import to_kv_str, TwoWayDict
from ..rams import to_t_minutes
from ..types_core import PathLike

# Allow empty meshes globally, since the individual varspecs handle whether to actually
# include them
pv.global_theme.allow_empty_mesh = True


def add_mesh_to_subplots(
    pv_config: PVConfig,
    pv_mesh: PVMesh,
) -> Dict[Tuple[int, int], Any]:
    """
    Add PVMesh to specified subplots within a multi-subplot plotter.

    Args:
        pv_config: PyVista configuration object containing plotter and subplot settings.
        pv_mesh: PVMesh object containing mesh and its specification.

    Returns:
        dict: Dictionary mapping subplot keys to actor objects.
    """
    actors = {}

    # Determine which subplots to add mesh to
    if len(pv_config.subplot_included_meshes) == 0:
        subplot_keys_to_add = get_subplot_keys(pv_config.plotter)
    else:
        subplot_keys_to_add = [
            subplot_key
            for subplot_key, included_meshes in pv_config.subplot_included_meshes.items()
            if pv_mesh.varspec.varname in included_meshes
        ]

    # Use add_mesh_kwargs from the varspec
    add_mesh_kwargs = pv_mesh.varspec.add_mesh_kwargs.copy()

    # Add scalar bar configuration if needed
    if pv_mesh.varspec.scalar_bar and "scalar_bar_args" not in add_mesh_kwargs:
        add_mesh_kwargs["scalar_bar_args"] = {
            "title": (
                pv_mesh.varspec.scalar
                if isinstance(pv_mesh.varspec, PVTrajectorySpec)
                else pv_mesh.varspec.varname
            )
        }

    # Add mesh to each specified subplot
    print(pv_mesh)
    for subplot_key in subplot_keys_to_add:
        pv_config.plotter.subplot(*subplot_key)
        # Get the right function, which is plotter.add_mesh for all except volumes
        # and plotter.add_volume for volumes
        add_mesh_fn = (
            pv_config.plotter.add_volume
            if isinstance(pv_mesh.varspec, PVVolumeSpec)
            else pv_config.plotter.add_mesh
        )
        actors[subplot_key] = add_mesh_fn(
            pv_mesh.mesh,
            show_scalar_bar=pv_mesh.varspec.scalar_bar,
            # Remember to add it with the varspec name (which doesn't have time in it)
            # rather than the mesh name, which does, so that pyvista replaces
            # the old mesh with the new
            name=pv_mesh.varspec.name,
            **add_mesh_kwargs,
        )

    return actors


def _add_meshes_to_plotter(
    pv_config: PVConfig,
    meshes: List[PVMesh],
):
    """
    Add collection of PVMesh objects to plotter with appropriate styling.

    Internal function that handles adding different types of meshes (trajectory,
    simulation, grid) to the plotter using their embedded specifications.
    Sets the actor attribute on each PVMesh object.

    Args:
        pv_config: PyVista configuration object containing plotter and settings.
        meshes: List of PVMesh objects containing meshes and their specifications.
    """
    for pv_mesh in meshes:
        if not pv_mesh.mesh:
            continue

        # Handle trajectory-specific coloring/styling
        if isinstance(pv_mesh.varspec, PVTrajectorySpec):
            mesh = pv_mesh.mesh
            varspec = pv_mesh.varspec
            # Save colormap info if using scalar coloring
            if (
                varspec.scalar
                and hasattr(mesh, "array_names")
                and varspec.scalar in mesh.array_names
            ):
                if varspec.cmap:
                    mesh.field_data["arrow_color_cmap"] = varspec.cmap

        # Add mesh to plotter and set actor on PVMesh object
        try:
            # Even if this mesh is allowed to be empty, pyvista will complain when
            # it tries to calculate the range of any scalars on it; just skip it
            # and leave .actor empty if it's empty
            if not pv_mesh.mesh_empty:
                pv_mesh.actor = add_mesh_to_subplots(pv_config, pv_mesh)
        except:
            print(f"Encountered error adding mesh {pv_mesh.name}:")
            raise


# ============================================================================
# MESH CREATION FOR ATMOSPHERIC DATA
# ============================================================================


def _create_meshes_for_frame(pv_datas: List[PVData], current_time):
    """
    Create all meshes needed for a single visualization frame.

    Internal function that creates trajectory, contour, and 2D meshes for a
    specific time frame using variable specifications.

    Args:
        rams_data: RAMS data object containing simulation dataset.
        trajectory_data: Trajectory data object containing parcel dataset.
        varspecs: List of variable specifications for rendering.
        current_time: Current time being processed.

    Returns:
        list: List of PVMesh objects.

    Raises:
        ValueError: If simulation_ds contains time dimension or parcel_ds is empty.
    """
    meshes: list[PVMesh] = []

    for pv_data in pv_datas:
        if isinstance(pv_data, PVRamsData):
            # =============================================================================
            # RAMS data processing
            # =============================================================================
            this_time_simulation_ds = pv_data.ds.sel({"time": current_time})

            # Separate varspecs by type
            contour_specs = [
                spec for spec in pv_data.varspecs if isinstance(spec, PVContourSpec)
            ]
            vector_specs = [
                spec for spec in pv_data.varspecs if isinstance(spec, PVVectorSpec)
            ]
            twod_specs = [
                spec for spec in pv_data.varspecs if isinstance(spec, PV2DSpec)
            ]
            volume_specs = [
                spec for spec in pv_data.varspecs if isinstance(spec, PVVolumeSpec)
            ]
            # Only allow one 2D spec per dataset right n ow
            if len(twod_specs) > 1:
                raise NotImplementedError(
                    "Only one 2D varpsec per dataset is allowed at present"
                )

            # Create 3D grid for contours and vectors if needed
            grid_mesh_3d = None
            if contour_specs or vector_specs or volume_specs:
                grid_mesh_3d = pv.RectilinearGrid(
                    this_time_simulation_ds["x"].values,
                    this_time_simulation_ds["y"].values,
                    this_time_simulation_ds["z"].values,
                )

                # Add the grid itself as an invisible mesh to maintain domain bounds
                # when no 2D slices are present (prevents PyVista from zooming to
                # just the contour/vector geometry)
                if not twod_specs and not volume_specs:
                    # Don't need to do this if we have a volume
                    bounds_varspec = PVContourSpec(
                        varname="_domain_bounds",
                        add_mesh_kwargs={"opacity": 0.0, "style": "wireframe"},
                    )
                    meshes.append(
                        PVMesh(
                            varspec=bounds_varspec,
                            time=current_time,
                            mesh=grid_mesh_3d,
                        )
                    )

                # -----------------------------------------------------------------------------
                # Process volume variables
                # -----------------------------------------------------------------------------
                for this_volume_spec in volume_specs:
                    # Simplest way is to add a scalar on the grid we already created
                    varname = this_volume_spec.varname
                    grid_mesh_3d[varname] = this_time_simulation_ds[
                        varname
                    ].values.ravel(order="F")
                    # Then just add this as a pvmesh
                    meshes.append(
                        PVMesh(
                            varspec=this_volume_spec,
                            time=current_time,
                            mesh=grid_mesh_3d,
                        )
                    )

                # -----------------------------------------------------------------------------
                # Process contour variables
                # -----------------------------------------------------------------------------
                for this_contour_spec in contour_specs:
                    varname = this_contour_spec.varname
                    grid_mesh_3d[varname] = this_time_simulation_ds[
                        varname
                    ].values.ravel(order="F")

                    isosurfaces = this_contour_spec.isosurfaces
                    if isosurfaces is not None and not isinstance(
                        isosurfaces, (list, tuple, np.ndarray)
                    ):
                        raise ValueError(
                            "isosurfaces must be a list, tuple, or numpy array"
                        )
                    contour_kwargs = dict(
                        **{"scalars": varname},
                        **(
                            {"isosurfaces": this_contour_spec.isosurfaces}
                            if this_contour_spec.isosurfaces is not None
                            else {}
                        ),
                        **this_contour_spec.create_mesh_kwargs,
                    )
                    this_mesh = grid_mesh_3d.contour(**contour_kwargs)

                    # If a different scalar variable is specified for coloring, sample it onto the isosurface
                    if this_contour_spec.scalars is not None:
                        scalar_varname = this_contour_spec.scalars
                        # Add the scalar variable to the grid if not already present
                        if scalar_varname not in grid_mesh_3d.array_names:
                            grid_mesh_3d[scalar_varname] = this_time_simulation_ds[
                                scalar_varname
                            ].values.ravel(order="F")
                        # Sample the scalar variable onto the isosurface mesh
                        this_mesh = this_mesh.sample(
                            grid_mesh_3d,
                            pass_point_arrays=False,
                            pass_cell_arrays=False,
                        )
                        # Set the scalar variable as the active scalars
                        this_mesh.set_active_scalars(scalar_varname)

                    meshes.append(
                        PVMesh(
                            varspec=this_contour_spec,
                            time=current_time,
                            mesh=this_mesh,
                        )
                    )
                # -----------------------------------------------------------------------------
                # Process vector variables
                # -----------------------------------------------------------------------------
                for vector_spec in vector_specs:
                    u_var = vector_spec.u_varname
                    v_var = vector_spec.v_varname
                    w_var = vector_spec.w_varname

                    # Get flow components
                    flow_component_arrs = {
                        "u": this_time_simulation_ds[u_var].values.ravel(order="F"),
                        "v": this_time_simulation_ds[v_var].values.ravel(order="F"),
                        "w": this_time_simulation_ds[w_var].values.ravel(order="F"),
                    }

                    this_mesh_vectors = np.vstack(list(flow_component_arrs.values())).T
                    vector_name = vector_spec.varname

                    # Add vectors to the 3D grid
                    grid_mesh_3d[vector_name] = this_mesh_vectors
                    for component, arr in flow_component_arrs.items():
                        grid_mesh_3d[component] = arr

                    # Set as active vector and create glyphs
                    grid_mesh_3d.set_active_vectors(vector_name)
                    # Want to default to passing the vector name as the orient and
                    # scale arguments to glyph if it wasn't passed, so clean up
                    # the create_mesh_kwargs to make this possible
                    create_mesh_kwargs = {
                        k: v for k, v in vector_spec.create_mesh_kwargs.items()
                    }
                    # Manually handle orient and scale
                    # For orient, default to the active vector if nothing was passed,
                    # since we'd need to do a lot more work to handle orienting
                    # this vector by some other vector
                    create_mesh_kwargs["orient"] = create_mesh_kwargs.get(
                        "orient", vector_name
                    )
                    # For scale, if something other than the name of the active vector
                    # was passed, we assume it's a scalar and add it to grid_mesh_3d
                    create_mesh_kwargs["scale"] = create_mesh_kwargs.get(
                        "scale", vector_name
                    )
                    if create_mesh_kwargs["scale"] != vector_name:
                        # Add this data to grid_mesh_3d
                        grid_mesh_3d[create_mesh_kwargs["scale"]] = (
                            this_time_simulation_ds[
                                create_mesh_kwargs["scale"]
                            ].values.ravel(order="F")
                        )
                    this_isosurface_mesh = grid_mesh_3d.glyph(**create_mesh_kwargs)

                    meshes.append(
                        PVMesh(
                            varspec=vector_spec,
                            time=current_time,
                            mesh=this_isosurface_mesh,
                        )
                    )

            # -----------------------------------------------------------------------------
            # Process 2D variables
            # -----------------------------------------------------------------------------
            if twod_specs:
                # Already checked it can only be one spec
                twod_spec = twod_specs[0]

                # Get slice dimension and value
                slice_dim = twod_spec.slice_dim
                slice_value = twod_spec.slice_value

                # Select the slice from the data
                if slice_value is None:
                    # Use first index if no value specified
                    slice_sel = {
                        slice_dim: this_time_simulation_ds[slice_dim].values[0]
                    }
                else:
                    slice_sel = {slice_dim: slice_value}

                # Extract 2D data slice
                sliced_data = this_time_simulation_ds.sel(
                    slice_sel, method=twod_spec.slice_method
                )

                # Determine which dimensions remain after slicing
                remaining_dims = [d for d in ["x", "y", "z"] if d != slice_dim]
                dim1, dim2 = remaining_dims

                # Create meshgrid for the 2D surface
                coord1 = sliced_data[dim1].values
                coord2 = sliced_data[dim2].values
                grid1, grid2 = np.meshgrid(coord1, coord2, indexing="ij")

                # Create constant coordinate for the sliced dimension
                if slice_value is None:
                    slice_coord_value = sliced_data[slice_dim].values.item()
                else:
                    slice_coord_value = slice_value
                grid_sliced = np.ones_like(grid1) * slice_coord_value

                # Map coordinates to x, y, z in correct order
                coords = {slice_dim: grid_sliced, dim1: grid1, dim2: grid2}
                grid_mesh_2d = pv.StructuredGrid(
                    coords["x"],
                    coords["y"],
                    coords["z"],
                    **twod_spec.create_mesh_kwargs,
                )

                # Add variable data to mesh
                varname = twod_spec.varname
                var_data = sliced_data[varname].values
                grid_mesh_2d[varname] = var_data.ravel(
                    order="F"
                )  # Fortran order for correct mapping

                meshes.append(
                    PVMesh(
                        varspec=twod_spec,
                        time=current_time,
                        mesh=grid_mesh_2d,
                    )
                )

        # =============================================================================
        # Trajectories
        # =============================================================================
        elif isinstance(pv_data, PVTrajectoryData):
            this_time_trajectory_ds = pv_data.ds.sel(
                {"time": slice(None, current_time)}
            )
            # Don't include these if we have fewer than two time points
            if len(this_time_trajectory_ds["time"].values) >= 2:
                # Get trajectory specs
                trajectory_specs = pv_data.varspecs

                for trajectory_spec in trajectory_specs:
                    # Use optimized single mesh trajectory generation
                    trajectory_mesh = generate_trajectory_mesh(
                        trajectory_ds=this_time_trajectory_ds,
                        trajectory_spec=trajectory_spec,  # type: ignore
                    )

                    # Store the mesh object
                    meshes.append(
                        PVMesh(
                            varspec=trajectory_spec,
                            time=current_time,
                            mesh=trajectory_mesh,
                        )
                    )

    # Filter out empty meshes if we're supposed to
    meshes = [x for x in meshes if not (x.mesh_empty and not x.varspec.empty_ok)]
    return meshes


# ============================================================================
# TIME PROCESSING AND ANIMATION
# ============================================================================


def _process_time_point(
    pv_config: PVConfig,
    pv_datas: List[PVData],
    current_time,
):
    """
    Process single time point: create meshes, add to plotter, execute callback.

    Internal function that handles all operations needed for a single time point
    in either still image or animation context. Creates appropriate data slices,
    generates meshes, adds them to plotter, manages scalar bars, adds timestamp
    title, and executes user callback.

    Args:
        pv_config: PyVista configuration object containing plotter and other settings.
        rams_data: RAMS data object containing simulation dataset.
        trajectory_data: Trajectory data object containing parcel dataset.
        varspecs: List of variable specifications for rendering.
        current_time: Time value to process.

    Returns:
        list: List of PVMesh objects with actor attributes set.
    """

    # Create meshes for this frame
    meshes = _create_meshes_for_frame(
        pv_datas=pv_datas,
        current_time=current_time,
    )

    # Add meshes to plotter (sets actor attribute on each mesh)
    _add_meshes_to_plotter(
        pv_config=pv_config,
        meshes=meshes,
    )

    # Handle scalar bar management - this could be moved to PVConfig if needed
    # TODO: Add scalar bar management to PVConfig

    # Add timestamp title
    try:
        time_ds = pv_datas[0].ds  # type: ignore
        time_minutes = time_ds.sel({"time": current_time})["t_minutes"].values
        pv_config.plotter.add_text(
            f"t={time_minutes:.0f} minutes",
            position="upper_edge",
            name="title_text",
        )
    except (KeyError, AttributeError):
        # No t_minutes coordinate available
        pass

    # Execute user callback if provided
    if pv_config.callback:
        raise NotImplementedError
        trajectory_meshes = {
            mesh.varspec.varname: mesh.mesh
            for mesh in meshes
            if isinstance(mesh.varspec, PVTrajectorySpec)
        }
        pv_config.callback(
            pv_config.plotter,
            current_time,
            rams_data.ds if rams_data else None,
            trajectory_data.ds if trajectory_data else None,
            trajectory_meshes,
        )

    return meshes


def sanitize_inputs(pv_datas: List[PVData]):
    """
    Validate and prepare input datasets for visualization.

    Performs input validation, dimension reordering, and time compatibility
    checking. Ensures parcel_ds times are a subset of simulation_ds times
    when both datasets are provided.

    Args:
        rams_config (PVRamsConfig): RAMS configuration containing simulation_ds and contour specs.
        trajectory_config (PVTrajectoryConfig): Trajectory configuration containing trajectory_ds and limit.

    Returns:
        tuple: (cleaned_simulation_ds, cleaned_parcel_ds)

    Raises:
        ValueError: If parcel_ds has empty time dimension or parcel times
            are not a subset of simulation times.

    Example:
        >>> rams_cfg = PVRamsConfig(simulation_ds=sim_ds, kwargs_contour={'temp': {'isosurfaces': [15]}})
        >>> traj_cfg = PVTrajectoryConfig(trajectory_ds=parcel_ds)
        >>> sim_ds, parcel_ds = sanitize_inputs(rams_cfg, traj_cfg)
    """

    # Call individual sanitize methods
    if len(pv_datas) == 0:
        raise ValueError("Must pass at least one PVData object")

    # Check that all PVDatas of the same type have the same time values
    # Group PVData objects by type
    rams_datas = [pv_data for pv_data in pv_datas if isinstance(pv_data, PVRamsData)]
    trajectory_datas = [
        pv_data for pv_data in pv_datas if isinstance(pv_data, PVTrajectoryData)
    ]

    # If any of the datasets have a time dimension, they all need to
    if not all(["time" not in data.ds.dims for data in pv_datas]):
        # Check that all RAMS data have the same time values
        for datas, datas_type in [
            (rams_datas, "PVRamsData"),
            (trajectory_datas, "PVTrajectoryData"),
        ]:

            if not datas:
                continue
            # Check that they all have a time dimension
            for data in datas:
                if "time" not in data.ds.dims:
                    raise ValueError(
                        f"A {datas_type} does not have a 'time' dimension; 'time'"
                        " dimension must either be present in no datasets or all"
                        " datasets. If preselecting a single time in RAMS data, pass"
                        " it as a list to avoid this error, e.g."
                        " ds.sel(dict(time=[some_time]))"
                    )
            # a time dimension
            if not len({frozenset(data.ds["time"].values) for data in datas}) == 1:
                raise ValueError(
                    f"All PVData objects of each type must have the same time values"
                )

        # Check time compatibility between RAMS and trajectory data
        if rams_datas and trajectory_datas:
            rams_times = set(rams_datas[0].ds["time"].values)
            traj_times = set(trajectory_datas[0].ds["time"].values)
            # One of them needs to be a subset of the other
            if not (traj_times.issubset(rams_times) or rams_times.issubset(traj_times)):
                raise ValueError(
                    "Times in trajectory data must be a subset of times in RAMS data,"
                    "or vice versa. "
                    f"Extra times in trajectory data: {traj_times - rams_times}"
                )

    # Since we usually get the time values by just taking pv_datas[0], need to ensure
    # that any rams datasets present are first in the list, since they should contain
    # all time values, whereas parcels can be a subset
    # Sort pv_datas so that PVRamsData objects come first
    pv_datas = sorted(pv_datas, key=lambda x: 0 if isinstance(x, PVRamsData) else 1)
    return pv_datas


# ============================================================================
# MAIN PLOTTING FUNCTIONS
# ============================================================================


def plot_rams_and_trajectories(
    pv_config: PVConfig, pv_datas: List[PVData]
) -> Dict[Any, List[PVMesh]]:
    """
    Create 3D visualization of atmospheric data as still image or animation.

    Unified function for creating both static visualizations and animations of
    atmospheric simulation data and trajectory data. Handles isosurface generation,
    trajectory rendering, camera control, and export options.

    Args:
        parcel_ds (xr.Dataset, optional): Trajectory dataset with 'parcel_ix' and
            'time' dimensions, containing 'x', 'y', 'z' position variables.
        simulation_ds (xr.Dataset, optional): Atmospheric simulation dataset with
            'x', 'y', 'z', and optionally 'time' dimensions.
        plotter (pv.Plotter, optional): PyVista plotter instance. If None, creates new one.
        kwargs_contour (dict, optional): 3D contour specifications. Format:
            {'variable_name': {'isosurfaces': [val1, val2], 'opacity': 0.5, ...}}.
        kwargs_2d (dict, optional): 2D slice specifications. Format:
            {'variable_name': {'opacity': 0.8, ...}}.
        subplot_included_meshes (dict, optional): Subplot inclusion mapping for
            multi-panel plots. Format: {(row, col): ['mesh1', 'mesh2']}.
        trajectory_color_or_scalar (str, optional): Trajectory coloring. Can be
            variable name for scalar coloring or color name for solid coloring.
        trajectory_cmap (str, optional): Colormap name for trajectory scalar coloring.
        trajectory_silhouettes (bool, optional): Enable trajectory silhouettes.
            Defaults to False.
        title (str, optional): Custom title text for the plot.
        trajectory_clim (list, optional): Color limits [min, max] for trajectory scalars.
        show (bool, optional): Whether to display the plot. Defaults to True.
        opacity (float, optional): Global opacity for meshes (0.0-1.0).
        scalar_bars (list, optional): List of scalar bar names to keep. Others removed.
        screenshot_path (Path, optional): Path for saving screenshot image.
        export_html (bool, optional): Export HTML version alongside screenshot.
            Defaults to False.
        jupyter_backend (str, optional): Jupyter display backend ('pythreejs', etc.).
        label_trajectories (bool, optional): Add 3D text labels to trajectories.
            Not recommended for animations. Defaults to False.
        particles (bool, optional): Render trajectories as particles instead of arrows.
            Defaults to False.
        add_grid_mesh (bool, optional): Add background coordinate grid. Defaults to False.
        bunny (bool, optional): Special bunny-shaped particles mode. Defaults to False.
        trajectory_scalar_bar_args (dict, optional): Scalar bar customization arguments.
        interactive (bool, optional): Enable interactive controls. Defaults to True.
        individual_meshes (bool, optional): Create separate mesh for each isosurface
            value instead of single mesh with multiple isosurfaces. Defaults to False.
        add_meshes (bool, optional): Whether to add meshes to plotter. If False,
            only creates and returns meshes. Defaults to True.
        animate (bool, optional): Create animation instead of still image. Defaults to False.
        gif_path (str, optional): Output path for animation GIF. Required if animate=True.
        fps (int, optional): Animation frames per second. Defaults to 10.
        show_gif (bool, optional): Display animation in notebook after creation.
            Defaults to True.
        callback (callable, optional): User callback function called for each frame.
            Signature: callback(plotter, current_time, simulation_ds, parcel_ds, trajectory_meshes).
        show_interactive_slider (bool, optional): When animate=True, also display
            an interactive PyVista render with time slider for exploring animation data.
            Defaults to True.
        **trajectory_arrow_kwargs: Additional arguments for trajectory arrow styling
            (body_radius, head_length_frac, etc.).

    Returns:
        dict: Dictionary mapping times to lists of PVMesh objects.
            For animations, keyed by time. For still images, keyed by last_time.

    Raises:
        ValueError: If animate=True but gif_path is not provided, or if datasets
            have incompatible time dimensions.

    Examples:
        Create still image with trajectories and temperature isosurfaces:
        >>> meshes = plot_trajectories(
        ...     simulation_ds=sim_data,
        ...     parcel_ds=trajectory_data,
        ...     kwargs_contour={'temperature': {'isosurfaces': [15, 20, 25]}},
        ...     trajectory_color_or_scalar='temperature',
        ...     animate=False
        ... )

        Create animation:
        >>> plot_trajectories(
        ...     simulation_ds=sim_data,
        ...     parcel_ds=trajectory_data,
        ...     kwargs_contour={'temperature': {'isosurfaces': [20]}},
        ...     animate=True,
        ...     gif_path='temperature_evolution.gif',
        ...     fps=15
        ... )

        Individual isosurface meshes:
        >>> meshes = plot_trajectories(
        ...     simulation_ds=sim_data,
        ...     kwargs_contour={'RV': {'isosurfaces': [0.01, 0.02, 0.03]}},
        ...     individual_meshes=True  # Creates RV_iso-0.01, RV_iso-0.02, RV_iso-0.03
        ... )
    """

    # Validate and prepare input datasets
    pv_datas = sanitize_inputs(pv_datas)

    # Make dicts for storing all of the meshes and actors
    meshes = {}
    if pv_config.animation:
        if not pv_config.interactive:
            # Initialize animation
            pv_config.plotter.open_gif(
                str(pv_config.gif_path), fps=pv_config.fps, subrectangles=True
            )

        # Animation loop - use simulation_ds times if available, otherwise parcel_ds times
        time_ds = pv_datas[0].ds
        time_values = time_ds["time"].values

        for frame_idx, current_time in enumerate(
            tqdm(time_values, desc="Creating meshes by frame")
        ):

            this_frame_meshes = _process_time_point(
                pv_config=pv_config,
                pv_datas=pv_datas,
                current_time=current_time,
            )

            # Show plotter on first frame (required for trajectory-only animations)
            if frame_idx == 0 and pv_config.show and not pv_config.interactive:
                pv_config.plotter.show(auto_close=False)

            # Export HTML for individual frames if requested
            if pv_config.export_html:
                gif_path = (
                    Path(pv_config.gif_path)
                    if pv_config.gif_path
                    else Path("animation.gif")
                )
                html_path = gif_path.with_name(
                    f"{gif_path.stem}_frame-{frame_idx+1}.html"
                )
                pv_config.plotter.export_html(str(html_path))

            # Write animation frame
            if not pv_config.interactive:
                pv_config.plotter.write_frame()

            # Store the meshes and actors
            meshes[current_time] = this_frame_meshes

        # Finalize animation
        if not pv_config.interactive:
            pv_config.plotter.close()

        # Display animation in notebook if requested
        if pv_config.show and not pv_config.interactive:
            if pv_config.gif_scrubber:
                display(create_gif_scrubber(gif_path))
            else:
                with open(pv_config.gif_path, "rb") as f:
                    display(Image(data=f.read(), format="png"))

        # If we are interactive, now add the last meshes to the plotter and
        # set up the slider
        if pv_config.interactive:
            # Add meshes from first time
            _add_meshes_to_plotter(pv_config=pv_config, meshes=meshes[time_values[0]])

            # Set up slider
            _create_interactive_time_slider(
                pv_config=pv_config,
                meshes_by_time=meshes,
                pv_datas=pv_datas,
            )
            if pv_config.show:
                print("show")
                pv_config.plotter.show(auto_close=False)

    else:
        # STILL IMAGE MODE
        # Use simulation_ds times if available, otherwise parcel_ds times
        time_ds = pv_datas[0].ds
        last_time = (
            time_ds["time"].values[-1]
            if "time" in time_ds.dims
            else time_ds["time"].values  # I.e. if it's a singleton dimension
        )

        # Process single time point
        this_frame_meshes = _process_time_point(
            pv_config=pv_config,
            pv_datas=pv_datas,
            current_time=last_time,
        )

        # Display plot
        if pv_config.show:
            pv_config.plotter.show(
                auto_close=False,
            )

        # Add custom title if provided
        if pv_config.title:
            pv_config.plotter.add_text(
                pv_config.title, position="upper_edge", name="title_text"
            )

        # Save output files
        if pv_config.screenshot_path:
            screenshot_path = Path(pv_config.screenshot_path)
            if pv_config.export_html:
                pv_config.plotter.export_html(str(screenshot_path.with_suffix(".html")))
            pv_config.plotter.screenshot(pv_config.screenshot_path, scale=3)

        # Store meshes
        meshes[last_time] = this_frame_meshes

    return TwoWayDict(
        {dt: {mesh.varspec.name: mesh for mesh in dt_l} for dt, dt_l in meshes.items()}
    )


def _create_interactive_time_slider(
    pv_config: PVConfig, meshes_by_time: Dict[Any, List[PVMesh]], pv_datas: List[PVData]
):
    """
    Create interactive PyVista render with time slider for animation data.

    Creates a new PyVista plotter with an interactive time slider that allows
    users to step through different time points of the animation data.

    Args:
        pv_config: PyVista configuration object.
        meshes_by_time: Dictionary of meshes keyed by time from animation.
        rams_data: Optional RAMS data object.
        trajectory_data: Optional trajectory data object.
    """
    # Get time values and sort them
    time_values = sorted(meshes_by_time.keys())
    if not time_values:
        print("No time values found in meshes")
        return
    # Get these as t_minutes also
    try:
        import blt_utils as blt

        start_time = blt.SIMULATION_START_TIME
    except:
        print("Unable to import blt")
        start_time = time_values[0]
    time_values_minutes = to_t_minutes(time_values=time_values, start_time=start_time)

    # Create new plotter for interactive display
    interactive_plotter = pv_config.plotter or initialize_plotter()
    interactive_plotter.title = "Interactive Time Slider View"

    # Initialize with the last frame
    current_time_index = len(time_values) - 1
    current_time = time_values[current_time_index]

    # Create new config for interactive plotter
    interactive_config = PVConfig(
        plotter=interactive_plotter,
        subplot_included_meshes=pv_config.subplot_included_meshes,
    )

    # Add initial meshes from the last frame
    _add_meshes_to_plotter(
        pv_config=interactive_config,
        meshes=meshes_by_time[current_time],
    )

    # Add timestamp title
    try:
        time_ds = pv_datas[0].ds
        if time_ds and "t_minutes" in time_ds.coords:
            time_minutes = time_ds.sel({"time": current_time})["t_minutes"].values
            interactive_plotter.add_text(
                f"t={time_minutes:.0f} minutes",
                position="upper_edge",
                name="title_text",
            )
    except (KeyError, AttributeError):
        pass

    def update_time_slider(time_minutes):
        """Callback function to update meshes based on slider value."""
        # Convert slider value (0-100) to time index
        # Find the nearest value in time_values_minutes that is less than or equal to t_minutes
        valid_indices = [
            i for i, t in enumerate(time_values_minutes) if t <= time_minutes
        ]
        if valid_indices:
            time_index = valid_indices[-1]  # Get the last (largest) valid index
        else:
            time_index = 0  # Fallback to first frame if no valid times found

        selected_time = time_values[time_index]

        # Add meshes for selected time
        _add_meshes_to_plotter(
            pv_config=interactive_config,
            meshes=meshes_by_time[selected_time],
        )

        # Update timestamp title
        interactive_plotter.add_text(
            f"t={time_minutes:.0f} minutes",
            position="upper_edge",
            name="title_text",
        )

    # Track whether slider has been added to avoid multiple additions
    slider = interactive_plotter.add_slider_widget(
        callback=update_time_slider,
        rng=[min(time_values_minutes), max(time_values_minutes)],
        value=min(time_values_minutes),
        title="Simulation time (minutes)",
        pointa=(0.1, 0.05),
        pointb=(0.9, 0.05),
        style="modern",
        tube_width=0.01,
        slider_width=0.02,
        title_height=0.03,
    )

    # Ensure the slider is enabled and visible
    slider.GetRepresentation().SetVisibility(True)
    slider.On()
    slider.EnabledOn()

    return interactive_plotter


# Convenience function for only exporting meshes for blender, i.e. not plotting
# ============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# ============================================================================


def plot_trajectory_frame(**kwargs):
    """
    Legacy function for single-frame trajectory plotting.

    This legacy function is no longer supported with the new class-based approach.
    Please use plot_rams_and_trajectories with PVConfig, PVRamsData, PVTrajectoryData
    and varspecs instead.

    Returns:
        dict: Dictionary of created meshes.
    """
    raise NotImplementedError(
        "Legacy functions are not supported. Please use plot_rams_and_trajectories "
        "with PVConfig, PVRamsData, PVTrajectoryData and varspecs."
    )


def animate_trajectories(plotter, gif_path, **kwargs):
    """
    Legacy function for trajectory animation.

    This legacy function is no longer supported with the new class-based approach.
    Please use plot_rams_and_trajectories with PVConfig, PVRamsData, PVTrajectoryData
    and varspecs instead.

    Args:
        plotter: PyVista plotter object.
        gif_path (str): Output path for animation GIF.
        **kwargs: All other arguments passed to plot_trajectories.

    Returns:
        dict: Empty dictionary (for consistency with animation behavior).
    """
    raise NotImplementedError(
        "Legacy functions are not supported. Please use plot_rams_and_trajectories "
        "with PVConfig, PVRamsData, PVTrajectoryData and varspecs."
    )


# ============================================================================
# CAMERA AND UTILITY FUNCTIONS
# ============================================================================


def rectangle_mesh(x, y, z):
    """
    Create rectangular mesh from coordinate arrays.

    Creates a 2D rectangular mesh where exactly one of x, y, z must be
    a single value (defines the plane), and the other two define the
    rectangular grid.

    Args:
        x, y, z (array-like): Coordinate arrays.

    Returns:
        pyvista.StructuredGrid: Rectangular mesh.

    Raises:
        ValueError: If not exactly one coordinate is singular.
    """
    dim_vals = {"x": x, "y": y, "z": z}
    dim_lengths = {k: len(v) for k, v in dim_vals.items()}
    singleton_dim = [k for k, v in dim_lengths.items() if v == 1]

    if not singleton_dim or len(singleton_dim) > 1:
        raise ValueError("Exactly one of x, y, or z must be of length 1")

    singleton_dim = singleton_dim[0]
    full_dims = [k for k in dim_vals.keys() if k != singleton_dim]

    # Create 2D grid
    full_dim_vals1, full_dim_vals2 = np.meshgrid(
        dim_vals[full_dims[0]], dim_vals[full_dims[1]]
    )
    singleton_dim_vals = (
        np.ones(shape=full_dim_vals1.shape) * dim_vals[singleton_dim][0]
    )

    grid_values = {
        full_dims[0]: full_dim_vals1,
        full_dims[1]: full_dim_vals2,
        singleton_dim: singleton_dim_vals,
    }

    return pv.StructuredGrid(grid_values["x"], grid_values["y"], grid_values["z"])


# ============================================================================
# EXPORT AND UTILITY FUNCTIONS
# ============================================================================


def screenshot_render(plotter, screenshot_path, wait=None):
    """
    Create screenshot of plotter render (deprecated).

    This function is deprecated and provided for backward compatibility only.
    Use plotter.screenshot() directly instead.

    Args:
        plotter: PyVista plotter object.
        screenshot_path (Path): Output path for screenshot.
        wait (int, optional): Wait time (unused).

    Raises:
        ImportError: If playwright package is not installed.
    """
    try:
        import playwright
    except ImportError:
        raise ImportError(
            "The `playwright` package is needed to screenshot renders, but is not"
            " installed; setup instructions for the shot-scraper program used for"
            " screenshots can be found at https://github.com/simonw/shot-scraper"
        )

    screenshot_path = Path(screenshot_path)
    # Export HTML version
    html_path = screenshot_path.with_suffix(".html")
    plotter.export_html(str(html_path))


# ============================================================================
# MESH MANAGEMENT AND PLOTTING UTILITIES
# ============================================================================


def get_subplot_keys(plotter: pv.Plotter) -> Iterator[Tuple[int, int]]:
    """
    Get all subplot indices for a plotter.

    Args:
        plotter: PyVista plotter object.

    Returns:
        itertools.product: Iterator over (row, col) subplot indices.
    """
    return product(range(plotter.shape[0]), range(plotter.shape[1]))


def create_gif_scrubber(gif_path):
    import PIL.Image as pilimage

    # Load GIF
    gif = pilimage.open(gif_path)
    frames = []

    try:
        while True:
            frames.append(gif.copy())
            gif.seek(len(frames))
    except EOFError:
        pass

    def show_frame(frame_idx):
        plt.figure(figsize=(8, 6))
        plt.imshow(frames[frame_idx])
        plt.axis("off")
        plt.show()

    slider = widgets.IntSlider(
        value=0, min=0, max=len(frames) - 1, step=1, description="Frame:"
    )

    widgets.interact(show_frame, frame_idx=slider)


def extract_isosurface_to_vtk(da, iso_value, vtk_filepath):
    """
    Extract an isosurface from a 3D NetCDF dataset and save as VTK file using meshio.

    Parameters:
    -----------
    netcdf_file : str
        Path to the NetCDF file
    variable_name : str
        Name of the variable to extract isosurface from
    iso_value : float
        Value for the isosurface
    output_vtk_file : str
        Output VTK file path
    """

    # Get the coordinate arrays (assuming x, y, z or similar)
    # Adjust coordinate names based on your dataset
    coords = list(da.dims)
    if len(coords) != 3:
        raise ValueError(f"Expected 3D data, got {len(coords)} dimensions")

    # Get coordinate values for proper scaling
    coord_arrays = [da.coords[coord].values for coord in coords]

    # Convert to numpy array
    volume = da.values

    # Handle NaN values by setting them to a value far from iso_value
    if np.isnan(volume).any():
        nan_replacement = iso_value + (np.nanmax(volume) - np.nanmin(volume)) * 2
        volume = np.nan_to_num(volume, nan=nan_replacement)

    # Extract isosurface using marching cubes
    try:
        vertices, faces, normals, values = marching_cubes(
            volume,
            level=iso_value,
            spacing=(
                (
                    coord_arrays[0][1] - coord_arrays[0][0]
                    if len(coord_arrays[0]) > 1
                    else 1.0
                ),
                (
                    coord_arrays[1][1] - coord_arrays[1][0]
                    if len(coord_arrays[1]) > 1
                    else 1.0
                ),
                (
                    coord_arrays[2][1] - coord_arrays[2][0]
                    if len(coord_arrays[2]) > 1
                    else 1.0
                ),
            ),
        )
    except ValueError as e:
        raise ValueError(
            f"No isosurface found at level {iso_value}. Check your data range and"
            " iso_value."
        )

    # Adjust vertices to real coordinate system
    for i in range(3):
        vertices[:, i] += coord_arrays[i].min()

    # Prepare data for meshio
    # meshio expects cells as a list of (cell_type, connectivity_array) tuples
    cells = [("triangle", faces)]

    # Prepare point data
    point_data = {"iso": values, "normals": normals}

    # Create meshio mesh
    mesh = meshio.Mesh(points=vertices, cells=cells, point_data=point_data)

    # Save as VTK file
    # meshio automatically detects format from extension
    mesh.write(vtk_filepath)

    print(f"Isosurface extracted and saved to {vtk_filepath}")
    print(f"Number of vertices: {len(vertices)}")
    print(f"Number of faces: {len(faces)}")

    return mesh


def extract_isosurface_to_multiple_formats(
    netcdf_file, variable_name, iso_value, output_base
):
    """
    Extract an isosurface and save in multiple formats using meshio.

    Parameters:
    -----------
    netcdf_file : str
        Path to the NetCDF file
    variable_name : str
        Name of the variable to extract isosurface from
    iso_value : float
        Value for the isosurface
    output_base : str
        Base name for output files (without extension)
    """

    # Load the dataset
    ds = xr.open_dataset(netcdf_file)
    data = ds[variable_name]
    coords = list(data.dims)
    coord_arrays = [data.coords[coord].values for coord in coords]
    volume = data.values

    # Handle NaN values
    if np.isnan(volume).any():
        nan_replacement = iso_value + (np.nanmax(volume) - np.nanmin(volume)) * 2
        volume = np.nan_to_num(volume, nan=nan_replacement)

    # Extract isosurface
    vertices, faces, normals, values = marching_cubes(
        volume,
        level=iso_value,
        spacing=(
            (
                coord_arrays[0][1] - coord_arrays[0][0]
                if len(coord_arrays[0]) > 1
                else 1.0
            ),
            (
                coord_arrays[1][1] - coord_arrays[1][0]
                if len(coord_arrays[1]) > 1
                else 1.0
            ),
            (
                coord_arrays[2][1] - coord_arrays[2][0]
                if len(coord_arrays[2]) > 1
                else 1.0
            ),
        ),
    )

    # Adjust vertices to real coordinate system
    for i in range(3):
        vertices[:, i] += coord_arrays[i].min()

    # Create meshio mesh
    mesh = meshio.Mesh(
        points=vertices,
        cells=[("triangle", faces)],
        point_data={f"{variable_name}_iso": values, "normals": normals},
    )

    # Save in multiple formats
    formats = {
        ".vtk": "VTK legacy format",
        ".vtu": "VTK XML unstructured grid",
        ".ply": "Stanford PLY format",
        ".stl": "STL format",
        ".obj": "Wavefront OBJ format",
    }

    for ext, description in formats.items():
        try:
            output_file = f"{output_base}{ext}"
            mesh.write(output_file)
            print(f"Saved {description}: {output_file}")
        except Exception as e:
            print(f"Failed to save {description}: {e}")

    ds.close()
    return mesh
