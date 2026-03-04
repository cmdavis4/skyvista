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

from .types_sv import (
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
from .trajectories import generate_trajectory_mesh
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

from skyutils import TwoWayDict, to_t_minutes

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
            # Handle datasets with or without time dimension
            if current_time is not None and "time" in pv_data.ds.dims:
                this_time_simulation_ds = pv_data.ds.sel({"time": current_time})
            else:
                # No time dimension or no time selection needed
                this_time_simulation_ds = pv_data.ds

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
            # Handle datasets with or without time dimension
            if current_time is not None and "time" in pv_data.ds.dims:
                this_time_trajectory_ds = pv_data.ds.sel(
                    {"time": slice(None, current_time)}
                )
            else:
                # No time dimension or no time selection needed
                this_time_trajectory_ds = pv_data.ds

            # Don't include these if we have fewer than two time points
            # (only check if time dimension exists)
            if "time" not in this_time_trajectory_ds.dims or len(this_time_trajectory_ds["time"].values) >= 2:
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


def plot_gridded_and_trajectories(
    pv_config: PVConfig, pv_datas: List[PVData]
) -> Dict[Any, List[PVMesh]]:
    """
    Create 3D visualization of gridded atmospheric data as still image or animation.

    Unified function for creating both static visualizations and animations of
    gridded simulation data and trajectory data. Handles isosurface generation,
    trajectory rendering, camera control, and export options.

    Args:
        pv_config (PVConfig): Configuration object for plotter settings, animation, exports.
        pv_datas (List[PVData]): List of data objects (PVGriddedData and/or PVTrajectoryData).

    Returns:
        dict: Dictionary mapping times to PVMesh objects, organized by variable spec name.

    Examples:
        >>> config = PVConfig(animation=True, gif_path='output.gif')
        >>> gridded_data = PVGriddedData(
        ...     simulation_ds=sim_ds,
        ...     varspecs=(PVContourSpec(varname='THETA', isosurfaces=[300, 310]),)
        ... )
        >>> traj_data = PVTrajectoryData(
        ...     trajectory_ds=traj_ds,
        ...     varspecs=(PVTrajectorySpec(scalar='temperature'),)
        ... )
        >>> meshes = plot_gridded_and_trajectories(config, [gridded_data, traj_data])
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

        # Handle datasets with or without time dimension
        if "time" in time_ds.dims:
            last_time = time_ds["time"].values[-1]
        elif "time" in time_ds.coords:
            # It's a singleton dimension
            last_time = time_ds["time"].values
        else:
            # No time dimension at all - use None
            last_time = None

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


# Deprecated alias for backwards compatibility
def plot_rams_and_trajectories(
    pv_config: PVConfig, pv_datas: List[PVData]
) -> Dict[Any, List[PVMesh]]:
    """
    Deprecated: Use plot_gridded_and_trajectories instead.

    This function is kept for backwards compatibility and will be removed in a future version.
    """
    from warnings import warn

    warn(
        "plot_rams_and_trajectories is deprecated. Use plot_gridded_and_trajectories"
        " instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return plot_gridded_and_trajectories(pv_config, pv_datas)


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
