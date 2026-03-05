from typing import Any

import pyvista as pv


def initialize_plotter(
    background: str = "#f8f6f1",
    add_axes: bool = True,
    show_grid: bool = False,
    **kwargs: Any,
) -> pv.Plotter:
    """
    Initialize a PyVista plotter with atmospheric modeling defaults.

    Creates a plotter optimized for atmospheric data visualization with
    appropriate lighting, background, and coordinate scaling suitable for
    atmospheric data.

    Args:
        background (str, optional): Background color hex code, or None for
            no background. Defaults to "#f8f6f1" (light beige).
        add_axes (bool, optional): Whether to add 3D axes widget to viewport.
            Defaults to True.
        show_grid (bool, optional): DEPRECATED. This parameter no longer has any effect.
            To show grid with correct bounds, either:
            - Use PVConfig(show_grid=True) when plotting, or
            - Call plotter.show_grid() manually after adding meshes
        **kwargs: Additional arguments passed to pv.Plotter constructor.

    Returns:
        pyvista.Plotter: Configured plotter object ready for atmospheric data.

    Example:
        >>> plotter = initialize_plotter(background="white")
        >>> # Add your meshes...
        >>> plotter.show_grid()  # Show grid after meshes for correct bounds
    """
    # Initialize with sensible lighting and jupyter compatibility
    p = pv.Plotter(off_screen=True, lighting="three lights", **kwargs)

    # Configure all subplots if multiple are present
    for row in range(p.shape[0]):
        for col in range(p.shape[1]):
            p.subplot(row, col)
            if background:
                p.set_background(background)
            if add_axes:
                p.add_axes(viewport=(0.0, 0.0, 0.3, 0.3))
            # Exaggerate z scale for atmospheric data (typical aspect ratio)
            p.set_scale(zscale=3)
            # Note: show_grid parameter is deprecated - don't call it here
            # Grid should be shown after meshes are added for correct bounds
            # Use PVConfig.show_grid instead or call plotter.show_grid() manually

    return p
