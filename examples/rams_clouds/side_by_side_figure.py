"""
Compose two Blender renders into one labelled side-by-side figure.

Shared by the RAMS figure scripts that pair a physically based cloud (left)
with the same view showing something inside it (right): streamlines, dye.
A thin layer over :mod:`skyvista.compose`, which does the drawing; this
module fixes the layout those scripts share. Blender's own colorbar overlay
is hard to read at figure size, so the scripts turn it off and the composer
draws a larger one.
"""

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import skyvista as sv

# Gap between the two panels (px).
PANEL_DIVIDER_WIDTH = 6
# Text heights, as fractions of the panel height.
TITLE_SIZE, LABEL_SIZE = 0.031, 0.024


@dataclass
class ColorbarSpec:
    """
    A horizontal colorbar for the right panel.

    Attributes:
        cmap: Matplotlib colormap name.
        clim: Data values at the left and right ends of the bar.
        tick_values: Data values to label.
        title: Text above the bar.
        tick_format: Format of each tick label, e.g. "{:g}".
        tick_scale: Factor applied to tick values before formatting, e.g.
            0.001 to label meters in kilometers.
    """

    cmap: str
    clim: Tuple[float, float]
    tick_values: Sequence[float]
    title: str
    tick_format: str = "{:g}"
    tick_scale: float = 1.0


def compose_side_by_side(
    left_png, right_png, labels: Dict[str, str], colorbar: ColorbarSpec, output_png
) -> None:
    """
    Place both renders side by side, with titles, a note and a colorbar.

    Args:
        left_png, right_png: The two renders, of equal size.
        labels: Text: ``left_title``, ``right_title``, ``time`` (under the
            left title) and ``right_note`` (lower left of the right panel).
        colorbar: The colorbar, drawn in the lower right of the right panel.
        output_png: Destination file.
    """
    compose = sv.compose
    left_panel = compose.Panel(
        left_png,
        [
            compose.Text(labels["left_title"], size=TITLE_SIZE),
            compose.Text(labels["time"], size=LABEL_SIZE, offset=(0.0, -0.045)),
        ],
    )
    right_panel = compose.Panel(
        right_png,
        [
            compose.Text(labels["right_title"], size=TITLE_SIZE),
            compose.Text(labels["right_note"], anchor="lower left", size=LABEL_SIZE),
            compose.Colorbar(
                cmap=colorbar.cmap,
                clim=colorbar.clim,
                label=colorbar.title,
                ticks=list(colorbar.tick_values),
                tick_format=colorbar.tick_format,
                tick_scale=colorbar.tick_scale,
                anchor="lower right",
                length=0.3,
                thickness=0.025,
                margin=0.07,
                text_size=LABEL_SIZE,
            ),
        ],
    )
    compose.Figure(panels=[left_panel, right_panel], gap=PANEL_DIVIDER_WIDTH).render(
        output_png
    )
