"""
Compose two Blender renders into one labelled side-by-side figure.

Shared by the RAMS figure scripts that pair a physically based cloud (left)
with the same view showing something inside it (right): streamlines, dye.
Blender's own colorbar overlay is hard to read at figure size, so the
scripts turn it off and this module draws a larger one with PIL.
"""

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Figure layout (px).
PANEL_DIVIDER_WIDTH = 6
TITLE_FONT_SIZE, LABEL_FONT_SIZE = 34, 26
COLORBAR_WIDTH, COLORBAR_HEIGHT = 420, 26
COLORBAR_MARGIN_RIGHT, COLORBAR_MARGIN_BOTTOM = 40, 110


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
    Paste both renders side by side and draw titles, a note and a colorbar.

    Args:
        left_png, right_png: The two renders, of equal size.
        labels: Text: ``left_title``, ``right_title``, ``time`` (under the
            left title) and ``right_note`` (lower left of the right panel).
        colorbar: The colorbar, drawn in the lower right of the right panel.
        output_png: Destination file.
    """
    from matplotlib import colormaps

    left_panel = Image.open(left_png).convert("RGB")
    right_panel = Image.open(right_png).convert("RGB")
    panel_width, panel_height = left_panel.size
    canvas = Image.new(
        "RGB", (2 * panel_width + PANEL_DIVIDER_WIDTH, panel_height), "white"
    )
    canvas.paste(left_panel, (0, 0))
    right_left = panel_width + PANEL_DIVIDER_WIDTH
    canvas.paste(right_panel, (right_left, 0))
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default(size=TITLE_FONT_SIZE)
    label_font = ImageFont.load_default(size=LABEL_FONT_SIZE)
    text_style = {"fill": "white", "stroke_width": 3, "stroke_fill": "black"}

    # Titles, time and note.
    draw.text((24, 20), labels["left_title"], font=title_font, **text_style)
    draw.text(
        (right_left + 24, 20), labels["right_title"], font=title_font, **text_style
    )
    draw.text((24, 66), labels["time"], font=label_font, **text_style)
    draw.multiline_text(
        (right_left + 24, panel_height - 100),
        labels["right_note"],
        font=label_font,
        spacing=8,
        **text_style,
    )

    # Colorbar in the lower right of the right panel.
    bar_left = canvas.width - COLORBAR_WIDTH - COLORBAR_MARGIN_RIGHT
    bar_top = panel_height - COLORBAR_MARGIN_BOTTOM
    bar_colors = colormaps[colorbar.cmap](np.linspace(0, 1, COLORBAR_WIDTH))[:, :3]
    bar_pixels = np.repeat(bar_colors[None], COLORBAR_HEIGHT, axis=0)
    canvas.paste(
        Image.fromarray((bar_pixels * 255).astype(np.uint8)), (bar_left, bar_top)
    )
    draw.rectangle(
        [
            bar_left - 1,
            bar_top - 1,
            bar_left + COLORBAR_WIDTH,
            bar_top + COLORBAR_HEIGHT,
        ],
        outline="black",
        width=2,
    )
    # Tick labels under the bar, placed linearly in data value.
    bar_minimum, bar_maximum = colorbar.clim
    for tick_value in colorbar.tick_values:
        tick_x = bar_left + (tick_value - bar_minimum) / (bar_maximum - bar_minimum) * (
            COLORBAR_WIDTH - 1
        )
        draw.text(
            (tick_x, bar_top + COLORBAR_HEIGHT + 10),
            colorbar.tick_format.format(tick_value * colorbar.tick_scale),
            font=label_font,
            anchor="ma",
            **text_style,
        )
    draw.text(
        (bar_left + COLORBAR_WIDTH / 2, bar_top - 8),
        colorbar.title,
        font=label_font,
        anchor="md",
        **text_style,
    )
    canvas.save(output_png)
