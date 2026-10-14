"""
Compose rendered images into labelled figures.

A :class:`Figure` is a grid of :class:`Panel` objects. Each panel shows one
image (a Blender render, a PyVista screenshot, any PNG) and carries overlays
placed by a corner anchor: :class:`Text`, :class:`TimeStamp`,
:class:`Colorbar` and :class:`Inset`. The figure can also carry overlays of
its own, placed relative to the whole canvas.

Drawing uses matplotlib on a canvas exactly the size of the composed images,
in pixels: each panel is an axes covering its image's pixels, and overlay
sizes are fractions of the panel height. A layout therefore looks the same at
any render resolution, and matplotlib supplies real fonts, math text (e.g.
``"m s$^{-1}$"``) and colorbar ticks.

Animation frames reuse one figure as a template. Any text, and any panel
image, may be a function of a :class:`FrameContext` instead of a constant, so
the same figure definition composes every frame of a video (e.g. a
:class:`TimeStamp` that reads each frame's model time, and panels whose image
is that frame's render).

Example::

    figure = Figure(
        panels=[
            Panel("left.png", [Text("Physical cloud", anchor="upper left")]),
            Panel(
                "right.png",
                [Colorbar(cmap="RdBu_r", clim=(-12, 12), label="w (m s$^{-1}$)")],
            ),
        ]
    )
    figure.render("figure.png")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

# =============================================================================
# Frame context: what a per-frame text or image source receives
# =============================================================================


@dataclass
class FrameContext:
    """
    Describes one frame of an animation, for per-frame overlay content.

    Attributes:
        index: Zero-based frame index.
        n_frames: Number of frames in the animation.
        time: The model time this frame shows, as the animation was given it.
        time_seconds: The same time in seconds, for formatting.
        render_path: This frame's rendered image, when composing an animation.
    """

    index: int = 0
    n_frames: int = 1
    time: Any = None
    time_seconds: Optional[float] = None
    render_path: Optional[Path] = None


# A constant, or a function of the frame.
TextSource = Union[str, Callable[[FrameContext], str]]
ImageSource = Union[str, Path, np.ndarray, Callable[[FrameContext], Any]]


def _resolve(source, context: FrameContext):
    """Evaluate a per-frame source, or return a constant unchanged."""
    return source(context) if callable(source) else source


def _load_image(source, context: FrameContext) -> np.ndarray:
    """An image source as an (height, width, channels) float array in [0, 1]."""
    import matplotlib.image as mpimg

    resolved = _resolve(source, context)
    if isinstance(resolved, np.ndarray):
        image = resolved
    else:
        image = mpimg.imread(str(resolved))
    # Integer images (e.g. JPEG) come back as 0-255.
    if image.dtype.kind in "ui":
        image = image.astype(float) / 255.0
    return image


# =============================================================================
# Anchors: where an overlay sits within its panel
# =============================================================================

# Anchor name -> (x, y) of the corner in axes fractions, and the matching
# horizontal / vertical text alignment.
ANCHORS = {
    "upper left": ((0.0, 1.0), "left", "top"),
    "upper center": ((0.5, 1.0), "center", "top"),
    "upper right": ((1.0, 1.0), "right", "top"),
    "center left": ((0.0, 0.5), "left", "center"),
    "center": ((0.5, 0.5), "center", "center"),
    "center right": ((1.0, 0.5), "right", "center"),
    "lower left": ((0.0, 0.0), "left", "bottom"),
    "lower center": ((0.5, 0.0), "center", "bottom"),
    "lower right": ((1.0, 0.0), "right", "bottom"),
}


def _anchor_point(
    anchor: str, margin: float, aspect: float
) -> Tuple[Tuple[float, float], str, str]:
    """
    The anchor's position, inset by ``margin`` (a fraction of panel height).

    The margin is measured in panel heights on both axes, so it is the same
    number of pixels horizontally as vertically; ``aspect`` (width / height)
    converts it to a horizontal axes fraction.
    """
    if anchor not in ANCHORS:
        raise ValueError(f"Unknown anchor {anchor!r}; choose from {sorted(ANCHORS)}.")
    (x, y), horizontal_alignment, vertical_alignment = ANCHORS[anchor]
    horizontal_margin = margin / aspect
    # Move inward from whichever edge the anchor touches.
    x += horizontal_margin if x == 0.0 else (-horizontal_margin if x == 1.0 else 0.0)
    y += margin if y == 0.0 else (-margin if y == 1.0 else 0.0)
    return (x, y), horizontal_alignment, vertical_alignment


def _outline_effects(outline: bool, width_points: float):
    """A dark stroke behind light text, so it reads on any background."""
    if not outline:
        return None
    import matplotlib.patheffects as path_effects

    return [path_effects.withStroke(linewidth=width_points, foreground="black")]


# =============================================================================
# Overlays
# =============================================================================


@dataclass
class Text:
    """
    Text placed at a corner of a panel (or of the figure).

    Attributes:
        text: The text, or a function of the :class:`FrameContext`.
        anchor: Corner or edge to place it at (see ``ANCHORS``).
        size: Font height as a fraction of the panel height.
        margin: Inset from the anchored edges, as a fraction of panel height.
        offset: Further (x, y) shift, in panel heights, e.g. to stack lines.
        color: Text color.
        outline: Draw a dark stroke behind the text.
        weight: Font weight ("normal", "bold").
    """

    text: TextSource
    anchor: str = "upper left"
    size: float = 0.035
    margin: float = 0.025
    offset: Tuple[float, float] = (0.0, 0.0)
    color: str = "white"
    outline: bool = True
    weight: str = "normal"

    def draw(self, axes, panel_height_px: float, dpi: float, context) -> None:
        """Draw onto ``axes`` (whose extent is the panel)."""
        aspect = axes.bbox.width / axes.bbox.height
        (x, y), horizontal_alignment, vertical_alignment = _anchor_point(
            self.anchor, self.margin, aspect
        )
        x += self.offset[0] / aspect
        y += self.offset[1]
        # Points per pixel at this dpi turn a pixel height into a font size.
        font_size_points = self.size * panel_height_px * 72.0 / dpi
        axes.text(
            x,
            y,
            _resolve(self.text, context),
            transform=axes.transAxes,
            fontsize=font_size_points,
            color=self.color,
            fontweight=self.weight,
            horizontalalignment=horizontal_alignment,
            verticalalignment=vertical_alignment,
            path_effects=_outline_effects(self.outline, font_size_points * 0.15),
            zorder=10,
        )


def format_elapsed_time(seconds: float) -> str:
    """Elapsed model time as ``h:mm:ss`` (e.g. 4520 s -> ``1:15:20``)."""
    whole_seconds = int(round(seconds))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds_part:02d}"


@dataclass
class TimeStamp(Text):
    """
    The frame's model time, e.g. ``"t = 1:15:20"``.

    Attributes:
        text: Unused unless ``formatter`` is None; then a constant label.
        formatter: Function of the time in seconds giving the stamp text.
            The default prints ``"t = h:mm:ss"``.
    """

    text: TextSource = ""
    formatter: Optional[Callable[[float], str]] = None

    def __post_init__(self):
        formatter = self.formatter or (
            lambda seconds: f"t = {format_elapsed_time(seconds)}"
        )

        def stamp(context: FrameContext) -> str:
            if context.time_seconds is None:
                return ""
            return formatter(context.time_seconds)

        self.text = stamp


@dataclass
class Colorbar:
    """
    A colorbar placed at a corner of a panel.

    Attributes:
        cmap: Matplotlib colormap name.
        clim: Data values at the two ends of the bar.
        label: Title of the bar (drawn above a horizontal bar).
        anchor: Corner or edge to place it at.
        orientation: "horizontal" or "vertical".
        length: Bar length as a fraction of the panel width (horizontal) or
            height (vertical).
        thickness: Bar thickness as a fraction of the panel height.
        margin: Inset from the anchored edges, as a fraction of panel height.
            Leaves room for tick labels on the outer side.
        ticks: Data values to label; None lets matplotlib choose.
        tick_format: Format of each tick label, applied after ``tick_scale``.
        tick_scale: Factor applied to tick values before formatting, e.g.
            0.001 to label meters in kilometers.
        text_size: Label and tick font height, as a fraction of panel height.
        color: Label and tick color.
    """

    cmap: str
    clim: Tuple[float, float]
    label: str = ""
    anchor: str = "lower right"
    orientation: str = "horizontal"
    length: float = 0.3
    thickness: float = 0.025
    margin: float = 0.08
    ticks: Optional[Sequence[float]] = None
    tick_format: str = "{:g}"
    tick_scale: float = 1.0
    text_size: float = 0.028
    color: str = "white"

    def draw(self, axes, panel_height_px: float, dpi: float, context) -> None:
        """Draw the bar as its own axes, positioned inside the panel."""
        import matplotlib as mpl
        from matplotlib.ticker import FixedLocator, FuncFormatter

        aspect = axes.bbox.width / axes.bbox.height
        (anchor_x, anchor_y), _, _ = _anchor_point(self.anchor, self.margin, aspect)

        # Bar size in axes fractions of the panel.
        if self.orientation == "horizontal":
            bar_width, bar_height = self.length, self.thickness
        else:
            bar_width, bar_height = self.thickness / aspect, self.length
        # Place the bar so the anchored corner of the bar sits at the anchor.
        horizontal_fraction = ANCHORS[self.anchor][0][0]
        vertical_fraction = ANCHORS[self.anchor][0][1]
        left = anchor_x - horizontal_fraction * bar_width
        bottom = anchor_y - vertical_fraction * bar_height
        bar_axes = axes.inset_axes([left, bottom, bar_width, bar_height])

        # The bar itself.
        norm = mpl.colors.Normalize(vmin=self.clim[0], vmax=self.clim[1])
        colorbar = mpl.colorbar.ColorbarBase(
            bar_axes,
            cmap=mpl.colormaps[self.cmap],
            norm=norm,
            orientation=self.orientation,
        )
        colorbar.outline.set_edgecolor("black")
        colorbar.outline.set_linewidth(1.0)

        # Ticks, labelled with the scaled, formatted value.
        font_size_points = self.text_size * panel_height_px * 72.0 / dpi
        tick_axis = (
            bar_axes.xaxis if self.orientation == "horizontal" else bar_axes.yaxis
        )
        if self.ticks is not None:
            tick_axis.set_major_locator(FixedLocator(list(self.ticks)))
        tick_axis.set_major_formatter(
            FuncFormatter(
                lambda value, _: self.tick_format.format(value * self.tick_scale)
            )
        )
        effects = _outline_effects(True, font_size_points * 0.15)
        bar_axes.tick_params(
            labelsize=font_size_points, labelcolor=self.color, color=self.color
        )
        for tick_label in tick_axis.get_ticklabels():
            tick_label.set_path_effects(effects)

        # Title: above a horizontal bar, beside a vertical one.
        if self.label:
            if self.orientation == "horizontal":
                bar_axes.set_title(
                    self.label,
                    fontsize=font_size_points,
                    color=self.color,
                    path_effects=effects,
                    pad=font_size_points * 0.4,
                )
            else:
                colorbar.set_label(
                    self.label,
                    fontsize=font_size_points,
                    color=self.color,
                    path_effects=effects,
                )


@dataclass
class Inset:
    """
    A small image (e.g. a plan-view map) placed at a corner of a panel.

    Attributes:
        image: Path or array, or a function of the :class:`FrameContext`.
        anchor: Corner or edge to place it at.
        width: Inset width as a fraction of the panel width; the height
            follows from the image's aspect ratio.
        margin: Inset from the anchored edges, as a fraction of panel height.
        border: Draw a thin white border around the inset.
    """

    image: ImageSource
    anchor: str = "upper right"
    width: float = 0.25
    margin: float = 0.025
    border: bool = True

    def draw(self, axes, panel_height_px: float, dpi: float, context) -> None:
        image = _load_image(self.image, context)
        aspect = axes.bbox.width / axes.bbox.height
        (anchor_x, anchor_y), _, _ = _anchor_point(self.anchor, self.margin, aspect)
        # Height in axes fractions that keeps the image's pixel aspect ratio.
        image_height_px, image_width_px = image.shape[:2]
        inset_height = self.width * aspect * image_height_px / image_width_px
        left = anchor_x - ANCHORS[self.anchor][0][0] * self.width
        bottom = anchor_y - ANCHORS[self.anchor][0][1] * inset_height
        inset_axes = axes.inset_axes([left, bottom, self.width, inset_height])
        inset_axes.imshow(image, interpolation="antialiased")
        inset_axes.set_xticks([])
        inset_axes.set_yticks([])
        for spine in inset_axes.spines.values():
            spine.set_visible(self.border)
            spine.set_edgecolor("white")
            spine.set_linewidth(1.5)


Overlay = Union[Text, Colorbar, Inset]


# =============================================================================
# Panels and figures
# =============================================================================


@dataclass
class Panel:
    """
    One image of a figure, with its overlays.

    Attributes:
        image: Path or array, or a function of the :class:`FrameContext`
            (e.g. ``lambda frame: frame.render_path`` for an animation).
        overlays: Overlays drawn on this panel, in order.
    """

    image: ImageSource
    overlays: List[Overlay] = field(default_factory=list)


@dataclass
class Figure:
    """
    A grid of panels on one canvas, with optional figure-wide overlays.

    Panels fill the grid row by row. Every column is as wide as its widest
    image and every row as tall as its tallest, so panels of equal size (the
    usual case: renders with the same resolution) tile without padding.

    Attributes:
        panels: The panels, in reading order.
        n_columns: Panels per row; None puts every panel in one row.
        gap: Space between panels, in pixels.
        background: Canvas color behind and between the panels.
        overlays: Overlays placed relative to the whole canvas.
    """

    panels: List[Panel]
    n_columns: Optional[int] = None
    gap: int = 6
    background: str = "white"
    overlays: List[Overlay] = field(default_factory=list)

    def render(
        self, output_path, context: Optional[FrameContext] = None, dpi: float = 100.0
    ) -> Path:
        """
        Draw the figure and save it as an image exactly the panels' size.

        Args:
            output_path: Destination file (format from the extension).
            context: The frame, for per-frame sources; a still uses the default.
            dpi: Resolution used to convert pixels to points. It does not
                change the output size, only the internal unit conversion.

        Returns:
            The output path.
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure as MatplotlibFigure

        context = context or FrameContext()
        images = [_load_image(panel.image, context) for panel in self.panels]

        # Grid shape and the pixel size of every column and row.
        n_columns = self.n_columns or len(images)
        n_rows = int(np.ceil(len(images) / n_columns))
        column_widths = [0] * n_columns
        row_heights = [0] * n_rows
        for panel_index, image in enumerate(images):
            row, column = divmod(panel_index, n_columns)
            column_widths[column] = max(column_widths[column], image.shape[1])
            row_heights[row] = max(row_heights[row], image.shape[0])
        canvas_width = sum(column_widths) + self.gap * (n_columns - 1)
        canvas_height = sum(row_heights) + self.gap * (n_rows - 1)

        # A canvas of exactly canvas_width x canvas_height pixels.
        figure = MatplotlibFigure(
            figsize=(canvas_width / dpi, canvas_height / dpi),
            dpi=dpi,
            facecolor=self.background,
        )
        FigureCanvasAgg(figure)

        # Each panel is an axes covering its image's pixels (origin top left).
        for panel_index, (panel, image) in enumerate(
            zip(self.panels, images, strict=True)
        ):
            row, column = divmod(panel_index, n_columns)
            left_px = sum(column_widths[:column]) + self.gap * column
            top_px = sum(row_heights[:row]) + self.gap * row
            height_px, width_px = image.shape[:2]
            axes = figure.add_axes(
                [
                    left_px / canvas_width,
                    1.0 - (top_px + height_px) / canvas_height,
                    width_px / canvas_width,
                    height_px / canvas_height,
                ]
            )
            axes.imshow(image, interpolation="none", aspect="auto")
            axes.set_axis_off()
            for overlay in panel.overlays:
                overlay.draw(axes, height_px, dpi, context)

        # Figure-wide overlays sit on an invisible axes over the whole canvas.
        if self.overlays:
            canvas_axes = figure.add_axes([0.0, 0.0, 1.0, 1.0])
            canvas_axes.set_axis_off()
            canvas_axes.patch.set_alpha(0.0)
            for overlay in self.overlays:
                overlay.draw(canvas_axes, canvas_height, dpi, context)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=dpi, facecolor=self.background)
        return output_path
