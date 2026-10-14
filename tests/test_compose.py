"""Tests for the figure composer (skyvista.compose)."""

import matplotlib.image as mpimg
import numpy as np
import pytest

from skyvista.compose import (
    Colorbar,
    Figure,
    FrameContext,
    Inset,
    Panel,
    Text,
    TimeStamp,
    format_elapsed_time,
)

# Size of the synthetic panels (px).
PANEL_HEIGHT, PANEL_WIDTH = 120, 160


def solid_panel(gray_level: float) -> np.ndarray:
    """A uniform RGB image of the panel size."""
    return np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), gray_level)


def test_canvas_is_exactly_the_panels_plus_gaps(tmp_path):
    figure = Figure(panels=[Panel(solid_panel(0.2)), Panel(solid_panel(0.8))], gap=6)
    output = mpimg.imread(figure.render(tmp_path / "figure.png"))
    assert output.shape[:2] == (PANEL_HEIGHT, 2 * PANEL_WIDTH + 6)
    # Each panel's pixels land where its image is, unchanged.
    assert output[PANEL_HEIGHT // 2, 10, 0] == pytest.approx(0.2, abs=0.01)
    assert output[PANEL_HEIGHT // 2, PANEL_WIDTH + 6 + 10, 0] == pytest.approx(
        0.8, abs=0.01
    )


def test_grid_layout_wraps_rows(tmp_path):
    panels = [Panel(solid_panel(level)) for level in (0.1, 0.3, 0.5, 0.7)]
    figure = Figure(panels=panels, n_columns=2, gap=4)
    output = mpimg.imread(figure.render(tmp_path / "grid.png"))
    assert output.shape[:2] == (2 * PANEL_HEIGHT + 4, 2 * PANEL_WIDTH + 4)
    # Third panel: second row, first column.
    assert output[PANEL_HEIGHT + 4 + 10, 10, 0] == pytest.approx(0.5, abs=0.01)


def test_text_is_drawn_at_its_anchor_only(tmp_path):
    """White text in the upper left changes that corner and nothing opposite."""
    figure = Figure(
        panels=[
            Panel(solid_panel(0.5), [Text("Title", anchor="upper left", size=0.15)])
        ]
    )
    output = mpimg.imread(figure.render(tmp_path / "text.png"))[..., 0]
    upper_left = output[: PANEL_HEIGHT // 3, : PANEL_WIDTH // 2]
    lower_right = output[2 * PANEL_HEIGHT // 3 :, PANEL_WIDTH // 2 :]
    assert np.abs(upper_left - 0.5).max() > 0.3
    assert np.abs(lower_right - 0.5).max() < 0.01


def test_per_frame_sources_receive_the_frame(tmp_path):
    """Images and text can be functions of the frame context."""
    seen_frames = []

    def image_for_frame(frame):
        seen_frames.append(frame.index)
        return solid_panel(0.25 * (frame.index + 1))

    figure = Figure(panels=[Panel(image_for_frame, [TimeStamp()])])
    context = FrameContext(index=2, n_frames=5, time_seconds=4520.0)
    output = mpimg.imread(figure.render(tmp_path / "frame.png", context))
    assert seen_frames == [2]
    assert output[PANEL_HEIGHT // 2, PANEL_WIDTH // 2, 0] == pytest.approx(
        0.75, abs=0.01
    )


def test_elapsed_time_format():
    assert format_elapsed_time(4520.0) == "1:15:20"
    assert format_elapsed_time(59.6) == "0:01:00"


def test_colorbar_and_inset_draw(tmp_path):
    """A colorbar in the lower right shows the colormap's two ends."""
    import matplotlib

    panel = Panel(
        np.full((400, 600, 3), 0.5),
        [
            Colorbar(
                cmap="viridis",
                clim=(0.0, 10.0),
                label="value",
                anchor="lower right",
                length=0.4,
                thickness=0.05,
                margin=0.1,
            ),
            Inset(np.zeros((50, 100, 3)), anchor="upper right", width=0.2),
        ],
    )
    output = mpimg.imread(Figure(panels=[panel]).render(tmp_path / "bar.png"))
    # Bar spans x = 600 * (1 - 0.1 * 400 / 600 - 0.4) .. at y just above margin.
    bar_row = int(400 * (1 - 0.1 - 0.025))
    bar_right = int(600 - 0.1 * 400) - 4
    bar_left = int(600 - 0.1 * 400 - 0.4 * 600) + 4
    viridis = matplotlib.colormaps["viridis"]
    np.testing.assert_allclose(
        output[bar_row, bar_left, :3], viridis(0.0)[:3], atol=0.06
    )
    np.testing.assert_allclose(
        output[bar_row, bar_right, :3], viridis(1.0)[:3], atol=0.06
    )
    # The black inset darkens the upper right.
    assert output[20, 600 - 30, 0] < 0.1
