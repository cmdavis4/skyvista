"""
Standalone: a RAMS storm, side by side, as a physically based cloud and cut
open along a vertical plane with the 3-D wind drawn on the cut.

Everything needed is in this one file; it depends only on xarrams (to read
RAMS output), skyvista (to export to Blender), and numpy / scipy / matplotlib
/ Pillow. Example::

    python standalone_rams_wind_cutaway.py \\
        /path/to/output/a-L-2000-01-01-013420-g1.h5  wind_figure \\
        --blender ~/programs/blender-5.0.1-linux-x64/blender

The environment needs OpenVDB's Python bindings for the cloud volumes
(conda-forge ``openvdb<13``: Blender 5.0.1 cannot read OpenVDB 13 files) and
``hdf5plugin`` for ZFP-compressed lite files (xarrams imports it).

What it does
------------
1. Read one output time with xarrams: hydrometeor mixing ratios and number
   concentrations, base-state density, and the winds ``UC``, ``VC``, ``WC``.
   The winds are used as stored, on RAMS's staggered C grid (each component
   half a cell from the cell centre, in its own direction); they are not
   averaged onto cell centres.
2. Left panel: export and render the physically based cloud with skyvista's
   automatic camera.
3. Right panel: the same camera (pinned from the left panel's manifest), the
   cloud with its camera-side half (y below the arrow plane) cut away, and
   arrows on that plane:
   * the plane passes through the strongest updraft;
   * arrows sit on every ``ARROW_STRIDE``-th grid point in x and z, are drawn
     ``ARROW_OFFSET_TOWARD_CAMERA`` meters in front of the cut face, and show
     the full 3-D wind (length proportional to speed, color by w);
   * an arrow is dropped when its vertical motion is weak AND it is far from
     any cloud, since weak flow in clear air far from the storm is clutter.
4. Compose the two renders with titles, labels and a w colorbar.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import xarrams
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

import skyvista as sv

# =============================================================================
# Settings
# =============================================================================
# ---- Data
# RAMS variables read: mixing ratios (kg/kg), number concentrations (#/kg),
# base-state density (kg/m^3), and the staggered wind components (m/s).
RAMS_MIXING_RATIOS = ("RCP", "RDP", "RRP", "RPP", "RSP", "RAP", "RGP", "RHP")
RAMS_NUMBER_CONCENTRATIONS = ("CCP", "CDP", "CRP", "CPP", "CSP", "CAP", "CGP", "CHP")
RAMS_WIND_COMPONENTS = ("UC", "VC", "WC")
RAMS_AIR_DENSITY = "DN0"
# Model start, for the "t = ... min" label (RAMS idealized runs start here).
RUN_START = np.datetime64("2000-01-01T00:00:00")
# Levels kept: above the ground (RAMS's first level is below it, at -dz/2) and
# below the stratosphere, which clouds rarely reach.
BOTTOM_HEIGHT_M = 0.0
TOP_HEIGHT_M = 16500.0

# ---- Render
PANEL_WIDTH_PX, PANEL_HEIGHT_PX = 1440, 1080
SUN_ELEVATION_DEG = 30.0
# Direction toward the sun, counterclockwise from +x.
SUN_AZIMUTH_DEG = 15.0
# Camera direction from the scene centre (counterclockwise from +x), height
# angle (negative = low camera looking up), and distance in bounding-box
# diagonals.
CAMERA_AZIMUTH_DEG = -55.0
CAMERA_ELEVATION_DEG = -8.0
CAMERA_DISTANCE_FACTOR = 1.1

# ---- Arrows
# Keep every n-th grid point along x and z (50 m grid -> 600 m apart).
ARROW_STRIDE = 12
# Arrow length (m) per m/s of wind speed, and shaft diameter (m).
ARROW_LENGTH_PER_MPS = 80.0
ARROW_SHAFT_WIDTH_M = 65.0
# Share of each arrow's on-screen color it emits itself; sun and sky add
# shading on top. 1 keeps colors readable against shaded cloud.
ARROW_SELF_ILLUMINATION = 1.0
# Arrows are drawn this far toward the camera (-y) from the cut, so arrows
# whose wind points into the cloud are not hidden behind the cut face.
ARROW_OFFSET_TOWARD_CAMERA_M = 300.0
# Colormap ends (m/s) for w; stronger w saturates at the end colors.
W_COLOR_LIMIT_MPS = 6.0
# Blue -> mid-grey -> red: the grey centre reads against the white cut face
# and the blue sky; hue gives the sign of w and saturation its strength.
ARROW_COLORMAP_NAME = "blue_grey_red"
ARROW_COLORMAP_COLORS = ("#2166ac", "#808080", "#b2182b")

# ---- Clutter filter: drop an arrow when BOTH hold
# |w| below this (m/s) ...
WEAK_W_MPS = 2.0
# ... and farther than this (m, 3-D) from any cloud.
NEAR_CLOUD_DISTANCE_M = 1000.0
# A cell is "cloud" when its total hydrometeor mixing ratio exceeds this.
CLOUD_MIXING_RATIO_THRESHOLD = 1.0e-5

# ---- Figure
PANEL_DIVIDER_WIDTH_PX = 6
TITLE_FONT_SIZE, LABEL_FONT_SIZE = 34, 26
COLORBAR_WIDTH_PX, COLORBAR_HEIGHT_PX = 420, 26
COLORBAR_MARGIN_RIGHT_PX, COLORBAR_MARGIN_BOTTOM_PX = 40, 110


# =============================================================================
# Data
# =============================================================================
def read_rams_time(data_path: Path):
    """
    Read one RAMS output time with xarrams and load it into memory.

    xarrams names the dimensions (z, y, x) and assigns the header's
    coordinates. ``parallel=True`` is required: in xarrams 1.1.0 the serial
    path skips that dimension naming. The result is cropped to
    [BOTTOM_HEIGHT_M, TOP_HEIGHT_M], reduced to its single time, and loaded,
    because every later step reads the arrays several times.
    """
    wanted_variables = (
        list(RAMS_MIXING_RATIOS)
        + list(RAMS_NUMBER_CONCENTRATIONS)
        + list(RAMS_WIND_COMPONENTS)
        + [RAMS_AIR_DENSITY]
    )
    dataset = xarrams.read_rams_output(
        [data_path], parallel=True, silent=True, keep_vars=wanted_variables
    )
    # One time: drop the time dimension and the per-time file name, but keep
    # the time itself (for the label) as an attribute.
    model_time = dataset["time"].values[0]
    dataset = dataset.isel(time=0).drop_vars(["time", "source_file"])
    dataset = dataset.sel(z=slice(BOTTOM_HEIGHT_M, TOP_HEIGHT_M)).load()
    return dataset, model_time


def strongest_updraft_y_index(dataset) -> int:
    """y index of the grid column that holds the maximum vertical velocity."""
    w_values = dataset["WC"].transpose("z", "y", "x").values
    _, y_index, _ = np.unravel_index(int(np.nanargmax(w_values)), w_values.shape)
    return int(y_index)


def distance_to_cloud_on_plane(dataset, plane_y_index: int) -> np.ndarray:
    """
    3-D distance (m) from each cell of one y row to the nearest cloudy cell.

    "Cloudy" means total hydrometeor mixing ratio above
    CLOUD_MIXING_RATIO_THRESHOLD. The Euclidean distance transform runs on a
    band of rows around the plane, wide enough to hold any cloud within
    NEAR_CLOUD_DISTANCE_M, instead of the whole domain: distances up to that
    are exact, and larger ones only need to be "larger", which is all the
    filter asks. Returns a (z, x) array; inf when the band has no cloud.
    """
    # Cloud mask from the sum of every hydrometeor mixing ratio present.
    total_mixing_ratio = sum(
        dataset[name] for name in RAMS_MIXING_RATIOS if name in dataset
    )
    # Grid spacings (m) along z, y, x, for an anisotropic distance.
    spacing_zyx = [
        float(np.mean(np.diff(dataset[axis].values))) for axis in ("z", "y", "x")
    ]
    # Rows within the filter distance (+1 for rounding) of the plane.
    band_half_width = int(np.ceil(NEAR_CLOUD_DISTANCE_M / spacing_zyx[1])) + 1
    band_start = max(plane_y_index - band_half_width, 0)
    band_stop = min(plane_y_index + band_half_width + 1, dataset.sizes["y"])
    cloudy_band = (
        total_mixing_ratio.isel(y=slice(band_start, band_stop))
        .transpose("z", "y", "x")
        .values
        > CLOUD_MIXING_RATIO_THRESHOLD
    )
    if not cloudy_band.any():
        return np.full((dataset.sizes["z"], dataset.sizes["x"]), np.inf)
    # Distance from each clear cell to the nearest cloudy one (0 inside cloud).
    distance_band = ndimage.distance_transform_edt(~cloudy_band, sampling=spacing_zyx)
    return distance_band[:, plane_y_index - band_start, :]


def arrow_plane(dataset, plane_y_index: int):
    """
    The wind on one y row, with the clutter mask, moved toward the camera.

    Keeps y as a length-1 dimension (skyvista's glyphs need x, y and z). The
    mask variable ``show_arrow`` is True where |w| is at least WEAK_W_MPS or
    the cell lies within NEAR_CLOUD_DISTANCE_M of cloud. The y coordinate is
    then shifted by -ARROW_OFFSET_TOWARD_CAMERA_M so the arrows render in
    front of the cut face; the mask was computed at the true position.
    """
    plane = dataset[list(RAMS_WIND_COMPONENTS)].isel(y=[plane_y_index])
    plane = plane.transpose("z", "y", "x")
    # Clutter filter.
    strong_vertical_motion = np.abs(plane["WC"].values[:, 0, :]) >= WEAK_W_MPS
    near_cloud = (
        distance_to_cloud_on_plane(dataset, plane_y_index) <= NEAR_CLOUD_DISTANCE_M
    )
    show_arrow = (strong_vertical_motion | near_cloud)[:, None, :]
    plane = plane.assign(show_arrow=(("z", "y", "x"), show_arrow))
    # Draw in front of the cut face.
    return plane.assign_coords(y=plane["y"] - ARROW_OFFSET_TOWARD_CAMERA_M)


# =============================================================================
# Blender export
# =============================================================================
def cloud_export_config() -> sv.BlenderExportConfig:
    """Physical-sky cloud settings with this figure's framing and quality."""
    config = sv.BlenderExportConfig.for_clouds()
    config.render.resolution = (PANEL_WIDTH_PX, PANEL_HEIGHT_PX)
    config.world.sun_elevation_deg = SUN_ELEVATION_DEG
    config.world.sun_azimuth_deg = SUN_AZIMUTH_DEG
    config.camera.azimuth_deg = CAMERA_AZIMUTH_DEG
    config.camera.elevation_deg = CAMERA_ELEVATION_DEG
    config.camera.distance_factor = CAMERA_DISTANCE_FACTOR
    return config


def pinned_view(bundle_dir: Path):
    """Camera location, aim point and scene origin shift of a built bundle."""
    manifest = json.loads((bundle_dir / "scene.json").read_text())
    (keyframe,) = manifest["camera"]["keyframes"]
    return (
        tuple(keyframe["location"]),
        tuple(keyframe["look_at"]),
        tuple(manifest["transform"]["origin_shift"]),
    )


def register_arrow_colormap() -> None:
    """Register the blue-grey-red map by name (the exporter looks it up so)."""
    if ARROW_COLORMAP_NAME not in colormaps:
        colormaps.register(
            LinearSegmentedColormap.from_list(
                ARROW_COLORMAP_NAME, ARROW_COLORMAP_COLORS
            )
        )


def make_arrow_spec():
    """Instanced arrows: full 3-D wind, length by speed, color by w, masked."""
    arrows = sv.make_vectors(
        "wind",
        u="UC",
        v="VC",
        w="WC",
        stride={"x": ARROW_STRIDE, "z": ARROW_STRIDE},
        color_by="w",
        mask_by="show_arrow",
        cmap=ARROW_COLORMAP_NAME,
        clim=(-W_COLOR_LIMIT_MPS, W_COLOR_LIMIT_MPS),
        factor=ARROW_LENGTH_PER_MPS,
        glyph_width=ARROW_SHAFT_WIDTH_M,
    )
    # Self-lit, so an arrow's color reads on its shadowed side too.
    arrows.appearance.shader = "self_lit"
    arrows.appearance.shader_overrides = {"emission_strength": ARROW_SELF_ILLUMINATION}
    return arrows


# =============================================================================
# Figure
# =============================================================================
def draw_w_colorbar(canvas, draw, left: int, top: int, font, text_style) -> None:
    """Horizontal colorbar of the arrow colormap, with ticks and a title."""
    # Bar pixels: the colormap sampled across the bar's width.
    fractions = np.linspace(0.0, 1.0, COLORBAR_WIDTH_PX)
    bar_colors = colormaps[ARROW_COLORMAP_NAME](fractions)[:, :3]
    bar_pixels = np.repeat(bar_colors[None, :, :], COLORBAR_HEIGHT_PX, axis=0)
    canvas.paste(Image.fromarray((bar_pixels * 255).astype(np.uint8)), (left, top))
    draw.rectangle(
        [left - 1, top - 1, left + COLORBAR_WIDTH_PX, top + COLORBAR_HEIGHT_PX],
        outline="black",
        width=2,
    )
    # Five ticks from -limit to +limit.
    for tick_value in np.linspace(-W_COLOR_LIMIT_MPS, W_COLOR_LIMIT_MPS, 5):
        tick_fraction = (tick_value + W_COLOR_LIMIT_MPS) / (2 * W_COLOR_LIMIT_MPS)
        tick_x = left + tick_fraction * (COLORBAR_WIDTH_PX - 1)
        draw.line(
            [tick_x, top + COLORBAR_HEIGHT_PX, tick_x, top + COLORBAR_HEIGHT_PX + 8],
            fill="black",
            width=2,
        )
        draw.text(
            (tick_x, top + COLORBAR_HEIGHT_PX + 10),
            f"{tick_value:g}",
            font=font,
            anchor="ma",
            **text_style,
        )
    draw.text(
        (left + COLORBAR_WIDTH_PX / 2, top - 8),
        "Vertical velocity w (m/s)",
        font=font,
        anchor="md",
        **text_style,
    )


def compose_side_by_side(
    left_png: Path, right_png: Path, labels: dict, output_png: Path
) -> None:
    """
    Both renders side by side, with titles, the time, a note and a colorbar.

    ``labels`` holds "left_title", "right_title", "time" and "right_note".
    Text is white with a black outline, so it reads on sky, cloud and ground.
    """
    left_panel = Image.open(left_png).convert("RGB")
    right_panel = Image.open(right_png).convert("RGB")
    panel_width, panel_height = left_panel.size
    canvas = Image.new(
        "RGB", (2 * panel_width + PANEL_DIVIDER_WIDTH_PX, panel_height), "white"
    )
    canvas.paste(left_panel, (0, 0))
    right_panel_left = panel_width + PANEL_DIVIDER_WIDTH_PX
    canvas.paste(right_panel, (right_panel_left, 0))

    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default(size=TITLE_FONT_SIZE)
    label_font = ImageFont.load_default(size=LABEL_FONT_SIZE)
    text_style = {"fill": "white", "stroke_width": 3, "stroke_fill": "black"}
    # Titles and the model time.
    draw.text((24, 20), labels["left_title"], font=title_font, **text_style)
    draw.text(
        (right_panel_left + 24, 20),
        labels["right_title"],
        font=title_font,
        **text_style,
    )
    draw.text((24, 66), labels["time"], font=label_font, **text_style)
    # Arrow note, lower left of the right panel.
    draw.multiline_text(
        (right_panel_left + 24, panel_height - 100),
        labels["right_note"],
        font=label_font,
        spacing=8,
        **text_style,
    )
    # Colorbar, lower right of the right panel.
    draw_w_colorbar(
        canvas,
        draw,
        canvas.width - COLORBAR_WIDTH_PX - COLORBAR_MARGIN_RIGHT_PX,
        panel_height - COLORBAR_MARGIN_BOTTOM_PX,
        label_font,
        text_style,
    )
    canvas.save(output_png)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_path", type=Path, help="RAMS a-L/a-A ...-g1.h5 file.")
    parser.add_argument("output_dir", type=Path, help="Directory for the outputs.")
    parser.add_argument(
        "--blender",
        default=None,
        help="Blender executable (default: $SKYVISTA_BLENDER, then PATH).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=16,
        help="Cycles samples per pixel (16 is plenty with the denoiser).",
    )
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data
    dataset, model_time = read_rams_time(arguments.data_path)
    minutes = float((model_time - RUN_START) / np.timedelta64(1, "s")) / 60.0
    cloud_fields = dataset[
        [
            name
            for name in RAMS_MIXING_RATIOS + RAMS_NUMBER_CONCENTRATIONS
            if name in dataset
        ]
        + [RAMS_AIR_DENSITY]
    ]

    # ---- Left panel: the physically based cloud, automatic camera.
    realistic_config = cloud_export_config()
    realistic_config.render.samples = arguments.samples
    realistic_dir = arguments.output_dir / "realistic"
    sv.Scene().add_clouds(cloud_fields, microphysics="rams").to_blender(
        realistic_dir,
        config=realistic_config,
        render=True,
        blender_executable=arguments.blender,
    )

    # ---- Right panel: same view, cut cloud, arrows on the cut.
    camera_location, camera_look_at, origin_shift = pinned_view(realistic_dir)
    cutaway_config = cloud_export_config()
    cutaway_config.render.samples = arguments.samples
    cutaway_config.transform.origin_shift = origin_shift
    cutaway_config.camera.location = camera_location
    cutaway_config.camera.look_at = camera_look_at
    # The figure draws its own colorbar.
    cutaway_config.overlay_colorbars = False

    # The plane through the strongest updraft; the cut removes y below it.
    plane_y_index = strongest_updraft_y_index(dataset)
    plane_y = float(dataset["y"].values[plane_y_index])
    register_arrow_colormap()
    cutaway_scene = sv.Scene().add_clouds(
        cloud_fields,
        microphysics="rams",
        cutaway=((0.0, plane_y, 0.0), (0.0, -1.0, 0.0)),
    )
    cutaway_scene.add(arrow_plane(dataset, plane_y_index), make_arrow_spec())
    cutaway_dir = arguments.output_dir / "cutaway"
    cutaway_scene.to_blender(
        cutaway_dir,
        config=cutaway_config,
        render=True,
        blender_executable=arguments.blender,
    )

    # ---- Side by side.
    arrow_spacing_m = ARROW_STRIDE * float(np.diff(dataset["x"].values[:2])[0])
    output_png = arguments.output_dir / f"winds_cutaway_t{minutes:05.1f}min.png"
    compose_side_by_side(
        realistic_dir / "render" / "frame_0001.png",
        cutaway_dir / "render" / "frame_0001.png",
        {
            "left_title": "Cloud, physically based",
            "right_title": f"Cut open at y = {plane_y / 1000:.2f} km, with the 3-D wind",
            "time": f"t = {minutes:.1f} min",
            "right_note": (
                f"Arrows every {arrow_spacing_m:.0f} m; "
                f"{ARROW_LENGTH_PER_MPS:g} m of arrow per m/s\n"
                f"Arrows drawn {ARROW_OFFSET_TOWARD_CAMERA_M:g} m in front of the cut"
            ),
        },
        output_png,
    )
    print(f"Figure: {output_png}")


if __name__ == "__main__":
    main()
