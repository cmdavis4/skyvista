"""
Render a RAMS storm twice, side by side: the physically based cloud, and the
same view with the 3-D wind drawn as arrows on a vertical plane through the
storm core. The cloud in the arrow panel is either thinned (default) or, with
``--cutaway``, cut open along the arrow plane.

Example (from the repository root, in the pixi ``blender`` environment)::

    pixi run -e blender python examples/rams_clouds/render_rams_wind_side_by_side.py \\
        /path/to/output/a-L-2000-01-01-013420-g1.h5  wind_figure \\
        --blender ~/programs/blender-5.0.1-linux-x64/blender --width 1440

How the two panels are kept identical except for what they show:

1. The left bundle (cloud only) is exported and rendered first, with the
   usual automatic camera.
2. Its manifest's camera position, aim point and scene origin are then pinned
   in the right bundle's config. The right bundle contains extra objects
   (the arrows) whose bounds would otherwise move the automatic camera.

The arrows are instanced in Blender with geometry nodes from exported points
(``Scene.add_vectors``), not baked meshes. Each arrow is the full 3-D wind at
its point, centred from RAMS's staggered grid, so it can point out of the
plane it sits on.

Two presentation choices:

* Cutaway (``--cutaway``): the cloud on the camera side of the arrow plane
  (y below it) is removed, and the arrows are drawn ``--arrow-offset`` meters
  toward the camera, so arrows whose wind points into the cloud are not
  swallowed by the cut face. The remaining half keeps its physical optics.
* Clutter filter (both modes): an arrow is dropped when its vertical motion
  is weak (``|w| < --weak-w``) and it is farther than
  ``--near-cloud-distance`` from any cloud (3-D distance, counting the
  cut-away half too). Weak flow in clear air far from the storm carries
  little information and crowds the figure.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from cloud_render_config import add_render_arguments, make_cloud_export_config
from PIL import Image, ImageDraw, ImageFont
from rams_lite_loader import load_rams_lite_hydrometeors, load_rams_lite_winds
from scipy import ndimage

import skyvista as sv

# Gap between the two panels (px).
PANEL_DIVIDER_WIDTH = 6
# Font sizes of the panel titles and of the smaller labels (px).
TITLE_FONT_SIZE = 34
LABEL_FONT_SIZE = 26
# Arrow colormap over a thinned cloud: diverging, white at w = 0 so weak
# vertical motion still stands out against the blue sky and the dark ground.
THINNED_ARROW_COLORMAP = "RdBu_r"
# Arrow colormap over a cut face: the face is bright white, so the centre is
# mid-grey instead (blue -> grey -> red, the end colours of RdBu). Hue gives
# the sign of w and saturation its strength.
CUTAWAY_ARROW_COLORMAP = "skyvista_blue_grey_red"
CUTAWAY_ARROW_COLORMAP_COLORS = ("#2166ac", "#808080", "#b2182b")
# Horizontal colorbar drawn in the lower right of the arrow panel (px).
COLORBAR_WIDTH, COLORBAR_HEIGHT = 420, 26
COLORBAR_MARGIN_RIGHT, COLORBAR_MARGIN_BOTTOM = 40, 110
# RAMS hydrometeor mixing ratios that count as "cloud" for the clutter filter.
HYDROMETEOR_MIXING_RATIOS = ("RCP", "RDP", "RRP", "RPP", "RSP", "RAP", "RGP", "RHP")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_path", type=Path, help="RAMS a-L/a-A ...-g1.h5 file.")
    parser.add_argument("output_dir", type=Path, help="Directory for both bundles.")
    parser.add_argument(
        "--blender",
        default=None,
        help="Blender executable (default: $SKYVISTA_BLENDER, then PATH).",
    )
    parser.add_argument(
        "--slice-y",
        type=float,
        default=None,
        help="y (m) of the vertical arrow plane (default: through the strongest updraft).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=16,
        help="Keep every n-th grid point along x and z for arrows.",
    )
    parser.add_argument(
        "--arrow-length-per-mps",
        type=float,
        default=110.0,
        help="Arrow length (m) per m/s of wind speed.",
    )
    parser.add_argument(
        "--arrow-width",
        type=float,
        default=90.0,
        help="Arrow shaft diameter (m); the head scales with it.",
    )
    parser.add_argument(
        "--arrow-self-illumination",
        type=float,
        default=1.0,
        help=(
            "Share of each arrow's on-screen color that it emits itself"
            " (0 = lit only by sun and sky); lighting adds shading on top."
        ),
    )
    parser.add_argument(
        "--w-limit",
        type=float,
        default=6.0,
        help="Vertical velocity (m/s) at the ends of the arrow colormap.",
    )
    parser.add_argument(
        "--cloud-thinning",
        type=float,
        default=0.03,
        help="Factor on cloud extinction in the arrow panel, so arrows inside show.",
    )
    parser.add_argument(
        "--cutaway",
        action="store_true",
        help="Cut the cloud open along the arrow plane instead of thinning it.",
    )
    parser.add_argument(
        "--arrow-offset",
        type=float,
        default=300.0,
        help="With --cutaway, draw the arrows this far (m) toward the camera (-y).",
    )
    parser.add_argument(
        "--weak-w",
        type=float,
        default=2.0,
        help="|w| (m/s) below which an arrow far from cloud is dropped.",
    )
    parser.add_argument(
        "--near-cloud-distance",
        type=float,
        default=1000.0,
        help="Arrows within this distance (m) of cloud are always kept.",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.35,
        help=(
            "Factor on the focal length of both panels. The camera stays where"
            " the automatic framing puts it, so this narrows the view onto the"
            " storm without changing the perspective."
        ),
    )
    parser.add_argument(
        "--arrow-panel-zoom",
        type=float,
        default=1.2,
        help=(
            "Extra focal-length factor for the arrow panel only. Above 1, the"
            " arrow panel is also re-aimed at the middle of the drawn arrows,"
            " so it zooms onto them; the two panels then differ in framing."
        ),
    )
    parser.add_argument(
        "--cloud-mixing-ratio",
        type=float,
        default=1.0e-5,
        help="Total hydrometeor mixing ratio (kg/kg) above which a cell is cloud.",
    )
    add_render_arguments(parser)
    return parser.parse_args()


def strongest_updraft_y(winds) -> float:
    """y (m) of the grid column holding the domain's maximum vertical velocity."""
    flat_index = int(np.nanargmax(winds["W"].values))
    _, y_index, _ = np.unravel_index(flat_index, winds["W"].shape)
    return float(winds["y"].values[y_index])


def distance_to_cloud_on_plane(
    clouds, plane_y_index: int, cloud_mixing_ratio: float, max_distance: float
) -> np.ndarray:
    """
    3-D distance (m) from each cell of one y row to the nearest cloudy cell.

    "Cloudy" means total hydrometeor mixing ratio above
    ``cloud_mixing_ratio``. The Euclidean distance transform runs on a band of
    rows around the plane, wide enough to contain any cloud within
    ``max_distance``, instead of the whole domain: every distance up to
    ``max_distance`` is then exact, and larger ones only need to be "larger",
    which is all the filter asks. Returns a (z, x) array; inf when the band
    holds no cloud at all.
    """
    # Cloud mask from the sum of every hydrometeor mixing ratio present.
    total_mixing_ratio = sum(
        clouds[name] for name in HYDROMETEOR_MIXING_RATIOS if name in clouds
    )
    # Grid spacings (m) along z, y, x for an anisotropic distance.
    spacing_zyx = [
        float(np.mean(np.diff(clouds[axis].values))) for axis in ("z", "y", "x")
    ]
    # Rows within max_distance (+1 for rounding) of the plane.
    band_half_width = int(np.ceil(max_distance / spacing_zyx[1])) + 1
    band_start = max(plane_y_index - band_half_width, 0)
    band_stop = min(plane_y_index + band_half_width + 1, clouds.sizes["y"])
    cloudy_band = (
        total_mixing_ratio.isel(y=slice(band_start, band_stop))
        .transpose("z", "y", "x")
        .values
        > cloud_mixing_ratio
    )
    plane_row_in_band = plane_y_index - band_start
    if not cloudy_band.any():
        return np.full((clouds.sizes["z"], clouds.sizes["x"]), np.inf)
    # Distance from every clear cell to the nearest cloudy one (0 inside cloud).
    distance_band = ndimage.distance_transform_edt(~cloudy_band, sampling=spacing_zyx)
    return distance_band[:, plane_row_in_band, :]


def pinned_view(bundle_dir: Path):
    """Camera location, aim point and scene origin shift from a built bundle."""
    manifest = json.loads((bundle_dir / "scene.json").read_text())
    (keyframe,) = manifest["camera"]["keyframes"]
    return (
        tuple(keyframe["location"]),
        tuple(keyframe["look_at"]),
        tuple(manifest["transform"]["origin_shift"]),
    )


def arrow_colormap_name(cutaway: bool) -> str:
    """
    Name of the arrow colormap, registering the cut-away one on first use.

    The exporter bakes arrow colors and draws its colorbar PNG by colormap
    name, so the custom map must be registered with matplotlib by name.
    """
    from matplotlib import colormaps
    from matplotlib.colors import LinearSegmentedColormap

    if not cutaway:
        return THINNED_ARROW_COLORMAP
    if CUTAWAY_ARROW_COLORMAP not in colormaps:
        colormaps.register(
            LinearSegmentedColormap.from_list(
                CUTAWAY_ARROW_COLORMAP, CUTAWAY_ARROW_COLORMAP_COLORS
            )
        )
    return CUTAWAY_ARROW_COLORMAP


def draw_horizontal_colorbar(
    canvas, draw, left, top, w_limit, colormap_name, font, text_style
):
    """
    Draw the arrow colormap as a horizontal bar with ticks and a title.

    The bar samples the same matplotlib colormap and limits the exporter baked
    the arrow colors with, so a color on an arrow reads off the bar directly.
    """
    from matplotlib import colormaps

    # Bar pixels: the colormap sampled across the bar's width.
    fractions = np.linspace(0.0, 1.0, COLORBAR_WIDTH)
    bar_colors = colormaps[colormap_name](fractions)[:, :3]
    bar_pixels = np.repeat(bar_colors[None, :, :], COLORBAR_HEIGHT, axis=0)
    canvas.paste(Image.fromarray((bar_pixels * 255).astype(np.uint8)), (left, top))
    draw.rectangle(
        [left - 1, top - 1, left + COLORBAR_WIDTH, top + COLORBAR_HEIGHT],
        outline="black",
        width=2,
    )
    # Five evenly spaced ticks from -w_limit to +w_limit; ends are clipped.
    for tick_value in np.linspace(-w_limit, w_limit, 5):
        tick_x = left + (tick_value + w_limit) / (2 * w_limit) * (COLORBAR_WIDTH - 1)
        draw.line(
            [tick_x, top + COLORBAR_HEIGHT, tick_x, top + COLORBAR_HEIGHT + 8],
            fill="black",
            width=2,
        )
        draw.text(
            (tick_x, top + COLORBAR_HEIGHT + 10),
            f"{tick_value:g}",
            font=font,
            anchor="ma",
            **text_style,
        )
    draw.text(
        (left + COLORBAR_WIDTH / 2, top - 8),
        "Vertical velocity w (m/s)",
        font=font,
        anchor="md",
        **text_style,
    )


def compose_side_by_side(
    left_png: Path,
    right_png: Path,
    labels: dict,
    w_limit: float,
    colormap_name: str,
    output_png: Path,
) -> None:
    """
    Place the two renders side by side and add titles, labels and a colorbar.

    ``labels`` has "left_title", "right_title", "time" and "right_note"
    (drawn in the lower left of the right panel). The arrow colorbar goes in
    the lower right of the right panel.
    """
    left_panel = Image.open(left_png).convert("RGB")
    right_panel = Image.open(right_png).convert("RGB")
    panel_width, panel_height = left_panel.size
    canvas = Image.new(
        "RGB", (2 * panel_width + PANEL_DIVIDER_WIDTH, panel_height), "white"
    )
    canvas.paste(left_panel, (0, 0))
    right_panel_left = panel_width + PANEL_DIVIDER_WIDTH
    canvas.paste(right_panel, (right_panel_left, 0))

    # Outlined white text stays readable on sky, cloud and ground alike.
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default(size=TITLE_FONT_SIZE)
    label_font = ImageFont.load_default(size=LABEL_FONT_SIZE)
    text_style = {"fill": "white", "stroke_width": 3, "stroke_fill": "black"}
    draw.text((24, 20), labels["left_title"], font=title_font, **text_style)
    draw.text(
        (right_panel_left + 24, 20),
        labels["right_title"],
        font=title_font,
        **text_style,
    )
    draw.text((24, 66), labels["time"], font=label_font, **text_style)
    draw.multiline_text(
        (right_panel_left + 24, panel_height - 100),
        labels["right_note"],
        font=label_font,
        spacing=8,
        **text_style,
    )
    draw_horizontal_colorbar(
        canvas,
        draw,
        canvas.width - COLORBAR_WIDTH - COLORBAR_MARGIN_RIGHT,
        panel_height - COLORBAR_MARGIN_BOTTOM,
        w_limit,
        colormap_name,
        label_font,
        text_style,
    )
    canvas.save(output_png)


def main() -> None:
    arguments = parse_arguments()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    # Hydrometeors and cell-centred winds of this output time.
    clouds = load_rams_lite_hydrometeors(
        arguments.data_path, top_height=arguments.top_height
    )
    winds = load_rams_lite_winds(arguments.data_path, top_height=arguments.top_height)
    minutes = float(clouds["time"]) / 60.0

    # ---- Left panel: the physically based cloud, automatic camera.
    realistic_dir = arguments.output_dir / "realistic"
    realistic_config = make_cloud_export_config(arguments)
    realistic_config.camera.lens_mm *= arguments.zoom
    sv.Scene().add_clouds(clouds, microphysics="rams").to_blender(
        realistic_dir,
        config=realistic_config,
        render=True,
        blender_executable=arguments.blender,
    )

    # ---- Right panel: same view, thinned cloud, arrows on a vertical plane.
    camera_location, camera_look_at, origin_shift = pinned_view(realistic_dir)
    wind_config = make_cloud_export_config(arguments, origin_shift=origin_shift)
    wind_config.camera.location = camera_location
    wind_config.camera.look_at = camera_look_at
    wind_config.camera.lens_mm *= arguments.zoom
    # The figure draws its own, larger colorbar.
    wind_config.overlay_colorbars = False

    # The arrow plane: one y row (kept as a length-1 dimension).
    slice_y = (
        arguments.slice_y
        if arguments.slice_y is not None
        else strongest_updraft_y(winds)
    )
    slice_index = int(np.argmin(np.abs(winds["y"].values - slice_y)))
    wind_plane = winds.isel(y=[slice_index])
    plane_y = float(wind_plane["y"].values[0])
    plane_y_km = plane_y / 1000.0

    # Clutter filter: keep an arrow if its vertical motion is strong or it
    # is near cloud.
    distance_to_cloud = distance_to_cloud_on_plane(
        clouds, slice_index, arguments.cloud_mixing_ratio, arguments.near_cloud_distance
    )
    strong_vertical_motion = (
        np.abs(wind_plane["W"].isel(y=0).values) >= arguments.weak_w
    )
    near_cloud = distance_to_cloud <= arguments.near_cloud_distance
    wind_plane = wind_plane.assign(
        show_arrow=(("z", "y", "x"), (strong_vertical_motion | near_cloud)[:, None, :])
    )

    # With a cutaway, draw the arrows slightly in front of the cut face.
    if arguments.cutaway:
        wind_plane = wind_plane.assign_coords(
            y=wind_plane["y"] - arguments.arrow_offset
        )

    # Optionally zoom the arrow panel onto the arrows: aim at the middle of
    # the drawn arrows' bounding box, from the same camera position.
    if arguments.arrow_panel_zoom != 1.0:
        arrow_lattice = wind_plane.isel(
            x=slice(None, None, arguments.stride), z=slice(None, None, arguments.stride)
        )
        drawn_arrows = arrow_lattice["show_arrow"].isel(y=0)
        drawn_z_values = drawn_arrows["z"].values[drawn_arrows.any("x").values]
        drawn_x_values = drawn_arrows["x"].values[drawn_arrows.any("z").values]
        wind_config.camera.look_at = (
            float(drawn_x_values.min() + drawn_x_values.max()) / 2.0,
            float(wind_plane["y"].values[0]),
            float(drawn_z_values.min() + drawn_z_values.max()) / 2.0,
        )
        wind_config.camera.lens_mm *= arguments.arrow_panel_zoom

    # Arrows: full 3-D wind, colored by w, length proportional to speed.
    arrows = sv.make_vectors(
        "wind",
        u="U",
        v="V",
        w="W",
        stride={"x": arguments.stride, "z": arguments.stride},
        color_by="w",
        cmap=arrow_colormap_name(arguments.cutaway),
        clim=(-arguments.w_limit, arguments.w_limit),
        factor=arguments.arrow_length_per_mps,
        glyph_width=arguments.arrow_width,
        mask_by="show_arrow",
    )
    # Self-lit, so an arrow's color reads on its shadowed side and against
    # the darker, shaded parts of the cloud.
    arrows.appearance.shader = "self_lit"
    arrows.appearance.shader_overrides = {
        "emission_strength": arguments.arrow_self_illumination
    }

    if arguments.cutaway:
        # Remove the camera-side half (y below the plane); physical optics.
        wind_dir = arguments.output_dir / "winds_cutaway"
        wind_scene = sv.Scene().add_clouds(
            clouds,
            microphysics="rams",
            cutaway=((0.0, plane_y, 0.0), (0.0, -1.0, 0.0)),
        )
    else:
        wind_dir = arguments.output_dir / "winds"
        wind_scene = sv.Scene().add_clouds(
            clouds, microphysics="rams", extinction_scale=arguments.cloud_thinning
        )
    wind_scene.add(wind_plane, arrows)
    wind_scene.to_blender(
        wind_dir,
        config=wind_config,
        render=True,
        blender_executable=arguments.blender,
    )

    # ---- Side by side.
    figure_kind = "cutaway" if arguments.cutaway else "thinned"
    output_png = (
        arguments.output_dir
        / f"winds_{figure_kind}_side_by_side_t{minutes:05.1f}min.png"
    )
    lattice_spacing_m = arguments.stride * float(np.diff(winds["x"].values[:2])[0])
    if arguments.cutaway:
        right_title = f"Cut open at y = {plane_y_km:.2f} km, with the 3-D wind"
        cloud_note = f"Arrows drawn {arguments.arrow_offset:g} m in front of the cut"
    else:
        right_title = f"3-D wind on the plane y = {plane_y_km:.2f} km"
        cloud_note = f"Cloud extinction x {arguments.cloud_thinning:g}"
    compose_side_by_side(
        realistic_dir / "render" / "frame_0001.png",
        wind_dir / "render" / "frame_0001.png",
        {
            "left_title": "Cloud, physically based",
            "right_title": right_title,
            "time": f"t = {minutes:.1f} min",
            "right_note": (
                f"Arrows every {lattice_spacing_m:.0f} m; "
                f"{arguments.arrow_length_per_mps:g} m of arrow per m/s\n"
                f"{cloud_note}"
            ),
        },
        arguments.w_limit,
        arrow_colormap_name(arguments.cutaway),
        output_png,
    )
    print(f"Side-by-side figure: {output_png}")


if __name__ == "__main__":
    main()
