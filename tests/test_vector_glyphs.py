"""
Tests for vector glyphs: lattice striding, glyph coloring, and the Blender
export that writes per-frame point files for geometry-nodes instancing.

Blender itself is not needed: the export side writes plain ``.npz`` files and a
manifest, which these tests read back. (Arrow placement inside Blender was
checked by evaluating the instances of a built scene.)
"""

import json

import numpy as np
import pytest
import xarray as xr

import skyvista as sv
from skyvista.varspec import VectorSpec


@pytest.fixture
def wind_dataset():
    """A 9 x 7 x 5 wind field with known components at every point."""
    x = np.arange(9) * 100.0
    y = np.arange(7) * 100.0
    z = np.arange(5) * 100.0
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing="ij")
    return xr.Dataset(
        {
            # u = x / 100, v = -y / 100, w = 2: easy to check per point.
            "U": (("x", "y", "z"), x_grid / 100.0),
            "V": (("x", "y", "z"), -y_grid / 100.0),
            "W": (("x", "y", "z"), np.full_like(z_grid, 2.0)),
            "T": (("x", "y", "z"), 300.0 + z_grid / 100.0),
        },
        coords={"x": x, "y": y, "z": z},
    )


def make_wind_spec(**geometry_kwargs) -> VectorSpec:
    """VectorSpec for the fixture's U/V/W with the given geometry options."""
    return sv.make_vectors("wind", u="U", v="V", w="W", **geometry_kwargs)


def test_integer_stride_thins_every_axis(wind_dataset):
    spec = make_wind_spec(stride=4)
    thinned = spec.subsample(wind_dataset)
    # Indices 0, 4, 8 on x; 0, 4 on y; 0, 4 on z.
    assert dict(thinned.sizes) == {"x": 3, "y": 2, "z": 2}
    np.testing.assert_array_equal(thinned.x.values, [0.0, 400.0, 800.0])


def test_per_axis_stride_leaves_unlisted_axes_whole(wind_dataset):
    spec = make_wind_spec(stride={"x": 2})
    thinned = spec.subsample(wind_dataset)
    assert dict(thinned.sizes) == {"x": 5, "y": 7, "z": 5}


def test_color_by_derived_and_dataset_scalars(wind_dataset):
    spec_by_speed = make_wind_spec(color_by="speed")
    components = spec_by_speed.vector_components(wind_dataset)
    expected_speed = np.sqrt(sum(component**2 for component in components))
    np.testing.assert_allclose(
        spec_by_speed.color_values(wind_dataset, components), expected_speed
    )
    spec_by_w = make_wind_spec(color_by="w")
    np.testing.assert_allclose(spec_by_w.color_values(wind_dataset, components), 2.0)
    spec_by_temperature = make_wind_spec(color_by="T")
    np.testing.assert_allclose(
        spec_by_temperature.color_values(wind_dataset, components),
        wind_dataset["T"].values.ravel(order="F"),
    )


def test_auto_factor_sizes_longest_arrow_to_half_the_lattice_spacing(wind_dataset):
    spec = make_wind_spec(stride=4)
    thinned = spec.subsample(wind_dataset)
    components = spec.vector_components(thinned)
    factor = spec.auto_factor(thinned, components)
    longest = np.max(np.sqrt(sum(component**2 for component in components)))
    # Strided lattice spacing is 400 m on every axis.
    assert factor * longest == pytest.approx(200.0)


def test_pyvista_glyphs_follow_the_stride(wind_dataset):
    full = make_wind_spec().create_mesh(wind_dataset, None)
    thinned = make_wind_spec(stride=4).create_mesh(wind_dataset, None)
    assert 0 < thinned.n_points < full.n_points


def test_blender_export_writes_instancing_point_files(wind_dataset, tmp_path):
    scene = sv.Scene().add_vectors(
        wind_dataset,
        "wind",
        u="U",
        v="V",
        w="W",
        stride=4,
        color_by="w",
        factor=25.0,
        cmap="RdBu_r",
        clim=(-4.0, 4.0),
    )
    bundle_dir = tmp_path / "bundle"
    scene.to_blender(bundle_dir)
    manifest = json.loads((bundle_dir / "scene.json").read_text())
    (vector_entry,) = manifest["objects"]
    geometry = vector_entry["geometry"]
    assert geometry["carrier"] == "point_glyphs"
    assert geometry["length_per_unit"] == 25.0
    assert vector_entry["material"]["coloring"]["mode"] == "instance_color"
    assert vector_entry["material"]["coloring"]["clim"] == [-4.0, 4.0]

    point_path = bundle_dir / geometry["path_pattern"].replace("####", "0001")
    with np.load(point_path) as point_file:
        positions = point_file["positions"]
        vectors = point_file["vectors"]
        colors = point_file["colors"]
    # 3 x 2 x 2 strided lattice.
    assert positions.shape == vectors.shape == colors.shape == (12, 3)
    # Each glyph's vector is the model value at its own position.
    np.testing.assert_allclose(vectors[:, 0], positions[:, 0] / 100.0)
    np.testing.assert_allclose(vectors[:, 1], -positions[:, 1] / 100.0)
    np.testing.assert_allclose(vectors[:, 2], 2.0)
    # w = 2 with clim (-4, 4) sits 3/4 along the colormap, for every glyph.
    assert np.allclose(colors, colors[0])
    # A colorbar is produced, as for colormapped meshes.
    assert manifest["annotations"]["colorbars"][0]["for"] == vector_entry["name"]


def test_blender_export_drops_glyphs_without_a_vector(wind_dataset, tmp_path):
    masked = wind_dataset.copy(deep=True)
    masked["U"][0, 0, 0] = np.nan
    scene = sv.Scene().add_vectors(masked, "wind", u="U", v="V", w="W")
    scene.to_blender(tmp_path / "bundle")
    with np.load(
        tmp_path / "bundle" / "data" / "vector_wind" / "points_0001.npz"
    ) as point_file:
        assert len(point_file["positions"]) == 9 * 7 * 5 - 1


def test_blender_instancing_off_falls_back_to_alembic(wind_dataset, tmp_path):
    pytest.importorskip("alembic3d")
    scene = sv.Scene().add_vectors(
        wind_dataset, "wind", u="U", v="V", w="W", stride=4, blender_instancing=False
    )
    scene.to_blender(tmp_path / "bundle")
    manifest = json.loads((tmp_path / "bundle" / "scene.json").read_text())
    assert manifest["objects"][0]["geometry"]["carrier"] == "alembic"


def test_explicit_camera_view_is_used_verbatim(wind_dataset, tmp_path):
    config = sv.BlenderExportConfig()
    config.camera.location = (5000.0, -6000.0, 1500.0)
    config.camera.look_at = (400.0, 300.0, 200.0)
    sv.Scene().add_vectors(wind_dataset, "wind", u="U", v="V", w="W").to_blender(
        tmp_path / "bundle", config=config
    )
    manifest = json.loads((tmp_path / "bundle" / "scene.json").read_text())
    (keyframe,) = manifest["camera"]["keyframes"]
    assert keyframe["location"] == [5000.0, -6000.0, 1500.0]
    assert keyframe["look_at"] == [400.0, 300.0, 200.0]


def test_constant_width_arrows_record_fixed_part_sizes(wind_dataset, tmp_path):
    scene = sv.Scene().add_vectors(
        wind_dataset, "wind", u="U", v="V", w="W", glyph_width=40.0
    )
    scene.to_blender(tmp_path / "bundle")
    manifest = json.loads((tmp_path / "bundle" / "scene.json").read_text())
    geometry = manifest["objects"][0]["geometry"]
    assert geometry["width_mode"] == "constant"
    assert geometry["shaft_radius"] == pytest.approx(20.0)
    # The head is wider than the shaft and has a fixed length.
    assert geometry["head_radius"] > geometry["shaft_radius"]
    assert geometry["head_length"] > 0


def test_mask_by_keeps_only_selected_glyphs(wind_dataset, tmp_path):
    masked_dataset = wind_dataset.assign(keep=wind_dataset["x"] >= 400.0)
    # Broadcast the 1-D mask to the full grid, as a real mask would be.
    masked_dataset["keep"] = masked_dataset["keep"].broadcast_like(masked_dataset["U"])
    scene = sv.Scene().add_vectors(
        masked_dataset, "wind", u="U", v="V", w="W", mask_by="keep"
    )
    scene.to_blender(tmp_path / "bundle")
    with np.load(
        tmp_path / "bundle" / "data" / "vector_wind" / "points_0001.npz"
    ) as point_file:
        positions = point_file["positions"]
    # x = 400 ... 800 m: 5 of the 9 columns.
    assert len(positions) == 5 * 7 * 5
    assert positions[:, 0].min() == pytest.approx(400.0)
    # The PyVista path draws glyphs only at the kept points too.
    spec = make_wind_spec(mask_by="keep")
    unmasked_points = make_wind_spec().create_mesh(masked_dataset, None).n_points
    assert 0 < spec.create_mesh(masked_dataset, None).n_points < unmasked_points


def test_self_lit_preset_is_exposure_relative():
    from skyvista.shaders import resolve_shader

    shader = resolve_shader("self_lit")
    assert shader["emission_from"] == "color"
    assert shader["emission_relative_to_exposure"] is True
    # The default look stays a plain, unlit-by-itself matte surface.
    assert resolve_shader(None)["emission_relative_to_exposure"] is False


def test_shader_overrides_replace_preset_fields():
    appearance = sv.VectorAppearance(
        shader="self_lit", shader_overrides={"emission_strength": 1.0}
    )
    assert appearance.to_blender_material()["shader"]["emission_strength"] == 1.0
    with pytest.raises(ValueError, match="Unknown shader field"):
        sv.VectorAppearance(shader_overrides={"glowiness": 2}).to_blender_material()
