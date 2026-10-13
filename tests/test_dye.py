"""
Tests for passive-tracer dye (DyeSpec): the dyed fraction and label, the
displacement filter, the PyVista preview, and the Blender (VDB) export.

The export tests need OpenVDB (conda-forge ``openvdb``) and are skipped
without it.
"""

import json

import numpy as np
import pytest
import xarray as xr

import skyvista as sv

# Three layered tracers: 0-1, 1-2 and 2-3 km, labelled by mid-layer height.
LAYER_MID_HEIGHTS = [500.0, 1500.0, 2500.0]
TRACER_NAMES = ["TRACER1", "TRACER2", "TRACER3"]
# Tracer value of undiluted air from one layer.
REFERENCE_VALUE = 10.0
# Plume: a column of air lifted from the lowest layer into the middle one.
PLUME_LIFTED_SHARE = 0.8


@pytest.fixture
def layered_tracer_dataset():
    """
    Three tracer layers, plus a lifted plume in the middle of the domain.

    Outside the plume each tracer is undiluted (value 10) inside its own layer
    and 0 elsewhere. In the plume column, from 1.5 to 2.5 km, 80% of the air
    came from the lowest layer and 20% is local, from the layer it sits in.
    """
    x = np.arange(0.0, 2001.0, 100.0)
    y = np.arange(0.0, 2001.0, 100.0)
    z = np.arange(50.0, 3000.0, 100.0)
    heights = z[:, None, None] * np.ones((1, len(y), len(x)))
    tracers = {}
    for tracer_name, (layer_bottom, layer_top) in zip(
        TRACER_NAMES, [(0, 1000), (1000, 2000), (2000, 3000)], strict=True
    ):
        in_layer = (heights >= layer_bottom) & (heights < layer_top)
        tracers[tracer_name] = np.where(in_layer, REFERENCE_VALUE, 0.0)

    # The plume column: 5 x 5 points around the center, 1.5-2.5 km.
    in_plume = np.zeros_like(heights, dtype=bool)
    in_plume[(z >= 1500) & (z <= 2500), 8:13, 8:13] = True
    for tracer_name in TRACER_NAMES:
        tracers[tracer_name] = np.where(
            in_plume,
            tracers[tracer_name] * (1.0 - PLUME_LIFTED_SHARE),
            tracers[tracer_name],
        )
    tracers["TRACER1"] = np.where(
        in_plume, PLUME_LIFTED_SHARE * REFERENCE_VALUE, tracers["TRACER1"]
    )
    return xr.Dataset(
        {name: (("z", "y", "x"), values) for name, values in tracers.items()},
        coords={"x": x, "y": y, "z": z},
    )


def make_layer_dye(**kwargs):
    """A dye of the three layers, labelled by starting height."""
    return sv.make_dye(
        TRACER_NAMES,
        tracer_values=LAYER_MID_HEIGHTS,
        reference_value=REFERENCE_VALUE,
        **kwargs,
    )


class TestDyeFields:
    def test_fraction_and_label_without_displacement_filter(
        self, layered_tracer_dataset
    ):
        """Every cell is fully dyed; the plume's label is the mixed height."""
        fraction, label = make_layer_dye().dye_fields(layered_tracer_dataset)
        np.testing.assert_allclose(fraction.values, 1.0)
        # Plume at 1.95 km: 80% from 500 m, 20% from the local 1.5 km layer.
        plume_label = float(label.sel(x=1000.0, y=1000.0, z=1950.0))
        assert plume_label == pytest.approx(0.8 * 500.0 + 0.2 * 1500.0)

    def test_displacement_filter_keeps_only_moved_air(self, layered_tracer_dataset):
        """Undisturbed layers vanish; the lifted share of the plume remains."""
        spec = make_layer_dye(min_vertical_displacement=600.0)
        fraction, label = spec.dye_fields(layered_tracer_dataset)
        # The environment has not moved: no dye.
        assert float(fraction.sel(x=0.0, y=0.0).max()) == 0.0
        # At 1.95 km the local air (from 1.5 km) moved < 600 m and is hidden.
        plume_point = {"x": 1000.0, "y": 1000.0, "z": 1950.0}
        assert float(fraction.sel(plume_point)) == pytest.approx(PLUME_LIFTED_SHARE)
        assert float(label.sel(plume_point)) == pytest.approx(500.0)
        # No dye means no label.
        assert np.isnan(float(label.sel(x=0.0, y=0.0, z=1950.0)))

    def test_default_color_limits_span_tracer_values(self):
        spec = make_layer_dye()
        assert spec.appearance.clim == (500.0, 2500.0)

    def test_mismatched_tracer_values_are_rejected(self):
        with pytest.raises(ValueError, match="one value per tracer"):
            sv.make_dye(TRACER_NAMES, tracer_values=[1.0, 2.0])

    def test_displacement_filter_needs_heights(self):
        with pytest.raises(ValueError, match="tracer_values"):
            sv.make_dye(TRACER_NAMES, min_vertical_displacement=500.0)

    def test_preview_is_a_labelled_isosurface(self, layered_tracer_dataset):
        spec = make_layer_dye(min_vertical_displacement=600.0)
        surface = spec.create_mesh(layered_tracer_dataset, None)
        assert surface.n_points > 0
        assert surface.active_scalars_name == "dye_label"
        # Only the plume is dyed, so the surface hugs the plume column.
        x_min, x_max = surface.bounds[0], surface.bounds[1]
        assert 700.0 <= x_min and x_max <= 1300.0


class TestDyeExport:
    @pytest.fixture(autouse=True)
    def _require_openvdb(self):
        pytest.importorskip("openvdb")

    def test_bundle_has_dye_material_and_cropped_grids(
        self, layered_tracer_dataset, tmp_path
    ):
        import openvdb

        spec = make_layer_dye(
            min_vertical_displacement=600.0,
            cmap="viridis",
            extinction_per_km=3.0,
            self_illumination=0.4,
        )
        sv.Scene().add(layered_tracer_dataset, spec).to_blender(tmp_path / "bundle")
        manifest = json.loads((tmp_path / "bundle" / "scene.json").read_text())
        dye_object = manifest["objects"][0]

        # Material: extinction per meter, colormap over the starting heights.
        material = dye_object["material"]
        assert material["type"] == "dye"
        assert material["extinction_at_full_strength"] == pytest.approx(3.0e-3)
        assert material["self_illumination"] == pytest.approx(0.4)
        assert material["coloring"]["clim"] == [500.0, 2500.0]
        assert (tmp_path / "bundle" / material["coloring"]["colormap_lut"]).exists()

        # The VDB holds both fields, cropped to the plume.
        vdb_path = tmp_path / "bundle" / "data" / "dye" / "dye_0001.vdb"
        grids, _ = openvdb.readAll(str(vdb_path))
        grids_by_name = {grid.name: grid for grid in grids}
        assert sorted(grids_by_name) == ["dye_fraction", "dye_label"]
        _, stored_maximum = grids_by_name["dye_fraction"].evalMinMax()
        assert stored_maximum == pytest.approx(PLUME_LIFTED_SHARE, rel=1e-5)
        # The lifted air came from 500 m, so the stored label is 500 m. Every
        # voxel equals the background then, so read values, not active voxels.
        label_grid = grids_by_name["dye_label"]
        first_index, last_index = grids_by_name[
            "dye_fraction"
        ].evalActiveVoxelBoundingBox()
        dyed_box_shape = tuple(
            last - first + 1
            for first, last in zip(first_index, last_index, strict=True)
        )
        label_values = np.zeros(dyed_box_shape, dtype=np.float32)
        label_grid.copyToArray(label_values, ijk=first_index)
        np.testing.assert_allclose(label_values, 500.0, rtol=1e-5)
        # Scene bounds cover only the plume (plus a one-cell margin).
        # Bounds are [[x, y, z minimum], [x, y, z maximum]].
        x_min, x_max = manifest["bounds"][0][0], manifest["bounds"][1][0]
        assert x_min >= 600.0 and x_max <= 1400.0
