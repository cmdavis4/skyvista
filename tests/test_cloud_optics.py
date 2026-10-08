"""
Tests for physically based cloud rendering: hydrometeor optics, the CloudSpec
preview, and the Blender (VDB) export of cloud scenes.

The optics tests check closed-form results that the gamma-distribution
integrals must reproduce. The export tests need the OpenVDB Python bindings
(conda-forge ``openvdb``) and are skipped without them.
"""

import json
import math
import warnings

import numpy as np
import pytest
import xarray as xr

import skyvista as sv
from skyvista.cloud_optics import (
    DROPLET_ASYMMETRY_PARAMETER,
    HydrometeorSpecies,
    compute_extinction,
    compute_slope_and_number,
    delta_scaled_optics,
    detect_microphysics_preset,
    extinction_efficiency,
    generic_single_moment_species,
    mie_fit_diameter_microns,
    rams_hydrometeor_species,
    resolve_cloud_species,
    standard_atmosphere_density,
)

# =============================================================================
# FIXTURES
# =============================================================================
# Mass content (kg m^-3) used for the single-cell optics checks: 1 g m^-3.
ONE_GRAM_PER_CUBIC_METER = 1.0e-3


@pytest.fixture
def rams_like_dataset():
    """
    A small RAMS-style dataset: a cloud-water bubble, a rain shaft, and ice.

    The vertical grid is stretched (spacing grows with height) so the export's
    uniform resampling is exercised. Variables follow RAMS names and units:
    mixing ratios in kg/kg, number concentrations per kg, DN0 in kg m^-3.
    """
    x_coordinates = np.arange(12) * 500.0
    y_coordinates = np.arange(10) * 500.0
    level_spacings = np.clip(100.0 * 1.15 ** np.arange(20), 100.0, 600.0)
    z_coordinates = np.cumsum(level_spacings) - 50.0
    x_grid, y_grid, z_grid = np.meshgrid(
        x_coordinates - x_coordinates.mean(),
        y_coordinates - y_coordinates.mean(),
        z_coordinates,
        indexing="ij",
    )
    radius = np.sqrt(x_grid**2 + y_grid**2)

    # Cloud water: a bubble between 1.5 and 5 km.
    cloud_water = 1.0e-3 * np.exp(-((radius / 1500.0) ** 2)) * (z_grid > 1500.0)
    cloud_water *= z_grid < 5000.0
    # Rain: a shaft below 3 km.
    rain = 0.5e-3 * np.exp(-((radius / 1000.0) ** 2)) * (z_grid < 3000.0)
    # Pristine ice: a layer around 6 km.
    pristine_ice = 0.3e-3 * np.exp(-(((z_grid - 6000.0) / 600.0) ** 2))

    dimensions = ("time", "x", "y", "z")
    fields = {
        "RCP": cloud_water,
        "CCP": np.where(cloud_water > 0, 3.0e8, 0.0),
        "RRP": rain,
        "CRP": np.where(rain > 0, 2.0e3, 0.0),
        "RPP": pristine_ice,
        "CPP": np.where(pristine_ice > 0, 1.0e5, 0.0),
        "DN0": np.broadcast_to(1.2 * np.exp(-z_coordinates / 8500.0), x_grid.shape),
    }
    return xr.Dataset(
        {
            name: (dimensions, np.array(values, dtype=np.float32)[None, ...])
            for name, values in fields.items()
        },
        coords={
            "time": [0.0],
            "x": x_coordinates,
            "y": y_coordinates,
            "z": z_coordinates,
        },
    )


# =============================================================================
# SIZE-DISTRIBUTION OPTICS
# =============================================================================
class TestExtinction:
    """Closed-form checks of the geometric-optics extinction."""

    def test_droplets_match_textbook_formula(self):
        """beta = Q_ext * 3 M / (2 rho_w D_eff) for spheres of water."""
        cloud = generic_single_moment_species()[0]
        mass_content = np.array([ONE_GRAM_PER_CUBIC_METER])
        extinction, effective_diameter = compute_extinction(mass_content, cloud)
        efficiency = extinction_efficiency(cloud, effective_diameter)
        expected = efficiency * 1.5 * mass_content / (1000.0 * effective_diameter)
        np.testing.assert_allclose(extinction, expected, rtol=1e-12)

    def test_marshall_palmer_rain_closed_form(self):
        """Exponential rain with fixed N0: lambda = (pi rho_w N0 / M)**(1/4)."""
        rain = generic_single_moment_species()[1]
        extinction, effective_diameter = compute_extinction(
            np.array([ONE_GRAM_PER_CUBIC_METER]), rain
        )
        slope = (math.pi * 1000.0 * 8.0e6 / ONE_GRAM_PER_CUBIC_METER) ** 0.25
        # Extinction = Q_ext * (pi/4) * N0 * 2 / lambda**3, Q_ext ~ 2.
        expected_extinction = extinction_efficiency(rain, effective_diameter) * (
            math.pi * 8.0e6 / (2.0 * slope**3)
        )
        np.testing.assert_allclose(extinction, expected_extinction, rtol=1e-10)
        np.testing.assert_allclose(effective_diameter, 3.0 / slope, rtol=1e-10)

    def test_number_closure_conserves_mass(self):
        """The recomputed number concentration reproduces the input mass."""
        for species in rams_hydrometeor_species():
            mass_content = np.array([1.0e-4, ONE_GRAM_PER_CUBIC_METER])
            slope, number = compute_slope_and_number(
                mass_content, species, np.array([1.0e3, 1.0e5])
            )
            # Mass = a N G(b) / lambda**b for the gamma distribution.
            reconstructed_mass = (
                species.mass_coefficient
                * number
                * species.moment_factor(species.mass_exponent)
                / slope**species.mass_exponent
            )
            np.testing.assert_allclose(reconstructed_mass, mass_content, rtol=1e-10)

    def test_diameter_clamp_keeps_mass(self):
        """An absurdly small number is clamped to the largest mean diameter."""
        rain = rams_hydrometeor_species()[2]
        slope, number = compute_slope_and_number(
            np.array([ONE_GRAM_PER_CUBIC_METER]), rain, np.array([1.0e-20])
        )
        mean_diameter = (rain.gamma_shape + 1.0) / slope
        np.testing.assert_allclose(mean_diameter, rain.mean_diameter_bounds[1])
        assert number[0] > 1.0e-20

    def test_zero_mass_gives_zero_extinction(self):
        cloud = generic_single_moment_species()[0]
        extinction, effective_diameter = compute_extinction(np.zeros(3), cloud)
        assert np.all(extinction == 0.0)
        assert np.all(effective_diameter == 0.0)

    def test_droplet_edge_correction_magnitude(self):
        """Q_ext is ~2.17 at 10 um and tends to 2 for millimetre drops."""
        cloud = generic_single_moment_species()[0]
        efficiency = extinction_efficiency(cloud, np.array([10.0e-6, 2.0e-3]))
        np.testing.assert_allclose(efficiency, [2.162, 2.005], atol=0.01)


class TestPhaseFunctions:
    """Delta scaling and the Mie-fit diameter conversion."""

    def test_delta_scaling_preserves_transport_coefficient(self):
        extinction = np.array([0.01, 0.2])
        for asymmetry in (0.75, 0.88):
            scaled_extinction, scaled_asymmetry = delta_scaled_optics(
                extinction, asymmetry
            )
            np.testing.assert_allclose(
                scaled_extinction * (1.0 - scaled_asymmetry),
                extinction * (1.0 - asymmetry),
            )

    def test_droplets_default_to_mie_theory_henyey_greenstein(self):
        for preset in (rams_hydrometeor_species, generic_single_moment_species):
            cloud = preset()[0]
            assert cloud.phase_function == "henyey_greenstein"
            assert cloud.asymmetry_parameter == DROPLET_ASYMMETRY_PARAMETER
            assert not cloud.delta_scale_diffraction

    def test_mie_with_delta_scaling_is_rejected(self):
        with pytest.raises(ValueError, match="delta_scale_diffraction"):
            HydrometeorSpecies(
                name="cloud",
                mixing_ratio_variable="qc",
                mass_coefficient=523.6,
                mass_exponent=3.0,
                phase_function="mie",
                delta_scale_diffraction=True,
            )

    def test_mie_fit_diameter_conversion_and_clamp(self):
        """D_fit = D_eff * exp(-2 * 0.25**2), clamped to [0.5, 50] um."""
        fit_diameter = mie_fit_diameter_microns(np.array([20.0e-6, 1.0e-3, 1.0e-9]))
        np.testing.assert_allclose(
            fit_diameter, [20.0 * math.exp(-0.125), 50.0, 0.5], rtol=1e-12
        )


class TestDatasetHelpers:
    """Preset detection, species resolution, and the air-density fallback."""

    def test_detect_presets(self):
        assert detect_microphysics_preset(xr.Dataset({"RCP": ("z", [0.0])})) == "rams"
        morrison = xr.Dataset({"qc": ("z", [0.0]), "ncr": ("z", [0.0])})
        assert detect_microphysics_preset(morrison) == "cm1_morrison"
        assert detect_microphysics_preset(xr.Dataset({"qc": ("z", [0.0])})) == (
            "generic"
        )

    def test_resolve_keeps_only_present_species(self, rams_like_dataset):
        resolved = resolve_cloud_species(rams_like_dataset)
        assert [species.name for species in resolved] == [
            "cloud",
            "rain",
            "pristine_ice",
        ]

    def test_missing_number_variable_falls_back_with_warning(self, rams_like_dataset):
        """RAMS cloud has a fixed-number fallback when CCP is absent."""
        without_number = rams_like_dataset.drop_vars("CCP")
        with pytest.warns(UserWarning, match="CCP"):
            resolved = resolve_cloud_species(without_number)
        assert resolved[0].number_variable is None

    def test_standard_atmosphere_density(self):
        density = standard_atmosphere_density(np.array([0.0, 10000.0]))
        np.testing.assert_allclose(density, [1.225, 0.4135], rtol=2e-3)


# =============================================================================
# CLOUDSPEC (PYVISTA PREVIEW)
# =============================================================================
class TestCloudSpec:
    def test_add_clouds_preview_mesh(self, rams_like_dataset):
        scene = sv.Scene().add_clouds(rams_like_dataset)
        spec = scene._specs[0][1]
        assert isinstance(spec, sv.CloudSpec)
        mesh = spec.create_mesh(rams_like_dataset, 0.0)
        extinction = mesh.point_data["cloud_extinction"]
        assert extinction.max() > 0.0
        assert np.all(np.isfinite(extinction))

    def test_preview_opacity_is_beer_lambert(self, rams_like_dataset):
        spec = sv.make_clouds()
        spec.create_mesh(rams_like_dataset, 0.0)
        kwargs = spec.get_pyvista_kwargs()
        opacity = np.asarray(kwargs["opacity"])
        # Not 256 long: PyVista would copy it unscaled into an 8-bit alpha.
        assert len(opacity) != 256
        sampled_extinction = np.linspace(0.0, kwargs["clim"][1], len(opacity))
        expected = 1.0 - np.exp(-sampled_extinction * kwargs["opacity_unit_distance"])
        np.testing.assert_allclose(opacity, expected)

    def test_extinction_scale_multiplies_extinction(self, rams_like_dataset):
        physical = sv.make_clouds().create_mesh(rams_like_dataset, 0.0)
        halved = sv.make_clouds(extinction_scale=0.5).create_mesh(
            rams_like_dataset, 0.0
        )
        np.testing.assert_allclose(
            halved.point_data["cloud_extinction"],
            0.5 * physical.point_data["cloud_extinction"],
            rtol=1e-6,
        )


# =============================================================================
# BLENDER EXPORT
# =============================================================================
class TestVerticalResampling:
    def test_linear_profile_is_preserved(self):
        from skyvista.blender import interpolate_to_levels

        native_heights = np.array([0.0, 100.0, 300.0, 700.0])
        profile = xr.DataArray(
            2.0 * native_heights + 1.0,
            dims=["z"],
            coords={"z": native_heights},
        )
        target_heights = np.arange(0.0, 701.0, 50.0)
        resampled = interpolate_to_levels(profile, "z", native_heights, target_heights)
        np.testing.assert_allclose(resampled.values, 2.0 * target_heights + 1.0)

    def test_spacing_uses_cloudy_levels_and_horizontal_floor(self):
        from skyvista.blender import choose_uniform_vertical_spacing

        heights = np.array([0.0, 20.0, 60.0, 160.0, 400.0])
        cloudy = np.array([False, False, False, True, True])
        # Finest spacing touching a cloudy level is 100 m (60 -> 160).
        assert choose_uniform_vertical_spacing(heights, cloudy) == 100.0
        # A 1 km horizontal grid floors it at 250 m.
        assert choose_uniform_vertical_spacing(heights, cloudy, 1000.0) == 250.0


class TestCloudExport:
    @pytest.fixture(autouse=True)
    def _require_openvdb(self):
        pytest.importorskip("openvdb")

    def test_bundle_has_cloud_material_and_grids(self, rams_like_dataset, tmp_path):
        import openvdb

        scene = sv.Scene().add_clouds(rams_like_dataset)
        scene.to_blender(tmp_path / "bundle")
        manifest = json.loads((tmp_path / "bundle" / "scene.json").read_text())

        # Cloud scenes get the physical defaults when no config is passed.
        assert manifest["world"]["preset"] == "physical_sky"
        assert manifest["world"]["ground_albedo"] == pytest.approx(0.15)

        cloud_object = manifest["objects"][0]
        material = cloud_object["material"]
        assert material["type"] == "cloud"
        assert material["extinction_grid_units"] == "1/km"
        species_names = [species["name"] for species in material["species"]]
        assert species_names == ["cloud", "rain", "pristine_ice"]
        # Rain and ice are delta scaled: g' = (g - 1/2) / (1/2).
        anisotropy = {
            species["name"]: species["anisotropy"] for species in material["species"]
        }
        assert anisotropy["cloud"] == pytest.approx(0.86)
        assert anisotropy["rain"] == pytest.approx(0.76)
        assert anisotropy["pristine_ice"] == pytest.approx(0.5)

        # The VDB holds one extinction grid per species, on uniform levels.
        vdb_path = tmp_path / "bundle" / "data" / "clouds" / "clouds_0001.vdb"
        grids, _ = openvdb.readAll(str(vdb_path))
        grid_names = sorted(grid.name for grid in grids)
        assert grid_names == sorted(f"extinction_{name}" for name in species_names)
        vertical_spacing = cloud_object["geometry"]["vertical_spacing"]
        assert vertical_spacing >= 0.25 * 500.0

    def test_vdb_extinction_matches_optics(self, rams_like_dataset, tmp_path):
        """The stored cloud extinction (km^-1) equals the computed optics."""
        import openvdb

        spec = sv.make_clouds()
        scene = sv.Scene().add(rams_like_dataset, spec)
        scene.to_blender(tmp_path / "bundle")
        vdb_path = tmp_path / "bundle" / "data" / "clouds" / "clouds_0001.vdb"
        grid = openvdb.read(str(vdb_path), "extinction_cloud")
        _, stored_maximum = grid.evalMinMax()

        optics = spec.compute_optics(rams_like_dataset.isel(time=0), "z")
        expected_maximum = float(optics[0].render_extinction.max()) * 1000.0
        # Vertical resampling interpolates between levels, so the peak may sit
        # slightly below the native maximum, never above it.
        assert stored_maximum <= expected_maximum * (1.0 + 1e-5)
        assert stored_maximum > 0.8 * expected_maximum

    def test_z_exaggeration_warns_for_clouds(self, rams_like_dataset, tmp_path):
        config = sv.BlenderExportConfig.for_clouds()
        config.transform.z_exaggeration = 2.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sv.Scene().add_clouds(rams_like_dataset).to_blender(
                tmp_path / "bundle", config=config
            )
        assert any("z_exaggeration" in str(item.message) for item in caught)


def test_world_preset_is_validated():
    from skyvista.blender import BlenderWorldConfig

    with pytest.raises(ValueError, match="Unknown world preset"):
        BlenderWorldConfig(preset="sunset")
