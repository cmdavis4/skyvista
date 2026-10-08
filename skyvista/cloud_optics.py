"""
Visible-wavelength optical properties of simulated hydrometeors.

This module turns the bulk microphysics output of a cloud model -- mass mixing
ratios, and number concentrations when the scheme predicts them -- into the
quantities a physically based volume renderer needs: an *extinction
coefficient* (how much light is removed per meter) and a *phase function* (in
which directions the removed light is re-emitted) for every hydrometeor
species. The Blender backend writes these fields to VDB grids and builds one
Cycles volume shader from them (see ``build_scene.make_cloud_material``).

Physics in brief
----------------
Every hydrometeor is much larger than visible wavelengths (a 10 um droplet
already has a size parameter ``pi * D / lambda`` of about 60), so the
geometric-optics limit applies:

* **Extinction** is twice the particles' projected area per unit volume. The
  factor of two is the "extinction paradox": a large particle removes the light
  that hits it *and* diffracts an equal amount around its edge. For a
  population with size distribution ``n(D)``::

      beta_ext = Q_ext * integral( A(D) n(D) dD ),   Q_ext ~= 2

  For spheres this reduces to the textbook ``beta = 3 LWC / (2 rho_w r_e)``.
* **Absorption** is negligible: the imaginary refractive index of water and ice
  is ~1e-9 across 400-700 nm, so the single-scattering albedo is 1 to better
  than 1e-4 even for millimetre raindrops. Clouds are pure, *spectrally
  neutral* scatterers. Their colour comes only from the illumination.
* **Phase functions** are strongly forward peaked. Every species uses a
  Henyey-Greenstein lobe whose asymmetry parameter ``g`` comes from Mie
  theory or from measured ice optics (values below). Large particles are
  delta scaled (see below). Cloud droplets may instead use Cycles' native
  "Mie" phase function (opt-in; see "Validation in Cycles").

Size distributions
------------------
Each species follows the bulk-scheme convention of a gamma size distribution
with power-law mass and projected area::

    n(D) = N0 * D**mu * exp(-lambda * D)
    m(D) = a * D**b                  (mass, kg; D is the maximum dimension, m)
    A(D) = gamma * D**sigma          (orientation-averaged projected area, m^2)

Its k-th moment is ``N * G(k) / lambda**k``, with
``G(k) = Gamma(mu + k + 1) / Gamma(mu + 1)``. Given the mass content
``M = rho_air * q`` and one more constraint, the slope ``lambda`` follows in
closed form:

* a predicted number concentration (two-moment schemes) or a fixed one
  (single-moment cloud water): ``lambda = (a N G(b) / M) ** (1 / b)``;
* a fixed intercept ``N0`` (single-moment precipitation, Marshall-Palmer
  style): ``lambda = (a N0 Gamma(mu + b + 1) / M) ** (1 / (mu + b + 1))``.

Using each scheme's own ``a, b, mu`` keeps the optics consistent with the
particle sizes the model itself assumed when it computed fall speeds and
collection rates.

Delta scaling of large particles
--------------------------------
For particles much larger than the wavelength, the diffracted half of the
extinction goes into a lobe only ``lambda / D`` radians wide: about 0.03 deg
for a 1 mm raindrop, narrower than the 0.5 deg solar disc. That light is
indistinguishable from light that was never scattered. Rendering it as a
scattering event wastes a path vertex and smears a needle-thin peak into a
broad Henyey-Greenstein lobe. The standard radiative-transfer remedy is delta
scaling (Joseph et al. 1976): remove the forward-peak fraction ``f`` from the
extinction and from the asymmetry parameter::

    beta' = beta * (1 - f)        g' = (g - f) / (1 - f)        f = 1/2

The transport coefficient ``beta * (1 - g)``, which governs the diffuse light
field, is unchanged. Rain shafts and snow curtains then transmit the
background with their *visual* (half) optical depth, as they do in reality.

Asymmetry parameters (total, including diffraction)
--------------------------------------------------
* cloud droplets 0.86: Mie theory for water spheres, averaged over a gamma
  distribution (mu = 2) and 450-650 nm. It is 0.849 at an effective diameter
  of 10 um, 0.858 at 20 um, and 0.870 at 40 um. Droplets are not delta
  scaled, because their diffraction lobe (``lambda / D`` ~ 1.6 deg at 20 um)
  is wider than the solar disc and forms the visible aureole.
* drizzle 0.88, rain 0.88: Mie theory for water spheres, averaged over a gamma
  distribution and 450-650 nm (D_eff 100 um: 0.879; 0.5-2 mm: 0.883-0.884).
* pristine ice, snow, aggregates, graupel 0.75: the severely roughened
  aggregate habit (Yang et al. 2013), whose g = 0.75 matches the in-situ and
  satellite constraints used for MODIS Collection 6 (Baum et al. 2014; Cole
  et al. 2014). Rimed graupel surfaces are rough, so graupel shares it; this is
  an assumption, not a fit.
* hail 0.80: assumed. It lies between a clear ice sphere (0.89 from Mie
  theory) and roughened ice (0.75), since hailstones are often bubbly or lobed.

Validation in Cycles (Blender 5.0.1)
------------------------------------
The pipeline was checked against a discrete-ordinates solver (DISORT) and an
independent Monte Carlo code for a homogeneous slab of optical depth 10,
lit by a sun at 60 deg elevation and viewed at nadir:

* isotropic scattering: Cycles is within 1.5% of DISORT, which confirms the
  extinction units and the object-space density convention.
* Henyey-Greenstein, g = 0.72-0.86: Cycles reflects 9-11% too little and
  transmits correspondingly more. Energy is conserved (reflectance plus
  transmittance is 1.000), so this is a redistribution inside Cycles'
  volume integrator, not lost light. Delta scaling does not change it.
* Cycles' "Mie" phase function (droplets, d = 15 um): ~20% too little
  reflection against a Monte Carlo solution with the same phase function.
  The Jendersie & d'Eon fit is also more forward peaked than full Mie theory
  (g 0.864 vs 0.849 at 10 um; 0.879 vs 0.858 at 20 um).

Henyey-Greenstein with the Mie-theory ``g`` is therefore the default for
droplets: it gives the more accurate cloud brightness in Cycles. The "mie"
phase function remains available for close-ups of sunlit cloud edges, where
its narrow forward peak reproduces the silver lining more faithfully.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

# =============================================================================
# Physical constants
# =============================================================================
# Bulk density of liquid water and of solid (bubble-free) ice, kg m^-3.
WATER_DENSITY = 1000.0
SOLID_ICE_DENSITY = 917.0

# Projected-area coefficients for a sphere: A = (pi / 4) * D**2.
SPHERE_AREA_COEFFICIENT = math.pi / 4.0
SPHERE_AREA_EXPONENT = 2.0

# Mitchell (1996) area-dimensional power law for aggregates of side planes,
# columns and bullets, A = 0.2285 D**1.88 in CGS units (cm^2, cm). Converted to
# SI (m^2, m) exactly as the P3 scheme does: 0.2285 * 100**1.88 / 100**2. P3
# and Morrison & Grabowski (2008) use it for all unrimed non-spherical ice.
NONSPHERICAL_ICE_AREA_COEFFICIENT = 0.2285 * 100.0**1.88 / 100.0**2
NONSPHERICAL_ICE_AREA_EXPONENT = 1.88

# Asymmetry parameter of cloud droplets from Mie theory (gamma mu = 2
# populations, 450-650 nm): 0.849 at D_eff = 10 um, 0.858 at 20 um, 0.870 at
# 40 um. 0.86 represents typical cloud droplets to within 0.01.
DROPLET_ASYMMETRY_PARAMETER = 0.86

# Geometric-optics limit of the extinction efficiency (the extinction paradox).
LARGE_PARTICLE_EXTINCTION_EFFICIENCY = 2.0

# Edge-effect correction for droplets: Q_ext = 2 + c * (D_eff / 1 um)**(-2/3).
# The -2/3 power is the leading asymptotic correction from Mie theory
# (Nussenzveig & Wiscombe 1980). c = 0.75 is a fit to Mie calculations for
# gamma(mu = 2) droplet populations averaged over 450-650 nm; it reproduces
# Q_ext to ~1% from D_eff = 10 um (Q_ext = 2.17) to 2 mm (Q_ext = 2.005).
DROPLET_EXTINCTION_EDGE_COEFFICIENT = 0.75

# Fraction of a large particle's extinction carried by the diffraction peak.
# By Babinet's principle diffraction removes exactly the geometric
# cross-section, which is half of Q_ext = 2.
DIFFRACTION_FRACTION_OF_EXTINCTION = 0.5

# The Cycles "Mie" phase function (Jendersie & d'Eon 2023) is parameterised by
# the mean diameter of a log-normal droplet population with log-space standard
# deviation 0.25, fitted over 5 < d < 50 um. The effective diameter (third over
# second moment) of that population is exp(2 * 0.25**2) = 1.133 times its mean
# diameter. We convert our effective diameter to the fit's mean diameter so the
# two populations have the same effective diameter, and hence the same optics.
MIE_FIT_LOGNORMAL_SIGMA = 0.25
MIE_FIT_DIAMETER_PER_EFFECTIVE_DIAMETER = math.exp(-2.0 * MIE_FIT_LOGNORMAL_SIGMA**2)

# Clamp the fit diameter to where the fit was validated (upper end) and to a
# positive value (lower end; Cycles has separate small-droplet branches down to
# ~0 um, so values under 5 um are still meaningful).
MIE_FIT_DIAMETER_BOUNDS_MICRONS = (0.5, 50.0)

# Names a dataset might use for the air density, in priority order. "rho" is
# CM1's output name (output_rho = 1); "DN0" is RAMS's base-state density.
AIR_DENSITY_VARIABLE_CANDIDATES = (
    "rho",
    "RHO",
    "DN0",
    "dn0",
    "air_density",
    "rho_air",
    "density",
)

# Phase-function names a species may request.
PHASE_FUNCTIONS = ("mie", "henyey_greenstein")


# =============================================================================
# Species definition
# =============================================================================
@dataclass(frozen=True)
class HydrometeorSpecies:
    """
    One hydrometeor category: where its data lives and how its particles look.

    The size distribution is a gamma distribution in maximum dimension ``D``,
    ``n(D) = N0 D**mu exp(-lambda D)``. Mass and projected area follow power
    laws, ``m = a D**b`` and ``A = gamma D**sigma``. The slope ``lambda`` is
    closed by the first available of these, in priority order:

    1. ``number_variable``: a predicted number concentration in the dataset
       (number per kg of air, as two-moment schemes write it).
    2. ``fixed_number_concentration``: a constant number per m^3 of air.
    3. ``fixed_intercept``: a constant ``N0``, in m^-(4 + mu).

    Attributes:
        name: Short species name ("cloud", "rain", ...). Used to name the
            VDB grids, so it must be a valid identifier.
        mixing_ratio_variable: Dataset variable holding the mass mixing ratio,
            in kg per kg of air.
        mass_coefficient: ``a`` in ``m = a D**b`` (SI: kg, m).
        mass_exponent: ``b`` in ``m = a D**b``.
        area_coefficient: ``gamma`` in ``A = gamma D**sigma`` (SI: m^2, m).
        area_exponent: ``sigma`` in ``A = gamma D**sigma``.
        gamma_shape: ``mu`` in ``n(D) ~ D**mu exp(-lambda D)``. RAMS's ``GNU``
            is ``mu + 1``.
        number_variable: Dataset variable holding the number concentration
            per kg of air, or None.
        fixed_number_concentration: Constant number per m^3, used when
            ``number_variable`` is None or absent from the dataset.
        fixed_intercept: Constant intercept ``N0`` in m^-(4 + mu), used when
            no number concentration is available.
        phase_function: "henyey_greenstein" (default) or "mie", the Cycles
            water-droplet Mie fit (valid for effective diameters up to ~50 um;
            see "Validation in Cycles" in the module docstring).
        asymmetry_parameter: Total asymmetry parameter ``g`` (mean scattering
            cosine, including diffraction). Ignored for "mie".
        delta_scale_diffraction: Remove the diffraction peak from extinction
            and ``g`` (see the module docstring). Must be False for "mie",
            whose phase function already contains the peak.
        mean_diameter_bounds: (min, max) allowed mean diameter ``(mu + 1) /
            lambda`` in m. Inconsistent mass/number pairs (e.g. from
            interpolation or numerical noise) are clamped into this range with
            the number adjusted so the mass is preserved, as bulk schemes do.
    """

    name: str
    mixing_ratio_variable: str
    mass_coefficient: float
    mass_exponent: float
    area_coefficient: float = SPHERE_AREA_COEFFICIENT
    area_exponent: float = SPHERE_AREA_EXPONENT
    gamma_shape: float = 0.0
    number_variable: Optional[str] = None
    fixed_number_concentration: Optional[float] = None
    fixed_intercept: Optional[float] = None
    phase_function: str = "henyey_greenstein"
    asymmetry_parameter: float = 0.75
    delta_scale_diffraction: bool = True
    mean_diameter_bounds: Tuple[float, float] = (1.0e-6, 0.05)

    def __post_init__(self):
        # The name becomes part of VDB grid names and Blender node labels.
        if not self.name.isidentifier():
            raise ValueError(
                f"Species name {self.name!r} must be a valid identifier "
                "(letters, digits, underscores)."
            )
        if self.phase_function not in PHASE_FUNCTIONS:
            raise ValueError(
                f"Unknown phase_function {self.phase_function!r}; "
                f"choose from {PHASE_FUNCTIONS}."
            )
        # The Mie fit already includes the diffraction peak, so removing it
        # from the extinction as well would double count the correction.
        if self.phase_function == "mie" and self.delta_scale_diffraction:
            raise ValueError(
                f"Species {self.name!r}: delta_scale_diffraction must be False "
                "with the 'mie' phase function, which models the forward peak."
            )
        if not -1.0 < self.asymmetry_parameter < 1.0:
            raise ValueError("asymmetry_parameter must lie strictly in (-1, 1).")

    def with_overrides(self, **changes) -> "HydrometeorSpecies":
        """Return a copy with some fields changed (e.g. a different variable)."""
        return replace(self, **changes)

    def moment_factor(self, moment_order: float) -> float:
        """
        ``G(k) = Gamma(mu + k + 1) / Gamma(mu + 1)``.

        The k-th moment of the gamma distribution is ``N * G(k) / lambda**k``,
        so this is the only special function the optics need. Computed through
        log-gamma to avoid overflow for large shape parameters.
        """
        return math.exp(
            math.lgamma(self.gamma_shape + moment_order + 1.0)
            - math.lgamma(self.gamma_shape + 1.0)
        )


@dataclass
class SpeciesOptics:
    """
    Render-ready optical fields for one species at one time.

    Attributes:
        species: The species these fields describe.
        extinction: Full geometric-optics extinction coefficient, m^-1.
        render_extinction: Extinction actually given to the renderer, m^-1:
            ``extinction`` itself for "mie", or the delta-scaled value.
        render_anisotropy: Henyey-Greenstein ``g`` given to the renderer (the
            delta-scaled value when delta scaling is on); None for "mie".
        effective_diameter: Third-over-second moment diameter, m (meaningful
            for spheres; used to drive the Mie fit).
        mie_fit_diameter_microns: Diameter passed to the Cycles Mie phase
            function, um; None unless the species uses "mie".
    """

    species: HydrometeorSpecies
    extinction: xr.DataArray
    render_extinction: xr.DataArray
    render_anisotropy: Optional[float]
    effective_diameter: xr.DataArray
    mie_fit_diameter_microns: Optional[xr.DataArray]


# =============================================================================
# Core size-distribution optics (pure numpy)
# =============================================================================
def compute_slope_and_number(
    mass_content: np.ndarray,
    species: HydrometeorSpecies,
    number_concentration: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Close the gamma distribution: slope ``lambda`` and number ``N`` per m^3.

    The mass content ``M = integral(a D**b n(D) dD) = a N G(b) / lambda**b``
    fixes one constraint. The second comes from the predicted or fixed number
    concentration, or from a fixed intercept ``N0`` (see the class docstring
    for the priority order). After solving, the mean diameter ``(mu + 1) /
    lambda`` is clamped into ``species.mean_diameter_bounds`` and ``N`` is
    recomputed from ``M`` and the clamped slope. Mass is therefore always
    conserved, which is the same repair bulk schemes apply to inconsistent
    mass/number pairs.

    Args:
        mass_content: Hydrometeor mass per volume of air, kg m^-3. Cells with
            zero (or negative) mass return lambda = inf and N = 0.
        species: The species definition (supplies a, b, mu and the closure).
        number_concentration: Number per m^3 of air, broadcastable to
            ``mass_content``; None to use the species' fixed closure.

    Returns:
        (slope_parameter, number_concentration), both shaped like
        ``mass_content``. Units m^-1 and m^-3.
    """
    mass_content = np.asarray(mass_content, dtype=np.float64)
    has_mass = mass_content > 0.0
    # Avoid dividing by zero in cells without hydrometeors; masked out below.
    safe_mass_content = np.where(has_mass, mass_content, 1.0)

    a = species.mass_coefficient
    b = species.mass_exponent
    mu = species.gamma_shape

    # ---- Pick the closure: number concentration first, then fixed intercept.
    if number_concentration is None and species.fixed_number_concentration is not None:
        number_concentration = species.fixed_number_concentration

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if number_concentration is not None:
            # Number known: M = a N G(b) / lambda**b  ->  solve for lambda.
            number_concentration = np.broadcast_to(
                np.asarray(number_concentration, dtype=np.float64),
                mass_content.shape,
            )
            # Cells with mass but no number (numerical noise) get a tiny number,
            # which the diameter clamp below turns into the largest allowed
            # particles: the least-extinguishing reading of the mass.
            positive_number = np.maximum(number_concentration, 1.0e-30)
            slope_parameter = (
                a * positive_number * species.moment_factor(b) / safe_mass_content
            ) ** (1.0 / b)
        elif species.fixed_intercept is not None:
            # Intercept known: M = a N0 Gamma(mu + b + 1) / lambda**(mu + b + 1).
            total_power = mu + b + 1.0
            slope_parameter = (
                a
                * species.fixed_intercept
                * math.exp(math.lgamma(total_power))
                / safe_mass_content
            ) ** (1.0 / total_power)
        else:
            raise ValueError(
                f"Species {species.name!r} has no size closure: provide a "
                "number_variable present in the dataset, a "
                "fixed_number_concentration, or a fixed_intercept."
            )

        # ---- Clamp the mean diameter (mu + 1) / lambda into physical bounds.
        min_mean_diameter, max_mean_diameter = species.mean_diameter_bounds
        slope_parameter = np.clip(
            slope_parameter,
            (mu + 1.0) / max_mean_diameter,
            (mu + 1.0) / min_mean_diameter,
        )

        # ---- Recompute N from M and the (possibly clamped) slope, so mass is
        # conserved exactly whatever closure was used.
        number_from_mass = (
            safe_mass_content * slope_parameter**b / (a * species.moment_factor(b))
        )

    slope_parameter = np.where(has_mass, slope_parameter, np.inf)
    number_from_mass = np.where(has_mass, number_from_mass, 0.0)
    return slope_parameter, number_from_mass


def extinction_efficiency(
    species: HydrometeorSpecies, effective_diameter: np.ndarray
) -> Union[float, np.ndarray]:
    """
    Extinction efficiency ``Q_ext`` for a species.

    Spheres (mass exponent 3, area exponent 2) get the Mie edge-effect
    correction ``2 + 0.75 * (D_eff / 1 um)**(-2/3)``: about +8% at 10 um, +5%
    at 20 um, and under +0.5% for raindrops. Non-spherical ice keeps
    ``Q_ext = 2``, since the correction was derived for spheres and those
    particles are large enough that it would be under 2% anyway.
    """
    is_sphere = species.mass_exponent == 3.0 and species.area_exponent == 2.0
    if not is_sphere:
        return LARGE_PARTICLE_EXTINCTION_EFFICIENCY
    effective_diameter_microns = np.maximum(effective_diameter * 1.0e6, 1.0e-3)
    return LARGE_PARTICLE_EXTINCTION_EFFICIENCY + (
        DROPLET_EXTINCTION_EDGE_COEFFICIENT * effective_diameter_microns ** (-2.0 / 3.0)
    )


def compute_extinction(
    mass_content: np.ndarray,
    species: HydrometeorSpecies,
    number_concentration: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Geometric-optics extinction coefficient and effective diameter.

    Integrates the projected area over the closed gamma distribution,
    ``A_total = gamma N G(sigma) / lambda**sigma``, and multiplies by the
    extinction efficiency (~2). For spheres this equals
    ``Q_ext * 3 M / (2 rho_p D_eff)``. The effective diameter is the
    third-over-second moment ratio ``(mu + 3) / lambda``, which is what sets
    the optics of a sphere population and what drives the Mie phase fit.

    Args:
        mass_content: Hydrometeor mass per volume of air, kg m^-3.
        species: Species definition.
        number_concentration: Number per m^3, or None for the fixed closure.

    Returns:
        (extinction in m^-1, effective_diameter in m), shaped like
        ``mass_content``. Both are 0 where there is no mass.
    """
    slope_parameter, number_per_volume = compute_slope_and_number(
        mass_content, species, number_concentration
    )
    has_mass = np.isfinite(slope_parameter)
    safe_slope_parameter = np.where(has_mass, slope_parameter, 1.0)

    # Effective diameter: ratio of the 3rd to the 2nd moment of n(D).
    effective_diameter = np.where(
        has_mass, (species.gamma_shape + 3.0) / safe_slope_parameter, 0.0
    )

    # Total projected area per volume: gamma * N * G(sigma) / lambda**sigma.
    projected_area_per_volume = (
        species.area_coefficient
        * number_per_volume
        * species.moment_factor(species.area_exponent)
        / safe_slope_parameter**species.area_exponent
    )

    # Extinction = efficiency * projected area (efficiency ~2: extinction paradox).
    extinction = np.where(
        has_mass,
        extinction_efficiency(species, effective_diameter) * projected_area_per_volume,
        0.0,
    )
    return extinction, effective_diameter


def delta_scaled_optics(
    extinction: np.ndarray, asymmetry_parameter: float
) -> Tuple[np.ndarray, float]:
    """
    Remove the diffraction peak: ``beta(1 - f)`` and ``(g - f) / (1 - f)``.

    ``f = 1/2`` is the share of extinction diffracted into a lobe narrower
    than the solar disc for large particles. The transport coefficient
    ``beta (1 - g)`` is unchanged, so the diffuse light field is preserved
    while the direct beam passes through with the visual optical depth.
    """
    forward_peak_fraction = DIFFRACTION_FRACTION_OF_EXTINCTION
    scaled_extinction = extinction * (1.0 - forward_peak_fraction)
    scaled_asymmetry = (asymmetry_parameter - forward_peak_fraction) / (
        1.0 - forward_peak_fraction
    )
    return scaled_extinction, scaled_asymmetry


def mie_fit_diameter_microns(effective_diameter: np.ndarray) -> np.ndarray:
    """
    Convert an effective diameter (m) to the Cycles Mie-fit diameter (um).

    The fit is indexed by the mean diameter of a narrow log-normal population.
    We pick the fit population whose *effective* diameter matches ours, then
    clamp to the fit's range (see ``MIE_FIT_DIAMETER_BOUNDS_MICRONS``).
    """
    fit_diameter = effective_diameter * 1.0e6 * MIE_FIT_DIAMETER_PER_EFFECTIVE_DIAMETER
    return np.clip(fit_diameter, *MIE_FIT_DIAMETER_BOUNDS_MICRONS)


# =============================================================================
# Species presets
# =============================================================================
def _sphere_mass_coefficient(bulk_density: float) -> float:
    """``a`` for spheres of a given bulk density: m = (pi / 6) rho D**3."""
    return math.pi / 6.0 * bulk_density


# RAMS default gamma shape parameters GNU, indexed like RAMSIN's GNU list:
# cloud, rain, pristine, snow, aggregates, graupel, hail, drizzle. RAMS's
# opspec requires GNU >= 4 for cloud and drizzle; 2 is the usual default else.
RAMS_DEFAULT_GNU = (4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0)


def rams_hydrometeor_species(
    gnu: Sequence[float] = RAMS_DEFAULT_GNU,
    fallback_cloud_number_concentration: float = 3.0e8,
) -> List[HydrometeorSpecies]:
    """
    Species for the RAMS bin-emulating two-moment scheme (Meyers et al. 1997;
    Saleeby & van den Heever 2013).

    Mass power laws are the ``cfmas``/``pwmas`` columns of
    ``dstprms1`` in RAMS ``mic_init.f90``. Pristine ice and snow use the
    column habit, RAMS's default. RAMS's gamma form is ``(D/Dn)**(GNU-1)``, so
    ``mu = GNU - 1``. Liquid, graupel and hail are spheres in RAMS. Pristine
    ice, snow and aggregates take the Mie/P3 non-spherical-ice area law, since
    RAMS itself defines no projected area for them.

    Args:
        gnu: The eight RAMSIN ``GNU`` values (cloud, rain, pristine, snow,
            aggregates, graupel, hail, drizzle).
        fallback_cloud_number_concentration: Droplets per m^3 used if ``CCP``
            is absent (a single-moment cloud run). RAMS's default ``CPARM`` is
            3e8 per kg, which is ~3e8 per m^3 near the surface.
    """
    gnu_cloud, gnu_rain, gnu_pristine, gnu_snow, gnu_aggregates = gnu[:5]
    gnu_graupel, gnu_hail, gnu_drizzle = gnu[5:8]
    return [
        HydrometeorSpecies(
            name="cloud",
            mixing_ratio_variable="RCP",
            number_variable="CCP",
            fixed_number_concentration=fallback_cloud_number_concentration,
            mass_coefficient=524.0,
            mass_exponent=3.0,
            gamma_shape=gnu_cloud - 1.0,
            asymmetry_parameter=DROPLET_ASYMMETRY_PARAMETER,
            delta_scale_diffraction=False,
            mean_diameter_bounds=(2.0e-6, 50.0e-6),
        ),
        HydrometeorSpecies(
            name="drizzle",
            mixing_ratio_variable="RDP",
            number_variable="CDP",
            mass_coefficient=524.0,
            mass_exponent=3.0,
            gamma_shape=gnu_drizzle - 1.0,
            asymmetry_parameter=0.88,
            mean_diameter_bounds=(50.0e-6, 200.0e-6),
        ),
        HydrometeorSpecies(
            name="rain",
            mixing_ratio_variable="RRP",
            number_variable="CRP",
            mass_coefficient=524.0,
            mass_exponent=3.0,
            gamma_shape=gnu_rain - 1.0,
            asymmetry_parameter=0.88,
            mean_diameter_bounds=(0.1e-3, 5.0e-3),
        ),
        HydrometeorSpecies(
            name="pristine_ice",
            mixing_ratio_variable="RPP",
            number_variable="CPP",
            mass_coefficient=110.8,
            mass_exponent=2.91,
            area_coefficient=NONSPHERICAL_ICE_AREA_COEFFICIENT,
            area_exponent=NONSPHERICAL_ICE_AREA_EXPONENT,
            gamma_shape=gnu_pristine - 1.0,
            asymmetry_parameter=0.75,
            mean_diameter_bounds=(15.0e-6, 125.0e-6),
        ),
        HydrometeorSpecies(
            name="snow",
            mixing_ratio_variable="RSP",
            number_variable="CSP",
            mass_coefficient=2.739e-3,
            mass_exponent=1.74,
            area_coefficient=NONSPHERICAL_ICE_AREA_COEFFICIENT,
            area_exponent=NONSPHERICAL_ICE_AREA_EXPONENT,
            gamma_shape=gnu_snow - 1.0,
            asymmetry_parameter=0.75,
            mean_diameter_bounds=(0.1e-3, 10.0e-3),
        ),
        HydrometeorSpecies(
            name="aggregates",
            mixing_ratio_variable="RAP",
            number_variable="CAP",
            mass_coefficient=0.496,
            mass_exponent=2.4,
            area_coefficient=NONSPHERICAL_ICE_AREA_COEFFICIENT,
            area_exponent=NONSPHERICAL_ICE_AREA_EXPONENT,
            gamma_shape=gnu_aggregates - 1.0,
            asymmetry_parameter=0.75,
            mean_diameter_bounds=(0.1e-3, 10.0e-3),
        ),
        HydrometeorSpecies(
            name="graupel",
            mixing_ratio_variable="RGP",
            number_variable="CGP",
            mass_coefficient=157.0,
            mass_exponent=3.0,
            gamma_shape=gnu_graupel - 1.0,
            asymmetry_parameter=0.75,
            mean_diameter_bounds=(0.1e-3, 5.0e-3),
        ),
        HydrometeorSpecies(
            name="hail",
            mixing_ratio_variable="RHP",
            number_variable="CHP",
            mass_coefficient=471.0,
            mass_exponent=3.0,
            gamma_shape=gnu_hail - 1.0,
            asymmetry_parameter=0.80,
            mean_diameter_bounds=(0.8e-3, 10.0e-3),
        ),
    ]


def cm1_morrison_hydrometeor_species(
    droplet_number_concentration: float = 250.0e6,
    droplet_gamma_shape: float = 5.0,
    graupel_density: float = 400.0,
) -> List[HydrometeorSpecies]:
    """
    Species for the Morrison double-moment scheme in CM1 (``ptype = 5``).

    Morrison treats every species as spheres of a fixed bulk density
    (``morrison.F``: water 997, cloud ice 500, snow 100, graupel 400 or 900
    with ``ihail = 1``) and every precipitating/ice species as exponential
    (``mu = 0``). Droplet number is constant (namelist ``ndcnst``, cm^-3)
    unless the ``ncc`` variable is written, in which case it is used.

    Morrison diagnoses the droplet shape from droplet number; 5 corresponds to
    ~250 cm^-3, and the shape barely changes the extinction (~3% from 2 to 10).

    Args:
        droplet_number_concentration: ``ndcnst`` converted to m^-3.
        droplet_gamma_shape: Fixed droplet ``mu``.
        graupel_density: 400 (graupel) or 900 (``ihail = 1``), kg m^-3.
    """
    return [
        HydrometeorSpecies(
            name="cloud",
            mixing_ratio_variable="qc",
            number_variable="ncc",
            fixed_number_concentration=droplet_number_concentration,
            mass_coefficient=_sphere_mass_coefficient(997.0),
            mass_exponent=3.0,
            gamma_shape=droplet_gamma_shape,
            asymmetry_parameter=DROPLET_ASYMMETRY_PARAMETER,
            delta_scale_diffraction=False,
            mean_diameter_bounds=(1.0e-6, 50.0e-6),
        ),
        HydrometeorSpecies(
            name="rain",
            mixing_ratio_variable="qr",
            number_variable="ncr",
            mass_coefficient=_sphere_mass_coefficient(997.0),
            mass_exponent=3.0,
            asymmetry_parameter=0.88,
            mean_diameter_bounds=(20.0e-6, 2.8e-3),
        ),
        HydrometeorSpecies(
            name="cloud_ice",
            mixing_ratio_variable="qi",
            number_variable="nci",
            mass_coefficient=_sphere_mass_coefficient(500.0),
            mass_exponent=3.0,
            asymmetry_parameter=0.75,
            mean_diameter_bounds=(1.0e-6, 2.0e-3),
        ),
        HydrometeorSpecies(
            name="snow",
            mixing_ratio_variable="qs",
            number_variable="ncs",
            mass_coefficient=_sphere_mass_coefficient(100.0),
            mass_exponent=3.0,
            asymmetry_parameter=0.75,
            mean_diameter_bounds=(10.0e-6, 2.0e-3),
        ),
        HydrometeorSpecies(
            name="graupel",
            mixing_ratio_variable="qg",
            number_variable="ncg",
            mass_coefficient=_sphere_mass_coefficient(graupel_density),
            mass_exponent=3.0,
            asymmetry_parameter=0.75 if graupel_density < 900.0 else 0.80,
            mean_diameter_bounds=(50.0e-6, 2.0e-3),
        ),
    ]


def generic_single_moment_species() -> List[HydrometeorSpecies]:
    """
    Species for a generic single-moment (Kessler/Lin-type) scheme.

    Uses CM1-style lower-case names. Cloud water and cloud ice have fixed
    number concentrations. Precipitation has the classic fixed exponential
    intercepts and densities:

    * rain: Marshall & Palmer (1948), ``N0 = 8e6 m^-4``;
    * snow: ``N0 = 3e6 m^-4``, density 100 (Lin et al. 1983);
    * graupel: ``N0 = 4e6 m^-4``, density 400 (Rutledge & Hobbs 1984);
    * hail: ``N0 = 4e4 m^-4``, density 917 (Lin et al. 1983).

    The cloud-ice concentration (1e5 m^-3 = 100 per litre) and droplet
    concentration (250 cm^-3) are representative assumptions. Override them
    with ``HydrometeorSpecies.with_overrides`` to match a specific scheme.
    """
    return [
        HydrometeorSpecies(
            name="cloud",
            mixing_ratio_variable="qc",
            fixed_number_concentration=250.0e6,
            mass_coefficient=_sphere_mass_coefficient(WATER_DENSITY),
            mass_exponent=3.0,
            gamma_shape=4.0,
            asymmetry_parameter=DROPLET_ASYMMETRY_PARAMETER,
            delta_scale_diffraction=False,
            mean_diameter_bounds=(1.0e-6, 50.0e-6),
        ),
        HydrometeorSpecies(
            name="rain",
            mixing_ratio_variable="qr",
            fixed_intercept=8.0e6,
            mass_coefficient=_sphere_mass_coefficient(WATER_DENSITY),
            mass_exponent=3.0,
            asymmetry_parameter=0.88,
            mean_diameter_bounds=(50.0e-6, 5.0e-3),
        ),
        HydrometeorSpecies(
            name="cloud_ice",
            mixing_ratio_variable="qi",
            fixed_number_concentration=1.0e5,
            mass_coefficient=_sphere_mass_coefficient(500.0),
            mass_exponent=3.0,
            asymmetry_parameter=0.75,
            mean_diameter_bounds=(1.0e-6, 1.0e-3),
        ),
        HydrometeorSpecies(
            name="snow",
            mixing_ratio_variable="qs",
            fixed_intercept=3.0e6,
            mass_coefficient=_sphere_mass_coefficient(100.0),
            mass_exponent=3.0,
            asymmetry_parameter=0.75,
            mean_diameter_bounds=(50.0e-6, 10.0e-3),
        ),
        HydrometeorSpecies(
            name="graupel",
            mixing_ratio_variable="qg",
            fixed_intercept=4.0e6,
            mass_coefficient=_sphere_mass_coefficient(400.0),
            mass_exponent=3.0,
            asymmetry_parameter=0.75,
            mean_diameter_bounds=(50.0e-6, 10.0e-3),
        ),
        HydrometeorSpecies(
            name="hail",
            mixing_ratio_variable="qh",
            fixed_intercept=4.0e4,
            mass_coefficient=_sphere_mass_coefficient(SOLID_ICE_DENSITY),
            mass_exponent=3.0,
            asymmetry_parameter=0.80,
            mean_diameter_bounds=(0.5e-3, 50.0e-3),
        ),
    ]


# Named microphysics presets. Each value builds a fresh species list.
MICROPHYSICS_PRESETS = {
    "rams": rams_hydrometeor_species,
    "cm1_morrison": cm1_morrison_hydrometeor_species,
    "generic": generic_single_moment_species,
}


# =============================================================================
# Dataset-level helpers
# =============================================================================
def _find_variable(dataset: xr.Dataset, variable_name: Optional[str]) -> Optional[str]:
    """Return the dataset's name for a variable, matching case-insensitively."""
    if variable_name is None:
        return None
    if variable_name in dataset.data_vars:
        return variable_name
    lowercase_lookup = {str(name).lower(): name for name in dataset.data_vars}
    return lowercase_lookup.get(variable_name.lower())


def detect_microphysics_preset(dataset: xr.Dataset) -> str:
    """
    Guess which microphysics preset matches a dataset's variable names.

    RAMS is identified by its upper-case ``R?P`` mixing ratios, CM1 Morrison by
    its ``nc?`` number concentrations. Anything else falls back to "generic".
    """
    variable_names = set(map(str, dataset.data_vars))
    if variable_names & {"RCP", "RRP", "RPP", "RSP", "RAP", "RGP", "RHP", "RDP"}:
        return "rams"
    if variable_names & {"ncr", "nci", "ncs", "ncg"}:
        return "cm1_morrison"
    return "generic"


def resolve_cloud_species(
    dataset: xr.Dataset,
    species: Optional[Sequence[HydrometeorSpecies]] = None,
    microphysics: Optional[str] = None,
) -> List[HydrometeorSpecies]:
    """
    Decide which species to render, with variable names as the dataset has them.

    Args:
        dataset: The model output.
        species: Explicit species list. When given, ``microphysics`` is ignored.
        microphysics: Preset name from :data:`MICROPHYSICS_PRESETS`, or None to
            auto-detect with :func:`detect_microphysics_preset`.

    Returns:
        Species whose mixing ratio is present in the dataset, with
        ``mixing_ratio_variable``/``number_variable`` rewritten to the
        dataset's spelling. A ``number_variable`` absent from the dataset is
        set to None, so the species falls back to its fixed closure.

    Raises:
        ValueError: If no listed species has its mixing ratio in the dataset.
    """
    if species is None:
        preset_name = microphysics or detect_microphysics_preset(dataset)
        if preset_name not in MICROPHYSICS_PRESETS:
            raise ValueError(
                f"Unknown microphysics preset {preset_name!r}; choose from "
                f"{sorted(MICROPHYSICS_PRESETS)} or pass species=[...]."
            )
        species = MICROPHYSICS_PRESETS[preset_name]()

    resolved_species: List[HydrometeorSpecies] = []
    for candidate in species:
        mixing_ratio_name = _find_variable(dataset, candidate.mixing_ratio_variable)
        if mixing_ratio_name is None:
            continue
        number_name = _find_variable(dataset, candidate.number_variable)
        resolved = candidate.with_overrides(
            mixing_ratio_variable=mixing_ratio_name, number_variable=number_name
        )
        # Without a number variable, the fixed closure must exist.
        if (
            number_name is None
            and resolved.fixed_number_concentration is None
            and resolved.fixed_intercept is None
        ):
            raise ValueError(
                f"Species {candidate.name!r}: number variable "
                f"{candidate.number_variable!r} is not in the dataset and no "
                "fixed_number_concentration or fixed_intercept is set."
            )
        if candidate.number_variable is not None and number_name is None:
            warnings.warn(
                f"Cloud species {candidate.name!r}: number concentration "
                f"{candidate.number_variable!r} not found; using the fixed "
                "size closure instead.",
                stacklevel=2,
            )
        resolved_species.append(resolved)

    if not resolved_species:
        wanted = [candidate.mixing_ratio_variable for candidate in species]
        raise ValueError(
            f"None of the hydrometeor mixing ratios {wanted} are in the dataset. "
            "Pass species=[HydrometeorSpecies(...)] naming your variables."
        )
    return resolved_species


def standard_atmosphere_density(height_m: np.ndarray) -> np.ndarray:
    """
    Air density of the 1976 US Standard Atmosphere, kg m^-3, for 0-20 km.

    Troposphere: linear temperature lapse of 6.5 K/km from 288.15 K and the
    matching hydrostatic pressure. Above 11 km: isothermal at 216.65 K with an
    exponential pressure decay. Used only when the dataset carries no density.
    """
    height_m = np.clip(np.asarray(height_m, dtype=np.float64), -500.0, 20000.0)
    # Physical constants of the standard atmosphere.
    gas_constant_dry_air = 287.053
    gravity = 9.80665
    lapse_rate = 0.0065
    surface_temperature = 288.15
    surface_pressure = 101325.0
    tropopause_height = 11000.0
    tropopause_temperature = surface_temperature - lapse_rate * tropopause_height
    hydrostatic_exponent = gravity / (gas_constant_dry_air * lapse_rate)
    tropopause_pressure = (
        surface_pressure
        * (tropopause_temperature / surface_temperature) ** hydrostatic_exponent
    )

    # Troposphere: T falls linearly, p follows the hydrostatic power law.
    tropospheric_temperature = surface_temperature - lapse_rate * height_m
    tropospheric_pressure = (
        surface_pressure
        * (tropospheric_temperature / surface_temperature) ** hydrostatic_exponent
    )

    # Lower stratosphere: isothermal, p decays exponentially with height.
    stratospheric_pressure = tropopause_pressure * np.exp(
        -gravity
        * (height_m - tropopause_height)
        / (gas_constant_dry_air * tropopause_temperature)
    )

    in_troposphere = height_m <= tropopause_height
    temperature = np.where(
        in_troposphere, tropospheric_temperature, tropopause_temperature
    )
    pressure = np.where(in_troposphere, tropospheric_pressure, stratospheric_pressure)
    return pressure / (gas_constant_dry_air * temperature)


def resolve_air_density(
    dataset: xr.Dataset,
    air_density: Union[str, float, None],
    vertical_coordinate_name: str,
) -> Union[xr.DataArray, float]:
    """
    Air density in kg m^-3: a dataset variable, a constant, or an estimate.

    Mixing ratios are per kg of *air*, so converting them to mass per volume
    needs the air density. A cloud at 10 km sees air ~3x thinner than at the
    surface, so a constant would misstate upper-level extinction by that
    factor. When no density is supplied or found under a common name, the
    1976 standard atmosphere evaluated at the vertical coordinate is used
    (with a warning). It is accurate to ~5-10% for most convective cases.

    Args:
        dataset: The model output.
        air_density: Variable name, constant value, or None to search
            :data:`AIR_DENSITY_VARIABLE_CANDIDATES` and fall back.
        vertical_coordinate_name: Name of the height coordinate (meters).
    """
    if isinstance(air_density, (int, float)):
        return float(air_density)
    if isinstance(air_density, str):
        variable_name = _find_variable(dataset, air_density)
        if variable_name is None:
            raise ValueError(f"Air density variable {air_density!r} not in dataset.")
        return dataset[variable_name]

    # Search the usual names before falling back to the standard atmosphere.
    for candidate in AIR_DENSITY_VARIABLE_CANDIDATES:
        variable_name = _find_variable(dataset, candidate)
        if variable_name is not None:
            return dataset[variable_name]

    warnings.warn(
        "No air-density variable found (tried "
        f"{AIR_DENSITY_VARIABLE_CANDIDATES}); using the 1976 US standard "
        f"atmosphere at coordinate {vertical_coordinate_name!r}, assumed to be "
        "height in meters. Pass air_density='<variable>' for exact extinction.",
        stacklevel=2,
    )
    height_values = dataset[vertical_coordinate_name]
    return xr.DataArray(
        standard_atmosphere_density(height_values.values),
        coords=height_values.coords,
        dims=height_values.dims,
    )


def compute_cloud_optics(
    dataset: xr.Dataset,
    species: Sequence[HydrometeorSpecies],
    air_density: Union[xr.DataArray, float],
) -> List[SpeciesOptics]:
    """
    Compute render-ready optics for every species at one time.

    For each species: mass content ``rho_air * q`` -> closed gamma
    distribution -> geometric-optics extinction and effective diameter ->
    the renderer's phase-function inputs (Mie fit diameter, or delta-scaled
    extinction and Henyey-Greenstein ``g``).

    Args:
        dataset: Model output at a single time (no time dimension needed).
        species: Species already resolved against this dataset
            (:func:`resolve_cloud_species`).
        air_density: From :func:`resolve_air_density`.

    Returns:
        One :class:`SpeciesOptics` per species, with DataArrays on the mixing
        ratio's grid.
    """
    optics_per_species: List[SpeciesOptics] = []
    for one_species in species:
        mixing_ratio = dataset[one_species.mixing_ratio_variable]

        # Negative mixing ratios (advection undershoots) carry no particles.
        mixing_ratio = mixing_ratio.clip(min=0.0).fillna(0.0)
        # Mass per volume of air, kg m^-3, broadcast onto the mixing ratio grid.
        mass_content, density_on_grid = xr.broadcast(
            mixing_ratio * air_density, mixing_ratio * 0.0 + air_density
        )

        # Number per volume of air, m^-3 (numbers are written per kg of air).
        number_concentration = None
        if one_species.number_variable is not None:
            number_per_kg = dataset[one_species.number_variable].clip(min=0.0)
            number_concentration = (
                (number_per_kg.fillna(0.0) * density_on_grid)
                .transpose(*mass_content.dims)
                .values
            )

        # Geometric-optics extinction and effective diameter.
        extinction_values, effective_diameter_values = compute_extinction(
            mass_content.values, one_species, number_concentration
        )
        extinction = mass_content.copy(data=extinction_values)
        effective_diameter = mass_content.copy(data=effective_diameter_values)

        # Renderer inputs: Mie fit diameter, or delta-scaled HG optics.
        if one_species.phase_function == "mie":
            render_extinction = extinction
            render_anisotropy = None
            mie_diameter = mass_content.copy(
                data=mie_fit_diameter_microns(effective_diameter_values)
            )
        elif one_species.delta_scale_diffraction:
            scaled_values, render_anisotropy = delta_scaled_optics(
                extinction_values, one_species.asymmetry_parameter
            )
            render_extinction = mass_content.copy(data=scaled_values)
            mie_diameter = None
        else:
            render_extinction = extinction
            render_anisotropy = one_species.asymmetry_parameter
            mie_diameter = None

        optics_per_species.append(
            SpeciesOptics(
                species=one_species,
                extinction=extinction,
                render_extinction=render_extinction,
                render_anisotropy=render_anisotropy,
                effective_diameter=effective_diameter,
                mie_fit_diameter_microns=mie_diameter,
            )
        )
    return optics_per_species


def total_render_extinction(
    optics_per_species: Sequence[SpeciesOptics],
) -> xr.DataArray:
    """Sum of the species' render extinctions: the mixture's extinction, m^-1."""
    total = optics_per_species[0].render_extinction
    for optics in optics_per_species[1:]:
        total = total + optics.render_extinction
    return total


def species_summary(species: HydrometeorSpecies) -> Dict[str, object]:
    """Plain-dict description of a species for the manifest (provenance)."""
    return {
        "name": species.name,
        "mixing_ratio_variable": species.mixing_ratio_variable,
        "number_variable": species.number_variable,
        "fixed_number_concentration": species.fixed_number_concentration,
        "fixed_intercept": species.fixed_intercept,
        "mass_power_law": [species.mass_coefficient, species.mass_exponent],
        "area_power_law": [species.area_coefficient, species.area_exponent],
        "gamma_shape": species.gamma_shape,
        "phase_function": species.phase_function,
        "asymmetry_parameter": species.asymmetry_parameter,
        "delta_scale_diffraction": species.delta_scale_diffraction,
    }
