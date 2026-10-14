"""
Blender-side build script for a skyvista bundle.

Run inside Blender (this file uses ``bpy``; it is never imported by skyvista
itself)::

    blender --background --python build_scene.py -- /path/to/bundle [--render]

It reads the bundle's ``scene.json`` and assembles a ``.blend``:

* a parent Empty ("skyvista_root") carrying the single scene transform
  (origin shift + scale + vertical exaggeration), with every data object
  parented to it so the geometry stays in physical units and the whole figure
  is one movable/scalable Blender-native unit;
* Alembic mesh sequences imported via Mesh Sequence Cache modifiers, with a
  material that reads the baked per-vertex "col_data" attribute (or a solid color);
* VDB volume sequences with a Principled Volume shader whose density and color
  are driven by the named grid through a color ramp rebuilt from the bundle's
  colormap LUT;
* physically rendered clouds: a VDB sequence with one extinction grid per
  hydrometeor species, rendered by one Volume Scatter closure per species
  summed in a single shader (see ``make_cloud_material``);
* vector glyphs: per-frame point files (position, vector, color) turned into
  point clouds, on which a geometry-nodes tree instances one arrow per point,
  aligned with and scaled by its vector (see ``make_glyph_node_group``);
* streamlines: per-frame polylines rebuilt as Curves objects, swept into
  tubes with downstream-pointing cones (see ``make_streamline_node_group``);
  with ``arrow_travel_time`` the cones move with the air as the manifest's
  flow clock advances (see ``add_moving_streamline_markers``);
* a camera (from the manifest, or a sensible default framing the data);
* render settings (engine, samples, resolution, Standard view transform).

The design goal is a script-free, hand-editable result: after this runs, the
``.blend`` references the Alembic/VDB caches through standard modifiers and can
be edited with ordinary Blender tools -- no skyvista or this script required.

Everything is wrapped so one failing object does not abort the whole build; a
summary is printed at the end.
"""

import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


# ---------------------------------------------------------------------------
# Argument / manifest loading
# ---------------------------------------------------------------------------
def parse_arguments():
    """Return (bundle_dir, do_render) from CLI args after the ``--`` separator."""
    if "--" not in sys.argv:
        raise SystemExit(
            "Usage: blender --background --python build_scene.py -- <bundle> [--render]"
        )
    arguments = sys.argv[sys.argv.index("--") + 1 :]
    if not arguments:
        raise SystemExit("No bundle directory provided after '--'.")
    bundle_dir = Path(arguments[0]).resolve()
    do_render = "--render" in arguments[1:]
    return bundle_dir, do_render


def load_manifest(bundle_dir: Path) -> dict:
    manifest_path = bundle_dir / "scene.json"
    if not manifest_path.exists():
        raise SystemExit(f"No scene.json in bundle: {bundle_dir}")
    with open(manifest_path) as manifest_file:
        return json.load(manifest_file)


# ---------------------------------------------------------------------------
# Scene reset + render configuration
# ---------------------------------------------------------------------------
def reset_scene():
    """Delete everything so we build into a clean scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras):
        for datablock in list(collection):
            collection.remove(datablock)


# Volume bounces used for scenes with clouds when the manifest leaves it unset.
# On a synthetic cumulus of optical depth ~300, the sunlit side's brightness
# was within ~1% of its 256-bounce value at 64 bounces; 128 adds margin for
# deeper convection.
CLOUD_DEFAULT_VOLUME_BOUNCES = 128


def configure_render(scene, manifest, scene_has_clouds=False):
    """
    Apply engine, samples, resolution, color management, and frame range.

    For scenes with physically rendered clouds this also sets the settings
    the radiative transfer depends on:

    * ``volume_bounces``: Blender's default of 0 means single scattering only,
      which renders thick clouds grey-brown instead of white.
    * no indirect clamp: clamping removes energy from the multiply scattered
      light that makes up most of a cloud's brightness.
    * the unbiased (null-scattering) volume integrator, whose transmittance
      estimates stay accurate however optically thick a voxel is.
    """
    render_config = manifest.get("render", {})
    engine = render_config.get("engine", "CYCLES")
    try:
        scene.render.engine = engine
    except TypeError:
        scene.render.engine = "CYCLES"

    if scene.render.engine == "CYCLES":
        scene.cycles.samples = int(render_config.get("samples", 128))
    else:
        # EEVEE (Next) uses a different sample attribute name.
        try:
            scene.eevee.taa_render_samples = int(render_config.get("samples", 128))
        except AttributeError:
            pass

    resolution = render_config.get("resolution", [1920, 1080])
    scene.render.resolution_x = int(resolution[0])
    scene.render.resolution_y = int(resolution[1])
    scene.render.film_transparent = bool(render_config.get("film_transparent", True))

    if scene.render.engine == "CYCLES":
        # Denoising also removes the rare negative pixels that the unbiased
        # volume transmittance estimator produces in thick, sunlit clouds.
        scene.cycles.use_denoising = bool(render_config.get("denoise", True))
        volume_bounces = render_config.get("volume_bounces")
        if volume_bounces is None and scene_has_clouds:
            volume_bounces = CLOUD_DEFAULT_VOLUME_BOUNCES
        if volume_bounces is not None:
            scene.cycles.volume_bounces = int(volume_bounces)
            # The total bounce limit caps every other limit, so raise it too.
            scene.cycles.max_bounces = max(
                scene.cycles.max_bounces, int(volume_bounces) + 8
            )
        if scene_has_clouds:
            scene.cycles.sample_clamp_indirect = 0.0
            if hasattr(scene.cycles, "volume_biased"):
                scene.cycles.volume_biased = False

    # Data-faithful color: Standard view transform (not AgX/Filmic).
    try:
        scene.view_settings.view_transform = render_config.get(
            "view_transform", "Standard"
        )
    except TypeError:
        pass

    time_config = manifest.get("time", {})
    scene.render.fps = int(round(time_config.get("fps", 24)))
    scene.frame_start = int(time_config.get("frame_start", 1))
    scene.frame_end = int(time_config.get("frame_end", scene.frame_start))


# ---------------------------------------------------------------------------
# The single scene transform (as a parent Empty)
# ---------------------------------------------------------------------------
def create_root_empty(scene, transform):
    """
    Create the parent Empty implementing blender = (data - origin_shift) * scale
    with an extra vertical factor. Objects parented to it stay in physical units.
    """
    scale = float(transform.get("scale", 1.0e-3))
    z_exaggeration = float(transform.get("z_exaggeration", 1.0))
    origin_shift = transform.get("origin_shift") or [0.0, 0.0, 0.0]

    root_empty = bpy.data.objects.new("skyvista_root", None)
    root_empty.empty_display_type = "PLAIN_AXES"
    scene.collection.objects.link(root_empty)

    # world = location + scale_vec ⊙ local, with location = -scale_vec ⊙ origin.
    scale_vector = Vector((scale, scale, scale * z_exaggeration))
    root_empty.scale = scale_vector
    root_empty.location = Vector(
        (
            -scale_vector.x * origin_shift[0],
            -scale_vector.y * origin_shift[1],
            -scale_vector.z * origin_shift[2],
        )
    )
    return root_empty


def parent_keep_local(child, root_empty):
    """Parent child to root so world = root.matrix_world @ child_local_coords."""
    child.parent = root_empty
    # Identity parent-inverse => the child's local coords ARE the data coords.
    child.matrix_parent_inverse = Matrix.Identity(4)


def transform_point(transform, point):
    """Apply the scene transform to a single data-space point (for the camera)."""
    scale = float(transform.get("scale", 1.0e-3))
    z_exaggeration = float(transform.get("z_exaggeration", 1.0))
    origin_shift = transform.get("origin_shift") or [0.0, 0.0, 0.0]
    return Vector(
        (
            (point[0] - origin_shift[0]) * scale,
            (point[1] - origin_shift[1]) * scale,
            (point[2] - origin_shift[2]) * scale * z_exaggeration,
        )
    )


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
def srgb_to_linear(channel_value):
    """Convert one sRGB channel [0,1] to linear (Blender works in linear)."""
    if channel_value <= 0.04045:
        return channel_value / 12.92
    return ((channel_value + 0.055) / 1.055) ** 2.4


NAMED_COLORS = {
    "white": (1, 1, 1),
    "black": (0, 0, 0),
    "gray": (0.5, 0.5, 0.5),
    "grey": (0.5, 0.5, 0.5),
    "red": (1, 0, 0),
    "green": (0, 0.5, 0),
    "blue": (0, 0, 1),
    "orange": (1, 0.5, 0),
    "yellow": (1, 1, 0),
}


def parse_color(color_spec):
    """Parse '#rrggbb' or a basic color name to a linear RGBA tuple."""
    if (
        isinstance(color_spec, str)
        and color_spec.startswith("#")
        and len(color_spec) == 7
    ):
        srgb = tuple(int(color_spec[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
    else:
        srgb = NAMED_COLORS.get(str(color_spec).lower(), (0.8, 0.8, 0.8))
    return (*(srgb_to_linear(c) for c in srgb), 1.0)


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------
def _principled_emission_color_input(principled):
    """
    Return the Principled BSDF's emission *color* input across Blender versions.

    The socket was renamed over time: 3.x/early-4.x expose "Emission"; 4.x+
    renamed it to "Emission Color" (adding a separate "Emission Strength"). Try
    the current name first, fall back to the old one, and return None if neither
    exists so callers can skip emission gracefully.
    """
    for socket_name in ("Emission Color", "Emission"):
        if socket_name in principled.inputs:
            return principled.inputs[socket_name]
    return None


# Custom material property holding an exposure-relative emission fraction.
RELATIVE_EMISSION_PROPERTY = "skyvista_relative_emission"


def apply_exposure_relative_emission(scene):
    """
    Turn exposure-relative emission fractions into absolute strengths.

    With the Standard view transform the displayed (linear) value is
    ``radiance * 2**exposure``. An emission of strength ``S`` and color ``c``
    has radiance ``S * c``, so ``S = fraction * 2**-exposure`` makes the
    emission alone display as ``fraction * c``, independent of how bright the
    sky is. Must run after the exposure is configured.
    """
    exposure_factor = 2.0 ** (-float(scene.view_settings.exposure))
    for material in bpy.data.materials:
        fraction = material.get(RELATIVE_EMISSION_PROPERTY)
        if fraction is None or not material.node_tree:
            continue
        # A dye volume carries its scale on a Value node (see make_dye_material).
        dye_emission_scale = material.node_tree.nodes.get(DYE_EMISSION_SCALE_NODE)
        if dye_emission_scale is not None:
            dye_emission_scale.outputs[0].default_value = (
                float(fraction) * exposure_factor
            )
            continue
        principled = material.node_tree.nodes.get("Principled BSDF")
        if principled is None or "Emission Strength" not in principled.inputs:
            continue
        principled.inputs["Emission Strength"].default_value = (
            float(fraction) * exposure_factor
        )


def make_surface_material(name, material_spec):
    """
    Build a Principled-BSDF surface material from a manifest material block.

    The color source (a baked per-vertex color attribute, or a solid color) is
    computed once and then wired into Base Color; if the shader preset is
    emissive (``shader.emission_from == "color"``), the same source also drives
    the Emission Color so the surface self-illuminates in its own data color --
    the basis of the "glow" look, which a scene-wide bloom pass then haloes.
    """
    material = bpy.data.materials.new(name=f"{name}_mat")
    material.use_nodes = True
    node_tree = material.node_tree
    links = node_tree.links
    principled = node_tree.nodes.get("Principled BSDF")

    shader = material_spec.get("shader", {})
    if principled is not None:
        if "Roughness" in principled.inputs:
            principled.inputs["Roughness"].default_value = float(
                shader.get("roughness", 0.4)
            )
        if "Metallic" in principled.inputs:
            principled.inputs["Metallic"].default_value = float(
                shader.get("metallic", 0.0)
            )

    # ---- Resolve the single color source (a shader output socket, or a value).
    # For vertex-colored objects it's an Attribute node's Color output; for
    # solid-colored ones it's a constant RGBA. We keep the socket (if any) so it
    # can feed both Base Color and, when emissive, Emission Color.
    coloring = material_spec.get("coloring", {})
    color_output_socket = None
    solid_color_value = None
    if coloring.get("mode") in ("vertex_color", "instance_color"):
        # Read the baked per-vertex color attribute (default "col_data" -- NOT
        # "color", which collides with a reserved attribute in Cycles and
        # renders grey there while EEVEE looks fine).
        attribute_node = node_tree.nodes.new("ShaderNodeAttribute")
        attribute_node.attribute_name = coloring.get("attribute", "col_data")
        if coloring.get("mode") == "instance_color":
            # Instanced glyphs carry their color on the instance, copied from
            # the point they were placed on, not on the arrow mesh's vertices.
            attribute_node.attribute_type = "INSTANCER"
        attribute_node.location = (-350, 0)
        color_output_socket = attribute_node.outputs["Color"]
    else:
        solid_color_value = parse_color(coloring.get("color", "#cccccc"))

    # ---- Base Color from the resolved source.
    if principled is not None:
        if color_output_socket is not None:
            links.new(color_output_socket, principled.inputs["Base Color"])
        else:
            principled.inputs["Base Color"].default_value = solid_color_value

    # ---- Emission: for an emissive preset, drive Emission Color from the same
    # color source and set Emission Strength so the surface self-illuminates.
    emission_strength = float(shader.get("emission_strength", 0.0))
    emission_from = shader.get("emission_from")
    if principled is not None and emission_from == "color" and emission_strength > 0.0:
        emission_color_input = _principled_emission_color_input(principled)
        if emission_color_input is not None:
            if color_output_socket is not None:
                links.new(color_output_socket, emission_color_input)
            else:
                emission_color_input.default_value = solid_color_value
        strength_input = principled.inputs.get("Emission Strength")
        if strength_input is not None:
            strength_input.default_value = emission_strength
        if shader.get("emission_relative_to_exposure"):
            # Final strength depends on the exposure, set later by
            # apply_exposure_relative_emission.
            material[RELATIVE_EMISSION_PROPERTY] = emission_strength

    # Opacity via alpha; enable alpha blending for EEVEE (Cycles honors it too).
    opacity = float(material_spec.get("opacity", 1.0))
    if opacity < 1.0 and principled is not None and "Alpha" in principled.inputs:
        principled.inputs["Alpha"].default_value = opacity
        for blend_attr in ("blend_method", "surface_render_method"):
            if hasattr(material, blend_attr):
                try:
                    setattr(
                        material,
                        blend_attr,
                        "BLEND" if blend_attr == "blend_method" else "BLENDED",
                    )
                except (TypeError, AttributeError):
                    pass
    return material


def load_colormap_stops(bundle_dir, coloring):
    """Load [pos, r, g, b] colormap stops referenced by a volume material."""
    lut_rel = coloring.get("colormap_lut")
    if not lut_rel:
        return [[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    lut_path = bundle_dir / lut_rel
    with open(lut_path) as lut_file:
        return json.load(lut_file)["stops"]


def make_volume_material(name, material_spec, bundle_dir):
    """
    Build a Principled Volume material whose density and color are driven by the
    named VDB grid, with the colormap rebuilt from the bundle's LUT.
    """
    material = bpy.data.materials.new(name=f"{name}_mat")
    material.use_nodes = True
    node_tree = material.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    # Replace the default surface BSDF with a Principled Volume.
    for node in list(nodes):
        if node.type not in ("OUTPUT_MATERIAL",):
            nodes.remove(node)
    output_node = nodes.get("Material Output") or nodes.new("ShaderNodeOutputMaterial")
    volume_node = nodes.new("ShaderNodeVolumePrincipled")
    volume_node.location = (0, 0)
    links.new(volume_node.outputs["Volume"], output_node.inputs["Volume"])

    coloring = material_spec.get("coloring", {})
    density_spec = material_spec.get("density", {})
    grid_name = density_spec.get("grid") or coloring.get("attribute", "density")
    clim = coloring.get("clim", [0.0, 1.0])

    # Read the named grid.
    attribute_node = nodes.new("ShaderNodeAttribute")
    attribute_node.attribute_name = grid_name
    attribute_node.location = (-800, 0)

    # Normalise the raw scalar through clim -> [0, 1].
    map_range_node = nodes.new("ShaderNodeMapRange")
    map_range_node.location = (-600, 0)
    map_range_node.inputs["From Min"].default_value = float(clim[0])
    map_range_node.inputs["From Max"].default_value = float(clim[1])
    map_range_node.inputs["To Min"].default_value = 0.0
    map_range_node.inputs["To Max"].default_value = 1.0
    links.new(attribute_node.outputs["Fac"], map_range_node.inputs["Value"])

    # Color ramp rebuilt from the baked colormap stops.
    color_ramp_node = nodes.new("ShaderNodeValToRGB")
    color_ramp_node.location = (-400, 200)
    stops = load_colormap_stops(bundle_dir, coloring)
    ramp = color_ramp_node.color_ramp
    while len(ramp.elements) > 1:
        ramp.elements.remove(ramp.elements[-1])
    for stop_index, (position, r, g, b) in enumerate(stops):
        element = ramp.elements[0] if stop_index == 0 else ramp.elements.new(position)
        element.position = float(position)
        element.color = (float(r), float(g), float(b), 1.0)
    links.new(map_range_node.outputs["Result"], color_ramp_node.inputs["Fac"])
    links.new(color_ramp_node.outputs["Color"], volume_node.inputs["Color"])

    # Density driven by the normalised scalar times a user-tunable strength.
    density_scale = float(density_spec.get("scale", 1.0))
    if abs(density_scale - 1.0) > 1e-9:
        multiply_node = nodes.new("ShaderNodeMath")
        multiply_node.operation = "MULTIPLY"
        multiply_node.location = (-200, -200)
        multiply_node.inputs[1].default_value = density_scale
        links.new(map_range_node.outputs["Result"], multiply_node.inputs[0])
        links.new(multiply_node.outputs["Value"], volume_node.inputs["Density"])
    else:
        links.new(map_range_node.outputs["Result"], volume_node.inputs["Density"])
    return material


def _set_volume_scatter_phase(scatter_node, phase_function):
    """
    Select a Volume Scatter phase function, returning False if unavailable.

    Blender 4.x added the phase enum (with "MIE"); older versions only have
    Henyey-Greenstein. Blender 5.0 also *hides* the inputs belonging to the
    other phase functions, so the phase must be set before the matching input
    ("Anisotropy" or "Diameter") is looked up.
    """
    identifier = {"mie": "MIE", "henyey_greenstein": "HENYEY_GREENSTEIN"}[
        phase_function
    ]
    if not hasattr(scatter_node, "phase"):
        return phase_function == "henyey_greenstein"
    available = {
        item.identifier for item in scatter_node.bl_rna.properties["phase"].enum_items
    }
    if identifier not in available:
        return False
    scatter_node.phase = identifier
    return True


def _cutaway_mask(nodes, links, cutaway_spec):
    """
    Shader nodes giving 1 where the cloud is kept and 0 on the cut-away side.

    The rule is ``(position - point) . normal > 0`` -> removed, evaluated at
    every volume sample from the Object texture coordinate. The volume's
    object space is the VDB's physical (data) space in meters, the same space
    the manifest's point and normal are given in. Zeroing density can only
    lower the extinction, so Cycles' majorant (its upper bound on extinction,
    taken from the grids) stays valid. Returns None when there is no cutaway.
    """
    if not cutaway_spec:
        return None
    coordinates = nodes.new("ShaderNodeTexCoord")
    coordinates.location = (-1400, 400)
    offset_from_plane = nodes.new("ShaderNodeVectorMath")
    offset_from_plane.operation = "SUBTRACT"
    offset_from_plane.location = (-1200, 400)
    offset_from_plane.inputs[1].default_value = cutaway_spec["point"]
    links.new(coordinates.outputs["Object"], offset_from_plane.inputs[0])
    signed_distance = nodes.new("ShaderNodeVectorMath")
    signed_distance.operation = "DOT_PRODUCT"
    signed_distance.location = (-1000, 400)
    signed_distance.inputs[1].default_value = cutaway_spec["normal"]
    links.new(offset_from_plane.outputs["Vector"], signed_distance.inputs[0])
    kept = nodes.new("ShaderNodeMath")
    kept.operation = "LESS_THAN"
    kept.label = "cutaway: 1 = kept"
    kept.location = (-800, 400)
    kept.inputs[1].default_value = 0.0
    links.new(signed_distance.outputs["Value"], kept.inputs[0])
    return kept.outputs["Value"]


def make_cloud_material(name, material_spec):
    """
    Build the physically based cloud shader from a manifest cloud material.

    Every species contributes one Volume Scatter closure::

        Attribute(extinction_<species>, km^-1)
          -> x (0.001 * extinction_scale)        [km^-1 -> m^-1]
          -> Volume Scatter(Density, white, phase of that species)

    and the closures are summed with Add Shader nodes. Summing is exact: for a
    mixture of independent scatterers the extinction coefficients add, and the
    effective phase function is the scattering-weighted mean of the species'
    phase functions, which is what Cycles samples from a sum of closures. No
    absorption closure is added because water and ice absorb negligibly in
    the visible (single-scattering albedo 1 to better than 1e-4).

    With three or more species Cycles prints "Maximum number of closures
    exceeded: 96 > 64". That count is a conservative compile-time estimate
    (32 slots reserved per volume closure). It is harmless here: the render
    does not depend on the order of the species' closures, which it would if
    any were dropped (checked in Blender 5.0.1 with three and with eight
    distinct g values).

    Units: the grids are in km^-1, and the shader converts them to m^-1. The
    volume object's render space is set to OBJECT, where one unit is one meter
    of the VDB's physical transform, so the optical depth is physical for any
    *uniform* scene scale. Vertical exaggeration is different: Cycles
    compensates density for a single scale factor, so stretching z inflates
    vertical optical depths (an isotropic slab of optical depth 10 reflected
    0.856, 0.910 and 0.958 at 1x, 2x and 5x). The exporter warns about it.
    """
    material = bpy.data.materials.new(name=f"{name}_mat")
    material.use_nodes = True
    node_tree = material.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    # Start from just the output node.
    for node in list(nodes):
        if node.type != "OUTPUT_MATERIAL":
            nodes.remove(node)
    output_node = nodes.get("Material Output") or nodes.new("ShaderNodeOutputMaterial")
    output_node.location = (600, 0)

    # Grid units -> m^-1, times any deliberate user scaling.
    units_to_per_meter = (
        1.0e-3 if material_spec.get("extinction_grid_units") == "1/km" else 1.0
    )
    density_factor = units_to_per_meter * float(
        material_spec.get("extinction_scale", 1.0)
    )

    # Optional cutaway: a 0/1 mask, 0 on the removed side of the plane.
    cutaway_mask_socket = _cutaway_mask(nodes, links, material_spec.get("cutaway"))

    accumulated_shader_socket = None
    for species_index, species in enumerate(material_spec.get("species", [])):
        row_y = -300 * species_index

        # Extinction grid of this species, converted to m^-1.
        extinction_attribute = nodes.new("ShaderNodeAttribute")
        extinction_attribute.attribute_name = species["extinction_grid"]
        extinction_attribute.label = f"{species['name']} extinction (km^-1)"
        extinction_attribute.location = (-800, row_y)
        to_per_meter = nodes.new("ShaderNodeMath")
        to_per_meter.operation = "MULTIPLY"
        to_per_meter.label = f"{species['name']}: km^-1 -> m^-1"
        to_per_meter.inputs[1].default_value = density_factor
        to_per_meter.location = (-600, row_y)
        links.new(extinction_attribute.outputs["Fac"], to_per_meter.inputs[0])

        # Pure, spectrally neutral scattering with this species' phase function.
        scatter = nodes.new("ShaderNodeVolumeScatter")
        scatter.label = f"{species['name']} scattering"
        scatter.location = (-350, row_y)
        scatter.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        density_socket = to_per_meter.outputs[0]
        if cutaway_mask_socket is not None:
            # No extinction on the cut-away side.
            cut = nodes.new("ShaderNodeMath")
            cut.operation = "MULTIPLY"
            cut.label = f"{species['name']}: cutaway"
            cut.location = (-480, row_y)
            links.new(density_socket, cut.inputs[0])
            links.new(cutaway_mask_socket, cut.inputs[1])
            density_socket = cut.outputs[0]
        links.new(density_socket, scatter.inputs["Density"])

        wants_mie = species.get("phase_function") == "mie"
        if wants_mie and _set_volume_scatter_phase(scatter, "mie"):
            # Per-voxel droplet diameter (um) drives the Mie fit.
            diameter_attribute = nodes.new("ShaderNodeAttribute")
            diameter_attribute.attribute_name = species["mie_diameter_grid"]
            diameter_attribute.label = f"{species['name']} Mie diameter (um)"
            diameter_attribute.location = (-600, row_y - 150)
            links.new(diameter_attribute.outputs["Fac"], scatter.inputs["Diameter"])
        else:
            if wants_mie:
                print(
                    f"  [WARN] {name}: no Mie phase in this Blender; "
                    f"{species['name']} falls back to Henyey-Greenstein g=0.86"
                )
            _set_volume_scatter_phase(scatter, "henyey_greenstein")
            anisotropy = species.get("anisotropy")
            scatter.inputs["Anisotropy"].default_value = float(
                0.86 if anisotropy is None else anisotropy
            )

        # Sum this species' closure into the mixture.
        if accumulated_shader_socket is None:
            accumulated_shader_socket = scatter.outputs[0]
        else:
            add_shader = nodes.new("ShaderNodeAddShader")
            add_shader.location = (0, row_y)
            links.new(accumulated_shader_socket, add_shader.inputs[0])
            links.new(scatter.outputs[0], add_shader.inputs[1])
            accumulated_shader_socket = add_shader.outputs[0]

    if accumulated_shader_socket is not None:
        links.new(accumulated_shader_socket, output_node.inputs["Volume"])
    return material


# Name of the Value node holding a dye's emission scale, which
# apply_exposure_relative_emission sets once the exposure is known.
DYE_EMISSION_SCALE_NODE = "skyvista_dye_emission_scale"


def make_dye_material(name, material_spec, bundle_dir):
    """
    Build the passive-tracer dye shader from a manifest dye material.

    The dye is a colored participating medium::

        Attribute(dye_fraction) x extinction_at_full_strength -> sigma (m^-1)
        Attribute(dye_label) -> clim -> [0, 1] -> colormap -> albedo c (linear)
        Volume Scatter(c, sigma, isotropic) + Volume Absorption(c, sigma)

    Cycles' Volume Scatter has scattering coefficient ``sigma * c`` and
    Volume Absorption has absorption coefficient ``sigma * (1 - c)``, per
    color channel. Their sum therefore has extinction ``sigma`` in every
    channel and single-scattering albedo ``c``. The dye scatters its own color
    and absorbs the rest, so it looks like ink in water.

    Optional self-illumination adds a Volume Emission of ``c * sigma * k``.
    Along a ray, emission proportional to extinction saturates at radiance
    ``k * c`` inside thick dye (the source function), and stays faint in
    thin wisps. ``k`` is set from the exposure in
    :func:`apply_exposure_relative_emission`, so thick dye displays as
    ``self_illumination * c`` whatever the sky brightness.

    Colormap stops arrive as sRGB and are converted to linear here, because
    ColorRamp colors (and so the albedo) are scene-linear.
    """
    material = bpy.data.materials.new(name=f"{name}_mat")
    material.use_nodes = True
    node_tree = material.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    # Start from just the output node.
    for node in list(nodes):
        if node.type != "OUTPUT_MATERIAL":
            nodes.remove(node)
    output_node = nodes.get("Material Output") or nodes.new("ShaderNodeOutputMaterial")
    output_node.location = (600, 0)

    # Extinction (m^-1): dyed fraction times the full-strength extinction.
    fraction_attribute = nodes.new("ShaderNodeAttribute")
    fraction_attribute.attribute_name = material_spec["fraction_grid"]
    fraction_attribute.label = "dyed fraction"
    fraction_attribute.location = (-800, -200)
    extinction = nodes.new("ShaderNodeMath")
    extinction.operation = "MULTIPLY"
    extinction.label = "dye extinction (m^-1)"
    extinction.inputs[1].default_value = float(
        material_spec["extinction_at_full_strength"]
    )
    extinction.location = (-600, -200)
    links.new(fraction_attribute.outputs["Fac"], extinction.inputs[0])

    # Albedo: the dye label through the colormap.
    coloring = material_spec["coloring"]
    label_attribute = nodes.new("ShaderNodeAttribute")
    label_attribute.attribute_name = material_spec["label_grid"]
    label_attribute.label = "dye label"
    label_attribute.location = (-800, 200)
    label_to_unit = nodes.new("ShaderNodeMapRange")
    label_to_unit.location = (-600, 200)
    label_to_unit.inputs["From Min"].default_value = float(coloring["clim"][0])
    label_to_unit.inputs["From Max"].default_value = float(coloring["clim"][1])
    links.new(label_attribute.outputs["Fac"], label_to_unit.inputs["Value"])
    color_ramp = nodes.new("ShaderNodeValToRGB")
    color_ramp.location = (-400, 200)
    ramp = color_ramp.color_ramp
    while len(ramp.elements) > 1:
        ramp.elements.remove(ramp.elements[-1])
    for stop_index, (position, r, g, b) in enumerate(
        load_colormap_stops(bundle_dir, coloring)
    ):
        element = ramp.elements[0] if stop_index == 0 else ramp.elements.new(position)
        element.position = float(position)
        element.color = (srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), 1.0)
    links.new(label_to_unit.outputs["Result"], color_ramp.inputs["Fac"])
    albedo_socket = color_ramp.outputs["Color"]

    # Scattering (albedo c) plus absorption (1 - c): extinction sigma overall.
    scatter = nodes.new("ShaderNodeVolumeScatter")
    scatter.label = "dye scattering"
    scatter.location = (-100, 150)
    _set_volume_scatter_phase(scatter, "henyey_greenstein")
    scatter.inputs["Anisotropy"].default_value = 0.0
    links.new(albedo_socket, scatter.inputs["Color"])
    links.new(extinction.outputs[0], scatter.inputs["Density"])
    absorption = nodes.new("ShaderNodeVolumeAbsorption")
    absorption.label = "dye absorption"
    absorption.location = (-100, -50)
    links.new(albedo_socket, absorption.inputs["Color"])
    links.new(extinction.outputs[0], absorption.inputs["Density"])
    medium = nodes.new("ShaderNodeAddShader")
    medium.location = (150, 50)
    links.new(scatter.outputs[0], medium.inputs[0])
    links.new(absorption.outputs[0], medium.inputs[1])
    volume_socket = medium.outputs[0]

    # Optional self-illumination: emission proportional to extinction.
    self_illumination = float(material_spec.get("self_illumination", 0.0))
    if self_illumination > 0.0:
        emission_scale = nodes.new("ShaderNodeValue")
        emission_scale.name = DYE_EMISSION_SCALE_NODE
        emission_scale.label = "emission scale (set from exposure)"
        emission_scale.outputs[0].default_value = self_illumination
        emission_scale.location = (-600, -400)
        emission_strength = nodes.new("ShaderNodeMath")
        emission_strength.operation = "MULTIPLY"
        emission_strength.location = (-400, -350)
        links.new(extinction.outputs[0], emission_strength.inputs[0])
        links.new(emission_scale.outputs[0], emission_strength.inputs[1])
        emission = nodes.new("ShaderNodeEmission")
        emission.label = "dye self-illumination"
        emission.location = (-100, -300)
        links.new(albedo_socket, emission.inputs["Color"])
        links.new(emission_strength.outputs[0], emission.inputs["Strength"])
        with_emission = nodes.new("ShaderNodeAddShader")
        with_emission.location = (350, -50)
        links.new(volume_socket, with_emission.inputs[0])
        links.new(emission.outputs[0], with_emission.inputs[1])
        volume_socket = with_emission.outputs[0]
        # Picked up by apply_exposure_relative_emission.
        material[RELATIVE_EMISSION_PROPERTY] = self_illumination

    links.new(volume_socket, output_node.inputs["Volume"])
    return material


# ---------------------------------------------------------------------------
# Object import
# ---------------------------------------------------------------------------
def import_alembic_object(bundle_dir, object_spec, root_empty):
    """Import an Alembic mesh sequence and attach its material."""
    geometry = object_spec["geometry"]
    abc_path = bundle_dir / geometry["path"]

    before = set(bpy.context.scene.objects)
    # set_frame_range=False: don't let the importer overwrite the manifest-driven
    # frame range (configure_render already set it). Each Alembic archive would
    # otherwise reset the scene range to its own sample span on import.
    bpy.ops.wm.alembic_import(
        filepath=str(abc_path),
        as_background_job=False,
        validate_meshes=True,
        set_frame_range=False,
    )
    imported = [o for o in bpy.context.scene.objects if o not in before]
    if not imported:
        raise RuntimeError(f"Alembic import produced no object: {abc_path}")

    # Heterogeneous (changing vertex/face count) sequences must NOT vertex-
    # interpolate: the Mesh Sequence Cache modifier defaults to interpolation
    # on, which corrupts geometry between frames of differing vertex count (and
    # would garble the baked per-vertex colors). Turn it off for those; a
    # homogeneous sequence can keep interpolation for smoother sub-frames.
    is_heterogeneous = geometry.get("topology") == "heterogeneous"

    material = make_surface_material(object_spec["name"], object_spec["material"])
    for imported_object in imported:
        imported_object.name = object_spec["name"]
        parent_keep_local(imported_object, root_empty)
        imported_object.data.materials.clear()
        imported_object.data.materials.append(material)
        if is_heterogeneous:
            for modifier in imported_object.modifiers:
                if modifier.type == "MESH_SEQUENCE_CACHE" and hasattr(
                    modifier, "use_vertex_interpolation"
                ):
                    modifier.use_vertex_interpolation = False
    return imported


# Name of the per-point vector attribute that orients and sizes each glyph.
GLYPH_VECTOR_ATTRIBUTE = "glyph_vector"
# Polygon count of the arrow's round cross-section.
GLYPH_CROSS_SECTION_VERTICES = 20


def make_glyph_node_group(name, geometry_spec, material):
    """
    Build the geometry-nodes tree that turns a point cloud into vector glyphs.

    Each point carries a vector attribute (``glyph_vector``). The tree::

        unit arrow (shaft cylinder + head cone, length 1 along +Z, tail at 0)
          -> Set Material
        Instance on Points:
          rotation = Align Rotation to Vector(Z axis -> glyph_vector)
          scale    = length_per_unit * |glyph_vector|   (uniform)

    so every arrow starts at its point, points along the vector, and is
    ``length_per_unit`` data units long per unit of magnitude. The arrow is
    scaled uniformly, like PyVista's glyphs, so its proportions are the same
    for every arrow. Point attributes (the baked "col_data" color) carry over
    to the instances, where the material reads them as INSTANCER attributes.

    With ``width_mode == "constant"`` the arrow is instead built from two
    instance sets (see :func:`_add_constant_width_arrows`) so that shaft width
    and head size stay fixed and only the length follows the magnitude.
    """
    node_group = bpy.data.node_groups.new(f"{name}_glyphs", "GeometryNodeTree")
    node_group.interface.new_socket(
        name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
    )
    node_group.interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )
    nodes = node_group.nodes
    links = node_group.links
    group_input = nodes.new("NodeGroupInput")
    group_output = nodes.new("NodeGroupOutput")
    group_input.location = (-900, 0)
    group_output.location = (500, 0)

    head_length = float(geometry_spec.get("head_length_fraction", 0.28))
    head_radius = float(geometry_spec.get("head_radius_fraction", 0.08))
    shaft_radius = float(geometry_spec.get("shaft_radius_fraction", 0.03))
    glyph_type = geometry_spec.get("glyph_type", "arrow")

    if glyph_type == "sphere":
        # Particles: a native point cloud. Cycles ray-traces each point as an
        # exact sphere of the given radius; no sphere meshes are built. The
        # vertices' color attribute carries over to the points.
        to_points = nodes.new("GeometryNodeMeshToPoints")
        to_points.location = (-300, 0)
        to_points.mode = "VERTICES"
        to_points.inputs["Radius"].default_value = float(geometry_spec["radius"])
        links.new(group_input.outputs["Geometry"], to_points.inputs["Mesh"])
        set_material = nodes.new("GeometryNodeSetMaterial")
        set_material.location = (0, 0)
        set_material.inputs["Material"].default_value = material
        links.new(to_points.outputs["Points"], set_material.inputs["Geometry"])
        links.new(set_material.outputs["Geometry"], group_output.inputs["Geometry"])
        return node_group

    if geometry_spec.get("width_mode") == "constant":
        instances_socket = _add_constant_width_arrows(
            nodes, links, group_input.outputs["Geometry"], geometry_spec, material
        )
        links.new(instances_socket, group_output.inputs["Geometry"])
        return node_group

    def translated(geometry_socket, z_offset, location):
        """Move a primitive up by z_offset."""
        transform = nodes.new("GeometryNodeTransform")
        transform.location = location
        transform.inputs["Translation"].default_value = (0.0, 0.0, z_offset)
        links.new(geometry_socket, transform.inputs["Geometry"])
        return transform.outputs["Geometry"]

    # ---- The unit glyph, tail at the origin, tip at z = 1.
    if glyph_type == "cone":
        # A single cone spanning the whole length.
        cone = nodes.new("GeometryNodeMeshCone")
        cone.location = (-700, -300)
        cone.inputs["Vertices"].default_value = GLYPH_CROSS_SECTION_VERTICES
        cone.inputs["Radius Bottom"].default_value = head_radius
        cone.inputs["Radius Top"].default_value = 0.0
        cone.inputs["Depth"].default_value = 1.0
        # Mesh Cone already has its base at z = 0.
        glyph_geometry = cone.outputs["Mesh"]
    else:
        # Shaft: a cylinder from z = 0 to the base of the head. Mesh
        # Cylinder is centered on the origin, so it is moved up by half its
        # length; Mesh Cone already has its base at z = 0.
        shaft_length = 1.0 - head_length
        shaft = nodes.new("GeometryNodeMeshCylinder")
        shaft.location = (-700, -250)
        shaft.inputs["Vertices"].default_value = GLYPH_CROSS_SECTION_VERTICES
        shaft.inputs["Radius"].default_value = shaft_radius
        shaft.inputs["Depth"].default_value = shaft_length
        shaft_geometry = translated(
            shaft.outputs["Mesh"], shaft_length / 2, (-500, -250)
        )
        # Head: a cone from the end of the shaft to the tip.
        head = nodes.new("GeometryNodeMeshCone")
        head.location = (-700, -450)
        head.inputs["Vertices"].default_value = GLYPH_CROSS_SECTION_VERTICES
        head.inputs["Radius Bottom"].default_value = head_radius
        head.inputs["Radius Top"].default_value = 0.0
        head.inputs["Depth"].default_value = head_length
        head_geometry = translated(head.outputs["Mesh"], shaft_length, (-500, -450))
        join = nodes.new("GeometryNodeJoinGeometry")
        join.location = (-300, -350)
        links.new(shaft_geometry, join.inputs["Geometry"])
        links.new(head_geometry, join.inputs["Geometry"])
        glyph_geometry = join.outputs["Geometry"]

    set_material = nodes.new("GeometryNodeSetMaterial")
    set_material.location = (-100, -350)
    set_material.inputs["Material"].default_value = material
    links.new(glyph_geometry, set_material.inputs["Geometry"])

    # ---- Orientation and size from the per-point vector.
    vector_attribute = nodes.new("GeometryNodeInputNamedAttribute")
    vector_attribute.location = (-700, 200)
    vector_attribute.data_type = "FLOAT_VECTOR"
    vector_attribute.inputs["Name"].default_value = GLYPH_VECTOR_ATTRIBUTE
    align = nodes.new("FunctionNodeAlignRotationToVector")
    align.location = (-400, 250)
    align.axis = "Z"
    links.new(vector_attribute.outputs["Attribute"], align.inputs["Vector"])
    magnitude = nodes.new("ShaderNodeVectorMath")
    magnitude.location = (-400, 100)
    magnitude.operation = "LENGTH"
    links.new(vector_attribute.outputs["Attribute"], magnitude.inputs[0])
    length = nodes.new("ShaderNodeMath")
    length.location = (-200, 100)
    length.operation = "MULTIPLY"
    length.inputs[1].default_value = float(geometry_spec.get("length_per_unit", 1.0))
    links.new(magnitude.outputs["Value"], length.inputs[0])

    # ---- One glyph per point.
    instance = nodes.new("GeometryNodeInstanceOnPoints")
    instance.location = (200, 0)
    links.new(group_input.outputs["Geometry"], instance.inputs["Points"])
    links.new(set_material.outputs["Geometry"], instance.inputs["Instance"])
    links.new(align.outputs["Rotation"], instance.inputs["Rotation"])
    links.new(length.outputs["Value"], instance.inputs["Scale"])
    links.new(instance.outputs["Instances"], group_output.inputs["Geometry"])
    return node_group


def _add_constant_width_arrows(nodes, links, points_socket, geometry_spec, material):
    """
    Constant-width arrows: fixed shaft width and head size, variable length.

    One mesh scaled per instance cannot do this (stretching it would stretch
    the head too), so each arrow is two instances placed on the same point:

    * shaft: a unit cylinder (radius 1, z from 0 to 1) scaled to
      (shaft_radius, shaft_radius, shaft_length), where
      shaft_length = max(L - head_length, 0) and L = length_per_unit * |v|;
    * head: a fixed-size cone (base at 0, tip at head_length), moved from the
      point to the end of the shaft, and shrunk uniformly by min(L /
      head_length, 1) so an arrow shorter than a head is a small cone.

    Both are aligned with the vector by the same rotation. Returns the output
    socket of the joined instances.
    """
    shaft_radius = float(geometry_spec["shaft_radius"])
    head_radius = float(geometry_spec["head_radius"])
    head_length = float(geometry_spec["head_length"])

    def math_node(operation, location, constant=None):
        node = nodes.new("ShaderNodeMath")
        node.operation = operation
        node.location = location
        if constant is not None:
            node.inputs[1].default_value = constant
        return node

    # ---- Per-point quantities from the vector attribute.
    vector_attribute = nodes.new("GeometryNodeInputNamedAttribute")
    vector_attribute.location = (-900, 300)
    vector_attribute.data_type = "FLOAT_VECTOR"
    vector_attribute.inputs["Name"].default_value = GLYPH_VECTOR_ATTRIBUTE
    vector_socket = vector_attribute.outputs["Attribute"]
    align = nodes.new("FunctionNodeAlignRotationToVector")
    align.location = (-600, 450)
    align.axis = "Z"
    links.new(vector_socket, align.inputs["Vector"])
    direction = nodes.new("ShaderNodeVectorMath")
    direction.operation = "NORMALIZE"
    direction.location = (-600, 300)
    links.new(vector_socket, direction.inputs[0])
    magnitude = nodes.new("ShaderNodeVectorMath")
    magnitude.operation = "LENGTH"
    magnitude.location = (-600, 150)
    links.new(vector_socket, magnitude.inputs[0])
    # Arrow length L in data units.
    arrow_length = math_node(
        "MULTIPLY", (-400, 150), float(geometry_spec.get("length_per_unit", 1.0))
    )
    links.new(magnitude.outputs["Value"], arrow_length.inputs[0])
    # Shaft length: what is left after the head, never negative.
    shaft_length_raw = math_node("SUBTRACT", (-200, 150), head_length)
    links.new(arrow_length.outputs["Value"], shaft_length_raw.inputs[0])
    shaft_length = math_node("MAXIMUM", (0, 150), 0.0)
    links.new(shaft_length_raw.outputs["Value"], shaft_length.inputs[0])
    # Head shrink factor for arrows shorter than one head.
    head_fraction = math_node("DIVIDE", (-200, 0), head_length)
    links.new(arrow_length.outputs["Value"], head_fraction.inputs[0])
    head_scale = math_node("MINIMUM", (0, 0), 1.0)
    links.new(head_fraction.outputs["Value"], head_scale.inputs[0])

    # ---- Shaft instances.
    shaft = nodes.new("GeometryNodeMeshCylinder")
    shaft.location = (-600, -250)
    shaft.inputs["Vertices"].default_value = GLYPH_CROSS_SECTION_VERTICES
    shaft.inputs["Radius"].default_value = 1.0
    shaft.inputs["Depth"].default_value = 1.0
    shaft_base_at_origin = nodes.new("GeometryNodeTransform")
    shaft_base_at_origin.location = (-400, -250)
    shaft_base_at_origin.inputs["Translation"].default_value = (0.0, 0.0, 0.5)
    links.new(shaft.outputs["Mesh"], shaft_base_at_origin.inputs["Geometry"])
    shaft_material = nodes.new("GeometryNodeSetMaterial")
    shaft_material.location = (-200, -250)
    shaft_material.inputs["Material"].default_value = material
    links.new(
        shaft_base_at_origin.outputs["Geometry"], shaft_material.inputs["Geometry"]
    )
    shaft_scale = nodes.new("ShaderNodeCombineXYZ")
    shaft_scale.location = (200, 150)
    shaft_scale.inputs["X"].default_value = shaft_radius
    shaft_scale.inputs["Y"].default_value = shaft_radius
    links.new(shaft_length.outputs["Value"], shaft_scale.inputs["Z"])
    shaft_instances = nodes.new("GeometryNodeInstanceOnPoints")
    shaft_instances.location = (400, 100)
    links.new(points_socket, shaft_instances.inputs["Points"])
    links.new(shaft_material.outputs["Geometry"], shaft_instances.inputs["Instance"])
    links.new(align.outputs["Rotation"], shaft_instances.inputs["Rotation"])
    links.new(shaft_scale.outputs["Vector"], shaft_instances.inputs["Scale"])

    # ---- Head instances, at the end of each shaft.
    head = nodes.new("GeometryNodeMeshCone")
    head.location = (-600, -500)
    head.inputs["Vertices"].default_value = GLYPH_CROSS_SECTION_VERTICES
    head.inputs["Radius Bottom"].default_value = head_radius
    head.inputs["Radius Top"].default_value = 0.0
    head.inputs["Depth"].default_value = head_length
    # Mesh Cone already has its base at z = 0 (Mesh Cylinder is centered).
    head_material = nodes.new("GeometryNodeSetMaterial")
    head_material.location = (-200, -500)
    head_material.inputs["Material"].default_value = material
    links.new(head.outputs["Mesh"], head_material.inputs["Geometry"])
    shaft_offset = nodes.new("ShaderNodeVectorMath")
    shaft_offset.operation = "SCALE"
    shaft_offset.location = (200, -100)
    links.new(direction.outputs["Vector"], shaft_offset.inputs[0])
    links.new(shaft_length.outputs["Value"], shaft_offset.inputs["Scale"])
    head_points = nodes.new("GeometryNodeSetPosition")
    head_points.location = (400, -150)
    links.new(points_socket, head_points.inputs["Geometry"])
    links.new(shaft_offset.outputs["Vector"], head_points.inputs["Offset"])
    head_instances = nodes.new("GeometryNodeInstanceOnPoints")
    head_instances.location = (600, -200)
    links.new(head_points.outputs["Geometry"], head_instances.inputs["Points"])
    links.new(head_material.outputs["Geometry"], head_instances.inputs["Instance"])
    links.new(align.outputs["Rotation"], head_instances.inputs["Rotation"])
    links.new(head_scale.outputs["Value"], head_instances.inputs["Scale"])

    # ---- Both parts together.
    join = nodes.new("GeometryNodeJoinGeometry")
    join.location = (800, 0)
    links.new(shaft_instances.outputs["Instances"], join.inputs["Geometry"])
    links.new(head_instances.outputs["Instances"], join.inputs["Geometry"])
    return join.outputs["Geometry"]


def make_glyph_point_cloud(name, positions, vectors, colors_srgb):
    """
    A vertex-only mesh: one vertex per glyph, with its vector and color.

    ``vectors`` may be None (particles have no vector).

    Colors arrive as sRGB (matplotlib's output) and are linearized here,
    because Blender treats float color attributes as scene-linear.
    """
    import numpy as np

    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(len(positions))
    mesh.vertices.foreach_set("co", np.ascontiguousarray(positions, np.float32).ravel())
    if vectors is not None:
        vector_attribute = mesh.attributes.new(
            GLYPH_VECTOR_ATTRIBUTE, "FLOAT_VECTOR", "POINT"
        )
        vector_attribute.data.foreach_set(
            "vector", np.ascontiguousarray(vectors, np.float32).ravel()
        )
    # sRGB -> linear, then RGBA.
    colors_srgb = np.asarray(colors_srgb, dtype=np.float64)
    colors_linear = np.where(
        colors_srgb <= 0.04045,
        colors_srgb / 12.92,
        ((colors_srgb + 0.055) / 1.055) ** 2.4,
    )
    colors_rgba = np.concatenate(
        [colors_linear, np.ones((len(colors_linear), 1))], axis=1
    ).astype(np.float32)
    color_attribute = mesh.attributes.new("col_data", "FLOAT_COLOR", "POINT")
    color_attribute.data.foreach_set("color", colors_rgba.ravel())
    mesh.update()
    return mesh


def _show_only_on_frame(blender_object, frame, frame_start, frame_end):
    """Keyframe render/viewport visibility so the object shows on one frame."""
    for keyed_frame, hidden in (
        (frame - 1, True),
        (frame, False),
        (frame + 1, True),
    ):
        if keyed_frame < frame_start or keyed_frame > frame_end:
            continue
        blender_object.hide_render = hidden
        blender_object.hide_viewport = hidden
        blender_object.keyframe_insert(data_path="hide_render", frame=keyed_frame)
        blender_object.keyframe_insert(data_path="hide_viewport", frame=keyed_frame)
    # Visibility must switch, not blend, between keys.
    animation_data = blender_object.animation_data
    if animation_data and animation_data.action:
        for fcurve in _action_fcurves(animation_data.action):
            for keyframe_point in fcurve.keyframe_points:
                keyframe_point.interpolation = "CONSTANT"


def _action_fcurves(action):
    """F-curves of an action, across the layered-action API of Blender 4.4+."""
    if hasattr(action, "fcurves") and len(action.fcurves):
        return list(action.fcurves)
    fcurves = []
    for layer in getattr(action, "layers", []):
        for strip in layer.strips:
            for channelbag in getattr(strip, "channelbags", []):
                fcurves.extend(channelbag.fcurves)
    return fcurves


def import_point_glyphs(bundle_dir, object_spec, root_empty, frame_start):
    """
    Build instanced vector glyphs from a ``point_glyphs`` manifest entry.

    Each frame's ``.npz`` becomes one point-cloud object carrying the shared
    glyph geometry-nodes modifier; in an animation each object is visible on
    its own frame only. The glyph arrays are tiny, so one object per frame is
    simpler and more robust than swapping data from a frame-change handler
    (which a rendered ``.blend`` would need a registered script for).
    """
    import numpy as np

    geometry = object_spec["geometry"]
    name = object_spec["name"]
    n_frames = int(geometry.get("n_frames", 1))
    frame_end = frame_start + n_frames - 1
    material = make_surface_material(name, object_spec["material"])
    node_group = make_glyph_node_group(name, geometry, material)

    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    created = []
    for frame_offset in range(n_frames):
        frame = frame_start + frame_offset
        point_path = bundle_dir / geometry["path_pattern"].replace(
            "####", f"{frame_offset + 1:04d}"
        )
        with np.load(point_path) as point_file:
            mesh = make_glyph_point_cloud(
                f"{name}_{frame:04d}",
                point_file["positions"],
                point_file["vectors"] if "vectors" in point_file else None,
                point_file["colors"],
            )
        point_cloud = bpy.data.objects.new(f"{name}_{frame:04d}", mesh)
        collection.objects.link(point_cloud)
        parent_keep_local(point_cloud, root_empty)
        modifier = point_cloud.modifiers.new("glyphs", "NODES")
        modifier.node_group = node_group
        if n_frames > 1:
            _show_only_on_frame(point_cloud, frame, frame_start, frame_end)
        created.append(point_cloud)
    return created


def make_streamline_node_group(name, geometry_spec, material):
    """
    Geometry nodes that turn streamline curves into tubes with direction cones.

    The curves arrive ordered downstream, so each curve's tangent is the flow
    direction. The tree::

        tubes:  Curve to Mesh(profile = circle of tube_radius, caps filled)
        cones:  Curve to Points(every arrow_spacing along each curve)
                -> Instance on Points(cone, base at the point, tip along +Z;
                   rotation = Align Rotation to Vector(Z -> tangent))
                -> Realize Instances
        Join(tubes, cones) -> Set Material

    Point attributes (the baked "col_data" color) are interpolated onto the
    tube vertices and the cone points, and realized onto the cones, so one
    GEOMETRY attribute lookup in the material colors both.
    """
    node_group = bpy.data.node_groups.new(f"{name}_streamlines", "GeometryNodeTree")
    node_group.interface.new_socket(
        name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
    )
    node_group.interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )
    nodes = node_group.nodes
    links = node_group.links
    group_input = nodes.new("NodeGroupInput")
    group_input.location = (-900, 0)
    group_output = nodes.new("NodeGroupOutput")
    group_output.location = (700, 0)

    # ---- Tubes.
    profile = nodes.new("GeometryNodeCurvePrimitiveCircle")
    profile.location = (-650, -150)
    profile.inputs["Resolution"].default_value = GLYPH_CROSS_SECTION_VERTICES
    profile.inputs["Radius"].default_value = float(geometry_spec["tube_radius"])
    tubes = nodes.new("GeometryNodeCurveToMesh")
    tubes.location = (-400, 0)
    tubes.inputs["Fill Caps"].default_value = True
    links.new(group_input.outputs["Geometry"], tubes.inputs["Curve"])
    links.new(profile.outputs["Curve"], tubes.inputs["Profile Curve"])

    join = nodes.new("GeometryNodeJoinGeometry")
    join.location = (200, 0)
    links.new(tubes.outputs["Mesh"], join.inputs["Geometry"])

    # ---- Direction cones: moving with the flow clock, or fixed every
    # arrow_spacing along each curve.
    arrow_spacing = geometry_spec.get("arrow_spacing")
    arrow_travel_time = geometry_spec.get("arrow_travel_time")
    if arrow_travel_time:
        moving_markers = add_moving_streamline_markers(
            node_group,
            group_input.outputs["Geometry"],
            float(arrow_travel_time),
            geometry_spec,
        )
        links.new(moving_markers, join.inputs["Geometry"])
    elif arrow_spacing:
        marker_points = nodes.new("GeometryNodeCurveToPoints")
        marker_points.location = (-400, -300)
        marker_points.mode = "LENGTH"
        marker_points.inputs["Length"].default_value = float(arrow_spacing)
        links.new(group_input.outputs["Geometry"], marker_points.inputs["Curve"])
        align = nodes.new("FunctionNodeAlignRotationToVector")
        align.location = (-200, -450)
        align.axis = "Z"
        links.new(marker_points.outputs["Tangent"], align.inputs["Vector"])
        cone = add_streamline_cone(node_group, geometry_spec, (-400, -600))
        markers = nodes.new("GeometryNodeInstanceOnPoints")
        markers.location = (0, -350)
        links.new(marker_points.outputs["Points"], markers.inputs["Points"])
        links.new(cone, markers.inputs["Instance"])
        links.new(align.outputs["Rotation"], markers.inputs["Rotation"])
        realize = nodes.new("GeometryNodeRealizeInstances")
        realize.location = (100, -250)
        links.new(markers.outputs["Instances"], realize.inputs["Geometry"])
        links.new(realize.outputs["Geometry"], join.inputs["Geometry"])

    set_material = nodes.new("GeometryNodeSetMaterial")
    set_material.location = (450, 0)
    set_material.inputs["Material"].default_value = material
    links.new(join.outputs["Geometry"], set_material.inputs["Geometry"])
    links.new(set_material.outputs["Geometry"], group_output.inputs["Geometry"])
    return node_group


# Names of the travel-time attributes on streamline curves (written by
# make_polyline_curves) and of the flow-clock node in the streamline tree.
TRAVEL_TIME_ATTRIBUTE = "skyvista_travel_time"
TRAVEL_TIME_START_ATTRIBUTE = "skyvista_travel_time_start"
TRAVEL_TIME_END_ATTRIBUTE = "skyvista_travel_time_end"
FLOW_CLOCK_NODE = "skyvista_flow_clock"
# Moving markers grow over this fraction of the marker travel time after
# entering a streamline, and shrink over it before leaving, so none pops.
MARKER_END_FADE_FRACTION = 0.25


def add_streamline_cone(node_group, geometry_spec, location):
    """
    A direction cone, base at the origin and tip along +Z; returns its socket.

    The cone is ``arrow_head_length`` long with a base of
    ``arrow_head_radius``, so an instance placed on a streamline point with
    +Z turned along the tangent starts at that point and points downstream.
    Blender's Mesh Cone already has its base at z = 0 (unlike Mesh Cylinder,
    which is centered on the origin), so no offset is needed.
    """
    cone = node_group.nodes.new("GeometryNodeMeshCone")
    cone.location = location
    cone.inputs["Vertices"].default_value = GLYPH_CROSS_SECTION_VERTICES
    cone.inputs["Radius Bottom"].default_value = float(
        geometry_spec["arrow_head_radius"]
    )
    cone.inputs["Radius Top"].default_value = 0.0
    cone.inputs["Depth"].default_value = float(geometry_spec["arrow_head_length"])
    return cone.outputs["Mesh"]


class _NodeBuilder:
    """
    Small helpers for wiring math-heavy geometry-node graphs from Python.

    Each method creates one node, connects its inputs (an input may be a
    socket, which is linked, or a plain number, which is set as the default)
    and returns the output socket, so a graph reads like the formula it
    computes. Nodes are laid out left to right in creation order.
    """

    def __init__(self, node_group, origin):
        self.nodes = node_group.nodes
        self.links = node_group.links
        self.x, self.y = origin
        self.column = 0

    def _place(self, node):
        node.location = (self.x + 180 * self.column, self.y - 40 * (self.column % 6))
        self.column += 1
        return node

    def _connect(self, socket, value):
        if hasattr(value, "is_output"):
            self.links.new(value, socket)
        else:
            socket.default_value = value

    def math(self, operation, first, second=0.0):
        """Float math: ``operation(first, second)``."""
        node = self._place(self.nodes.new("ShaderNodeMath"))
        node.operation = operation
        self._connect(node.inputs[0], first)
        self._connect(node.inputs[1], second)
        return node.outputs[0]

    def named_attribute(self, name, data_type="FLOAT"):
        """A named attribute of the geometry being evaluated."""
        node = self._place(self.nodes.new("GeometryNodeInputNamedAttribute"))
        node.data_type = data_type
        node.inputs["Name"].default_value = name
        return node.outputs["Attribute"]

    def store(self, geometry, name, value, data_type="FLOAT", domain="POINT"):
        """Store a field as a named attribute; returns the geometry."""
        node = self._place(self.nodes.new("GeometryNodeStoreNamedAttribute"))
        node.data_type = data_type
        node.domain = domain
        self._connect(node.inputs["Geometry"], geometry)
        node.inputs["Name"].default_value = name
        self._connect(node.inputs["Value"], value)
        return node.outputs["Geometry"]

    def sample_index(self, geometry, value, index, data_type="FLOAT", domain="CURVE"):
        """``value`` of element ``index`` of ``geometry``'s ``domain``."""
        node = self._place(self.nodes.new("GeometryNodeSampleIndex"))
        node.data_type = data_type
        node.domain = domain
        self._connect(node.inputs["Geometry"], geometry)
        self._connect(node.inputs["Value"], value)
        self._connect(node.inputs["Index"], index)
        return node.outputs["Value"]

    def sample_curve_at_length(self, curves, value, curve_index, length, data_type):
        """``value`` interpolated at arc ``length`` along curve ``curve_index``."""
        node = self._place(self.nodes.new("GeometryNodeSampleCurve"))
        node.data_type = data_type
        node.mode = "LENGTH"
        node.use_all_curves = False
        self._connect(node.inputs["Curves"], curves)
        self._connect(node.inputs["Value"], value)
        self._connect(node.inputs["Curve Index"], curve_index)
        self._connect(node.inputs["Length"], length)
        return node.outputs["Value"]


def add_moving_streamline_markers(node_group, curves, travel_time_step, spec):
    """
    Direction cones that move along the streamlines with the air.

    The idea: a marker sits wherever the travel time from the seed (the
    ``skyvista_travel_time`` point attribute) equals ``clock + k * step`` for
    some integer k. As the flow clock advances, every marker moves downstream
    exactly as far as the air does in that time, so markers crowd and slow
    where the wind is weak and spread and speed up where it is strong.

    Placing points at given travel times needs the inverse of travel time
    along a curve, which no node computes directly. The trick is to
    *straighten* each curve: every point's real position and tangent are
    saved as attributes, then the point is moved to x = travel time. On the
    straightened curve, arc length *is* travel time, so ``Sample Curve`` at
    length ``t - t_start`` reads the saved position, tangent and color at
    travel time t, interpolated linearly between points.

    The markers come from a ``Points`` node with ``M`` points per curve, where
    ``M`` is enough for the longest curve (its travel-time span over the
    step, plus one). Point i belongs to curve ``i // M`` and is marker
    ``k = i % M`` on it. The first marker's time is the first time after the
    curve start that is in phase with the clock::

        first = t_start + ((clock - t_start) mod step)
        t_k = first + k * step

    Markers past the curve end are deleted. Each remaining marker gets a
    cone, turned along the saved tangent and scaled down to nothing within
    ``MARKER_END_FADE_FRACTION`` of a step of either curve end, so markers
    grow in at the upstream end and shrink away at the downstream end
    instead of popping.

    The clock is a Value node named ``FLOW_CLOCK_NODE`` (model seconds),
    keyframed per frame by ``key_flow_clock``.

    Returns:
        The realized cones' geometry socket.
    """
    nodes = node_group.nodes
    links = node_group.links
    build = _NodeBuilder(node_group, (-1800, -900))

    # The flow clock (s), keyframed per frame.
    clock_node = nodes.new("ShaderNodeValue")
    clock_node.name = FLOW_CLOCK_NODE
    clock_node.label = "Flow clock (s)"
    clock_node.location = (-2000, -700)
    clock_node.outputs[0].default_value = 0.0
    clock = clock_node.outputs[0]

    # ---- Save each point's real position and flow direction, then
    # straighten the curves so that arc length equals travel time.
    position = nodes.new("GeometryNodeInputPosition")
    position.location = (-2000, -900)
    tangent = nodes.new("GeometryNodeInputTangent")
    tangent.location = (-2000, -1000)
    saved = build.store(
        curves, "skyvista_position", position.outputs["Position"], "FLOAT_VECTOR"
    )
    saved = build.store(
        saved, "skyvista_tangent", tangent.outputs["Tangent"], "FLOAT_VECTOR"
    )
    travel_time = build.named_attribute(TRAVEL_TIME_ATTRIBUTE)
    straight_position = nodes.new("ShaderNodeCombineXYZ")
    straight_position.location = (-1400, -1100)
    links.new(travel_time, straight_position.inputs["X"])
    straighten = nodes.new("GeometryNodeSetPosition")
    straighten.location = (-1200, -1000)
    links.new(saved, straighten.inputs["Geometry"])
    links.new(straight_position.outputs["Vector"], straighten.inputs["Position"])
    straight_curves = straighten.outputs["Geometry"]

    # ---- Markers per curve: enough for the longest travel-time span.
    curve_count_node = nodes.new("GeometryNodeAttributeDomainSize")
    curve_count_node.component = "CURVE"
    curve_count_node.location = (-1000, -1300)
    links.new(straight_curves, curve_count_node.inputs["Geometry"])
    span = build.math(
        "SUBTRACT",
        build.named_attribute(TRAVEL_TIME_END_ATTRIBUTE),
        build.named_attribute(TRAVEL_TIME_START_ATTRIBUTE),
    )
    span_statistics = nodes.new("GeometryNodeAttributeStatistic")
    span_statistics.data_type = "FLOAT"
    span_statistics.domain = "CURVE"
    span_statistics.location = (-800, -1400)
    links.new(straight_curves, span_statistics.inputs["Geometry"])
    links.new(span, span_statistics.inputs["Attribute"])
    markers_per_curve = build.math(
        "ADD",
        build.math(
            "CEIL",
            build.math("DIVIDE", span_statistics.outputs["Max"], travel_time_step),
        ),
        1.0,
    )
    marker_count = build.math(
        "MULTIPLY", markers_per_curve, curve_count_node.outputs["Spline Count"]
    )
    points_node = nodes.new("GeometryNodePoints")
    points_node.location = (-200, -1300)
    links.new(marker_count, points_node.inputs["Count"])
    marker_points = points_node.outputs[0]

    # ---- Each marker's curve and travel time.
    index = nodes.new("GeometryNodeInputIndex")
    index.location = (-400, -1500)
    curve_index = build.math(
        "FLOOR", build.math("DIVIDE", index.outputs["Index"], markers_per_curve)
    )
    marker_number = build.math(
        "SUBTRACT",
        index.outputs["Index"],
        build.math("MULTIPLY", curve_index, markers_per_curve),
    )
    curve_start = build.sample_index(
        straight_curves,
        build.named_attribute(TRAVEL_TIME_START_ATTRIBUTE),
        curve_index,
    )
    curve_end = build.sample_index(
        straight_curves,
        build.named_attribute(TRAVEL_TIME_END_ATTRIBUTE),
        curve_index,
    )
    # Floored modulo: (clock - start) - step * floor((clock - start) / step).
    clock_since_start = build.math("SUBTRACT", clock, curve_start)
    phase = build.math(
        "SUBTRACT",
        clock_since_start,
        build.math(
            "MULTIPLY",
            build.math(
                "FLOOR", build.math("DIVIDE", clock_since_start, travel_time_step)
            ),
            travel_time_step,
        ),
    )
    marker_time = build.math(
        "ADD",
        build.math("ADD", curve_start, phase),
        build.math("MULTIPLY", marker_number, travel_time_step),
    )

    # Keep the per-marker values: indices shift once markers are deleted.
    marker_points = build.store(marker_points, "skyvista_curve", curve_index, "INT")
    marker_points = build.store(marker_points, "skyvista_marker_time", marker_time)
    marker_points = build.store(marker_points, "skyvista_curve_start", curve_start)
    marker_points = build.store(marker_points, "skyvista_curve_end", curve_end)

    # ---- Drop markers past their curve's downstream end.
    past_end = nodes.new("FunctionNodeCompare")
    past_end.data_type = "FLOAT"
    past_end.operation = "GREATER_THAN"
    past_end.location = (600, -1300)
    links.new(build.named_attribute("skyvista_marker_time"), past_end.inputs["A"])
    links.new(build.named_attribute("skyvista_curve_end"), past_end.inputs["B"])
    delete = nodes.new("GeometryNodeDeleteGeometry")
    delete.domain = "POINT"
    delete.location = (800, -1200)
    links.new(marker_points, delete.inputs["Geometry"])
    links.new(past_end.outputs["Result"], delete.inputs["Selection"])
    kept_markers = delete.outputs["Geometry"]

    # ---- Read position, direction and color at each marker's travel time.
    stored_curve = build.named_attribute("skyvista_curve", "INT")
    stored_time = build.named_attribute("skyvista_marker_time")
    stored_start = build.named_attribute("skyvista_curve_start")
    stored_end = build.named_attribute("skyvista_curve_end")
    length_along = build.math("SUBTRACT", stored_time, stored_start)
    marker_position = build.sample_curve_at_length(
        straight_curves,
        build.named_attribute("skyvista_position", "FLOAT_VECTOR"),
        stored_curve,
        length_along,
        "FLOAT_VECTOR",
    )
    marker_tangent = build.sample_curve_at_length(
        straight_curves,
        build.named_attribute("skyvista_tangent", "FLOAT_VECTOR"),
        stored_curve,
        length_along,
        "FLOAT_VECTOR",
    )
    marker_color = build.sample_curve_at_length(
        straight_curves,
        build.named_attribute("col_data", "FLOAT_COLOR"),
        stored_curve,
        length_along,
        "FLOAT_COLOR",
    )
    placed = nodes.new("GeometryNodeSetPosition")
    placed.location = (1400, -1200)
    links.new(kept_markers, placed.inputs["Geometry"])
    links.new(marker_position, placed.inputs["Position"])
    colored = build.store(
        placed.outputs["Geometry"], "col_data", marker_color, "FLOAT_COLOR"
    )

    # ---- Size: full inside the curve, shrinking to zero at either end.
    fade_time = MARKER_END_FADE_FRACTION * travel_time_step
    distance_to_nearest_end = build.math(
        "MINIMUM", length_along, build.math("SUBTRACT", stored_end, stored_time)
    )
    marker_scale = build.math(
        "MINIMUM",
        build.math("MAXIMUM", build.math("DIVIDE", distance_to_nearest_end, fade_time)),
        1.0,
    )

    # ---- One cone per marker, turned downstream.
    align = nodes.new("FunctionNodeAlignRotationToVector")
    align.axis = "Z"
    align.location = (1600, -1400)
    links.new(marker_tangent, align.inputs["Vector"])
    cone = add_streamline_cone(node_group, spec, (1400, -1600))
    instances = nodes.new("GeometryNodeInstanceOnPoints")
    instances.location = (1900, -1200)
    links.new(colored, instances.inputs["Points"])
    links.new(cone, instances.inputs["Instance"])
    links.new(align.outputs["Rotation"], instances.inputs["Rotation"])
    links.new(marker_scale, instances.inputs["Scale"])
    realize = nodes.new("GeometryNodeRealizeInstances")
    realize.location = (2100, -1200)
    links.new(instances.outputs["Instances"], realize.inputs["Geometry"])
    return realize.outputs["Geometry"]


def key_flow_clock(node_group, flow_clock):
    """
    Keyframe the streamline tree's flow clock from the manifest.

    ``flow_clock`` is the manifest's ``{"keyframes": [{"frame", "seconds"}]}``
    (one per video frame). Interpolation is linear, so the clock also runs
    smoothly between keys if the frame rate is raised in Blender. Without
    keyframes the clock stays at zero and the markers stand still.
    """
    clock_node = node_group.nodes.get(FLOW_CLOCK_NODE)
    if clock_node is None or not flow_clock or not flow_clock.get("keyframes"):
        return
    clock_socket = clock_node.outputs[0]
    for keyframe in flow_clock["keyframes"]:
        clock_socket.default_value = float(keyframe["seconds"])
        clock_socket.keyframe_insert("default_value", frame=int(keyframe["frame"]))
    animation_data = node_group.animation_data
    if animation_data and animation_data.action:
        for fcurve in _action_fcurves(animation_data.action):
            for keyframe_point in fcurve.keyframe_points:
                keyframe_point.interpolation = "LINEAR"


def make_polyline_curves(name, positions, curve_sizes, colors_srgb, travel_times=None):
    """
    A Curves datablock with one poly curve per streamline and its colors.

    Colors arrive as sRGB and are linearized, as for glyph points. Travel
    times (s from each streamline's seed), when given, become the point
    attribute TRAVEL_TIME_ATTRIBUTE, and each curve's first and last travel
    time become the curve attributes TRAVEL_TIME_START_ATTRIBUTE and
    TRAVEL_TIME_END_ATTRIBUTE, which the moving markers read.
    """
    import numpy as np

    curves = bpy.data.hair_curves.new(name)
    curves.add_curves([int(size) for size in curve_sizes])
    curves.attributes["position"].data.foreach_set(
        "vector", np.ascontiguousarray(positions, np.float32).ravel()
    )
    colors_srgb = np.asarray(colors_srgb, dtype=np.float64)
    colors_linear = np.where(
        colors_srgb <= 0.04045,
        colors_srgb / 12.92,
        ((colors_srgb + 0.055) / 1.055) ** 2.4,
    )
    colors_rgba = np.concatenate(
        [colors_linear, np.ones((len(colors_linear), 1))], axis=1
    ).astype(np.float32)
    color_attribute = curves.attributes.new("col_data", "FLOAT_COLOR", "POINT")
    color_attribute.data.foreach_set("color", colors_rgba.ravel())

    if travel_times is not None:
        travel_times = np.asarray(travel_times, dtype=np.float32)
        curves.attributes.new(TRAVEL_TIME_ATTRIBUTE, "FLOAT", "POINT").data.foreach_set(
            "value", travel_times
        )
        # Index of each curve's first and last point in the flat point arrays.
        curve_ends = np.cumsum(np.asarray(curve_sizes, dtype=np.int64))
        curve_starts = curve_ends - np.asarray(curve_sizes, dtype=np.int64)
        for attribute_name, point_indices in (
            (TRAVEL_TIME_START_ATTRIBUTE, curve_starts),
            (TRAVEL_TIME_END_ATTRIBUTE, curve_ends - 1),
        ):
            curves.attributes.new(attribute_name, "FLOAT", "CURVE").data.foreach_set(
                "value", np.ascontiguousarray(travel_times[point_indices])
            )
    return curves


def import_polylines(bundle_dir, object_spec, root_empty, frame_start, flow_clock=None):
    """
    Build streamline tubes from a ``polylines`` manifest entry.

    Each frame's file becomes one Curves object carrying the shared
    streamline geometry-nodes modifier, visible on its own frame only (as for
    glyph point clouds). The tree's flow clock, which moves markers with
    ``arrow_travel_time``, is keyed from the manifest's ``flow_clock``.
    """
    import numpy as np

    geometry = object_spec["geometry"]
    name = object_spec["name"]
    n_frames = int(geometry.get("n_frames", 1))
    frame_end = frame_start + n_frames - 1
    material = make_surface_material(name, object_spec["material"])
    node_group = make_streamline_node_group(name, geometry, material)
    key_flow_clock(node_group, flow_clock)

    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    created = []
    for frame_offset in range(n_frames):
        frame = frame_start + frame_offset
        polyline_path = bundle_dir / geometry["path_pattern"].replace(
            "####", f"{frame_offset + 1:04d}"
        )
        with np.load(polyline_path) as polyline_file:
            if len(polyline_file["curve_sizes"]) == 0:
                continue
            curves = make_polyline_curves(
                f"{name}_{frame:04d}",
                polyline_file["positions"],
                polyline_file["curve_sizes"],
                polyline_file["colors"],
                (
                    polyline_file["travel_times"]
                    if "travel_times" in polyline_file.files
                    else None
                ),
            )
        curves_object = bpy.data.objects.new(f"{name}_{frame:04d}", curves)
        collection.objects.link(curves_object)
        parent_keep_local(curves_object, root_empty)
        modifier = curves_object.modifiers.new("streamlines", "NODES")
        modifier.node_group = node_group
        if n_frames > 1:
            _show_only_on_frame(curves_object, frame, frame_start, frame_end)
        created.append(curves_object)
    return created


def import_volume_object(bundle_dir, object_spec, root_empty, frame_start, n_frames):
    """Import a VDB sequence as a Volume object and attach its material."""
    geometry = object_spec["geometry"]
    # Reconstruct the first file from the '####' pattern.
    pattern = geometry["path_pattern"]
    first_file = bundle_dir / pattern.replace("####", f"{frame_start:04d}")
    if not first_file.exists():
        # Fall back to the first matching file on disk.
        matches = sorted((bundle_dir / Path(pattern).parent).glob("*.vdb"))
        if not matches:
            raise RuntimeError(f"No VDB files for pattern {pattern}")
        first_file = matches[0]

    before = set(bpy.context.scene.objects)
    bpy.ops.object.volume_import(filepath=str(first_file), use_sequence_detection=True)
    imported = [o for o in bpy.context.scene.objects if o not in before]
    if not imported:
        raise RuntimeError(f"Volume import produced no object: {first_file}")
    volume_object = imported[0]
    volume_object.name = object_spec["name"]

    # Play the files as a sequence over the animation range. A single file
    # (a still, or one data time held over many video frames) is shown on
    # every frame instead: as a one-frame sequence it would be hidden after
    # its frame.
    volume_data = volume_object.data
    n_files = int(geometry.get("n_frames", n_frames))
    if hasattr(volume_data, "is_sequence"):
        volume_data.is_sequence = n_files > 1
        if n_files > 1:
            try:
                volume_data.frame_start = frame_start
                volume_data.frame_duration = n_files
            except AttributeError:
                pass

    parent_keep_local(volume_object, root_empty)
    material_spec = object_spec["material"]
    material_type = material_spec.get("type")
    if material_type in ("cloud", "dye"):
        material = (
            make_cloud_material(object_spec["name"], material_spec)
            if material_type == "cloud"
            else make_dye_material(object_spec["name"], material_spec, bundle_dir)
        )
        render_settings = volume_data.render
        # Density per object unit (= per meter of the VDB transform), so the
        # optical depth is physical for any uniform scene scale.
        if hasattr(render_settings, "space"):
            render_settings.space = "OBJECT"
        # Grid values below the export's floor are clear air.
        if hasattr(render_settings, "clipping"):
            render_settings.clipping = float(material_spec.get("clipping", 0.0))
    else:
        material = make_volume_material(object_spec["name"], material_spec, bundle_dir)
    volume_object.data.materials.clear()
    volume_object.data.materials.append(material)
    return [volume_object]


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
def aim_camera(camera_object, location, look_at):
    """Position a camera and rotate it to look at a target point."""
    camera_object.location = location
    direction = Vector(look_at) - Vector(location)
    if direction.length > 0:
        camera_object.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_camera(scene, manifest, transform, all_objects):
    """Create a camera from the manifest, or a default 3/4 view of the data."""
    camera_data = bpy.data.cameras.new("skyvista_camera")
    camera_object = bpy.data.objects.new("skyvista_camera", camera_data)
    scene.collection.objects.link(camera_object)
    scene.camera = camera_object

    camera_spec = manifest.get("camera")
    if camera_spec and camera_spec.get("keyframes"):
        camera_data.lens = float(camera_spec.get("lens_mm", 50))
        for keyframe in camera_spec["keyframes"]:
            frame = int(keyframe["frame"])
            location = transform_point(transform, keyframe["location"])
            look_at = transform_point(transform, keyframe.get("look_at", [0, 0, 0]))
            aim_camera(camera_object, location, look_at)
            camera_object.keyframe_insert(data_path="location", frame=frame)
            camera_object.keyframe_insert(data_path="rotation_euler", frame=frame)
            # A camera path may also animate the focal length.
            if "lens_mm" in keyframe:
                camera_data.lens = float(keyframe["lens_mm"])
                camera_data.keyframe_insert(data_path="lens", frame=frame)
        return camera_object

    # Default: frame the combined bounding box of all imported world geometry.
    minimum = Vector((1e30, 1e30, 1e30))
    maximum = Vector((-1e30, -1e30, -1e30))
    for obj in all_objects:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            minimum = Vector(map(min, minimum, world_corner))
            maximum = Vector(map(max, maximum, world_corner))
    center = (minimum + maximum) / 2.0
    span = (maximum - minimum).length or 1.0
    location = center + Vector((0.9, -1.3, 0.7)) * span
    aim_camera(camera_object, location, center)
    return camera_object


# ---------------------------------------------------------------------------
# World / lighting
# ---------------------------------------------------------------------------
# Flat (non-sky) presets: the World background is a single constant color and
# the object is lit by the key sun plus a little ambient from this color.
FLAT_WORLD_COLORS = {
    "studio": (0.20, 0.20, 0.21, 1.0),  # neutral mid-gray, product-shot look
    "dark": (0.02, 0.02, 0.03, 1.0),  # near-black, flatters glowing volumes
    "white": (1.0, 1.0, 1.0, 1.0),  # clean print-figure background
}

# How much each preset's background contributes to *lighting*, as a fraction of
# its camera-visible strength. A physically bright environment (the sky, or a
# white backdrop) otherwise blows out every upward/outward-facing surface and
# washes the baked scientific colors to white; damping only the lighting rays
# (see setup_world's Is-Camera-Ray mix) keeps the background bright to the camera
# while letting the subject's color read. Values calibrated empirically in
# Blender 5.0.1 against a mid-value surface. Presets whose background is already
# dark ("dark", "studio") keep the full contribution (1.0).
WORLD_LIGHT_FACTORS = {
    "sky": 0.30,
    "physical_sky": 1.0,
    "white": 0.15,
    "studio": 1.0,
    "dark": 1.0,
}

# Radiance of a white Lambertian horizontal plane lit by the physical sky
# (sun disc on, default air/aerosol/ozone densities and altitude, sun
# intensity 1), as (sun elevation in degrees, linear radiance). Measured by
# rendering such a plane in Cycles, Blender 5.0.1, MULTIPLE_SCATTERING sky.
# The sky's diffuse share of this is 16% at 35 deg, close to clear-sky values.
PHYSICAL_SKY_WHITE_PLANE_RADIANCE = (
    (5.0, 2.707),
    (10.0, 6.484),
    (20.0, 14.934),
    (35.0, 27.114),
    (50.0, 37.391),
    (70.0, 46.711),
    (90.0, 50.052),
)

# Automatic exposure maps that white plane to this linear value: 80% of
# display white, leaving headroom for forward-scattering cloud edges.
AUTO_EXPOSURE_WHITE_TARGET = 0.8


def physical_sky_white_plane_radiance(sun_elevation_deg):
    """Interpolate PHYSICAL_SKY_WHITE_PLANE_RADIANCE (clamped to its range)."""
    table = PHYSICAL_SKY_WHITE_PLANE_RADIANCE
    elevation = min(max(float(sun_elevation_deg), table[0][0]), table[-1][0])
    for (low_elevation, low_value), (high_elevation, high_value) in zip(
        table[:-1], table[1:], strict=True
    ):
        if elevation <= high_elevation:
            fraction = (elevation - low_elevation) / (high_elevation - low_elevation)
            return low_value + fraction * (high_value - low_value)
    return table[-1][1]


def auto_exposure_ev(sun_elevation_deg, background_strength):
    """
    Exposure (EV) a camera metering a white card in full sun would choose.

    The physical sky is radiometrically scaled, so a sunlit white surface has
    a linear radiance of ~27 at 35 deg sun elevation, far above display white.
    We expose so that a white Lambertian surface lying flat in full sun and
    skylight renders at AUTO_EXPOSURE_WHITE_TARGET. Everything else keeps its
    physical brightness relative to that card.
    """
    white_radiance = physical_sky_white_plane_radiance(sun_elevation_deg) * max(
        float(background_strength), 1.0e-6
    )
    return math.log2(AUTO_EXPOSURE_WHITE_TARGET / white_radiance)


def aim_sun(sun_object, sun_elevation_deg, sun_azimuth_deg):
    """
    Orient a Sun lamp so the sun sits at (elevation, azimuth) in the sky.

    Azimuth is measured CCW from +x about +z, toward the sun. A Blender Sun
    emits along its local -Z axis. Tilting about X by (90 - elevation) and
    then rotating about Z by psi makes the light *come from* the direction
    (sin psi cos e, -cos psi cos e, sin e). That equals the requested
    (cos A cos e, sin A cos e, sin e) when psi = A + 90 deg. Euler order 'XYZ'
    applies the X tilt first, then the Z rotation.
    """
    elevation = math.radians(sun_elevation_deg)
    lamp_rotation_about_z = math.radians(sun_azimuth_deg) + math.pi / 2.0
    sun_object.rotation_euler = (math.pi / 2.0 - elevation, 0.0, lamp_rotation_about_z)


def make_sky_texture(
    nodes, sun_elevation_deg, sun_azimuth_deg, sun_disc=False, altitude_m=0.0
):
    """
    Create a physically-based Sky Texture node aimed at the given sun direction.

    The sky-model enum was renamed across Blender versions: 4.x exposes
    "NISHITA", while 5.0 replaced it with "MULTIPLE_SCATTERING" (same underlying
    model, same sun_* attributes). Pick the best physically-based model that is
    actually available so this works on either. The Nishita parameters are node
    attributes (not input sockets); set them defensively so a renamed attribute
    in some Blender version can't abort the whole build.

    By default the sky's own sun disc is disabled: the crisp key light comes
    from the explicit, matched Sun lamp, while the sky still contributes
    realistic blue skylight. With ``sun_disc=True`` (the "physical_sky"
    preset) the sky's sun is the only sun, which keeps the sun/sky radiance
    ratio of the physical model. Cycles importance-samples that disc directly.

    Azimuth convention: the sky's sun lies toward
    (sin r cos e, cos r cos e, sin e) for rotation r, so r = 0 puts it toward
    +y and r increases clockwise seen from above. This was established by
    rendering a pole's shadow at four azimuths. Our azimuth A is measured CCW
    from +x toward the sun, so r = 90 deg - A (wrapped into [0, 360)).

    ``altitude_m`` places the sky model's observer above the ground. With
    aerial perspective on, it is the top of the haze box: the sky model then
    supplies only the atmosphere above the box, and the box supplies the air
    inside it, instead of both supplying the low atmosphere.
    """
    sky = nodes.new("ShaderNodeTexSky")
    available_sky_types = {
        item.identifier for item in sky.bl_rna.properties["sky_type"].enum_items
    }
    for candidate in ("MULTIPLE_SCATTERING", "NISHITA", "HOSEK_WILKIE"):
        if candidate in available_sky_types:
            sky.sky_type = candidate
            break
    if hasattr(sky, "sun_elevation"):
        sky.sun_elevation = math.radians(sun_elevation_deg)
    if hasattr(sky, "sun_rotation"):
        sky.sun_rotation = math.radians((90.0 - sun_azimuth_deg) % 360.0)
    if hasattr(sky, "sun_disc"):
        sky.sun_disc = sun_disc
    if altitude_m and hasattr(sky, "altitude"):
        sky.altitude = float(altitude_m)
    return sky


def haze_box_top_altitude_m(manifest):
    """Height (m above the ground) of the haze box top, or 0 with no haze."""
    world_block = manifest.get("world", {})
    if not world_block.get("aerial_perspective"):
        return 0.0
    transform = manifest.get("transform", {})
    box_bounds = _haze_box_bounds(manifest, transform)
    if box_bounds is None:
        return 0.0
    scale = float(transform.get("scale", 1.0e-3))
    z_exaggeration = float(transform.get("z_exaggeration", 1.0))
    origin_z = float((transform.get("origin_shift") or [0.0, 0.0, 0.0])[2])
    top_data_z = box_bounds[1].z / (scale * z_exaggeration) + origin_z
    return max(top_data_z - float(world_block.get("ground_height", 0.0)), 0.0)


def setup_world(scene, manifest):
    """
    Build the world background + key light from the manifest's ``world`` block.

    Presets:
      - "sky":    Nishita physical daytime sky; the key sun is aligned to the
                  sky's sun direction, and the sky's own sun disc is disabled so
                  the crisp shadows come from the (single) Sun lamp while the sky
                  still provides realistic blue ambient light.
      - "physical_sky": the same sky with its own sun disc as the only sun and
                  undamped sky light, so sun and sky keep the physical model's
                  radiance ratio; exposure is set by ``configure_exposure``.
      - "studio": neutral mid-gray environment + soft key sun.
      - "dark":   near-black background + sun (flatters glowing volumes).
      - "white":  pure white background + sun (clean print look).
    Unknown preset names fall back to "studio".

    The background is wired so its *camera-visible* brightness and its *lighting*
    contribution can differ: a Light Path "Is Camera Ray" node mixes a
    full-strength background (what the camera sees) with a damped one (what
    illuminates the scene). This keeps a bright, pretty sky/backdrop while
    preventing it from washing the baked scientific colors to white -- the
    over-exposure that a single bright environment otherwise causes.
    """
    # Read the world block with defaults matching BlenderWorldConfig.
    world_block = manifest.get("world", {})
    preset = world_block.get("preset", "studio")
    sun_elevation_deg = world_block.get("sun_elevation_deg", 35.0)
    sun_azimuth_deg = world_block.get("sun_azimuth_deg", 40.0)
    sun_strength = world_block.get("sun_strength", 2.0)
    background_strength = world_block.get("background_strength", 1.0)

    # Fraction of the background strength that reaches lighting rays (see
    # WORLD_LIGHT_FACTORS). Camera rays always see the full background_strength.
    light_factor = WORLD_LIGHT_FACTORS.get(preset, 1.0)

    # Fresh world; rebuild the node tree from scratch so the mix rig below is
    # deterministic regardless of the default nodes a new world ships with.
    world = bpy.data.worlds.new("skyvista_world")
    scene.world = world
    world.use_nodes = True
    node_tree = world.node_tree
    nodes = node_tree.nodes
    links = node_tree.links
    nodes.clear()

    world_output = nodes.new("ShaderNodeOutputWorld")

    # Background color source: physical sky for the sky presets, else a
    # constant color. Only "physical_sky" keeps the sky's own sun disc.
    if preset in ("sky", "physical_sky"):
        sky = make_sky_texture(
            nodes,
            sun_elevation_deg,
            sun_azimuth_deg,
            sun_disc=(preset == "physical_sky"),
            altitude_m=haze_box_top_altitude_m(manifest),
        )
        color_output = sky.outputs["Color"]
        constant_color = None
    else:
        color_output = None
        constant_color = FLAT_WORLD_COLORS.get(preset, FLAT_WORLD_COLORS["studio"])

    # Two Background shaders sharing the same color: one seen by the camera at
    # full strength, one that lights the scene at a damped strength.
    background_seen_by_camera = nodes.new("ShaderNodeBackground")
    background_that_lights = nodes.new("ShaderNodeBackground")
    background_seen_by_camera.inputs["Strength"].default_value = background_strength
    background_that_lights.inputs["Strength"].default_value = (
        background_strength * light_factor
    )
    if color_output is not None:
        links.new(color_output, background_seen_by_camera.inputs["Color"])
        links.new(color_output, background_that_lights.inputs["Color"])
    else:
        background_seen_by_camera.inputs["Color"].default_value = constant_color
        background_that_lights.inputs["Color"].default_value = constant_color

    # Mix by "Is Camera Ray": camera rays (fac=1) take the full background;
    # lighting/indirect rays (fac=0) take the damped one. Mix Shader input[1] is
    # the fac=0 shader and input[2] is the fac=1 shader.
    light_path = nodes.new("ShaderNodeLightPath")
    mix_shader = nodes.new("ShaderNodeMixShader")
    links.new(light_path.outputs["Is Camera Ray"], mix_shader.inputs["Fac"])
    links.new(background_that_lights.outputs["Background"], mix_shader.inputs[1])
    links.new(background_seen_by_camera.outputs["Background"], mix_shader.inputs[2])
    links.new(mix_shader.outputs["Shader"], world_output.inputs["Surface"])

    # The physical sky's sun disc *is* the sun; a lamp would double it.
    if preset == "physical_sky":
        return

    # Key sun: one directional lamp, aimed from the configured sky direction so
    # shadows are consistent regardless of which world preset is active.
    sun_data = bpy.data.lights.new("skyvista_sun", type="SUN")
    sun_data.energy = sun_strength
    sun_object = bpy.data.objects.new("skyvista_sun", sun_data)
    aim_sun(sun_object, sun_elevation_deg, sun_azimuth_deg)
    scene.collection.objects.link(sun_object)


def setup_ground(scene, manifest, transform, root_empty):
    """
    Add a matte, spectrally neutral ground plane if the manifest asks for one.

    The plane sits at ``world.ground_height`` (data units) under the scene
    transform and extends 10,000 km in every direction. Its edge must lie
    below the sky texture's horizon: seen from height h at distance L it sits
    atan(h / L) below horizontal, and between that edge and the horizon the
    sky's dark below-horizon region would show as a stripe. At 10,000 km the
    edge is within 0.15 deg of the horizon even from 25 km up. It is parented to the root Empty like the
    data, so it follows any transform edit. A Diffuse BSDF is exactly
    Lambertian, so the albedo means what it says: the fraction of light
    reflected, which then brightens cloud bases from below.
    """
    world_block = manifest.get("world", {})
    ground_albedo = world_block.get("ground_albedo")
    if ground_albedo is None:
        return None

    # A square plane in data units (meters), parented under the root Empty.
    half_width_data_units = 1.0e7
    ground_height = float(world_block.get("ground_height", 0.0))
    mesh = bpy.data.meshes.new("skyvista_ground")
    corners = [
        (-half_width_data_units, -half_width_data_units, ground_height),
        (half_width_data_units, -half_width_data_units, ground_height),
        (half_width_data_units, half_width_data_units, ground_height),
        (-half_width_data_units, half_width_data_units, ground_height),
    ]
    mesh.from_pydata(corners, [], [(0, 1, 2, 3)])
    ground_object = bpy.data.objects.new("skyvista_ground", mesh)
    scene.collection.objects.link(ground_object)
    # Keep the ground in data units: apply the root transform, minus any
    # horizontal origin shift (the plane is centered on the data anyway).
    parent_keep_local(ground_object, root_empty)
    origin_shift = transform.get("origin_shift") or [0.0, 0.0, 0.0]
    ground_object.location = (origin_shift[0], origin_shift[1], 0.0)

    # Lambertian, grey material with the requested albedo (linear).
    material = bpy.data.materials.new("skyvista_ground_mat")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    for node in list(nodes):
        if node.type != "OUTPUT_MATERIAL":
            nodes.remove(node)
    diffuse = nodes.new("ShaderNodeBsdfDiffuse")
    albedo = float(ground_albedo)
    diffuse.inputs["Color"].default_value = (albedo, albedo, albedo, 1.0)
    material.node_tree.links.new(
        diffuse.outputs[0], nodes["Material Output"].inputs["Surface"]
    )
    mesh.materials.append(material)

    # Let the camera see the ground out to its edge, not just the default
    # 1000-unit clip distance (the plane is 20,000 km across).
    if scene.camera is not None:
        plane_reach = (
            2.0 * half_width_data_units * float(transform.get("scale", 1.0e-3))
        )
        scene.camera.data.clip_end = max(scene.camera.data.clip_end, plane_reach)
    return ground_object


# Molecular (Rayleigh) scattering coefficients of air at sea level, m^-1, for
# the red / green / blue channels taken as 680 / 550 / 440 nm (Bruneton &
# Neyret 2008), and the e-folding height of air density.
RAYLEIGH_SCATTERING_SEA_LEVEL_RGB = (5.8e-6, 13.5e-6, 33.1e-6)
RAYLEIGH_SCALE_HEIGHT_M = 8000.0
# Koschmieder: visibility V (m) where contrast falls to 2%, beta = -ln(0.02)/V.
KOSCHMIEDER_CONSTANT = 3.912
# Haze box margin beyond the data and camera, as a fraction of its span.
HAZE_BOX_MARGIN_FRACTION = 0.1
# Minimum horizontal half-width of the haze box (m). Wide enough that sky rays
# leave through the top, where the sky model (observer at the box top) takes
# over; rays leaving through the sides would miss the low atmosphere beyond.
HAZE_BOX_MIN_HALF_WIDTH_M = 300000.0


def _haze_box_bounds(manifest, transform):
    """
    Blender-space (min, max) corners of the box the haze fills.

    The box encloses the data bounds and every camera position, plus a margin,
    from the ground to above the higher of the data top and the camera, and
    is at least HAZE_BOX_MIN_HALF_WIDTH_M wide each way. The physical sky is
    computed for an observer at the box top (see ``haze_box_top_altitude_m``),
    so the box carries the air below that height and the sky model the air
    above it. The width makes nearly every sky ray leave through the top,
    where that split is exact.
    """
    data_bounds = manifest.get("bounds")
    camera_keyframes = (manifest.get("camera") or {}).get("keyframes", [])
    corner_points = []
    if data_bounds:
        corner_points += [transform_point(transform, corner) for corner in data_bounds]
    corner_points += [
        transform_point(transform, keyframe["location"])
        for keyframe in camera_keyframes
    ]
    if not corner_points:
        return None
    minimum = Vector([min(point[axis] for point in corner_points) for axis in range(3)])
    maximum = Vector([max(point[axis] for point in corner_points) for axis in range(3)])
    margin = (maximum - minimum) * HAZE_BOX_MARGIN_FRACTION
    ground = transform_point(
        transform,
        (0.0, 0.0, float(manifest.get("world", {}).get("ground_height", 0.0))),
    )
    minimum = Vector((minimum.x - margin.x, minimum.y - margin.y, ground.z))
    maximum = Vector((maximum.x + margin.x, maximum.y + margin.y, maximum.z + margin.z))
    # Widen horizontally around the centre to the minimum half-width.
    min_half_width = HAZE_BOX_MIN_HALF_WIDTH_M * float(transform.get("scale", 1.0e-3))
    for axis in (0, 1):
        center = (minimum[axis] + maximum[axis]) / 2
        half_width = max((maximum[axis] - minimum[axis]) / 2, min_half_width)
        minimum[axis] = center - half_width
        maximum[axis] = center + half_width
    return minimum, maximum


def setup_aerial_perspective(scene, manifest, transform):
    """
    Fill the clear air with physical haze: Rayleigh molecules plus aerosol.

    A box mesh around the data and camera carries a volume shader whose
    coefficients decay exponentially with height above the ground::

        molecules: Volume Scatter(RAYLEIGH phase),
                   sigma_rgb = RAYLEIGH_SCATTERING_SEA_LEVEL_RGB * exp(-h / 8 km)
        aerosol:   Volume Scatter(HG, g) + Volume Absorption,
                   beta = (3.912 / V - sigma_550) * exp(-h / H_aerosol),
                   split by the single-scattering albedo

    Cycles measures a mesh volume's density per Blender unit, and the box is
    not parented to the scene root, so coefficients in m^-1 are divided by the
    scene scale (Blender units per meter). The height in meters comes from the
    shading position through the inverse of the scene transform.
    """
    world_block = manifest.get("world", {})
    if not world_block.get("aerial_perspective"):
        return None
    box_bounds = _haze_box_bounds(manifest, transform)
    if box_bounds is None:
        print("  [WARN] aerial perspective skipped: no bounds or camera")
        return None
    minimum, maximum = box_bounds

    # ---- The box, in Blender space (identity transform).
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    haze_box = bpy.context.active_object
    haze_box.name = "skyvista_haze"
    haze_box.location = (minimum + maximum) / 2
    haze_box.scale = maximum - minimum

    # ---- Coefficients, per Blender unit.
    scale = float(transform.get("scale", 1.0e-3))
    z_exaggeration = float(transform.get("z_exaggeration", 1.0))
    origin_z = float((transform.get("origin_shift") or [0.0, 0.0, 0.0])[2])
    ground_height = float(world_block.get("ground_height", 0.0))
    meters_to_blender = 1.0 / scale
    rayleigh_rgb = RAYLEIGH_SCATTERING_SEA_LEVEL_RGB
    rayleigh_max = max(rayleigh_rgb)
    visibility_m = float(world_block.get("visibility_km", 50.0)) * 1000.0
    aerosol_extinction = max(KOSCHMIEDER_CONSTANT / visibility_m - rayleigh_rgb[1], 0.0)
    single_scattering_albedo = float(
        world_block.get("aerosol_single_scattering_albedo", 0.9)
    )

    material = bpy.data.materials.new("skyvista_haze_mat")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    for node in list(nodes):
        if node.type != "OUTPUT_MATERIAL":
            nodes.remove(node)
    output_node = nodes.get("Material Output")

    def math_node(operation, value=None, location=(0, 0)):
        node = nodes.new("ShaderNodeMath")
        node.operation = operation
        node.location = location
        if value is not None:
            node.inputs[1].default_value = value
        return node

    # Height above ground (m) from the world-space shading position:
    # z_data = z_blender / (scale * z_exaggeration) + origin_z.
    geometry = nodes.new("ShaderNodeNewGeometry")
    geometry.location = (-1200, 0)
    separate = nodes.new("ShaderNodeSeparateXYZ")
    separate.location = (-1000, 0)
    links.new(geometry.outputs["Position"], separate.inputs[0])
    to_meters = math_node("MULTIPLY", 1.0 / (scale * z_exaggeration), (-800, 0))
    links.new(separate.outputs["Z"], to_meters.inputs[0])
    above_ground = math_node("ADD", origin_z - ground_height, (-600, 0))
    links.new(to_meters.outputs[0], above_ground.inputs[0])

    def density_profile(scale_height_m, surface_coefficient, location):
        """surface_coefficient * exp(-h / H), per Blender unit."""
        ratio = math_node("DIVIDE", -scale_height_m, location)
        links.new(above_ground.outputs[0], ratio.inputs[0])
        decay = math_node("EXPONENT", None, (location[0] + 200, location[1]))
        links.new(ratio.outputs[0], decay.inputs[0])
        coefficient = math_node(
            "MULTIPLY",
            surface_coefficient * meters_to_blender,
            (location[0] + 400, location[1]),
        )
        links.new(decay.outputs[0], coefficient.inputs[0])
        return coefficient.outputs[0]

    # ---- Molecules: spectral scattering with the Rayleigh phase function.
    rayleigh = nodes.new("ShaderNodeVolumeScatter")
    rayleigh.location = (0, 250)
    rayleigh.phase = "RAYLEIGH"
    rayleigh.inputs["Color"].default_value = (
        *(coefficient / rayleigh_max for coefficient in rayleigh_rgb),
        1.0,
    )
    links.new(
        density_profile(RAYLEIGH_SCALE_HEIGHT_M, rayleigh_max, (-500, 250)),
        rayleigh.inputs["Density"],
    )

    # ---- Aerosol: grey scattering (HG) plus grey absorption.
    aerosol_scale_height = float(world_block.get("aerosol_scale_height_m", 1500.0))
    aerosol_scatter = nodes.new("ShaderNodeVolumeScatter")
    aerosol_scatter.location = (0, -50)
    aerosol_scatter.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    aerosol_scatter.inputs["Anisotropy"].default_value = float(
        world_block.get("aerosol_asymmetry", 0.7)
    )
    links.new(
        density_profile(
            aerosol_scale_height,
            aerosol_extinction * single_scattering_albedo,
            (-500, -50),
        ),
        aerosol_scatter.inputs["Density"],
    )
    aerosol_absorption = nodes.new("ShaderNodeVolumeAbsorption")
    aerosol_absorption.location = (0, -300)
    # Black: every channel absorbed at the full Density.
    aerosol_absorption.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    links.new(
        density_profile(
            aerosol_scale_height,
            aerosol_extinction * (1.0 - single_scattering_albedo),
            (-500, -300),
        ),
        aerosol_absorption.inputs["Density"],
    )

    add_scatterers = nodes.new("ShaderNodeAddShader")
    add_scatterers.location = (250, 100)
    links.new(rayleigh.outputs[0], add_scatterers.inputs[0])
    links.new(aerosol_scatter.outputs[0], add_scatterers.inputs[1])
    add_absorption = nodes.new("ShaderNodeAddShader")
    add_absorption.location = (450, 0)
    links.new(add_scatterers.outputs[0], add_absorption.inputs[0])
    links.new(aerosol_absorption.outputs[0], add_absorption.inputs[1])
    links.new(add_absorption.outputs[0], output_node.inputs["Volume"])
    haze_box.data.materials.append(material)
    print(
        f"  aerial perspective: visibility {visibility_m / 1000:g} km, "
        f"aerosol beta {aerosol_extinction * 1000:.3f} km^-1 at the ground"
    )
    return haze_box


def configure_exposure(scene, manifest):
    """
    Set the film exposure: explicit from the manifest, else auto-metered.

    Auto-metering applies only to the "physical_sky" world, whose radiances
    are physical and far above display range. Other presets are tuned for an
    exposure of 0 and keep it.
    """
    render_config = manifest.get("render", {})
    world_block = manifest.get("world", {})
    exposure = render_config.get("exposure")
    if exposure is None and world_block.get("preset") == "physical_sky":
        exposure = auto_exposure_ev(
            world_block.get("sun_elevation_deg", 35.0),
            world_block.get("background_strength", 1.0),
        )
    if exposure is not None:
        scene.view_settings.exposure = float(exposure)
        print(f"  exposure: {float(exposure):+.2f} EV")


# ---------------------------------------------------------------------------
# Colorbar compositing (best-effort overlay of the baked colorbar PNGs)
# ---------------------------------------------------------------------------
def _set_compositor_value(node, property_name, socket_label, value):
    """
    Set a compositor-node setting that may be a property or an input socket.

    Blender 5.0 moved several Glare settings from node properties to input
    sockets, and the two use different keys (property ``threshold`` vs socket
    "Threshold"). Try the property, then the socket, and silently skip if
    neither exists so a renamed setting can't abort the build.
    """
    if hasattr(node, property_name):
        try:
            setattr(node, property_name, value)
            return
        except (TypeError, AttributeError):
            pass
    socket = node.inputs.get(socket_label)
    if socket is not None:
        try:
            socket.default_value = value
        except (TypeError, AttributeError):
            pass


def _select_glare_bloom_type(glare):
    """
    Set a Glare node to a bloom-style type across Blender versions.

    The control moved between releases: <= 4.x exposes a ``glare_type`` enum
    *property* with UPPERCASE identifiers ("BLOOM" was added in 4.4; "FOG_GLOW"
    is the older soft-halo fallback that reads the same); 5.0 replaced it with a
    "Type" menu *input socket* whose values are title-case labels ("Bloom",
    "Fog Glow"). The default is "Streaks" (a star/streak look), so we must set
    this explicitly or we get streaks instead of a halo. Try both mechanisms.
    """
    if hasattr(glare, "glare_type"):
        try:
            available = {
                item.identifier
                for item in glare.bl_rna.properties["glare_type"].enum_items
            }
        except (KeyError, AttributeError):
            available = set()
        for candidate in ("BLOOM", "FOG_GLOW"):
            if candidate in available:
                glare.glare_type = candidate
                return
    type_socket = glare.inputs.get("Type")
    if type_socket is not None:
        for candidate in ("Bloom", "Fog Glow"):
            try:
                type_socket.default_value = candidate
                return
            except (TypeError, ValueError):
                pass


def _set_glare_size(glare):
    """
    Set a soft, large bloom radius, handling the int- vs float-size split.

    <= 4.x: an integer ``size`` property (1..9, a power-of-two kernel size).
    5.0: a float "Size" input socket (0..1, relative to the image). Set whichever
    this Blender has, with a magnitude appropriate to that scale.
    """
    if hasattr(glare, "size"):
        try:
            glare.size = 7
            return
        except (TypeError, AttributeError):
            pass
    size_socket = glare.inputs.get("Size")
    if size_socket is not None:
        try:
            size_socket.default_value = 0.6
        except (TypeError, ValueError):
            pass


def add_bloom_glare(node_tree, input_socket):
    """
    Insert a Glare (bloom) node after ``input_socket`` and return its output.

    Bloom is what turns bright emissive surfaces (the "glow" shader preset) into
    soft haloed light -- the NCAR "fountain" look. Cycles has no built-in bloom,
    so it is done here in the compositor as a full-frame post-process; a single
    glowing object is enough to warrant it, and it composes with the colorbar
    overlay by feeding this node's output on into that chain.

    The Glare node's type and settings vary across Blender versions (5.0 turned
    them into input sockets and made "Type" a menu), so select the type and set
    values defensively via the version-aware helpers above.
    """
    glare = node_tree.nodes.new("CompositorNodeGlare")
    _select_glare_bloom_type(glare)
    # Only pixels above the threshold bloom; ~1.0 means just the HDR
    # (emission_strength > 1) emitters halo, not the whole lit scene.
    _set_compositor_value(glare, "threshold", "Threshold", 1.0)
    _set_glare_size(glare)

    node_tree.links.new(input_socket, glare.inputs["Image"])
    return glare.outputs["Image"]


def setup_compositing(scene, manifest, bundle_dir, add_bloom=False):
    """
    Build the compositor chain: render layers -> [bloom] -> [colorbars] -> out.

    Overlays the baked colorbar PNGs onto the render and, when ``add_bloom`` is
    set (any object uses a bloom shader preset), inserts a Glare/bloom pass
    first so emissive surfaces halo.

    Best-effort: compositor socket names vary across Blender versions, so the
    whole thing is wrapped by the caller; if it fails the render still works and
    the colorbar PNGs remain in the bundle for manual compositing.
    """
    annotations = manifest.get("annotations", {})
    colorbars = annotations.get("colorbars", [])
    # The exporter can ask for a clean render (the figure draws its own legend).
    if not annotations.get("overlay_colorbars", True):
        colorbars = []
    if not colorbars and not add_bloom:
        return

    # Resolve the compositor node tree across Blender versions. 5.0 removed
    # scene.node_tree / scene.use_nodes in favor of a CompositorNodeTree
    # datablock on scene.compositing_node_group, whose final output is a Group
    # Output node (CompositorNodeComposite no longer exists). Fall back to the
    # <=4.x scene.node_tree + Composite path.
    uses_node_group = hasattr(scene, "compositing_node_group")
    if uses_node_group:
        node_tree = scene.compositing_node_group
        if node_tree is None:
            node_tree = bpy.data.node_groups.new(
                "skyvista_compositor", "CompositorNodeTree"
            )
            scene.compositing_node_group = node_tree
    else:
        scene.use_nodes = True
        node_tree = scene.node_tree

    nodes = node_tree.nodes
    links = node_tree.links

    render_layers = next((n for n in nodes if n.type == "R_LAYERS"), None)
    if render_layers is None:
        render_layers = nodes.new("CompositorNodeRLayers")

    if uses_node_group:
        # The group needs an image output on its interface; the Group Output
        # node's input mirrors it and carries the final composite.
        if not any(item.in_out == "OUTPUT" for item in node_tree.interface.items_tree):
            node_tree.interface.new_socket(
                name="Image", in_out="OUTPUT", socket_type="NodeSocketColor"
            )
        output_node = next((n for n in nodes if n.type == "GROUP_OUTPUT"), None)
        if output_node is None:
            output_node = nodes.new("NodeGroupOutput")
    else:
        output_node = next((n for n in nodes if n.type == "COMPOSITE"), None)
        if output_node is None:
            output_node = nodes.new("CompositorNodeComposite")
    final_output_socket = output_node.inputs[0]  # "Image" (composite) / group out

    resolution_x = scene.render.resolution_x
    resolution_y = scene.render.resolution_y
    current_image_socket = render_layers.outputs["Image"]

    # Bloom first (if requested), so the colorbar overlay sits crisply on top of
    # the haloed render rather than being bloomed itself.
    if add_bloom:
        current_image_socket = add_bloom_glare(node_tree, current_image_socket)

    for colorbar_index, colorbar in enumerate(colorbars):
        image = bpy.data.images.load(
            str(bundle_dir / colorbar["image"]), check_existing=True
        )
        image_node = nodes.new("CompositorNodeImage")
        image_node.image = image

        scale_node = nodes.new("CompositorNodeScale")
        # Scale mode: <=4.x exposed a node .space enum ("RELATIVE"); 5.0 replaced
        # it with a "Type" menu input socket (whose default is already
        # "Relative"). Set whichever this Blender has.
        if hasattr(scale_node, "space"):
            scale_node.space = "RELATIVE"
        else:
            type_socket = scale_node.inputs.get("Type")
            if type_socket is not None:
                type_socket.default_value = "Relative"
        scale_node.inputs["X"].default_value = 0.18
        scale_node.inputs["Y"].default_value = 0.18

        # Stack colorbars down the right edge of the frame.
        translate_node = nodes.new("CompositorNodeTranslate")
        translate_node.inputs["X"].default_value = resolution_x * 0.40
        translate_node.inputs["Y"].default_value = (
            resolution_y * 0.30 - colorbar_index * resolution_y * 0.32
        )

        alpha_over_node = nodes.new("CompositorNodeAlphaOver")
        # 5.0 renamed AlphaOver's two image inputs to Background/Foreground and
        # added a separate Factor input (the old layout was Fac, Image, Image at
        # indices 0/1/2). Prefer names, fall back to the old positional inputs.
        background_input = (
            alpha_over_node.inputs.get("Background") or alpha_over_node.inputs[1]
        )
        foreground_input = (
            alpha_over_node.inputs.get("Foreground") or alpha_over_node.inputs[2]
        )
        factor_input = alpha_over_node.inputs.get("Factor") or alpha_over_node.inputs[0]
        factor_input.default_value = 1.0

        links.new(image_node.outputs["Image"], scale_node.inputs["Image"])
        links.new(scale_node.outputs["Image"], translate_node.inputs["Image"])
        links.new(current_image_socket, background_input)
        links.new(translate_node.outputs["Image"], foreground_input)
        current_image_socket = alpha_over_node.outputs["Image"]

    links.new(current_image_socket, final_output_socket)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    bundle_dir, do_render = parse_arguments()
    manifest = load_manifest(bundle_dir)
    scene = bpy.context.scene

    print(f"Building {bundle_dir.name} in Blender {bpy.app.version_string} ...")
    reset_scene()
    scene_has_clouds = any(
        object_spec.get("material", {}).get("type") == "cloud"
        for object_spec in manifest.get("objects", [])
    )
    configure_render(scene, manifest, scene_has_clouds=scene_has_clouds)

    transform = manifest.get("transform", {})
    root_empty = create_root_empty(scene, transform)

    time_config = manifest.get("time", {})
    frame_start = int(time_config.get("frame_start", 1))
    n_frames = int(time_config.get("frame_end", frame_start)) - frame_start + 1

    built_objects = []
    successes, failures = 0, 0
    for object_spec in manifest.get("objects", []):
        name = object_spec.get("name", "?")
        carrier = object_spec.get("geometry", {}).get("carrier")
        try:
            if carrier == "alembic":
                built_objects += import_alembic_object(
                    bundle_dir, object_spec, root_empty
                )
            elif carrier == "vdb_sequence":
                built_objects += import_volume_object(
                    bundle_dir, object_spec, root_empty, frame_start, n_frames
                )
            elif carrier == "polylines":
                built_objects += import_polylines(
                    bundle_dir,
                    object_spec,
                    root_empty,
                    frame_start,
                    flow_clock=manifest.get("flow_clock"),
                )
            elif carrier == "point_glyphs":
                built_objects += import_point_glyphs(
                    bundle_dir, object_spec, root_empty, frame_start
                )
            else:
                print(f"  [SKIP] {name}: unknown carrier '{carrier}'")
                continue
            print(f"  [OK]   {name} ({carrier})")
            successes += 1
        except Exception as build_error:  # keep going; report at the end
            print(f"  [FAIL] {name}: {type(build_error).__name__}: {build_error}")
            failures += 1

    setup_camera(scene, manifest, transform, built_objects)
    setup_world(scene, manifest)
    setup_ground(scene, manifest, transform, root_empty)
    setup_aerial_perspective(scene, manifest, transform)
    configure_exposure(scene, manifest)
    apply_exposure_relative_emission(scene)

    # A bloom pass is warranted if any surface object opted into it via its
    # shader preset (the "glow" look). Bloom is a full-frame effect, so one
    # emitter turns it on for the whole render.
    needs_bloom = any(
        object_spec.get("material", {}).get("shader", {}).get("bloom")
        for object_spec in manifest.get("objects", [])
    )

    # Compositing (bloom + colorbars) is best-effort: never let it abort build.
    try:
        setup_compositing(scene, manifest, bundle_dir, add_bloom=needs_bloom)
    except Exception as compositing_error:
        print(
            f"  [WARN] compositing skipped: "
            f"{type(compositing_error).__name__}: {compositing_error}"
        )

    # Render output settings are stored in the .blend, so it can be rendered
    # later or by several processes: ``blender -b file.blend -a``.
    render_dir = bundle_dir / "render"
    render_dir.mkdir(parents=True, exist_ok=True)
    # PNG + alpha so a transparent film (film_transparent) is preserved on
    # disk; Blender appends the frame number and extension to this prefix.
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    is_animation = n_frames > 1
    # Skip frames already on disk, and claim each frame with a placeholder
    # when it starts. A long animation interrupted part-way then resumes where
    # it stopped when the same command is run again, and several processes
    # rendering the same .blend share out the frames between them.
    scene.render.use_overwrite = False
    scene.render.use_placeholder = True
    # An animation appends frame numbers to this prefix itself; a still does
    # not, so name it after its frame explicitly.
    scene.render.filepath = (
        str(render_dir / "frame_")
        if is_animation
        else str(render_dir / f"frame_{scene.frame_start:04d}")
    )

    blend_path = bundle_dir / f"{bundle_dir.name}.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    print(f"Saved {blend_path}  ({successes} objects, {failures} failed)")

    if do_render:
        bpy.ops.render.render(animation=is_animation, write_still=not is_animation)
        print(f"Rendered {n_frames} frame(s) to {render_dir}")


if __name__ == "__main__":
    main()
