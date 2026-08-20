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


def configure_render(scene, manifest):
    """Apply engine, samples, resolution, color management, and frame range."""
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
    if isinstance(color_spec, str) and color_spec.startswith("#") and len(color_spec) == 7:
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
    if coloring.get("mode") == "vertex_color":
        # Read the baked per-vertex color attribute (default "col_data" -- NOT
        # "color", which collides with a reserved attribute in Cycles and
        # renders grey there while EEVEE looks fine).
        attribute_node = node_tree.nodes.new("ShaderNodeAttribute")
        attribute_node.attribute_name = coloring.get("attribute", "col_data")
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
    if (
        principled is not None
        and emission_from == "color"
        and emission_strength > 0.0
    ):
        emission_color_input = _principled_emission_color_input(principled)
        if emission_color_input is not None:
            if color_output_socket is not None:
                links.new(color_output_socket, emission_color_input)
            else:
                emission_color_input.default_value = solid_color_value
        strength_input = principled.inputs.get("Emission Strength")
        if strength_input is not None:
            strength_input.default_value = emission_strength

    # Opacity via alpha; enable alpha blending for EEVEE (Cycles honors it too).
    opacity = float(material_spec.get("opacity", 1.0))
    if opacity < 1.0 and principled is not None and "Alpha" in principled.inputs:
        principled.inputs["Alpha"].default_value = opacity
        for blend_attr in ("blend_method", "surface_render_method"):
            if hasattr(material, blend_attr):
                try:
                    setattr(material, blend_attr, "BLEND"
                            if blend_attr == "blend_method" else "BLENDED")
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

    # Ensure it plays as a sequence over the animation range.
    volume_data = volume_object.data
    if hasattr(volume_data, "is_sequence"):
        volume_data.is_sequence = True
        try:
            volume_data.frame_start = frame_start
            volume_data.frame_duration = n_frames
        except AttributeError:
            pass

    parent_keep_local(volume_object, root_empty)
    material = make_volume_material(object_spec["name"], object_spec["material"], bundle_dir)
    volume_object.data.materials.clear()
    volume_object.data.materials.append(material)
    return [volume_object]


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
def aim_camera(camera_object, location, look_at):
    """Position a camera and rotate it to look at a target point."""
    camera_object.location = location
    direction = (Vector(look_at) - Vector(location))
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
    "white": 0.15,
    "studio": 1.0,
    "dark": 1.0,
}


def aim_sun(sun_object, sun_elevation_deg, sun_azimuth_deg):
    """
    Orient a Sun lamp so its light arrives from a given sky direction.

    A Blender Sun emits along its local -Z axis. At zero rotation that points
    straight down (sun at the zenith). Tilting about X by (90 - elevation) drops
    the apparent sun to the requested height above the horizon; rotating about Z
    sets the compass azimuth. Euler order 'XYZ' applies the X tilt first, then
    the Z azimuth, which is exactly what we want.
    """
    elevation = math.radians(sun_elevation_deg)
    azimuth = math.radians(sun_azimuth_deg)
    sun_object.rotation_euler = (math.pi / 2.0 - elevation, 0.0, azimuth)


def make_sky_texture(nodes, sun_elevation_deg, sun_azimuth_deg):
    """
    Create a physically-based Sky Texture node aimed at the given sun direction.

    The sky-model enum was renamed across Blender versions: 4.x exposes
    "NISHITA", while 5.0 replaced it with "MULTIPLE_SCATTERING" (same underlying
    model, same sun_* attributes). Pick the best physically-based model that is
    actually available so this works on either. The Nishita parameters are node
    attributes (not input sockets); set them defensively so a renamed attribute
    in some Blender version can't abort the whole build.

    The sky's own sun disc is disabled: the crisp key light comes from the
    explicit, matched Sun lamp, while the sky still contributes realistic blue
    skylight. This avoids a blown-out sun disc and doubled directional light.
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
        sky.sun_rotation = math.radians(sun_azimuth_deg)
    if hasattr(sky, "sun_disc"):
        sky.sun_disc = False
    return sky


def setup_world(scene, manifest):
    """
    Build the world background + key light from the manifest's ``world`` block.

    Presets:
      - "sky":    Nishita physical daytime sky; the key sun is aligned to the
                  sky's sun direction, and the sky's own sun disc is disabled so
                  the crisp shadows come from the (single) Sun lamp while the sky
                  still provides realistic blue ambient light.
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

    # Background color source: physical sky for "sky", else a constant color.
    if preset == "sky":
        sky = make_sky_texture(nodes, sun_elevation_deg, sun_azimuth_deg)
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

    # Key sun: one directional lamp, aimed from the configured sky direction so
    # shadows are consistent regardless of which world preset is active.
    sun_data = bpy.data.lights.new("skyvista_sun", type="SUN")
    sun_data.energy = sun_strength
    sun_object = bpy.data.objects.new("skyvista_sun", sun_data)
    aim_sun(sun_object, sun_elevation_deg, sun_azimuth_deg)
    scene.collection.objects.link(sun_object)


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
    colorbars = manifest.get("annotations", {}).get("colorbars", [])
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
    configure_render(scene, manifest)

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
                built_objects += import_alembic_object(bundle_dir, object_spec, root_empty)
            elif carrier == "vdb_sequence":
                built_objects += import_volume_object(
                    bundle_dir, object_spec, root_empty, frame_start, n_frames
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
        print(f"  [WARN] compositing skipped: "
              f"{type(compositing_error).__name__}: {compositing_error}")

    blend_path = bundle_dir / f"{bundle_dir.name}.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    print(f"Saved {blend_path}  ({successes} objects, {failures} failed)")

    if do_render:
        render_dir = bundle_dir / "render"
        render_dir.mkdir(parents=True, exist_ok=True)
        # PNG + alpha so a transparent film (film_transparent) is preserved on
        # disk; Blender appends the frame number and extension to this prefix.
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.filepath = str(render_dir / "frame_")
        is_animation = n_frames > 1
        bpy.ops.render.render(animation=is_animation, write_still=not is_animation)
        print(f"Rendered {n_frames} frame(s) to {render_dir}")


if __name__ == "__main__":
    main()
