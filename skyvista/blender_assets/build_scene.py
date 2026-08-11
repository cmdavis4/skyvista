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
def make_surface_material(name, material_spec):
    """Build a Principled-BSDF surface material from a manifest material block."""
    material = bpy.data.materials.new(name=f"{name}_mat")
    material.use_nodes = True
    node_tree = material.node_tree
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

    coloring = material_spec.get("coloring", {})
    if coloring.get("mode") == "vertex_color":
        # Read the baked per-vertex color attribute into Base Color.
        attribute_node = node_tree.nodes.new("ShaderNodeAttribute")
        # The manifest names the baked color attribute (default "col_data" --
        # NOT "color", which collides with a reserved attribute in Cycles and
        # renders grey there while EEVEE looks fine).
        attribute_node.attribute_name = coloring.get("attribute", "col_data")
        attribute_node.location = (-350, 0)
        if principled is not None:
            node_tree.links.new(
                attribute_node.outputs["Color"], principled.inputs["Base Color"]
            )
    else:
        if principled is not None:
            principled.inputs["Base Color"].default_value = parse_color(
                coloring.get("color", "#cccccc")
            )

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
def setup_world(scene, manifest):
    """A simple, decent-looking world: soft gray ambient + a key sun."""
    world = bpy.data.worlds.new("skyvista_world")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (0.05, 0.05, 0.06, 1.0)
        background.inputs["Strength"].default_value = 1.0

    sun_data = bpy.data.lights.new("skyvista_sun", type="SUN")
    sun_data.energy = 3.0
    sun_object = bpy.data.objects.new("skyvista_sun", sun_data)
    sun_object.rotation_euler = (0.6, 0.2, 0.5)
    scene.collection.objects.link(sun_object)


# ---------------------------------------------------------------------------
# Colorbar compositing (best-effort overlay of the baked colorbar PNGs)
# ---------------------------------------------------------------------------
def setup_colorbar_compositing(scene, manifest, bundle_dir):
    """
    Overlay the baked colorbar PNGs onto the render via the compositor.

    Best-effort: compositor socket names vary across Blender versions, so the
    whole thing is wrapped by the caller; if it fails the render still works and
    the colorbar PNGs remain in the bundle for manual compositing.
    """
    colorbars = manifest.get("annotations", {}).get("colorbars", [])
    if not colorbars:
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

    # Colorbar compositing is best-effort: never let it abort the build.
    try:
        setup_colorbar_compositing(scene, manifest, bundle_dir)
    except Exception as compositing_error:
        print(f"  [WARN] colorbar compositing skipped: "
              f"{type(compositing_error).__name__}: {compositing_error}")

    blend_path = bundle_dir / f"{bundle_dir.name}.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    print(f"Saved {blend_path}  ({successes} objects, {failures} failed)")

    if do_render:
        render_path = bundle_dir / "render"
        scene.render.filepath = str(render_path) + "/frame_"
        bpy.ops.render.render(animation=n_frames > 1, write_still=n_frames == 1)
        print(f"Rendered to {render_path}")


if __name__ == "__main__":
    main()
