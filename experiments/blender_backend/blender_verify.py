"""
Blender-side verification of a changing-topology Alembic cache.

Run OUTSIDE this script's environment, inside Blender's bundled Python:

    blender --background --python blender_verify.py -- changing_topology_sequence.abc

(The ``--`` separates Blender's args from ours; the path after it is the
``.abc`` written by ``test_alembic_roundtrip.py``.)

Use a recent Blender (4.x strongly recommended). Changing-topology Alembic
import and the fix that stops vertex interpolation from corrupting meshes
with varying vertex counts both matured in the 3.6 -> 4.x line.

What this checks (the questions the write-side round-trip cannot answer)
-----------------------------------------------------------------------
1. Blender imports the cache and auto-adds a Mesh Sequence Cache modifier.
2. Scrubbing the timeline actually changes the evaluated mesh's vertex count
   frame-to-frame -- i.e. Blender swaps topology rather than freezing frame 0
   or trying to interpolate (which is what corrupts changing-topology caches).
3. Which per-vertex attribute survives import, informing the design choice:
      * baked RGB "color"  -> a Color Attribute (robust, colormap already applied)
      * raw scalar "THETA" -> a generic float attribute (flexible, color-ramp
                              done inside Blender by the user)
   Whether (b) survives is the genuinely uncertain part; this reports it.
"""

import sys

import bpy


def get_argument_after_double_dash() -> str:
    """Return the first CLI argument that follows the Blender ``--`` separator."""
    if "--" not in sys.argv:
        raise SystemExit(
            "Usage: blender --background --python blender_verify.py -- <path.abc>"
        )
    arguments_after_separator = sys.argv[sys.argv.index("--") + 1 :]
    if not arguments_after_separator:
        raise SystemExit("No .abc path provided after '--'.")
    return arguments_after_separator[0]


def clear_default_scene() -> None:
    """Remove the default cube/camera/light so only imported objects remain."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def import_alembic(alembic_path: str) -> None:
    """Import the Alembic archive; Blender adds a Mesh Sequence Cache modifier."""
    bpy.ops.wm.alembic_import(
        filepath=alembic_path,
        set_frame_range=True,  # match the scene frame range to the cache
        validate_meshes=True,  # repair any corrupt data on import (recommended)
    )


def get_first_imported_mesh_object():
    """Return the first mesh object in the scene (the imported poly-mesh)."""
    for scene_object in bpy.context.scene.objects:
        if scene_object.type == "MESH":
            return scene_object
    raise SystemExit("No mesh object found after Alembic import.")


def report_mesh_sequence_cache_modifier(mesh_object) -> None:
    """Confirm Blender attached a Mesh Sequence Cache modifier on import."""
    cache_modifiers = [
        modifier
        for modifier in mesh_object.modifiers
        if modifier.type == "MESH_SEQUENCE_CACHE"
    ]
    if cache_modifiers:
        print(f"  [PASS] Mesh Sequence Cache modifier present on "
              f"'{mesh_object.name}'")
    else:
        print(f"  [WARN] No Mesh Sequence Cache modifier on "
              f"'{mesh_object.name}' -- topology may be frozen at frame 0")


def evaluate_mesh_at_current_frame(mesh_object):
    """Return the evaluated mesh (modifiers applied) at the current scene frame."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_object = mesh_object.evaluated_get(depsgraph)
    return evaluated_object.data


def main() -> int:
    alembic_path = get_argument_after_double_dash()
    print(f"Importing {alembic_path} into Blender {bpy.app.version_string} ...")

    clear_default_scene()
    import_alembic(alembic_path)
    mesh_object = get_first_imported_mesh_object()

    print("\nChecking import ...")
    report_mesh_sequence_cache_modifier(mesh_object)

    scene = bpy.context.scene
    frame_start = scene.frame_start
    frame_end = scene.frame_end
    print(f"  [INFO] scene frame range: {frame_start}..{frame_end}")

    observed_vertex_counts = []
    color_attribute_present_each_frame = []
    scalar_attribute_present_each_frame = []

    for current_frame in range(frame_start, frame_end + 1):
        scene.frame_set(current_frame)
        evaluated_mesh = evaluate_mesh_at_current_frame(mesh_object)

        vertex_count = len(evaluated_mesh.vertices)
        observed_vertex_counts.append(vertex_count)

        # Color attribute (baked colormap) -> Blender Color Attribute
        color_attribute = evaluated_mesh.color_attributes.get("color")
        color_attribute_present_each_frame.append(color_attribute is not None)

        # Raw scalar -> generic float attribute (the uncertain one)
        scalar_attribute = evaluated_mesh.attributes.get("THETA")
        scalar_attribute_present_each_frame.append(scalar_attribute is not None)

        color_status = "yes" if color_attribute is not None else "NO"
        scalar_status = "yes" if scalar_attribute is not None else "NO"
        print(
            f"  frame {current_frame}: vertices={vertex_count:4d}  "
            f"color_attr={color_status}  scalar_attr={scalar_status}"
        )

    print("\nSummary")
    topology_changes = len(set(observed_vertex_counts)) > 1
    print(f"  observed vertex counts: {observed_vertex_counts}")
    print(f"  [{'PASS' if topology_changes else 'FAIL'}] "
          f"changing topology survives import (counts differ across frames)")

    all_color_present = all(color_attribute_present_each_frame)
    print(f"  [{'PASS' if all_color_present else 'FAIL'}] baked 'color' "
          f"Color Attribute present on every frame")

    all_scalar_present = all(scalar_attribute_present_each_frame)
    print(f"  [{'PASS' if all_scalar_present else 'INFO'}] raw 'THETA' float "
          f"attribute present on every frame "
          f"(if FAIL: prefer baking colors on the skyvista side)")

    print("\nDesign takeaway:")
    if all_color_present and topology_changes:
        print("  Baked vertex-color path is viable -> reproduce the scientific")
        print("  colormap on the skyvista side and store RGB as vertex colors.")
    if all_scalar_present:
        print("  Raw-scalar path is ALSO viable -> you can defer the color ramp")
        print("  to a Blender material node, editable by the end user.")
    elif not all_scalar_present:
        print("  Raw-scalar attribute did NOT import on this Blender version;")
        print("  bake colors on the skyvista side (or add a color ramp per-frame).")

    return 0 if (topology_changes and all_color_present) else 1


if __name__ == "__main__":
    sys.exit(main())
