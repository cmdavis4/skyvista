"""
Tests for the Blender cloud shader: species that render with the same phase
function share one Volume Scatter closure, because Cycles keeps only 8 phase
closures per point. Runs ``build_scene.make_cloud_material`` inside Blender.
"""

import json
import subprocess
import textwrap
from pathlib import Path

import pytest

import skyvista.blender as skyvista_blender


def find_blender_runner():
    """How skyvista would run Blender here (program or bpy module), or None."""
    try:
        return skyvista_blender._resolve_blender_runner()
    except FileNotFoundError:
        return None


# Runs inside Blender: builds the cloud material from the spec given as JSON,
# then prints each scatter node's label and anisotropy, and the attribute
# grids feeding the material.
CLOUD_MATERIAL_PROBE = textwrap.dedent("""
    import json, sys
    arguments = sys.argv[sys.argv.index("--") + 1 :]
    sys.path.insert(0, arguments[0])
    import build_scene
    material_spec = json.loads(arguments[1])
    material = build_scene.make_cloud_material("clouds", material_spec)
    nodes = material.node_tree.nodes
    scatter_nodes = [
        {"label": node.label, "anisotropy": node.inputs["Anisotropy"].default_value}
        for node in nodes
        if node.bl_idname == "ShaderNodeVolumeScatter"
    ]
    grids = sorted(
        node.attribute_name for node in nodes if node.bl_idname == "ShaderNodeAttribute"
    )
    print("PROBE" + json.dumps({"scatter": scatter_nodes, "grids": grids}))
    """)


@pytest.mark.slow
@pytest.mark.skipif(find_blender_runner() is None, reason="needs Blender")
def test_species_with_equal_anisotropy_share_one_closure(tmp_path):
    # The rendered anisotropies of the RAMS preset: four ice species at 0.5.
    species_anisotropies = {
        "cloud": 0.86,
        "rain": 0.76,
        "pristine_ice": 0.5,
        "snow": 0.5,
        "aggregates": 0.5,
        "graupel": 0.5,
        "hail": 0.6000000000000001,
    }
    material_spec = {
        "extinction_grid_units": "1/km",
        "species": [
            {
                "name": species_name,
                "extinction_grid": f"extinction_{species_name}",
                "phase_function": "henyey_greenstein",
                "anisotropy": anisotropy,
            }
            for species_name, anisotropy in species_anisotropies.items()
        ],
    }
    probe_path = tmp_path / "probe.py"
    probe_path.write_text(CLOUD_MATERIAL_PROBE)
    build_scene_dir = Path(skyvista_blender.__file__).parent / "blender_assets"
    completed = subprocess.run(
        find_blender_runner().script_command(
            probe_path, [str(build_scene_dir), json.dumps(material_spec)]
        ),
        capture_output=True,
        text=True,
        check=True,
    )
    (probe_line,) = [
        line for line in completed.stdout.splitlines() if line.startswith("PROBE")
    ]
    probe = json.loads(probe_line[len("PROBE") :])

    # Four distinct phase functions -> four closures, in the species' order.
    scatter_labels = [scatter["label"] for scatter in probe["scatter"]]
    assert scatter_labels == [
        "cloud scattering",
        "rain scattering",
        "pristine_ice + snow + aggregates + graupel scattering",
        "hail scattering",
    ]
    rendered_anisotropies = [scatter["anisotropy"] for scatter in probe["scatter"]]
    assert rendered_anisotropies == pytest.approx([0.86, 0.76, 0.5, 0.6])
    # Every species still reads its own extinction grid.
    assert probe["grids"] == sorted(
        f"extinction_{species_name}" for species_name in species_anisotropies
    )
