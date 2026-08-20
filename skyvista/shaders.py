"""
Named *shader presets* for the Blender export backend.

A shader preset is a small, renderer-agnostic bundle of surface-material
parameters -- roughness, metallic, self-illumination (emission), and whether the
render should get a bloom/glow post-process. The user selects one by name
(``shader="glow"``) instead of hand-tuning a node graph, exactly the way
:data:`skyvista.blender.WORLD_PRESETS` lets them pick a world/lighting look.

Keeping the registry here (rather than only on the Blender side) means callers
get validated names, the full set is documented in one place, and the same
resolved block is what lands in the bundle manifest for the Blender-side
``build_scene.py`` to construct.

The motivating look is the classic NCAR "glowing fountain" trajectory render:
emissive particles/streaks whose color comes from a scalar colormap, wrapped in
a soft bloom halo. That is the ``"glow"`` preset; ``"emissive"`` is the same
self-illumination without the bloom, and the rest are ordinary matte/glossy/
metal surfaces.

Resolved shader block (the JSON that goes in the manifest ``material.shader``)::

    {
        "preset": "glow",         # the chosen preset name (provenance)
        "base": "principled",     # node family the build script constructs
        "roughness": 0.5,         # Principled BSDF roughness
        "metallic": 0.0,          # Principled BSDF metallic
        "emission_strength": 6.0, # Principled emission strength (0 = none)
        "emission_from": "color", # where emission color comes from:
                                  #   None    -> not emissive
                                  #   "color" -> the object's own color
                                  #              (baked vertex colors or the
                                  #              solid appearance color)
        "bloom": true,            # request a scene-wide bloom/glare pass
    }
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

# The default preset: a clean, matte physically-based surface. This reproduces
# skyvista's previous fixed shader (Principled, roughness 0.4, no emission), so
# existing exports look identical unless a preset is explicitly chosen.
DEFAULT_SHADER_PRESET = "matte"


@dataclass(frozen=True)
class ShaderPreset:
    """
    One named surface look, resolvable to a manifest shader block.

    Attributes:
        base: Node family the Blender build script constructs. Only
            "principled" today, but named so future presets (e.g. a
            toon/holdout base) can select a different graph.
        roughness: Principled BSDF roughness (0 = mirror, 1 = fully diffuse).
        metallic: Principled BSDF metallic (0 = dielectric, 1 = metal).
        emission_strength: Principled emission strength. 0 disables emission;
            values > 1 push pixels into HDR so a bloom pass has something to
            bloom.
        emission_from: Source of the emission *color*. ``None`` means the
            surface does not self-illuminate; ``"color"`` drives the emission
            color from the object's own color (its baked per-vertex colormap
            colors, or its solid appearance color when no scalar is mapped), so
            an emissive trajectory glows in whatever color the data gave it.
        bloom: Whether a render using this preset should get a scene-wide
            bloom/glare post-process (the soft halo around bright emitters).
            Bloom is a full-frame compositor effect, so a single object opting
            in turns it on for the whole render.
    """

    base: str = "principled"
    roughness: float = 0.4
    metallic: float = 0.0
    emission_strength: float = 0.0
    emission_from: Optional[str] = None
    bloom: bool = False

    def to_dict(self, preset_name: str) -> Dict[str, Any]:
        """Resolve to the manifest ``material.shader`` block for this preset."""
        return {
            "preset": preset_name,
            "base": self.base,
            "roughness": self.roughness,
            "metallic": self.metallic,
            "emission_strength": self.emission_strength,
            "emission_from": self.emission_from,
            "bloom": self.bloom,
        }


# The registry. Names here are what a user passes as ``shader=...``.
SHADER_PRESET_DEFINITIONS: Dict[str, ShaderPreset] = {
    # Default: clean matte surface (unchanged from skyvista's prior look).
    "matte": ShaderPreset(roughness=0.4, metallic=0.0),
    # Low-roughness dielectric: crisp highlights, a slightly "wet"/ceramic read.
    "glossy": ShaderPreset(roughness=0.08, metallic=0.0),
    # Metal: full metallic with moderate roughness (brushed-metal look).
    "metal": ShaderPreset(roughness=0.25, metallic=1.0),
    # Glow: the NCAR "glowing fountain" look. Self-illuminates in the object's
    # own (colormap) color and adds a scene-wide bloom halo. Strength is tuned
    # so the color still reads through the bloom rather than clipping to white.
    "glow": ShaderPreset(
        roughness=0.5,
        metallic=0.0,
        emission_strength=6.0,
        emission_from="color",
        bloom=True,
    ),
    # Emissive: same self-illumination as "glow" but no bloom -- a flatter,
    # self-lit surface for when the halo would be too much (dense fields, print).
    "emissive": ShaderPreset(
        roughness=0.5,
        metallic=0.0,
        emission_strength=2.0,
        emission_from="color",
    ),
}

# Public tuple of valid preset names, mirroring blender.WORLD_PRESETS' role.
SHADER_PRESETS = tuple(SHADER_PRESET_DEFINITIONS)


def resolve_shader(shader: Optional[str]) -> Dict[str, Any]:
    """
    Resolve a preset name (or ``None``) to a manifest shader block.

    Args:
        shader: A preset name from :data:`SHADER_PRESETS`, or ``None`` to use
            the default (:data:`DEFAULT_SHADER_PRESET`, a matte surface).

    Returns:
        The resolved ``material.shader`` dict for the manifest.

    Raises:
        ValueError: If ``shader`` names a preset that does not exist, with the
            list of valid names.
    """
    preset_name = shader if shader is not None else DEFAULT_SHADER_PRESET
    if preset_name not in SHADER_PRESET_DEFINITIONS:
        valid = ", ".join(sorted(SHADER_PRESET_DEFINITIONS))
        raise ValueError(
            f"Unknown shader preset {preset_name!r}. Valid presets are: {valid}."
        )
    return SHADER_PRESET_DEFINITIONS[preset_name].to_dict(preset_name)
