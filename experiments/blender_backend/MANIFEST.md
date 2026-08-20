# `scene.json` manifest — draft schema (v0.1)

The manifest is the **contract** between skyvista (writer) and the sciblend
fork / headless build script (reader). Skyvista emits it; the Blender side never
needs to import skyvista. It is declarative: it names data files and describes
how to build objects, materials, camera, and render settings — but contains no
Blender API calls.

Design rules:

- **One object per `VarSpec`**, keyed by `VarSpec.name` (already auto-generated
  in every spec's `__post_init__`).
- **One global transform**, applied identically to every object, the camera, and
  the lights, so spatial relationships are preserved and the mapping back to
  physical units is recoverable.
- **Renderer-agnostic appearance → `material`**, the Blender analog of
  `Appearance.to_pyvista_kwargs()`.
- Everything needed to rebuild the figure is inside the bundle directory; all
  `path` fields are relative to the manifest.

## Bundle layout

```
storm_figure/
  scene.json                     # this file
  provenance.json                # dataset hashes, skyvista version, echoed specs
  data/
    contour_THETA_iso300/sequence.abc
    volume_W/W_0001.vdb ... W_0100.vdb
    trajectory_tube_altitude/sequence.abc
  assets/
    colormaps/viridis.png        # baked LUT for scalar_ramp materials
    colorbars/THETA.png          # rendered separately, composited
    hdri/studio.exr              # optional world lighting
  storm_figure.blend             # produced by the build script (git-ignored)
```

## Example

```jsonc
{
  "skyvista_manifest_version": "0.1",
  "generated_by": { "skyvista_version": "1.0.0", "timestamp": "2026-08-10T20:45:00Z" },

  // Applied identically to all objects, camera, lights. Data coords (e.g.
  // meters) minus origin_shift, times scale, with an extra vertical factor.
  // Recorded so the figure is reversible back to physical units.
  "transform": {
    "origin_shift": [128000.0, 128000.0, 0.0],
    "scale": 1.0e-4,
    "z_exaggeration": 1.0
  },

  // Frame <-> data-time mapping. sample_times_seconds are the Alembic sample
  // times; data_times are the original coordinate values (for the timestamp).
  "time": {
    "fps": 24,
    "frame_start": 1,
    "frame_end": 100,
    "sample_times_seconds": [0.0, 0.04167, 0.08333, "..."],
    "data_times": ["2016-05-24T21:00:00", "..."]
  },

  "render": {
    "engine": "CYCLES",              // or "EEVEE" for fast preview
    "samples": 128,
    "resolution": [1920, 1080],
    "film_transparent": true,
    "view_transform": "Standard"     // data-faithful color; NOT AgX/Filmic
  },

  // World background + key light. `preset` is one of "sky" (Nishita physical
  // sky, sun aligned to it), "studio" (neutral gray), "dark", or "white". The
  // sun_* / background_strength knobs tune the chosen preset. Built by
  // build_scene.py:setup_world().
  "world": {
    "preset": "sky",
    "sun_elevation_deg": 35.0,   // sun height above horizon (also drives sky)
    "sun_azimuth_deg": 40.0,     // sun compass direction, CCW from +x about +z
    "sun_strength": 2.0,         // key Sun lamp irradiance
    "background_strength": 1.0   // camera-visible background brightness; its
                                 // *lighting* contribution is damped per preset
                                 // (Is-Camera-Ray mix) so a bright sky doesn't
                                 // wash the baked colors to white
  },
  "lights": [],

  "camera": {
    "type": "perspective",
    "lens_mm": 50,
    "clip_start": 0.01,
    "clip_end": 1000.0,
    // Static camera => single keyframe. Following => one keyframe per frame,
    // generated from camera.py. look_at/up are in transformed Blender space.
    "keyframes": [
      { "frame": 1,   "location": [12, -18, 9], "look_at": [0, 0, 3], "up": [0, 0, 1] },
      { "frame": 100, "location": [15, -12, 9], "look_at": [0, 0, 3], "up": [0, 0, 1] }
    ]
  },

  "objects": [
    {
      "name": "contour_THETA_iso300",       // from ContourSpec.name
      "spec_type": "contour",
      "geometry": {
        "carrier": "alembic",
        "path": "data/contour_THETA_iso300/sequence.abc",
        "object_path": "/storm_isosurface", // path within the .abc
        "topology": "heterogeneous"
      },
      "material": {
        "type": "surface",
        // coloring is the crux; two mutually exclusive modes:
        "coloring": {
          "mode": "vertex_color",           // DEFAULT: colormap baked on skyvista
          "attribute": "color"              // side -> Blender Color Attribute
          // -- OR (editable-in-Blender path) --
          // "mode": "scalar_ramp",         // re-rampable scalar
          // "attribute": "scalar",         // MUST be a *float-color* attribute
          //                                //   (raw float geom params do NOT
          //                                //   import in Blender 5.0; only
          //                                //   recognized color/UV/velocity
          //                                //   params survive). Carry the
          //                                //   scalar as C4f gray = (s,s,s,1).
          // "cmap": "viridis",
          // "clim": [290.0, 320.0],
          // "colormap_lut": "assets/colormaps/viridis.png"
        },
        "opacity": 0.7,
        // shader = a resolved *preset* (skyvista.shaders.SHADER_PRESETS:
        //   matte | glossy | metal | glow | emissive). "glow" is the NCAR
        //   emissive-fountain look: emission_from "color" drives Emission Color
        //   from the same source as Base Color (the vertex colormap or solid
        //   color), and bloom=true requests a scene-wide Glare/bloom pass.
        "shader": { "preset": "matte", "base": "principled",
                    "roughness": 0.4, "metallic": 0.0,
                    "emission_strength": 0.0, "emission_from": null,
                    "bloom": false }
      }
    },

    {
      "name": "volume_W",                    // from VolumeSpec.name
      "spec_type": "volume",
      "geometry": {
        "carrier": "vdb_sequence",
        "path_pattern": "data/volume_W/W_####.vdb",
        "grid_name": "W"
      },
      "material": {
        "type": "volume",
        "coloring": { "mode": "scalar_ramp", "cmap": "coolwarm", "clim": [-30, 30],
                      "colormap_lut": "assets/colormaps/coolwarm.png" },
        "density": { "attribute": "W", "scale": 0.1 },
        "emission": { "strength": 0.0 }
      }
    },

    {
      "name": "trajectory_tube_altitude",    // from TrajectorySpec.name
      "spec_type": "trajectory",
      "geometry": { "carrier": "alembic",
                    "path": "data/trajectory_tube_altitude/sequence.abc",
                    "object_path": "/trajectories", "topology": "heterogeneous" },
      "material": { "type": "surface",
                    "coloring": { "mode": "vertex_color", "attribute": "color" },
                    "opacity": 1.0,
                    "shader": { "base": "principled", "roughness": 0.3 } }
    }
  ],

  // Things Blender won't draw itself; skyvista supplies them.
  "annotations": {
    "colorbars": [
      { "for": "contour_THETA_iso300", "cmap": "viridis", "clim": [290, 320],
        "label": "THETA (K)", "image": "assets/colorbars/THETA.png" }
    ],
    "timestamp": { "enabled": true, "format": "t={t_minutes:.0f} min" },
    "title": null,
    "show_grid": true
  }
}
```

## Mapping to existing skyvista classes

| manifest field | source in skyvista |
|---|---|
| `objects[].name`, `spec_type` | `VarSpec.name`, subclass |
| `objects[].geometry` | `VarSpec.create_mesh()` per timestep → written to the carrier |
| `objects[].material` | `Appearance.to_blender_material()` (new; parallels `to_pyvista_kwargs()`) |
| `time.*` | `Scene._get_all_times()` + `animation.FPS.to_fps()` |
| `camera.keyframes` | `camera.py` (static or following) |
| `render.view_transform`, `world`, `annotations.show_grid`, `title` | `Scene` fields (`background`, `title`, `show_grid`) + a new `BlenderExportConfig` |
| `transform` | new `BlenderExportConfig` (origin_shift, scale, z_exaggeration) |
| `provenance` | dataset ids/hashes + echoed spec params |

## Open questions (decide before v0.1 freezes)

1. **Default coloring mode** — RESOLVED: default `vertex_color` (bake the
   colormap on the skyvista side). Verified on Blender 5.0.1: the baked C3f
   color attribute imports on every frame, but a raw float `arbGeomParam` does
   NOT import (only recognized color/UV/velocity params survive Alembic import).
   The editable `scalar_ramp` path is still possible but must carry the scalar
   as a **float-color attribute** (gray `(s,s,s,1)`), not a float param —
   needs a quick verify before it's promised as a feature.
2. **Transform defaults** — auto-derive `origin_shift`/`scale` from merged data
   bounds (center at origin, fit to ~10 Blender units), or require explicit?
   Auto with override is the likely answer.
3. **One `.abc` per object vs one per bundle** — per-object is simpler to map and
   lets a Blender user toggle objects independently; keep per-object.
4. **Camera parameterization** — store `look_at`/`up` (convert to Blender
   quaternion on the reader side) vs store the matrix directly. `look_at` is more
   human-editable; prefer it.
5. **Colorbar compositing** — bake into the render via the Blender compositor, or
   leave `assets/colorbars/*.png` for the user to overlay? Start with the latter.
