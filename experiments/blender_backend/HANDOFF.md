# Skyvista → Blender backend: handoff

**Audience:** a Claude Code instance running on the machine that *has Blender*
(the user has **Blender 5.0.1**). The development so far happened on a machine
**without** Blender, so everything on the skyvista side is written and validated
by round-trip, but the **Blender-side build script has never actually run in
Blender.** Your main job is to run it, then fix whatever real `bpy` surfaces.

Everything below is on git branch **`blender`**.

---

## 1. TL;DR — what you're picking up

Skyvista is a Python library for scientifically-accurate 3D visualization of
gridded atmospheric data, built on PyVista/VTK. We are (re)adding the ability to
export a scene to **Blender** for paper-quality figures/animations.

The design: skyvista (normal Python env, **never imports `bpy`**) writes a
self-contained **bundle** on disk — geometry caches plus a declarative
`scene.json` **manifest**. A **build script that runs inside Blender** reads the
bundle and assembles a `.blend`. The resulting `.blend` is script-free and
hand-editable with standard Blender tools.

**Status:**
- ✅ skyvista side (bundle writer, Alembic + VDB caches, materials, camera,
  colorbars, manifest) — built and validated end-to-end by round-trip in
  throwaway envs.
- ❓ `skyvista/blender_assets/build_scene.py` (the Blender-side assembler) —
  **written but unrun in Blender.** This is where bugs will be.
- ❓ The "editable-scalar spike" verdict — needs one Blender run.

---

## 2. Architecture and the decisions behind it

**Blender is a render *backend* of `Scene`,** sitting beside `export_html()` and
`animate()` — not a separate API. Users specify the figure once
(`scene.add_contour(...)`, `add_volume(...)`, etc.) and pick a render target.
`Scene.to_blender(path)` writes the bundle.

**The process boundary is the central fact.** Blender has its own bundled Python;
you cannot reliably `import bpy` from skyvista's env. So skyvista only ever
*writes data + a declarative description*; all `bpy` code lives in
`build_scene.py` on the Blender side. The contract between them is the manifest
schema + the on-disk file formats.

**Geometry carriers:**
- Time-varying **meshes** (contours/isosurfaces, slices, vector glyphs,
  trajectory tubes) → **Alembic `.abc`** with *changing topology* (isosurfaces
  change vertex/face count every frame). Blender ingests these via a **Mesh
  Sequence Cache** modifier.
- **Volumes** → **VDB sequence** (`name_0001.vdb`, …), rendered with a Principled
  Volume shader. This is the killer feature (real volumetric light scattering).

**Animation is baked, not scripted.** All geometry lives in the caches; camera
and lights are keyframed. The build script *constructs* the scene once, but the
runtime `.blend` has **no script dependency** — a Blender artist can edit it
normally. (This directly serves the user's priority: "modifiable by Blender
experts using best practices.")

**Coloring:**
- **Meshes:** the scientific colormap is **baked on the skyvista side** into
  per-vertex **C3f vertex colors** (attribute named `color`). This is the only
  per-vertex attribute that survives Blender's Alembic importer, and baking with
  a fixed global `clim` keeps colors from drifting across frames.
- **Volumes:** the VDB stores the **raw scalar**; the material rebuilds the
  colormap as a **ColorRamp** from a baked LUT (`assets/colormaps/<cmap>.json`),
  because Blender's Python has no matplotlib.

**One scene transform.** Atmospheric coords are huge (meters). A single transform
`blender = (data − origin_shift) · scale` (with an extra vertical factor) is
applied to **everything** via a parent **Empty** (`skyvista_root`); geometry
stays in physical units so it's inspectable/reversible. Defaults: **scale =
1/1000**, `origin_shift` auto-centers on the merged data bounds, `z_exaggeration
= 1.0`.

**Color management:** render view transform defaults to **Standard** (not
AgX/Filmic) so a scientific figure's colors aren't silently altered.

---

## 3. Critical gotchas (do not relearn these the hard way)

1. **Alembic writer = `alembic3d` on PyPI** — the 3D Alembic bindings. **NOT
   `alembic`** (that's the SQLAlchemy DB-migration tool; mixing them up is what
   killed an earlier attempt). Wheels: CPython 3.10–3.13 (no 3.14 yet).
2. **OpenVDB has no usable PyPI wheel** for modern Python (only cp37/2020). Use
   **conda-forge `openvdb`** — import name is `openvdb`. The pixi `blender`
   environment provides it. (`_require_openvdb()` tries `openvdb` then
   `pyopenvdb`.)
3. **Require Blender ≥ 4.x.** Changing-topology Alembic import (and the fix that
   stops vertex interpolation from corrupting varying-vertex-count meshes)
   matured across 3.6 → 4.x. The earlier failed attempt was a Blender-version
   problem, *not* the export package. User is on **5.0.1** — good.
4. **Raw float `arbGeomParams` do NOT import** into Blender from Alembic; only
   recognized color/UV/velocity params survive. That's why mesh colors are baked
   as C3f. (Confirmed on 5.0.1.)
5. `ISampleSelector(int)` is a **time in seconds**, not a sample index — select
   Alembic samples by `frame/fps`. (Only matters in the test scripts.)
6. **Use the Mesh *Sequence* Cache modifier** (Alembic/USD), never the Mesh Cache
   (MDD/PC2) modifier, which needs fixed topology.

---

## 4. Branch, commits, and what's validated

Branch `blender`, most recent last:

| commit | content |
|---|---|
| `1e1a5df` | contour path (first cut) |
| `c80c49b` | mesh-type specs (slice/vector/trajectory) + VDB volumes + `varspec.py` `pass_point_data` fix |
| `f0c4a0e` | **build script** (`build_scene.py`) + pixi `blender` feature + colormap LUT + `Scene.to_blender(build=True)` |
| `09e0b1d` | **camera** keyframes in the manifest (static/orbit/follow) |
| `0ec9f46` | **colorbars** (matplotlib PNG per scalar) |
| `fc8621c` | **editable-scalar spike** wired into the test scripts |

**Validated on the skyvista side** (round-trip in throwaway envs; a py3.13 venv
with the mesh stack, and a py3.11 conda env with `openvdb` + the full skyvista
stack):
- contour/slice/vector/trajectory → Alembic, baked C3f vertex colors,
  heterogeneous-topology flag recorded, colors + geometry round-trip;
- volume → VDB sequence, grid named after the variable, **anisotropic
  index→world transform correct** (`dx`, `dz` distinct), thresholded values
  exact;
- transform auto-centering, camera keyframes (follow tracks storm position),
  colormap LUTs, colorbar PNGs, manifest assembly.

**NOT validated (your job):** `build_scene.py` inside Blender, and the
editable-scalar storage-type verdict.

---

## 5. Repo map

| path | what |
|---|---|
| `skyvista/blender.py` | The exporter. Config dataclasses (`BlenderExportConfig`, `BlenderTransform`, `BlenderRenderConfig`, `BlenderCameraConfig`), colormap baking, Alembic writer (`write_mesh_sequence_alembic`), VDB writer (`write_volume_sequence_vdb`, `_require_openvdb`), colorbar/LUT writers, camera derivation, `export_scene_to_blender`. |
| `skyvista/blender_assets/build_scene.py` | **The Blender-side script you will run.** Reads the bundle, builds the `.blend`. |
| `skyvista/appearance.py` | `Appearance.to_blender_material()` (Blender analog of `to_pyvista_kwargs()`). |
| `skyvista/scene.py` | `Scene.to_blender(path, config=None, times=None, build=False, blender_executable="blender")`. |
| `pixi.toml` | `blender` feature/environment (openvdb via conda, alembic3d via pypi). |
| `experiments/blender_backend/MANIFEST.md` | The `scene.json` schema (design reference). |
| `experiments/blender_backend/test_alembic_roundtrip.py` | Write/read-back proof + the editable-scalar spike write side (no Blender needed). |
| `experiments/blender_backend/blender_verify.py` | **Run in Blender** to confirm topology + attribute import and the spike verdict. |

---

## 6. Bundle layout and manifest (the contract)

```
<bundle>/
  scene.json                 # the manifest (below)
  build_scene.py             # copied in; runnable on its own
  provenance.json
  data/
    contour_THETA_iso305.0/sequence.abc     # Alembic (mesh, changing topology)
    volume_W/W_0001.vdb ... W_000N.vdb      # numbered VDB sequence
    ...
  assets/
    colormaps/coolwarm.json    # LUT stops for volume ColorRamps
    colorbars/volume_W.png     # matplotlib colorbar per scalar-colored object
```

`scene.json` (abridged, annotated):

```jsonc
{
  "skyvista_manifest_version": "0.1",
  "transform": { "origin_shift": [cx,cy,cz], "scale": 0.001, "z_exaggeration": 1.0 },
  "time": { "fps": 24, "frame_start": 1, "frame_end": N, "data_times": [...] },
  "render": { "engine": "CYCLES", "samples": 128, "resolution": [1920,1080],
              "view_transform": "Standard", "film_transparent": true },
  "world": { "preset": "sky", "sun_elevation_deg": 35.0, "sun_azimuth_deg": 40.0,
             "sun_strength": 2.0, "background_strength": 1.0 },
  "camera": { "type": "perspective", "lens_mm": 50,
              "keyframes": [ { "frame": 1, "location": [...], "look_at": [...], "up": [0,0,1] } ] },
  "objects": [
    { "name": "contour_THETA_iso305.0", "spec_type": "contour",
      "geometry": { "carrier": "alembic", "path": "data/.../sequence.abc",
                    "object_path": "/contour_THETA_iso305.0", "topology": "heterogeneous" },
      "material": { "type": "surface", "opacity": 0.7,
                    "shader": { "base": "principled", "roughness": 0.4 },
                    "coloring": { "mode": "vertex_color", "attribute": "color",
                                  "cmap": "viridis", "clim": [..], "label": "W" } } },
    { "name": "volume_W", "spec_type": "volume",
      "geometry": { "carrier": "vdb_sequence", "path_pattern": "data/volume_W/W_####.vdb",
                    "grid_name": "W" },
      "material": { "type": "volume",
                    "coloring": { "mode": "scalar_ramp", "attribute": "W", "cmap": "coolwarm",
                                  "clim": [..], "colormap_lut": "assets/colormaps/coolwarm.json" },
                    "density": { "grid": "W", "clim": [..], "scale": 1.0 } } }
  ],
  "annotations": { "title": null, "show_grid": true, "background": "#f8f6f1",
                   "colorbars": [ { "for": "volume_W", "cmap": "coolwarm", "clim": [..],
                                    "label": "W", "image": "assets/colorbars/volume_W.png" } ] }
}
```

Camera keyframes and object geometry are in **data (physical) units**; the build
script applies the scene transform to them.

---

## 7. How to run it (the main task)

### 7a. Environment

```bash
pixi install -e blender          # openvdb (conda) + alembic3d (pypi) + skyvista
```

Sanity check the writers import:

```bash
pixi run -e blender python -c "import openvdb, alembic3d; print('vdb + abc OK')"
```

### 7b. Produce a bundle (no Blender needed)

Save as `make_bundle.py` and run `pixi run -e blender python make_bundle.py`.
This synthetic dataset exercises a contour (colored by a *different* scalar,
which uses the `pass_point_data` path fixed in `c80c49b`), a slice, and a volume:

```python
import numpy as np, xarray as xr, skyvista as sv

nx, ny, nz, nt = 32, 32, 20, 8
x = np.arange(nx) * 1000.0; y = np.arange(ny) * 1000.0; z = np.arange(nz) * 500.0
times = np.array([f"2016-05-24T21:{m:02d}" for m in range(0, 40, 5)], dtype="datetime64[m]")
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
theta = np.empty((nt, nx, ny, nz)); w = np.empty((nt, nx, ny, nz))
for t in range(nt):
    cx = 8000 + 2500 * t                      # moving, growing blob -> changing topology
    theta[t] = 290 + 30 * np.exp(-(((X-cx)/(6000+800*t))**2 + ((Y-16000)/6000)**2 + ((Z-7000)/4000)**2))
    w[t] = theta[t] - 290
ds = xr.Dataset({"THETA": (["time","x","y","z"], theta), "W": (["time","x","y","z"], w)},
                coords={"x": x, "y": y, "z": z, "time": times})

scene = sv.Scene(force_bounds=False)
scene.add_contour(ds, "THETA", isosurfaces=[305.0], scalar="W", cmap="viridis", opacity=0.7)
scene.add_slice(ds, "THETA", dim="z", value=7000.0, cmap="magma")
scene.add_volume(ds, "W", threshold=(3.0, None), cmap="coolwarm")

# Static camera by default; try mode="orbit" for a turntable.
scene.to_blender("storm_bundle")          # writes storm_bundle/
print("bundle written")
```

### 7c. Build the `.blend` in Blender

```bash
blender --background --python storm_bundle/build_scene.py -- storm_bundle
```

(or in one step from Python: `scene.to_blender("storm_bundle", build=True)`.)

Success looks like `[OK] <name> (alembic|vdb_sequence)` lines and
`Saved storm_bundle/storm_bundle.blend`. Then open the `.blend` in the GUI and
scrub the timeline: meshes should swap topology each frame; the volume should
render as a cloud in Cycles.

Add `--render` after the bundle path to also render frames to `storm_bundle/render/`.

---

## 8. Where `build_scene.py` will most likely need fixing

It is written defensively (per-object `try/except`, best-effort compositing), so
partial failures won't abort the build — read the `[FAIL]`/`[WARN]` lines. Likely
trouble spots, all `bpy`-version-sensitive:

1. **Compositor socket names** (colorbar overlay): `CompositorNodeScale` /
   `Translate` / `AlphaOver` input names (`"X"`, `"Y"`, indices `1`/`2`). Already
   wrapped so it won't kill the build, but may silently skip.
2. **Principled Volume wiring:** node id `ShaderNodeVolumePrincipled`, inputs
   `"Density"`/`"Color"`; the **Attribute** node reading the VDB grid uses output
   `"Fac"` (verify it's `Fac` and the grid name matches `geometry.grid_name`).
3. **Volume sequence playback:** we set `volume.data.is_sequence = True`,
   `frame_start`, `frame_duration`. Confirm the volume actually advances frames
   (Object Data Properties → the grid should update on scrub).
4. **Alembic import object capture:** we diff `scene.objects` before/after
   `wm.alembic_import`. If import nests objects in a collection, adjust.
5. **Alpha/opacity:** `material.blend_method` vs `surface_render_method` changed
   across versions; guarded but verify transparency shows.
6. **Camera clip planes** at scaled coordinates (scale 1/1000 → data spans a few
   hundred Blender units). If the scene is clipped, set `camera.data.clip_end`.

### The known *fidelity* open question — check this deliberately

Baked mesh colors are matplotlib **sRGB** values in `[0,1]`, stored in a float
color attribute. Blender tends to treat float color attributes as **linear**, so
the colormap may look **washed out / too bright**. Compare the Blender render's
colormap against a matplotlib reference. If they differ, the fix options are:
(a) bake **linear** RGB in skyvista (`srgb_to_linear` before writing the C3f), or
(b) drive Base Color through an sRGB→linear node, or (c) use the vertex color as
**Emission** (unlit) so the stored RGB shows exactly. Decide and implement on the
**skyvista side** (`bake_scalar_to_rgb` in `skyvista/blender.py`) if needed.

---

## 9. The editable-scalar spike (one decision to close)

Question: can we offer a **"re-ramp the scalar in Blender without re-exporting"**
mode for meshes? Raw float attrs don't import; C3f colors do — but we only proved
that for values in `[0,1]`. The spike carries the scalar as a **float-color
(gray)** attribute in two variants (raw physical + normalized) and asks whether
Blender preserves float magnitude.

Run:
```bash
pixi run -e blender python experiments/blender_backend/test_alembic_roundtrip.py   # writes changing_topology_sequence.abc
blender --background --python experiments/blender_backend/blender_verify.py -- changing_topology_sequence.abc
```

`blender_verify.py` prints a verdict:
- **FLOAT_COLOR + magnitude preserved** → offer `scalar_ramp` carrying the **raw**
  scalar as a float-color attribute; user re-ramps in physical units.
- **BYTE_COLOR / clamped** → carry the **normalized** scalar + `clim`; the
  material maps `[0,1] → clim` before the ramp.
- **neither imports** → keep baking colors (the current default).

Whatever the verdict, the *default* stays baked vertex colors — this only decides
whether to add the optional editable mode (and in which form) to
`_build_mesh_entry` in `skyvista/blender.py`.

---

## 10. Suggested order of work for you

1. `pixi install -e blender`; confirm `openvdb` + `alembic3d` import.
2. Produce `storm_bundle` (§7b).
3. Run `build_scene.py` (§7c); capture full stdout. Iterate on `[FAIL]`/`[WARN]`
   lines — these are `bpy` API mismatches in `build_scene.py`.
4. Open the `.blend`: verify topology swaps per frame, volume renders, camera
   frames the data, colorbars composite.
5. **Check color fidelity** (§8) and fix in `bake_scalar_to_rgb` if washed out.
6. Run the editable-scalar spike (§9); record the verdict; wire the mode if viable.
7. Nice-to-haves: HDRI/world presets (currently a simple sun + gray ambient in
   `setup_world`), non-rectilinear volume resampling (VDB export currently
   requires a rectilinear grid).

Commit fixes on the `blender` branch, one logical change per commit. All
skyvista-side outputs are already validated, so bugs are almost certainly in
`build_scene.py` (bpy) or the color-fidelity baking — start there.
