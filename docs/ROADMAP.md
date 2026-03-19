# Skyvista Roadmap

## Vision

Enable users to construct a scene using skyvista's PyVista-based functionality and convert it to a Blender scene with a single function call, regardless of Blender expertise.

---

## Phase 1: Stabilize & Unify the PyVista API

**Goal:** Clean up the existing codebase so the Scene class has a solid foundation.

### Tasks

- [ ] Audit and consolidate the type hierarchy in `types_sv.py`
  - `PVVarSpec` hierarchy is good but could use a common `render_type` enum
  - Consider whether `create_mesh_kwargs`/`add_mesh_kwargs` split is the right abstraction
- [ ] Unify naming: `PVRamsData` → `PVGriddedData` (alias exists, complete migration)
- [ ] Review and fix the `PVConfig` class - it mixes plotter instance with output settings
- [ ] Address any known bugs in trajectory rendering, volume rendering, etc.
- [ ] Expand test coverage (`tests/` has minimal tests)
- [ ] Document the current API patterns (the convenience API in `convenience.py` is well-designed)

---

## Phase 2: Design the Scene Abstraction

**Goal:** Create a `Scene` class that captures all information needed to recreate a visualization in either PyVista or Blender.

### Key Design Considerations

- Scene should be **renderer-agnostic** (not tied to PyVista plotter or Blender scene)
- Must capture:
  - All data sources (`PVData` objects)
  - All visualization specs (`PVVarSpec` objects)
  - Camera position/orientation
  - Lighting configuration
  - Material/shader assignments
  - Animation state (time range, fps, etc.)
  - Background/environment

### Tasks

- [ ] Design `SceneConfig` dataclass for renderer-agnostic configuration
- [ ] Design `CameraConfig` dataclass (position, focal point, up vector, FOV)
- [ ] Design `LightingConfig` dataclass (type, position, intensity, color)
- [ ] Design `MaterialConfig` dataclass (abstract material properties)
- [ ] Create `Scene` class that composes these configs + data
- [ ] Refactor `PVConfig` to be a "render target" that consumes a `Scene`
- [ ] Add `Scene.render_pyvista()` method
- [ ] Add `Scene.export_blender()` method stub

---

## Phase 3: Implement PyVista → Blender Conversion

**Goal:** Make `Scene.export_blender()` produce a complete Blender scene.

### Current State

- `blender/pv_to_blender.py` only exports meshes to `.vtk` files
- `blender/blender_run.py` requires the user to manually set up `kwargs_data` with materials, scales, collections
- No transfer of camera, lighting, or materials from PyVista

### Tasks

- [ ] Expand mesh export to preserve more metadata (scalar names, colormaps, etc.)
- [ ] Implement camera conversion (PyVista → Blender coordinate system)
- [ ] Implement coordinate system transformation (PyVista Z-up → Blender Z-up, but different handedness)
- [ ] Map PyVista `add_mesh` kwargs to Blender material properties
  - `color` → principled BSDF base color
  - `opacity` → alpha/transmission
  - `cmap` → vertex colors or procedural texture
- [ ] Auto-generate Blender materials from visualization specs
- [ ] Create a unified `Scene.export_blender(output_dir)` that:
  1. Exports all meshes
  2. Generates a Python script to reconstruct the scene in Blender
  3. Optionally exports a `.blend` file directly

---

## Phase 4: Create Atmospheric Science Presets

**Goal:** Ship sensible defaults for common atmospheric visualization scenarios.

### Material Presets

- [ ] `cloud_material` - volumetric white/gray with subsurface scattering
- [ ] `rain_material` - semi-transparent blue particles
- [ ] `updraft_material` - warm colors (orange/red) for positive W
- [ ] `downdraft_material` - cool colors (blue) for negative W
- [ ] `trajectory_material` - options for tubes, arrows, ribbons

### Lighting Presets

- [ ] `daytime_hdri` - realistic sky lighting
- [ ] `sunset_hdri` - dramatic low-angle lighting
- [ ] `studio_lighting` - clean presentation lighting
- [ ] `scientific_lighting` - neutral, even illumination

### Camera Presets

- [ ] `overview` - 45 degree angle showing domain
- [ ] `storm_core` - looking into updraft region
- [ ] `ground_level` - low angle showing structure

### Scene Presets (Composing the Above)

- [ ] `quick_preview` - fast EEVEE render, simple materials
- [ ] `publication_quality` - Cycles, volumetrics, high samples
- [ ] `animation_optimized` - balanced quality/speed

---

## Phase 5: Polish & Documentation

**Goal:** Make the library production-ready and easy to adopt.

### Tasks

- [ ] Write user guide for the Scene-based workflow
- [ ] Create example notebooks demonstrating PyVista → Blender
- [ ] Add type stubs for better IDE support
- [ ] Publish to PyPI (if not already)

---

## Summary

| Phase | Description | Dependencies |
|-------|-------------|--------------|
| 1 | Stabilize PyVista API | None |
| 2 | Scene Abstraction | Phase 1 |
| 3 | PyVista → Blender Conversion | Phase 2 |
| 4 | Atmospheric Presets | Phase 3 |
| 5 | Documentation & Polish | All |

---

## Notes

- The dependency flow is: **Stable PyVista API → Scene Abstraction → Conversion Mechanics → Presets**
- Phase 1 must come first because the Scene abstraction needs to wrap a stable, well-defined set of primitives
- Phases can have some overlap (e.g., basic presets can be developed alongside Phase 3)
