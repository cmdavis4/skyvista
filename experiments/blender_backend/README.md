# Blender backend — feasibility experiments

Scratch/validation code for reintroducing Blender visualization as a render
backend of `skyvista.Scene`. Nothing here ships yet; it exists to de-risk the
design before touching the library.

## Decision (validated)

- Blender is a **backend of `Scene`**, not a separate API. Skyvista stays
  `bpy`-free and emits a self-contained **bundle** (data files + a `scene.json`
  manifest). The Blender side (the sciblend fork) consumes the bundle.
- **Geometry carriers:** time-varying *meshes* (contours, slices, glyph meshes,
  trajectory tubes) → **Alembic `.abc`** with changing topology; *volumes* →
  **VDB sequence**.
- Animation is **baked** (Alembic caches + camera/light keyframes), not runtime
  scripting, so the resulting `.blend` is editable with standard Blender tools.

See `MANIFEST.md` for the draft manifest schema (the skyvista ↔ sciblend contract).

## Scripts

| file | env | what it proves |
|---|---|---|
| `test_alembic_roundtrip.py` | plain Python (no Blender) | write path: changing-topology `.abc` + baked vertex color + raw scalar all round-trip |
| `blender_verify.py` | Blender's Python | import side: topology changes per frame, which attributes survive import |

## Running

Write/read-back proof (no Blender needed):

```bash
python -m venv venv && . venv/bin/activate
pip install alembic3d numpy        # NOTE: alembic3d, NOT alembic (that's the SQL tool)
python test_alembic_roundtrip.py   # writes changing_topology_sequence.abc, verifies it
```

Import side (needs **Blender 4.x** — older versions mishandle changing topology):

```bash
blender --background --python blender_verify.py -- changing_topology_sequence.abc
```

## Gotchas found

- **`alembic3d`, not `alembic`** — the latter is the SQLAlchemy DB-migration tool.
- Wheels cover CPython **3.10–3.13** only (no 3.14 yet). Pin the export extra `python <3.14`.
- `ISampleSelector(int)` is a **time in seconds**, not a sample index; an
  out-of-range time silently clamps to the last frame. Select by `frame/fps`.
- Use Blender's **Mesh Sequence Cache** modifier (Alembic/USD), not **Mesh
  Cache** (MDD/PC2) which requires fixed topology.
- The earlier attempt failed on the **import** side; a Blender major-version
  upgrade fixed it. Require Blender 4.x.
- `usd-core` (Pixar/Apple) is a viable, better-maintained alternative carrier
  worth a parallel spike if `alembic3d` supply-chain trust matters.
