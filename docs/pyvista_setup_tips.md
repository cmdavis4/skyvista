# PyVista setup tips

Setting up PyVista used to be onerous on remote/headless machines. With modern
VTK (>= 9.3) most of that pain is gone: VTK can render **offscreen with no X
server at all**, and skyvista auto-configures it for you. This page describes the
easy path first, then keeps the legacy Xvfb instructions at the bottom for the
rare cases that still need them.

## TL;DR — check whether it already works

After installing skyvista, run the built-in environment doctor:

```bash
python -m skyvista        # or, in Python:  import skyvista as sv; sv.doctor()
```

It runs a real offscreen render and tells you the active backend and whether the
image came back non-blank, with a specific fix for anything that's wrong. If you
see `✔ offscreen render` you can write figures/animations to disk immediately —
no `DISPLAY`, no `Xvfb`, no admin-installed system libraries.

A second quick check is to render the PyVista sample bunny; if you see it (and,
interactively, can rotate it), you're in business:

```python
from pyvista import examples
examples.download_bunny().plot(cpos="xy")
```

## How offscreen rendering works now

Modern VTK wheels ship with two X-free rendering backends and pick one
automatically:

- **EGL** — GPU rendering with no X server. Used automatically when the node has
  a GPU + drivers. This is the ideal case on a compute server.
- **OSMesa** — CPU software rendering, no GPU and no X server. The universal
  fallback that works anywhere. If your VTK build doesn't include it, install an
  OSMesa-enabled VTK (e.g. the conda-forge `vtk` osmesa variant, or the
  `vtk-osmesa` wheel).

skyvista handles the setup for you: on `import skyvista` it detects a headless
environment, enables offscreen mode (`PYVISTA_OFF_SCREEN`), and clears a **stale
`DISPLAY`** — a dead `Xvfb` pointer like `:99.0` left in a shell profile — so the
headless detection isn't fooled into thinking a display exists.

You will likely still see one line like this on the first render:

```
vtkXOpenGLRenderWindow: bad X server connection. DISPLAY=
```

That warning is harmless and expected. VTK always probes X before falling back,
so it prints the warning and then renders successfully via EGL/OSMesa. Confirm
with `python -m skyvista`: if the doctor reports `✔ offscreen render`, the
warning can be ignored.

- Opt out of the auto-setup with `SKYVISTA_NO_AUTOCONFIG=1` before importing.
- Override anything explicitly with `skyvista.configure(...)` (see its docstring
  for `off_screen`, `jupyter_backend`, `server_proxy`, ...).

## Interactive plots inside Jupyter

Writing to disk "just works" as above. Getting **live, interactive** plots inside
a notebook is the part that can still need a nudge, and the fix differs by
frontend.

### VSCode

If interactive PyVista plots show up blank in VSCode's notebook, it's a
port-forwarding quirk ([PyVista #5296](https://github.com/pyvista/pyvista/issues/5296)).
Two settings fix it:

- `remote.autoForwardPortsSource`: `process`
- `remote.localPortHost`: `localhost`

Then, after running a plotting cell that comes up blank, toggle the cell's output
type back and forth once ([demo](https://github.com/pyvista/pyvista/issues/5296#issuecomment-2374315543)).
For me this persists until the VSCode window is reloaded.

### Browser Jupyter behind a proxy

When Jupyter is served in a browser behind a proxy, PyVista's trame backend needs
`server_proxy` mode. skyvista wraps the boilerplate:

```python
import skyvista as sv
sv.configure(server_proxy=True, server_proxy_prefix="/proxy/")
```

(Equivalent to setting `pv.global_theme.trame.server_proxy_enabled = True`, the
prefix, and `pv.set_jupyter_backend("trame")` by hand.)

## Legacy: the Xvfb path (only if you can't get EGL/OSMesa)

You should not need this with a modern VTK. It's here for old VTK builds, or a
CPU-only node whose VTK lacks OSMesa, where a virtual X display is the last
resort.

Install the system packages (needs sudo — ask an admin):
`libgl1-mesa-glx` and `xvfb` (the old docs also mention `python-qt4`, which is
outdated and unnecessary).

```bash
export DISPLAY=:99.0
export PYVISTA_OFF_SCREEN=true
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &   # rerun after every reboot
```

If you hit lock-file errors, delete `/tmp/.X99-lock` (or the matching file under
`/tmp/.X11-unix/`). From Python you can instead call `pyvista.start_xvfb()`.
Whenever possible, prefer the EGL/OSMesa path above and skip all of this.
