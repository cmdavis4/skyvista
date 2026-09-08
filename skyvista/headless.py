"""
Headless / offscreen rendering setup and diagnostics.

Modern VTK (>= 9.3) wheels can render offscreen with no X server at all, via
**EGL** (GPU) or **OSMesa** (CPU). That makes the classic Xvfb dance described in
``docs/pyvista_setup_tips.md`` -- ask an admin for system GL libraries, export
``DISPLAY``, start ``Xvfb`` on every reboot, delete stale ``.X99-lock`` files --
unnecessary for the common case of writing figures/animations to disk.

This module turns that already-working path into a zero-configuration one:

- :func:`configure` idempotently puts PyVista into offscreen mode when there is
  no usable display, and cleans up a stale ``DISPLAY`` (a dead ``Xvfb`` pointer
  left in a shell profile) so the headless detection is not fooled into thinking
  a display exists. Note that clearing ``DISPLAY`` does *not* silence VTK's "bad
  X server connection" warning: VTK always probes X before falling back, so the
  warning still appears (with an empty ``DISPLAY=``) and is harmless -- the
  render then succeeds via EGL/OSMesa. It also exposes the two
  interactive-notebook knobs (Jupyter backend, ``server_proxy``) so users can
  stop copy-pasting trame boilerplate.

- :func:`doctor` runs a real offscreen render self-test and prints exactly what
  is (and is not) working -- the active render-window backend, whether the image
  came back non-blank, the Jupyter situation -- with an actionable fix for each
  problem. This converts the classic opaque "blank render window" into a
  specific instruction.

:func:`configure` runs automatically on ``import skyvista`` (it is cheap: it only
reads environment variables and sets flags; it does **not** render). Opt out by
setting the environment variable ``SKYVISTA_NO_AUTOCONFIG=1`` before import.
"""

from __future__ import annotations

import os
import re
import socket
import sys
from typing import Any, Dict, Optional

# Remembers a DISPLAY value we cleared for offscreen hygiene, so configure() can
# restore it (e.g. if a caller later re-enables an interactive display).
_cleared_display_value: Optional[str] = None


# =============================================================================
# Display detection
# =============================================================================
def _x_server_reachable(display: str) -> bool:
    """
    Cheaply test whether the X server named by ``display`` actually accepts
    connections, without pulling in any X libraries.

    A ``DISPLAY`` being *set* does not mean an X server is *running*: remote
    setups routinely leave ``DISPLAY=:99.0`` in a shell profile pointing at an
    Xvfb that is no longer up. We check the two ways X listens: the local unix
    socket ``/tmp/.X11-unix/X<n>`` and TCP port ``6000 + <n>``.
    """
    # DISPLAY looks like "[host]:<number>[.<screen>]", e.g. ":99.0" or "box:0".
    display_match = re.match(r"^([\w.\-]*):(\d+)", display)
    if display_match is None:
        return False
    host_part = display_match.group(1)
    display_number = int(display_match.group(2))

    # Local displays use a unix-domain socket; its mere existence is a good
    # (dependency-free) signal that a server is present.
    if not host_part or host_part == "unix":
        if os.path.exists(f"/tmp/.X11-unix/X{display_number}"):
            return True

    # Otherwise probe the TCP port X would listen on, with a tight timeout so a
    # dead pointer costs milliseconds, not seconds.
    target_host = host_part or "127.0.0.1"
    try:
        probe_socket = socket.create_connection(
            (target_host, 6000 + display_number), timeout=0.2
        )
        probe_socket.close()
        return True
    except OSError:
        return False


def has_working_display() -> bool:
    """
    True if this process appears to have a usable window system.

    On macOS/Windows the native window system is always present. On Linux we
    require either a Wayland display or an X ``DISPLAY`` that actually answers
    (see :func:`_x_server_reachable`) -- a stale ``DISPLAY`` string does not
    count.
    """
    if not sys.platform.startswith("linux"):
        return True
    if os.environ.get("WAYLAND_DISPLAY"):
        return True
    display = os.environ.get("DISPLAY", "")
    if not display:
        return False
    return _x_server_reachable(display)


# =============================================================================
# Configuration
# =============================================================================
def configure(
    *,
    off_screen: Optional[bool] = None,
    jupyter_backend: Optional[str] = None,
    server_proxy: Optional[bool] = None,
    server_proxy_prefix: str = "/proxy/",
    silence_stale_display: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Prepare PyVista for headless rendering with sensible, overridable defaults.

    This is safe to call repeatedly and at any point (including before PyVista is
    imported -- it sets the ``PYVISTA_OFF_SCREEN`` environment variable that
    PyVista reads at its own import time).

    Args:
        off_screen: Force offscreen (True) or interactive (False) rendering.
            ``None`` (default) auto-detects: offscreen when there is no working
            display (see :func:`has_working_display`).
        jupyter_backend: If given and running under Jupyter, set PyVista's
            Jupyter backend (e.g. "trame", "static", "html", "none").
        server_proxy: If True, enable PyVista's trame ``server_proxy`` mode --
            the boilerplate needed for interactive plots in a browser-based
            Jupyter served behind a proxy. Leave ``None``/False for VSCode.
        server_proxy_prefix: URL prefix used when ``server_proxy`` is enabled.
        silence_stale_display: When going offscreen, clear a ``DISPLAY`` that
            points at a non-responding X server, so nothing downstream mistakes
            the dead pointer for a live display. VTK still probes X and still
            prints its "bad X server connection" warning before falling back to
            EGL/OSMesa; clearing only empties the ``DISPLAY=`` in that message.
        verbose: Print what was changed.

    Returns:
        A dict describing the resolved settings (useful for tests / doctor).
    """
    global _cleared_display_value

    # Decide offscreen vs interactive.
    resolved_off_screen = (
        off_screen if off_screen is not None else not has_working_display()
    )

    resolved_settings: Dict[str, Any] = {
        "off_screen": resolved_off_screen,
        "cleared_display": None,
        "jupyter_backend": None,
        "server_proxy": False,
    }

    if resolved_off_screen:
        # PyVista reads this at import; setdefault so we never clobber an
        # explicit user choice already in the environment.
        os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

        # Drop a dead DISPLAY pointer so later display checks see the truth.
        # This does not stop VTK's own X probe or its "bad X server" warning.
        # Only do this when the display is genuinely unreachable, and remember
        # the value so it can be restored.
        display = os.environ.get("DISPLAY")
        if silence_stale_display and display and not _x_server_reachable(display):
            _cleared_display_value = display
            del os.environ["DISPLAY"]
            resolved_settings["cleared_display"] = display

    # Reflect the offscreen choice onto PyVista if it is already imported. If it
    # isn't, the environment variable set above will carry the choice through
    # when PyVista is first imported.
    pyvista_module = sys.modules.get("pyvista")
    if pyvista_module is None and (jupyter_backend or server_proxy):
        # These two knobs require touching the live PyVista object.
        import pyvista as pyvista_module  # noqa: F401

    if pyvista_module is not None:
        if resolved_off_screen:
            pyvista_module.OFF_SCREEN = True

        # The interactive-notebook knobs only make sense inside Jupyter.
        if jupyter_backend and _running_in_jupyter():
            pyvista_module.set_jupyter_backend(jupyter_backend)
            resolved_settings["jupyter_backend"] = jupyter_backend

        if server_proxy:
            trame_theme = pyvista_module.global_theme.trame
            trame_theme.server_proxy_enabled = True
            trame_theme.server_proxy_prefix = server_proxy_prefix
            resolved_settings["server_proxy"] = True

    if verbose:
        print(f"[skyvista] headless configure -> {resolved_settings}")

    return resolved_settings


def _running_in_jupyter() -> bool:
    """True if we appear to be inside an IPython/Jupyter kernel (not a plain REPL)."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    if shell is None:
        return False
    # ZMQInteractiveShell is the Jupyter kernel; TerminalInteractiveShell is the
    # plain `ipython` REPL, where the notebook knobs don't apply.
    return shell.__class__.__name__ == "ZMQInteractiveShell"


def _autoconfigure_on_import() -> None:
    """
    Best-effort automatic headless setup at ``import skyvista``.

    Deliberately conservative: it only enables offscreen + stale-DISPLAY hygiene
    (never a Jupyter backend or server_proxy, which depend on details we can't
    reliably detect). Fully non-fatal -- import must never fail because of it --
    and skippable via ``SKYVISTA_NO_AUTOCONFIG=1``.
    """
    if os.environ.get("SKYVISTA_NO_AUTOCONFIG"):
        return
    try:
        configure()
    except Exception:
        # Headless convenience must never break `import skyvista`.
        pass


# =============================================================================
# Diagnostics
# =============================================================================
def _classify_render_window(window_class_name: str) -> str:
    """Map a VTK render-window class name to a human description of the backend."""
    lowered = window_class_name.lower()
    if "egl" in lowered:
        return "EGL (GPU, no X server) -- ideal for a headless server"
    if "osmesa" in lowered or "osopengl" in lowered:
        return "OSMesa (CPU software rendering, no X server) -- works anywhere"
    if "xopengl" in lowered or "glx" in lowered:
        return "X/GLX (needs a display or Xvfb) -- works, but not ideal on a server"
    if "win32" in lowered or "cocoa" in lowered:
        return "native desktop window system"
    return "unknown backend"


def doctor(render_test: bool = True) -> Dict[str, Any]:
    """
    Diagnose the rendering environment and print an actionable report.

    Runs a real offscreen render (unless ``render_test=False``) and reports the
    active backend, whether the produced image was non-blank, and the Jupyter
    situation -- with a specific fix printed for anything that looks wrong.

    Returns:
        A structured dict of findings (also printed as a human-readable report).
    """
    report: Dict[str, Any] = {}
    fixes: list[str] = []

    # ---- Basic environment ------------------------------------------------
    report["platform"] = sys.platform
    report["python"] = sys.version.split()[0]
    try:
        import skyvista

        report["skyvista_version"] = skyvista.__version__
    except Exception:
        report["skyvista_version"] = "unknown"

    try:
        import pyvista

        report["pyvista_version"] = pyvista.__version__
    except Exception as pyvista_import_error:
        report["pyvista_version"] = f"NOT IMPORTABLE ({pyvista_import_error})"

    try:
        import vtk

        report["vtk_version"] = vtk.vtkVersion.GetVTKVersion()
    except Exception:
        report["vtk_version"] = "unknown"

    # ---- Display / offscreen state ---------------------------------------
    report["DISPLAY"] = os.environ.get("DISPLAY", "(unset)")
    report["PYVISTA_OFF_SCREEN"] = os.environ.get("PYVISTA_OFF_SCREEN", "(unset)")
    report["has_working_display"] = has_working_display()

    # ---- Offscreen render self-test --------------------------------------
    report["render_ok"] = None
    report["render_window"] = None
    report["render_backend"] = None
    report["image_nonblank"] = None
    if render_test:
        try:
            import numpy as np
            import pyvista as pv

            pv.OFF_SCREEN = True
            test_plotter = pv.Plotter(off_screen=True)
            test_plotter.add_mesh(pv.Sphere())
            rendered_image = test_plotter.screenshot(return_img=True)
            window_class_name = test_plotter.render_window.GetClassName()
            test_plotter.close()

            report["render_window"] = window_class_name
            report["render_backend"] = _classify_render_window(window_class_name)
            report["image_nonblank"] = bool(
                rendered_image is not None and np.asarray(rendered_image).sum() > 0
            )
            report["render_ok"] = report["image_nonblank"]
            if not report["image_nonblank"]:
                fixes.append(
                    "Offscreen render produced a blank image. Try a software "
                    "backend: install an OSMesa-enabled VTK, or as a last resort "
                    "start a virtual display with `pyvista.start_xvfb()`."
                )
        except Exception as render_error:
            report["render_ok"] = False
            report["render_error"] = repr(render_error)
            fixes.append(
                f"Offscreen render raised {render_error!r}. Your VTK likely has "
                "no usable GL backend on this node. Install an OSMesa-enabled VTK "
                "(CPU rendering, no GPU/X needed), or run `pyvista.start_xvfb()`."
            )

    # ---- Jupyter situation ------------------------------------------------
    report["in_jupyter"] = _running_in_jupyter()
    if report["in_jupyter"]:
        # We can't reliably tell VSCode from a browser frontend, so advise both
        # of the documented interactive fixes.
        fixes.append(
            "Interactive plots blank in VSCode? Set `remote.autoForwardPortsSource"
            "=process` and `remote.localPortHost=localhost`, then toggle the cell "
            "output type once (PyVista issue #5296)."
        )
        fixes.append(
            "Interactive plots in a browser Jupyter behind a proxy? Run "
            "`skyvista.configure(server_proxy=True)` before plotting."
        )

    _print_doctor_report(report, fixes)
    report["fixes"] = fixes
    return report


def _print_doctor_report(report: Dict[str, Any], fixes: list[str]) -> None:
    """Pretty-print a :func:`doctor` report with check marks and fixes."""

    def status_mark(ok: Optional[bool]) -> str:
        if ok is None:
            return "•"
        return "✔" if ok else "✖"

    print("skyvista doctor")
    print("=" * 60)
    print(f"  platform            : {report['platform']}")
    print(f"  python              : {report['python']}")
    print(f"  skyvista            : {report['skyvista_version']}")
    print(f"  pyvista             : {report['pyvista_version']}")
    print(f"  vtk                 : {report['vtk_version']}")
    print("-" * 60)
    print(f"  DISPLAY             : {report['DISPLAY']}")
    print(f"  PYVISTA_OFF_SCREEN  : {report['PYVISTA_OFF_SCREEN']}")
    print(f"  working display     : {report['has_working_display']}")
    if report["render_ok"] is not None:
        print("-" * 60)
        print(f"  {status_mark(report['render_ok'])} offscreen render")
        if report.get("render_window"):
            print(f"      window  : {report['render_window']}")
            print(f"      backend : {report['render_backend']}")
        if report.get("render_error"):
            print(f"      error   : {report['render_error']}")
    print(f"  {status_mark(None)} in Jupyter        : {report['in_jupyter']}")
    print("=" * 60)
    if fixes:
        print("Suggested fixes:")
        for fix_index, fix_text in enumerate(fixes, start=1):
            print(f"  {fix_index}. {fix_text}")
    else:
        print("Everything looks good. 🎉")


if __name__ == "__main__":
    doctor()
