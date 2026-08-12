"""Tests for headless/offscreen rendering setup and diagnostics."""

import os

import pytest


def test_headless_public_api():
    """configure/doctor/has_working_display are exported from the package."""
    import skyvista as sv

    assert hasattr(sv, "configure")
    assert hasattr(sv, "doctor")
    assert hasattr(sv, "has_working_display")


def test_x_server_reachable_rejects_dead_display():
    """A DISPLAY pointing at no running X server reports unreachable."""
    from skyvista import headless

    # :999 is almost certainly not running: no /tmp/.X11-unix/X999 socket and
    # TCP 6999 refused. The check must be fast (tight timeout) and return False.
    assert headless._x_server_reachable(":999") is False
    # A malformed DISPLAY string is also not reachable.
    assert headless._x_server_reachable("not-a-display") is False


def test_configure_forced_offscreen_sets_env(monkeypatch):
    """Forcing off_screen=True sets PYVISTA_OFF_SCREEN and reports it."""
    from skyvista import headless

    monkeypatch.delenv("PYVISTA_OFF_SCREEN", raising=False)
    resolved = headless.configure(off_screen=True)
    assert resolved["off_screen"] is True
    assert os.environ.get("PYVISTA_OFF_SCREEN") == "true"


def test_configure_interactive_preserves_display(monkeypatch):
    """With off_screen=False we must not clear a caller's DISPLAY."""
    from skyvista import headless

    monkeypatch.setenv("DISPLAY", ":0")
    resolved = headless.configure(off_screen=False)
    assert resolved["off_screen"] is False
    assert resolved["cleared_display"] is None
    assert os.environ.get("DISPLAY") == ":0"


def test_configure_clears_stale_display(monkeypatch):
    """A dead DISPLAY is cleared when going offscreen so VTK skips the X path."""
    from skyvista import headless

    monkeypatch.setenv("DISPLAY", ":999")  # unreachable
    resolved = headless.configure(off_screen=True, silence_stale_display=True)
    assert resolved["cleared_display"] == ":999"
    assert "DISPLAY" not in os.environ


def test_configure_is_idempotent():
    """configure() can be called repeatedly without error."""
    from skyvista import headless

    first = headless.configure()
    second = headless.configure()
    assert first["off_screen"] == second["off_screen"]


def test_doctor_without_render_returns_report():
    """doctor(render_test=False) returns a structured report and skips rendering."""
    import skyvista as sv

    report = sv.doctor(render_test=False)
    for expected_key in ("platform", "python", "skyvista_version", "fixes"):
        assert expected_key in report
    # No render was attempted, so the render fields stay unset.
    assert report["render_ok"] is None


@pytest.mark.gui
def test_doctor_offscreen_render_succeeds():
    """A real offscreen render self-test works headless (EGL or OSMesa)."""
    import skyvista as sv

    report = sv.doctor(render_test=True)
    assert report["render_ok"] is True
    assert report["image_nonblank"] is True
