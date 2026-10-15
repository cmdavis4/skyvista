"""
Tests for launching Blender: the quiet mode that writes Blender's console
output to a log file, and reports a failure with the log's last lines; and
the choice between a Blender program and the bpy module.
A small shell script stands in for the Blender program.
"""

import stat
import subprocess
import sys

import pytest

import skyvista.blender as skyvista_blender
from skyvista.blender import _launch_blender_build


def make_fake_blender(directory, exit_code):
    """A stand-in for Blender that prints two lines and exits with exit_code."""
    script_path = directory / "fake_blender"
    script_path.write_text(
        "#!/bin/sh\necho 'Building bundle ...'\necho 'Error: bad volume' >&2\n"
        f"exit {exit_code}\n"
    )
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)
    return script_path


def test_quiet_launch_writes_the_log(tmp_path, capfd):
    fake_blender = make_fake_blender(tmp_path, exit_code=0)
    log_path = tmp_path / "blender.log"
    _launch_blender_build(
        fake_blender, tmp_path / "build_scene.py", tmp_path, log_path=log_path
    )
    # Both streams go to the log; nothing reaches the console.
    assert "Building bundle" in log_path.read_text()
    assert "bad volume" in log_path.read_text()
    assert capfd.readouterr().out == ""


def test_quiet_launch_failure_shows_the_end_of_the_log(tmp_path):
    fake_blender = make_fake_blender(tmp_path, exit_code=3)
    with pytest.raises(RuntimeError, match="(?s)code 3.*bad volume"):
        _launch_blender_build(
            fake_blender,
            tmp_path / "build_scene.py",
            tmp_path,
            log_path=tmp_path / "blender.log",
        )


# ---------------------------------------------------------------------------
# Choosing between the Blender program and the bpy module
# ---------------------------------------------------------------------------
def test_explicit_program_beats_the_bpy_module(tmp_path, monkeypatch):
    fake_blender = make_fake_blender(tmp_path, exit_code=0)
    monkeypatch.delenv("SKYVISTA_BLENDER", raising=False)
    monkeypatch.setattr(skyvista_blender, "_bpy_module_is_installed", lambda: True)
    runner = skyvista_blender._resolve_blender_runner(fake_blender)
    assert not runner.uses_bpy_module
    assert runner.executable == str(fake_blender)


def test_environment_variable_beats_the_bpy_module(tmp_path, monkeypatch):
    fake_blender = make_fake_blender(tmp_path, exit_code=0)
    monkeypatch.setenv("SKYVISTA_BLENDER", str(fake_blender))
    monkeypatch.setattr(skyvista_blender, "_bpy_module_is_installed", lambda: True)
    runner = skyvista_blender._resolve_blender_runner()
    assert not runner.uses_bpy_module


def test_bpy_module_runs_in_this_interpreter(monkeypatch):
    monkeypatch.delenv("SKYVISTA_BLENDER", raising=False)
    monkeypatch.setattr(skyvista_blender, "_bpy_module_is_installed", lambda: True)
    runner = skyvista_blender._resolve_blender_runner()
    assert runner.uses_bpy_module
    assert runner.executable == sys.executable


def test_the_two_modes_pass_the_script_the_same_arguments():
    program = skyvista_blender.BlenderRunner("/opt/blender", uses_bpy_module=False)
    module = skyvista_blender.BlenderRunner(sys.executable, uses_bpy_module=True)
    program_command = program.script_command("s.py", ["4"], blend_path="b.blend")
    module_command = module.script_command("s.py", ["4"], blend_path="b.blend")
    # The program opens the file itself, before --python.
    assert program_command[:3] == ["/opt/blender", "--background", "b.blend"]
    # The module launcher takes the file and the script before "--".
    assert module_command[2:4] == ["b.blend", "s.py"]
    # Both end with the script's own arguments after "--".
    assert program_command[-2:] == module_command[-2:] == ["--", "4"]


@pytest.mark.slow
@pytest.mark.skipif(
    not skyvista_blender._bpy_module_is_installed(), reason="needs the bpy module"
)
def test_bpy_module_runs_a_script_with_blender_arguments(tmp_path):
    # The script prints the argv it sees and the Blender version.
    script_path = tmp_path / "probe.py"
    script_path.write_text(
        "import sys, bpy\nprint('ARGV', sys.argv[1:])\n"
        "print('VERSION', bpy.app.version_string)\n"
    )
    module = skyvista_blender.BlenderRunner(sys.executable, uses_bpy_module=True)
    completed = subprocess.run(
        module.script_command(script_path, ["a", "b"]),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "ARGV ['--', 'a', 'b']" in completed.stdout
    assert "VERSION" in completed.stdout


@pytest.mark.slow
@pytest.mark.skipif(
    not skyvista_blender._bpy_module_is_installed(), reason="needs the bpy module"
)
def test_bpy_module_script_error_exits_non_zero(tmp_path):
    script_path = tmp_path / "broken.py"
    script_path.write_text("raise ValueError('broken script')\n")
    module = skyvista_blender.BlenderRunner(sys.executable, uses_bpy_module=True)
    completed = subprocess.run(
        module.script_command(script_path), capture_output=True, text=True
    )
    assert completed.returncode != 0
    assert "broken script" in completed.stderr
