"""
Tests for launching Blender: the quiet mode that writes Blender's console
output to a log file, and reports a failure with the log's last lines.
A small shell script stands in for Blender.
"""

import stat

import pytest

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
