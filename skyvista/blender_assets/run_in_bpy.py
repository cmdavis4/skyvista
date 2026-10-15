"""
Run a Blender script with the ``bpy`` module, the way the Blender program would.

skyvista can drive Blender in two ways. One is the standalone Blender program::

    blender --background [file.blend] --python script.py -- <arguments>

The other is the ``bpy`` package from PyPI (the ``blender`` extra), which is
Blender built as a Python module. It has no command line of its own, so this
file stands in for one::

    python run_in_bpy.py <file.blend or ""> <script.py> -- <arguments>

It does what the Blender program does with those options:

1. it opens the ``.blend`` file, when one is given (an empty string means
   none, so the script starts from Blender's default scene, as it would in
   the program);
2. it rewrites ``sys.argv`` so the script sees ``[script.py, "--",
   <arguments>]`` -- every skyvista Blender script reads its arguments after
   the ``--`` separator, so the same script runs unchanged in both modes;
3. it runs the script as ``__main__``.

An exception in the script propagates, so the process exits with a non-zero
code. (The standalone program gets the same behavior from
``--python-exit-code``.)
"""

import runpy
import sys

import bpy


def main():
    """Open the optional .blend, then run the script with Blender-style argv."""
    # Everything before the "--" separator belongs to this launcher.
    separator_index = sys.argv.index("--")
    blend_path, script_path = sys.argv[1:separator_index]
    script_arguments = sys.argv[separator_index + 1 :]

    # Load the saved scene first, as `blender file.blend --python ...` does.
    if blend_path:
        bpy.ops.wm.open_mainfile(filepath=blend_path)

    # The script sees the same argv layout as under the Blender program.
    sys.argv = [script_path, "--", *script_arguments]
    runpy.run_path(script_path, run_name="__main__")


if __name__ == "__main__":
    main()
