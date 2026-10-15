"""
Render the animation of an already open ``.blend`` file, with a fixed thread
count.

skyvista runs this script after Blender has opened a bundle's ``.blend``::

    blender --background bundle.blend --python render_blend.py -- <threads>

It is the scripted equivalent of ``blender -b bundle.blend --threads N -a``.
A script is used instead of those options because the ``bpy`` module (the
PyPI build of Blender) has no command line; see ``run_in_bpy.py``.

The render settings themselves -- output path, skip-existing frames, and the
placeholders that let several processes share the frames -- were saved into
the ``.blend`` by ``build_scene.py``, so this script only sets the threads.
"""

import sys

import bpy


def main():
    """Set the render thread count from argv, then render every frame."""
    # The only argument after "--" is the number of CPU threads to use.
    thread_count = int(sys.argv[sys.argv.index("--") + 1])

    # A fixed count, so parallel processes do not each claim every core.
    render_settings = bpy.context.scene.render
    render_settings.threads_mode = "FIXED"
    render_settings.threads = thread_count

    bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
