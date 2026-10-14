"""
Shared setup for the skyvista example notebooks.

Every notebook starts with ``import notebook_setup as setup``. Importing this
module:

1. puts the example script folders (``../rams_clouds``, ``../cm1_storms``) on
   the import path, so the notebooks can reuse their data loaders;
2. tells skyvista where Blender and ffmpeg are (``$SKYVISTA_BLENDER`` and
   ``$SKYVISTA_FFMPEG``, unless already set);
3. silences an h5py version warning that says nothing useful here.

It also holds the data paths the notebooks use. **Edit the paths below to
point at your own output.** The notebooks write everything they make under
``WORK_DIR`` (``examples/notebooks/work``, ignored by git).
"""

import io
import os
import sys
import warnings
from pathlib import Path

# =============================================================================
# Paths: edit these for your machine
# =============================================================================
# Blender 5.0 executable, and an ffmpeg for encoding videos.
BLENDER_EXECUTABLE = Path.home() / "programs/blender-5.0.1-linux-x64/blender"
FFMPEG_EXECUTABLE = Path(
    "/home/cmdavis4/projects/dc_entrainment/.pixi/envs/default/bin/ffmpeg"
)

# A RAMS run of an isolated storm, and one lite output time near its peak.
RAMS_STORM_OUTPUT_DIR = Path(
    "/moonbow/cmdavis4/projects/thermals/tracer_soundings/20220602_20220603/"
    "20220602.124500CDT_20220602.174500UTC/ccn-500_regen-no/output"
)
RAMS_STORM_FILE = RAMS_STORM_OUTPUT_DIR / "a-L-2000-01-01-013420-g1.h5"

# A RAMS run with layered passive tracers (ITRACER > 0).
RAMS_TRACER_FILE = Path(
    "/moonbow/cmdavis4/projects/thermals/single_chain/ccn-500_ftccn-True/"
    "fullrun_tracers/output/a-L-2000-01-01-011000-g1.h5"
)

# CM1 runs (Morrison microphysics) with tobac tracking in ``derived/``.
CM1_DATA_ROOT = Path("/moonbow/cmdavis4/projects/dc_entrainment")
CM1_ISOLATED_STORM_RUN = (
    CM1_DATA_ROOT / "ic/higher_res/test_compression_tapfrq-10_prclfrq-10"
)
CM1_SUPERCELL_RUN = CM1_DATA_ROOT / "sc/higher_res/dx-250_nstar-1e-07-nx-1500_ny-1060"

# Where the notebooks write bundles, renders and videos.
WORK_DIR = Path(__file__).resolve().parent / "work"

# =============================================================================
# Preview quality
# =============================================================================
# Renders in these notebooks use few samples and a small image so each takes
# seconds to a few minutes. For a final figure, raise both (e.g. 128 samples
# at 1920 x 1080); a mature storm then takes ~10 min on 64 cores.
PREVIEW_SAMPLES = 16
PREVIEW_RESOLUTION = (960, 540)

# =============================================================================
# Environment
# =============================================================================
EXAMPLES_DIR = Path(__file__).resolve().parent.parent
for example_folder in ("rams_clouds", "cm1_storms"):
    folder_path = str(EXAMPLES_DIR / example_folder)
    if folder_path not in sys.path:
        sys.path.insert(0, folder_path)

# The pixi environment's h5py was built against a slightly older HDF5.
warnings.filterwarnings("ignore", message="h5py is running against HDF5")

# skyvista falls back to these when no executable is passed explicitly.
os.environ.setdefault("SKYVISTA_BLENDER", str(BLENDER_EXECUTABLE))
os.environ.setdefault("SKYVISTA_FFMPEG", str(FFMPEG_EXECUTABLE))

WORK_DIR.mkdir(exist_ok=True)


def apply_preview_quality(config, resolution=None):
    """
    Set a Blender export config to the notebooks' quick preview quality.

    Args:
        config: A ``skyvista.BlenderExportConfig`` (changed in place).
        resolution: (width, height) in pixels; default PREVIEW_RESOLUTION.

    Returns:
        The same config, for chaining.
    """
    config.render.samples = PREVIEW_SAMPLES
    config.render.resolution = tuple(resolution or PREVIEW_RESOLUTION)
    return config


def rendered_frame(bundle_dir, frame_number: int = 1) -> Path:
    """The rendered image of one frame of a bundle."""
    return Path(bundle_dir) / "render" / f"frame_{frame_number:04d}.png"


def show_image(image_path, width: int = 900):
    """
    Display an image in the notebook, shrunk and stored as JPEG.

    Renders are lossless PNGs of 1-3 MB. Embedding them as they are would make
    every notebook tens of megabytes, so the image is resized to ``width``
    pixels and re-encoded as a JPEG of ~100 kB before it is displayed. The
    file on disk is left untouched.

    Args:
        image_path: An image file, or a bundle directory (its first frame).
        width: Displayed width in pixels.
    """
    from IPython.display import Image as DisplayImage
    from IPython.display import display
    from PIL import Image

    image_path = Path(image_path)
    if image_path.is_dir():
        image_path = rendered_frame(image_path)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        height = round(image.height * width / image.width)
        resized = image.resize((width, height), Image.LANCZOS)
    jpeg_buffer = io.BytesIO()
    resized.save(jpeg_buffer, format="JPEG", quality=85)
    display(DisplayImage(data=jpeg_buffer.getvalue(), format="jpeg"))


def show_video(video_path, width: int = 900):
    """
    Display a video by reference (not embedded), with player controls.

    The notebook stores only the path, so the video plays where the file
    exists (after the notebook has been run on this machine).
    """
    from IPython.display import HTML, display

    relative_path = os.path.relpath(video_path, Path.cwd())
    display(
        HTML(
            f'<video src="{relative_path}" width="{width}" controls loop></video>'
            f"<p><code>{video_path}</code></p>"
        )
    )
