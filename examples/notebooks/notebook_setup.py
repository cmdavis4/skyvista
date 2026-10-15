"""
Shared setup for the skyvista example notebooks.

Every notebook starts with ``import notebook_setup as setup``. Importing this
module:

1. tells skyvista where ffmpeg is (``$SKYVISTA_FFMPEG``, unless already
   set); Blender itself comes from the pixi environment, as the ``bpy``
   module;
2. lets h5py read ZFP-compressed RAMS output (see below);
3. silences two warnings that say nothing useful here.

It also holds the data paths the notebooks use. **Edit the paths below to
point at your own output.** The notebooks write everything they make under
``WORK_DIR`` (``examples/notebooks/work``, ignored by git).
"""

import io
import os
import warnings
from pathlib import Path

# =============================================================================
# Paths: edit these for your machine
# =============================================================================
# An ffmpeg for encoding videos.
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
# Environment
# =============================================================================
# The pixi environment's h5py was built against a slightly older HDF5.
warnings.filterwarnings("ignore", message="h5py is running against HDF5")
# xarrams opens a file once without h5netcdf's phony_dims argument, to list
# its variables; xarray then warns about the argument's default.
warnings.filterwarnings("ignore", message="The 'phony_dims' kwarg now defaults")

# RAMS lite output may be compressed with the ZFP filter, which HDF5 does not
# include. Importing hdf5plugin registers it with h5py.
import hdf5plugin  # noqa: E402, F401

# skyvista falls back to this when no ffmpeg is passed explicitly.
os.environ.setdefault("SKYVISTA_FFMPEG", str(FFMPEG_EXECUTABLE))

WORK_DIR.mkdir(exist_ok=True)


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
