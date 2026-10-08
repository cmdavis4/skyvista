"""
Render a time-lapse of RAMS clouds with the skyvista Blender backend.

Example (from the repository root, in the pixi ``blender`` environment)::

    HDF5_PLUGIN_PATH=/path/to/hdf5plugin/plugins \\
    pixi run -e blender python examples/rams_clouds/render_rams_cloud_animation.py \\
        /path/to/case/output  storm_animation \\
        --start-minute 42 --end-minute 120 --cadence-seconds 40 \\
        --samples 64 --blender ~/programs/blender-5.0.1-linux-x64/blender

How it works
------------
1. **Parallel export.** Each output time is exported by its own worker
   process into a single-time bundle under ``<bundle>/frames/``. Reading a
   ZFP-compressed lite file and computing the optics takes ~1 min per time
   serially, so a pool of workers turns hours into minutes. Every frame uses
   the same fixed scene transform (domain center horizontally, z = 0 at the
   ground), so all frames share one coordinate system.
2. **Assembly.** The per-frame VDB files are moved into one numbered
   sequence (``data/clouds/clouds_0001.vdb`` ...), which Blender plays as an
   animated volume. Each VDB file carries its own index-to-world transform,
   so frames cropped to different cloud footprints still line up. The
   manifest takes the camera of the frame with the largest cloud (so the
   mature storm fits in view throughout) and the union of all frames'
   hydrometeor species.
3. **Render.** One Blender run builds the ``.blend`` and renders every frame
   to ``<bundle>/animation/render/frame_####.png``. Frames already rendered
   are skipped, so an interrupted run resumes when relaunched.
4. **Video.** If an ``ffmpeg`` executable is available (``--ffmpeg``, PATH,
   or the ``imageio-ffmpeg`` package), the frames are stamped with the model
   time and encoded to ``<bundle>/<bundle name>.mp4``.

Every stage skips work that is already on disk, so the command can simply be
rerun after an interruption.
"""

import argparse
import json
import os
import shutil
import subprocess
import time as clock
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from cloud_render_config import add_render_arguments, make_cloud_export_config
from rams_lite_loader import (
    load_rams_lite_hydrometeors,
    model_time_seconds,
    read_header_coordinates,
)

# Name of the cloud object (and its VDB file stem) inside every bundle.
CLOUD_OBJECT_NAME = "clouds"


# =============================================================================
# 1. Parallel per-time export
# =============================================================================
def find_output_files(
    output_dir: Path, start_minute: float, end_minute: float, cadence_seconds: int
) -> List[Path]:
    """
    The lite files at the requested times, in time order.

    Lists every ``a-L-*-g1.h5`` file once, keys it by model time, then picks
    the times start, start + cadence, ... up to end. A requested time with no
    file is an error, since a gap would silently speed up the time-lapse.
    """
    files_by_time = {
        model_time_seconds(path): path for path in output_dir.glob("a-L-*-g1.h5")
    }
    requested_times = np.arange(
        start_minute * 60.0, end_minute * 60.0 + 0.5, cadence_seconds
    )
    missing_times = [t for t in requested_times if float(t) not in files_by_time]
    if missing_times:
        raise FileNotFoundError(
            f"No lite file at model times (s): {missing_times[:10]}"
            f"{' ...' if len(missing_times) > 10 else ''}"
        )
    return [files_by_time[float(t)] for t in requested_times]


def export_one_frame(
    frame_index: int,
    data_path: Path,
    frame_bundle_dir: Path,
    arguments: argparse.Namespace,
    origin_shift: Tuple[float, float, float],
) -> Tuple[int, float]:
    """
    Worker: export one output time to its own single-time bundle.

    Runs in a separate process, so it imports skyvista itself and returns
    only small values. Skips the time if its bundle is already complete.

    Returns:
        (frame index, export seconds; 0 if it was already done).
    """
    import skyvista as sv

    if (frame_bundle_dir / "scene.json").exists():
        return frame_index, 0.0
    start = clock.time()
    dataset = load_rams_lite_hydrometeors(data_path, top_height=arguments.top_height)
    config = make_cloud_export_config(arguments, origin_shift=origin_shift)
    sv.Scene().add_clouds(dataset, microphysics="rams").to_blender(
        frame_bundle_dir, config=config
    )
    return frame_index, clock.time() - start


def export_all_frames(
    data_paths: List[Path],
    frames_dir: Path,
    arguments: argparse.Namespace,
    origin_shift: Tuple[float, float, float],
) -> List[Path]:
    """Export every time with a process pool; return the frame bundle dirs."""
    frame_bundle_dirs = [
        frames_dir / f"frame_{index + 1:04d}" for index in range(len(data_paths))
    ]
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        futures = [
            pool.submit(
                export_one_frame,
                index,
                data_path,
                frame_bundle_dirs[index],
                arguments,
                origin_shift,
            )
            for index, data_path in enumerate(data_paths)
        ]
        for n_done, future in enumerate(as_completed(futures), start=1):
            frame_index, seconds = future.result()
            print(
                f"  exported frame {frame_index + 1:4d}"
                f" ({n_done}/{len(futures)}, {seconds:5.1f} s)",
                flush=True,
            )
    return frame_bundle_dirs


# =============================================================================
# 2. Assembly into one animated bundle
# =============================================================================
def camera_distance(manifest: Dict) -> float:
    """Distance from a manifest's camera to its target (grows with cloud size)."""
    keyframe = manifest["camera"]["keyframes"][0]
    return float(np.linalg.norm(np.subtract(keyframe["location"], keyframe["look_at"])))


def assemble_animation_bundle(
    frame_bundle_dirs: List[Path],
    animation_dir: Path,
    model_times_seconds: List[float],
) -> Path:
    """
    Merge single-time bundles into one bundle with an animated VDB sequence.

    The VDB of frame i is moved to ``data/clouds/clouds_<i>.vdb``. The
    manifest is the largest-cloud frame's manifest with three changes: the
    time range covers every frame, the species list is the union over frames
    (a species absent from a frame's file simply reads as zero), and each
    species' maximum extinction is the maximum over frames.
    """
    manifests = [
        json.loads((frame_dir / "scene.json").read_text())
        for frame_dir in frame_bundle_dirs
    ]

    # The frame whose camera stands farthest back frames the biggest cloud.
    # Frames without any hydrometeor are excluded: their camera frames the
    # whole (empty) domain, which would otherwise win.
    candidate_distances = [
        (
            camera_distance(manifest)
            if manifest["objects"][0]["material"]["species"]
            else -1.0
        )
        for manifest in manifests
    ]
    reference_index = int(np.argmax(candidate_distances))
    animation_manifest = json.loads(json.dumps(manifests[reference_index]))
    print(f"  camera from frame {reference_index + 1} (largest cloud)")

    # Move each frame's VDB into one numbered sequence.
    sequence_dir = animation_dir / "data" / CLOUD_OBJECT_NAME
    sequence_dir.mkdir(parents=True, exist_ok=True)
    for frame_number, frame_dir in enumerate(frame_bundle_dirs, start=1):
        frame_vdb = (
            frame_dir / "data" / CLOUD_OBJECT_NAME / f"{CLOUD_OBJECT_NAME}_0001.vdb"
        )
        sequence_vdb = sequence_dir / f"{CLOUD_OBJECT_NAME}_{frame_number:04d}.vdb"
        if frame_vdb.exists():
            os.replace(frame_vdb, sequence_vdb)
        elif not sequence_vdb.exists():
            raise FileNotFoundError(
                f"Missing VDB for frame {frame_number}: {frame_vdb}"
            )

    # Union of species across frames, keeping each one's largest extinction.
    species_by_name: Dict[str, Dict] = {}
    for manifest in manifests:
        for species in manifest["objects"][0]["material"]["species"]:
            known = species_by_name.get(species["name"])
            if known is None:
                species_by_name[species["name"]] = dict(species)
            else:
                known["max_extinction"] = max(
                    known["max_extinction"], species["max_extinction"]
                )
    cloud_object = animation_manifest["objects"][0]
    cloud_object["material"]["species"] = list(species_by_name.values())

    # Time range: one Blender frame per exported output time.
    frame_start = animation_manifest["time"]["frame_start"]
    animation_manifest["time"]["frame_end"] = frame_start + len(frame_bundle_dirs) - 1
    animation_manifest["time"]["data_times"] = list(model_times_seconds)
    # The camera is static: one keyframe on the first frame.
    for keyframe in animation_manifest["camera"]["keyframes"]:
        keyframe["frame"] = frame_start

    (animation_dir / "scene.json").write_text(json.dumps(animation_manifest, indent=2))
    # The Blender build script travels with every bundle.
    build_script = animation_dir / "build_scene.py"
    shutil.copyfile(frame_bundle_dirs[reference_index] / "build_scene.py", build_script)
    return build_script


# =============================================================================
# 3. Render and 4. video
# =============================================================================
def render_animation(
    blender_executable: str, build_script: Path, animation_dir: Path
) -> None:
    """Build and render in one headless Blender run (resumes if interrupted)."""
    render_dir = animation_dir / "render"
    # Remove zero-byte placeholders left by a frame that was interrupted, so
    # the resumed run renders that frame again instead of skipping it.
    if render_dir.exists():
        for placeholder in render_dir.glob("frame_*.png"):
            if placeholder.stat().st_size == 0:
                placeholder.unlink()
    command = [
        blender_executable,
        "--background",
        "--python",
        str(build_script),
        "--",
        str(animation_dir),
        "--render",
    ]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def find_ffmpeg(explicit: Optional[str]) -> Optional[str]:
    """An ffmpeg executable: explicit path, PATH, or imageio-ffmpeg's binary."""
    if explicit:
        return explicit
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def stamp_and_encode(
    animation_dir: Path,
    model_times_seconds: List[float],
    output_video: Path,
    fps: float,
    ffmpeg_executable: Optional[str],
) -> None:
    """
    Stamp each frame with its model time, then encode an H.264 MP4.

    Stamped copies go to ``render_stamped/`` so the raw frames stay
    untouched. The label is drawn with Pillow (bundled with matplotlib), which
    avoids depending on ffmpeg's optional text filter.
    """
    from PIL import Image, ImageDraw, ImageFont

    render_dir = animation_dir / "render"
    stamped_dir = animation_dir / "render_stamped"
    stamped_dir.mkdir(exist_ok=True)
    for frame_number, model_seconds in enumerate(model_times_seconds, start=1):
        frame_path = render_dir / f"frame_{frame_number:04d}.png"
        image = Image.open(frame_path).convert("RGB")
        # Text height ~3.5% of the frame height, in the top-left corner.
        font_size = max(12, image.height // 28)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()
        label = f"t = {model_seconds / 60.0:5.1f} min"
        drawer = ImageDraw.Draw(image)
        margin = font_size // 2
        drawer.text((margin + 2, margin + 2), label, fill=(0, 0, 0), font=font)
        drawer.text((margin, margin), label, fill=(255, 255, 255), font=font)
        image.save(stamped_dir / frame_path.name)

    if ffmpeg_executable is None:
        print("No ffmpeg found; stamped frames are in", stamped_dir)
        return
    command = [
        ffmpeg_executable,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(stamped_dir / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "16",
        str(output_video),
    ]
    subprocess.run(command, check=True)
    print("Video written to", output_video)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output_dir", type=Path, help="RAMS case output directory.")
    parser.add_argument("bundle_dir", type=Path, help="Working/output directory.")
    parser.add_argument("--start-minute", type=float, required=True)
    parser.add_argument("--end-minute", type=float, required=True)
    parser.add_argument(
        "--cadence-seconds",
        type=int,
        default=60,
        help="Model time between frames (a multiple of the lite output interval).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Export processes (each needs ~10-15 GB for a 400x400x330 grid).",
    )
    parser.add_argument("--fps", type=float, default=15.0, help="Video frame rate.")
    parser.add_argument("--blender", default=None, help="Blender executable.")
    parser.add_argument("--ffmpeg", default=None, help="ffmpeg executable.")
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Stop after assembling the bundle.",
    )
    add_render_arguments(parser)
    arguments = parser.parse_args()

    from skyvista.blender import _resolve_blender_executable

    data_paths = find_output_files(
        arguments.output_dir,
        arguments.start_minute,
        arguments.end_minute,
        arguments.cadence_seconds,
    )
    model_times_seconds = [model_time_seconds(path) for path in data_paths]
    print(
        f"{len(data_paths)} frames from {model_times_seconds[0] / 60:.1f} to"
        f" {model_times_seconds[-1] / 60:.1f} min every {arguments.cadence_seconds} s"
    )

    # Fixed transform for every frame: domain center horizontally, ground at 0.
    coordinates = read_header_coordinates(
        data_paths[0].with_name(data_paths[0].name.replace("-g1.h5", "-head.txt"))
    )
    origin_shift = (
        float(np.mean(coordinates["x"])),
        float(np.mean(coordinates["y"])),
        0.0,
    )

    # 1. Export every time in parallel (resumable).
    start = clock.time()
    frame_bundle_dirs = export_all_frames(
        data_paths, arguments.bundle_dir / "frames", arguments, origin_shift
    )
    print(f"Export done in {(clock.time() - start) / 60:.1f} min", flush=True)

    # 2. Assemble the animated bundle.
    animation_dir = arguments.bundle_dir / "animation"
    build_script = assemble_animation_bundle(
        frame_bundle_dirs, animation_dir, model_times_seconds
    )
    if arguments.skip_render:
        print("Bundle assembled at", animation_dir)
        return

    # 3. Render all frames in one Blender run (resumable).
    start = clock.time()
    render_animation(
        _resolve_blender_executable(arguments.blender), build_script, animation_dir
    )
    print(f"Render done in {(clock.time() - start) / 3600:.2f} h", flush=True)

    # 4. Stamp model times and encode the video.
    stamp_and_encode(
        animation_dir,
        model_times_seconds,
        arguments.bundle_dir / f"{arguments.bundle_dir.name}.mp4",
        arguments.fps,
        find_ffmpeg(arguments.ffmpeg),
    )


if __name__ == "__main__":
    main()
