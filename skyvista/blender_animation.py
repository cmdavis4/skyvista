"""
Blender animations of model output that is too large to hold in memory.

``Scene.to_blender()`` exports a scene from one dataset holding every time.
A storm simulation's output does not fit that pattern: each output time is
its own multi-gigabyte file, and each frame may need processing of its own
(cropping, isolating a tracked cell). :class:`BlenderAnimation` instead takes
the list of times and a function ``build_frame(time) -> Scene``, and runs a
pipeline whose every stage skips work already on disk, so an interrupted run
resumes when rerun:

1. :meth:`~BlenderAnimation.export` -- each time is exported to its own
   single-frame bundle under ``<output>/frames/``, by a pool of worker
   processes. The first frame is exported alone first; its scene transform
   (the data point mapped to the Blender origin) is then fixed for every
   other frame, so all frames share one coordinate system.
2. :meth:`~BlenderAnimation.assemble` -- the frame bundles are merged into
   one animation bundle, ``<output>/bundle/``. Every per-frame data file
   (cloud and volume VDBs, point and streamline ``.npz`` files) is named by
   a ``####`` pattern, so merging is renumbering: frame *i*'s file becomes
   number *i* (hard-linked, not copied). Cloud species are unioned over the
   frames. The camera comes from a :class:`~skyvista.camera_path.CameraPath`
   (one keyframe per frame), or else is the static camera of the frame with
   the largest scene.
3. :meth:`~BlenderAnimation.render` -- Blender builds ``bundle.blend`` once,
   then one or more Blender processes render it. Each frame is claimed with
   a placeholder file when its render starts and skipped when it exists, so
   processes share the frames without coordination and a rerun resumes.
4. :meth:`~BlenderAnimation.compose` -- a :class:`skyvista.compose.Figure`
   template (time stamp, titles, colorbars...) is applied to every rendered
   frame, into ``<output>/composed/``. Raw renders are kept, so overlays can
   change without re-rendering.
5. :meth:`~BlenderAnimation.encode` -- ffmpeg encodes the frames to H.264.

Surface meshes (contours, slices, trajectory tubes: the Alembic carrier) are
not supported in animations yet; use the point and streamline carriers.
"""

import copy
import json
import os
import shutil
import subprocess
import time as clock
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from .camera_path import CameraPath
from .compose import Figure, FrameContext

# Marker written into a frame bundle once its export finished.
FRAME_COMPLETE_MARKER = ".skyvista_frame_complete"

# Carriers whose per-frame files follow a '####' path pattern.
SEQUENCE_CARRIERS = ("vdb_sequence", "point_glyphs", "polylines")

# The animation being exported or composed, for forked worker processes.
# Workers are forked, so they inherit it (and the user's build_frame, which
# may be a closure or lambda) without pickling.
_ACTIVE_ANIMATION: Optional["BlenderAnimation"] = None
_ACTIVE_FIGURE: Optional[Union[Figure, Callable[[FrameContext], Figure]]] = None


def default_time_seconds(time: Any) -> float:
    """
    Seconds for a model time given as a number, timedelta or numpy timedelta.

    Datetimes have no meaningful zero, so they need an explicit
    ``time_seconds`` function on the animation.
    """
    if isinstance(time, np.timedelta64):
        return float(time / np.timedelta64(1, "s"))
    if hasattr(time, "total_seconds"):
        return float(time.total_seconds())
    if isinstance(time, (np.datetime64,)) or hasattr(time, "year"):
        raise TypeError(
            "Datetime frame times need time_seconds=..., a function giving "
            "seconds since a reference time."
        )
    return float(time)


def encode_video(
    frames_pattern,
    output_path,
    fps: float,
    start_number: int = 1,
    repeat: int = 1,
    crf: int = 20,
    ffmpeg: Optional[str] = None,
) -> Optional[Path]:
    """
    Encode numbered frames to an H.264 MP4 that plays everywhere.

    The frames are padded to even dimensions and stored as yuv420p, which most
    players require. ``repeat`` plays the whole sequence that many times in a
    row, e.g. to lengthen a seamless loop (such as a hold whose clock advance
    is a whole number of marker travel times) without rendering it again.

    Args:
        frames_pattern: printf-style frame path, e.g. ``render/frame_%04d.png``.
        output_path: Destination video.
        fps: Frame rate.
        start_number: Number of the first frame.
        repeat: Times the sequence is played.
        crf: x264 quality (lower is better; 18-23 is typical).
        ffmpeg: ffmpeg executable (see :func:`find_ffmpeg`).

    Returns:
        The video path, or None if no ffmpeg was found.
    """
    executable = find_ffmpeg(ffmpeg)
    if executable is None:
        print("No ffmpeg found; pass ffmpeg=... or set SKYVISTA_FFMPEG.")
        return None
    output_path = Path(output_path)
    command = [
        executable,
        "-y",
        "-loglevel",
        "error",
        # Extra passes through the input, before it is opened.
        "-stream_loop",
        str(max(int(repeat), 1) - 1),
        "-framerate",
        str(fps),
        "-start_number",
        str(start_number),
        "-i",
        str(frames_pattern),
        # yuv420p (needed by most players) requires even dimensions.
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        "slow",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    print(f"Video: {output_path}", flush=True)
    return output_path


def find_ffmpeg(explicit: Optional[str] = None) -> Optional[str]:
    """
    An ffmpeg executable, or None.

    Looked up in order: the explicit path, ``$SKYVISTA_FFMPEG``, ``ffmpeg`` on
    the PATH, and the binary bundled with the ``imageio-ffmpeg`` package.
    """
    for candidate in (explicit, os.environ.get("SKYVISTA_FFMPEG")):
        if candidate:
            return str(candidate)
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def _link_or_copy(source: Path, destination: Path) -> None:
    """Hard-link a file into place (instant, no extra disk), else copy it."""
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _export_frame_worker(frame_index: int) -> int:
    """Export one frame in a forked worker (see ``_ACTIVE_ANIMATION``)."""
    _ACTIVE_ANIMATION._export_frame(frame_index)
    return frame_index


def _compose_frame_worker(frame_index: int) -> int:
    """Compose one frame in a forked worker (see ``_ACTIVE_FIGURE``)."""
    _ACTIVE_ANIMATION._compose_frame(frame_index, _ACTIVE_FIGURE)
    return frame_index


class BlenderAnimation:
    """
    A Blender animation built frame by frame from ``build_frame(time)``.

    Args:
        output_dir: Directory for every stage's output.
        times: The model times to animate, one per frame, in order. Each is
            passed to ``build_frame`` as given.
        build_frame: Function returning the :class:`skyvista.Scene` of one
            time. It runs in worker processes; it should load only that
            time's data.
        config: Export settings shared by every frame. Its
            ``transform.origin_shift`` is fixed from the first frame if unset.
        camera_path: The camera's motion; None uses a static camera.
        time_seconds: Converts a frame time to seconds, for the camera path
            and time stamps (default: numbers and timedeltas as seconds).
        fps: Frames per second of the encoded video.
    """

    def __init__(
        self,
        output_dir,
        times: Sequence[Any],
        build_frame: Callable[[Any], Any],
        config=None,
        camera_path: Optional[CameraPath] = None,
        time_seconds: Optional[Callable[[Any], float]] = None,
        fps: float = 24.0,
    ):
        from .blender import BlenderExportConfig

        self.output_dir = Path(output_dir)
        self.times = list(times)
        if not self.times:
            raise ValueError("A BlenderAnimation needs at least one time.")
        self.build_frame = build_frame
        self.config = copy.deepcopy(config) if config is not None else None
        if self.config is None:
            self.config = BlenderExportConfig.for_clouds()
        self.camera_path = camera_path
        self.time_seconds_fn = time_seconds or default_time_seconds
        self.fps = fps

    # ------------------------------------------------------------------ paths
    @property
    def n_frames(self) -> int:
        return len(self.times)

    @property
    def frames_dir(self) -> Path:
        return self.output_dir / "frames"

    @property
    def bundle_dir(self) -> Path:
        return self.output_dir / "bundle"

    @property
    def render_dir(self) -> Path:
        return self.bundle_dir / "render"

    @property
    def composed_dir(self) -> Path:
        return self.output_dir / "composed"

    def frame_bundle_dir(self, frame_index: int) -> Path:
        return self.frames_dir / f"frame_{frame_index + 1:04d}"

    def time_seconds(self, frame_index: int) -> float:
        return self.time_seconds_fn(self.times[frame_index])

    def render_path(self, frame_index: int) -> Path:
        """The rendered image of a frame (Blender numbers from frame_start)."""
        frame_number = self.config.frame_start + frame_index
        return self.render_dir / f"frame_{frame_number:04d}.png"

    def frame_context(self, frame_index: int) -> FrameContext:
        return FrameContext(
            index=frame_index,
            n_frames=self.n_frames,
            time=self.times[frame_index],
            time_seconds=self.time_seconds(frame_index),
            render_path=self.render_path(frame_index),
        )

    # ----------------------------------------------------------------- export
    def _frame_is_exported(self, frame_index: int) -> bool:
        return (self.frame_bundle_dir(frame_index) / FRAME_COMPLETE_MARKER).exists()

    def _frame_config(self, frame_index: int):
        """The export config of one frame (camera placed on the path)."""
        frame_config = copy.deepcopy(self.config)
        if self.camera_path is not None:
            # Not used by the assembled animation, but keeps each frame bundle
            # a correct still of its own frame.
            state = self.camera_path.at(self.time_seconds(frame_index))
            frame_config.camera.location = state.location
            frame_config.camera.look_at = state.look_at
            frame_config.camera.lens_mm = state.lens_mm
        return frame_config

    def _export_frame(self, frame_index: int) -> None:
        """Export one frame's bundle; a partial earlier attempt is replaced."""
        frame_dir = self.frame_bundle_dir(frame_index)
        if self._frame_is_exported(frame_index):
            return
        if frame_dir.exists():
            shutil.rmtree(frame_dir)
        scene = self.build_frame(self.times[frame_index])
        scene.to_blender(frame_dir, config=self._frame_config(frame_index))
        (frame_dir / FRAME_COMPLETE_MARKER).write_text(
            json.dumps({"time_seconds": self.time_seconds(frame_index)})
        )

    def _fix_transform_from_first_frame(self) -> None:
        """Every frame uses the first frame's origin shift."""
        if self.config.transform.origin_shift is not None:
            return
        self._export_frame(0)
        manifest = json.loads((self.frame_bundle_dir(0) / "scene.json").read_text())
        self.config.transform.origin_shift = tuple(
            manifest["transform"]["origin_shift"]
        )

    def export(self, workers: int = 4) -> None:
        """
        Export every frame not already exported, ``workers`` at a time.

        Raises RuntimeError listing the frames that failed, after attempting
        all of them.
        """
        global _ACTIVE_ANIMATION

        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._fix_transform_from_first_frame()
        pending = [
            frame_index
            for frame_index in range(self.n_frames)
            if not self._frame_is_exported(frame_index)
        ]
        print(
            f"Export: {self.n_frames - len(pending)} of {self.n_frames} frames "
            f"already done; exporting {len(pending)} with {workers} workers.",
            flush=True,
        )
        if not pending:
            return

        failures: Dict[int, str] = {}
        start = clock.time()
        _ACTIVE_ANIMATION = self
        try:
            if workers <= 1:
                for done_count, frame_index in enumerate(pending, start=1):
                    try:
                        self._export_frame(frame_index)
                    except Exception:
                        failures[frame_index] = traceback.format_exc()
                    self._report_progress("Export", done_count, len(pending), start)
            else:
                with ProcessPoolExecutor(
                    max_workers=workers, mp_context=get_context("fork")
                ) as pool:
                    futures = {
                        pool.submit(_export_frame_worker, frame_index): frame_index
                        for frame_index in pending
                    }
                    for done_count, future in enumerate(as_completed(futures), 1):
                        try:
                            future.result()
                        except Exception:
                            failures[futures[future]] = traceback.format_exc()
                        self._report_progress("Export", done_count, len(pending), start)
        finally:
            _ACTIVE_ANIMATION = None
        if failures:
            first_index = min(failures)
            raise RuntimeError(
                f"{len(failures)} frame(s) failed to export: "
                f"{sorted(index + 1 for index in failures)}. First failure "
                f"(frame {first_index + 1}):\n{failures[first_index]}"
            )

    @staticmethod
    def _report_progress(stage: str, done: int, total: int, start: float) -> None:
        """Print progress every ~5% of the work, with a time estimate."""
        if done != total and done % max(1, total // 20):
            return
        elapsed = clock.time() - start
        remaining = elapsed / done * (total - done)
        print(
            f"  {stage}: {done}/{total} ({elapsed / 60:.1f} min elapsed, "
            f"~{remaining / 60:.1f} min left)",
            flush=True,
        )

    # --------------------------------------------------------------- assemble
    def assemble(self) -> Path:
        """
        Merge the frame bundles into one animation bundle.

        Returns:
            The animation bundle directory.
        """
        from .blender import _copy_build_script

        missing = [
            frame_index + 1
            for frame_index in range(self.n_frames)
            if not self._frame_is_exported(frame_index)
        ]
        if missing:
            raise RuntimeError(f"Frames not exported yet: {missing[:20]} ...")
        manifests = [
            json.loads((self.frame_bundle_dir(index) / "scene.json").read_text())
            for index in range(self.n_frames)
        ]
        self.bundle_dir.mkdir(parents=True, exist_ok=True)

        # Every frame must share one coordinate system.
        transforms = {json.dumps(manifest["transform"]) for manifest in manifests}
        if len(transforms) > 1:
            raise RuntimeError(
                "Frame bundles have different scene transforms; export them "
                "with one fixed config.transform.origin_shift."
            )

        merged = copy.deepcopy(manifests[0])
        merged["objects"] = [
            self._merge_object(name, manifests)
            for name in self._object_names(manifests)
        ]
        merged["bounds"] = self._union_bounds(manifests)
        merged["time"] = {
            **merged.get("time", {}),
            "frame_start": self.config.frame_start,
            "frame_end": self.config.frame_start + self.n_frames - 1,
            "fps": self.fps,
            "times_seconds": [self.time_seconds(i) for i in range(self.n_frames)],
        }
        merged["camera"] = self._merged_camera(manifests)
        # The flow clock (streamline markers) runs with model time.
        merged["flow_clock"] = {
            "keyframes": [
                {
                    "frame": self.config.frame_start + frame_index,
                    "seconds": self.time_seconds(frame_index),
                }
                for frame_index in range(self.n_frames)
            ]
        }

        # Colormap tables and the build script.
        for frame_index in range(self.n_frames):
            assets_dir = self.frame_bundle_dir(frame_index) / "assets"
            if assets_dir.exists():
                shutil.copytree(
                    assets_dir, self.bundle_dir / "assets", dirs_exist_ok=True
                )
        _copy_build_script(self.bundle_dir)
        (self.bundle_dir / "scene.json").write_text(json.dumps(merged, indent=2))
        print(f"Assembled {self.n_frames} frames into {self.bundle_dir}", flush=True)
        return self.bundle_dir

    @staticmethod
    def _object_names(manifests: List[Dict]) -> List[str]:
        """Object names in first-appearance order; each must be in every frame."""
        names: List[str] = []
        for manifest in manifests:
            for entry in manifest["objects"]:
                if entry["name"] not in names:
                    names.append(entry["name"])
        for name in names:
            absent = [
                frame_index + 1
                for frame_index, manifest in enumerate(manifests)
                if all(entry["name"] != name for entry in manifest["objects"])
            ]
            if absent:
                raise RuntimeError(
                    f"Object {name!r} is missing from frames {absent[:20]}; every "
                    "frame's scene must hold the same objects (an empty one is "
                    "fine, e.g. a cloud with no condensate yet)."
                )
        return names

    def _merge_object(self, name: str, manifests: List[Dict]) -> Dict:
        """One object's animation entry, with its files renumbered by frame."""
        entries = [
            next(entry for entry in manifest["objects"] if entry["name"] == name)
            for manifest in manifests
        ]
        merged_entry = copy.deepcopy(entries[0])
        geometry = merged_entry["geometry"]
        carrier = geometry.get("carrier")
        if carrier not in SEQUENCE_CARRIERS:
            raise NotImplementedError(
                f"Object {name!r} uses the {carrier!r} carrier, which animations "
                "do not support yet (surface meshes are baked into Alembic "
                "files that cannot be renumbered)."
            )
        path_pattern = geometry["path_pattern"]

        # Frame i's single file becomes file number i of the sequence.
        for frame_index, entry in enumerate(entries):
            source = self.frame_bundle_dir(frame_index) / entry["geometry"][
                "path_pattern"
            ].replace("####", "0001")
            destination = self.bundle_dir / path_pattern.replace(
                "####", f"{frame_index + 1:04d}"
            )
            _link_or_copy(source, destination)
        geometry["n_frames"] = self.n_frames

        material = merged_entry.get("material", {})
        # Clouds: the union of species over all frames, at their largest.
        if material.get("type") == "cloud":
            species_by_name: Dict[str, Dict] = {}
            for entry in entries:
                for species in entry["material"].get("species", []):
                    known = species_by_name.get(species["name"])
                    if (
                        known is None
                        or species["max_extinction"] > known["max_extinction"]
                    ):
                        species_by_name[species["name"]] = species
            material["species"] = list(species_by_name.values())
        # A colormap whose limits change from frame to frame would recolor
        # the same value over time.
        color_limits = {
            json.dumps(entry.get("material", {}).get("coloring", {}).get("clim"))
            for entry in entries
        }
        if len(color_limits) > 1:
            warnings.warn(
                f"Object {name!r} has different color limits in different "
                "frames, so its colors are not comparable over time. Pass an "
                "explicit clim.",
                stacklevel=3,
            )
        return merged_entry

    @staticmethod
    def _frame_has_content(manifest: Dict) -> bool:
        """
        Whether a frame shows anything.

        A cloud with no condensate yet is exported over the whole (empty)
        domain, so its bounds and camera describe the domain, not a cloud.
        Such frames must not size the scene or place a static camera.
        """
        for entry in manifest["objects"]:
            material = entry.get("material", {})
            if material.get("type") != "cloud" or material.get("species"):
                return True
        return False

    def _content_manifests(self, manifests: List[Dict]) -> List[Dict]:
        """The frames that show something (all of them if none do)."""
        with_content = [m for m in manifests if self._frame_has_content(m)]
        return with_content or manifests

    def _union_bounds(self, manifests: List[Dict]) -> Optional[List[List[float]]]:
        """Bounding box of every frame that shows something."""
        corners = [
            manifest["bounds"]
            for manifest in self._content_manifests(manifests)
            if manifest.get("bounds")
        ]
        if not corners:
            return None
        corners_array = np.array(corners, dtype=float)
        return [
            corners_array[:, 0, :].min(axis=0).tolist(),
            corners_array[:, 1, :].max(axis=0).tolist(),
        ]

    def _merged_camera(self, manifests: List[Dict]) -> Optional[Dict]:
        """The camera path's keyframes, or the largest frame's static camera."""
        if self.camera_path is not None:
            keyframes = self.camera_path.manifest_keyframes(
                [self.time_seconds(i) for i in range(self.n_frames)],
                frame_start=self.config.frame_start,
            )
            return {
                "type": "perspective",
                "lens_mm": keyframes[0]["lens_mm"],
                "keyframes": keyframes,
            }

        # Static: the camera that frames the largest scene keeps every frame
        # in view.
        def diagonal(manifest):
            bounds = manifest.get("bounds")
            if not bounds:
                return 0.0
            return float(np.linalg.norm(np.subtract(bounds[1], bounds[0])))

        largest = max(self._content_manifests(manifests), key=diagonal)
        camera = copy.deepcopy(largest.get("camera"))
        if camera and camera.get("keyframes"):
            camera["keyframes"] = [
                {**camera["keyframes"][0], "frame": self.config.frame_start}
            ]
        return camera

    # ----------------------------------------------------------------- render
    def render(
        self,
        blender_executable=None,
        processes: int = 1,
        threads: Optional[int] = None,
    ) -> None:
        """
        Build ``bundle.blend`` (if stale) and render every missing frame.

        Args:
            blender_executable: Blender command (see ``Scene.to_blender``).
            processes: Blender processes rendering in parallel. Each claims
                frames with placeholders, so they share the work.
            threads: CPU threads per process; None splits the machine's cores
                evenly between the processes.
        """
        from .blender import _launch_blender_build, _resolve_blender_executable

        executable = _resolve_blender_executable(blender_executable)
        manifest_path = self.bundle_dir / "scene.json"
        blend_path = self.bundle_dir / f"{self.bundle_dir.name}.blend"
        if (
            not blend_path.exists()
            or blend_path.stat().st_mtime < manifest_path.stat().st_mtime
        ):
            _launch_blender_build(
                executable, self.bundle_dir / "build_scene.py", self.bundle_dir
            )

        # Zero-byte files are placeholders of renders that were interrupted.
        self.render_dir.mkdir(parents=True, exist_ok=True)
        for frame_path in self.render_dir.glob("frame_*.png"):
            if frame_path.stat().st_size == 0:
                frame_path.unlink()
        n_missing = sum(
            not self.render_path(index).exists() for index in range(self.n_frames)
        )
        print(f"Render: {n_missing} of {self.n_frames} frames to render.", flush=True)
        if n_missing == 0:
            return

        threads = threads or max(1, (os.cpu_count() or 1) // processes)
        logs_dir = self.bundle_dir / "render_logs"
        logs_dir.mkdir(exist_ok=True)
        running = []
        for process_index in range(processes):
            log_file = open(logs_dir / f"process_{process_index + 1}.log", "a")
            command = [
                executable,
                "--background",
                str(blend_path),
                "--threads",
                str(threads),
                "--render-anim",
            ]
            running.append(
                (
                    subprocess.Popen(
                        command, stdout=log_file, stderr=subprocess.STDOUT
                    ),
                    log_file,
                )
            )
            # Stagger the starts so the processes claim different frames.
            clock.sleep(2.0)
        for process, log_file in running:
            process.wait()
            log_file.close()
        failed = [process.returncode for process, _ in running if process.returncode]
        n_missing = sum(
            not self.render_path(index).exists() for index in range(self.n_frames)
        )
        if failed or n_missing:
            raise RuntimeError(
                f"Rendering incomplete: {n_missing} frames missing, exit codes "
                f"{failed}. See {logs_dir}."
            )

    # ---------------------------------------------------------------- compose
    def _compose_frame(self, frame_index: int, figure) -> None:
        context = self.frame_context(frame_index)
        frame_figure = (
            figure(context)
            if callable(figure) and not isinstance(figure, Figure)
            else figure
        )
        frame_figure.render(
            self.composed_dir / f"frame_{frame_index + 1:04d}.png", context
        )

    def compose(self, figure, workers: int = 8, overwrite: bool = False) -> None:
        """
        Apply a figure template to every rendered frame.

        Args:
            figure: A :class:`skyvista.compose.Figure` whose per-frame sources
                read the :class:`~skyvista.compose.FrameContext` (its
                ``render_path`` is this animation's render of the frame), or
                a function of the context returning a Figure.
            workers: Frames composed in parallel.
            overwrite: Recompose frames that already exist, e.g. after
                changing an overlay.
        """
        global _ACTIVE_ANIMATION, _ACTIVE_FIGURE

        self.composed_dir.mkdir(parents=True, exist_ok=True)
        pending = [
            frame_index
            for frame_index in range(self.n_frames)
            if overwrite
            or not (self.composed_dir / f"frame_{frame_index + 1:04d}.png").exists()
        ]
        print(f"Compose: {len(pending)} of {self.n_frames} frames.", flush=True)
        _ACTIVE_ANIMATION, _ACTIVE_FIGURE = self, figure
        try:
            if workers <= 1:
                for frame_index in pending:
                    self._compose_frame(frame_index, figure)
            else:
                with ProcessPoolExecutor(
                    max_workers=workers, mp_context=get_context("fork")
                ) as pool:
                    for future in as_completed(
                        [pool.submit(_compose_frame_worker, index) for index in pending]
                    ):
                        future.result()
        finally:
            _ACTIVE_ANIMATION, _ACTIVE_FIGURE = None, None

    # ----------------------------------------------------------------- encode
    def encode(
        self,
        output_path=None,
        source: str = "composed",
        fps: Optional[float] = None,
        crf: int = 20,
        ffmpeg: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Encode the frames to an H.264 MP4 that plays everywhere.

        Args:
            output_path: Destination; default ``<output>/<output name>.mp4``.
            source: "composed" or "render" frames.
            fps: Frame rate; default the animation's.
            crf: x264 quality (lower is better; 18-23 is typical).
            ffmpeg: ffmpeg executable (see :func:`find_ffmpeg`).

        Returns:
            The video path, or None if no ffmpeg was found.
        """
        if source == "composed":
            frames_pattern = self.composed_dir / "frame_%04d.png"
            start_number = 1
        else:
            frames_pattern = self.render_dir / "frame_%04d.png"
            start_number = self.config.frame_start
        return encode_video(
            frames_pattern,
            output_path or self.output_dir / f"{self.output_dir.name}.mp4",
            fps=fps or self.fps,
            start_number=start_number,
            crf=crf,
            ffmpeg=ffmpeg,
        )

    # -------------------------------------------------------------- all of it
    def run(
        self,
        figure=None,
        export_workers: int = 4,
        blender_executable=None,
        render_processes: int = 1,
        compose_workers: int = 8,
        ffmpeg: Optional[str] = None,
    ) -> Optional[Path]:
        """Export, assemble, render, compose (if a figure is given), encode."""
        self.export(workers=export_workers)
        self.assemble()
        self.render(blender_executable=blender_executable, processes=render_processes)
        if figure is not None:
            self.compose(figure, workers=compose_workers)
        return self.encode(
            source="composed" if figure is not None else "render", ffmpeg=ffmpeg
        )
