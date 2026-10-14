"""
Tests for BlenderAnimation's export and assembly stages.

Rendering needs Blender and is exercised by the example scripts; the VDB
(cloud) test needs OpenVDB and is skipped without it.
"""

import json

import numpy as np
import pytest
import xarray as xr

import skyvista as sv

# Three frames, 60 s apart.
FRAME_TIMES = [0.0, 60.0, 120.0]


def wind_dataset(time_seconds: float) -> xr.Dataset:
    """A uniform eastward wind whose box drifts east with time."""
    x = np.linspace(0.0, 1000.0, 6) + time_seconds
    y = np.linspace(0.0, 1000.0, 6)
    z = np.linspace(0.0, 500.0, 3)
    shape = (len(z), len(y), len(x))
    return xr.Dataset(
        {
            "u": (("z", "y", "x"), np.full(shape, 5.0 + time_seconds / 60.0)),
            "v": (("z", "y", "x"), np.zeros(shape)),
            "w": (("z", "y", "x"), np.zeros(shape)),
        },
        coords={"x": x, "y": y, "z": z},
    )


def build_wind_frame(time_seconds: float) -> sv.Scene:
    return sv.Scene().add_vectors(
        wind_dataset(time_seconds),
        "wind",
        u="u",
        v="v",
        w="w",
        color_by="speed",
        clim=(0.0, 10.0),
    )


def make_animation(tmp_path, **kwargs) -> sv.BlenderAnimation:
    return sv.BlenderAnimation(
        tmp_path / "animation",
        FRAME_TIMES,
        build_wind_frame,
        config=sv.BlenderExportConfig(),
        **kwargs,
    )


@pytest.mark.parametrize("workers", [1, 2])
def test_export_and_assemble_renumber_point_frames(tmp_path, workers):
    animation = make_animation(tmp_path)
    animation.export(workers=workers)
    bundle_dir = animation.assemble()
    manifest = json.loads((bundle_dir / "scene.json").read_text())

    (wind_entry,) = manifest["objects"]
    assert wind_entry["geometry"]["n_frames"] == 3
    for frame_number in (1, 2, 3):
        point_file = bundle_dir / wind_entry["geometry"]["path_pattern"].replace(
            "####", f"{frame_number:04d}"
        )
        assert point_file.exists()
    # Frame 3's points are frame 3's data: the box drifted 120 m east.
    with np.load(
        bundle_dir / wind_entry["geometry"]["path_pattern"].replace("####", "0003")
    ) as frame_three:
        with np.load(
            bundle_dir / wind_entry["geometry"]["path_pattern"].replace("####", "0001")
        ) as frame_one:
            shift = (
                frame_three["positions"][:, 0].min()
                - frame_one["positions"][:, 0].min()
            )
    # Positions are in data units (meters) relative to the fixed origin shift.
    assert shift == pytest.approx(120.0, rel=1e-6)
    assert manifest["time"]["frame_end"] - manifest["time"]["frame_start"] == 2
    assert (bundle_dir / "build_scene.py").exists()


def test_every_frame_shares_the_first_frames_transform(tmp_path):
    animation = make_animation(tmp_path)
    animation.export(workers=1)
    shifts = {
        json.dumps(
            json.loads((animation.frame_bundle_dir(index) / "scene.json").read_text())[
                "transform"
            ]["origin_shift"]
        )
        for index in range(3)
    }
    assert len(shifts) == 1


def test_export_resumes_without_redoing_frames(tmp_path):
    calls = []

    def counting_build_frame(time_seconds):
        calls.append(time_seconds)
        return build_wind_frame(time_seconds)

    animation = sv.BlenderAnimation(
        tmp_path / "animation",
        FRAME_TIMES,
        counting_build_frame,
        config=sv.BlenderExportConfig(),
    )
    animation.export(workers=1)
    assert sorted(calls) == FRAME_TIMES
    animation.export(workers=1)
    assert sorted(calls) == FRAME_TIMES


def test_camera_path_gives_one_keyframe_per_frame(tmp_path):
    path = sv.CameraPath(
        [
            sv.CameraKeyframe(0.0, azimuth_deg=-90.0, horizontal_distance=3000.0),
            sv.CameraKeyframe(120.0, azimuth_deg=0.0, horizontal_distance=3000.0),
        ],
        target=(500.0, 500.0),
    )
    animation = make_animation(tmp_path, camera_path=path)
    animation.export(workers=1)
    manifest = json.loads((animation.assemble() / "scene.json").read_text())
    keyframes = manifest["camera"]["keyframes"]
    assert [keyframe["frame"] for keyframe in keyframes] == [1, 2, 3]
    # Last frame: azimuth 0, so the camera is due east of the target.
    assert keyframes[2]["location"][0] == pytest.approx(3500.0)
    assert keyframes[2]["location"][1] == pytest.approx(500.0, abs=1e-6)


def test_objects_missing_from_a_frame_are_reported(tmp_path):
    def build_frame_missing_wind_late(time_seconds):
        if time_seconds > 100:
            return sv.Scene().add_vectors(
                wind_dataset(time_seconds), "gusts", u="u", v="v", w="w", clim=(0, 10)
            )
        return build_wind_frame(time_seconds)

    animation = sv.BlenderAnimation(
        tmp_path / "animation",
        FRAME_TIMES,
        build_frame_missing_wind_late,
        config=sv.BlenderExportConfig(),
    )
    animation.export(workers=1)
    with pytest.raises(RuntimeError, match="missing from frames"):
        animation.assemble()


def test_compose_applies_the_template_to_every_frame(tmp_path):
    """Compose reads each frame's render; here, stand-in images."""
    import matplotlib.image as mpimg

    animation = make_animation(tmp_path)
    animation.render_dir.mkdir(parents=True)
    for index in range(3):
        mpimg.imsave(animation.render_path(index), np.full((40, 60, 3), 0.3 * index))
    figure = sv.compose.Figure(
        panels=[
            sv.compose.Panel(lambda frame: frame.render_path, [sv.compose.TimeStamp()])
        ]
    )
    animation.compose(figure, workers=2)
    composed = sorted(animation.composed_dir.glob("frame_*.png"))
    assert [path.name for path in composed] == [
        "frame_0001.png",
        "frame_0002.png",
        "frame_0003.png",
    ]


class TestCloudAnimation:
    @pytest.fixture(autouse=True)
    def _require_openvdb(self):
        pytest.importorskip("openvdb")

    def test_cloud_frames_become_one_vdb_sequence(self, tmp_path):
        def build_cloud_frame(time_seconds):
            x = np.arange(0.0, 2000.0, 100.0)
            z = np.arange(0.0, 2000.0, 100.0)
            mixing_ratio = np.zeros((len(z), len(x), len(x)))
            mixing_ratio[5:10, 8:12, 8:12] = 1e-3 * (1 + time_seconds / 60.0)
            # Rain appears only in the last frame.
            rain = np.zeros_like(mixing_ratio)
            if time_seconds > 100:
                rain[2:5, 9:11, 9:11] = 1e-4
            dataset = xr.Dataset(
                {
                    "qc": (("z", "y", "x"), mixing_ratio),
                    "qr": (("z", "y", "x"), rain),
                },
                coords={"x": x, "y": x, "z": z},
            )
            return sv.Scene().add_clouds(dataset, microphysics="generic")

        animation = sv.BlenderAnimation(
            tmp_path / "animation", FRAME_TIMES, build_cloud_frame
        )
        animation.export(workers=1)
        manifest = json.loads((animation.assemble() / "scene.json").read_text())
        (cloud_entry,) = manifest["objects"]
        pattern = cloud_entry["geometry"]["path_pattern"]
        for frame_number in (1, 2, 3):
            assert (
                animation.bundle_dir / pattern.replace("####", f"{frame_number:04d}")
            ).exists()
        # Rain exists only in frame 3, but the animation's material has it.
        species_names = {
            species["name"] for species in cloud_entry["material"]["species"]
        }
        assert "rain" in species_names
