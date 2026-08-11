"""
Alembic feasibility test for skyvista's Blender backend.

Purpose
-------
Before committing to Alembic (``.abc``) as the on-disk carrier for
time-varying *mesh* geometry (isosurfaces, slices, glyph meshes, trajectory
tubes), we need to confirm two things that killed the earlier attempt:

1.  We can WRITE an animated poly-mesh whose *topology changes every frame*
    (different vertex and face counts per timestep) from skyvista's own
    Python environment -- i.e. WITHOUT Blender / ``bpy`` -- using the
    ``alembic3d`` PyPI wheel (the 3D Alembic bindings, renamed to avoid the
    SQLAlchemy ``alembic`` database-migration package that almost certainly
    caused the previous confusion).

2.  A per-vertex scalar field survives the round-trip, carried two ways:
      * as baked RGB *vertex colors* (an ``OC3fGeomParam``), which is the
        robust path for reproducing a scientific colormap in Blender, and
      * as the raw scalar itself (an ``OFloatGeomParam``), which is the
        flexible path (do the color-ramp inside Blender, editable by a user).

This script does NOT need Blender. It writes an ``.abc`` and reads it back
with the same library, asserting that geometry and both attributes are
recoverable frame-by-frame and that the archive records *heterogeneous*
topology (the flag Blender uses to swap meshes instead of interpolating
vertices, which is what corrupts changing-topology caches).

The Blender *read* side is checked separately by ``blender_verify.py``.

Mapping to the real skyvista pipeline
-------------------------------------
In production these arrays come straight off the PyVista mesh returned by
``VarSpec.create_mesh(ds, time)``:
    points  <- mesh.points                      # (n_points, 3) float
    faces   <- mesh.faces reshaped / triangulated
    scalar  <- mesh[spec.geometry.scalar]       # (n_points,) float
Here we synthesise an analytic changing-topology surface so the test has no
scientific-data dependency, but the array shapes and dtypes are identical.
"""

import sys

import numpy as np

import imath
from alembic3d.Abc import IArchive, ISampleSelector, OArchive
from alembic3d.AbcCoreAbstract import TimeSampling
from alembic3d.AbcGeom import (
    GeometryScope,
    IPolyMesh,
    OC3fGeomParam,
    OC3fGeomParamSample,
    OFloatGeomParam,
    OFloatGeomParamSample,
    OPolyMesh,
    OPolyMeshSchemaSample,
)

# Names under which the attributes are stored in the archive. Blender's
# Alembic importer turns a C3f/C4f arbGeomParam into a Color Attribute and
# (Blender >= 3.x) a float arbGeomParam into a generic float attribute.
VERTEX_COLOR_PARAM_NAME = "color"
VERTEX_SCALAR_PARAM_NAME = "THETA"

# Editable-scalar spike: carry the scalar as a *float-color* attribute (gray,
# value in all channels), in two variants -- raw physical units and normalized
# to [0, 1]. Blender imports C3f params as Color Attributes, but the storage
# type decides usefulness: FLOAT_COLOR preserves magnitude (so the raw variant
# is re-rampable in physical units), while BYTE_COLOR clamps to [0, 1] (so only
# the normalized variant survives, and the ramp must map [0,1] -> clim). This
# script only *writes* both; blender_verify.py reports which storage Blender uses.
SCALAR_GRAY_RAW_PARAM_NAME = "scalar_gray_raw"
SCALAR_GRAY_NORM_PARAM_NAME = "scalar_gray_norm"
PHYSICAL_SCALAR_MIN = 290.0  # e.g. THETA in kelvin
PHYSICAL_SCALAR_MAX = 320.0

# Frame rate the archive advertises. Blender maps Alembic sample times to
# frames using the scene fps; 24 keeps the test aligned with Blender's
# default so sample i lands exactly on frame (i + 1).
FRAMES_PER_SECOND = 24.0


# -----------------------------------------------------------------------------
# Synthetic changing-topology data (stand-in for skyvista isosurfaces)
# -----------------------------------------------------------------------------
def make_viridis_like_colormap_lut(n_colors: int = 256) -> np.ndarray:
    """
    Build a small viridis-ish RGB lookup table without importing matplotlib.

    The exact colormap is irrelevant to the round-trip test; we only need a
    smooth scalar -> RGB mapping whose values we can compare after the write.
    Returns an (n_colors, 3) float array in [0, 1].
    """
    # A handful of viridis control points (perceptually close enough for a test)
    control_point_positions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    control_point_rgb = np.array(
        [
            [0.267, 0.005, 0.329],  # dark purple
            [0.229, 0.322, 0.545],  # blue
            [0.128, 0.567, 0.551],  # teal
            [0.369, 0.788, 0.383],  # green
            [0.993, 0.906, 0.144],  # yellow
        ]
    )
    sample_positions = np.linspace(0.0, 1.0, n_colors)
    # Interpolate each RGB channel independently across the control points
    lut = np.stack(
        [
            np.interp(sample_positions, control_point_positions, control_point_rgb[:, c])
            for c in range(3)
        ],
        axis=1,
    )
    return lut


def apply_colormap(scalar_values_normalized: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Map scalars already normalised to [0, 1] onto the LUT -> (n, 3) RGB."""
    lut_indices = np.clip(
        (scalar_values_normalized * (len(lut) - 1)).round().astype(int), 0, len(lut) - 1
    )
    return lut[lut_indices]


def make_changing_topology_frame(frame_index: int):
    """
    Produce one timestep of a triangulated surface whose vertex/face counts
    change from frame to frame -- the essential property that distinguishes a
    marching-cubes isosurface animation from an ordinary deforming mesh.

    We vary the grid resolution per frame so that both the vertex count and
    the face count genuinely differ across the sequence, and move a Gaussian
    bump so the scalar field (used for coloring) actually evolves.

    Returns
    -------
    points_xyz : (n_points, 3) float32
    triangle_vertex_indices : (n_triangles, 3) int32
    scalar_field : (n_points,) float32   -- raw scalar, here the bump height
    """
    # Deliberately different grid resolution each frame -> changing topology
    grid_resolution_per_dim = 8 + 3 * frame_index  # 8, 11, 14, 17, ...

    x_coordinates = np.linspace(0.0, 1.0, grid_resolution_per_dim, dtype=np.float32)
    y_coordinates = np.linspace(0.0, 1.0, grid_resolution_per_dim, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_coordinates, y_coordinates, indexing="ij")

    # A Gaussian bump whose center drifts with time -> evolving scalar field
    bump_center_x = 0.3 + 0.1 * frame_index
    bump_center_y = 0.5
    squared_distance_to_center = (x_grid - bump_center_x) ** 2 + (
        y_grid - bump_center_y
    ) ** 2
    bump_height = np.exp(-squared_distance_to_center / 0.05).astype(np.float32)

    # Stack into an (n_points, 3) point cloud; z is the bump height (relief)
    points_xyz = np.column_stack(
        [x_grid.ravel(), y_grid.ravel(), bump_height.ravel()]
    ).astype(np.float32)

    # Triangulate the regular grid (two triangles per quad cell)
    triangle_list = []
    n = grid_resolution_per_dim
    for i in range(n - 1):
        for j in range(n - 1):
            top_left = i * n + j
            top_right = i * n + (j + 1)
            bottom_left = (i + 1) * n + j
            bottom_right = (i + 1) * n + (j + 1)
            triangle_list.append([top_left, bottom_left, top_right])
            triangle_list.append([top_right, bottom_left, bottom_right])
    triangle_vertex_indices = np.array(triangle_list, dtype=np.int32)

    scalar_field = bump_height.ravel().astype(np.float32)
    return points_xyz, triangle_vertex_indices, scalar_field


# -----------------------------------------------------------------------------
# numpy -> imath array converters
# -----------------------------------------------------------------------------
def numpy_points_to_imath(points_xyz: np.ndarray) -> "imath.V3fArray":
    """Convert an (n, 3) float array to an imath.V3fArray of vertex positions."""
    n_points = len(points_xyz)
    imath_points = imath.V3fArray(n_points)
    for point_index in range(n_points):
        px, py, pz = points_xyz[point_index]
        imath_points[point_index] = imath.V3f(float(px), float(py), float(pz))
    return imath_points


def numpy_faces_to_imath(triangle_vertex_indices: np.ndarray):
    """
    Convert an (n_tri, 3) triangle index array to the flat (indices, counts)
    representation Alembic expects: one flat index stream plus a per-face
    vertex-count stream (all 3 here, since every face is a triangle).
    """
    flat_indices = triangle_vertex_indices.ravel()
    imath_face_indices = imath.IntArray(len(flat_indices))
    for k in range(len(flat_indices)):
        imath_face_indices[k] = int(flat_indices[k])

    n_faces = len(triangle_vertex_indices)
    imath_face_counts = imath.IntArray(n_faces)
    for face_index in range(n_faces):
        imath_face_counts[face_index] = 3
    return imath_face_indices, imath_face_counts


def numpy_rgb_to_imath(rgb_values: np.ndarray) -> "imath.C3fArray":
    """Convert an (n, 3) RGB float array to an imath.C3fArray of vertex colors."""
    n_colors = len(rgb_values)
    imath_colors = imath.C3fArray(n_colors)
    for color_index in range(n_colors):
        r, g, b = rgb_values[color_index]
        imath_colors[color_index] = imath.Color3f(float(r), float(g), float(b))
    return imath_colors


def numpy_scalar_to_imath(scalar_values: np.ndarray) -> "imath.FloatArray":
    """Convert an (n,) float array to an imath.FloatArray."""
    imath_scalars = imath.FloatArray(len(scalar_values))
    for scalar_index in range(len(scalar_values)):
        imath_scalars[scalar_index] = float(scalar_values[scalar_index])
    return imath_scalars


# -----------------------------------------------------------------------------
# Write path
# -----------------------------------------------------------------------------
def write_animated_alembic(output_path: str, n_frames: int):
    """
    Write an animated poly-mesh with per-frame changing topology plus a
    per-vertex color attribute and a per-vertex raw-scalar attribute.

    Returns the per-frame reference data so the read path can verify fidelity.
    """
    colormap_lut = make_viridis_like_colormap_lut()

    # Establish a global scalar range so the colormap is consistent across
    # frames (never renormalise per frame -- that would make colors drift,
    # which is a fidelity bug in an animated scientific figure).
    global_scalar_min = 0.0
    global_scalar_max = 1.0

    archive = OArchive(output_path)

    # Uniform time sampling: one sample every 1/fps seconds, starting at t=0.
    time_sampling = TimeSampling(1.0 / FRAMES_PER_SECOND, 0.0)
    time_sampling_index = archive.addTimeSampling(time_sampling)

    # Create the mesh object bound to that time sampling.
    poly_mesh = OPolyMesh(archive.getTop(), "storm_isosurface", time_sampling_index)
    mesh_schema = poly_mesh.getSchema()

    # Create the two arbitrary geom params ONCE, then set them each frame.
    arb_geom_params = mesh_schema.getArbGeomParams()
    color_param = OC3fGeomParam(
        arb_geom_params,
        VERTEX_COLOR_PARAM_NAME,
        False,  # not indexed
        GeometryScope.kVertexScope,  # one value per vertex
        1,  # extent
        time_sampling_index,
    )
    scalar_param = OFloatGeomParam(
        arb_geom_params,
        VERTEX_SCALAR_PARAM_NAME,
        False,
        GeometryScope.kVertexScope,
        1,
        time_sampling_index,
    )
    # The spike: the same scalar carried as float-color (gray) attributes.
    scalar_gray_raw_param = OC3fGeomParam(
        arb_geom_params,
        SCALAR_GRAY_RAW_PARAM_NAME,
        False,
        GeometryScope.kVertexScope,
        1,
        time_sampling_index,
    )
    scalar_gray_norm_param = OC3fGeomParam(
        arb_geom_params,
        SCALAR_GRAY_NORM_PARAM_NAME,
        False,
        GeometryScope.kVertexScope,
        1,
        time_sampling_index,
    )

    reference_frames = []
    for frame_index in range(n_frames):
        points_xyz, triangle_indices, scalar_field = make_changing_topology_frame(
            frame_index
        )

        # Bake the scalar to RGB via the fixed global range + colormap
        scalar_normalized = (scalar_field - global_scalar_min) / (
            global_scalar_max - global_scalar_min
        )
        rgb_values = apply_colormap(scalar_normalized, colormap_lut)

        # Geometry sample (this is where topology changes frame to frame)
        imath_points = numpy_points_to_imath(points_xyz)
        imath_face_indices, imath_face_counts = numpy_faces_to_imath(triangle_indices)
        mesh_sample = OPolyMeshSchemaSample(
            imath_points, imath_face_indices, imath_face_counts
        )
        mesh_schema.set(mesh_sample)

        # Attribute samples (must be reset each frame to match the new vertex count)
        color_param.set(
            OC3fGeomParamSample(
                numpy_rgb_to_imath(rgb_values), GeometryScope.kVertexScope
            )
        )
        scalar_param.set(
            OFloatGeomParamSample(
                numpy_scalar_to_imath(scalar_field), GeometryScope.kVertexScope
            )
        )

        # Spike: physical-range and normalized scalar carried as gray colors.
        physical_scalar = PHYSICAL_SCALAR_MIN + scalar_normalized * (
            PHYSICAL_SCALAR_MAX - PHYSICAL_SCALAR_MIN
        )
        gray_raw = np.repeat(physical_scalar[:, None], 3, axis=1)
        gray_norm = np.repeat(scalar_normalized[:, None], 3, axis=1)
        scalar_gray_raw_param.set(
            OC3fGeomParamSample(
                numpy_rgb_to_imath(gray_raw), GeometryScope.kVertexScope
            )
        )
        scalar_gray_norm_param.set(
            OC3fGeomParamSample(
                numpy_rgb_to_imath(gray_norm), GeometryScope.kVertexScope
            )
        )

        reference_frames.append(
            {
                "n_points": len(points_xyz),
                "n_triangles": len(triangle_indices),
                "points_xyz": points_xyz,
                "scalar_field": scalar_field,
                "rgb_values": rgb_values,
                "gray_raw": gray_raw,
                "gray_norm": gray_norm,
            }
        )

    # Archive is finalised when it goes out of scope; drop our reference.
    del mesh_schema, poly_mesh, color_param, scalar_param
    del scalar_gray_raw_param, scalar_gray_norm_param, archive
    return reference_frames


# -----------------------------------------------------------------------------
# Read path (round-trip verification, no Blender required)
# -----------------------------------------------------------------------------
def read_and_verify(input_path: str, reference_frames: list) -> bool:
    """Read the archive back and check every frame against the written data."""
    archive = IArchive(input_path)
    top_object = archive.getTop()

    # Navigate to the single mesh child we wrote.
    poly_mesh = IPolyMesh(top_object, "storm_isosurface")
    mesh_schema = poly_mesh.getSchema()

    all_checks_passed = True

    def check(condition: bool, description: str):
        nonlocal all_checks_passed
        status = "PASS" if condition else "FAIL"
        if not condition:
            all_checks_passed = False
        print(f"  [{status}] {description}")

    n_samples = mesh_schema.getNumSamples()
    check(
        n_samples == len(reference_frames),
        f"sample count {n_samples} == {len(reference_frames)} written frames",
    )

    # Topology variance: the archive should report HETEROGENEOUS topology,
    # which is exactly what tells Blender to swap meshes per frame instead of
    # trying to interpolate vertex positions (the interpolation path is what
    # corrupts changing-topology caches).
    topology_variance = mesh_schema.getTopologyVariance()
    print(f"  [INFO] topology variance code = {int(topology_variance)} "
          f"(0=constant, 1=homogeneous, 2=heterogeneous)")
    check(
        int(topology_variance) == 2,
        "archive records HETEROGENEOUS topology (changing vertex counts)",
    )

    # Pull the two arb geom params back out.
    arb_geom_params = mesh_schema.getArbGeomParams()
    from alembic3d.AbcGeom import IC3fGeomParam, IFloatGeomParam

    color_param = IC3fGeomParam(arb_geom_params, VERTEX_COLOR_PARAM_NAME)
    scalar_param = IFloatGeomParam(arb_geom_params, VERTEX_SCALAR_PARAM_NAME)
    scalar_gray_raw_param = IC3fGeomParam(
        arb_geom_params, SCALAR_GRAY_RAW_PARAM_NAME
    )
    scalar_gray_norm_param = IC3fGeomParam(
        arb_geom_params, SCALAR_GRAY_NORM_PARAM_NAME
    )

    observed_vertex_counts = []
    for frame_index, reference in enumerate(reference_frames):
        # IMPORTANT: ISampleSelector(int) is interpreted as a *time in seconds*,
        # not a sample index (a plain int hits the time constructor, leaving
        # getRequestedIndex() == -1). Passing an out-of-range time silently
        # clamps to the last sample. Select by the sample's actual time
        # instead -- which is exactly how Blender maps frames to Alembic
        # samples, so this mirrors the real import behaviour.
        sample_time_seconds = frame_index / FRAMES_PER_SECOND
        sample_selector = ISampleSelector(sample_time_seconds)

        mesh_sample = mesh_schema.getValue(sample_selector)
        positions = mesh_sample.getPositions()
        face_counts = mesh_sample.getFaceCounts()

        n_points_read = len(positions)
        n_faces_read = len(face_counts)
        observed_vertex_counts.append(n_points_read)

        check(
            n_points_read == reference["n_points"],
            f"frame {frame_index}: vertex count {n_points_read} "
            f"== {reference['n_points']}",
        )
        check(
            n_faces_read == reference["n_triangles"],
            f"frame {frame_index}: face count {n_faces_read} "
            f"== {reference['n_triangles']}",
        )

        # Verify a couple of actual vertex positions round-tripped exactly.
        first_point_read = np.array(
            [positions[0][0], positions[0][1], positions[0][2]]
        )
        position_matches = np.allclose(
            first_point_read, reference["points_xyz"][0], atol=1e-5
        )
        check(position_matches, f"frame {frame_index}: vertex[0] position matches")

        # Verify baked vertex colors round-tripped.
        color_sample = color_param.getExpandedValue(sample_selector)
        color_values = color_sample.getVals()
        check(
            len(color_values) == reference["n_points"],
            f"frame {frame_index}: color attr length "
            f"{len(color_values)} == n_points",
        )
        first_color_read = np.array(
            [color_values[0][0], color_values[0][1], color_values[0][2]]
        )
        color_matches = np.allclose(
            first_color_read, reference["rgb_values"][0], atol=1e-4
        )
        check(color_matches, f"frame {frame_index}: vertex[0] color matches colormap")

        # Verify the raw scalar attribute round-tripped.
        scalar_sample = scalar_param.getExpandedValue(sample_selector)
        scalar_values = scalar_sample.getVals()
        first_scalar_read = float(scalar_values[0])
        scalar_matches = np.isclose(
            first_scalar_read, reference["scalar_field"][0], atol=1e-5
        )
        check(scalar_matches, f"frame {frame_index}: vertex[0] raw scalar matches")

        # Verify the spike's gray float-color attributes round-tripped (the
        # write side always works for C3f; blender_verify checks the import).
        gray_raw_read = scalar_gray_raw_param.getExpandedValue(
            sample_selector
        ).getVals()
        check(
            np.isclose(gray_raw_read[0][0], reference["gray_raw"][0][0], atol=1e-3),
            f"frame {frame_index}: gray_raw float-color matches physical scalar",
        )
        gray_norm_read = scalar_gray_norm_param.getExpandedValue(
            sample_selector
        ).getVals()
        check(
            np.isclose(gray_norm_read[0][0], reference["gray_norm"][0][0], atol=1e-4),
            f"frame {frame_index}: gray_norm float-color matches normalized scalar",
        )

    # The whole point: vertex counts must actually differ across frames.
    check(
        len(set(observed_vertex_counts)) == len(observed_vertex_counts),
        f"vertex counts differ every frame: {observed_vertex_counts}",
    )

    return all_checks_passed


def main():
    output_path = "changing_topology_sequence.abc"
    n_frames = 5

    print(f"Writing animated Alembic ({n_frames} frames, changing topology) ...")
    reference_frames = write_animated_alembic(output_path, n_frames)
    print(f"  wrote {output_path}")
    print(
        "  per-frame vertex counts: "
        + str([f["n_points"] for f in reference_frames])
    )

    print("\nReading back and verifying fidelity ...")
    passed = read_and_verify(output_path, reference_frames)

    print("\n" + "=" * 60)
    if passed:
        print("RESULT: PASS -- Alembic write path is usable for skyvista.")
        print("  changing topology + baked vertex color + raw scalar +")
        print("  gray float-color scalar (raw & normalized) all round-tripped,")
        print("  with heterogeneous topology recorded.")
        print("\n  Next: run blender_verify.py inside Blender. It reports which")
        print("  attributes import and -- for the editable-scalar spike -- whether")
        print("  the gray scalar imports as FLOAT_COLOR (re-rampable in raw units)")
        print("  or BYTE_COLOR (clamped; use the normalized variant + clim).")
    else:
        print("RESULT: FAIL -- see failed checks above.")
    print("=" * 60)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
