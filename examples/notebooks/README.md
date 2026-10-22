# skyvista example notebooks

These notebooks show how to render atmospheric model output with skyvista's
Blender backend. Each one covers one kind of figure, from loading the data to
the final image or video. They use RAMS and CM1 runs of isolated storms and a
supercell.

| Notebook | What it shows |
| --- | --- |
| `01_storm_portrait.ipynb` | A physically based storm from RAMS or CM1 output: the shortest path to an image, framing the camera, placing the sun, an explicit camera, aerial perspective (haze), Morrison microphysics. |
| `02_winds_at_a_fixed_time.ipynb` | The 3-D wind at one output time: arrows on a vertical plane, a cloud cut open along a plane (a *cutaway*), storm-relative streamlines, streamlines inside a cutaway, and a labelled side-by-side figure. |
| `03_passive_tracer_dye.ipynb` | RAMS passive tracers drawn as a colored dye, colored by the height the air started from. |
| `04_animation_and_tracking.ipynb` | Animations over model time: a time-lapse with a fixed camera, then a camera that follows a storm along its tobac track, with the neighbouring storm faded out and per-frame captions. |
| `05_flow_at_a_fixed_time.ipynb` | One output time held still while cones move along the streamlines at the speed of the air, rendered as a seamless loop. |
| `06_parcel_trajectories.ipynb` | CM1 parcel trajectories: the parcels in a storm's updraft drawn as tapered trails, alone and inside a see-through cloud, and an animation in which the trails follow the parcels. |
| `07_squall_line_animation.ipynb` | A CM1 squall line animated without a storm track: the camera follows the centroid of the surface rain, the periodic line is tiled so its ends stay out of frame, and camera positions are tried on single still images first. |

Start with notebook 01. Notebooks 02 and 03 use its camera and sun.
Notebook 05 builds on notebook 02, and notebook 07 on notebook 04. Each notebook runs on its own.

The notebooks read the model output with
[xarrams](https://pypi.org/project/xarrams/), a separate package for RAMS
and CM1 output. skyvista does not depend on it: any xarray `Dataset` with
`x`, `y` and `z` coordinates in meters works.

## Running them

1. Edit the paths to the model output at the top of `notebook_setup.py`.
2. Start Jupyter in the pixi `blender` environment, from the repository
   root. This environment includes Blender itself (the `bpy` module),
   `xarrams` and ffmpeg:

   ```bash
   pixi run -e blender jupyter lab examples/notebooks
   ```

The renders use skyvista's "preview" quality (16 samples per pixel) and
small images, so each takes from seconds to a few minutes on a many-core
machine. Notebooks 04 and 07 are the longest, at about 15 and 30 minutes. Everything a notebook makes goes into `work/` (ignored by
git).

The command-line scripts in `../rams_clouds` and `../cm1_storms` make the
same figures at full quality. Their outputs are in the `outputs/` folders
next to them.
