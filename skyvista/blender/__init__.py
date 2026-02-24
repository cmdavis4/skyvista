# Blender module for atmospheric data visualization
# Re-export all public functions for backwards compatibility

# Core functions
from .blender_run import run_atmospheric_animation
from .blender_core import (
    DATA_FILE_SUFFIXES,
    deselect_all,
    multiply_scales,
    reset_scene,
    get_object_by_name,
    resize_object,
    select_object,
    translate_object,
    resize_with_constant_xy_position,
    add_object_to_scene,
    get_object_name,
    get_collection,
    create_collection,
    assign_material,
    move_to_collection,
    setup_object,
    update_object_geometry,
    create_grass_field,
    position_text_for_camera,
    set_sky_background,
    add_map,
)

# Import functions moved to utils
from .blender_core import (
    uvw_to_magnitude_and_rotation,
)

# Utility functions
from skyutils.utils import (
    recursive_reload,
)

# Camera functions
from .blender_camera import (
    setup_camera,
)

# Animation functions
from .blender_animation import (
    register_frame_change_handlers,
    set_f_curve_interpolation,
    calculate_frames_per_timestep,
    update_for_new_frame,
    setup_visibility_keyframes,
)

# Import functions
from .blender_import import (
    get_data_filepaths,
    to_data_type_collection_name,
    import_assets_library,
    import_single_data_file,
    import_data_for_time,
    import_all_timesteps,
    load_wind_data,
    import_mesh,
)

# Render functions
from .blender_render import (
    setup_render_settings,
)

# Configuration
from .blender_config import (
    DEFAULT_BLENDER_CONFIG,
)

# Types (re-export for convenience)
from skyutils.types_skyutils import (
    BlenderObject,
    BlenderCollection,
    ConfigDict,
    PathLike,
)
