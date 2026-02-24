from skyutils.types_skyutils import ConfigDict

DEFAULT_BLENDER_CONFIG: ConfigDict = {
    "frames_per_timestep": 12,
    "grass_field_size": 150,
    "grass_density": 1500,
    "resolution_x": 1920,
    "resolution_y": 1080,
    "render_engine": "BLENDER_EEVEE_NEXT",
    "render_samples": 128,
    "sun_energy": 4,
    "background_color": (0.02, 0.05, 0.2, 1.0),
    "background_strength": 0.4,
}
