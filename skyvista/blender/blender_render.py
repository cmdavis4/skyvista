from pathlib import Path
from typing import Tuple
import bpy
from skyutils.types_skyutils import PathLike


def setup_render_settings(
    output_dir: PathLike,
    resolution: Tuple[int, int] = (1920, 1080),
    engine: str = "BLENDER_EEVEE_NEXT",
    quality: str = "LOW",
) -> bpy.types.RenderSettings:
    """
    Configure Blender render settings optimized for atmospheric animation.

    Sets up render engine, resolution, output format, and quality settings
    suitable for atmospheric visualization. Supports both Cycles and EEVEE
    render engines with different quality presets.

    Args:
        output_dir: Directory where rendered frames will be saved
        resolution: (width, height) tuple for render resolution (default: 1920x1080)
        engine: Render engine to use ("CYCLES" or "BLENDER_EEVEE_NEXT")
        quality: Quality preset ("DOGWATER", "LOW", "MEDIUM", "HIGH", "ULTRA")

    Returns:
        Blender render settings object for further customization

    Note:
        - EEVEE is faster for previews, Cycles for final quality
        - Higher quality settings increase render time significantly
        - Output format is set to PNG with RGB color mode
    """
    scene = bpy.context.scene
    render = scene.render

    # Need to turn this on since we're using callback that alter data
    bpy.types.RenderSettings.use_lock_interface = True

    # Basic settings
    render.engine = engine
    render.resolution_x = resolution[0]
    render.resolution_y = resolution[1]
    render.resolution_percentage = 100

    # Output settings
    render.filepath = str(Path(output_dir) / "frame")
    render.image_settings.color_mode = "RGB"
    # Set exposure
    scene.view_settings.exposure = -3

    qs = {
        "DOGWATER": {
            "samples": 16,
            "denoise_samples": 8,
            "noise_threshold": 2.0,
            "volume_step_rate": 10,
            "volume_max_steps": 16,
        },
        "LOW": {"samples": 64, "denoise_samples": 32, "noise_threshold": 1.0},
        "MEDIUM": {"samples": 128, "denoise_samples": 64},
        "HIGH": {"samples": 256, "denoise_samples": 128},
        "ULTRA": {"samples": 512, "denoise_samples": 256},
    }[quality]

    print(f"Setting quality to {quality}")
    # EEVEE settings for atmospheric rendering
    if engine == "CYCLES":
        cycles = scene.cycles
        cycles.device = "GPU"
        cycles.samples = qs.get("samples", 4096)
        cycles.use_adaptive_sampling = True
        cycles.adaptive_threshold = qs.get("noise_threshold", 0.01)
        cycles.adaptive_max_samples = qs.get("denoise_max_samples", 128)
        cycles.denoiser = "OPENIMAGEDENOISE"  # Best quality denoiser
        cycles.denoising_input_passes = (  # More data for better denoising
            "RGB_ALBEDO_NORMAL"
        )

        # Performance optimizations
        cycles.max_bounces = qs.get("max_bounces", 12)
        # cycles.diffuse_bounces = 4
        # cycles.glossy_bounces = 4
        # cycles.transmission_bounces = 12
        # cycles.volume_bounces = 0
        # cycles.transparent_max_bounces = 8

        # Caustics (usually disabled for performance)
        cycles.caustics_reflective = False
        cycles.caustics_refractive = False

        # Lower quality volumes
        cycles.volume_step_rate = qs.get("volume_step_rate", 1.0)
        cycles.volume_max_steps = qs.get("volume_max_steps", 1028.0)

        # Set preview render settings
        bpy.context.scene.cycles.preview_samples = 12
        bpy.context.scene.cycles.use_preview_denoising = True
        bpy.context.scene.cycles.preview_denoiser = "OPENIMAGEDENOISE"
        bpy.context.scene.cycles.preview_denoising_input_passes = "RGB_ALBEDO_NORMAL"
        # These next two settings are only applied on the last sample so the
        # performance impact is minimal, and it really helps the result
        bpy.context.scene.cycles.preview_denoising_prefilter = "ACCURATE"
        bpy.context.scene.cycles.preview_denoising_quality = "HIGH"
        bpy.context.scene.cycles.preview_adaptive_threshold = 1
        bpy.context.scene.cycles.preview_denoising_start_sample = 12

    elif engine == "BLENDER_EEVEE_NEXT":
        eevee = scene.eevee
        eevee.taa_render_samples = qs.get("samples", 4096)
        # eevee.use_bloom = True
        # eevee.use_volumetric_lights = True
        eevee.volumetric_samples = 32

    return render
