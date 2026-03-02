"""
Laura Image Studio - ComfyUI Custom Nodes
A comprehensive all-in-one image generation, editing, and upscaling solution
"""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Import all node modules with fault tolerance
# If one module fails, the rest still load
_modules_to_load = [
    "generation",
    "models",
    "toggle",
    "checkpoint",
    "video",
    "dressing",
    "face",
    "inpainting",
    "upscaling",
    "background",
    "quantization",
    "video_advanced",
    "batch_processing",
    "tile_processing",
    "comparison",
]

for _mod_name in _modules_to_load:
    try:
        import importlib

        _mod = importlib.import_module(f".nodes.{_mod_name}", package=__name__)
        NODE_CLASS_MAPPINGS.update(getattr(_mod, "NODE_CLASS_MAPPINGS", {}))
        NODE_DISPLAY_NAME_MAPPINGS.update(
            getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {})
        )
    except Exception as e:
        print(f"[Laura Image Studio] WARNING: Failed to load {_mod_name}: {e}")

WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
