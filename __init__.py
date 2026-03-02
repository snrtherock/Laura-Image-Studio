"""
Laura Image Studio - ComfyUI Custom Nodes
A comprehensive all-in-one image generation, editing, and upscaling solution
"""

import sys
import os
import subprocess


# ============== AUTO-DEPENDENCY CHECKER ==============
# This ensures that no matter IF you use Conda, Portable, or System Python,
# the nodes will install their own requirements to the CORRECT environment.
def _check_dependencies():
    _req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if not os.path.exists(_req_file):
        return

    # Use a hidden flag to prevent infinite loops or redundant checks
    if os.environ.get("LAURA_STUDIO_REQS_CHECKED") == "1":
        return

    print(f"## [snrtherock/Laura Studio] Checking environment: {sys.executable}")

    try:
        # sys.executable ensures we use the EXACT python that ComfyUI is currently using
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                _req_file,
                "--quiet",
                "--no-warn-script-location",
            ]
        )
        os.environ["LAURA_STUDIO_REQS_CHECKED"] = "1"
        print("## [snrtherock/Laura Studio] All dependencies verified/installed.")
    except Exception as e:
        print(f"## [snrtherock/Laura Studio] AUTO-INSTALL WARNING: {e}")
        print("## Manual install may be required: pip install -r requirements.txt")


# Run dependency check BEFORE loading any nodes
_check_dependencies()

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
