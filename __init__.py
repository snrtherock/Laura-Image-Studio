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
        # This handles Portable (python_embeded) vs Conda vs System Python automatically.
        # --only-binary :all: prevents pip from trying to build packages from source,
        # which fails on Portable Python (no compiler / meson / build tools).
        # If a wheel isn't available, that package is simply skipped rather than
        # crashing the entire install with a mesonpy/compiler error.
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
                "--only-binary",
                ":all:",
            ]
        )
        os.environ["LAURA_STUDIO_REQS_CHECKED"] = "1"
        print("## [snrtherock/Laura Studio] All dependencies verified/installed.")
    except subprocess.CalledProcessError:
        # Wheel-only install failed (some packages lack wheels for this platform).
        # Fall back to a normal install — if user has build tools it'll work,
        # otherwise the warning below tells them what to do.
        try:
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
            os.environ["LAURA_STUDIO_REQS_CHECKED"] = "1"  # Don't retry every restart
            print(f"## [snrtherock/Laura Studio] AUTO-INSTALL WARNING: {e}")
            print("## This is usually harmless if deps are already installed.")
            print("## Manual install if needed: pip install -r requirements.txt")
    except Exception as e:
        os.environ["LAURA_STUDIO_REQS_CHECKED"] = "1"  # Don't retry every restart
        print(f"## [snrtherock/Laura Studio] AUTO-INSTALL WARNING: {e}")
        print("## This is usually harmless if deps are already installed.")
        print("## Manual install if needed: pip install -r requirements.txt")


# Run dependency check BEFORE loading any nodes
_check_dependencies()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Import all node modules with fault tolerance
# If one module fails, the rest still load
_modules_to_load = [
    "model_registry",   # Pure data — no nodes, must load first
    "model_manager",    # Config + management nodes, load second
    "control_center",   # Control Center supernode, load third
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
    "flux_tools",
]

# Core modules always load, even if config says otherwise
_CORE_MODULES = {
    "model_registry", "model_manager", "control_center", "models", "toggle",
    "quantization", "checkpoint", "batch_processing",
    "tile_processing", "comparison",
}

# Load laura_config.json if available for module enable/disable
_config = None
_config_path = os.path.join(os.path.dirname(__file__), "laura_config.json")
if os.path.exists(_config_path):
    try:
        import json as _json
        with open(_config_path, "r") as _cf:
            _config = _json.load(_cf)
        print("[Laura Image Studio] Loaded module config from laura_config.json")
    except Exception as _e:
        print(f"[Laura Image Studio] WARNING: Failed to load config: {_e}")

for _mod_name in _modules_to_load:
    # Check if module is disabled in config (core modules always load)
    if _config and _mod_name not in _CORE_MODULES:
        _mod_config = _config.get("modules", {}).get(_mod_name, {})
        if isinstance(_mod_config, dict) and not _mod_config.get("enabled", True):
            print(f"[Laura Image Studio] SKIPPED: {_mod_name} (disabled in config)")
            continue

    try:
        import importlib

        _mod = importlib.import_module(f".nodes.{_mod_name}", package=__name__)
        NODE_CLASS_MAPPINGS.update(getattr(_mod, "NODE_CLASS_MAPPINGS", {}))
        NODE_DISPLAY_NAME_MAPPINGS.update(
            getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {})
        )
    except Exception as e:
        print(f"[Laura Image Studio] WARNING: Failed to load {_mod_name}: {e}")

# Enable web extension directory for JavaScript-based UI enhancements
# This serves the web/js/model_links.js extension for clickable download links
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
