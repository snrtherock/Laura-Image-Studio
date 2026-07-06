"""
Laura Image Studio - ComfyUI Custom Nodes
A comprehensive all-in-one image generation, editing, and upscaling solution
"""

import sys
import os
import subprocess
import importlib


# ============== AUTO-DEPENDENCY CHECKER ==============
def _check_dependencies():
    _req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if not os.path.exists(_req_file):
        return

    if os.environ.get("LAURA_STUDIO_REQS_CHECKED") == "1":
        return

    print(f"## [Laura Studio] Checking core dependencies: {sys.executable}")

    try:
        subprocess.check_call(
            [
                sys.executable, "-m", "pip", "install",
                "-r", _req_file,
                "--quiet", "--no-warn-script-location",
                "--only-binary", ":all:",
            ]
        )
        os.environ["LAURA_STUDIO_REQS_CHECKED"] = "1"
        print("## [Laura Studio] Core dependencies OK.")
    except subprocess.CalledProcessError:
        try:
            subprocess.check_call(
                [
                    sys.executable, "-m", "pip", "install",
                    "-r", _req_file,
                    "--quiet", "--no-warn-script-location",
                ]
            )
            os.environ["LAURA_STUDIO_REQS_CHECKED"] = "1"
            print("## [Laura Studio] Core dependencies OK.")
        except Exception as e:
            os.environ["LAURA_STUDIO_REQS_CHECKED"] = "1"
            print(f"## [Laura Studio] AUTO-INSTALL WARNING: {e}")
            print("## Manual install: pip install -r requirements.txt")
    except Exception as e:
        os.environ["LAURA_STUDIO_REQS_CHECKED"] = "1"
        print(f"## [Laura Studio] AUTO-INSTALL WARNING: {e}")
        print("## Manual install: pip install -r requirements.txt")


def _check_optional_dependencies(group):
    """Install optional dependency group if the requirements file exists.

    Args:
        group: One of 'video', 'face', 'upscale', 'advanced'.
    """
    env_key = f"LAURA_STUDIO_OPT_{group.upper()}_CHECKED"
    if os.environ.get(env_key) == "1":
        return True

    req_file = os.path.join(
        os.path.dirname(__file__), f"requirements-optional-{group}.txt"
    )
    if not os.path.exists(req_file):
        return False

    try:
        subprocess.check_call(
            [
                sys.executable, "-m", "pip", "install",
                "-r", req_file,
                "--quiet", "--no-warn-script-location",
                "--only-binary", ":all:",
            ]
        )
        os.environ[env_key] = "1"
        return True
    except Exception:
        os.environ[env_key] = "1"
        return False


_check_dependencies()

# ============== IMPORT HEALTH TRACKING ==============

_import_health = {}

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Modules grouped by dependency tier
_CORE_MODULES = [
    "model_registry",
    "model_manager",
    "control_center",
    "hardware_profiler",
    "model_recommender",
    "workflow_health_check",
    "stage_switch",
    "terminal_save",
    "model_catalog",
]

_STANDARD_MODULES = [
    "generation",
    "models",
    "toggle",
    "checkpoint",
    "quantization",
    "batch_processing",
    "tile_processing",
    "comparison",
    "inpainting",
    "background",
]

_VIDEO_MODULES = [
    "video",
    "video_advanced",
]

_FACE_MODULES = [
    "face",
    "dressing",
]

_UPSCALE_MODULES = [
    "upscaling",
]

_OTHER_MODULES = [
    "flux_tools",
]

_ALWAYS_LOAD = set(_CORE_MODULES) | set(_STANDARD_MODULES)

# Load laura_config.json if available for module enable/disable
_config = None
_config_path = os.path.join(os.path.dirname(__file__), "laura_config.json")
if os.path.exists(_config_path):
    try:
        import json as _json
        with open(_config_path, "r") as _cf:
            _config = _json.load(_cf)
        print("[Laura Studio] Loaded config from laura_config.json")
    except Exception as _e:
        print(f"[Laura Studio] WARNING: Failed to load config: {_e}")


def _load_module(mod_name):
    """Load a single node module with fault tolerance."""
    try:
        _mod = importlib.import_module(f".nodes.{mod_name}", package=__name__)
        NODE_CLASS_MAPPINGS.update(getattr(_mod, "NODE_CLASS_MAPPINGS", {}))
        NODE_DISPLAY_NAME_MAPPINGS.update(
            getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {})
        )
        _import_health[mod_name] = "ok"
    except Exception as e:
        _import_health[mod_name] = str(e)
        if mod_name in _ALWAYS_LOAD:
            print(f"[Laura Studio] WARNING: Failed to load {mod_name}: {e}")


def _load_module_group(modules, optional_group=None):
    """Load a group of modules, optionally installing deps first."""
    if optional_group:
        _check_optional_dependencies(optional_group)

    for mod_name in modules:
        if _config and mod_name not in _ALWAYS_LOAD:
            mod_config = _config.get("modules", {}).get(mod_name, {})
            if isinstance(mod_config, dict) and not mod_config.get("enabled", True):
                _import_health[mod_name] = "disabled"
                continue
        _load_module(mod_name)


_load_module_group(_CORE_MODULES)
_load_module_group(_STANDARD_MODULES)
_load_module_group(_VIDEO_MODULES, optional_group="video")
_load_module_group(_FACE_MODULES, optional_group="face")
_load_module_group(_UPSCALE_MODULES, optional_group="upscale")
_load_module_group(_OTHER_MODULES)

_loaded = sum(1 for v in _import_health.values() if v == "ok")
_total = len(_import_health)
_failed = [k for k, v in _import_health.items() if v not in ("ok", "disabled")]
if _failed:
    print(f"[Laura Studio] {_loaded}/{_total} modules loaded. Failed: {', '.join(_failed)}")
else:
    print(f"[Laura Studio] {_loaded}/{_total} modules loaded OK.")


def get_import_health():
    """Return the import health dict for health check nodes."""
    return dict(_import_health)


WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
