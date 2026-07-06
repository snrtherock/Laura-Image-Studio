#!/usr/bin/env python3
"""
Smoke-test for Laura Image Studio node pack.

Imports the package without ComfyUI and validates every node class that
successfully loads.  Runnable with plain Python -- no external deps.

Usage:
    python smoke_test_import.py
    python smoke_test_import.py --verbose
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import sys
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Colour helpers (degrade gracefully when piped / Windows legacy console)
# ---------------------------------------------------------------------------

def _supports_colour() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
        return True
    return False

_USE_COLOUR = _supports_colour()

def _green(text: str) -> str:
    return f"\033[92m{text}\033[0m" if _USE_COLOUR else text

def _red(text: str) -> str:
    return f"\033[91m{text}\033[0m" if _USE_COLOUR else text

def _yellow(text: str) -> str:
    return f"\033[93m{text}\033[0m" if _USE_COLOUR else text

def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m" if _USE_COLOUR else text

# ---------------------------------------------------------------------------
# Stubs for modules that only exist inside ComfyUI
# ---------------------------------------------------------------------------

_COMFY_STUBS = [
    "folder_paths",
    "comfy",
    "comfy.model_management",
    "comfy.sd",
    "comfy.utils",
    "comfy.samplers",
    "comfy.sample",
    "comfy.latent_formats",
    "comfy.model_base",
    "comfy.supported_models",
    "comfy.controlnet",
    "comfy.clip_vision",
    "comfy.diffusers_convert",
    "comfy.model_patcher",
    "comfy.ldm",
    "comfy.ldm.modules",
    "comfy_extras",
    "execution",
    "nodes",
    "server",
]


def _install_stubs() -> list[str]:
    """Insert lightweight stub modules so imports don't crash."""
    installed: list[str] = []
    for name in _COMFY_STUBS:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []  # type: ignore[attr-defined]
            sys.modules[name] = mod
            installed.append(name)
    return installed


def _remove_stubs(names: list[str]) -> None:
    for name in names:
        sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


def _check_class(node_id: str, cls: type, verbose: bool) -> tuple[str, list[str]]:
    """Validate a single node class.  Returns (status, messages)."""
    issues: list[str] = []
    status = PASS

    # INPUT_TYPES ---------------------------------------------------------
    if not hasattr(cls, "INPUT_TYPES"):
        issues.append("missing INPUT_TYPES")
        status = FAIL
    elif not callable(getattr(cls, "INPUT_TYPES")):
        issues.append("INPUT_TYPES is not callable")
        status = FAIL

    # RETURN_TYPES --------------------------------------------------------
    if not hasattr(cls, "RETURN_TYPES"):
        issues.append("missing RETURN_TYPES")
        status = FAIL
    else:
        rt = getattr(cls, "RETURN_TYPES")
        if not isinstance(rt, tuple):
            issues.append(f"RETURN_TYPES is {type(rt).__name__}, expected tuple")
            status = FAIL

    # FUNCTION ------------------------------------------------------------
    if not hasattr(cls, "FUNCTION"):
        issues.append("missing FUNCTION")
        status = FAIL
    else:
        func_name = getattr(cls, "FUNCTION")
        if not isinstance(func_name, str):
            issues.append(f"FUNCTION is {type(func_name).__name__}, expected str")
            status = FAIL
        elif not hasattr(cls, func_name):
            issues.append(f"method '{func_name}' referenced by FUNCTION not found")
            status = FAIL

    # CATEGORY ------------------------------------------------------------
    if not hasattr(cls, "CATEGORY"):
        issues.append("missing CATEGORY")
        status = FAIL
    else:
        cat = getattr(cls, "CATEGORY")
        if not isinstance(cat, str):
            issues.append(f"CATEGORY is {type(cat).__name__}, expected str")
            status = FAIL

    # OUTPUT_NODE (optional) -- just note it in verbose mode
    if verbose and hasattr(cls, "OUTPUT_NODE"):
        on = getattr(cls, "OUTPUT_NODE")
        if not isinstance(on, bool):
            issues.append(f"OUTPUT_NODE is {type(on).__name__}, expected bool")
            if status == PASS:
                status = WARN

    return status, issues


def run(verbose: bool = False) -> int:
    """Run the smoke test.  Returns exit code (0 = all OK, 1 = failures)."""

    # -- path setup -------------------------------------------------------
    package_dir = Path(__file__).resolve().parent.parent          # Laura_Image_Studio/
    parent_dir = package_dir.parent                                # custom_nodes/

    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    package_name = package_dir.name  # "Laura_Image_Studio"

    print(_bold(f"=== Laura Image Studio Smoke Test ==="))
    print(f"Package : {package_dir}")
    print()

    # -- stub ComfyUI internals -------------------------------------------
    stubs = _install_stubs()
    if verbose and stubs:
        print(f"Installed {len(stubs)} ComfyUI stubs")

    # -- import -----------------------------------------------------------
    import_warnings: list[str] = []
    try:
        pkg = importlib.import_module(package_name)
    except Exception as exc:
        print(_red(f"FATAL: could not import {package_name}: {exc}"))
        _remove_stubs(stubs)
        return 1

    node_class_mappings: dict = getattr(pkg, "NODE_CLASS_MAPPINGS", None) or {}
    node_display_mappings: dict = getattr(pkg, "NODE_DISPLAY_NAME_MAPPINGS", None) or {}

    if not node_class_mappings:
        print(_yellow("WARNING: NODE_CLASS_MAPPINGS is empty -- nothing to validate."))
        print("  (This may happen if all submodules require ComfyUI at import time.)")
        _remove_stubs(stubs)
        return 0

    print(f"Loaded {len(node_class_mappings)} node(s)")
    print()

    # -- duplicate check --------------------------------------------------
    # NODE_CLASS_MAPPINGS is a dict so true duplicates are impossible at the
    # Python level, but we flag display-name collisions.
    seen_display: dict[str, str] = {}
    for nid, display in node_display_mappings.items():
        if display in seen_display.values():
            dup_id = [k for k, v in seen_display.items() if v == display][0]
            import_warnings.append(
                f"Duplicate display name '{display}' used by '{nid}' and '{dup_id}'"
            )
        seen_display[nid] = display

    # -- per-node checks --------------------------------------------------
    counts = {PASS: 0, WARN: 0, FAIL: 0}
    details: list[tuple[str, str, list[str]]] = []

    for node_id, cls in sorted(node_class_mappings.items()):
        status, issues = _check_class(node_id, cls, verbose)
        counts[status] += 1
        details.append((node_id, status, issues))

    # -- report -----------------------------------------------------------
    for node_id, status, issues in details:
        if status == PASS:
            marker = _green(f"[{PASS}]")
        elif status == WARN:
            marker = _yellow(f"[{WARN}]")
        else:
            marker = _red(f"[{FAIL}]")

        if verbose or status != PASS:
            print(f"  {marker} {node_id}")
            for issue in issues:
                print(f"         - {issue}")

    if not verbose and counts[PASS] and not counts[FAIL] and not counts[WARN]:
        print(f"  {_green('[PASS]')} All {counts[PASS]} node(s) passed")

    print()

    # Import warnings
    for w in import_warnings:
        print(f"  {_yellow('[WARN]')} {w}")
    if import_warnings:
        print()

    # Summary line
    total = sum(counts.values())
    summary_parts = [
        f"Total: {total}",
        _green(f"Passed: {counts[PASS]}"),
    ]
    if counts[WARN]:
        summary_parts.append(_yellow(f"Warnings: {counts[WARN]}"))
    if counts[FAIL]:
        summary_parts.append(_red(f"Failures: {counts[FAIL]}"))

    print(_bold("Summary: ") + "  |  ".join(summary_parts))

    _remove_stubs(stubs)

    if counts[FAIL]:
        print()
        print(_red("RESULT: FAIL"))
        return 1

    print()
    print(_green("RESULT: OK"))
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test Laura Image Studio node classes without ComfyUI.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show passing nodes and extra diagnostics.",
    )
    args = parser.parse_args()
    sys.exit(run(verbose=args.verbose))


if __name__ == "__main__":
    main()
