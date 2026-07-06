#!/usr/bin/env python3
"""Audit ComfyUI workflow JSON files for missing/unknown node types.

Classifies every node type found in workflow files into:
  - Laura   : registered in Laura Image Studio's NODE_CLASS_MAPPINGS
  - Built-in: known ComfyUI core nodes (hardcoded set)
  - Community: everything else (likely from other custom-node packs)

Usage:
    python audit_workflow_nodes.py workflows/*.json
    python audit_workflow_nodes.py workflows/my_workflow.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Set

# ---------------------------------------------------------------------------
# Resolve Laura Studio package so we can import NODE_CLASS_MAPPINGS
# ---------------------------------------------------------------------------

_STUDIO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CUSTOM_NODES_DIR = os.path.dirname(_STUDIO_DIR)

# Add the custom_nodes parent so `import Laura_Image_Studio` works, and also
# add custom_nodes itself so relative imports inside the package resolve.
for _p in (_CUSTOM_NODES_DIR, os.path.dirname(_CUSTOM_NODES_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Install lightweight stubs for ComfyUI-only modules so the full node set loads
import types as _types
for _stub_name in [
    "folder_paths", "comfy", "comfy.model_management", "comfy.sd", "comfy.utils",
    "comfy.samplers", "comfy.sample", "comfy.latent_formats", "comfy.model_base",
    "comfy.supported_models", "comfy.controlnet", "comfy.clip_vision",
    "comfy.diffusers_convert", "comfy.model_patcher", "comfy.ldm",
    "comfy.ldm.modules", "comfy_extras", "comfy.graph_utils",
    "execution", "nodes", "server",
]:
    if _stub_name not in sys.modules:
        _m = _types.ModuleType(_stub_name)
        _m.__path__ = []  # type: ignore[attr-defined]
        sys.modules[_stub_name] = _m

# ---------------------------------------------------------------------------
# Known ComfyUI built-in node types
# ---------------------------------------------------------------------------

BUILTIN_NODES: Set[str] = {
    # Sampling
    "KSampler",
    "KSamplerAdvanced",
    # Loaders
    "CheckpointLoaderSimple",
    "CLIPLoader",
    "DualCLIPLoader",
    "VAELoader",
    "UNETLoader",
    "LoraLoader",
    "ControlNetLoader",
    "LoadImage",
    "LoadImageMask",
    # CLIP / Conditioning
    "CLIPTextEncode",
    "CLIPSetLastLayer",
    "ConditioningCombine",
    "ConditioningSetArea",
    "ControlNetApply",
    # VAE
    "VAEDecode",
    "VAEEncode",
    # Latent
    "EmptyLatentImage",
    "LatentUpscale",
    "LatentUpscaleBy",
    "LatentBlend",
    "LatentComposite",
    "RepeatLatentBatch",
    "LatentFromBatch",
    "RebatchLatentBatch",
    "SetLatentNoiseMask",
    # Image
    "SaveImage",
    "PreviewImage",
    "ImageScale",
    "ImageScaleBy",
    "ImageCrop",
    "ImagePadForOutpaint",
    "ImageCompositeMasked",
    "ImageToMask",
    "MaskToImage",
    # Mask
    "SolidMask",
    "InvertMask",
    "GrowMask",
    "FeatherMask",
    "CropMask",
    "ThresholdMask",
    # Advanced / Model patches
    "ModelSamplingDiscrete",
    "FreeU",
    "FreeU_V2",
    "SelfAttentionGuidance",
    "PatchModelAddDownscale",
    "RescaleCFG",
    # Legacy loader aliases
    "CheckpointLoader",
    "UpscaleModelLoader",
    # Utility
    "Reroute",
    "ShowText|textbox",
    "Note",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_laura_node_names() -> Set[str]:
    """Import Laura Image Studio and return the set of registered node type names."""
    try:
        # Suppress noisy prints during import
        _old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            import Laura_Image_Studio  # noqa: F811
            mappings = getattr(Laura_Image_Studio, "NODE_CLASS_MAPPINGS", {})
        finally:
            sys.stdout.close()
            sys.stdout = _old_stdout
        return set(mappings.keys())
    except Exception as exc:
        print(f"[WARN] Could not import Laura_Image_Studio: {exc}")
        print("       Laura nodes will all be classified as 'Community'.")
        return set()


def _extract_node_types(workflow: Dict[str, Any]) -> List[str]:
    """Extract node type strings from a workflow JSON.

    Handles both the standard web UI format (top-level "nodes" array) and
    the API format (top-level dict of id -> node, each with a "class_type").
    """
    node_types: List[str] = []

    # Web UI format: {"nodes": [{...}, ...], "links": [...], ...}
    if "nodes" in workflow and isinstance(workflow["nodes"], list):
        for node in workflow["nodes"]:
            ntype = node.get("type")
            if ntype:
                node_types.append(ntype)
        return node_types

    # API format: {"3": {"class_type": "KSampler", ...}, ...}
    # Every top-level value is a dict with "class_type".
    if all(
        isinstance(v, dict) and "class_type" in v
        for v in workflow.values()
        if isinstance(v, dict)
    ):
        for node in workflow.values():
            if isinstance(node, dict):
                ctype = node.get("class_type")
                if ctype:
                    node_types.append(ctype)

    return node_types


def classify_nodes(
    node_types: List[str],
    laura_nodes: Set[str],
) -> Dict[str, List[str]]:
    """Classify node types into Laura / Built-in / Community buckets."""
    buckets: Dict[str, List[str]] = {
        "Laura": [],
        "Built-in": [],
        "Community": [],
    }
    seen: Set[str] = set()
    for ntype in sorted(set(node_types)):
        if ntype in seen:
            continue
        seen.add(ntype)
        if ntype in laura_nodes:
            buckets["Laura"].append(ntype)
        elif ntype in BUILTIN_NODES:
            buckets["Built-in"].append(ntype)
        else:
            buckets["Community"].append(ntype)
    return buckets


# ---------------------------------------------------------------------------
# Report generators
# ---------------------------------------------------------------------------


def _generate_markdown(
    results: Dict[str, Dict[str, List[str]]],
    laura_nodes: Set[str],
) -> str:
    """Build a Markdown report string."""
    lines: List[str] = []
    lines.append("# Workflow Node Audit Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Laura Studio nodes available: {len(laura_nodes)}")
    lines.append(f"Workflows scanned: {len(results)}")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Workflow | Laura | Built-in | Community |")
    lines.append("|----------|------:|--------:|---------:|")
    for wf_name, buckets in sorted(results.items()):
        lines.append(
            f"| {wf_name} "
            f"| {len(buckets['Laura'])} "
            f"| {len(buckets['Built-in'])} "
            f"| {len(buckets['Community'])} |"
        )
    lines.append("")

    # Per-workflow details
    lines.append("## Details")
    lines.append("")
    for wf_name, buckets in sorted(results.items()):
        lines.append(f"### {wf_name}")
        lines.append("")
        for category in ("Laura", "Built-in", "Community"):
            nodes = buckets[category]
            if nodes:
                lines.append(f"**{category}** ({len(nodes)}):")
                for n in sorted(nodes):
                    lines.append(f"- `{n}`")
                lines.append("")
        if not any(buckets.values()):
            lines.append("_No nodes found._")
            lines.append("")

    return "\n".join(lines)


def _generate_missing_json(
    results: Dict[str, Dict[str, List[str]]],
) -> Dict[str, Any]:
    """Build a dict for missing_nodes.json (community/unknown nodes only)."""
    missing: Dict[str, List[str]] = {}
    for wf_name, buckets in sorted(results.items()):
        if buckets["Community"]:
            missing[wf_name] = sorted(buckets["Community"])
    return {
        "generated": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "description": "Node types not recognized as Laura Studio or ComfyUI built-in",
        "workflows": missing,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit ComfyUI workflow JSON files for missing/unknown node types.",
    )
    parser.add_argument(
        "workflows",
        nargs="+",
        help="Workflow JSON file paths (supports shell glob patterns).",
    )
    parser.add_argument(
        "--reports-dir",
        default=os.path.join(_STUDIO_DIR, "reports"),
        help="Directory for output reports (default: <studio>/reports/).",
    )
    args = parser.parse_args()

    # Expand globs (useful on Windows where the shell may not expand them)
    files: List[str] = []
    for pattern in args.workflows:
        expanded = glob.glob(pattern)
        if expanded:
            files.extend(expanded)
        else:
            # Treat as literal path
            files.append(pattern)

    if not files:
        print("No workflow files matched.", file=sys.stderr)
        sys.exit(1)

    # Load Laura node names
    laura_nodes = _load_laura_node_names()

    # Process each workflow
    results: Dict[str, Dict[str, List[str]]] = {}
    for filepath in files:
        wf_name = os.path.basename(filepath)
        if not os.path.isfile(filepath):
            print(f"[SKIP] Not a file: {filepath}")
            continue
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                workflow = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[SKIP] Failed to load {filepath}: {exc}")
            continue

        node_types = _extract_node_types(workflow)
        if not node_types:
            print(f"[WARN] No node types found in {filepath}")

        buckets = classify_nodes(node_types, laura_nodes)
        results[wf_name] = buckets

        # Per-file summary to stdout
        total = sum(len(v) for v in buckets.values())
        community_count = len(buckets["Community"])
        status = "OK" if community_count == 0 else f"{community_count} community"
        print(f"  {wf_name}: {total} nodes [{status}]")

    if not results:
        print("No valid workflows processed.", file=sys.stderr)
        sys.exit(1)

    # Write reports
    reports_dir = args.reports_dir
    os.makedirs(reports_dir, exist_ok=True)

    md_path = os.path.join(reports_dir, "workflow_node_audit.md")
    md_content = _generate_markdown(results, laura_nodes)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\nMarkdown report: {md_path}")

    missing_path = os.path.join(reports_dir, "missing_nodes.json")
    missing_data = _generate_missing_json(results)
    with open(missing_path, "w", encoding="utf-8") as f:
        json.dump(missing_data, f, indent=2)
    print(f"Missing nodes JSON: {missing_path}")

    # Exit with non-zero if any community nodes found
    total_community = sum(
        len(b["Community"]) for b in results.values()
    )
    if total_community:
        print(f"\nFound {total_community} community/unknown node type(s) across all workflows.")
        sys.exit(2)
    else:
        print("\nAll nodes are Laura Studio or ComfyUI built-in.")


if __name__ == "__main__":
    main()
