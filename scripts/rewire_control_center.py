#!/usr/bin/env python3
"""
Rewire Control Center Script
=============================
Replaces scattered model-loading / VRAM-management nodes with a single
LauraControlCenter supernode across all three master workflows.

Workflows processed:
  1. custom_nodes/Laura_Image_Studio/workflows/enhanced_master_workflow.json
  2. custom_nodes/Laura_Image_Studio/workflows/master_all_in_one.json
  3. workflows/master/Laura_Master_Studio_v0.8.json

For workflows 1 & 2 the script only removes nodes and dangling links, then
inserts the new LauraControlCenter node (no rewiring needed).

For workflow 3 (v0.8) the script performs full rewiring: every downstream
node that was previously fed by UniversalModelLoader (VAE) or MultiLoraStack
(MODEL / CLIP) is reconnected to the corresponding output slot of the new
LauraControlCenter node.
"""

import json
import os
import sys
import copy
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve base directory  (two levels up from this script)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SUBMODULE_ROOT = SCRIPT_DIR.parent                # custom_nodes/Laura_Image_Studio
PROJECT_ROOT = SUBMODULE_ROOT.parent.parent        # Best_Real_Image_Gen - Copy

WORKFLOW_PATHS = {
    "enhanced": SUBMODULE_ROOT / "workflows" / "enhanced_master_workflow.json",
    "all_in_one": SUBMODULE_ROOT / "workflows" / "master_all_in_one.json",
    "v08": PROJECT_ROOT / "workflows" / "master" / "Laura_Master_Studio_v0.8.json",
}

# ---------------------------------------------------------------------------
# Node IDs to remove per workflow
# ---------------------------------------------------------------------------
REMOVE_IDS = {
    "enhanced": {1000, 1001, 1002, 1003, 1004},
    "all_in_one": {1, 6, 117, 118, 120, 121, 1000, 1001, 1002, 1003, 1004},
    "v08": {1, 2, 916, 917, 919, 920},
}

# ---------------------------------------------------------------------------
# LauraControlCenter node template
# ---------------------------------------------------------------------------

def make_cc_node(node_id, pos=None):
    """Return a LauraControlCenter node dict ready for insertion."""
    return {
        "id": node_id,
        "type": "LauraControlCenter",
        "pos": pos or [50, 50],
        "size": [400, 600],
        "flags": {},
        "order": 0,
        "mode": 0,
        "properties": {
            "Node name for S&R": "LauraControlCenter"
        },
        "widgets_values": [
            "z_image_turbo",   # image_model
            "none",            # upscale_model
            "none",            # video_model
            "none",            # face_model
            "None",            # lora_1
            1.0,               # lora_1_strength
            "None",            # lora_2
            1.0,               # lora_2_strength
            "None",            # lora_3
            1.0,               # lora_3_strength
            True,              # auto_detect_vram
            "high",            # manual_vram_tier
            True,              # auto_quantization
            False,             # torch_compile
            1024,              # default_width
            1024,              # default_height
        ],
        "title": "Laura Control Center",
        "color": "#226",
        "bgcolor": "#337",
        "inputs": [],
        "outputs": [
            {"name": "model",          "type": "MODEL",  "links": [], "slot_index": 0},
            {"name": "clip",           "type": "CLIP",   "links": [], "slot_index": 1},
            {"name": "vae",            "type": "VAE",    "links": [], "slot_index": 2},
            {"name": "detected_type",  "type": "STRING", "links": [], "slot_index": 3},
            {"name": "default_width",  "type": "INT",    "links": [], "slot_index": 4},
            {"name": "default_height", "type": "INT",    "links": [], "slot_index": 5},
            {"name": "status_report",  "type": "STRING", "links": [], "slot_index": 6},
            {"name": "config_json",    "type": "STRING", "links": [], "slot_index": 7},
        ],
    }


# ===================================================================
#  Generic helpers
# ===================================================================

def load_workflow(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_workflow(path, data):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    print(f"  [saved] {path}")


def remove_nodes(data, ids_to_remove):
    """Remove nodes by ID and return the removed node dicts."""
    removed = []
    keep = []
    for node in data["nodes"]:
        if node["id"] in ids_to_remove:
            removed.append(node)
        else:
            keep.append(node)
    data["nodes"] = keep
    return removed


def remove_links_for_nodes(data, ids_to_remove):
    """Remove top-level links that touch any of the removed node IDs.

    Also cleans up per-node input/output link references in surviving nodes.
    Returns the set of removed link IDs.
    """
    removed_link_ids = set()
    keep_links = []
    for link in data.get("links", []):
        # link format: [link_id, src_node, src_slot, tgt_node, tgt_slot, type]
        src_node = link[1]
        tgt_node = link[3]
        if src_node in ids_to_remove or tgt_node in ids_to_remove:
            removed_link_ids.add(link[0])
        else:
            keep_links.append(link)
    data["links"] = keep_links

    # Clean surviving nodes' input/output references
    for node in data["nodes"]:
        for inp in node.get("inputs", []):
            if inp.get("link") in removed_link_ids:
                inp["link"] = None
        for out in node.get("outputs", []):
            if out.get("links"):
                out["links"] = [lid for lid in out["links"] if lid not in removed_link_ids]

    return removed_link_ids


# ===================================================================
#  Process enhanced_master_workflow.json
# ===================================================================

def process_enhanced(data):
    """Remove Laura model nodes 1000-1004, add LauraControlCenter."""
    ids = REMOVE_IDS["enhanced"]
    print(f"  Removing nodes: {sorted(ids)}")
    remove_nodes(data, ids)
    removed_links = remove_links_for_nodes(data, ids)
    print(f"  Removed {len(removed_links)} links")

    # Determine new node ID
    new_id = data["last_node_id"] + 1
    cc_node = make_cc_node(new_id, pos=[50, 50])
    data["nodes"].append(cc_node)
    data["last_node_id"] = new_id
    print(f"  Added LauraControlCenter as node {new_id}")


# ===================================================================
#  Process master_all_in_one.json
# ===================================================================

def process_all_in_one(data):
    """Remove model/VRAM nodes, add LauraControlCenter."""
    ids = REMOVE_IDS["all_in_one"]
    print(f"  Removing nodes: {sorted(ids)}")
    remove_nodes(data, ids)
    removed_links = remove_links_for_nodes(data, ids)
    print(f"  Removed {len(removed_links)} links")

    new_id = data["last_node_id"] + 1
    cc_node = make_cc_node(new_id, pos=[50, 50])
    data["nodes"].append(cc_node)
    data["last_node_id"] = new_id
    print(f"  Added LauraControlCenter as node {new_id}")


# ===================================================================
#  Process Laura_Master_Studio_v0.8.json  (full rewire)
# ===================================================================

# Mapping of connections to recreate:
#   CC slot 0 (MODEL) -> all former MultiLoraStack(2) slot 0 destinations
#   CC slot 1 (CLIP)  -> all former MultiLoraStack(2) slot 1 destinations
#   CC slot 2 (VAE)   -> all former UniversalModelLoader(1) slot 2 destinations

V08_MODEL_DESTINATIONS = [
    # (target_node_id, target_slot, type_string)
    (3,   0, "MODEL"),   # IPAdapterUnifiedLoader
    (63,  0, "MODEL"),   # LauraInpainter
    (66,  0, "MODEL"),   # LauraOutpainter
    (73,  1, "MODEL"),   # ImageToVideo
    (932, 0, "MODEL"),   # TileInpainter
    (951, 1, "MODEL"),   # DetailEnhancer
    (958, 2, "MODEL"),   # ExpressionTransfer
    (970, 0, "MODEL"),   # BackgroundGenerator
]

V08_CLIP_DESTINATIONS = [
    (6,   0, "CLIP"),    # CLIPTextEncode (Person Prompt)
    (7,   0, "CLIP"),    # CLIPTextEncode (Negative)
    (8,   0, "CLIP"),    # CLIPTextEncode (Background Prompt)
    (63,  1, "CLIP"),    # LauraInpainter
    (66,  1, "CLIP"),    # LauraOutpainter
    (73,  2, "CLIP"),    # ImageToVideo
    (932, 1, "CLIP"),    # TileInpainter
    (951, 2, "CLIP"),    # DetailEnhancer
    (958, 3, "CLIP"),    # ExpressionTransfer
    (970, 1, "CLIP"),    # BackgroundGenerator
]

V08_VAE_DESTINATIONS = [
    (15,  1, "VAE"),     # VAEDecode (Background)
    (52,  1, "VAE"),     # VAEDecode (Person)
    (63,  2, "VAE"),     # LauraInpainter
    (66,  2, "VAE"),     # LauraOutpainter
    (73,  3, "VAE"),     # ImageToVideo
    (932, 2, "VAE"),     # TileInpainter
    (951, 3, "VAE"),     # DetailEnhancer
    (958, 4, "VAE"),     # ExpressionTransfer
    (970, 2, "VAE"),     # BackgroundGenerator
]


def process_v08(data):
    """Remove model/VRAM nodes, add LauraControlCenter with full rewiring."""
    ids = REMOVE_IDS["v08"]
    print(f"  Removing nodes: {sorted(ids)}")

    # Also need to remove link 113 (VRAMAutoDetector -> ResolutionScaler)
    # even though ResolutionScaler (918) is kept, the link from removed
    # node 916 will be removed automatically.

    remove_nodes(data, ids)
    removed_links = remove_links_for_nodes(data, ids)
    print(f"  Removed {len(removed_links)} links")

    # -- Insert LauraControlCenter --
    new_node_id = data["last_node_id"] + 1     # 977
    next_link_id = data["last_link_id"] + 1    # 167

    cc_node = make_cc_node(new_node_id, pos=[50, 50])

    # Build a node-id -> node lookup for surviving nodes
    node_map = {n["id"]: n for n in data["nodes"]}

    new_links = []           # top-level link entries to add
    cc_model_link_ids = []   # link IDs for CC slot 0
    cc_clip_link_ids = []    # link IDs for CC slot 1
    cc_vae_link_ids = []     # link IDs for CC slot 2

    def add_link(cc_slot, cc_link_list, target_id, target_slot, type_str):
        nonlocal next_link_id
        lid = next_link_id
        next_link_id += 1

        # Top-level link
        new_links.append([lid, new_node_id, cc_slot, target_id, target_slot, type_str])
        cc_link_list.append(lid)

        # Update target node input
        target_node = node_map.get(target_id)
        if target_node is None:
            print(f"    [warn] target node {target_id} not found, skipping link {lid}")
            return
        inputs = target_node.get("inputs", [])
        # Find the right input slot
        found = False
        for inp in inputs:
            if inputs.index(inp) == target_slot:
                inp["link"] = lid
                found = True
                break
        if not found:
            # Input slot doesn't exist in the list yet or slot index mismatch;
            # try matching by position
            if target_slot < len(inputs):
                inputs[target_slot]["link"] = lid
            else:
                print(f"    [warn] target node {target_id} has no input slot {target_slot}")

    # Wire MODEL (CC slot 0)
    for tgt_id, tgt_slot, tgt_type in V08_MODEL_DESTINATIONS:
        add_link(0, cc_model_link_ids, tgt_id, tgt_slot, tgt_type)

    # Wire CLIP (CC slot 1)
    for tgt_id, tgt_slot, tgt_type in V08_CLIP_DESTINATIONS:
        add_link(1, cc_clip_link_ids, tgt_id, tgt_slot, tgt_type)

    # Wire VAE (CC slot 2)
    for tgt_id, tgt_slot, tgt_type in V08_VAE_DESTINATIONS:
        add_link(2, cc_vae_link_ids, tgt_id, tgt_slot, tgt_type)

    # Populate CC node output link arrays
    cc_node["outputs"][0]["links"] = cc_model_link_ids
    cc_node["outputs"][1]["links"] = cc_clip_link_ids
    cc_node["outputs"][2]["links"] = cc_vae_link_ids

    # Insert node and links
    data["nodes"].append(cc_node)
    data["links"].extend(new_links)
    data["last_node_id"] = new_node_id
    data["last_link_id"] = next_link_id - 1

    print(f"  Added LauraControlCenter as node {new_node_id}")
    print(f"  Created {len(new_links)} new links (IDs {new_links[0][0]}..{new_links[-1][0]})")
    print(f"  last_node_id = {data['last_node_id']}, last_link_id = {data['last_link_id']}")


# ===================================================================
#  Verification
# ===================================================================

def verify_workflow(name, data, expected_cc_id=None, expected_downstream=None):
    """Run sanity checks on the processed workflow."""
    errors = []

    # 1. LauraControlCenter exists
    cc_nodes = [n for n in data["nodes"] if n["type"] == "LauraControlCenter"]
    if not cc_nodes:
        errors.append("LauraControlCenter node NOT found")
    else:
        print(f"  [ok] LauraControlCenter node exists (id={cc_nodes[0]['id']})")

    # 2. Removed nodes are gone
    ids_to_remove = REMOVE_IDS[name]
    surviving_removed = [n for n in data["nodes"] if n["id"] in ids_to_remove]
    if surviving_removed:
        errors.append(f"Removed nodes still present: {[n['id'] for n in surviving_removed]}")
    else:
        print(f"  [ok] All {len(ids_to_remove)} removed nodes are gone")

    # 3. No orphaned links (links referencing non-existent node IDs)
    node_ids = {n["id"] for n in data["nodes"]}
    orphan_links = []
    for link in data.get("links", []):
        if link[1] not in node_ids or link[3] not in node_ids:
            orphan_links.append(link[0])
    if orphan_links:
        errors.append(f"Orphaned link IDs: {orphan_links}")
    else:
        print(f"  [ok] No orphaned links ({len(data.get('links', []))} links total)")

    # 4. For v08: verify downstream nodes have valid CC links
    if expected_downstream and cc_nodes:
        cc_id = cc_nodes[0]["id"]
        for ds_id in expected_downstream:
            ds_node = next((n for n in data["nodes"] if n["id"] == ds_id), None)
            if ds_node is None:
                errors.append(f"Expected downstream node {ds_id} not found")
                continue
            # Check if any input references the CC node via the links table
            ds_input_links = [inp.get("link") for inp in ds_node.get("inputs", []) if inp.get("link") is not None]
            cc_links_to_ds = [l for l in data["links"] if l[1] == cc_id and l[3] == ds_id]
            if not cc_links_to_ds:
                errors.append(f"Downstream node {ds_id} ({ds_node.get('type','?')}) has no link from CC")
            else:
                # Also verify the input slot references the link ID
                for cc_link in cc_links_to_ds:
                    link_id = cc_link[0]
                    target_slot = cc_link[4]
                    inputs = ds_node.get("inputs", [])
                    if target_slot < len(inputs):
                        actual_link = inputs[target_slot].get("link")
                        if actual_link != link_id:
                            errors.append(
                                f"Node {ds_id} input slot {target_slot}: "
                                f"expected link {link_id}, got {actual_link}"
                            )

        if not errors:
            print(f"  [ok] All {len(expected_downstream)} downstream nodes correctly wired to CC")

    if errors:
        print(f"  [ERRORS]")
        for e in errors:
            print(f"    - {e}")
        return False
    return True


# ===================================================================
#  Main
# ===================================================================

def main():
    print("=" * 60)
    print("Rewire Control Center - Laura Image Studio")
    print("=" * 60)

    all_ok = True

    # --- 1. enhanced_master_workflow.json ---
    print("\n[1] enhanced_master_workflow.json")
    path = WORKFLOW_PATHS["enhanced"]
    if not path.exists():
        print(f"  [skip] File not found: {path}")
    else:
        data = load_workflow(path)
        process_enhanced(data)
        save_workflow(path, data)
        ok = verify_workflow("enhanced", data)
        all_ok = all_ok and ok

    # --- 2. master_all_in_one.json ---
    print("\n[2] master_all_in_one.json")
    path = WORKFLOW_PATHS["all_in_one"]
    if not path.exists():
        print(f"  [skip] File not found: {path}")
    else:
        data = load_workflow(path)
        process_all_in_one(data)
        save_workflow(path, data)
        ok = verify_workflow("all_in_one", data)
        all_ok = all_ok and ok

    # --- 3. Laura_Master_Studio_v0.8.json (full rewire) ---
    print("\n[3] Laura_Master_Studio_v0.8.json")
    path = WORKFLOW_PATHS["v08"]
    if not path.exists():
        print(f"  [skip] File not found: {path}")
    else:
        data = load_workflow(path)
        process_v08(data)
        save_workflow(path, data)
        expected_ds = [3, 6, 7, 8, 15, 52, 63, 66, 73, 932, 951, 958, 970]
        ok = verify_workflow("v08", data, expected_downstream=expected_ds)
        all_ok = all_ok and ok

    # --- Summary ---
    print("\n" + "=" * 60)
    if all_ok:
        print("ALL WORKFLOWS PROCESSED SUCCESSFULLY")
    else:
        print("SOME WORKFLOWS HAD ERRORS - please review above")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
