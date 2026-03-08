#!/usr/bin/env python3
"""
Fix overlapping groups in Laura Image Studio master workflows.

For each workflow, detects whether groups are arranged vertically or horizontally,
identifies groups that actually overlap, and shifts the later group (and all
subsequent groups in the same row/column) to eliminate the overlap with a
minimum 100px gap. Also shifts all nodes contained within moved groups.

Handles node positions in both [x, y] list and {"0": x, "1": y} dict formats.
"""

import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MIN_GAP = 100  # minimum gap in pixels when fixing an overlap


def get_node_pos(node):
    """Return (x, y) from a node's 'pos' field, handling list or dict format."""
    pos = node.get("pos")
    if pos is None:
        return None, None
    if isinstance(pos, dict):
        return pos.get("0", 0), pos.get("1", 0)
    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
        return pos[0], pos[1]
    return None, None


def _preserve_num_type(original, new_val):
    """Keep the numeric type (int vs float) consistent with the original."""
    if isinstance(original, int) and isinstance(new_val, (int, float)):
        return int(new_val)
    return new_val


def set_node_pos(node, x, y):
    """Write (x, y) back into a node's 'pos' field, preserving format and numeric type."""
    pos = node.get("pos")
    if isinstance(pos, dict):
        pos["0"] = _preserve_num_type(pos.get("0", 0), x)
        pos["1"] = _preserve_num_type(pos.get("1", 0), y)
    elif isinstance(pos, (list, tuple)):
        node["pos"] = [
            _preserve_num_type(pos[0], x),
            _preserve_num_type(pos[1], y),
        ]


def node_inside_bbox(nx, ny, bbox):
    """Check if a node position falls within a group bounding box [x, y, w, h]."""
    gx, gy, gw, gh = bbox
    return gx <= nx < gx + gw and gy <= ny < gy + gh


def shift_nodes_in_bbox(nodes, bbox, dx, dy):
    """Shift all nodes whose positions fall within the given bbox by (dx, dy)."""
    count = 0
    for node in nodes:
        nx, ny = get_node_pos(node)
        if nx is None:
            continue
        if node_inside_bbox(nx, ny, bbox):
            set_node_pos(node, nx + dx, ny + dy)
            count += 1
    return count


def groups_share_column(g1, g2, tolerance=200):
    """Check if two groups are in the same vertical column (similar x)."""
    x1 = g1["bounding"][0]
    x2 = g2["bounding"][0]
    return abs(x1 - x2) <= tolerance


def groups_share_row(g1, g2, tolerance=200):
    """Check if two groups are in the same horizontal row (similar y)."""
    y1 = g1["bounding"][1]
    y2 = g2["bounding"][1]
    return abs(y1 - y2) <= tolerance


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------

def detect_layout(groups):
    """
    Determine whether the groups are stacked vertically or horizontally.
    Returns 'vertical' or 'horizontal'.

    Heuristic: if more than half the groups share a similar y-coordinate
    (within 50px), treat as horizontal; otherwise vertical.
    """
    if len(groups) <= 1:
        return "vertical"

    y_values = [g["bounding"][1] for g in groups]
    best_count = 0
    for y_ref in y_values:
        count = sum(1 for y in y_values if abs(y - y_ref) <= 50)
        best_count = max(best_count, count)

    if best_count > len(groups) / 2:
        return "horizontal"
    return "vertical"


# ---------------------------------------------------------------------------
# Core fixer
# ---------------------------------------------------------------------------

def fix_workflow(filepath):
    """
    Fix overlapping groups in a single workflow file.
    Returns a summary dict with changes made.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    groups = data.get("groups", [])
    nodes = data.get("nodes", [])

    if not groups:
        return {"file": filepath, "changes": [], "layout": "unknown"}

    layout = detect_layout(groups)
    changes = []

    if layout == "vertical":
        changes = fix_vertical(groups, nodes)
    else:
        changes = fix_horizontal(groups, nodes)

    # Write back
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return {"file": filepath, "changes": changes, "layout": layout}


def fix_vertical(groups, nodes):
    """
    Fix vertically-stacked groups. Only fixes ACTUAL overlaps (where one
    group's bottom edge extends past the next group's top edge) among groups
    that share the same x-column. Cascades shifts to all subsequent groups
    in the same column.
    """
    changes = []

    # Identify the main column: groups whose x is close to the most common x
    x_values = [g["bounding"][0] for g in groups]
    # Find dominant x
    x_counts = {}
    for x in x_values:
        # Round to nearest 50 for bucketing
        bucket = round(x / 50) * 50
        x_counts[bucket] = x_counts.get(bucket, 0) + 1
    dominant_x = max(x_counts, key=x_counts.get)

    # Sort groups in the main column by y
    column_groups = [g for g in groups if abs(g["bounding"][0] - dominant_x) <= 200]
    column_groups.sort(key=lambda g: g["bounding"][1])

    # Track cumulative shift that has been applied to each group
    cumulative_shift = 0

    for i in range(len(column_groups) - 1):
        g_curr = column_groups[i]
        g_next = column_groups[i + 1]

        bx1, by1, bw1, bh1 = g_curr["bounding"]
        bx2, by2, bw2, bh2 = g_next["bounding"]

        curr_end_y = by1 + bh1
        # Check for actual overlap (end of current > start of next)
        if curr_end_y > by2:
            overlap = curr_end_y - by2
            delta_y = overlap + MIN_GAP
            # Shift this group and ALL subsequent column groups down
            for j in range(i + 1, len(column_groups)):
                g = column_groups[j]
                old_bbox = list(g["bounding"])
                g["bounding"][1] += delta_y
                nodes_moved = shift_nodes_in_bbox(nodes, old_bbox, dx=0, dy=delta_y)
                changes.append(
                    f"Shifted '{g.get('title', '?')}' down by {delta_y}px "
                    f"(y: {old_bbox[1]} -> {g['bounding'][1]}, {nodes_moved} nodes moved)"
                )

    return changes


def fix_horizontal(groups, nodes):
    """
    Fix horizontally-arranged groups. Only fixes ACTUAL overlaps (where one
    group's right edge extends past the next group's left edge) among groups
    in the same y-row. Cascades shifts to all subsequent groups in the row.
    """
    changes = []

    # Identify the main row: groups whose y is close to the most common y
    y_values = [g["bounding"][1] for g in groups]
    y_counts = {}
    for y in y_values:
        bucket = round(y / 50) * 50
        y_counts[bucket] = y_counts.get(bucket, 0) + 1
    dominant_y = max(y_counts, key=y_counts.get)

    # Get groups in the main row and sort by x
    row_groups = [g for g in groups if abs(g["bounding"][1] - dominant_y) <= 200]
    row_groups.sort(key=lambda g: g["bounding"][0])

    for i in range(len(row_groups) - 1):
        g_curr = row_groups[i]
        g_next = row_groups[i + 1]

        bx1, by1, bw1, bh1 = g_curr["bounding"]
        bx2, by2, bw2, bh2 = g_next["bounding"]

        curr_end_x = bx1 + bw1
        # Check for actual overlap
        if curr_end_x > bx2:
            overlap = curr_end_x - bx2
            delta_x = overlap + MIN_GAP
            # Shift this group and ALL subsequent row groups right
            for j in range(i + 1, len(row_groups)):
                g = row_groups[j]
                old_bbox = list(g["bounding"])
                g["bounding"][0] += delta_x
                nodes_moved = shift_nodes_in_bbox(nodes, old_bbox, dx=delta_x, dy=0)
                changes.append(
                    f"Shifted '{g.get('title', '?')}' right by {delta_x}px "
                    f"(x: {old_bbox[0]} -> {g['bounding'][0]}, {nodes_moved} nodes moved)"
                )

    return changes


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def find_all_overlaps(filepath):
    """
    Check a workflow for any overlapping groups (pairwise check of ALL groups).
    Returns a list of overlap descriptions (empty = no overlaps).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    groups = data.get("groups", [])
    if not groups:
        return []

    overlaps = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1 = groups[i]
            g2 = groups[j]
            b1 = g1["bounding"]  # [x, y, w, h]
            b2 = g2["bounding"]

            # Check 2D rectangle overlap
            x_overlap = (b1[0] < b2[0] + b2[2]) and (b2[0] < b1[0] + b1[2])
            y_overlap = (b1[1] < b2[1] + b2[3]) and (b2[1] < b1[1] + b1[3])

            if x_overlap and y_overlap:
                overlaps.append(
                    f"'{g1.get('title')}' (bbox={b1}) overlaps "
                    f"'{g2.get('title')}' (bbox={b2})"
                )

    return overlaps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Determine project root
    # Script lives at: <project_root>/custom_nodes/Laura_Image_Studio/scripts/
    script_dir = Path(__file__).resolve().parent
    submodule_root = script_dir.parent  # custom_nodes/Laura_Image_Studio
    project_root = submodule_root.parent.parent  # Best_Real_Image_Gen - Copy

    workflow_files = [
        submodule_root / "workflows" / "enhanced_master_workflow.json",
        submodule_root / "workflows" / "master_all_in_one.json",
        project_root / "workflows" / "master" / "Laura_Master_Studio_v0.8.json",
    ]

    # Verify all files exist
    all_exist = True
    for wf in workflow_files:
        if not wf.exists():
            print(f"ERROR: Workflow file not found: {wf}")
            all_exist = False
    if not all_exist:
        sys.exit(1)

    print("=" * 70)
    print("Laura Image Studio - Workflow Group Layout Fixer")
    print("=" * 70)

    # Pre-check: show existing overlaps
    print("\n--- Pre-fix overlap check ---")
    total_pre_overlaps = 0
    for wf in workflow_files:
        overlaps = find_all_overlaps(str(wf))
        total_pre_overlaps += len(overlaps)
        if overlaps:
            print(f"\n{wf.name}: {len(overlaps)} overlap(s)")
            for o in overlaps:
                print(f"  {o}")
        else:
            print(f"\n{wf.name}: no overlaps")
    print(f"\nTotal overlaps found: {total_pre_overlaps}")

    # Fix
    print("\n--- Applying fixes ---")
    total_changes = 0

    for wf in workflow_files:
        print(f"\nProcessing: {wf.name}")
        print(f"  Path: {wf}")

        result = fix_workflow(str(wf))
        print(f"  Layout: {result['layout']}")

        if result["changes"]:
            print(f"  Changes ({len(result['changes'])}):")
            for change in result["changes"]:
                print(f"    {change}")
            total_changes += len(result["changes"])
        else:
            print("  No changes needed.")

    # Verification pass
    print("\n" + "=" * 70)
    print("Verification Pass")
    print("=" * 70)

    all_clear = True
    for wf in workflow_files:
        overlaps = find_all_overlaps(str(wf))
        if overlaps:
            print(f"\nFAIL: {wf.name} still has {len(overlaps)} overlap(s):")
            for o in overlaps:
                print(f"  {o}")
            all_clear = False
        else:
            print(f"\nOK: {wf.name} - no overlaps detected")

    print(f"\nTotal changes made: {total_changes}")

    if all_clear:
        print("\nAll workflows verified: 0 overlaps remain.")
    else:
        print("\nWARNING: Some overlaps still remain!")
        sys.exit(1)


if __name__ == "__main__":
    main()
