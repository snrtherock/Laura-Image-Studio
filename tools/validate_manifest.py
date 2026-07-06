#!/usr/bin/env python3
"""
Laura Image Studio - Model Registry Validator
Validates every entry in MODEL_REGISTRY for well-formed fields, consistent
cross-references, and sane numeric ranges.

Usage:
    python validate_manifest.py            # summary only
    python validate_manifest.py --verbose  # per-model detail

Exits 0 when no errors are found, 1 otherwise.
A Markdown report is written to  reports/manifest_validation.md
"""

import argparse
import os
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# Bootstrap: add the package root so we can import nodes.model_registry
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.dirname(_SCRIPT_DIR)           # Laura_Image_Studio/
sys.path.insert(0, _PACKAGE_ROOT)

from nodes.model_registry import MODEL_REGISTRY, CATEGORIES, VRAM_TIERS  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_CATEGORIES = set(CATEGORIES.keys())

REQUIRED_STRING_FIELDS = [
    "display_name",
    "family",
    "category",
    "repo",
    "license",
    "architecture",
    "status",
    "comfyui_type",
]

REQUIRED_FILE_KEYS = {"folder", "filename", "size_gb"}
REQUIRED_QVARIANT_KEYS = {"vram_gb", "quality", "speed"}

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

class ValidationResult:
    """Accumulates errors and warnings for one model key."""

    def __init__(self, key: str):
        self.key = key
        self.errors: list[str] = []
        self.warnings: list[str] = []

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def _is_nonempty_str(val) -> bool:
    return isinstance(val, str) and len(val.strip()) > 0


def validate_model(key: str, entry: dict) -> ValidationResult:
    """Run all checks on a single model entry, return result."""
    r = ValidationResult(key)

    # -- required string fields ------------------------------------------
    for field in REQUIRED_STRING_FIELDS:
        val = entry.get(field)
        if not _is_nonempty_str(val):
            r.error(f"Missing or empty required string field '{field}'")

    # -- category in CATEGORIES ------------------------------------------
    cat = entry.get("category", "")
    if _is_nonempty_str(cat) and cat not in VALID_CATEGORIES:
        r.error(
            f"category '{cat}' not in CATEGORIES "
            f"(valid: {sorted(VALID_CATEGORIES)})"
        )

    # -- files (dict, non-empty, each sub-entry well-formed) -------------
    files = entry.get("files")
    if not isinstance(files, dict) or len(files) == 0:
        r.error("'files' must be a non-empty dict")
    else:
        for fname, finfo in files.items():
            if not isinstance(finfo, dict):
                r.error(f"files['{fname}'] is not a dict")
                continue
            missing = REQUIRED_FILE_KEYS - set(finfo.keys())
            if missing:
                r.error(
                    f"files['{fname}'] missing keys: {sorted(missing)}"
                )
            else:
                if not _is_nonempty_str(finfo.get("folder")):
                    r.error(f"files['{fname}'].folder is empty")
                if not _is_nonempty_str(finfo.get("filename")):
                    r.error(f"files['{fname}'].filename is empty")
                sg = finfo.get("size_gb")
                if not isinstance(sg, (int, float)) or sg <= 0:
                    r.error(f"files['{fname}'].size_gb must be > 0")

    # -- total_size_gb ---------------------------------------------------
    total = entry.get("total_size_gb")
    if not isinstance(total, (int, float)) or total <= 0:
        r.error("'total_size_gb' must be a number > 0")

    # -- quantization_variants -------------------------------------------
    qv = entry.get("quantization_variants")
    if not isinstance(qv, dict):
        r.error("'quantization_variants' must be a dict")
    else:
        if len(qv) == 0:
            r.warn("'quantization_variants' is empty")
        for qname, qinfo in qv.items():
            if not isinstance(qinfo, dict):
                r.error(f"quantization_variants['{qname}'] is not a dict")
                continue
            missing = REQUIRED_QVARIANT_KEYS - set(qinfo.keys())
            if missing:
                r.error(
                    f"quantization_variants['{qname}'] missing keys: "
                    f"{sorted(missing)}"
                )
            else:
                vg = qinfo.get("vram_gb")
                if not isinstance(vg, (int, float)) or vg <= 0:
                    r.error(
                        f"quantization_variants['{qname}'].vram_gb "
                        f"must be > 0"
                    )

    # -- compatibility (dict) --------------------------------------------
    compat = entry.get("compatibility")
    if not isinstance(compat, dict):
        r.error("'compatibility' must be a dict")

    # -- quality_score (dict with elo_rank, strengths, weaknesses) -------
    qs = entry.get("quality_score")
    if not isinstance(qs, dict):
        r.error("'quality_score' must be a dict")
    else:
        if "elo_rank" not in qs:
            r.warn("quality_score missing 'elo_rank' key")
        if "strengths" not in qs:
            r.warn("quality_score missing 'strengths' key")
        if "weaknesses" not in qs:
            r.warn("quality_score missing 'weaknesses' key")

    # -- sanity: min VRAM across quant variants vs total_size_gb ---------
    if isinstance(qv, dict) and qv and isinstance(total, (int, float)):
        vram_values = [
            v.get("vram_gb")
            for v in qv.values()
            if isinstance(v, dict) and isinstance(v.get("vram_gb"), (int, float))
        ]
        if vram_values:
            min_vram = min(vram_values)
            # Sanity: minimum VRAM should not exceed 3x the total model size
            if min_vram > total * 3:
                r.warn(
                    f"min VRAM ({min_vram} GB) > 3x total_size_gb "
                    f"({total} GB) -- seems unusually high"
                )

    return r


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_report(results: list[ValidationResult]) -> str:
    """Return a Markdown report string."""
    total = len(results)
    valid = sum(1 for r in results if r.ok and not r.warnings)
    with_warnings = sum(1 for r in results if r.ok and r.warnings)
    with_errors = sum(1 for r in results if not r.ok)
    total_errors = sum(len(r.errors) for r in results)
    total_warnings = sum(len(r.warnings) for r in results)

    lines = [
        "# Model Registry Validation Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| Total models | {total} |",
        f"| Valid (no issues) | {valid} |",
        f"| Warnings only | {with_warnings} |",
        f"| With errors | {with_errors} |",
        f"| Total errors | {total_errors} |",
        f"| Total warnings | {total_warnings} |",
        "",
    ]

    if with_errors == 0 and total_warnings == 0:
        lines.append("All models passed validation.")
        lines.append("")

    # Per-model details (only models with issues)
    problem_results = [r for r in results if r.errors or r.warnings]
    if problem_results:
        lines.append("## Issues")
        lines.append("")
        for r in problem_results:
            status = "ERROR" if r.errors else "WARN"
            lines.append(f"### `{r.key}` [{status}]")
            lines.append("")
            if r.errors:
                for e in r.errors:
                    lines.append(f"- **Error:** {e}")
            if r.warnings:
                for w in r.warnings:
                    lines.append(f"- Warning: {w}")
            lines.append("")

    # Category breakdown
    cat_counts: dict[str, int] = {}
    for r in results:
        entry = MODEL_REGISTRY.get(r.key, {})
        cat = entry.get("category", "unknown")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    lines.append("## Models by Category")
    lines.append("")
    lines.append("| Category | Count |")
    lines.append("|----------|-------|")
    for cat in sorted(cat_counts):
        label = CATEGORIES.get(cat, {}).get("label", cat)
        lines.append(f"| {label} | {cat_counts[cat]} |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Laura Image Studio model registry"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-model validation details",
    )
    args = parser.parse_args()

    print(f"Validating MODEL_REGISTRY ({len(MODEL_REGISTRY)} models)...\n")

    results: list[ValidationResult] = []
    for key in sorted(MODEL_REGISTRY):
        entry = MODEL_REGISTRY[key]
        r = validate_model(key, entry)
        results.append(r)

        if args.verbose:
            status = "OK" if r.ok else "FAIL"
            if r.warnings and r.ok:
                status = "WARN"
            print(f"  [{status}] {key}")
            for e in r.errors:
                print(f"         ERROR: {e}")
            for w in r.warnings:
                print(f"         WARN:  {w}")

    # Totals
    total = len(results)
    n_valid = sum(1 for r in results if r.ok and not r.warnings)
    n_warn = sum(1 for r in results if r.ok and r.warnings)
    n_err = sum(1 for r in results if not r.ok)
    total_errors = sum(len(r.errors) for r in results)
    total_warnings = sum(len(r.warnings) for r in results)

    print(f"\n{'='*50}")
    print(f"  Total models:    {total}")
    print(f"  Valid:           {n_valid}")
    print(f"  Warnings only:   {n_warn}")
    print(f"  With errors:     {n_err}")
    print(f"  Total errors:    {total_errors}")
    print(f"  Total warnings:  {total_warnings}")
    print(f"{'='*50}")

    # Write report
    report_dir = os.path.join(_PACKAGE_ROOT, "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "manifest_validation.md")
    report = build_report(results)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    if n_err > 0:
        print("\nFAILED: validation errors found.")
        return 1
    else:
        print("\nPASSED: no errors found.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
