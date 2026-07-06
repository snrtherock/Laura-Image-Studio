"""
Laura Image Studio - Model Catalog Node
Browseable catalog of all models in the registry with filtering and sorting.
"""

from .model_registry import (
    MODEL_REGISTRY,
    CATEGORIES,
    get_models_by_category,
    get_recommended_quantization,
    get_model_display_name,
)
from .model_registry import get_registry_version

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_CATEGORY_CHOICES = ["all"] + list(CATEGORIES.keys())
_SORT_CHOICES = ["name", "vram_min", "family", "license"]


def _vram_range(model):
    variants = model.get("quantization_variants", {})
    if not variants:
        return None, None
    vram_values = [v.get("vram_gb", 999) for v in variants.values()]
    return min(vram_values), max(vram_values)


def _min_vram(model):
    lo, _ = _vram_range(model)
    return lo if lo is not None else 999


def _fits_vram(model, vram_limit):
    lo, _ = _vram_range(model)
    return lo is not None and lo <= vram_limit


def _sort_key(item, sort_by):
    key, model = item
    if sort_by == "name":
        return model.get("display_name", key).lower()
    if sort_by == "vram_min":
        return _min_vram(model)
    if sort_by == "family":
        return (model.get("family", ""), model.get("display_name", key).lower())
    if sort_by == "license":
        return (model.get("license", "").lower(), model.get("display_name", key).lower())
    return key


class LauraModelCatalog:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "category": (_CATEGORY_CHOICES, {"default": "all"}),
                "sort_by": (_SORT_CHOICES, {"default": "name"}),
            },
            "optional": {
                "vram_filter": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 999.0, "step": 0.5}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("catalog_text", "model_count", "model_keys")
    FUNCTION = "browse"
    CATEGORY = "Laura Studio/Core"

    def browse(self, category, sort_by, vram_filter=0.0):
        if category == "all":
            models = dict(MODEL_REGISTRY)
        else:
            models = get_models_by_category(category)

        if vram_filter > 0:
            models = {k: v for k, v in models.items() if _fits_vram(v, vram_filter)}

        sorted_items = sorted(models.items(), key=lambda item: _sort_key(item, sort_by))

        if category == "all":
            cat_label = "All Categories"
        else:
            cat_label = CATEGORIES.get(category, {}).get("label", category)

        lines = [
            "=== Laura Studio Model Catalog ===",
            f"Category: {cat_label} | Models: {len(sorted_items)}",
            "",
        ]

        matching_keys = []
        for key, model in sorted_items:
            matching_keys.append(key)
            display = get_model_display_name(key)
            family = model.get("family", "unknown")
            params = model.get("params") or "N/A"
            lic = model.get("license") or "N/A"
            status = model.get("status") or "unknown"

            lo, hi = _vram_range(model)
            if lo is not None and hi is not None:
                vram_str = f"{lo}-{hi} GB"
            else:
                vram_str = "N/A"

            lines.append(f"  {key} — {display}")
            lines.append(f"    Family: {family} | Params: {params} | License: {lic}")
            lines.append(f"    VRAM: {vram_str} | Status: {status}")
            lines.append("")

        version = get_registry_version()
        total = len(MODEL_REGISTRY)
        lines.append(f"Registry v{version} | Total: {total} models")

        catalog_text = "\n".join(lines)
        model_keys_str = ",".join(matching_keys)

        return (catalog_text, len(matching_keys), model_keys_str)


NODE_CLASS_MAPPINGS.update(
    {
        "LauraModelCatalog": LauraModelCatalog,
    }
)

NODE_DISPLAY_NAME_MAPPINGS.update(
    {
        "LauraModelCatalog": "Model Catalog (Laura)",
    }
)
