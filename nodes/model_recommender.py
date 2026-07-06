"""
Laura Image Studio - Model Recommender Node
Scores and ranks models from the registry based on hardware, task, and preferences.
"""

from .model_registry import MODEL_REGISTRY, VRAM_TIERS, get_models_by_category, get_recommended_quantization

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

WELL_KNOWN_FAMILIES = {"flux", "wan", "z_image", "stable_diffusion", "hunyuan", "ltx", "cogvideo"}

LICENSE_SCORES = {
    "Apache-2.0": 1.0,
    "MIT": 1.0,
    "BSD": 1.0,
}
NON_COMMERCIAL_KEYWORDS = {"non-commercial", "nc", "tencent", "stability ai community"}


def _license_score(license_str, prefer_commercial):
    if not license_str:
        return 0.3
    normalized = license_str.lower()
    if license_str in LICENSE_SCORES:
        return 1.0 if not prefer_commercial else 1.2
    for kw in NON_COMMERCIAL_KEYWORDS:
        if kw in normalized:
            return 0.0
    return 0.3


def _vram_tier_from_gb(vram_gb):
    for tier_key, tier_info in VRAM_TIERS.items():
        if tier_info["min_gb"] <= vram_gb < tier_info["max_gb"]:
            return tier_key
    return "hpc"


def _score_model(model_key, model, vram_tier, vram_gb, prefer_commercial):
    rec_quant = get_recommended_quantization(model_key, vram_tier)

    hw_score = 0.0
    chosen_quant = None
    if rec_quant:
        variant = model.get("quantization_variants", {}).get(rec_quant, {})
        variant_vram = variant.get("vram_gb", 999)
        if variant_vram <= vram_gb:
            hw_score = 1.0
            chosen_quant = rec_quant
        elif variant_vram <= vram_gb * 1.2:
            hw_score = 0.5
            chosen_quant = rec_quant
    if chosen_quant is None:
        for q_name, q_info in sorted(
            model.get("quantization_variants", {}).items(),
            key=lambda x: x[1].get("vram_gb", 999),
        ):
            if q_info.get("vram_gb", 999) <= vram_gb:
                hw_score = 0.6
                chosen_quant = q_name
                break
    if chosen_quant is None:
        return None

    task_score = 1.0

    status = model.get("status", "released")
    stability_map = {"released": 1.0, "brand_new": 0.5, "pending_release": 0.1}
    stability_score = stability_map.get(status, 0.5)

    strengths = model.get("quality_score", {}).get("strengths", [])
    quality_score = min(len(strengths) / 4.0, 1.0)

    lic_score = _license_score(model.get("license"), prefer_commercial)

    family = model.get("family", "")
    popularity_score = 1.0 if family in WELL_KNOWN_FAMILIES else 0.4

    total = (
        0.30 * hw_score
        + 0.25 * task_score
        + 0.15 * stability_score
        + 0.15 * quality_score
        + 0.10 * lic_score
        + 0.05 * popularity_score
    )

    return {
        "key": model_key,
        "score": round(total, 2),
        "quant": chosen_quant,
        "vram_gb": model.get("quantization_variants", {}).get(chosen_quant, {}).get("vram_gb", 0),
        "strengths": strengths,
        "display_name": model.get("display_name", model_key),
    }


class LauraModelRecommender:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vram_tier": ("STRING", {"default": "medium"}),
                "task": (["image_gen", "image_edit", "video_gen", "upscale", "audio", "virtual_tryon"],),
            },
            "optional": {
                "vram_gb": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 999.0, "step": 0.5}),
                "prefer_commercial": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("recommended_model", "recommended_quant", "recommendation_report")
    FUNCTION = "recommend"
    CATEGORY = "Laura Studio/Core"

    def recommend(self, vram_tier, task, vram_gb=8.0, prefer_commercial=False):
        if vram_tier not in VRAM_TIERS:
            vram_tier = _vram_tier_from_gb(vram_gb)

        tier_info = VRAM_TIERS.get(vram_tier, {})
        tier_label = tier_info.get("label", vram_tier)

        candidates = get_models_by_category(task)
        scored = []
        for model_key, model in candidates.items():
            result = _score_model(model_key, model, vram_tier, vram_gb, prefer_commercial)
            if result is not None:
                scored.append(result)

        scored.sort(key=lambda x: x["score"], reverse=True)

        if not scored:
            return (
                "none",
                "none",
                f"=== Laura Studio Model Recommendation ===\n"
                f"Task: {task} | VRAM: {vram_gb:.0f} GB ({vram_tier})\n\n"
                f"No compatible models found for this configuration.\n"
                f"Try increasing VRAM or selecting a different task.",
            )

        top = scored[0]
        lines = [
            "=== Laura Studio Model Recommendation ===",
            f"Task: {task} | VRAM: {vram_gb:.0f} GB ({vram_tier})",
            "",
        ]

        for i, entry in enumerate(scored[:3], 1):
            strengths_str = ", ".join(entry["strengths"][:3]) if entry["strengths"] else "N/A"
            lines.append(f"#{i} {entry['display_name']} (Score: {entry['score']:.2f})")
            lines.append(f"   Quant: {entry['quant']} | VRAM: {entry['vram_gb']} GB")
            lines.append(f"   ✓ {strengths_str}")
            lines.append("")

        report = "\n".join(lines).rstrip()

        return (top["key"], top["quant"], report)


NODE_CLASS_MAPPINGS.update(
    {
        "LauraModelRecommender": LauraModelRecommender,
    }
)

NODE_DISPLAY_NAME_MAPPINGS.update(
    {
        "LauraModelRecommender": "Model Recommender (Laura)",
    }
)
