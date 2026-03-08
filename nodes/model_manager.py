"""
Laura Image Studio - Model Manager Nodes
Provides 7 ComfyUI nodes for model management, VRAM detection, quantization
advice, download helpers, compatibility checking, auto-configuration, and
model download link generation.

Nodes:
    1. LauraModelManager      - Central model selector with VRAM-aware defaults
    2. ModuleConfigPanel       - Per-module enable/disable and settings
    3. LauraQuantizationAdvisor - Recommends quantization for current VRAM
    4. LauraModelDownloadHelper - Shows download URLs and sizes for a model
    5. LauraCompatibilityChecker - Reports feature compatibility for a model
    6. LauraAutoConfig          - One-click optimal config for detected hardware
    7. LauraModelLinks          - Formatted download links with directory tree
"""

import json
import os

import torch
import folder_paths

from .model_registry import (
    MODEL_REGISTRY,
    VRAM_TIERS,
    CATEGORIES,
    get_models_by_category,
    get_recommended_quantization,
    get_download_url,
    get_all_required_files,
    get_registry_version,
)


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _detect_vram():
    """Detect available GPU VRAM in GB and return (vram_gb, tier_key)."""
    if not torch.cuda.is_available():
        return 0, "ultra_low"
    try:
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = round(vram_bytes / (1024 ** 3), 1)
    except Exception:
        return 0, "ultra_low"

    tier_key = "ultra_low"
    for key, tier in VRAM_TIERS.items():
        if tier["min_gb"] <= vram_gb < tier["max_gb"]:
            tier_key = key
            break
    return vram_gb, tier_key


def _scan_installed_models():
    """Scan ComfyUI model folders and return a set of installed filenames."""
    installed = set()
    for folder_type in ("checkpoints", "diffusion_models", "text_encoders",
                        "vae", "loras", "upscale_models"):
        try:
            for f in folder_paths.get_filename_list(folder_type):
                installed.add(os.path.basename(f))
        except Exception:
            pass
    return installed


def _get_config_path():
    """Return path to the Laura Studio config JSON file."""
    output_dir = folder_paths.get_output_directory()
    config_dir = os.path.join(os.path.dirname(output_dir), "laura_studio_config")
    os.makedirs(config_dir, exist_ok=True)
    return os.path.join(config_dir, "model_manager_config.json")


def _load_config():
    """Load saved configuration, returning an empty dict on failure."""
    try:
        path = _get_config_path()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception:
        pass
    return {}


def _save_config(config):
    """Persist configuration to disk."""
    try:
        path = _get_config_path()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)
    except Exception:
        pass


# =============================================================================
# NODE 1: LauraModelManager
# =============================================================================

class LauraModelManager:
    """Central model selector with VRAM-aware defaults and installation status."""

    @classmethod
    def INPUT_TYPES(cls):
        model_choices = sorted(MODEL_REGISTRY.keys())
        return {
            "required": {
                "model_key": (model_choices, {"default": model_choices[0]}),
                "auto_detect_vram": ("BOOLEAN", {"default": True}),
                "manual_vram_tier": (
                    sorted(VRAM_TIERS.keys()),
                    {"default": "medium"},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = (
        "model_info",
        "recommended_quant",
        "install_status",
        "model_key",
        "total_size_gb",
    )
    FUNCTION = "manage"
    CATEGORY = "Laura Studio/Model Manager"
    DESCRIPTION = (
        "Central model selector. Detects VRAM, recommends quantization, "
        "and reports installation status for any registered model."
    )

    def manage(self, model_key, auto_detect_vram, manual_vram_tier):
        entry = MODEL_REGISTRY.get(model_key)
        if entry is None:
            return ("Model not found", "N/A", "unknown", model_key, 0.0)

        # VRAM detection
        if auto_detect_vram:
            vram_gb, tier_key = _detect_vram()
        else:
            tier_key = manual_vram_tier
            tier = VRAM_TIERS.get(tier_key, {})
            vram_gb = tier.get("max_gb", 0)

        # Recommended quantization
        rec_quant = get_recommended_quantization(model_key, tier_key) or "bf16"

        # Installation status
        installed = _scan_installed_models()
        files = entry.get("files", {})
        if not files:
            install_status = "no_files_listed"
        else:
            all_installed = True
            for _role, file_info in files.items():
                if not isinstance(file_info, dict):
                    continue
                fname = file_info.get("filename", "")
                if fname and fname not in installed:
                    all_installed = False
                    break
            install_status = "installed" if all_installed else "not_installed"

        # Build info string
        info_lines = [
            f"Model: {entry.get('display_name', model_key)}",
            f"Family: {entry.get('family', 'unknown')}",
            f"Category: {entry.get('category', 'unknown')}",
            f"Parameters: {entry.get('params', 'N/A')}",
            f"Architecture: {entry.get('architecture', 'N/A')}",
            f"Status: {entry.get('status', 'unknown')}",
            f"License: {entry.get('license', 'N/A')}",
            f"Total Size: {entry.get('total_size_gb', 0)} GB",
            f"Detected VRAM: {vram_gb} GB ({tier_key})",
            f"Recommended Quantization: {rec_quant}",
        ]
        model_info = "\n".join(info_lines)

        return (
            model_info,
            rec_quant,
            install_status,
            model_key,
            float(entry.get("total_size_gb", 0)),
        )


# =============================================================================
# NODE 2: ModuleConfigPanel
# =============================================================================

class ModuleConfigPanel:
    """Per-module enable/disable panel for Laura Studio modules."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable_generation": ("BOOLEAN", {"default": True}),
                "enable_inpainting": ("BOOLEAN", {"default": True}),
                "enable_upscaling": ("BOOLEAN", {"default": True}),
                "enable_video": ("BOOLEAN", {"default": True}),
                "enable_face": ("BOOLEAN", {"default": True}),
                "enable_comparison": ("BOOLEAN", {"default": True}),
                "save_config": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("config_summary",)
    FUNCTION = "configure"
    CATEGORY = "Laura Studio/Model Manager"
    DESCRIPTION = (
        "Enable or disable individual Laura Studio modules. "
        "Optionally save the configuration to disk."
    )

    def configure(
        self,
        enable_generation,
        enable_inpainting,
        enable_upscaling,
        enable_video,
        enable_face,
        enable_comparison,
        save_config,
    ):
        config = {
            "generation": enable_generation,
            "inpainting": enable_inpainting,
            "upscaling": enable_upscaling,
            "video": enable_video,
            "face": enable_face,
            "comparison": enable_comparison,
        }

        if save_config:
            full_config = _load_config()
            full_config["modules"] = config
            _save_config(full_config)

        enabled = [m for m, v in config.items() if v]
        disabled = [m for m, v in config.items() if not v]
        summary_lines = [
            "Laura Studio Module Configuration",
            "=" * 40,
            f"Enabled  ({len(enabled)}): {', '.join(enabled) if enabled else 'none'}",
            f"Disabled ({len(disabled)}): {', '.join(disabled) if disabled else 'none'}",
        ]
        if save_config:
            summary_lines.append("Configuration saved to disk.")

        return ("\n".join(summary_lines),)


# =============================================================================
# NODE 3: LauraQuantizationAdvisor
# =============================================================================

class LauraQuantizationAdvisor:
    """Recommends quantization variants for a model based on current VRAM."""

    @classmethod
    def INPUT_TYPES(cls):
        model_choices = sorted(MODEL_REGISTRY.keys())
        return {
            "required": {
                "model_key": (model_choices, {"default": model_choices[0]}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("recommendation", "all_variants")
    FUNCTION = "advise"
    CATEGORY = "Laura Studio/Model Manager"
    DESCRIPTION = (
        "Analyzes your GPU VRAM and recommends the best quantization "
        "variant for the selected model."
    )

    def advise(self, model_key):
        entry = MODEL_REGISTRY.get(model_key)
        if entry is None:
            return ("Model not found.", "{}")

        vram_gb, tier_key = _detect_vram()
        rec = get_recommended_quantization(model_key, tier_key) or "N/A"

        variants = entry.get("quantization_variants", {})
        if not variants:
            return (
                f"No quantization variants available for {entry.get('display_name', model_key)}.",
                "{}",
            )

        lines = [
            f"Quantization Advisor: {entry.get('display_name', model_key)}",
            f"Detected VRAM: {vram_gb} GB (tier: {tier_key})",
            f"Recommended: {rec}",
            "",
            "All available variants:",
        ]
        for name, info in variants.items():
            marker = " << RECOMMENDED" if name == rec else ""
            lines.append(
                f"  {name}: VRAM {info.get('vram_gb', '?')} GB, "
                f"quality={info.get('quality', '?')}, "
                f"speed={info.get('speed', '?')}{marker}"
            )

        return ("\n".join(lines), json.dumps(variants, indent=2))


# =============================================================================
# NODE 4: LauraModelDownloadHelper
# =============================================================================

class LauraModelDownloadHelper:
    """Shows download URLs, file sizes, and installation paths for a model."""

    @classmethod
    def INPUT_TYPES(cls):
        model_choices = sorted(MODEL_REGISTRY.keys())
        return {
            "required": {
                "model_key": (model_choices, {"default": model_choices[0]}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("download_info", "urls_json")
    FUNCTION = "get_downloads"
    CATEGORY = "Laura Studio/Model Manager"
    DESCRIPTION = (
        "Lists all required files for a model with download URLs, sizes, "
        "and target installation folders."
    )

    def get_downloads(self, model_key):
        entry = MODEL_REGISTRY.get(model_key)
        if entry is None:
            return ("Model not found.", "{}")

        files = get_all_required_files(model_key)
        if not files:
            return (
                f"No downloadable files listed for {entry.get('display_name', model_key)}.",
                "[]",
            )

        lines = [
            f"Download Helper: {entry.get('display_name', model_key)}",
            f"Total Size: {entry.get('total_size_gb', 0)} GB",
            f"Repository: {entry.get('repo', 'N/A')}",
            "",
        ]
        url_list = []
        for f in files:
            lines.append(f"  File: {f['filename']}")
            lines.append(f"    Folder: ComfyUI/models/{f['folder']}/")
            lines.append(f"    Size:   {f['size_gb']} GB")
            lines.append(f"    URL:    {f['url'] or 'N/A'}")
            lines.append("")
            if f["url"]:
                url_list.append({
                    "filename": f["filename"],
                    "folder": f["folder"],
                    "size_gb": f["size_gb"],
                    "url": f["url"],
                })

        return ("\n".join(lines), json.dumps(url_list, indent=2))


# =============================================================================
# NODE 5: LauraCompatibilityChecker
# =============================================================================

class LauraCompatibilityChecker:
    """Reports feature compatibility (LoRA, ControlNet, etc.) for a model."""

    @classmethod
    def INPUT_TYPES(cls):
        model_choices = sorted(MODEL_REGISTRY.keys())
        return {
            "required": {
                "model_key": (model_choices, {"default": model_choices[0]}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("compatibility_report", "compatibility_json")
    FUNCTION = "check"
    CATEGORY = "Laura Studio/Model Manager"
    DESCRIPTION = (
        "Checks and reports feature compatibility for a model: LoRA, "
        "ControlNet, IPAdapter, negative prompt, and more."
    )

    def check(self, model_key):
        entry = MODEL_REGISTRY.get(model_key)
        if entry is None:
            return ("Model not found.", "{}")

        compat = entry.get("compatibility", {})
        display = entry.get("display_name", model_key)

        # Feature labels
        features = [
            ("LoRA", compat.get("lora")),
            ("ControlNet", compat.get("controlnet")),
            ("IPAdapter", compat.get("ipadapter")),
            ("Negative Prompt", compat.get("negative_prompt")),
            ("Multi-Reference", compat.get("multi_reference")),
            ("Image Editing", compat.get("image_editing")),
        ]

        lines = [
            f"Compatibility Report: {display}",
            "=" * 40,
        ]
        for label, supported in features:
            if supported is True:
                status = "YES"
            elif supported is False:
                status = "NO"
            else:
                status = "UNKNOWN"
            lines.append(f"  {label}: {status}")

        # ControlNet models
        cn_models = compat.get("controlnet_models", [])
        if cn_models:
            lines.append(f"\n  ControlNet Models: {', '.join(cn_models)}")

        # LoRA zoo
        lora_zoo = compat.get("lora_zoo")
        if lora_zoo:
            lines.append(f"  LoRA Zoo: {lora_zoo}")
            zoo_models = compat.get("lora_zoo_models", [])
            for lm in zoo_models:
                lines.append(f"    - {lm.get('name', '?')}: {lm.get('description', '')}")

        # Quality scores
        quality = entry.get("quality_score", {})
        strengths = quality.get("strengths", [])
        weaknesses = quality.get("weaknesses", [])
        if strengths:
            lines.append(f"\n  Strengths: {', '.join(strengths)}")
        if weaknesses:
            lines.append(f"  Weaknesses: {', '.join(weaknesses)}")

        return ("\n".join(lines), json.dumps(compat, indent=2))


# =============================================================================
# NODE 6: LauraAutoConfig
# =============================================================================

class LauraAutoConfig:
    """One-click optimal configuration for detected hardware."""

    @classmethod
    def INPUT_TYPES(cls):
        category_choices = sorted(CATEGORIES.keys())
        return {
            "required": {
                "target_category": (category_choices, {"default": "image_gen"}),
                "prefer_speed": ("BOOLEAN", {"default": False}),
                "prefer_quality": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("auto_config", "recommended_model", "recommended_quant")
    FUNCTION = "auto_configure"
    CATEGORY = "Laura Studio/Model Manager"
    DESCRIPTION = (
        "Automatically detects hardware and recommends the best model "
        "and quantization for a given category."
    )

    def auto_configure(self, target_category, prefer_speed, prefer_quality):
        vram_gb, tier_key = _detect_vram()
        installed = _scan_installed_models()

        category_models = get_models_by_category(target_category)
        if not category_models:
            return (
                f"No models found for category '{target_category}'.",
                "N/A",
                "N/A",
            )

        # Score models
        best_key = None
        best_score = -1

        for mkey, mentry in category_models.items():
            score = 0

            # Prefer models with files listed
            if mentry.get("files"):
                score += 10

            # Prefer installed models
            all_installed = True
            for _role, finfo in mentry.get("files", {}).items():
                if not isinstance(finfo, dict):
                    continue
                fname = finfo.get("filename", "")
                if fname and fname not in installed:
                    all_installed = False
                    break
            if all_installed and mentry.get("files"):
                score += 20

            # Prefer models that fit in VRAM
            rec = get_recommended_quantization(mkey, tier_key)
            if rec:
                variant = mentry.get("quantization_variants", {}).get(rec, {})
                if variant.get("vram_gb", 999) <= vram_gb:
                    score += 15

            # Speed preference
            if prefer_speed:
                inference = mentry.get("inference", {})
                steps = inference.get("steps")
                if steps and steps <= 8:
                    score += 10
                elif steps and steps <= 20:
                    score += 5
                if not inference.get("cfg"):
                    score += 3

            # Quality preference
            if prefer_quality:
                compat = mentry.get("compatibility", {})
                if compat.get("lora"):
                    score += 5
                if compat.get("controlnet"):
                    score += 3
                if compat.get("negative_prompt"):
                    score += 2

            # Skip pending release
            if mentry.get("status") == "pending_release":
                score -= 50

            if score > best_score:
                best_score = score
                best_key = mkey

        if best_key is None:
            return ("Could not determine best model.", "N/A", "N/A")

        best_entry = MODEL_REGISTRY[best_key]
        rec_quant = get_recommended_quantization(best_key, tier_key) or "bf16"

        config_lines = [
            "Laura Studio Auto-Configuration",
            "=" * 40,
            f"Detected VRAM: {vram_gb} GB (tier: {tier_key})",
            f"Target Category: {target_category}",
            f"Preference: {'Speed' if prefer_speed else ''}"
            f"{'Quality' if prefer_quality else ''}",
            "",
            f"Recommended Model: {best_entry.get('display_name', best_key)}",
            f"  Key: {best_key}",
            f"  Parameters: {best_entry.get('params', 'N/A')}",
            f"  Total Size: {best_entry.get('total_size_gb', 0)} GB",
            f"  Quantization: {rec_quant}",
            f"  Status: {best_entry.get('status', 'unknown')}",
        ]

        # Inference settings
        inference = best_entry.get("inference", {})
        if inference.get("steps"):
            config_lines.append(f"\n  Inference Settings:")
            config_lines.append(f"    Steps: {inference['steps']}")
            if inference.get("cfg"):
                config_lines.append(f"    CFG Scale: {inference.get('cfg_scale', 'N/A')}")
            else:
                config_lines.append(f"    CFG: Disabled")
            if inference.get("default_resolution"):
                res = inference["default_resolution"]
                config_lines.append(
                    f"    Default Resolution: {res.get('width', '?')}x{res.get('height', '?')}"
                )

        return ("\n".join(config_lines), best_key, rec_quant)


# =============================================================================
# NODE 7: LauraModelLinks
# =============================================================================

_MODEL_FAMILY_CHOICES = [
    "z_image_turbo",
    "z_image_base",
    "flux1_dev",
    "flux1_schnell",
    "flux2_dev",
    "sd35_medium",
    "wan22_14b",
    "hunyuan_video",
    "firered_edit_1_1",
    "helios_distilled",
    "ltx_2_3",
    "all_image_gen",
    "all_video_gen",
]


def _build_file_url(repo, folder, filename):
    """Construct a HuggingFace resolve/main download URL.

    If the file's folder path is different from its role name (e.g. the file
    lives in a subfolder of the repo), we include it in the URL path.
    """
    if not repo or repo == "TBD" or not filename:
        return None
    # Some files live inside subfolders that match their folder type
    # (e.g. text_encoders/qwen_3_4b.safetensors in the repo tree).
    # We check whether the folder name implies a subfolder in the repo.
    # For diffusion_models the file is typically at root; for text_encoders/vae
    # it may be in a subfolder.  We include folder if it is one of the known
    # subfolder types.
    subfolder_types = {"text_encoders", "vae", "loras"}
    if folder in subfolder_types:
        return f"https://huggingface.co/{repo}/resolve/main/{folder}/{filename}"
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def _gather_model_files(model_key):
    """Return a list of dicts with folder, filename, size_gb, url for a model.

    The registry stores files as: {"role_name": {"folder": ..., "filename": ..., "size_gb": ...}}.
    Each value is a single file info dict with folder, filename, and size_gb.
    """
    entry = MODEL_REGISTRY.get(model_key)
    if entry is None:
        return [], None, 0
    repo = entry.get("repo", "")
    display_name = entry.get("display_name", model_key)
    total_size = entry.get("total_size_gb", 0)
    files_info = []
    for _role, file_info in entry.get("files", {}).items():
        if not isinstance(file_info, dict):
            continue
        folder = file_info.get("folder", _role)
        filename = file_info.get("filename", "")
        size_gb = file_info.get("size_gb", 0)
        url = _build_file_url(repo, folder, filename)
        if filename:
            files_info.append({
                "folder": folder,
                "filename": filename,
                "size_gb": size_gb,
                "url": url,
            })
    return files_info, display_name, total_size


def _format_model_links(model_keys, combined_label=None):
    """Build the three output strings for one or more model keys.

    Returns (model_links, directory_tree, download_urls_json).
    """
    registry_version = get_registry_version()

    # Collect all files grouped by folder type, across all requested models
    # Use ordered dict approach to keep folder ordering consistent
    folder_order = [
        "text_encoders", "clip", "clip_vision", "loras",
        "diffusion_models", "checkpoints", "vae", "upscale_models",
    ]
    files_by_folder = {}  # folder -> list of {filename, size_gb, url, model_display}
    all_urls = []
    grand_total = 0
    display_names = []

    for mkey in model_keys:
        files_info, display_name, total_size = _gather_model_files(mkey)
        if display_name and display_name not in display_names:
            display_names.append(display_name)
        grand_total += total_size
        for f in files_info:
            folder = f["folder"]
            if folder not in files_by_folder:
                files_by_folder[folder] = []
            # Avoid duplicate files (e.g. shared VAE)
            existing_filenames = [x["filename"] for x in files_by_folder[folder]]
            if f["filename"] not in existing_filenames:
                files_by_folder[folder].append({
                    "filename": f["filename"],
                    "size_gb": f["size_gb"],
                    "url": f["url"],
                    "model_display": display_name,
                })
                if f["url"]:
                    all_urls.append({
                        "filename": f["filename"],
                        "folder": folder,
                        "size_gb": f["size_gb"],
                        "url": f["url"],
                    })

    # Determine title
    if combined_label:
        title = combined_label
    elif len(display_names) == 1:
        title = display_names[0]
    else:
        title = ", ".join(display_names) if display_names else "Unknown"

    # ---- model_links output ----
    separator = "\u2550" * 47  # ═
    section_sep = "\u2500" * 30  # ─

    link_lines = [
        separator,
        f"  MODEL DOWNLOAD LINKS: {title}",
        f"  Registry v{registry_version}",
        separator,
    ]

    # Sort folders in canonical order
    ordered_folders = [f for f in folder_order if f in files_by_folder]
    # Add any remaining folders not in the predefined order
    for f in files_by_folder:
        if f not in ordered_folders:
            ordered_folders.append(f)

    for folder in ordered_folders:
        file_list = files_by_folder[folder]
        link_lines.append("")
        link_lines.append(f"  \U0001F4C1 {folder}")
        link_lines.append(f"  {section_sep}")
        for fentry in file_list:
            if fentry["url"]:
                link_lines.append(
                    f"    [LINK:{fentry['filename']}|{fentry['url']}]"
                    f"  ({fentry['size_gb']:.2f} GB)"
                )
            else:
                link_lines.append(
                    f"    {fentry['filename']}  ({fentry['size_gb']:.2f} GB)"
                    f"  [no download URL available]"
                )

    link_lines.append("")
    link_lines.append(f"  Total: {grand_total:.2f} GB")
    link_lines.append(separator)

    model_links = "\n".join(link_lines)

    # ---- directory_tree output ----
    tree_lines = ["Model Storage Location:", "ComfyUI/", "\u251C\u2500\u2500 models/"]

    for idx, folder in enumerate(ordered_folders):
        file_list = files_by_folder[folder]
        is_last_folder = (idx == len(ordered_folders) - 1)

        if is_last_folder:
            tree_lines.append(f"\u2502   \u2514\u2500\u2500 {folder}/")
            prefix = "\u2502       "
        else:
            tree_lines.append(f"\u2502   \u251C\u2500\u2500 {folder}/")
            prefix = "\u2502   \u2502   "

        for fidx, fentry in enumerate(file_list):
            is_last_file = (fidx == len(file_list) - 1)
            if is_last_file:
                if is_last_folder:
                    tree_lines.append(f"\u2502       \u2514\u2500\u2500 {fentry['filename']}")
                else:
                    tree_lines.append(f"\u2502   \u2502   \u2514\u2500\u2500 {fentry['filename']}")
            else:
                tree_lines.append(f"{prefix}\u251C\u2500\u2500 {fentry['filename']}")

    directory_tree = "\n".join(tree_lines)

    # ---- download_urls_json output ----
    download_urls_json = json.dumps(all_urls, indent=2)

    return model_links, directory_tree, download_urls_json


class LauraModelLinks:
    """Generates formatted download links and directory tree for model families.

    Outputs use [LINK:filename|url] markers that the JavaScript web extension
    parses into clickable download links.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_family": (_MODEL_FAMILY_CHOICES, {"default": "z_image_turbo"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("model_links", "directory_tree", "download_urls_json")
    FUNCTION = "get_links"
    CATEGORY = "Laura Studio/Model Manager"
    DESCRIPTION = (
        "Shows download links for model files with clickable URLs and an "
        "ASCII directory tree showing where each file goes in ComfyUI/models/. "
        "Select a specific model or use all_image_gen / all_video_gen for "
        "combined listings."
    )
    OUTPUT_NODE = True

    def get_links(self, model_family):
        # Handle combined categories
        if model_family == "all_image_gen":
            cat_models = get_models_by_category("image_gen")
            # Also include image_edit
            edit_models = get_models_by_category("image_edit")
            cat_models.update(edit_models)
            model_keys = list(cat_models.keys())
            return _format_model_links(model_keys, combined_label="All Image Generation Models")

        elif model_family == "all_video_gen":
            cat_models = get_models_by_category("video_gen")
            model_keys = list(cat_models.keys())
            return _format_model_links(model_keys, combined_label="All Video Generation Models")

        else:
            # Single model key
            if model_family not in MODEL_REGISTRY:
                return (
                    f"Model '{model_family}' not found in registry.",
                    "",
                    "[]",
                )
            return _format_model_links([model_family])


# =============================================================================
# REGISTER ALL NODES
# =============================================================================

NODE_CLASS_MAPPINGS["LauraModelManager"] = LauraModelManager
NODE_CLASS_MAPPINGS["ModuleConfigPanel"] = ModuleConfigPanel
NODE_CLASS_MAPPINGS["LauraQuantizationAdvisor"] = LauraQuantizationAdvisor
NODE_CLASS_MAPPINGS["LauraModelDownloadHelper"] = LauraModelDownloadHelper
NODE_CLASS_MAPPINGS["LauraCompatibilityChecker"] = LauraCompatibilityChecker
NODE_CLASS_MAPPINGS["LauraAutoConfig"] = LauraAutoConfig
NODE_CLASS_MAPPINGS["LauraModelLinks"] = LauraModelLinks

NODE_DISPLAY_NAME_MAPPINGS["LauraModelManager"] = "Laura Model Manager"
NODE_DISPLAY_NAME_MAPPINGS["ModuleConfigPanel"] = "Laura Module Config"
NODE_DISPLAY_NAME_MAPPINGS["LauraQuantizationAdvisor"] = "Laura Quantization Advisor"
NODE_DISPLAY_NAME_MAPPINGS["LauraModelDownloadHelper"] = "Laura Download Helper"
NODE_DISPLAY_NAME_MAPPINGS["LauraCompatibilityChecker"] = "Laura Compatibility Checker"
NODE_DISPLAY_NAME_MAPPINGS["LauraAutoConfig"] = "Laura Auto Config"
NODE_DISPLAY_NAME_MAPPINGS["LauraModelLinks"] = "Laura Model Links"
