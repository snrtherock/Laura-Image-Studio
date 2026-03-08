"""
Laura Image Studio - Control Center Supernode

Provides a single "Laura Control Center" node that replaces 6+ scattered
model-loading nodes.  It handles VRAM detection, model selection across four
category dropdowns (image / upscale / video / face), three LoRA slots, and
outputs MODEL / CLIP / VAE plus metadata strings to the entire workflow.

Node:
    LauraControlCenter  -  One-stop pipeline configuration
"""

import json
import os
import traceback

import torch
import folder_paths
import comfy.sd
import comfy.utils

from .model_registry import (
    MODEL_REGISTRY,
    VRAM_TIERS,
    get_image_model_choices,
    get_video_model_choices,
    get_upscale_model_choices,
    get_recommended_quantization,
    get_all_required_files,
    get_model_display_name,
)
from .model_manager import _detect_vram, _scan_installed_models


# =============================================================================
# MODULE-LEVEL REGISTRATION DICTS  (same pattern as model_manager.py)
# =============================================================================

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


# =============================================================================
# HELPERS
# =============================================================================

def _lora_choices():
    """Return the list of available LoRA filenames, gracefully handling errors.

    If ComfyUI's ``folder_paths`` cannot enumerate the loras folder (e.g. the
    folder does not exist yet), an empty list is returned so that the dropdown
    still renders with just the ``"None"`` sentinel.
    """
    try:
        return folder_paths.get_filename_list("loras")
    except Exception:
        return []


def _load_image_model(model_key, model_info):
    """Load the primary image model and return (model, clip, vae).

    Dispatches between checkpoint-based loading and diffusion-model-based
    loading depending on the ``comfyui_type`` field in the registry entry.

    Args:
        model_key:  Registry key string (e.g. ``"flux1_dev"``).
        model_info: The full registry dict for this model.

    Returns:
        Tuple of ``(model, clip, vae)`` where any element may be ``None`` if
        the corresponding component is unavailable.
    """
    comfyui_type = model_info.get("comfyui_type", "checkpoints")
    files = model_info.get("files", {})

    if comfyui_type == "checkpoints":
        # ---- Checkpoint loading path ----
        ckpt_info = files.get("checkpoint", {})
        ckpt_folder = ckpt_info.get("folder", "checkpoints")
        ckpt_filename = ckpt_info.get("filename", "")
        if not ckpt_filename:
            raise FileNotFoundError(
                f"No checkpoint filename defined for model '{model_key}'."
            )

        ckpt_path = folder_paths.get_full_path(ckpt_folder, ckpt_filename)
        if ckpt_path is None:
            raise FileNotFoundError(
                f"Checkpoint file '{ckpt_filename}' not found in "
                f"'{ckpt_folder}' folder.  Please download the model first."
            )

        # Embedding directory for CLIP tokenizer
        embedding_dirs = folder_paths.get_folder_paths("embeddings")
        embedding_dir = embedding_dirs[0] if embedding_dirs else None

        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=embedding_dir,
        )
        # load_checkpoint_guess_config returns (model, clip, vae, ...)
        model = out[0]
        clip = out[1]
        vae = out[2]
        return model, clip, vae

    elif comfyui_type == "diffusion_models":
        # ---- Diffusion model loading path ----
        dm_info = files.get("diffusion_model", {})
        dm_folder = dm_info.get("folder", "diffusion_models")
        dm_filename = dm_info.get("filename", "")
        if not dm_filename:
            raise FileNotFoundError(
                f"No diffusion_model filename defined for model '{model_key}'."
            )

        dm_path = folder_paths.get_full_path(dm_folder, dm_filename)
        if dm_path is None:
            raise FileNotFoundError(
                f"Diffusion model file '{dm_filename}' not found in "
                f"'{dm_folder}' folder.  Please download the model first."
            )

        model = comfy.sd.load_diffusion_model(dm_path)

        # Optional VAE
        vae = None
        vae_info = files.get("vae", {})
        vae_filename = vae_info.get("filename", "")
        if vae_filename:
            vae_folder = vae_info.get("folder", "vae")
            vae_path = folder_paths.get_full_path(vae_folder, vae_filename)
            if vae_path is not None:
                vae_sd = comfy.utils.load_torch_file(vae_path, safe_load=True)
                vae = comfy.sd.VAE(sd=vae_sd)

        # Optional CLIP / text encoder
        clip = None
        te_info = files.get("text_encoder", {})
        te_filename = te_info.get("filename", "")
        if te_filename:
            te_folder = te_info.get("folder", "text_encoders")
            te_path = folder_paths.get_full_path(te_folder, te_filename)
            if te_path is not None:
                clip_sd = comfy.utils.load_torch_file(te_path, safe_load=True)
                clip = comfy.sd.load_clip(clip_sd)

        return model, clip, vae

    else:
        raise ValueError(
            f"Unsupported comfyui_type '{comfyui_type}' for model '{model_key}'."
        )


def _apply_lora(model, clip, lora_name, strength):
    """Load a LoRA file and apply it to the model and CLIP.

    Args:
        model:     The MODEL object to patch.
        clip:      The CLIP object to patch (may be ``None``).
        lora_name: Filename of the LoRA inside the ``loras`` folder.
        strength:  Weight multiplier for both model and clip.

    Returns:
        Tuple of ``(patched_model, patched_clip)``.
    """
    lora_path = folder_paths.get_full_path("loras", lora_name)
    if lora_path is None:
        print(f"[Laura Control Center] WARNING: LoRA '{lora_name}' not found, skipping.")
        return model, clip

    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
    patched_model, patched_clip = comfy.sd.load_lora_for_models(
        model, clip, lora, strength, strength,
    )
    return patched_model, patched_clip


# =============================================================================
# MAIN NODE
# =============================================================================

class LauraControlCenter:
    """One-stop pipeline configuration node for Laura Image Studio.

    Replaces multiple scattered model-loading nodes with a single control
    surface.  Handles VRAM detection, model selection (image / upscale /
    video / face), up to three LoRA slots, and outputs MODEL / CLIP / VAE
    plus metadata for downstream nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        image_choices = get_image_model_choices()
        if not image_choices:
            image_choices = sorted(MODEL_REGISTRY.keys())

        upscale_choices = ["none"] + get_upscale_model_choices()
        video_choices = ["none"] + get_video_model_choices()
        face_choices = ["none", "inswapper_128"]

        lora_list = ["None"] + _lora_choices()

        return {
            "required": {
                "image_model": (image_choices, {
                    "default": image_choices[0] if image_choices else "z_image_turbo",
                }),
                "upscale_model": (upscale_choices, {"default": "none"}),
                "video_model": (video_choices, {"default": "none"}),
                "face_model": (face_choices, {"default": "none"}),
                "lora_1": (lora_list, {"default": "None"}),
                "lora_1_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                }),
                "lora_2": (lora_list, {"default": "None"}),
                "lora_2_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                }),
                "lora_3": (lora_list, {"default": "None"}),
                "lora_3_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                }),
                "auto_detect_vram": ("BOOLEAN", {"default": True}),
                "manual_vram_tier": (sorted(VRAM_TIERS.keys()), {"default": "high"}),
                "auto_quantization": ("BOOLEAN", {"default": True}),
                "torch_compile": ("BOOLEAN", {"default": False}),
                "default_width": ("INT", {
                    "default": 1024, "min": 512, "max": 4096, "step": 64,
                }),
                "default_height": ("INT", {
                    "default": 1024, "min": 512, "max": 4096, "step": 64,
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = (
        "model",
        "clip",
        "vae",
        "detected_type",
        "default_width",
        "default_height",
        "status_report",
        "config_json",
    )
    FUNCTION = "configure_pipeline"
    CATEGORY = "Laura Studio/Control Center"
    DESCRIPTION = (
        "All-in-one pipeline configuration node.  Selects models across four "
        "categories (image, upscale, video, face), detects VRAM, applies up to "
        "three LoRAs, and outputs MODEL / CLIP / VAE plus metadata for the "
        "entire workflow."
    )

    # -----------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------

    def configure_pipeline(
        self,
        image_model,
        upscale_model,
        video_model,
        face_model,
        lora_1,
        lora_1_strength,
        lora_2,
        lora_2_strength,
        lora_3,
        lora_3_strength,
        auto_detect_vram,
        manual_vram_tier,
        auto_quantization,
        torch_compile,
        default_width,
        default_height,
    ):
        """Configure the full generation pipeline.

        Returns:
            8-tuple: (model, clip, vae, detected_type, default_width,
                      default_height, status_report, config_json)
        """
        status_lines = []

        # ----------------------------------------------------------
        # 1.  VRAM detection
        # ----------------------------------------------------------
        if auto_detect_vram:
            vram_gb, tier_key = _detect_vram()
            status_lines.append(f"VRAM: {vram_gb} GB (auto-detected, tier: {tier_key})")
        else:
            tier_key = manual_vram_tier
            tier_info = VRAM_TIERS.get(tier_key, {})
            vram_gb = tier_info.get("max_gb", 0)
            status_lines.append(f"VRAM: {vram_gb} GB (manual tier: {tier_key})")

        # ----------------------------------------------------------
        # 2.  Registry lookup for the selected image model
        # ----------------------------------------------------------
        model_info = MODEL_REGISTRY.get(image_model)
        if model_info is None:
            raise ValueError(
                f"Image model '{image_model}' not found in the Laura Studio "
                f"Model Registry.  Available keys: "
                f"{', '.join(sorted(MODEL_REGISTRY.keys()))}"
            )

        detected_type = model_info.get("family", image_model)
        display_name = model_info.get("display_name", image_model)
        status_lines.append(f"Image model: {display_name} (family: {detected_type})")

        # ----------------------------------------------------------
        # 3.  Quantization selection
        # ----------------------------------------------------------
        if auto_quantization:
            quant = get_recommended_quantization(image_model, tier_key) or "bf16"
            status_lines.append(f"Quantization: {quant} (auto)")
        else:
            quant = "bf16"
            status_lines.append(f"Quantization: bf16 (manual)")

        # ----------------------------------------------------------
        # 4.  Load the primary image model
        # ----------------------------------------------------------
        try:
            model, clip, vae = _load_image_model(image_model, model_info)
            status_lines.append(f"Model loaded successfully: {display_name}")
        except Exception as exc:
            status_lines.append(f"ERROR loading model: {exc}")
            raise

        # ----------------------------------------------------------
        # 5.  Apply LoRAs
        # ----------------------------------------------------------
        lora_slots = [
            (lora_1, lora_1_strength),
            (lora_2, lora_2_strength),
            (lora_3, lora_3_strength),
        ]
        applied_loras = []
        for idx, (lora_name, lora_strength) in enumerate(lora_slots, start=1):
            if lora_name and lora_name != "None" and lora_strength > 0.0:
                try:
                    model, clip = _apply_lora(model, clip, lora_name, lora_strength)
                    applied_loras.append(f"LoRA {idx}: {lora_name} @ {lora_strength:.2f}")
                    status_lines.append(
                        f"Applied LoRA {idx}: {lora_name} (strength={lora_strength:.2f})"
                    )
                except Exception as exc:
                    status_lines.append(
                        f"WARNING: Failed to apply LoRA {idx} '{lora_name}': {exc}"
                    )

        if not applied_loras:
            status_lines.append("LoRAs: none applied")

        # ----------------------------------------------------------
        # 6.  Installed / missing file report
        # ----------------------------------------------------------
        installed = _scan_installed_models()
        required_files = get_all_required_files(image_model)
        installed_files = []
        missing_files = []
        for finfo in required_files:
            fname = finfo.get("filename", "")
            if fname in installed:
                installed_files.append(fname)
            else:
                missing_files.append(fname)

        if installed_files:
            status_lines.append(f"Installed files ({len(installed_files)}): "
                                f"{', '.join(installed_files)}")
        if missing_files:
            status_lines.append(f"MISSING files ({len(missing_files)}): "
                                f"{', '.join(missing_files)}")

        # Secondary model selections
        if upscale_model != "none":
            status_lines.append(f"Upscale model: {get_model_display_name(upscale_model)}")
        if video_model != "none":
            status_lines.append(f"Video model: {get_model_display_name(video_model)}")
        if face_model != "none":
            status_lines.append(f"Face model: {face_model}")

        if torch_compile:
            status_lines.append("torch.compile: enabled")

        status_report = "\n".join(status_lines)

        # ----------------------------------------------------------
        # 7.  Build config JSON for downstream nodes
        # ----------------------------------------------------------
        config = {
            "image_model": image_model,
            "image_model_display": display_name,
            "detected_type": detected_type,
            "upscale_model": upscale_model,
            "video_model": video_model,
            "face_model": face_model,
            "vram_gb": vram_gb,
            "vram_tier": tier_key,
            "quantization": quant,
            "auto_quantization": auto_quantization,
            "torch_compile": torch_compile,
            "default_width": default_width,
            "default_height": default_height,
            "loras": [
                {"name": name, "strength": strength}
                for name, strength in lora_slots
                if name and name != "None"
            ],
            "installed_files": installed_files,
            "missing_files": missing_files,
        }
        config_json = json.dumps(config, indent=2)

        # ----------------------------------------------------------
        # 8.  Return 8-tuple
        # ----------------------------------------------------------
        return (
            model,
            clip,
            vae,
            detected_type,
            default_width,
            default_height,
            status_report,
            config_json,
        )


# =============================================================================
# REGISTER NODE
# =============================================================================

NODE_CLASS_MAPPINGS["LauraControlCenter"] = LauraControlCenter
NODE_DISPLAY_NAME_MAPPINGS["LauraControlCenter"] = "Laura Control Center"
