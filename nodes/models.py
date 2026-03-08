"""
Laura Image Studio - Universal Model Support
Multi-model generation nodes supporting SDXL, Flux, Wan 2.2, SD 1.5, SD 3, and more
"""

import time

import torch
from PIL import Image
import numpy as np
import folder_paths
import os

from .model_registry import detect_model_key_from_filename, MODEL_REGISTRY, get_all_required_files

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Initialize colorama once at module level
try:
    from colorama import Fore, Style, init as _colorama_init

    _colorama_init(autoreset=True)
except ImportError:
    # Fallback if colorama is not installed
    class Fore:
        CYAN = RED = YELLOW = ""

    class Style:
        RESET_ALL = ""


# ============== ADVANCED LOGGER ==============
class LauraLogger:
    @staticmethod
    def info(msg):
        print(f"{Fore.CYAN}[snrtherock/Laura Studio] {Style.RESET_ALL}{msg}")

    @staticmethod
    def warn(msg):
        print(f"{Fore.YELLOW}[snrtherock/Laura Studio] WARNING: {Style.RESET_ALL}{msg}")

    @staticmethod
    def error(msg):
        print(f"{Fore.RED}[snrtherock/Laura Studio] ERROR: {Style.RESET_ALL}{msg}")


# ============== MODEL HEALTH CHECK ==============
class ModelHealthCheck:
    """Scan and verify that SOTA 2026 models are correctly installed"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "check_now": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("status_report", "all_present", "vram_advice")
    FUNCTION = "check_health"
    CATEGORY = "Laura Studio/Utility"
    DESCRIPTION = "Check if required SOTA models (Flux, Wan, Cosmos) are present and provide VRAM-specific advice"

    def check_health(self, check_now):
        if not check_now:
            return ("Check skipped", False, "")

        # Build required files list from the model registry
        report = ["--- LAURA STUDIO MODEL HEALTH ---"]
        all_ok = True

        # Cache folder listings to avoid repeated calls
        _folder_cache = {}

        def _get_folder_files(folder_name):
            if folder_name not in _folder_cache:
                try:
                    _folder_cache[folder_name] = [
                        f.lower() for f in folder_paths.get_filename_list(folder_name)
                    ]
                except Exception:
                    _folder_cache[folder_name] = []
            return _folder_cache[folder_name]

        for model_key, entry in MODEL_REGISTRY.items():
            files = entry.get("files", {})
            if not files:
                continue

            display_name = entry.get("display_name", model_key)
            model_files = get_all_required_files(model_key)
            if not model_files:
                continue

            # Check the primary file (first file entry) for this model
            primary = model_files[0]
            folder = primary.get("folder", "checkpoints")
            filename = primary.get("filename", "")
            if not filename:
                continue

            present_files = _get_folder_files(folder)
            found = any(filename.lower() in p for p in present_files)
            if found:
                report.append(f"\u2705 {display_name}: Found")
            else:
                report.append(f"\u274c {display_name}: MISSING")
                all_ok = False

        # VRAM Advice
        advice = "--- VRAM ADVICE ---"
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram < 8:
                advice = f"GPU: {vram:.1f}GB (ULTRA LOW)\n- Use SD1.5 or Pixart\n- Avoid Flux/Wan\n- Use FP8 for everything"
            elif vram < 12:
                advice = f"GPU: {vram:.1f}GB (LOW)\n- Flux Schnell (FP8) Recommended\n- Use 'Extreme' VRAM optimization"
            elif vram < 16:
                advice = f"GPU: {vram:.1f}GB (MEDIUM)\n- Flux Dev/Schnell (FP8/BF16) OK\n- Wan 2.2 1.3B OK"
            elif vram < 24:
                advice = f"GPU: {vram:.1f}GB (HIGH)\n- All models supported\n- Wan 2.2 14B (FP8) OK"
            else:
                advice = f"GPU: {vram:.1f}GB (EXTREME)\n- Full resolution Wan 2.2 / Hunyuan OK\n- Batch size 4+ OK"
        else:
            advice = "CPU MODE: Very slow generation expected."

        return ("\n".join(report), all_ok, advice)


# ============== LAURA STAGE PREVIEW ==============
class LauraStagePreview:
    """Lightweight preview/thumbnail node for any pipeline stage.

    Drop this node between any two stages (e.g. after generation, after
    upscaling, after face-swap) to get a quick visual preview, resolution
    info, and optional on-disk debug thumbnail.  It passes the image
    through unchanged so it never breaks the pipeline.

    Output ``stage_info`` is a human-readable string with dimensions,
    pixel count, dtype, and the stage label.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "stage_label": (
                    "STRING",
                    {"default": "Stage", "multiline": False},
                ),
            },
            "optional": {
                "save_thumbnail": ("BOOLEAN", {"default": False}),
                "thumbnail_max_size": (
                    "INT",
                    {"default": 256, "min": 64, "max": 1024, "step": 32},
                ),
                "output_dir": (
                    "STRING",
                    {"default": "laura_previews", "multiline": False},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "stage_info")
    FUNCTION = "preview_stage"
    CATEGORY = "Laura Studio/Utility"
    DESCRIPTION = "Preview image at any pipeline stage — logs metadata and optionally saves a debug thumbnail"
    OUTPUT_NODE = True

    def preview_stage(
        self,
        image,
        stage_label,
        save_thumbnail=False,
        thumbnail_max_size=256,
        output_dir="laura_previews",
    ):
        # Gather image metadata
        if image.ndim == 4:
            batch, height, width, channels = image.shape
        elif image.ndim == 3:
            height, width, channels = image.shape
            batch = 1
        else:
            batch, height, width, channels = 1, 0, 0, 0

        megapixels = (width * height) / 1_000_000
        dtype_str = str(image.dtype)

        info_lines = [
            f"--- {stage_label} ---",
            f"Resolution : {width} x {height}  ({megapixels:.2f} MP)",
            f"Batch/Frames: {batch}",
            f"Channels    : {channels}",
            f"Dtype       : {dtype_str}",
            f"Tensor shape: {list(image.shape)}",
        ]
        stage_info = "\n".join(info_lines)

        LauraLogger.info(
            f"[StagePreview] {stage_label}: {width}x{height} "
            f"({megapixels:.2f}MP), batch={batch}, dtype={dtype_str}"
        )

        # Optionally save a debug thumbnail to disk
        if save_thumbnail and width > 0 and height > 0:
            try:
                # Resolve output directory relative to ComfyUI output
                out_base = folder_paths.get_output_directory()
                thumb_dir = os.path.join(out_base, output_dir)
                os.makedirs(thumb_dir, exist_ok=True)

                # Take the first image in the batch
                first_img = image[0] if image.ndim == 4 else image
                # Convert to numpy uint8
                img_np = (
                    first_img.cpu().numpy()
                    if hasattr(first_img, "cpu")
                    else np.array(first_img)
                )
                img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

                pil_img = Image.fromarray(img_np)

                # Resize to thumbnail
                pil_img.thumbnail(
                    (thumbnail_max_size, thumbnail_max_size), Image.LANCZOS
                )

                # Safe filename
                safe_label = "".join(
                    c if c.isalnum() or c in "-_" else "_" for c in stage_label
                )
                filename = f"{safe_label}_{int(time.time())}.png"
                filepath = os.path.join(thumb_dir, filename)
                pil_img.save(filepath)

                LauraLogger.info(
                    f"[StagePreview] Saved thumbnail: {filepath} "
                    f"({pil_img.width}x{pil_img.height})"
                )
            except Exception as e:
                LauraLogger.warn(f"[StagePreview] Failed to save thumbnail: {e}")

        # Pass image through unchanged
        return (image, stage_info)


# ============== MODEL TYPE DETECTOR ==============
class ModelTypeDetector:
    """Detect model type from filename or metadata"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("checkpoints"),),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("model_type", "width", "height")
    FUNCTION = "detect_type"
    CATEGORY = "Laura Studio/Models"
    DESCRIPTION = "Auto-detect model architecture and set optimal base resolution"

    def detect_type(self, model_name):
        LauraLogger.info(f"Detecting architecture for: {model_name}")

        # --- Registry-based detection (preferred) ---
        registry_key = detect_model_key_from_filename(model_name)
        if registry_key:
            type_map = {
                "z_image_turbo": "zimage_turbo",
                "z_image_base": "zimage",
                "z_image_edit": "zimage_edit",
                "flux2_dev": "flux2",
                "flux1_dev": "flux",
                "flux1_schnell": "flux_schnell",
                "qwen_image_2512": "qwen",
                "glm_image": "glm_image",
                "firered_edit_1_1": "firered",
                "helios_distilled": "helios",
                "helios_base": "helios",
                "ltx_2_3": "ltx23",
                "sd35_medium": "sd35",
            }
            model_type = type_map.get(registry_key)
            if model_type:
                model_info = MODEL_REGISTRY.get(registry_key, {})
                default_res = model_info.get("inference", {}).get("default_resolution", {"width": 1024, "height": 1024})
                if default_res is None:
                    default_res = {"width": 1024, "height": 1024}
                width = default_res.get("width", 1024) if isinstance(default_res, dict) else 1024
                height = default_res.get("height", 1024) if isinstance(default_res, dict) else 1024
                LauraLogger.info(f"Registry match: {registry_key} -> {model_type} ({width}x{height})")
                return (model_type, width, height)

        # --- Fallback: pattern-based detection ---
        model_name = model_name.lower()

        # 2025/2026 SOTA Detection
        if "wan2" in model_name:
            return ("wan22", 1024, 1024)
        if "z-image" in model_name or "zimage" in model_name:
            if "turbo" in model_name:
                return ("zimage_turbo", 1024, 1024)
            return ("zimage", 1024, 1024)
        if "qwen" in model_name:
            return ("qwen", 1024, 1024)
        if "cosmos" in model_name:
            return ("cosmos", 1024, 1024)

        # Standard Detection
        if "flux" in model_name:
            if "schnell" in model_name:
                return ("flux_schnell", 1024, 1024)
            return ("flux", 1024, 1024)
        if "sdxl" in model_name:
            return ("sdxl", 1024, 1024)
        if "sd3.5" in model_name or "sd35" in model_name:
            return ("sd35", 1024, 1024)
        if "sd3" in model_name:
            return ("sd3", 1024, 1024)
        if "sd1.5" in model_name or "sd15" in model_name:
            return ("sd15", 512, 512)

        return ("unknown", 1024, 1024)


# ============== CHARACTER LORA LOADER ==============
class CharacterLoRALoader:
    """Specialized loader for trained character LoRAs (Influencer models)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05},
                ),
                "trigger_word": ("STRING", {"default": "character_name"}),
                "append_trigger": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "updated_prompt")
    FUNCTION = "load_character_lora"
    CATEGORY = "Laura Studio/Models"
    DESCRIPTION = "Load specialized character LoRAs (Laura, Zoriana, etc.) and auto-inject trigger words"

    def load_character_lora(
        self, model, clip, lora_name, strength, trigger_word, append_trigger
    ):
        from nodes import LoraLoader

        LauraLogger.info(f"Applying Character LoRA: {lora_name}")
        # Load LoRA using standard loader
        new_model, new_clip = LoraLoader().load_lora(
            model, clip, lora_name, strength, strength
        )

        # Return the trigger word to be appended to the prompt
        prompt_addition = f", {trigger_word}" if append_trigger and trigger_word else ""

        LauraLogger.info(
            f"Character LoRA '{lora_name}' loaded with trigger '{trigger_word}'."
        )
        return (new_model, new_clip, prompt_addition)


# ============== UNIVERSAL MODEL LOADER ==============
class UniversalModelLoader:
    """Universal model loader with model type detection"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("checkpoints"),),
                "model_type": (
                    [
                        "auto",
                        "sdxl",
                        "flux",
                        "flux_schnell",
                        "flux2",
                        "flux2_schnell",
                        "sd15",
                        "sd3",
                        "sd35",
                        "sd35_medium",
                        "wan21",
                        "wan22",
                        "zimage",
                        "zimage_turbo",
                        "zimage_edit",
                        "qwen",
                        "playground",
                        "pixart",
                        "aura",
                        "kolors",
                        "glm_image",
                        "firered",
                        "helios",
                        "ltx23",
                    ],
                ),
            },
            "optional": {
                "default_width": ("INT", {"default": 1024, "min": 256, "max": 2048}),
                "default_height": ("INT", {"default": 1024, "min": 256, "max": 2048}),
                "torch_compile": (
                    "BOOLEAN",
                    {"default": False},
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "detected_type")
    FUNCTION = "load_model"
    CATEGORY = "Laura Studio/Models"
    DESCRIPTION = (
        "Load any model with auto-detection of architecture and optimal settings. "
        "Enable torch_compile for ~15-40% faster inference (first run is slow due to compilation)."
    )

    def load_model(
        self,
        model_name,
        model_type,
        default_width=1024,
        default_height=1024,
        torch_compile=False,
    ):
        # Auto-detect if needed
        if model_type == "auto":
            model_type, width, height = ModelTypeDetector().detect_type(model_name)
        else:
            width, height = default_width, default_height

        LauraLogger.info(f"Universal Loading: {model_name} as {model_type}")

        # Set default resolutions per model type (only if user didn't override)
        resolution_map = {
            "sdxl": (1024, 1024),
            "flux": (512, 512),
            "flux_schnell": (512, 512),
            "flux2": (1024, 1024),
            "flux2_schnell": (1024, 1024),
            "sd15": (512, 512),
            "sd3": (1024, 1024),
            "sd35": (1024, 1024),
            "sd35_medium": (1024, 1024),
            "wan21": (512, 512),
            "wan22": (512, 512),
            "zimage": (1024, 1024),
            "zimage_turbo": (1024, 1024),
            "zimage_edit": (1024, 1024),
            "qwen": (2512, 2512),
            "glm_image": (1024, 1024),
            "firered": (1024, 1024),
            "helios": (1024, 1024),
            "ltx23": (768, 768),
            "playground": (1024, 1024),
            "pixart": (1024, 1024),
            "aura": (512, 512),
            "kolors": (1024, 1024),
        }

        # Use resolution_map as a refinement when user used default dimensions
        if default_width == 1024 and default_height == 1024 and model_type != "auto":
            width, height = resolution_map.get(model_type, (width, height))

        # Load model using ComfyUI's built-in
        from nodes import CheckpointLoaderSimple

        result = CheckpointLoaderSimple().load_checkpoint(model_name)
        model, clip, vae = result[0], result[1], result[2]

        # torch.compile() optimization (v0.9) — reduces overhead by ~15-40%
        if torch_compile:
            model = self._apply_torch_compile(model, model_name)

        return (model, clip, vae, model_type)

    @staticmethod
    def _apply_torch_compile(model, model_name="model"):
        """Apply torch.compile with reduce-overhead mode to the inner diffusion model."""
        try:
            if (
                hasattr(torch, "compile")
                and hasattr(model, "model")
                and hasattr(model.model, "diffusion_model")
            ):
                model.model.diffusion_model = torch.compile(
                    model.model.diffusion_model,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
                LauraLogger.info(
                    f"torch.compile(mode='reduce-overhead') applied to {model_name}"
                )
            else:
                LauraLogger.warn(
                    "torch.compile requested but model structure not compatible or torch.compile unavailable"
                )
        except Exception as e:
            LauraLogger.warn(f"torch.compile failed (will run without): {e}")
        return model


# ============== ADVANCED MODEL LOADER ==============
class AdvancedModelLoader:
    """Advanced model loader with auto-detection and quantization support"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("checkpoints"),),
                "model_type": (
                    [
                        "auto",
                        "sdxl",
                        "flux",
                        "flux_schnell",
                        "flux2",
                        "flux2_schnell",
                        "sd15",
                        "sd3",
                        "sd35",
                        "sd35_medium",
                        "wan21",
                        "wan22",
                        "zimage",
                        "zimage_turbo",
                        "zimage_edit",
                        "qwen",
                        "playground",
                        "pixart",
                        "aura",
                        "kolors",
                        "glm_image",
                        "firered",
                        "helios",
                        "ltx23",
                    ],
                ),
                "attention_mode": (
                    ["auto", "xformers", "sdpa", "sub_quad", "sliced"],
                    {"default": "auto"},
                ),
            },
            "optional": {
                "quant_config": ("QUANT_CONFIG",),
                "default_width": ("INT", {"default": 1024, "min": 256, "max": 2048}),
                "default_height": ("INT", {"default": 1024, "min": 256, "max": 2048}),
                "torch_compile": (
                    "BOOLEAN",
                    {"default": False},
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "detected_type")
    FUNCTION = "load_model_advanced"
    CATEGORY = "Laura Studio/Models"
    DESCRIPTION = "Load model with optional precision/VRAM optimizations and torch.compile acceleration"

    def load_model_advanced(
        self,
        model_name,
        model_type,
        attention_mode="auto",
        quant_config=None,
        default_width=1024,
        default_height=1024,
        torch_compile=False,
    ):
        # Auto-detect if needed
        if model_type == "auto":
            model_type, width, height = ModelTypeDetector().detect_type(model_name)
        else:
            width, height = default_width, default_height

        # Auto VRAM Tiers logic Integration
        # If quant_config is not provided, we try to auto-generate one based on VRAM
        if quant_config is None:
            from .quantization import (
                VRAMAutoDetector,
                QuantizationSelector,
                ModelOffloadConfig,
            )

            vram_tier, vram_gb = VRAMAutoDetector().detect_vram("auto")
            weight_dtype = QuantizationSelector().select_quantization(
                vram_tier, model_type
            )[0]
            enable_offload, seq_offload = ModelOffloadConfig().get_offload_config(
                vram_tier
            )

            quant_config = {
                "weight_dtype": weight_dtype,
                "enable_cpu_offload": enable_offload,
                "sequential_offload": seq_offload,
            }
            LauraLogger.info(
                f"Auto-configured optimization for {vram_tier} tier ({vram_gb}GB): {weight_dtype}"
            )

        # Map attention mode for low VRAM
        if attention_mode == "auto":
            from .quantization import VRAMAutoDetector

            vram_tier = VRAMAutoDetector().detect_vram("auto")[0]
            if vram_tier in ["ultra_low", "low"]:
                attention_mode = "sub_quad"
            elif vram_tier in ["medium", "high"]:
                attention_mode = "xformers" if torch.cuda.is_available() else "sdpa"
            else:
                attention_mode = "sdpa"

        weight_dtype = quant_config.get("weight_dtype", "fp16")
        LauraLogger.info(
            f"Advanced Loading {model_name} | Type: {model_type} | Weight: {weight_dtype} | Attn: {attention_mode}"
        )

        from nodes import CheckpointLoaderSimple

        result = CheckpointLoaderSimple().load_checkpoint(model_name)
        model, clip, vae = result[0], result[1], result[2]

        # Apply weight dtype optimization via ComfyUI model patcher
        dtype_map = {
            "fp8_e4m3fn": torch.float8_e4m3fn
            if hasattr(torch, "float8_e4m3fn")
            else torch.float16,
            "fp8_e5m2": torch.float8_e5m2
            if hasattr(torch, "float8_e5m2")
            else torch.float16,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
            "int8": torch.int8,
        }
        target_dtype = dtype_map.get(weight_dtype)
        if target_dtype and target_dtype != torch.float32:
            try:
                # ComfyUI ModelPatcher exposes model_dtype() and can cast weights
                if hasattr(model, "model") and hasattr(model.model, "to"):
                    model.model.to(target_dtype)
                    LauraLogger.info(f"Applied weight dtype: {weight_dtype}")
            except Exception as e:
                LauraLogger.warn(f"Could not apply weight dtype {weight_dtype}: {e}")

        # Apply attention mode via ComfyUI's model options
        if attention_mode != "auto":
            try:
                if hasattr(model, "model_options"):
                    model.model_options["attention_mode"] = attention_mode
                    LauraLogger.info(f"Applied attention mode: {attention_mode}")
            except Exception as e:
                LauraLogger.warn(
                    f"Could not apply attention mode {attention_mode}: {e}"
                )

        # Apply CPU offload if configured
        if quant_config.get("enable_cpu_offload", False):
            try:
                import comfy.model_management

                comfy.model_management.soft_empty_cache()
                # Move model to CPU initially; ComfyUI will load to GPU on demand
                if hasattr(model, "model") and hasattr(model.model, "to"):
                    model.model.to("cpu")
                LauraLogger.info(
                    "CPU offload enabled — model stored on CPU, loaded to GPU on demand"
                )
            except Exception as e:
                LauraLogger.warn(f"Could not configure CPU offload: {e}")

        # torch.compile() optimization (v0.9)
        if torch_compile:
            model = UniversalModelLoader._apply_torch_compile(model, model_name)

        return (model, clip, vae, model_type)


# ============== LORA MANAGER ==============
class LoraManager:
    """Manage LoRAs for custom character (Laura/Zoriana)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_path": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0}),
                "strength_clip": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0}),
            },
            "optional": {
                "lora_name": ("STRING", {"default": ""}),
                "enable": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "apply_lora"
    CATEGORY = "Laura Studio/Models"
    DESCRIPTION = "Individual LoRA manager for character or style refinement"

    def apply_lora(
        self,
        model,
        clip,
        lora_path,
        strength_model,
        strength_clip,
        lora_name="",
        enable=True,
    ):
        if not enable or not lora_path or lora_path == "None":
            return (model, clip)

        LauraLogger.info(f"Applying LoRA: {lora_path}")

        if lora_name:
            lora_file = lora_name
        else:
            lora_file = lora_path

        try:
            from nodes import LoraLoader

            result = LoraLoader().load_lora(
                model, clip, lora_file, strength_model, strength_clip
            )
            model, clip = result[0], result[1]
        except Exception as e:
            LauraLogger.warn(
                f"LoRA '{lora_name or lora_path}' failed to load: {e}. Returning model without LoRA applied."
            )

        return (model, clip)


# ============== MULTI LORA STACK ==============
class MultiLoraStack:
    """Apply multiple LoRAs (character, style, etc.)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_1": (folder_paths.get_filename_list("loras"),),
                "lora_1_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0}),
            },
            "optional": {
                "lora_2": (folder_paths.get_filename_list("loras"),),
                "lora_2_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0}),
                "lora_3": (folder_paths.get_filename_list("loras"),),
                "lora_3_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "apply_loras"
    CATEGORY = "Laura Studio/Models"
    DESCRIPTION = "Apply a stack of up to 3 LoRAs (e.g., Character + Style + Lighting)"

    def apply_loras(
        self,
        model,
        clip,
        lora_1,
        lora_1_strength,
        lora_2=None,
        lora_2_strength=0.5,
        lora_3=None,
        lora_3_strength=0.3,
    ):
        LauraLogger.info("Applying Multi-LoRA Stack")
        from nodes import LoraLoader

        loader = LoraLoader()

        # Apply LoRA 1 (typically character - Zoriana/Laura)
        if lora_1 and lora_1 != "None":
            result = loader.load_lora(
                model, clip, lora_1, lora_1_strength, lora_1_strength
            )
            model, clip = result[0], result[1]

        # Apply LoRA 2 (typically style)
        if lora_2 and lora_2 != "None":
            result = loader.load_lora(
                model, clip, lora_2, lora_2_strength, lora_2_strength
            )
            model, clip = result[0], result[1]

        # Apply LoRA 3 (typically additional)
        if lora_3 and lora_3 != "None":
            result = loader.load_lora(
                model, clip, lora_3, lora_3_strength, lora_3_strength
            )
            model, clip = result[0], result[1]

        return (model, clip)


# ============== UNIVERSAL GENERATOR ==============
class UniversalGenerator:
    """Universal image generator for all model types"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "model_type": (
                    [
                        "auto",
                        "sdxl",
                        "flux",
                        "flux_schnell",
                        "flux2",
                        "flux2_schnell",
                        "sd15",
                        "sd3",
                        "sd35",
                        "sd35_medium",
                        "wan21",
                        "wan22",
                        "zimage",
                        "zimage_turbo",
                        "zimage_edit",
                        "qwen",
                        "playground",
                        "pixart",
                        "aura",
                        "kolors",
                        "glm_image",
                        "firered",
                        "helios",
                        "ltx23",
                    ],
                ),
                "positive_prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "deformed, blurry, bad anatomy, low quality",
                    },
                ),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
                "height": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 2048, "step": 8},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0}),
                "sampler_name": (
                    [
                        "euler",
                        "euler_ancestral",
                        "dpm_2",
                        "dpm_2_ancestral",
                        "dpmpp_2m",
                        "dpmpp_sde",
                        "uni_pc",
                        "uni_pc_bh2",
                        "ddpm",
                        "kdpm_2",
                        "kdpm_2_a",
                    ],
                ),
                "scheduler": (
                    ["normal", "karras", "exponential", "simple", "ddim_uniform"],
                ),
            },
            "optional": {
                "image_to_image": ("IMAGE",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16}),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "generate"
    CATEGORY = "Laura Studio/Generation"
    DESCRIPTION = (
        "Universal Generator with built-in logic for Flux, Wan, and SDXL architectures"
    )

    def generate(
        self,
        model,
        clip,
        vae,
        model_type,
        positive_prompt,
        negative_prompt,
        width,
        height,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        image_to_image=None,
        denoise=1.0,
        batch_size=1,
    ):
        LauraLogger.info(f"Generating image ({width}x{height}) - Type: {model_type}")

        # 0. VRAM-SAFE RESOLUTION SCALING (SOTA 2026 Optimization)
        from .quantization import VRAMAutoDetector, ResolutionScaler

        vram_tier = VRAMAutoDetector().detect_vram("auto")[0]
        width, height = ResolutionScaler().scale_resolution(vram_tier, width, height)
        LauraLogger.info(f"VRAM-Safe Resolution: {width}x{height} (Tier: {vram_tier})")

        # 1. AUTO-CHARACTER IDENTITY (Influencer Logic)
        # If character keywords aren't in prompt and the model type suggests a Laura persona,
        # inject identity triggers. Users can disable this by including "laura" in their prompt
        # or by using a non-Laura model type.
        laura_model_types = {"flux", "sdxl", "wan22"}
        character_triggers = {
            "flux": "laura influencer, professional digital style",
            "sdxl": "laura, highly detailed face, professional photography",
            "wan22": "laura, realistic skin, cinematic motion",
        }
        if model_type in laura_model_types and "laura" not in positive_prompt.lower():
            trigger = character_triggers.get(model_type, "")
            if trigger:
                positive_prompt = f"{trigger}, {positive_prompt}"
                LauraLogger.info(f"Auto-Injected Character Identity: {trigger}")

        # Adjust settings based on model type
        model_defaults = {
            "flux": {"steps": 8, "cfg": 2.0, "max_res": None},
            "flux_schnell": {"steps": 4, "cfg": 2.0, "max_res": None},
            "flux2": {"steps": 8, "cfg": 2.0, "max_res": None},
            "flux2_schnell": {"steps": 4, "cfg": 2.0, "max_res": None},
            "zimage_turbo": {"steps": 8, "cfg": 2.0, "max_res": None},
            "sd15": {"steps": 50, "cfg": 7.0, "max_res": 768},
            "wan21": {"steps": 25, "cfg": 5.0, "max_res": None},
            "wan22": {"steps": 25, "cfg": 5.0, "max_res": None},
        }
        defaults = model_defaults.get(model_type, {})
        if defaults.get("cfg") is not None and cfg > defaults["cfg"] + 1:
            LauraLogger.warn(
                f"CFG {cfg} exceeds recommended max for {model_type}, clamping to {defaults['cfg']}"
            )
            cfg = defaults["cfg"]
        if defaults.get("steps") and model_type in [
            "flux",
            "flux_schnell",
            "flux2",
            "flux2_schnell",
            "zimage_turbo",
        ]:
            steps = min(steps, defaults["steps"])
        if defaults.get("max_res"):
            max_r = defaults["max_res"]
            if width > max_r:
                width = max_r
            if height > max_r:
                height = max_r

        from nodes import (
            CLIPTextEncode,
            VAEEncode,
            EmptyLatentImage,
            KSampler,
            VAEDecode,
        )

        # Encode prompts
        positive = CLIPTextEncode().encode(clip, positive_prompt)[0]
        negative = CLIPTextEncode().encode(clip, negative_prompt)[0]

        # Create or encode latent
        if image_to_image is not None:
            # Image to image
            encoded = VAEEncode().encode(vae, image_to_image)[0]
            latent = {"samples": encoded["samples"]}
        else:
            # Text to image
            latent = EmptyLatentImage().generate(width, height, batch_size)[0]

        # Sample
        sampled = KSampler().sample(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent,
            denoise=denoise,
        )[0]

        # Decode
        decoded = VAEDecode().decode(vae, sampled)[0]

        return (decoded, sampled)


# ============== IMAGE TO IMAGE UNIVERSAL ==============
class UniversalImg2Img:
    """Universal image-to-image for any model"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "model_type": (
                    [
                        "auto",
                        "sdxl",
                        "flux",
                        "flux_schnell",
                        "flux2",
                        "flux2_schnell",
                        "sd15",
                        "sd3",
                        "sd35",
                        "sd35_medium",
                        "wan21",
                        "wan22",
                        "zimage",
                        "zimage_turbo",
                        "zimage_edit",
                        "qwen",
                        "playground",
                        "pixart",
                        "aura",
                        "kolors",
                        "glm_image",
                        "firered",
                        "helios",
                        "ltx23",
                    ],
                ),
                "positive_prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": (
                    "STRING",
                    {"multiline": True, "default": "deformed, blurry, low quality"},
                ),
                "denoise": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "img2img"
    CATEGORY = "Laura Studio/Generation"
    DESCRIPTION = "Universal image-to-image conversion"

    def img2img(
        self,
        model,
        clip,
        vae,
        image,
        model_type,
        positive_prompt,
        negative_prompt,
        denoise,
        seed,
        steps,
        cfg,
    ):
        result = UniversalGenerator().generate(
            model,
            clip,
            vae,
            model_type,
            positive_prompt,
            negative_prompt,
            image.shape[2],
            image.shape[1],
            seed,
            steps,
            cfg,
            "euler",
            "normal",
            image_to_image=image,
            denoise=denoise,
        )
        return (result[0],)


# ============== INPAINTING UNIVERSAL ==============
class UniversalInpainter:
    """Universal inpainting for any model"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "positive_prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": (
                    "STRING",
                    {"multiline": True, "default": "deformed, blurry"},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 20.0}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "inpaint"
    CATEGORY = "Laura Studio/Generation"
    DESCRIPTION = "Universal inpainting"

    def inpaint(
        self,
        model,
        clip,
        vae,
        image,
        mask,
        positive_prompt,
        negative_prompt,
        seed,
        steps,
        cfg,
        denoise,
    ):
        from nodes import CLIPTextEncode, KSampler, VAEDecode

        # Use VAEEncodeForInpaint for proper mask-aware encoding
        try:
            from nodes import VAEEncodeForInpaint

            # VAEEncodeForInpaint expects (pixels, vae, mask, grow_mask_by)
            # mask should be [B, H, W] with 1.0 = inpaint region
            inpaint_mask = mask
            if inpaint_mask.dim() == 4:
                inpaint_mask = inpaint_mask.squeeze(1)
            if inpaint_mask.dim() == 2:
                inpaint_mask = inpaint_mask.unsqueeze(0)
            encoded = VAEEncodeForInpaint().encode(vae, image, inpaint_mask, 6)[0]
            use_inpaint_encoder = True
        except ImportError:
            # Fallback to standard VAEEncode if VAEEncodeForInpaint unavailable
            from nodes import VAEEncode

            encoded = VAEEncode().encode(vae, image)[0]
            use_inpaint_encoder = False

        # Build latent-space mask for noise_mask (only needed if fallback encoder)
        import torch.nn.functional as F

        if not use_inpaint_encoder:
            # Process mask to [B, 1, H, W] for latent-space interpolation
            mask_for_latent = mask
            if mask_for_latent.dim() == 2:
                mask_for_latent = mask_for_latent.unsqueeze(0).unsqueeze(0)
            elif mask_for_latent.dim() == 3:
                mask_for_latent = mask_for_latent.unsqueeze(1)
            elif mask_for_latent.dim() == 4 and mask_for_latent.shape[1] > 1:
                mask_for_latent = mask_for_latent[:, 0:1, :, :]

            latent_h = encoded["samples"].shape[2]
            latent_w = encoded["samples"].shape[3]
            mask_latent = F.interpolate(
                mask_for_latent,
                size=(latent_h, latent_w),
                mode="bilinear",
                align_corners=False,
            )
            mask_latent = (mask_latent > 0.5).float()
        else:
            # VAEEncodeForInpaint already handles mask internally via noise_mask
            mask_latent = None

        # Encode prompts
        positive = CLIPTextEncode().encode(clip, positive_prompt)[0]
        negative = CLIPTextEncode().encode(clip, negative_prompt)[0]

        # Create latent
        latent = {"samples": encoded["samples"]}
        if mask_latent is not None:
            latent["noise_mask"] = mask_latent

        # Sample
        sampled = KSampler().sample(
            model,
            seed,
            steps,
            cfg,
            "dpmpp_2m",
            "karras",
            positive,
            negative,
            latent,
            denoise=denoise,
        )[0]

        # Decode
        result = VAEDecode().decode(vae, sampled)[0]

        return (result,)


# ============== CONTROL NET ==============
class ControlNetLoader:
    """Load ControlNet models"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_net_name": (folder_paths.get_filename_list("controlnet"),),
            }
        }

    RETURN_TYPES = ("CONTROL_NET",)
    RETURN_NAMES = ("control_net",)
    FUNCTION = "load_controlnet"
    CATEGORY = "Laura Studio/Models"
    DESCRIPTION = "Load ControlNet model"

    def load_controlnet(self, control_net_name):
        from nodes import ControlNetLoader as CNLoader

        result = CNLoader().load_controlnet(control_net_name)
        return result


# ============== CONTROL NET APPLY ==============
class ApplyControlNet:
    """Apply ControlNet to generation"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "control_net": ("CONTROL_NET",),
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply_controlnet"
    CATEGORY = "Laura Studio/Models"
    DESCRIPTION = "Apply ControlNet to positive conditioning"

    def apply_controlnet(self, conditioning, control_net, image, strength):
        try:
            from comfy_extras.nodes_controlnet import ControlNetApplyAdvanced

            # ControlNetApplyAdvanced requires both positive and negative CONDITIONING.
            # Since we only have positive, create a minimal empty negative conditioning.
            # CONDITIONING type is a list of [tensor, dict] pairs.
            empty_negative = [[torch.zeros((1, 1, 768), dtype=torch.float32), {}]]
            result = ControlNetApplyAdvanced().apply_controlnet(
                conditioning, empty_negative, control_net, image, strength, 0.0, 1.0
            )
            # Note: result[1] (modified negative) is intentionally discarded — only positive conditioning is returned
            return (result[0],)
        except Exception as e:
            LauraLogger.warn(
                f"ControlNet application failed: {e}. Returning conditioning unchanged."
            )
            return (conditioning,)


# Register all nodes
NODE_CLASS_MAPPINGS.update(
    {
        "ModelHealthCheck": ModelHealthCheck,
        "LauraStagePreview": LauraStagePreview,
        "ModelTypeDetector": ModelTypeDetector,
        "UniversalModelLoader": UniversalModelLoader,
        "AdvancedModelLoader": AdvancedModelLoader,
        "CharacterLoRALoader": CharacterLoRALoader,
        "LoraManager": LoraManager,
        "MultiLoraStack": MultiLoraStack,
        "UniversalGenerator": UniversalGenerator,
        "UniversalImg2Img": UniversalImg2Img,
        "UniversalInpainter": UniversalInpainter,
        "ControlNetLoader": ControlNetLoader,
        "ApplyControlNet": ApplyControlNet,
    }
)

NODE_DISPLAY_NAME_MAPPINGS.update(
    {
        "ModelHealthCheck": "Model Health Check",
        "LauraStagePreview": "Laura Stage Preview",
        "ModelTypeDetector": "Model Type Detector",
        "UniversalModelLoader": "Universal Model Loader",
        "AdvancedModelLoader": "Advanced Model Loader",
        "CharacterLoRALoader": "Character LoRA Loader",
        "LoraManager": "LoRA Manager (Character)",
        "MultiLoraStack": "Multi-LoRA Stack",
        "UniversalGenerator": "Universal Image Generator",
        "UniversalImg2Img": "Universal Image to Image",
        "UniversalInpainter": "Universal Inpainter",
        "ControlNetLoader": "ControlNet Loader",
        "ApplyControlNet": "Apply ControlNet",
    }
)
