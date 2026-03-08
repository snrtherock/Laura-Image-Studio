"""
Laura Image Studio - VRAM Optimization & Quantization Nodes
Automatically detect GPU capabilities and adjust model precision and resolution.
v0.9: Added FP8 Transformer Engine with Ada Lovelace+ auto-detection.
"""

import torch

from .model_registry import get_recommended_quantization

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _detect_fp8_capability():
    """Detect if GPU supports native FP8 (Ada Lovelace / SM 8.9+ or Hopper SM 9.0+).
    Returns (supports_fp8: bool, gpu_name: str, compute_capability: tuple|None).
    """
    if not torch.cuda.is_available():
        return False, "CPU", None
    if not hasattr(torch, "float8_e4m3fn"):
        return False, "unknown", None
    try:
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        cc = (props.major, props.minor)
        # Ada Lovelace = SM 8.9, Hopper = SM 9.0, Blackwell = SM 10.0+
        supports_fp8 = cc >= (8, 9)
        return supports_fp8, gpu_name, cc
    except (RuntimeError, AssertionError):
        return False, "unknown", None


# ============== VRAM AUTO DETECTOR ==============
class VRAMAutoDetector:
    """Detect available VRAM and categorize into a tier"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "force_tier": (
                    [
                        "auto",
                        "ultra_low",
                        "low",
                        "medium",
                        "high",
                        "very_high",
                        "ultra",
                        "extreme",
                        "hpc",
                    ],
                    {"default": "auto"},
                ),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("vram_tier", "vram_gb")
    FUNCTION = "detect_vram"
    CATEGORY = "Laura Studio/Optimization"
    DESCRIPTION = "Auto-detect GPU memory tier"

    def detect_vram(self, force_tier="auto"):
        if force_tier != "auto":
            # Map tier back to approximate GB for downstream nodes
            tier_to_gb = {
                "ultra_low": 4.0,
                "low": 6.0,
                "medium": 8.0,
                "high": 12.0,
                "very_high": 16.0,
                "ultra": 24.0,
                "extreme": 40.0,
                "hpc": 80.0,
            }
            return (force_tier, tier_to_gb.get(force_tier, 8.0))

        vram_gb = 8.0  # default fallback
        if torch.cuda.is_available():
            # Get total memory of the current device
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except (RuntimeError, AssertionError):
                pass

        if vram_gb < 4.5:
            tier = "ultra_low"
        elif vram_gb < 6.5:
            tier = "low"
        elif vram_gb < 8.5:
            tier = "medium"
        elif vram_gb < 12.5:
            tier = "high"
        elif vram_gb < 16.5:
            tier = "very_high"
        elif vram_gb < 24.5:
            tier = "ultra"
        elif vram_gb < 48.5:
            tier = "extreme"
        else:
            tier = "hpc"

        return (tier, round(vram_gb, 2))


# ============== QUANTIZATION SELECTOR ==============
class QuantizationSelector:
    """Recommend quantization settings based on VRAM tier and model type"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vram_tier": ("STRING", {"forceInput": True}),
                "model_type": (
                    [
                        "auto",
                        "sdxl",
                        "flux",
                        "flux_schnell",
                        "flux2",
                        "sd15",
                        "sd3",
                        "wan21",
                        "wan22",
                        "cogvideox",
                        "zimage_turbo",
                        "zimage",
                        "qwen",
                        "glm_image",
                        "firered",
                        "helios",
                        "ltx23",
                    ],
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("weight_dtype",)
    FUNCTION = "select_quantization"
    CATEGORY = "Laura Studio/Optimization"
    DESCRIPTION = "Select optimal weight precision"

    def select_quantization(self, vram_tier, model_type):
        # Registry-based lookup (preferred for newer models)
        type_to_key = {
            "zimage_turbo": "z_image_turbo",
            "zimage": "z_image_base",
            "flux": "flux1_dev",
            "flux_schnell": "flux1_schnell",
            "flux2": "flux2_dev",
            "qwen": "qwen_image_2512",
            "glm_image": "glm_image",
            "firered": "firered_edit_1_1",
            "helios": "helios_distilled",
            "ltx23": "ltx_2_3",
            "wan22": "wan22_14b",
            "cogvideox": "cogvideox_5b",
            "sd3": "sd35_medium",
        }
        registry_key = type_to_key.get(model_type)
        if registry_key:
            rec = get_recommended_quantization(registry_key, vram_tier)
            if rec:
                return (rec,)

        # Fallback: Check if GPU supports native FP8 for optimal recommendations
        supports_fp8, _, _ = _detect_fp8_capability()

        # Determine optimal precision
        if vram_tier in ["ultra_low", "low"]:
            # FP8 is better than int8 on capable hardware
            if supports_fp8:
                return ("fp8_e4m3fn",)
            return ("int8",)
        elif vram_tier in ["medium", "high"]:
            if model_type in ["flux", "flux_schnell", "wan21", "wan22", "cogvideox"]:
                if supports_fp8:
                    return ("fp8_e4m3fn",)
                return ("fp16",)
            else:
                return ("fp16",)
        else:
            # ultra, extreme, hpc — use higher precision
            if supports_fp8 and model_type in [
                "flux",
                "flux_schnell",
                "wan22",
                "cogvideox",
            ]:
                # FP8 on large models even on big GPUs saves VRAM for larger batch/resolution
                return ("fp8_e4m3fn",)
            if model_type == "cogvideox":
                return ("bf16",)
            return ("fp16",)


# ============== RESOLUTION SCALER ==============
class ResolutionScaler:
    """Scale generation resolution based on VRAM constraints"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vram_tier": ("STRING", {"forceInput": True}),
                "base_width": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "base_height": ("INT", {"default": 1024, "min": 256, "max": 4096}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("scaled_width", "scaled_height")
    FUNCTION = "scale_resolution"
    CATEGORY = "Laura Studio/Optimization"
    DESCRIPTION = "Auto-scale resolution to fit VRAM"

    def scale_resolution(self, vram_tier, base_width, base_height):
        max_pixels = {
            "ultra_low": 512 * 512,
            "low": 768 * 768,
            "medium": 1024 * 1024,
            "high": 1024 * 1024,
            "very_high": 1280 * 1280,
            "ultra": 1536 * 1536,
            "extreme": 2048 * 2048,
            "hpc": 4096 * 4096,
        }

        limit = max_pixels.get(vram_tier, 1024 * 1024)
        current_pixels = base_width * base_height

        if current_pixels <= limit:
            return (base_width, base_height)

        # Scale down while maintaining aspect ratio
        scale_factor = (limit / current_pixels) ** 0.5

        # Round to nearest multiple of 8
        scaled_w = int(round((base_width * scale_factor) / 8) * 8)
        scaled_h = int(round((base_height * scale_factor) / 8) * 8)

        return (scaled_w, scaled_h)


# ============== MODEL OFFLOAD CONFIG ==============
class ModelOffloadConfig:
    """Generate configuration for CPU/GPU model offloading"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vram_tier": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("enable_cpu_offload", "sequential_offload")
    FUNCTION = "get_offload_config"
    CATEGORY = "Laura Studio/Optimization"
    DESCRIPTION = "Generate memory management settings"

    def get_offload_config(self, vram_tier):
        enable_cpu = False
        seq_offload = False

        if vram_tier in ["ultra_low"]:
            enable_cpu = True
            seq_offload = True
        elif vram_tier in ["low", "medium"]:
            enable_cpu = True
            seq_offload = False

        return (enable_cpu, seq_offload)


# ============== APPLY QUANTIZATION CONFIG ==============
class QuantizationConfig:
    """Package optimization settings into a config object for AdvancedModelLoader"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "weight_dtype": ("STRING", {"forceInput": True}),
                "enable_cpu_offload": ("BOOLEAN", {"forceInput": True}),
                "sequential_offload": ("BOOLEAN", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("QUANT_CONFIG",)
    RETURN_NAMES = ("quant_config",)
    FUNCTION = "build_config"
    CATEGORY = "Laura Studio/Optimization"
    DESCRIPTION = "Package optimization settings"

    def build_config(self, weight_dtype, enable_cpu_offload, sequential_offload):
        config = {
            "weight_dtype": weight_dtype,
            "enable_cpu_offload": enable_cpu_offload,
            "sequential_offload": sequential_offload,
        }
        return (config,)


# ============== FP8 TRANSFORMER ENGINE CONFIG (v0.9) ==============
class FP8TransformerConfig:
    """Native FP8 Transformer Engine — auto-detects Ada Lovelace+ GPUs and provides
    FP8 (float8_e4m3fn) precision for ~50% memory reduction with minimal quality loss.
    Requires: PyTorch 2.1+ with FP8 support, CUDA SM 8.9+ (RTX 4060+, Ada Lovelace).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (
                    ["auto", "force_fp8", "force_fp16", "force_bf16"],
                    {"default": "auto"},
                ),
            },
            "optional": {
                "fp8_format": (
                    ["e4m3fn", "e5m2"],
                    {"default": "e4m3fn"},
                ),
                "fallback_dtype": (
                    ["fp16", "bf16"],
                    {"default": "fp16"},
                ),
                "cast_activations": (
                    "BOOLEAN",
                    {"default": False},
                ),
            },
        }

    RETURN_TYPES = ("QUANT_CONFIG", "STRING", "BOOLEAN")
    RETURN_NAMES = ("quant_config", "status_report", "fp8_active")
    FUNCTION = "configure_fp8"
    CATEGORY = "Laura Studio/Optimization"
    DESCRIPTION = "Configure native FP8 precision (Ada Lovelace+ GPUs). Auto-detects capability and falls back gracefully."

    def configure_fp8(
        self,
        mode="auto",
        fp8_format="e4m3fn",
        fallback_dtype="fp16",
        cast_activations=False,
    ):
        supports_fp8, gpu_name, cc = _detect_fp8_capability()
        cc_str = f"SM {cc[0]}.{cc[1]}" if cc else "N/A"

        report_lines = [
            "--- FP8 Transformer Engine ---",
            f"GPU: {gpu_name}",
            f"Compute Capability: {cc_str}",
            f"Native FP8 Support: {'Yes' if supports_fp8 else 'No'}",
            f"PyTorch float8: {'Available' if hasattr(torch, 'float8_e4m3fn') else 'Not Available'}",
        ]

        use_fp8 = False
        weight_dtype = fallback_dtype

        if mode == "force_fp8":
            if hasattr(torch, "float8_e4m3fn"):
                use_fp8 = True
                weight_dtype = f"fp8_{fp8_format}"
                if not supports_fp8:
                    report_lines.append(
                        "WARNING: FP8 forced but GPU lacks native support — emulated, may be slower"
                    )
                else:
                    report_lines.append("FP8 forced ON — native acceleration active")
            else:
                report_lines.append(
                    "ERROR: FP8 forced but torch.float8_e4m3fn not available — falling back"
                )
                weight_dtype = fallback_dtype
        elif mode == "force_fp16":
            weight_dtype = "fp16"
            report_lines.append("Forced FP16 precision")
        elif mode == "force_bf16":
            weight_dtype = "bf16"
            report_lines.append("Forced BF16 precision")
        elif mode == "auto":
            if supports_fp8:
                use_fp8 = True
                weight_dtype = f"fp8_{fp8_format}"
                report_lines.append(
                    f"Auto-detected Ada Lovelace+ — using FP8 ({fp8_format})"
                )
            else:
                weight_dtype = fallback_dtype
                report_lines.append(
                    f"GPU does not support native FP8 — using {fallback_dtype}"
                )

        report_lines.append(f"Active dtype: {weight_dtype}")

        config = {
            "weight_dtype": weight_dtype,
            "fp8_active": use_fp8,
            "fp8_format": fp8_format if use_fp8 else None,
            "cast_activations": cast_activations and use_fp8,
            "enable_cpu_offload": False,
            "sequential_offload": False,
        }

        return (config, "\n".join(report_lines), use_fp8)


NODE_CLASS_MAPPINGS.update(
    {
        "VRAMAutoDetector": VRAMAutoDetector,
        "QuantizationSelector": QuantizationSelector,
        "ResolutionScaler": ResolutionScaler,
        "ModelOffloadConfig": ModelOffloadConfig,
        "QuantizationConfig": QuantizationConfig,
        "FP8TransformerConfig": FP8TransformerConfig,
    }
)

NODE_DISPLAY_NAME_MAPPINGS.update(
    {
        "VRAMAutoDetector": "VRAM Auto-Detector",
        "QuantizationSelector": "Quantization Selector",
        "ResolutionScaler": "Resolution Scaler",
        "ModelOffloadConfig": "Model Offload Config",
        "QuantizationConfig": "Quantization Config Builder",
        "FP8TransformerConfig": "FP8 Transformer Engine",
    }
)
