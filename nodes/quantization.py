"""
Laura Image Studio - VRAM Optimization & Quantization Nodes
Automatically detect GPU capabilities and adjust model precision and resolution
"""

import torch
import os

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


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
            except:
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
                        "sd15",
                        "sd3",
                        "wan21",
                        "wan22",
                        "cogvideox",
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
        # Determine optimal precision
        if vram_tier in ["ultra_low", "low"]:
            return ("int8",)
        elif vram_tier in ["medium", "high"]:
            if model_type in ["flux", "flux_schnell", "wan21", "wan22", "cogvideox"]:
                return ("fp8_e4m3fn",)
            else:
                return ("fp16",)
        else:
            # ultra, extreme, hpc
            if model_type in ["flux", "flux_schnell", "wan21", "wan22"]:
                # Many large models still do best in fp16/bf16 even on big GPUs
                return ("fp16",)
            elif model_type == "cogvideox":
                return ("bf16",)
            else:
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


NODE_CLASS_MAPPINGS.update(
    {
        "VRAMAutoDetector": VRAMAutoDetector,
        "QuantizationSelector": QuantizationSelector,
        "ResolutionScaler": ResolutionScaler,
        "ModelOffloadConfig": ModelOffloadConfig,
        "QuantizationConfig": QuantizationConfig,
    }
)

NODE_DISPLAY_NAME_MAPPINGS.update(
    {
        "VRAMAutoDetector": "VRAM Auto-Detector",
        "QuantizationSelector": "Quantization Selector",
        "ResolutionScaler": "Resolution Scaler",
        "ModelOffloadConfig": "Model Offload Config",
        "QuantizationConfig": "Quantization Config Builder",
    }
)
