"""
Laura Image Studio - Hardware Profiler Node
Detects GPU, VRAM, precision capabilities, and classifies into VRAM tiers.
"""

import platform

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from .model_registry import VRAM_TIERS

TIER_NAMES = ["auto"] + list(VRAM_TIERS.keys())

SM_ARCH_NAMES = {
    (7, 0): "Volta",
    (7, 5): "Turing",
    (8, 0): "Ampere",
    (8, 6): "Ampere",
    (8, 9): "Ada Lovelace",
    (9, 0): "Hopper",
}


def _get_arch_name(major, minor):
    name = SM_ARCH_NAMES.get((major, minor))
    if name:
        return name
    for (m, _), n in SM_ARCH_NAMES.items():
        if m == major:
            return n
    return "Unknown"


def _classify_vram_tier(vram_gb):
    for tier_name, tier in VRAM_TIERS.items():
        if tier["min_gb"] <= vram_gb < tier["max_gb"]:
            return tier_name
    return "hpc"


class LauraHardwareProfiler:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "force_tier": (TIER_NAMES, {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("vram_tier", "vram_gb", "gpu_name", "profile_report")
    FUNCTION = "profile"
    CATEGORY = "Laura Studio/Core"
    DESCRIPTION = "Detect GPU, VRAM, precision support, and classify hardware tier"

    def profile(self, force_tier):
        gpu_name = "No GPU (CPU mode)"
        vram_gb = 0.0
        sm_major, sm_minor = 0, 0
        arch_name = "N/A"
        fp8_supported = False
        fp16_supported = False
        bf16_supported = False

        if HAS_TORCH and torch.cuda.is_available():
            dev = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(dev)
            vram_gb = round(torch.cuda.get_device_properties(dev).total_mem / (1024 ** 3), 1)
            sm_major, sm_minor = torch.cuda.get_device_capability(dev)
            arch_name = _get_arch_name(sm_major, sm_minor)
            fp8_supported = (sm_major > 8) or (sm_major == 8 and sm_minor >= 9)
            fp16_supported = sm_major >= 7
            bf16_supported = sm_major >= 8

        if force_tier != "auto" and force_tier in VRAM_TIERS:
            vram_tier = force_tier
        else:
            vram_tier = _classify_vram_tier(vram_gb)

        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1) if HAS_PSUTIL else 0.0

        os_info = f"{platform.system()} {platform.release()}"

        compute_str = f"SM {sm_major}.{sm_minor} ({arch_name})" if sm_major > 0 else "N/A"
        fp8_str = "Supported" if fp8_supported else "Not supported"
        fp16_str = "Supported" if fp16_supported else "Not supported"
        bf16_str = "Supported" if bf16_supported else "Not supported"

        report = (
            f"=== Laura Studio Hardware Profile ===\n"
            f"GPU: {gpu_name}\n"
            f"VRAM: {vram_gb} GB (tier: {vram_tier})\n"
            f"Compute: {compute_str}\n"
            f"FP8: {fp8_str}\n"
            f"FP16: {fp16_str}\n"
            f"BF16: {bf16_str}\n"
            f"RAM: {ram_gb} GB\n"
            f"OS: {os_info}"
        )

        return (vram_tier, vram_gb, gpu_name, report)


NODE_CLASS_MAPPINGS = {
    "LauraHardwareProfiler": LauraHardwareProfiler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LauraHardwareProfiler": "Laura Hardware Profiler",
}
