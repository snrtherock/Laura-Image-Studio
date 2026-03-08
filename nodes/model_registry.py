"""
Laura Image Studio - Model Registry
Centralized registry of all supported models with metadata, quantization variants,
VRAM requirements, compatibility info, and helper functions.

This module is PURE DATA + HELPERS -- no torch, no folder_paths, no ComfyUI imports.
It must be importable in a plain Python environment for tooling and testing.
"""

import os


# =============================================================================
# REGISTRY VERSION
# =============================================================================

_REGISTRY_VERSION = "1.0.0-2026.03.08"


def get_registry_version():
    """Return the current registry version string."""
    return _REGISTRY_VERSION


# =============================================================================
# VRAM TIERS
# =============================================================================

VRAM_TIERS = {
    "ultra_low": {
        "min_gb": 0,
        "max_gb": 4,
        "label": "Ultra Low (< 4 GB)",
    },
    "low": {
        "min_gb": 4,
        "max_gb": 6,
        "label": "Low (4-6 GB)",
    },
    "medium": {
        "min_gb": 6,
        "max_gb": 8,
        "label": "Medium (6-8 GB)",
    },
    "high": {
        "min_gb": 8,
        "max_gb": 12,
        "label": "High (8-12 GB)",
    },
    "very_high": {
        "min_gb": 12,
        "max_gb": 16,
        "label": "Very High (12-16 GB)",
    },
    "ultra": {
        "min_gb": 16,
        "max_gb": 24,
        "label": "Ultra (16-24 GB)",
    },
    "extreme": {
        "min_gb": 24,
        "max_gb": 48,
        "label": "Extreme (24-48 GB)",
    },
    "hpc": {
        "min_gb": 48,
        "max_gb": 999,
        "label": "HPC (48+ GB)",
    },
}


# =============================================================================
# CATEGORIES
# =============================================================================

CATEGORIES = {
    "image_gen": {
        "label": "Image Generation",
        "description": "Text-to-image and image-to-image generation models",
    },
    "image_edit": {
        "label": "Image Editing",
        "description": "Instruction-based and reference-based image editing models",
    },
    "video_gen": {
        "label": "Video Generation",
        "description": "Text-to-video and image-to-video generation models",
    },
    "upscale": {
        "label": "Upscaling",
        "description": "Super-resolution and detail-enhancement models",
    },
    "vae": {
        "label": "VAE",
        "description": "Variational autoencoders for encoding/decoding latents",
    },
    "text_encoder": {
        "label": "Text Encoder",
        "description": "CLIP, T5, and other text encoding models",
    },
    "controlnet": {
        "label": "ControlNet",
        "description": "Conditional control models for guided generation",
    },
}


# =============================================================================
# MODEL REGISTRY
# =============================================================================

MODEL_REGISTRY = {

    # =========================================================================
    # IMAGE GENERATION (9 models)
    # =========================================================================

    "z_image_turbo": {
        "display_name": "Z-Image Turbo",
        "family": "z_image",
        "category": "image_gen",
        "repo": "Tongyi-MAI/Z-Image-Turbo",
        "homepage": "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo",
        "license": "Apache-2.0",
        "params": "6B",
        "architecture": "S3-DiT",
        "status": "released",
        "files": {
            "diffusion_model": {
                "folder": "diffusion_models",
                "filename": "z_image_turbo_bf16.safetensors",
                "size_gb": 11.46,
            },
            "text_encoder": {
                "folder": "text_encoders",
                "filename": "qwen_3_4b.safetensors",
                "size_gb": 7.49,
            },
            "vae": {
                "folder": "vae",
                "filename": "ae.safetensors",
                "size_gb": 0.31,
            },
        },
        "total_size_gb": 19.26,
        "quantization_variants": {
            "bf16": {"vram_gb": 16, "quality": "reference", "speed": "1x"},
            "fp8": {"vram_gb": 10, "quality": "near-lossless", "speed": "1.2x"},
            "gguf_q8": {"vram_gb": 8, "quality": "high", "speed": "1.1x"},
            "gguf_q4": {"vram_gb": 6, "quality": "good", "speed": "1.3x"},
            "nvfp4": {"vram_gb": 6, "quality": "good", "speed": "1.5x"},
        },
        "quantization_recommendation": {
            "ultra_low": "gguf_q4",
            "low": "gguf_q4",
            "medium": "gguf_q8",
            "high": "fp8",
            "very_high": "bf16",
            "ultra": "bf16",
            "extreme": "bf16",
            "hpc": "bf16",
        },
        "inference": {
            "steps": 8,
            "cfg": False,
            "cfg_scale": None,
            "resolution_range": {"min": 512, "max": 2048},
            "default_resolution": {"width": 1024, "height": 1024},
        },
        "compatibility": {
            "lora": False,
            "controlnet": True,
            "controlnet_models": ["ControlNet Union 2.1"],
            "ipadapter": True,
            "negative_prompt": False,
            "multi_reference": False,
            "image_editing": False,
            "lora_zoo": None,
            "lora_zoo_models": [],
        },
        "quality_score": {
            "elo_rank": None,
            "strengths": ["Fastest Z-Image variant", "8-step generation", "No CFG needed"],
            "weaknesses": ["No LoRA support", "No negative prompt"],
        },
        "module_dependency": "generation",
        "comfyui_type": "diffusion_models",
    },

    "z_image_base": {
        "display_name": "Z-Image",
        "family": "z_image",
        "category": "image_gen",
        "repo": "Tongyi-MAI/Z-Image",
        "homepage": "https://huggingface.co/Tongyi-MAI/Z-Image",
        "license": "Apache-2.0",
        "params": "6B",
        "architecture": "S3-DiT",
        "status": "released",
        "files": {
            "diffusion_model": {
                "folder": "diffusion_models",
                "filename": "z_image_bf16.safetensors",
                "size_gb": 11.46,
            },
            "text_encoder": {
                "folder": "text_encoders",
                "filename": "qwen_3_4b.safetensors",
                "size_gb": 7.49,
            },
            "vae": {
                "folder": "vae",
                "filename": "ae.safetensors",
                "size_gb": 0.31,
            },
        },
        "total_size_gb": 19.26,
        "quantization_variants": {
            "bf16": {"vram_gb": 16, "quality": "reference", "speed": "1x"},
            "fp8": {"vram_gb": 10, "quality": "near-lossless", "speed": "1.2x"},
            "gguf_q8": {"vram_gb": 8, "quality": "high", "speed": "1.1x"},
            "gguf_q4": {"vram_gb": 6, "quality": "good", "speed": "1.3x"},
            "nvfp4": {"vram_gb": 6, "quality": "good", "speed": "1.5x"},
        },
        "quantization_recommendation": {
            "ultra_low": "gguf_q4",
            "low": "gguf_q4",
            "medium": "gguf_q8",
            "high": "fp8",
            "very_high": "bf16",
            "ultra": "bf16",
            "extreme": "bf16",
            "hpc": "bf16",
        },
        "inference": {
            "steps": 50,
            "cfg": True,
            "cfg_scale": 4.0,
            "resolution_range": {"min": 512, "max": 2048},
            "default_resolution": {"width": 1024, "height": 1024},
        },
        "compatibility": {
            "lora": True,
            "controlnet": True,
            "controlnet_models": ["ControlNet Union 2.1"],
            "ipadapter": True,
            "negative_prompt": True,
            "multi_reference": False,
            "image_editing": False,
            "lora_zoo": None,
            "lora_zoo_models": [],
        },
        "quality_score": {
            "elo_rank": None,
            "strengths": ["Full Z-Image quality", "LoRA support", "Negative prompt support"],
            "weaknesses": ["50-step generation is slower"],
        },
        "module_dependency": "generation",
        "comfyui_type": "diffusion_models",
    },

    "z_image_edit": {
        "display_name": "Z-Image Edit",
        "family": "z_image",
        "category": "image_gen",
        "repo": "Tongyi-MAI/Z-Image-Edit",
        "homepage": "https://huggingface.co/Tongyi-MAI/Z-Image-Edit",
        "license": "Apache-2.0",
        "params": "6B",
        "architecture": "S3-DiT",
        "status": "pending_release",
        "files": {},
        "total_size_gb": 0,
        "quantization_variants": {},
        "quantization_recommendation": {},
        "inference": {
            "steps": None,
            "cfg": None,
            "cfg_scale": None,
            "resolution_range": None,
            "default_resolution": None,
        },
        "compatibility": {
            "lora": None,
            "controlnet": None,
            "controlnet_models": [],
            "ipadapter": None,
            "negative_prompt": None,
            "multi_reference": None,
            "image_editing": True,
            "lora_zoo": None,
            "lora_zoo_models": [],
        },
        "quality_score": {
            "elo_rank": None,
            "strengths": ["Instruction-based editing from Z-Image family"],
            "weaknesses": ["Not yet released"],
        },
        "module_dependency": "inpainting",
        "comfyui_type": "diffusion_models",
    },

    "flux2_dev": {
        "display_name": "FLUX.2 Dev",
        "family": "flux",
        "category": "image_gen",
        "repo": "black-forest-labs/FLUX.2-dev",
        "homepage": "https://huggingface.co/black-forest-labs/FLUX.2-dev",
        "license": "FLUX Non-Commercial License",
        "params": "32B",
        "architecture": "Rectified Flow Transformer",
        "status": "released",
        "files": {
            "checkpoint": {
                "folder": "checkpoints",
                "filename": "flux2-dev.safetensors",
                "size_gb": 24.0,
            },
        },
        "total_size_gb": 24.0,
        "quantization_variants": {
            "bf16": {"vram_gb": 48, "quality": "reference", "speed": "1x"},
            "fp8": {"vram_gb": 24, "quality": "near-lossless", "speed": "1.2x"},
            "bnb_4bit": {
                "vram_gb": 8,
                "quality": "good",
                "speed": "1.0x",
                "alt_repo": "diffusers/FLUX.2-dev-bnb-4bit",
            },
            "nvfp4": {
                "vram_gb": 10,
                "quality": "good",
                "speed": "1.5x",
                "alt_repo": "black-forest-labs/FLUX.2-dev-NVFP4",
            },
        },
        "quantization_recommendation": {
            "ultra_low": "bnb_4bit",
            "low": "bnb_4bit",
            "medium": "bnb_4bit",
            "high": "nvfp4",
            "very_high": "nvfp4",
            "ultra": "fp8",
            "extreme": "bf16",
            "hpc": "bf16",
        },
        "inference": {
            "steps": 30,
            "cfg": True,
            "cfg_scale": 3.5,
            "resolution_range": {"min": 512, "max": 2048},
            "default_resolution": {"width": 1024, "height": 1024},
        },
        "compatibility": {
            "lora": True,
            "controlnet": True,
            "controlnet_models": ["FLUX ControlNet Union"],
            "ipadapter": True,
            "negative_prompt": True,
            "multi_reference": True,
            "image_editing": False,
            "lora_zoo": None,
            "lora_zoo_models": [],
        },
        "quality_score": {
            "elo_rank": None,
            "strengths": ["32B parameters", "Rectified Flow", "Full compatibility"],
            "weaknesses": ["Large VRAM requirement", "Non-commercial license"],
        },
        "module_dependency": "generation",
        "comfyui_type": "checkpoints",
    },

    "flux1_dev": {
        "display_name": "FLUX.1 Dev",
        "family": "flux",
        "category": "image_gen",
        "repo": "black-forest-labs/FLUX.1-dev",
        "homepage": "https://huggingface.co/black-forest-labs/FLUX.1-dev",
        "license": "FLUX Non-Commercial License",
        "params": "12B",
        "architecture": "Rectified Flow Transformer",
        "status": "released",
        "files": {
            "checkpoint": {
                "folder": "checkpoints",
                "filename": "flux1-dev.safetensors",
                "size_gb": 24.0,
            },
        },
        "total_size_gb": 24.0,
        "quantization_variants": {
            "bf16": {"vram_gb": 24, "quality": "reference", "speed": "1x"},
            "fp8": {"vram_gb": 12, "quality": "near-lossless", "speed": "1.2x"},
            "gguf_q8": {"vram_gb": 8, "quality": "high", "speed": "1.1x"},
        },
        "quantization_recommendation": {
            "ultra_low": "gguf_q8",
            "low": "gguf_q8",
            "medium": "gguf_q8",
            "high": "fp8",
            "very_high": "fp8",
            "ultra": "bf16",
            "extreme": "bf16",
            "hpc": "bf16",
        },
        "inference": {
            "steps": 30,
            "cfg": True,
            "cfg_scale": 3.5,
            "resolution_range": {"min": 512, "max": 2048},
            "default_resolution": {"width": 1024, "height": 1024},
        },
        "compatibility": {
            "lora": True,
            "controlnet": True,
            "controlnet_models": ["FLUX ControlNet Union"],
            "ipadapter": True,
            "negative_prompt": True,
            "multi_reference": True,
            "image_editing": False,
            "lora_zoo": None,
            "lora_zoo_models": [],
        },
        "quality_score": {
            "elo_rank": None,
            "strengths": ["Strong community ecosystem", "Proven quality", "Wide LoRA support"],
            "weaknesses": ["Non-commercial license"],
        },
        "module_dependency": "generation",
        "comfyui_type": "checkpoints",
    },

    "flux1_schnell": {
        "display_name": "FLUX.1 Schnell",
        "family": "flux",
        "category": "image_gen",
        "repo": "black-forest-labs/FLUX.1-schnell",
        "homepage": "https://huggingface.co/black-forest-labs/FLUX.1-schnell",
        "license": "Apache-2.0",
        "params": "12B",
        "architecture": "Rectified Flow Transformer",
        "status": "released",
        "files": {
            "checkpoint": {
                "folder": "checkpoints",
                "filename": "flux1-schnell.safetensors",
                "size_gb": 24.0,
            },
        },
        "total_size_gb": 24.0,
        "quantization_variants": {
            "bf16": {"vram_gb": 24, "quality": "reference", "speed": "1x"},
            "fp8": {"vram_gb": 12, "quality": "near-lossless", "speed": "1.2x"},
            "gguf_q8": {"vram_gb": 8, "quality": "high", "speed": "1.1x"},
        },
        "quantization_recommendation": {
            "ultra_low": "gguf_q8",
            "low": "gguf_q8",
            "medium": "gguf_q8",
            "high": "fp8",
            "very_high": "fp8",
            "ultra": "bf16",
            "extreme": "bf16",
            "hpc": "bf16",
        },
        "inference": {
            "steps": 4,
            "cfg": False,
            "cfg_scale": None,
            "resolution_range": {"min": 512, "max": 2048},
            "default_resolution": {"width": 1024, "height": 1024},
        },
        "compatibility": {
            "lora": False,
            "controlnet": True,
            "controlnet_models": ["FLUX ControlNet Union"],
            "ipadapter": True,
            "negative_prompt": False,
            "multi_reference": False,
            "image_editing": False,
            "lora_zoo": None,
            "lora_zoo_models": [],
        },
        "quality_score": {
            "elo_rank": None,
            "strengths": ["4-step distilled", "Apache-2.0 license", "Very fast"],
            "weaknesses": ["No LoRA support", "Lower quality than dev"],
        },
        "module_dependency": "generation",
        "comfyui_type": "checkpoints",
    },

    "qwen_image_2512": {
        "display_name": "Qwen Image 2512",
        "family": "qwen",
        "category": "image_gen",
        "repo": "QwenLM/Qwen-Image-2512",
        "homepage": "https://huggingface.co/QwenLM/Qwen-Image-2512",
        "license": "Apache-2.0",
        "params": "~7B",
        "architecture": None,
        "status": "released",
        "files": {},
        "total_size_gb": 0,
        "quantization_variants": {
            "bf16": {"vram_gb": 16, "quality": "reference", "speed": "1x"},
            "fp8": {"vram_gb": 10, "quality": "near-lossless", "speed": "1.2x"},
        },
        "quantization_recommendation": {
            "ultra_low": "fp8",
            "low": "fp8",
            "medium": "fp8",
            "high": "fp8",
            "very_high": "bf16",
            "ultra": "bf16",
            "extreme": "bf16",
            "hpc": "bf16",
        },
        "inference": {
            "steps": None,
            "cfg": None,
            "cfg_scale": None,
            "resolution_range": None,
            "default_resolution": None,
        },
        "compatibility": {
            "lora": True,
            "controlnet": False,
            "controlnet_models": [],
            "ipadapter": False,
            "negative_prompt": False,
            "multi_reference": False,
            "image_editing": False,
            "lora_zoo": None,
            "lora_zoo_models": [],
        },
        "quality_score": {
            "elo_rank": None,
            "strengths": ["Qwen ecosystem", "Apache-2.0 license", "LoRA support"],
            "weaknesses": ["Limited compatibility features"],
        },
        "module_dependency": "generation",
        "comfyui_type": "checkpoints",
    },

    "glm_image": {
        "display_name": "GLM-Image",
        "family": "glm",
        "category": "image_gen",
        "repo": "zai-org/GLM-Image",
        "homepage": "https://huggingface.co/zai-org/GLM-Image",
        "license": "Apache-2.0",
        "params": "16B",
        "architecture": "Auto-regressive Dense-knowledge",
        "status": "released",
        "files": {},
        "total_size_gb": 0,
        "quantization_variants": {
            "bf16": {"vram_gb": 32, "quality": "reference", "speed": "1x"},
            "fp8": {"vram_gb": 16, "quality": "near-lossless", "speed": "1.2x"},
            "gguf_q8": {"vram_gb": 12, "quality": "high", "speed": "1.1x"},
        },
        "quantization_recommendation": {
            "ultra_low": "gguf_q8",
            "low": "gguf_q8",
            "medium": "gguf_q8",
            "high": "gguf_q8",
            "very_high": "fp8",
            "ultra": "fp8",
            "extreme": "bf16",
            "hpc": "bf16",
        },
        "inference": {
            "steps": None,
            "cfg": None,
            "cfg_scale": None,
            "resolution_range": None,
            "default_resolution": None,
        },
        "compatibility": {
            "lora": False,
            "controlnet": False,
            "controlnet_models": [],
            "ipadapter": False,
            "negative_prompt": False,
            "multi_reference": False,
            "image_editing": False,
            "lora_zoo": None,
            "lora_zoo_models": [],
        },
        "quality_score": {
            "elo_rank": None,
            "strengths": ["16B auto-regressive model", "Dense-knowledge architecture"],
            "weaknesses": ["No LoRA/ControlNet/IPAdapter support"],
        },
        "module_dependency": "generation",
        "comfyui_type": "checkpoints",
    },

    "sd35_medium": {
        "display_name": "Stable Diffusion 3.5 Medium",
        "family": "stable_diffusion",
        "category": "image_gen",
        "repo": "stabilityai/stable-diffusion-3.5-medium",
        "homepage": "https://huggingface.co/stabilityai/stable-diffusion-3.5-medium",
        "license": "Stability AI Community License",
        "params": "2.5B",
        "architecture": "MMDiT",
        "status": "released",
        "files": {
            "checkpoint": {
                "folder": "checkpoints",
                "filename": "sd3.5_medium.safetensors",
                "size_gb": 6.0,
            },
        },
        "total_size_gb": 6.0,
        "quantization_variants": {
            "bf16": {"vram_gb": 8, "quality": "reference", "speed": "1x"},
            "fp8": {"vram_gb": 6, "quality": "near-lossless", "speed": "1.2x"},
        },
        "quantization_recommendation": {
            "ultra_low": "fp8",
            "low": "fp8",
            "medium": "fp8",
            "high": "bf16",
            "very_high": "bf16",
            "ultra": "bf16",
            "extreme": "bf16",
            "hpc": "bf16",
        },
        "inference": {
            "steps": 28,
            "cfg": True,
            "cfg_scale": 4.5,
            "resolution_range": {"min": 512, "max": 1536},
            "default_resolution": {"width": 1024, "height": 1024},
        },
        "compatibility": {
            "lora": True,
            "controlnet": True,
            "controlnet_models": ["SD3 ControlNet"],
            "ipadapter": True,
            "negative_prompt": True,
            "multi_reference": True,
            "image_editing": False,
            "lora_zoo": None,
            "lora_zoo_models": [],
        },
        "quality_score": {
            "elo_rank": None,
            "strengths": ["Small footprint", "High prompt adherence", "Full compatibility"],
            "weaknesses": ["Lower ceiling than larger models"],
        },
        "module_dependency": "generation",
        "comfyui_type": "checkpoints",
    },

    # =========================================================================
    # IMAGE EDITING (1 model)
    # =========================================================================

    "firered_edit_1_1": {
        "display_name": "FireRed Image Edit 1.1",
        "family": "firered",
        "category": "image_edit",
        "repo": "FireRedTeam/FireRed-Image-Edit-1.1",
        "homepage": "https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.1",
        "license": "Apache-2.0",
        "params": "~12B",
        "architecture": None,
        "status": "released",
        "files": {},
        "total_size_gb": 0,
        "quantization_variants": {
            "bf16": {"vram_gb": 30, "quality": "reference", "speed": "1x"},
            "gguf_q8": {
                "vram_gb": 10,
                "quality": "high",
                "speed": "1.1x",
                "alt_repo": "drbaph/FireRed-Image-Edit-1.0_ComfyUI_Quants",
            },
        },
        "quantization_recommendation": {
            "ultra_low": "gguf_q8",
            "low": "gguf_q8",
            "medium": "gguf_q8",
            "high": "gguf_q8",
            "very_high": "gguf_q8",
            "ultra": "bf16",
            "extreme": "bf16",
            "hpc": "bf16",
        },
        "inference": {
            "steps": None,
            "cfg": None,
            "cfg_scale": None,
            "resolution_range": None,
            "default_resolution": None,
        },
        "compatibility": {
            "lora": True,
            "controlnet": False,
            "controlnet_models": [],
            "ipadapter": False,
            "negative_prompt": False,
            "multi_reference": False,
            "image_editing": True,
            "lora_zoo": "FireRedTeam/FireRed-Image-Edit-LoRA-Zoo",
            "lora_zoo_models": [
                {"name": "Makeup", "description": "Makeup transfer and application"},
                {"name": "Covercraft", "description": "Cover art and creative styling"},
                {"name": "Style", "description": "Artistic style transfer"},
            ],
        },
        "quality_score": {
            "elo_rank": None,
            "strengths": ["SOTA instruction-based editing", "Rich LoRA zoo", "Multi-edit support"],
            "weaknesses": ["Large VRAM at bf16"],
        },
        "module_dependency": "inpainting",
        "comfyui_type": "checkpoints",
    },

    # =========================================================================
    # VIDEO GENERATION (6 models)
    # =========================================================================

    "helios_distilled": {
        "display_name": "Helios Distilled",
        "family": "helios",
        "category": "video_gen",
        "repo": "BestWishYsh/Helios-Distilled",
        "homepage": "https://huggingface.co/BestWishYsh/Helios-Distilled",
        "license": "Apache-2.0",
        "params": "14B",
        "architecture": "Distilled Transformer",
        "status": "brand_new",
        "released": "2026-03-04",
        "files": {},
        "total_size_gb": 0,
        "quantization_variants": {},
        "quantization_recommendation": {},
        "inference": {
            "steps": None,
            "cfg": None,
            "cfg_scale": None,
            "resolution_range": None,
            "default_resolution": None,
        },
        "compatibility": {
            "lora": False,
            "controlnet": False,
            "controlnet_models": [],
            "ipadapter": False,
            "negative_prompt": False,
            "multi_reference": False,
            "image_editing": False,
            "lora_zoo": None,
            "lora_zoo_models": [],
        },
        "quality_score": {
            "elo_rank": None,
            "strengths": ["14B distilled video model", "Brand new release (2026-03-04)"],
            "weaknesses": ["Newly released, limited community testing"],
        },
        "module_dependency": "video",
        "comfyui_type": "checkpoints",
    },

    "helios_base": {
        "display_name": "Helios Base",
        "family": "helios",
        "category": "video_gen",
        "repo": "BestWishYsh/Helios-Base",
        "homepage": "https://huggingface.co/BestWishYsh/Helios-Base",
        "license": "Apache-2.0",
        "params": "14B",
        "architecture": None,
        "status": "released",
        "files": {},
        "total_size_gb": 0,
        "quantization_variants": {},
        "quantization_recommendation": {},
        "inference": {
            "steps": None,
            "cfg": None,
            "cfg_scale": None,
            "resolution_range": None,
            "default_resolution": None,
        },
        "compatibility": {
            "lora": False,
            "controlnet": False,
            "controlnet_models": [],
            "ipadapter": False,
            "negative_prompt": False,
            "multi_reference": False,
            "image_editing": False,
            "lora_zoo": None,
            "lora_zoo_models": [],
        },
        "quality_score": {
            "elo_rank": None,
            "strengths": ["14B video generation model"],
            "weaknesses": [],
        },
        "module_dependency": "video",
        "comfyui_type": "checkpoints",
    },

    "ltx_2_3": {
        "display_name": "LTX 2.3",
        "family": "ltx",
        "category": "video_gen",
        "repo": "Lightricks/LTX-2.3",
        "homepage": "https://huggingface.co/Lightricks/LTX-2.3",
        "license": "Apache-2.0",
        "params": "~2B",
        "architecture": None,
        "status": "released",
        "files": {},
        "total_size_gb": 0,
        "quantization_variants": {
            "bf16": {"vram_gb": 12, "quality": "reference", "speed": "1x"},
            "fp8": {"vram_gb": 8, "quality": "near-lossless", "speed": "1.2x"},
        },
        "quantization_recommendation": {
            "ultra_low": "fp8",
            "low": "fp8",
            "medium": "fp8",
            "high": "bf16",
            "very_high": "bf16",
            "ultra": "bf16",
            "extreme": "bf16",
            "hpc": "bf16",
        },
        "inference": {
            "steps": None,
            "cfg": None,
            "cfg_scale": None,
            "resolution_range": None,
            "default_resolution": None,
        },
        "compatibility": {
            "lora": False,
            "controlnet": False,
            "controlnet_models": [],
            "ipadapter": False,
            "negative_prompt": False,
            "multi_reference": False,
            "image_editing": False,
            "lora_zoo": None,
            "lora_zoo_models": [],
        },
        "quality_score": {
            "elo_rank": None,
            "strengths": ["Audio generation", "Portrait mode", "Vertical video", "Low VRAM"],
            "weaknesses": ["Smaller model size"],
        },
        "module_dependency": "video",
        "comfyui_type": "checkpoints",
    },

    "wan22_14b": {
        "display_name": "Wan 2.2 14B T2V",
        "family": "wan",
        "category": "video_gen",
        "repo": "Wan-AI/Wan2.2-14B-T2V",
        "homepage": "https://huggingface.co/Wan-AI/Wan2.2-14B-T2V",
        "license": "Apache-2.0",
        "params": "14B",
        "architecture": None,
        "status": "released",
        "files": {
            "checkpoint": {
                "folder": "checkpoints",
                "filename": "diffusion_pytorch_model.safetensors",
                "size_gb": 28.0,
            },
        },
        "total_size_gb": 28.0,
        "quantization_variants": {
            "bf16": {"vram_gb": 28, "quality": "reference", "speed": "1x"},
            "fp8": {"vram_gb": 14, "quality": "near-lossless", "speed": "1.2x"},
            "gguf_q8": {"vram_gb": 10, "quality": "high", "speed": "1.1x"},
        },
        "quantization_recommendation": {
            "ultra_low": "gguf_q8",
            "low": "gguf_q8",
            "medium": "gguf_q8",
            "high": "gguf_q8",
            "very_high": "fp8",
            "ultra": "fp8",
            "extreme": "bf16",
            "hpc": "bf16",
        },
        "inference": {
            "steps": None,
            "cfg": None,
            "cfg_scale": None,
            "resolution_range": None,
            "default_resolution": None,
        },
        "compatibility": {
            "lora": True,
            "controlnet": True,
            "controlnet_models": ["Wan Motion Control"],
            "ipadapter": True,
            "negative_prompt": True,
            "multi_reference": False,
            "image_editing": False,
            "lora_zoo": None,
            "lora_zoo_models": [],
        },
        "quality_score": {
            "elo_rank": None,
            "strengths": ["Motion control", "I2V and T2V", "Rich ecosystem"],
            "weaknesses": ["Large VRAM at bf16"],
        },
        "module_dependency": "video",
        "comfyui_type": "checkpoints",
    },

    "hunyuan_video": {
        "display_name": "HunyuanVideo",
        "family": "hunyuan",
        "category": "video_gen",
        "repo": "tencent/HunyuanVideo",
        "homepage": "https://huggingface.co/tencent/HunyuanVideo",
        "license": "Tencent Hunyuan License",
        "params": "~13B",
        "architecture": None,
        "status": "released",
        "files": {
            "checkpoint": {
                "folder": "checkpoints",
                "filename": "hunyuan_video_v1.0.safetensors",
                "size_gb": 24.0,
            },
        },
        "total_size_gb": 24.0,
        "quantization_variants": {
            "bf16": {"vram_gb": 24, "quality": "reference", "speed": "1x"},
            "fp8": {"vram_gb": 12, "quality": "near-lossless", "speed": "1.2x"},
        },
        "quantization_recommendation": {
            "ultra_low": "fp8",
            "low": "fp8",
            "medium": "fp8",
            "high": "fp8",
            "very_high": "fp8",
            "ultra": "bf16",
            "extreme": "bf16",
            "hpc": "bf16",
        },
        "inference": {
            "steps": None,
            "cfg": None,
            "cfg_scale": None,
            "resolution_range": None,
            "default_resolution": None,
        },
        "compatibility": {
            "lora": False,
            "controlnet": False,
            "controlnet_models": [],
            "ipadapter": False,
            "negative_prompt": False,
            "multi_reference": False,
            "image_editing": False,
            "lora_zoo": None,
            "lora_zoo_models": [],
        },
        "quality_score": {
            "elo_rank": None,
            "strengths": ["13B video model", "Tencent quality"],
            "weaknesses": ["Limited compatibility features"],
        },
        "module_dependency": "video",
        "comfyui_type": "checkpoints",
    },

    "cogvideox_5b": {
        "display_name": "CogVideoX 5B",
        "family": "cogvideo",
        "category": "video_gen",
        "repo": "THUDM/CogVideoX-5b",
        "homepage": "https://huggingface.co/THUDM/CogVideoX-5b",
        "license": "Apache-2.0",
        "params": "5B",
        "architecture": None,
        "status": "released",
        "files": {},
        "total_size_gb": 0,
        "quantization_variants": {
            "bf16": {"vram_gb": 12, "quality": "reference", "speed": "1x"},
            "fp8": {"vram_gb": 8, "quality": "near-lossless", "speed": "1.2x"},
        },
        "quantization_recommendation": {
            "ultra_low": "fp8",
            "low": "fp8",
            "medium": "fp8",
            "high": "bf16",
            "very_high": "bf16",
            "ultra": "bf16",
            "extreme": "bf16",
            "hpc": "bf16",
        },
        "inference": {
            "steps": None,
            "cfg": None,
            "cfg_scale": None,
            "resolution_range": None,
            "default_resolution": None,
        },
        "compatibility": {
            "lora": False,
            "controlnet": False,
            "controlnet_models": [],
            "ipadapter": False,
            "negative_prompt": False,
            "multi_reference": False,
            "image_editing": False,
            "lora_zoo": None,
            "lora_zoo_models": [],
        },
        "quality_score": {
            "elo_rank": None,
            "strengths": ["5B efficient video model", "Low VRAM viable"],
            "weaknesses": ["Smaller than 14B competitors"],
        },
        "module_dependency": "video",
        "comfyui_type": "checkpoints",
    },

    # =========================================================================
    # UPSCALE (1 model)
    # =========================================================================

    "seedvr2": {
        "display_name": "SeedVR2",
        "family": "seedvr",
        "category": "upscale",
        "repo": "TBD",
        "homepage": None,
        "license": "TBD",
        "params": None,
        "architecture": None,
        "status": "community",
        "files": {},
        "total_size_gb": 0,
        "quantization_variants": {},
        "quantization_recommendation": {},
        "inference": {
            "steps": None,
            "cfg": None,
            "cfg_scale": None,
            "resolution_range": None,
            "default_resolution": None,
        },
        "compatibility": {
            "lora": False,
            "controlnet": False,
            "controlnet_models": [],
            "ipadapter": False,
            "negative_prompt": False,
            "multi_reference": False,
            "image_editing": False,
            "lora_zoo": None,
            "lora_zoo_models": [],
        },
        "quality_score": {
            "elo_rank": None,
            "strengths": ["Community upscaling model"],
            "weaknesses": ["Repo TBD", "Community status"],
        },
        "module_dependency": "upscaling",
        "comfyui_type": "upscale_models",
    },

    # =========================================================================
    # SHARED COMPONENTS (3 models)
    # =========================================================================

    "qwen_text_encoder": {
        "display_name": "Qwen 3 4B Text Encoder",
        "family": "z_image",
        "category": "text_encoder",
        "repo": "Tongyi-MAI/Z-Image-Turbo",
        "homepage": "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo",
        "license": "Apache-2.0",
        "params": "4B",
        "architecture": "Qwen 3",
        "status": "released",
        "files": {
            "text_encoder": {
                "folder": "text_encoders",
                "filename": "qwen_3_4b.safetensors",
                "size_gb": 7.49,
            },
        },
        "total_size_gb": 7.49,
        "shared_by": ["z_image_turbo", "z_image_base"],
        "quantization_variants": {},
        "quantization_recommendation": {},
        "inference": {
            "steps": None,
            "cfg": None,
            "cfg_scale": None,
            "resolution_range": None,
            "default_resolution": None,
        },
        "compatibility": {
            "lora": False,
            "controlnet": False,
            "controlnet_models": [],
            "ipadapter": False,
            "negative_prompt": False,
            "multi_reference": False,
            "image_editing": False,
            "lora_zoo": None,
            "lora_zoo_models": [],
        },
        "quality_score": {
            "elo_rank": None,
            "strengths": ["Shared text encoder for Z-Image family", "Qwen 3 architecture"],
            "weaknesses": [],
        },
        "module_dependency": "generation",
        "comfyui_type": "text_encoders",
    },

    "flux_vae": {
        "display_name": "FLUX VAE (ae.safetensors)",
        "family": "flux",
        "category": "vae",
        "repo": "black-forest-labs/FLUX.1-dev",
        "homepage": "https://huggingface.co/black-forest-labs/FLUX.1-dev",
        "license": "Apache-2.0",
        "params": None,
        "architecture": "Autoencoder",
        "status": "released",
        "files": {
            "vae": {
                "folder": "vae",
                "filename": "ae.safetensors",
                "size_gb": 0.31,
            },
        },
        "total_size_gb": 0.31,
        "shared_by": ["flux1_dev", "flux1_schnell", "z_image_turbo", "z_image_base"],
        "quantization_variants": {},
        "quantization_recommendation": {},
        "inference": {
            "steps": None,
            "cfg": None,
            "cfg_scale": None,
            "resolution_range": None,
            "default_resolution": None,
        },
        "compatibility": {
            "lora": False,
            "controlnet": False,
            "controlnet_models": [],
            "ipadapter": False,
            "negative_prompt": False,
            "multi_reference": False,
            "image_editing": False,
            "lora_zoo": None,
            "lora_zoo_models": [],
        },
        "quality_score": {
            "elo_rank": None,
            "strengths": ["Shared across FLUX and Z-Image models", "Small footprint"],
            "weaknesses": [],
        },
        "module_dependency": "generation",
        "comfyui_type": "vae",
    },

    "cosmos_vae": {
        "display_name": "Cosmos VAE",
        "family": "cosmos",
        "category": "vae",
        "repo": "nvidia/Cosmos-1.0-VAE",
        "homepage": "https://huggingface.co/nvidia/Cosmos-1.0-VAE",
        "license": "Apache-2.0",
        "params": None,
        "architecture": "Autoencoder",
        "status": "released",
        "files": {
            "vae": {
                "folder": "vae",
                "filename": "cosmos_vae.safetensors",
                "size_gb": 0.5,
            },
        },
        "total_size_gb": 0.5,
        "quantization_variants": {},
        "quantization_recommendation": {},
        "inference": {
            "steps": None,
            "cfg": None,
            "cfg_scale": None,
            "resolution_range": None,
            "default_resolution": None,
        },
        "compatibility": {
            "lora": False,
            "controlnet": False,
            "controlnet_models": [],
            "ipadapter": False,
            "negative_prompt": False,
            "multi_reference": False,
            "image_editing": False,
            "lora_zoo": None,
            "lora_zoo_models": [],
        },
        "quality_score": {
            "elo_rank": None,
            "strengths": ["NVIDIA Cosmos VAE", "Video-capable VAE"],
            "weaknesses": [],
        },
        "module_dependency": "video",
        "comfyui_type": "vae",
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_models_by_category(category):
    """Return a dict of model entries filtered by category.

    Args:
        category: One of the keys in CATEGORIES (e.g. 'image_gen', 'video_gen').

    Returns:
        Dict of {model_key: model_entry} for models matching the category.
    """
    return {
        key: entry
        for key, entry in MODEL_REGISTRY.items()
        if entry.get("category") == category
    }


def get_models_by_family(family):
    """Return a dict of model entries filtered by family.

    Args:
        family: Family identifier (e.g. 'flux', 'z_image', 'wan').

    Returns:
        Dict of {model_key: model_entry} for models matching the family.
    """
    return {
        key: entry
        for key, entry in MODEL_REGISTRY.items()
        if entry.get("family") == family
    }


def get_recommended_quantization(model_key, vram_tier):
    """Look up the recommended quantization variant for a model at a given VRAM tier.

    Args:
        model_key: Key in MODEL_REGISTRY (e.g. 'z_image_turbo').
        vram_tier: Key in VRAM_TIERS (e.g. 'medium', 'ultra').

    Returns:
        String quantization variant name (e.g. 'fp8', 'gguf_q8'), or None if
        no recommendation exists.
    """
    entry = MODEL_REGISTRY.get(model_key)
    if entry is None:
        return None
    recommendations = entry.get("quantization_recommendation", {})
    return recommendations.get(vram_tier)


def get_download_url(model_key):
    """Generate the primary HuggingFace download URL for a model.

    Uses the repo field and the first file entry to construct a URL of the form:
    https://huggingface.co/{repo}/resolve/main/{filename}

    Args:
        model_key: Key in MODEL_REGISTRY.

    Returns:
        URL string, or None if the model has no files or no repo.
    """
    entry = MODEL_REGISTRY.get(model_key)
    if entry is None:
        return None

    repo = entry.get("repo")
    if not repo or repo == "TBD":
        return None

    files = entry.get("files", {})
    if not files:
        return None

    # Use the first file entry
    first_file = next(iter(files.values()))
    filename = first_file.get("filename", "")
    if not filename:
        return None

    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def get_all_required_files(model_key):
    """Return a flat list of all required files for a model.

    Each item is a dict with: folder, filename, size_gb, url.

    Args:
        model_key: Key in MODEL_REGISTRY.

    Returns:
        List of file info dicts, or empty list if model not found or has no files.
    """
    entry = MODEL_REGISTRY.get(model_key)
    if entry is None:
        return []

    repo = entry.get("repo", "")
    files = entry.get("files", {})
    result = []

    for _role, file_info in files.items():
        folder = file_info.get("folder", "")
        filename = file_info.get("filename", "")
        size_gb = file_info.get("size_gb", 0)

        if repo and repo != "TBD" and filename:
            url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        else:
            url = None

        result.append({
            "folder": folder,
            "filename": filename,
            "size_gb": size_gb,
            "url": url,
        })

    return result


def detect_model_key_from_filename(filename):
    """Pattern-match a filename to a registry model key.

    Searches all registered model file entries for a matching filename.
    Falls back to substring matching if no exact match is found.

    Args:
        filename: The filename to look up (e.g. 'z_image_turbo_bf16.safetensors').

    Returns:
        The model_key string if found, or None.
    """
    if not filename:
        return None

    basename = os.path.basename(filename)

    # Pass 1: exact filename match across all file entries
    for model_key, entry in MODEL_REGISTRY.items():
        for _role, file_info in entry.get("files", {}).items():
            if file_info.get("filename") == basename:
                return model_key

    # Pass 2: substring / pattern match using model key fragments
    _FILENAME_PATTERNS = {
        "z_image_turbo": ["z_image_turbo"],
        "z_image_base": ["z_image_bf16", "z_image.safetensors"],
        "z_image_edit": ["z_image_edit"],
        "flux2_dev": ["flux2-dev", "flux2_dev"],
        "flux1_dev": ["flux1-dev", "flux1_dev"],
        "flux1_schnell": ["flux1-schnell", "flux1_schnell"],
        "qwen_image_2512": ["qwen_image", "qwen-image"],
        "glm_image": ["glm_image", "glm-image"],
        "sd35_medium": ["sd3.5_medium", "sd35_medium"],
        "firered_edit_1_1": ["firered", "fire_red"],
        "helios_distilled": ["helios_distilled", "helios-distilled"],
        "helios_base": ["helios_base", "helios-base"],
        "ltx_2_3": ["ltx_2.3", "ltx-2.3", "ltx_2_3"],
        "wan22_14b": ["wan2.2", "wan22"],
        "hunyuan_video": ["hunyuan_video", "hunyuan-video"],
        "cogvideox_5b": ["cogvideox_5b", "cogvideox-5b", "cogvideox5b"],
        "seedvr2": ["seedvr2", "seedvr_2"],
        "qwen_text_encoder": ["qwen_3_4b"],
        "flux_vae": ["ae.safetensors"],
        "cosmos_vae": ["cosmos_vae"],
    }

    basename_lower = basename.lower()
    for model_key, patterns in _FILENAME_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in basename_lower:
                return model_key

    return None
