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

_REGISTRY_VERSION = "2.0.0-2026.07.05"


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
    "audio": {
        "label": "Audio Generation",
        "description": "Text-to-speech, video-to-audio, and music generation models",
    },
    "virtual_tryon": {
        "label": "Virtual Try-On",
        "description": "Clothing swap and fashion try-on models",
    },
    "face": {
        "label": "Face / Identity",
        "description": "Face swap, restoration, and animation models",
    },
}


# =============================================================================
# MODEL REGISTRY
# =============================================================================

MODEL_REGISTRY = {

    # =========================================================================
    # IMAGE GENERATION (13 models)
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
        "files": {
            "checkpoint": {
                "folder": "diffusion_models",
                "filename": "z_image_edit.safetensors",
                "size_gb": 12.0,
            },
        },
        "total_size_gb": 12.0,
        "quantization_variants": {
            "bf16": {"vram_gb": 16, "quality": "reference", "speed": "1x"},
            "fp8": {"vram_gb": 8, "quality": "near-lossless", "speed": "1.2x"},
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
        "repo": "Qwen/Qwen-Image-2512",
        "homepage": "https://huggingface.co/Qwen/Qwen-Image-2512",
        "license": "Apache-2.0",
        "params": "20B",
        "architecture": "MMDiT",
        "status": "released",
        "files": {
            "diffusion_model": {
                "folder": "diffusion_models",
                "filename": "qwen_image_2512_bf16.safetensors",
                "size_gb": 40.9,
                "alt_repo": "Comfy-Org/Qwen-Image_ComfyUI",
            },
            "text_encoder": {
                "folder": "text_encoders",
                "filename": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "size_gb": 9.38,
                "alt_repo": "Comfy-Org/Qwen-Image_ComfyUI",
            },
            "vae": {
                "folder": "vae",
                "filename": "diffusion_pytorch_model.safetensors",
                "size_gb": 0.25,
            },
        },
        "total_size_gb": 50.53,
        "quantization_variants": {
            "bf16": {"vram_gb": 48, "quality": "reference", "speed": "1x"},
            "fp8": {"vram_gb": 24, "quality": "near-lossless", "speed": "1.2x"},
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
            "steps": 50,
            "cfg": True,
            "cfg_scale": 4.0,
            "resolution_range": "1024x1024 - 1328x1328",
            "default_resolution": "1024x1024",
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
            "strengths": ["20B MMDiT model", "Excellent text rendering", "Apache-2.0 license", "LoRA support"],
            "weaknesses": ["Very large model (48 GB VRAM at bf16)", "Limited compatibility features"],
        },
        "module_dependency": "generation",
        "comfyui_type": "diffusion_models",
    },

    "glm_image": {
        "display_name": "GLM-Image",
        "family": "glm",
        "category": "image_gen",
        "repo": "zai-org/GLM-Image",
        "homepage": "https://huggingface.co/zai-org/GLM-Image",
        "license": "Apache-2.0",
        "params": "16B",
        "architecture": "Auto-regressive Dense-knowledge DiT",
        "status": "released",
        "files": {
            "transformer": {
                "folder": "diffusion_models",
                "filename": "diffusion_pytorch_model-00001-of-00003.safetensors",
                "size_gb": 13.9,
                "note": "3 sharded files totalling 13.9 GB (4.97 + 4.98 + 3.9 GB)",
            },
            "text_encoder": {
                "folder": "text_encoders",
                "filename": "model-00001-of-00002.safetensors",
                "size_gb": 18.0,
                "note": "9B auto-regressive generator (GLM-4-9B based)",
            },
            "vae": {
                "folder": "vae",
                "filename": "diffusion_pytorch_model.safetensors",
                "size_gb": 0.81,
            },
        },
        "total_size_gb": 35.8,
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
            "resolution_range": "512x512 - 1536x1536",
            "default_resolution": "1024x1024",
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
            "strengths": ["16B auto-regressive model", "Dense-knowledge architecture", "Accurate text rendering"],
            "weaknesses": ["No LoRA/ControlNet/IPAdapter support", "Large VRAM requirement"],
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

    "qwen_image_2_0": {
        "display_name": "Qwen-Image-2.0",
        "family": "qwen",
        "category": "image_gen",
        "repo": "Qwen/Qwen-Image-2.0",
        "homepage": "https://huggingface.co/Qwen/Qwen-Image-2.0",
        "license": "Check model card",
        "params": "20B",
        "architecture": "MMDiT",
        "status": "released",
        "files": {
            "diffusion_model": {
                "folder": "diffusion_models",
                "filename": "qwen_image_2_0_bf16.safetensors",
                "size_gb": 40.0,
                "alt_repo": "Comfy-Org/Qwen-Image_ComfyUI",
            },
        },
        "total_size_gb": 40.0,
        "quantization_variants": {
            "bf16": {"vram_gb": 48, "quality": "reference", "speed": "1x"},
            "fp8": {"vram_gb": 24, "quality": "near-lossless", "speed": "1.2x"},
            "gguf_q4": {"vram_gb": 16, "quality": "good", "speed": "1.3x"},
        },
        "quantization_recommendation": {
            "ultra_low": "gguf_q4",
            "low": "gguf_q4",
            "medium": "gguf_q4",
            "high": "gguf_q4",
            "very_high": "fp8",
            "ultra": "fp8",
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
            "controlnet": False,
            "controlnet_models": [],
            "ipadapter": False,
            "negative_prompt": False,
            "multi_reference": False,
            "image_editing": True,
            "lora_zoo": None,
            "lora_zoo_models": [],
        },
        "quality_score": {
            "elo_rank": None,
            "strengths": ["Unified gen + edit model", "Native ComfyUI support", "Excellent text rendering"],
            "weaknesses": ["Very large VRAM requirement at bf16"],
        },
        "module_dependency": "generation",
        "comfyui_type": "diffusion_models",
    },

    "hidream": {
        "display_name": "HiDream",
        "family": "hidream",
        "category": "image_gen",
        "repo": "HiDream-ai/HiDream-I1-Full",
        "homepage": "https://huggingface.co/HiDream-ai/HiDream-I1-Full",
        "license": "Check model card",
        "params": None,
        "architecture": "DiT",
        "status": "released",
        "files": {
            "checkpoint": {
                "folder": "checkpoints",
                "filename": "hidream_i1_full.safetensors",
                "size_gb": 12.0,
            },
        },
        "total_size_gb": 12.0,
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
            "steps": 28,
            "cfg": True,
            "cfg_scale": 5.0,
            "resolution_range": {"min": 512, "max": 2048},
            "default_resolution": {"width": 1024, "height": 1024},
        },
        "compatibility": {
            "lora": False,
            "controlnet": False,
            "controlnet_models": [],
            "ipadapter": False,
            "negative_prompt": True,
            "multi_reference": False,
            "image_editing": False,
            "lora_zoo": None,
            "lora_zoo_models": [],
        },
        "quality_score": {
            "elo_rank": None,
            "strengths": ["Native ComfyUI support", "Good quality"],
            "weaknesses": ["Limited ecosystem"],
        },
        "module_dependency": "generation",
        "comfyui_type": "checkpoints",
    },

    "lumina_image_2": {
        "display_name": "Lumina Image 2.0",
        "family": "lumina",
        "category": "image_gen",
        "repo": "Alpha-VLLM/Lumina-Image-2.0",
        "homepage": "https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0",
        "license": "Check model card",
        "params": None,
        "architecture": "DiT",
        "status": "released",
        "files": {
            "checkpoint": {
                "folder": "checkpoints",
                "filename": "lumina_image_2_0.safetensors",
                "size_gb": 10.0,
            },
        },
        "total_size_gb": 10.0,
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
            "steps": 30,
            "cfg": True,
            "cfg_scale": 4.0,
            "resolution_range": {"min": 512, "max": 2048},
            "default_resolution": {"width": 1024, "height": 1024},
        },
        "compatibility": {
            "lora": False,
            "controlnet": False,
            "controlnet_models": [],
            "ipadapter": False,
            "negative_prompt": True,
            "multi_reference": False,
            "image_editing": False,
            "lora_zoo": None,
            "lora_zoo_models": [],
        },
        "quality_score": {
            "elo_rank": None,
            "strengths": ["Native ComfyUI support", "DiT-based"],
            "weaknesses": ["Smaller community"],
        },
        "module_dependency": "generation",
        "comfyui_type": "checkpoints",
    },

    "hunyuan_image_2_1": {
        "display_name": "HunyuanImage 2.1",
        "family": "hunyuan",
        "category": "image_gen",
        "repo": "tencent/HunyuanImage-2.1",
        "homepage": "https://huggingface.co/tencent/HunyuanImage-2.1",
        "license": "Tencent Hunyuan License",
        "params": None,
        "architecture": "MoE DiT",
        "status": "released",
        "files": {
            "checkpoint": {
                "folder": "checkpoints",
                "filename": "hunyuan_image_2_1.safetensors",
                "size_gb": 30.0,
            },
        },
        "total_size_gb": 30.0,
        "quantization_variants": {
            "bf16": {"vram_gb": 48, "quality": "reference", "speed": "1x"},
            "fp8": {"vram_gb": 24, "quality": "near-lossless", "speed": "1.2x"},
        },
        "quantization_recommendation": {
            "ultra_low": "fp8",
            "low": "fp8",
            "medium": "fp8",
            "high": "fp8",
            "very_high": "fp8",
            "ultra": "fp8",
            "extreme": "bf16",
            "hpc": "bf16",
        },
        "inference": {
            "steps": 50,
            "cfg": True,
            "cfg_scale": 5.0,
            "resolution_range": {"min": 512, "max": 2048},
            "default_resolution": {"width": 1024, "height": 1024},
        },
        "compatibility": {
            "lora": False,
            "controlnet": False,
            "controlnet_models": [],
            "ipadapter": False,
            "negative_prompt": True,
            "multi_reference": False,
            "image_editing": False,
            "lora_zoo": None,
            "lora_zoo_models": [],
        },
        "quality_score": {
            "elo_rank": None,
            "strengths": ["MoE architecture", "Extremely high quality", "Native ComfyUI"],
            "weaknesses": ["Very large VRAM", "Tencent license restrictions"],
        },
        "module_dependency": "generation",
        "comfyui_type": "checkpoints",
    },

    # =========================================================================
    # IMAGE EDITING (5 models)
    # =========================================================================

    "firered_edit_1_1": {
        "display_name": "FireRed Image Edit 1.1",
        "family": "firered",
        "category": "image_edit",
        "repo": "FireRedTeam/FireRed-Image-Edit-1.1",
        "homepage": "https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.1",
        "license": "Apache-2.0",
        "params": "~20B",
        "architecture": "MM-DiT (FLUX-based)",
        "status": "released",
        "files": {
            "transformer": {
                "folder": "diffusion_models",
                "filename": "FireRed-Image-Edit-1.1-transformer.safetensors",
                "size_gb": 40.9,
                "alt_repo": "FireRedTeam/FireRed-Image-Edit-1.1-ComfyUI",
            },
            "transformer_gguf_q4_k_m": {
                "folder": "diffusion_models",
                "filename": "FireRed-Image-Edit-1.1-transformer-q4_k_m.gguf",
                "size_gb": 13.1,
                "alt_repo": "FireRedTeam/FireRed-Image-Edit-1.1-ComfyUI",
            },
            "lightning_lora": {
                "folder": "loras",
                "filename": "FireRed-Image-Edit-1.1-Lightning-8steps.safetensors",
                "size_gb": 0.85,
                "alt_repo": "FireRedTeam/FireRed-Image-Edit-1.1-ComfyUI",
                "note": "8-step distilled LoRA for faster inference",
            },
        },
        "total_size_gb": 57.7,
        "quantization_variants": {
            "bf16": {"vram_gb": 30, "quality": "reference", "speed": "1x"},
            "gguf_q4_k_m": {
                "vram_gb": 10,
                "quality": "good",
                "speed": "1.1x",
                "alt_repo": "FireRedTeam/FireRed-Image-Edit-1.1-ComfyUI",
            },
            "gguf_q8": {
                "vram_gb": 16,
                "quality": "high",
                "speed": "1.1x",
                "alt_repo": "drbaph/FireRed-Image-Edit-1.0_ComfyUI_Quants",
            },
        },
        "quantization_recommendation": {
            "ultra_low": "gguf_q4_k_m",
            "low": "gguf_q4_k_m",
            "medium": "gguf_q4_k_m",
            "high": "gguf_q8",
            "very_high": "gguf_q8",
            "ultra": "bf16",
            "extreme": "bf16",
            "hpc": "bf16",
        },
        "inference": {
            "steps": 28,
            "cfg": False,
            "cfg_scale": 1.0,
            "resolution_range": "512x512 - 1536x1536",
            "default_resolution": "1024x1024",
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
            "strengths": ["SOTA instruction-based editing", "Rich LoRA zoo", "Multi-edit support", "Lightning 8-step LoRA"],
            "weaknesses": ["Large VRAM at bf16 (30 GB)"],
        },
        "module_dependency": "inpainting",
        "comfyui_type": "diffusion_models",
    },

    "omnigen2": {
        "display_name": "OmniGen 2",
        "family": "omnigen",
        "category": "image_edit",
        "repo": "Shitao/OmniGen2",
        "homepage": "https://huggingface.co/Shitao/OmniGen2",
        "license": "Check model card",
        "params": None,
        "architecture": "Multi-modal DiT",
        "status": "released",
        "files": {
            "checkpoint": {
                "folder": "checkpoints",
                "filename": "omnigen2.safetensors",
                "size_gb": 12.0,
            },
        },
        "total_size_gb": 12.0,
        "quantization_variants": {
            "bf16": {"vram_gb": 16, "quality": "reference", "speed": "1x"},
            "fp8": {"vram_gb": 10, "quality": "near-lossless", "speed": "1.2x"},
            "gguf_q8": {"vram_gb": 8, "quality": "high", "speed": "1.1x"},
        },
        "quantization_recommendation": {
            "ultra_low": "gguf_q8",
            "low": "gguf_q8",
            "medium": "gguf_q8",
            "high": "fp8",
            "very_high": "bf16",
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
            "lora": False,
            "controlnet": False,
            "controlnet_models": [],
            "ipadapter": False,
            "negative_prompt": False,
            "multi_reference": True,
            "image_editing": True,
            "lora_zoo": None,
            "lora_zoo_models": [],
        },
        "quality_score": {
            "elo_rank": None,
            "strengths": ["Multi-modal gen + edit", "Native ComfyUI", "Multi-image reference input"],
            "weaknesses": ["Newer model, limited community testing"],
        },
        "module_dependency": "inpainting",
        "comfyui_type": "checkpoints",
    },

    "flux_kontext": {
        "display_name": "Flux Kontext",
        "family": "flux",
        "category": "image_edit",
        "repo": "black-forest-labs/FLUX.1-Kontext-dev",
        "homepage": "https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev",
        "license": "FLUX Non-Commercial License",
        "params": "12B",
        "architecture": "Rectified Flow Transformer",
        "status": "released",
        "files": {
            "checkpoint": {
                "folder": "checkpoints",
                "filename": "flux1-kontext-dev.safetensors",
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
            "controlnet": False,
            "controlnet_models": [],
            "ipadapter": False,
            "negative_prompt": True,
            "multi_reference": True,
            "image_editing": True,
            "lora_zoo": None,
            "lora_zoo_models": [],
        },
        "quality_score": {
            "elo_rank": None,
            "strengths": ["Context-aware editing", "Native ComfyUI", "FLUX ecosystem LoRA compatible"],
            "weaknesses": ["Non-commercial license"],
        },
        "module_dependency": "inpainting",
        "comfyui_type": "checkpoints",
    },

    "hidream_e1_1": {
        "display_name": "HiDream E1.1",
        "family": "hidream",
        "category": "image_edit",
        "repo": "HiDream-ai/HiDream-E1.1",
        "homepage": "https://huggingface.co/HiDream-ai/HiDream-E1.1",
        "license": "Check model card",
        "params": None,
        "architecture": "DiT",
        "status": "released",
        "files": {
            "checkpoint": {
                "folder": "checkpoints",
                "filename": "hidream_e1_1.safetensors",
                "size_gb": 12.0,
            },
        },
        "total_size_gb": 12.0,
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
            "steps": 28,
            "cfg": True,
            "cfg_scale": 5.0,
            "resolution_range": {"min": 512, "max": 2048},
            "default_resolution": {"width": 1024, "height": 1024},
        },
        "compatibility": {
            "lora": False,
            "controlnet": False,
            "controlnet_models": [],
            "ipadapter": False,
            "negative_prompt": True,
            "multi_reference": False,
            "image_editing": True,
            "lora_zoo": None,
            "lora_zoo_models": [],
        },
        "quality_score": {
            "elo_rank": None,
            "strengths": ["Native ComfyUI editing", "HiDream ecosystem"],
            "weaknesses": ["Smaller community"],
        },
        "module_dependency": "inpainting",
        "comfyui_type": "checkpoints",
    },

    # =========================================================================
    # VIDEO GENERATION (10 models)
    # =========================================================================

    "helios_distilled": {
        "display_name": "Helios Distilled",
        "family": "helios",
        "category": "video_gen",
        "repo": "BestWishYsh/Helios-Distilled",
        "homepage": "https://huggingface.co/BestWishYsh/Helios-Distilled",
        "license": "Apache-2.0",
        "params": "14B",
        "architecture": "Wan2.1-based Adversarial Hierarchical Distilled Transformer",
        "status": "brand_new",
        "released": "2026-03-04",
        "files": {
            "transformer": {
                "folder": "diffusion_models",
                "filename": "diffusion_pytorch_model.safetensors",
                "size_gb": 27.0,
                "note": "Diffusers format, may be sharded; total repo ~138 GB with all components",
            },
            "transformer_ode": {
                "folder": "diffusion_models",
                "filename": "transformer_ode/diffusion_pytorch_model.safetensors",
                "size_gb": 27.0,
                "note": "ODE branch for Pyramid Unified Predictor Corrector",
            },
            "text_encoder": {
                "folder": "text_encoders",
                "filename": "model.safetensors",
                "size_gb": 18.0,
                "note": "Shared Wan2.1 text encoder",
            },
            "vae": {
                "folder": "vae",
                "filename": "diffusion_pytorch_model.safetensors",
                "size_gb": 0.3,
            },
        },
        "total_size_gb": 138.0,
        "quantization_variants": {
            "bf16": {"vram_gb": 28, "quality": "reference", "speed": "1x"},
            "fp8": {"vram_gb": 16, "quality": "near-lossless", "speed": "1.2x"},
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
            "steps": 3,
            "cfg": False,
            "cfg_scale": 1.0,
            "resolution_range": "480p - 720p",
            "default_resolution": "720p",
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
            "strengths": ["14B distilled video model", "Only 3 inference steps", "19.5 FPS on H100", "Brand new release (2026-03-04)"],
            "weaknesses": ["Newly released, limited community testing", "Very large download (138 GB)"],
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
        "architecture": "Wan2.1-based Transformer with Unified History Injection",
        "status": "released",
        "files": {
            "transformer": {
                "folder": "diffusion_models",
                "filename": "diffusion_pytorch_model.safetensors",
                "size_gb": 27.0,
                "note": "Diffusers format, may be sharded; total repo ~138 GB with all components",
            },
            "text_encoder": {
                "folder": "text_encoders",
                "filename": "model.safetensors",
                "size_gb": 18.0,
                "note": "Shared Wan2.1 text encoder",
            },
            "vae": {
                "folder": "vae",
                "filename": "diffusion_pytorch_model.safetensors",
                "size_gb": 0.3,
            },
        },
        "total_size_gb": 138.0,
        "quantization_variants": {
            "bf16": {"vram_gb": 28, "quality": "reference", "speed": "1x"},
            "fp8": {"vram_gb": 16, "quality": "near-lossless", "speed": "1.2x"},
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
            "steps": 50,
            "cfg": True,
            "cfg_scale": 5.0,
            "resolution_range": "480p - 720p",
            "default_resolution": "720p",
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
            "strengths": ["14B video generation model", "Minute-scale video generation", "T2V/I2V/V2V support"],
            "weaknesses": ["Very large download (138 GB)", "High VRAM requirement"],
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
        "params": "22B",
        "architecture": "Diffusion Transformer (rebuilt VAE + enhanced text connector)",
        "status": "released",
        "files": {
            "checkpoint": {
                "folder": "diffusion_models",
                "filename": "ltx-2.3-22b-dev.safetensors",
                "size_gb": 46.1,
            },
            "checkpoint_distilled": {
                "folder": "diffusion_models",
                "filename": "ltx-2.3-22b-distilled.safetensors",
                "size_gb": 46.1,
                "note": "8-step distilled variant for faster inference",
            },
            "checkpoint_fp8": {
                "folder": "diffusion_models",
                "filename": "ltx-2.3-22b-dev-fp8.safetensors",
                "size_gb": 29.1,
                "alt_repo": "Lightricks/LTX-2.3-fp8",
            },
            "distilled_lora": {
                "folder": "loras",
                "filename": "ltx-2.3-22b-distilled-lora-384.safetensors",
                "size_gb": 7.61,
                "note": "LoRA adapter for distillation behavior on dev model",
            },
            "spatial_upscaler": {
                "folder": "diffusion_models",
                "filename": "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
                "size_gb": 1.0,
                "note": "2x spatial upsampler for two-stage pipelines",
            },
        },
        "total_size_gb": 46.1,
        "quantization_variants": {
            "bf16": {"vram_gb": 48, "quality": "reference", "speed": "1x"},
            "fp8": {
                "vram_gb": 24,
                "quality": "near-lossless",
                "speed": "1.2x",
                "alt_repo": "Lightricks/LTX-2.3-fp8",
            },
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
            "steps": 50,
            "cfg": True,
            "cfg_scale": 3.5,
            "resolution_range": "480p - 1080p",
            "default_resolution": "720p",
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
            "strengths": ["22B large video model", "Audio generation", "Rebuilt VAE for sharper output", "LoRA training support"],
            "weaknesses": ["Very large model (46.1 GB bf16)", "High VRAM requirement (48 GB bf16)"],
        },
        "module_dependency": "video",
        "comfyui_type": "diffusion_models",
    },

    "wan22_14b": {
        "display_name": "Wan 2.2 14B T2V",
        "family": "wan",
        "category": "video_gen",
        "repo": "Wan-AI/Wan2.2-14B-T2V",
        "homepage": "https://huggingface.co/Wan-AI/Wan2.2-14B-T2V",
        "license": "Apache-2.0",
        "params": "14B",
        "architecture": "MoE Diffusion Transformer",
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
        "architecture": "DiT Video Transformer",
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
        "architecture": "Expert Transformer + 3D Causal VAE + 3D-RoPE",
        "status": "released",
        "files": {
            "transformer": {
                "folder": "diffusion_models",
                "filename": "diffusion_pytorch_model-00001-of-00002.safetensors",
                "size_gb": 11.15,
                "note": "2 sharded files (9.93 + 1.22 GB)",
            },
            "vae": {
                "folder": "vae",
                "filename": "diffusion_pytorch_model.safetensors",
                "size_gb": 0.86,
                "note": "3D Causal VAE for joint spatial-temporal compression",
            },
            "text_encoder": {
                "folder": "text_encoders",
                "filename": "model-00001-of-00002.safetensors",
                "size_gb": 9.4,
                "note": "T5-XXL text encoder",
            },
        },
        "total_size_gb": 21.5,
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
            "steps": 50,
            "cfg": True,
            "cfg_scale": 6.0,
            "resolution_range": "480x720",
            "default_resolution": "480x720",
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
            "strengths": ["5B efficient video model", "Low VRAM viable", "Expert Transformer with adaptive LayerNorm"],
            "weaknesses": ["Smaller than 14B competitors", "Limited to 480x720 resolution"],
        },
        "module_dependency": "video",
        "comfyui_type": "diffusion_models",
    },

    "wan22_ti2v_5b": {
        "display_name": "Wan 2.2 5B TI2V",
        "family": "wan",
        "category": "video_gen",
        "repo": "Wan-AI/Wan2.2-T2V-5B",
        "homepage": "https://huggingface.co/Wan-AI/Wan2.2-T2V-5B",
        "license": "Apache-2.0",
        "params": "5B",
        "architecture": "Wan DiT",
        "status": "released",
        "files": {
            "checkpoint": {
                "folder": "diffusion_models",
                "filename": "wan2.2_t2v_5b.safetensors",
                "size_gb": 10.0,
            },
        },
        "total_size_gb": 10.0,
        "quantization_variants": {
            "bf16": {"vram_gb": 12, "quality": "reference", "speed": "1x"},
            "fp8": {"vram_gb": 8, "quality": "near-lossless", "speed": "1.2x"},
            "gguf_q8": {"vram_gb": 6, "quality": "high", "speed": "1.1x"},
        },
        "quantization_recommendation": {
            "ultra_low": "gguf_q8",
            "low": "gguf_q8",
            "medium": "gguf_q8",
            "high": "fp8",
            "very_high": "bf16",
            "ultra": "bf16",
            "extreme": "bf16",
            "hpc": "bf16",
        },
        "inference": {
            "steps": 30,
            "cfg": True,
            "cfg_scale": 5.0,
            "resolution_range": "480p - 720p",
            "default_resolution": "480p",
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
            "strengths": ["Sweet spot for 12GB GPUs", "T2V and I2V", "Apache-2.0", "Rich ecosystem"],
            "weaknesses": ["Lower quality than 14B"],
        },
        "module_dependency": "video",
        "comfyui_type": "diffusion_models",
    },

    "wan21_t2v_1_3b": {
        "display_name": "Wan 2.1 1.3B T2V",
        "family": "wan",
        "category": "video_gen",
        "repo": "Wan-AI/Wan2.1-T2V-1.3B",
        "homepage": "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B",
        "license": "Apache-2.0",
        "params": "1.3B",
        "architecture": "Wan DiT",
        "status": "released",
        "files": {
            "checkpoint": {
                "folder": "diffusion_models",
                "filename": "wan2.1_t2v_1.3b.safetensors",
                "size_gb": 2.6,
            },
        },
        "total_size_gb": 2.6,
        "quantization_variants": {
            "bf16": {"vram_gb": 8, "quality": "reference", "speed": "1x"},
            "fp8": {"vram_gb": 6, "quality": "near-lossless", "speed": "1.2x"},
        },
        "quantization_recommendation": {
            "ultra_low": "fp8",
            "low": "fp8",
            "medium": "bf16",
            "high": "bf16",
            "very_high": "bf16",
            "ultra": "bf16",
            "extreme": "bf16",
            "hpc": "bf16",
        },
        "inference": {
            "steps": 30,
            "cfg": True,
            "cfg_scale": 5.0,
            "resolution_range": "480p",
            "default_resolution": "480p",
        },
        "compatibility": {
            "lora": True,
            "controlnet": False,
            "controlnet_models": [],
            "ipadapter": False,
            "negative_prompt": True,
            "multi_reference": False,
            "image_editing": False,
            "lora_zoo": None,
            "lora_zoo_models": [],
        },
        "quality_score": {
            "elo_rank": None,
            "strengths": ["Runs on 8GB GPU", "Apache-2.0", "Beginner-friendly"],
            "weaknesses": ["480p only", "Lower quality"],
        },
        "module_dependency": "video",
        "comfyui_type": "diffusion_models",
    },

    "ltx_video_2b": {
        "display_name": "LTX-Video 2B",
        "family": "ltx",
        "category": "video_gen",
        "repo": "Lightricks/LTX-Video",
        "homepage": "https://huggingface.co/Lightricks/LTX-Video",
        "license": "Check Lightricks terms",
        "params": "2B",
        "architecture": "Diffusion Transformer",
        "status": "released",
        "files": {
            "checkpoint": {
                "folder": "diffusion_models",
                "filename": "ltx-video-2b-v0.9.5.safetensors",
                "size_gb": 4.0,
            },
            "checkpoint_q8": {
                "folder": "diffusion_models",
                "filename": "ltx-video-2b-v0.9.5-q8.safetensors",
                "size_gb": 2.5,
                "note": "Q8 quantized, runs on 8GB GPUs",
            },
        },
        "total_size_gb": 4.0,
        "quantization_variants": {
            "bf16": {"vram_gb": 12, "quality": "reference", "speed": "1x"},
            "q8": {"vram_gb": 8, "quality": "near-lossless", "speed": "1.1x"},
        },
        "quantization_recommendation": {
            "ultra_low": "q8",
            "low": "q8",
            "medium": "q8",
            "high": "bf16",
            "very_high": "bf16",
            "ultra": "bf16",
            "extreme": "bf16",
            "hpc": "bf16",
        },
        "inference": {
            "steps": 30,
            "cfg": True,
            "cfg_scale": 3.0,
            "resolution_range": "480p - 720p",
            "default_resolution": "720x480",
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
            "strengths": ["Built into ComfyUI core", "Runs on 8GB (Q8)", "Fast inference", "LoRA support"],
            "weaknesses": ["Lower quality than larger models"],
        },
        "module_dependency": "video",
        "comfyui_type": "diffusion_models",
    },

    "hunyuan_video_1_5": {
        "display_name": "HunyuanVideo 1.5",
        "family": "hunyuan",
        "category": "video_gen",
        "repo": "tencent/HunyuanVideo-1.5",
        "homepage": "https://huggingface.co/tencent/HunyuanVideo-1.5",
        "license": "Tencent Hunyuan License",
        "params": "~13B",
        "architecture": "Dual-stream DiT",
        "status": "released",
        "files": {
            "checkpoint": {
                "folder": "checkpoints",
                "filename": "hunyuan_video_v1.5.safetensors",
                "size_gb": 26.0,
            },
        },
        "total_size_gb": 26.0,
        "quantization_variants": {
            "bf16": {"vram_gb": 80, "quality": "reference", "speed": "1x"},
            "fp8": {"vram_gb": 45, "quality": "near-lossless", "speed": "1.2x"},
        },
        "quantization_recommendation": {
            "ultra_low": "fp8",
            "low": "fp8",
            "medium": "fp8",
            "high": "fp8",
            "very_high": "fp8",
            "ultra": "fp8",
            "extreme": "fp8",
            "hpc": "bf16",
        },
        "inference": {
            "steps": 50,
            "cfg": True,
            "cfg_scale": 6.0,
            "resolution_range": "540p - 720p",
            "default_resolution": "720p",
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
            "strengths": ["Top-tier video quality", "Native ComfyUI", "720p 129-frame output"],
            "weaknesses": ["Extremely high VRAM (45GB FP8)", "Tencent license"],
        },
        "module_dependency": "video",
        "comfyui_type": "checkpoints",
    },

    # =========================================================================
    # UPSCALE (1 model)
    # =========================================================================

    "seedvr2": {
        "display_name": "SeedVR2 7B",
        "family": "seedvr",
        "category": "upscale",
        "repo": "ByteDance-Seed/SeedVR2-7B",
        "homepage": "https://huggingface.co/ByteDance-Seed/SeedVR2-7B",
        "license": "Apache-2.0",
        "params": "7B",
        "architecture": "Diffusion Adversarial Post-Training with Adaptive Window Attention",
        "status": "released",
        "files": {
            "checkpoint_fp16": {
                "folder": "upscale_models",
                "filename": "seedvr2_ema_7b_fp16.safetensors",
                "size_gb": 16.5,
                "alt_repo": "numz/SeedVR2_comfyUI",
            },
            "checkpoint_fp8": {
                "folder": "upscale_models",
                "filename": "seedvr2_ema_7b_fp8_e4m3fn.safetensors",
                "size_gb": 8.24,
                "alt_repo": "numz/SeedVR2_comfyUI",
            },
            "checkpoint_sharp_fp16": {
                "folder": "upscale_models",
                "filename": "seedvr2_ema_7b_sharp_fp16.safetensors",
                "size_gb": 16.5,
                "alt_repo": "numz/SeedVR2_comfyUI",
                "note": "Sharper variant for detail-focused upscaling",
            },
        },
        "total_size_gb": 16.5,
        "quantization_variants": {
            "fp16": {"vram_gb": 18, "quality": "reference", "speed": "1x"},
            "fp8": {"vram_gb": 10, "quality": "near-lossless", "speed": "1.3x"},
        },
        "quantization_recommendation": {
            "ultra_low": "fp8",
            "low": "fp8",
            "medium": "fp8",
            "high": "fp8",
            "very_high": "fp16",
            "ultra": "fp16",
            "extreme": "fp16",
            "hpc": "fp16",
        },
        "inference": {
            "steps": 1,
            "cfg": False,
            "cfg_scale": 1.0,
            "resolution_range": "up to 4K",
            "default_resolution": "2x input",
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
            "strengths": ["One-step super resolution", "Image and video upscaling", "ICLR 2026 paper", "Apache-2.0 license"],
            "weaknesses": ["May oversharpen lightly degraded inputs", "Not robust to heavy degradations"],
        },
        "module_dependency": "upscaling",
        "comfyui_type": "upscale_models",
    },

    # =========================================================================
    # AUDIO (4 models)
    # =========================================================================

    "mmaudio": {
        "display_name": "MMAudio",
        "family": "mmaudio",
        "category": "audio",
        "repo": "hkchengrex/MMAudio",
        "homepage": "https://huggingface.co/hkchengrex/MMAudio",
        "license": "Check model card",
        "params": None,
        "architecture": "Multi-modal Audio Transformer",
        "status": "released",
        "files": {
            "checkpoint": {
                "folder": "custom_nodes/MMAudio",
                "filename": "mmaudio.safetensors",
                "size_gb": 2.0,
            },
        },
        "total_size_gb": 2.0,
        "quantization_variants": {
            "bf16": {"vram_gb": 8, "quality": "reference", "speed": "1x"},
        },
        "quantization_recommendation": {
            "ultra_low": "bf16",
            "low": "bf16",
            "medium": "bf16",
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
            "strengths": ["Video-to-audio generation", "Synced sound effects"],
            "weaknesses": ["Requires video input"],
        },
        "module_dependency": None,
        "comfyui_type": "custom_node",
        "custom_node_repo": "https://github.com/hkchengrex/MMAudio",
    },

    "cosyvoice3": {
        "display_name": "CosyVoice3",
        "family": "cosyvoice",
        "category": "audio",
        "repo": "FunAudioLLM/CosyVoice3-0.5B",
        "homepage": "https://huggingface.co/FunAudioLLM/CosyVoice3-0.5B",
        "license": "Apache-2.0",
        "params": "0.5B",
        "architecture": "Audio LLM",
        "status": "released",
        "files": {
            "checkpoint": {
                "folder": "custom_nodes/TTS",
                "filename": "cosyvoice3.safetensors",
                "size_gb": 1.5,
            },
        },
        "total_size_gb": 1.5,
        "quantization_variants": {
            "bf16": {"vram_gb": 4, "quality": "reference", "speed": "1x"},
        },
        "quantization_recommendation": {
            "ultra_low": "bf16",
            "low": "bf16",
            "medium": "bf16",
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
            "strengths": ["Apache-2.0 commercial safe", "Emotion control", "Voice cloning"],
            "weaknesses": ["Requires TTS Audio Suite custom node"],
        },
        "module_dependency": None,
        "comfyui_type": "custom_node",
        "custom_node_repo": "https://github.com/ShmuelRonen/ComfyUI-TTS_Audio_Suite",
    },

    "qwen3_tts": {
        "display_name": "Qwen3-TTS",
        "family": "qwen",
        "category": "audio",
        "repo": "Qwen/Qwen3-TTS-0.6B",
        "homepage": "https://huggingface.co/Qwen/Qwen3-TTS-0.6B",
        "license": "Check model card",
        "params": "0.6B",
        "architecture": "Audio LLM",
        "status": "released",
        "files": {
            "checkpoint": {
                "folder": "custom_nodes/TTS",
                "filename": "qwen3_tts_0.6b.safetensors",
                "size_gb": 1.2,
            },
        },
        "total_size_gb": 1.2,
        "quantization_variants": {
            "bf16": {"vram_gb": 3, "quality": "reference", "speed": "1x"},
        },
        "quantization_recommendation": {
            "ultra_low": "bf16",
            "low": "bf16",
            "medium": "bf16",
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
            "strengths": ["Very low VRAM (3GB)", "High quality TTS", "Emotion/speed control"],
            "weaknesses": ["Requires TTS Audio Suite custom node"],
        },
        "module_dependency": None,
        "comfyui_type": "custom_node",
        "custom_node_repo": "https://github.com/ShmuelRonen/ComfyUI-TTS_Audio_Suite",
    },

    "ace_step_1_5": {
        "display_name": "ACE-Step 1.5",
        "family": "ace",
        "category": "audio",
        "repo": "ACE-Step/ACE-Step-v1-5",
        "homepage": "https://huggingface.co/ACE-Step/ACE-Step-v1-5",
        "license": "Check model card",
        "params": None,
        "architecture": "Music Generation Transformer",
        "status": "released",
        "files": {
            "checkpoint": {
                "folder": "custom_nodes/ACE-Step",
                "filename": "ace_step_v1.5.safetensors",
                "size_gb": 3.0,
            },
        },
        "total_size_gb": 3.0,
        "quantization_variants": {
            "bf16": {"vram_gb": 8, "quality": "reference", "speed": "1x"},
        },
        "quantization_recommendation": {
            "ultra_low": "bf16",
            "low": "bf16",
            "medium": "bf16",
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
            "strengths": ["Music generation", "Lyrics-to-song", "Multiple genres"],
            "weaknesses": ["Specialized use case"],
        },
        "module_dependency": None,
        "comfyui_type": "custom_node",
    },

    # =========================================================================
    # VIRTUAL TRY-ON (2 models)
    # =========================================================================

    "catvton": {
        "display_name": "CatVTON",
        "family": "catvton",
        "category": "virtual_tryon",
        "repo": "zhengchong/CatVTON",
        "homepage": "https://huggingface.co/zhengchong/CatVTON",
        "license": "Check model card (ICLR 2025)",
        "params": None,
        "architecture": "Lightweight try-on network",
        "status": "released",
        "files": {
            "checkpoint": {
                "folder": "custom_nodes/CatVTON",
                "filename": "catvton.safetensors",
                "size_gb": 2.0,
            },
        },
        "total_size_gb": 2.0,
        "quantization_variants": {
            "bf16": {"vram_gb": 8, "quality": "reference", "speed": "1x"},
        },
        "quantization_recommendation": {
            "ultra_low": "bf16",
            "low": "bf16",
            "medium": "bf16",
            "high": "bf16",
            "very_high": "bf16",
            "ultra": "bf16",
            "extreme": "bf16",
            "hpc": "bf16",
        },
        "inference": {
            "steps": 30,
            "cfg": True,
            "cfg_scale": 2.5,
            "resolution_range": "768x1024",
            "default_resolution": "768x1024",
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
            "strengths": ["<8GB VRAM at 1024x768", "ICLR 2025 paper", "ComfyUI node available"],
            "weaknesses": ["Single garment at a time"],
        },
        "module_dependency": "dressing",
        "comfyui_type": "custom_node",
        "custom_node_repo": "https://github.com/chflame163/ComfyUI-CatVTON",
    },

    "idm_vton": {
        "display_name": "IDM-VTON",
        "family": "idm_vton",
        "category": "virtual_tryon",
        "repo": "yisol/IDM-VTON",
        "homepage": "https://huggingface.co/yisol/IDM-VTON",
        "license": "Check model card",
        "params": None,
        "architecture": "Dual-stream try-on network",
        "status": "released",
        "files": {
            "checkpoint": {
                "folder": "custom_nodes/IDM-VTON",
                "filename": "idm_vton.safetensors",
                "size_gb": 4.0,
            },
        },
        "total_size_gb": 4.0,
        "quantization_variants": {
            "bf16": {"vram_gb": 12, "quality": "reference", "speed": "1x"},
        },
        "quantization_recommendation": {
            "ultra_low": "bf16",
            "low": "bf16",
            "medium": "bf16",
            "high": "bf16",
            "very_high": "bf16",
            "ultra": "bf16",
            "extreme": "bf16",
            "hpc": "bf16",
        },
        "inference": {
            "steps": 30,
            "cfg": True,
            "cfg_scale": 2.5,
            "resolution_range": "768x1024",
            "default_resolution": "768x1024",
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
            "strengths": ["High fidelity try-on", "Good garment detail preservation"],
            "weaknesses": ["Higher VRAM than CatVTON"],
        },
        "module_dependency": "dressing",
        "comfyui_type": "custom_node",
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
        "qwen_image_2512": ["qwen_image_2512", "qwen-image-2512"],
        "qwen_image_2_0": ["qwen_image_2_0", "qwen-image-2.0", "qwen_image_2.0"],
        "glm_image": ["glm_image", "glm-image"],
        "sd35_medium": ["sd3.5_medium", "sd35_medium"],
        "hidream": ["hidream_i1", "hidream-i1"],
        "lumina_image_2": ["lumina_image", "lumina-image"],
        "hunyuan_image_2_1": ["hunyuan_image", "hunyuan-image"],
        "firered_edit_1_1": ["firered", "fire_red"],
        "omnigen2": ["omnigen2", "omnigen_2"],
        "flux_kontext": ["kontext", "flux1-kontext"],
        "hidream_e1_1": ["hidream_e1", "hidream-e1"],
        "helios_distilled": ["helios_distilled", "helios-distilled"],
        "helios_base": ["helios_base", "helios-base"],
        "ltx_2_3": ["ltx_2.3", "ltx-2.3", "ltx_2_3"],
        "ltx_video_2b": ["ltx-video-2b", "ltx_video_2b"],
        "wan22_14b": ["wan2.2-14b", "wan22_14b"],
        "wan22_ti2v_5b": ["wan2.2_t2v_5b", "wan2.2-5b"],
        "wan21_t2v_1_3b": ["wan2.1_t2v_1.3b", "wan2.1-1.3b"],
        "hunyuan_video": ["hunyuan_video_v1.0", "hunyuan-video-v1.0"],
        "hunyuan_video_1_5": ["hunyuan_video_v1.5", "hunyuan-video-v1.5"],
        "cogvideox_5b": ["cogvideox_5b", "cogvideox-5b", "cogvideox5b"],
        "seedvr2": ["seedvr2", "seedvr_2"],
        "mmaudio": ["mmaudio"],
        "cosyvoice3": ["cosyvoice3", "cosyvoice_3"],
        "qwen3_tts": ["qwen3_tts", "qwen3-tts"],
        "ace_step_1_5": ["ace_step", "ace-step"],
        "catvton": ["catvton"],
        "idm_vton": ["idm_vton", "idm-vton"],
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


def get_image_model_choices():
    """Return sorted list of registry keys for image generation and editing models."""
    return sorted(
        k for k, v in MODEL_REGISTRY.items()
        if v.get("category") in ("image_gen", "image_edit")
    )


def get_video_model_choices():
    """Return sorted list of registry keys for video generation models."""
    return sorted(
        k for k, v in MODEL_REGISTRY.items()
        if v.get("category") == "video_gen"
    )


def get_upscale_model_choices():
    """Return sorted list of registry keys for upscale models."""
    return sorted(
        k for k, v in MODEL_REGISTRY.items()
        if v.get("category") == "upscale"
    )


def get_audio_model_choices():
    """Return sorted list of registry keys for audio models."""
    return sorted(
        k for k, v in MODEL_REGISTRY.items()
        if v.get("category") == "audio"
    )


def get_tryon_model_choices():
    """Return sorted list of registry keys for virtual try-on models."""
    return sorted(
        k for k, v in MODEL_REGISTRY.items()
        if v.get("category") == "virtual_tryon"
    )


def get_model_display_name(key):
    """Return human-readable display name for a registry key."""
    entry = MODEL_REGISTRY.get(key, {})
    return entry.get("display_name", key)


def get_models_for_vram(vram_gb):
    """Return models that can run at a given VRAM level.

    Args:
        vram_gb: Available VRAM in GB.

    Returns:
        Dict of {model_key: recommended_quantization} for runnable models.
    """
    tier = None
    for tier_key, tier_info in VRAM_TIERS.items():
        if tier_info["min_gb"] <= vram_gb < tier_info["max_gb"]:
            tier = tier_key
            break
    if tier is None:
        tier = "hpc"

    result = {}
    for model_key, entry in MODEL_REGISTRY.items():
        rec = entry.get("quantization_recommendation", {})
        quant = rec.get(tier)
        if quant:
            variant = entry.get("quantization_variants", {}).get(quant, {})
            if variant and variant.get("vram_gb", 999) <= vram_gb:
                result[model_key] = quant
    return result


def get_commercial_safe_models():
    """Return model keys with Apache-2.0 or equivalent commercial licenses."""
    safe_licenses = {"Apache-2.0", "MIT", "BSD"}
    return sorted(
        k for k, v in MODEL_REGISTRY.items()
        if v.get("license") in safe_licenses
    )
