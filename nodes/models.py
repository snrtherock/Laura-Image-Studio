"""
Laura Image Studio - Universal Model Support
Multi-model generation nodes supporting SDXL, Flux, Wan 2.2, SD 1.5, SD 3, and more
"""

import torch
from PIL import Image
import numpy as np
import folder_paths

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# ============== MODEL TYPE DETECTOR ==============
class ModelTypeDetector:
    """Detect model type from filename"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_type",)
    FUNCTION = "detect_type"
    CATEGORY = "Laura Studio/Models"
    DESCRIPTION = "Auto-detect model type from filename"

    def detect_type(self, model_name):
        name_lower = model_name.lower()

        # FLUX.2 models (check before flux1 since "flux" is substring)
        if any(x in name_lower for x in ["flux2", "flux-2", "flux.2", "flux_2"]):
            if "schnell" in name_lower:
                model_type = "flux2_schnell"
            else:
                model_type = "flux2"
        # FLUX.1 models
        elif any(x in name_lower for x in ["flux1", "flux-1", "flux.1", "flux_1", "flux_dev", "flux_schnell"]):
            if "schnell" in name_lower:
                model_type = "flux_schnell"
            else:
                model_type = "flux"
        elif "flux" in name_lower:
            model_type = "flux"
        # Z-Image (Alibaba)
        elif any(x in name_lower for x in ["zimage", "z-image", "z_image"]):
            if "turbo" in name_lower:
                model_type = "zimage_turbo"
            elif "edit" in name_lower:
                model_type = "zimage_edit"
            else:
                model_type = "zimage"
        # Qwen-Image (Alibaba)
        elif any(x in name_lower for x in ["qwen", "qwen-image", "qwen_image"]):
            model_type = "qwen"
        # SD 3.5 (check before sd3)
        elif any(x in name_lower for x in ["sd3.5", "sd35", "sd3_5"]):
            if "medium" in name_lower:
                model_type = "sd35_medium"
            else:
                model_type = "sd35"
        # SD 3.0
        elif any(x in name_lower for x in ["sd3", "sd_3", "stable_diffusion_3"]):
            model_type = "sd3"
        # SDXL models
        elif any(x in name_lower for x in ["sdxl", "juggernaut", "realistic", "dreamshaper", "sd_xl", "sdxl_base"]):
            model_type = "sdxl"
        # Wan 2.1
        elif any(x in name_lower for x in ["wan2.1", "wan21", "wan_21"]):
            model_type = "wan21"
        # Wan 2.2
        elif any(x in name_lower for x in ["wan", "wan2", "wan_2", "zoriana"]):
            model_type = "wan22"
        # SD 1.5 models
        elif any(x in name_lower for x in ["sd15", "sd_15", "v1-5", "v2-1"]):
            model_type = "sd15"
        # Playground
        elif "playground" in name_lower:
            model_type = "playground"
        # Pixart
        elif "pixart" in name_lower:
            model_type = "pixart"
        # Aura Flow
        elif "aura" in name_lower:
            model_type = "aura"
        # Kolors
        elif "kolors" in name_lower:
            model_type = "kolors"
        else:
            model_type = "unknown"

        return (model_type,)


# ============== UNIVERSAL MODEL LOADER ==============
class UniversalModelLoader:
    """Universal model loader with model type detection"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("checkpoints"),),
                "model_type": (["auto", "sdxl", "flux", "flux_schnell", "flux2", "flux2_schnell", "sd15", "sd3", "sd35", "sd35_medium", "wan21", "wan22", "zimage", "zimage_turbo", "zimage_edit", "qwen", "playground", "pixart", "aura", "kolors"],),
            },
            "optional": {
                "default_width": ("INT", {"default": 1024, "min": 256, "max": 2048}),
                "default_height": ("INT", {"default": 1024, "min": 256, "max": 2048}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "detected_type")
    FUNCTION = "load_model"
    CATEGORY = "Laura Studio/Models"
    DESCRIPTION = "Load any model with auto-detection"

    def load_model(self, model_name, model_type, default_width=1024, default_height=1024):

        # Auto-detect if needed
        if model_type == "auto":
            detected = ModelTypeDetector.detect_type(None, model_name)[0]
            model_type = detected if detected != "unknown" else "sdxl"

        # Set default resolutions per model type
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
            "qwen": (1024, 1024),
            "playground": (1024, 1024),
            "pixart": (1024, 1024),
            "aura": (512, 512),
            "kolors": (1024, 1024),
        }

        width, height = resolution_map.get(model_type, (1024, 1024))

        # Load model using ComfyUI's built-in
        from nodes import CheckpointLoaderSimple
        result = CheckpointLoaderSimple().load_checkpoint(model_name)
        model, clip, vae = result[0], result[1], result[2]

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
                "lora_name": ("STRING", {"default": ""}),
                "lora_path": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0}),
                "strength_clip": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0}),
            },
            "optional": {
                "enable": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "apply_lora"
    CATEGORY = "Laura Studio/Models"
    DESCRIPTION = "Apply LoRA for custom character (Laura/Zoriana)"

    def apply_lora(self, model, clip, lora_name, lora_path, strength_model, strength_clip, enable=True):

        if not enable or not lora_path or lora_path == "None":
            return (model, clip)

        if lora_name:
            lora_file = lora_name
        else:
            lora_file = lora_path

        try:
            from nodes import LoraLoader
            result = LoraLoader().load_lora(model, clip, lora_file, strength_model, strength_clip)
            model, clip = result[0], result[1]
        except Exception as e:
            print(f"LoRA loading error: {e}")

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
                "lora_2": (folder_paths.get_filename_list("loras"),),
                "lora_2_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0}),
                "lora_3": (folder_paths.get_filename_list("loras"),),
                "lora_3_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "apply_loras"
    CATEGORY = "Laura Studio/Models"
    DESCRIPTION = "Apply multiple LoRAs (character + style)"

    def apply_loras(self, model, clip, lora_1, lora_1_strength, lora_2, lora_2_strength, lora_3, lora_3_strength):

        from nodes import LoraLoader
        loader = LoraLoader()

        # Apply LoRA 1 (typically character - Zoriana/Laura)
        if lora_1 and lora_1 != "None":
            result = loader.load_lora(model, clip, lora_1, lora_1_strength, lora_1_strength)
            model, clip = result[0], result[1]

        # Apply LoRA 2 (typically style)
        if lora_2 and lora_2 != "None":
            result = loader.load_lora(model, clip, lora_2, lora_2_strength, lora_2_strength)
            model, clip = result[0], result[1]

        # Apply LoRA 3 (typically additional)
        if lora_3 and lora_3 != "None":
            result = loader.load_lora(model, clip, lora_3, lora_3_strength, lora_3_strength)
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
                "model_type": (["auto", "sdxl", "flux", "flux_schnell", "flux2", "flux2_schnell", "sd15", "sd3", "sd35", "sd35_medium", "wan21", "wan22", "zimage", "zimage_turbo", "zimage_edit", "qwen", "playground", "pixart", "aura", "kolors"],),
                "positive_prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "deformed, blurry, bad anatomy, low quality"}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0}),
                "sampler_name": (["euler", "euler_ancestral", "dpm_2", "dpm_2_ancestral", "dpmpp_2m", "dpmpp_sde", "uni_pc", "uni_pc_bh2", "ddpm", "kdpm_2", "kdpm_2_a"],),
                "scheduler": (["normal", "karras", "exponential", "simple", "ddim_uniform"],),
            },
            "optional": {
                "image_to_image": ("IMAGE",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "generate"
    CATEGORY = "Laura Studio/Generation"
    DESCRIPTION = "Universal image generator for all models"

    def generate(self, model, clip, vae, model_type, positive_prompt, negative_prompt,
                 width, height, seed, steps, cfg, sampler_name, scheduler,
                 image_to_image=None, denoise=1.0, batch_size=1):

        # Auto-detect model type from clip if possible
        if model_type == "auto":
            model_type = "sdxl"  # Default

        # Adjust settings based on model type
        model_defaults = {
            "flux":          {"steps": 8,  "cfg": 2.0, "max_res": None},
            "flux_schnell":  {"steps": 4,  "cfg": 2.0, "max_res": None},
            "flux2":         {"steps": 8,  "cfg": 2.0, "max_res": None},
            "flux2_schnell": {"steps": 4,  "cfg": 2.0, "max_res": None},
            "zimage_turbo":  {"steps": 8,  "cfg": 2.0, "max_res": None},
            "sd15":          {"steps": 50, "cfg": 7.0, "max_res": 768},
            "wan21":         {"steps": 25, "cfg": 5.0, "max_res": None},
            "wan22":         {"steps": 25, "cfg": 5.0, "max_res": None},
        }
        defaults = model_defaults.get(model_type, {})
        if defaults.get("cfg") and cfg > defaults["cfg"] + 1:
            cfg = defaults["cfg"]
        if defaults.get("steps") and model_type in ["flux", "flux_schnell", "flux2", "flux2_schnell", "zimage_turbo"]:
            steps = min(steps, defaults["steps"])
        if defaults.get("max_res"):
            max_r = defaults["max_res"]
            if width > max_r:
                width = max_r
            if height > max_r:
                height = max_r

        from nodes import CLIPTextEncode, VAEEncode, EmptyLatentImage, KSampler, VAEDecode

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
            model, seed, steps, cfg, sampler_name, scheduler,
            positive, negative, latent, denoise=denoise
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
                "model_type": (["auto", "sdxl", "flux", "flux_schnell", "flux2", "flux2_schnell", "sd15", "sd3", "sd35", "sd35_medium", "wan21", "wan22", "zimage", "zimage_turbo", "zimage_edit", "qwen", "playground", "pixart", "aura", "kolors"],),
                "positive_prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "deformed, blurry, low quality"}),
                "denoise": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "img2img"
    CATEGORY = "Laura Studio/Generation"
    DESCRIPTION = "Universal image-to-image conversion"

    def img2img(self, model, clip, vae, image, model_type, positive_prompt, negative_prompt,
                denoise, seed, steps, cfg):

        return UniversalGenerator().generate(
            model, clip, vae, model_type, positive_prompt, negative_prompt,
            image.shape[3], image.shape[2], seed, steps, cfg, "euler", "normal",
            image_to_image=image, denoise=denoise
        )


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
                "negative_prompt": ("STRING", {"multiline": True, "default": "deformed, blurry"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 20.0}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "inpaint"
    CATEGORY = "Laura Studio/Generation"
    DESCRIPTION = "Universal inpainting"

    def inpaint(self, model, clip, vae, image, mask, positive_prompt, negative_prompt,
                seed, steps, cfg, denoise):

        from nodes import VAEEncode, CLIPTextEncode, KSampler, VAEDecode

        # Encode image
        encoded = VAEEncode().encode(vae, image)[0]

        # Process mask
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(0)

        # Resize mask to latent size
        latent_h = encoded["samples"].shape[2]
        latent_w = encoded["samples"].shape[3]
        import torch.nn.functional as F
        mask_latent = F.interpolate(mask, size=(latent_h, latent_w), mode="bilinear", align_corners=False)
        mask_latent = (mask_latent > 0.5).float()

        # Encode prompts
        positive = CLIPTextEncode().encode(clip, positive_prompt)[0]
        negative = CLIPTextEncode().encode(clip, negative_prompt)[0]

        # Create latent
        latent = {"samples": encoded["samples"], "mask": mask_latent, "noise_mask": mask_latent}

        # Sample
        sampled = KSampler().sample(
            model, seed, steps, cfg, "dpmpp_2m", "karras",
            positive, negative, latent, denoise=denoise
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
    FUNCTION = "apply_controlnet"
    CATEGORY = "Laura Studio/Models"
    DESCRIPTION = "Apply ControlNet"

    def apply_controlnet(self, conditioning, control_net, image, strength):
        try:
            from comfy_extras.nodes_controlnet import ControlNetApplyAdvanced
            result = ControlNetApplyAdvanced().apply_controlnet(conditioning, conditioning, control_net, image, strength, 0.0, 1.0)
            return (result[0],)
        except Exception:
            # Fallback: return conditioning unchanged
            return (conditioning,)


# Register all nodes
NODE_CLASS_MAPPINGS.update({
    "ModelTypeDetector": ModelTypeDetector,
    "UniversalModelLoader": UniversalModelLoader,
    "LoraManager": LoraManager,
    "MultiLoraStack": MultiLoraStack,
    "UniversalGenerator": UniversalGenerator,
    "UniversalImg2Img": UniversalImg2Img,
    "UniversalInpainter": UniversalInpainter,
    "ControlNetLoader": ControlNetLoader,
    "ApplyControlNet": ApplyControlNet,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "ModelTypeDetector": "Model Type Detector",
    "UniversalModelLoader": "Universal Model Loader",
    "LoraManager": "LoRA Manager (Character)",
    "MultiLoraStack": "Multi-LoRA Stack",
    "UniversalGenerator": "Universal Image Generator",
    "UniversalImg2Img": "Universal Image to Image",
    "UniversalInpainter": "Universal Inpainter",
    "ControlNetLoader": "ControlNet Loader",
    "ApplyControlNet": "Apply ControlNet",
})
