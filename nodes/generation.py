"""
Laura Image Studio - Generation Nodes
Text-to-image and image-to-image generation with Laura character
"""

import torch
from PIL import Image
import numpy as np
from comfy_extras import nodes_custom_sdxl
from nodes import LoadImage, EmptyLatentImage, KSampler, VAEDecode, VAEEncode, CLIPTextEncode
import folder_paths

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# ============== LAURA SDXL GENERATOR ==============
class LauraSDXLGenerator:
    """Main image generation node for Laura character"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive_prompt": ("STRING", {"multiline": True, "default": "photo of laura, detailed face, professional lighting"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "deformed, blurry, bad anatomy, extra limbs, poorly drawn face"}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0}),
                "sampler_name": (["euler", "euler_ancestral", "dpm_2", "dpm_2_ancestral", "dpmpp_2m", "ddpm", "uni_pc", "uni_pc_bh2"],),
                "scheduler": (["normal", "karras", "exponential", "simple"],),
            },
            "optional": {
                "ipadapter_image": ("IMAGE",),
                "ipadapter_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "laura_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "generate"
    CATEGORY = "Laura Studio/Generation"
    DESCRIPTION = "Generate images of Laura using SDXL"

    def generate(self, model, clip, vae, positive_prompt, negative_prompt,
                 width, height, seed, steps, cfg, sampler_name, scheduler,
                 ipadapter_image=None, ipadapter_weight=1.0, laura_strength=0.8):

        # Encode prompts
        positive = CLIPTextEncode.encode(clip, positive_prompt)[0]
        negative = CLIPTextEncode.encode(clip, negative_prompt)[0]

        # Create empty latent
        latent = EmptyLatentImage.generate(width, height, batch_size=1)[0]

        # Sample
        sampled = KSampler.sample(
            model, seed, steps, cfg, sampler_name, scheduler,
            positive, negative, latent, denoise=1.0
        )

        # Decode
        decoded = VAEDecode.decode(vae, sampled)[0]

        return (decoded, sampled)


# ============== LAURA PROMPT BUILDER ==============
class LauraPromptBuilder:
    """Intelligent prompt construction for Laura generation"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_prompt": ("STRING", {"multiline": True, "default": "photo of a woman"}),
                "pose": (["standing", "sitting", "walking", "dynamic", "formal", "casual"],),
                "lighting": (["natural", "studio", "golden hour", "blue hour", "dramatic", "soft"],),
                "camera_angle": (["eye level", "low angle", "high angle", "worm's eye", "bird's eye"],),
                "style": (["portrait", "full body", "fashion", "commercial", "cinematic", "documentary"],),
            },
            "optional": {
                "clothing_description": ("STRING", {"default": ""}),
                "accessories": ("STRING", {"default": ""}),
                "mood": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "build_prompt"
    CATEGORY = "Laura Studio/Generation"
    DESCRIPTION = "Build optimized prompts for Laura generation"

    def build_prompt(self, base_prompt, pose, lighting, camera_angle, style,
                     clothing_description="", accessories="", mood=""):

        # Construct positive prompt
        parts = [base_prompt]

        # Add pose
        pose_map = {
            "standing": "standing pose, confident posture",
            "sitting": "sitting pose, relaxed",
            "walking": "walking motion, dynamic",
            "dynamic": "dynamic pose, movement",
            "formal": "formal pose, professional",
            "casual": "casual pose, relaxed"
        }
        parts.append(pose_map.get(pose, pose))

        # Add lighting
        lighting_map = {
            "natural": "natural lighting, window light",
            "studio": "studio lighting, professional",
            "golden hour": "golden hour, warm light, sunset",
            "blue hour": "blue hour, cool light, dusk",
            "dramatic": "dramatic lighting, chiaroscuro",
            "soft": "soft lighting, diffused"
        }
        parts.append(lighting_map.get(lighting, lighting))

        # Add camera
        camera_map = {
            "eye level": "eye level shot",
            "low angle": "low angle shot, looking up",
            "high angle": "high angle shot, looking down",
            "worm's eye": "worm's eye view, from below",
            "bird's eye": "bird's eye view, from above"
        }
        parts.append(camera_map.get(camera_angle, camera_angle))

        # Add style
        style_map = {
            "portrait": "portrait photography, close-up",
            "full body": "full body shot, entire figure visible",
            "fashion": "fashion photography, editorial",
            "commercial": "commercial photography, professional",
            "cinematic": "cinematic lighting, film grain",
            "documentary": "documentary style, candid"
        }
        parts.append(style_map.get(style, style))

        # Add optional elements
        if clothing_description:
            parts.append(clothing_description)
        if accessories:
            parts.append(f"wearing {accessories}")
        if mood:
            parts.append(mood)

        # Add quality tags
        parts.extend([
            "professional quality",
            "detailed",
            "sharp focus",
            "8k uhd",
            "dslr",
            "soft lighting",
            "high detail"
        ])

        positive = ", ".join(parts)

        # Build negative prompt
        negative = (
            "deformed, blurry, bad anatomy, extra limbs, poorly drawn face, "
            "mutation, mutated, ugly, disfigured, wrong hands, missing limbs, "
            "floating limbs, disconnected limbs, malformed hands, blur, out of focus, "
            "long neck, mutation, poorly drawn hands, missing fingers, "
            "digital artifacts, watermark, text, signature, logo, border"
        )

        return (positive, negative)


# ============== SEED CONTROL ==============
class SeedControl:
    """Control seed for reproducible generation"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["random", "fixed", "increment", "decrement"],),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "control_seed"
    CATEGORY = "Laura Studio/Generation"
    DESCRIPTION = "Control seed for reproducible generation"

    def control_seed(self, mode, seed):
        import random

        if mode == "random":
            result = random.randint(0, 0xffffffffffffffff)
        elif mode == "increment":
            result = seed + 1
        elif mode == "decrement":
            result = max(0, seed - 1)
        else:  # fixed
            result = seed

        return (result,)


# ============== IMAGE TO IMAGE ==============
class LauraImageToImage:
    """Image to image generation with Laura preservation"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "positive_prompt": ("STRING", {"multiline": True, "default": "photo of laura"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "deformed, blurry"}),
                "denoise": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0}),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "img2img"
    CATEGORY = "Laura Studio/Generation"
    DESCRIPTION = "Image to image with Laura preservation"

    def img2img(self, model, clip, vae, image, positive_prompt, negative_prompt,
                denoise, seed, steps, cfg, strength=0.8):

        # Encode image to latent
        encoded = VAEEncode.encode(vae, image)[0]

        # Encode prompts
        positive = CLIPTextEncode.encode(clip, positive_prompt)[0]
        negative = CLIPTextEncode.encode(clip, negative_prompt)[0]

        # Sample with encoded latent
        latent = {"samples": encoded["samples"]}
        sampled = KSampler.sample(
            model, seed, steps, cfg, "euler", "normal",
            positive, negative, latent, denoise=denoise
        )

        # Decode
        decoded = VAEDecode.decode(vae, sampled)[0]

        return (decoded,)


# ============== QUALITY NEGATIVE PROMPTS ==============
class LauraNegativePrompts:
    """Pre-built quality negative prompts"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (["maximum", "standard", "minimal", "custom"],),
                "custom": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_negative"
    CATEGORY = "Laura Studio/Generation"
    DESCRIPTION = "Get quality negative prompts"

    def get_negative(self, preset, custom=""):

        presets = {
            "maximum": (
                "deformed, blurry, bad anatomy, extra limbs, poorly drawn face, "
                "mutation, mutated, ugly, disfigured, wrong hands, missing limbs, "
                "floating limbs, disconnected limbs, malformed hands, blur, out of focus, "
                "long neck, mutation, poorly drawn hands, missing fingers, "
                "digital artifacts, watermark, text, signature, logo, border, "
                "low quality, worst quality, jpeg artifacts, noise, grain, "
                "nsfw, nudi, naked, nude"
            ),
            "standard": (
                "deformed, blurry, bad anatomy, extra limbs, poorly drawn face, "
                "mutation, ugly, disfigured, blur, out of focus, low quality, "
                "worst quality, jpeg artifacts"
            ),
            "minimal": (
                "deformed, blurry, bad anatomy, low quality, worst quality"
            ),
        }

        if preset == "custom":
            return (custom,)

        return (presets.get(preset, presets["standard"]),)


# ============== LORA LOADER ==============
class LauraLoRALoader:
    """Load LoRAs for Laura customization"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "load_lora"
    CATEGORY = "Laura Studio/Generation"
    DESCRIPTION = "Load LoRA for model customization"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        from nodes import LoraLoader

        # Use built-in LoRA loader
        model, clip = LoraLoader.load_lora(model, clip, lora_name, strength_model, strength_clip)
        return (model, clip)


# Register all nodes
NODE_CLASS_MAPPINGS.update({
    "LauraSDXLGenerator": LauraSDXLGenerator,
    "LauraPromptBuilder": LauraPromptBuilder,
    "SeedControl": SeedControl,
    "LauraImageToImage": LauraImageToImage,
    "LauraNegativePrompts": LauraNegativePrompts,
    "LauraLoRALoader": LauraLoRALoader,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "LauraSDXLGenerator": "LAURA SDXL Generator",
    "LauraPromptBuilder": "LAURA Prompt Builder",
    "SeedControl": "Seed Control",
    "LauraImageToImage": "LAURA Image to Image",
    "LauraNegativePrompts": "LAURA Negative Prompts",
    "LauraLoRALoader": "LAURA LoRA Loader",
})
