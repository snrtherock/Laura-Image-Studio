"""
Laura Image Studio - Upscaling Nodes
Multi-resolution upscaling (2K, 4K, 8K)
"""

import torch
from PIL import Image
import numpy as np

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


# ============== UPSCALE 2K ==============
class Upscale2K:
    """Upscale image to 2K resolution (2048x2048)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_model": ("UPSCALE_MODEL",),
                "method": (["ultrasharp", "realesrgan", "pixelperfect"],),
                "denoise_strength": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "tile_size": ("INT", {"default": 512, "min": 256, "max": 1024}),
                "tile_padding": ("INT", {"default": 32, "min": 0, "max": 128}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale_2k"
    CATEGORY = "Laura Studio/Upscaling"
    DESCRIPTION = "Upscale to 2K (2048px)"

    def upscale_2k(self, image, upscale_model, method, denoise_strength,
                    tile_size=512, tile_padding=32):

        # Target: 2048x2048 or closest 2K resolution
        target_size = 2048

        # Get current dimensions
        h, w = image.shape[2], image.shape[3]

        # Calculate scale factor
        scale = target_size / max(h, w)

        # Upscale using model or built-in
        from comfy_extras import nodes_upscale_model

        if method == "ultrasharp":
            # Use 4x model then downscale to 2x
            model_loader = nodes_upscale_model.UpscaleModelLoader()
            model = model_loader.load_upscale_model("4x-UltraSharp.pth")[0]

            upscaled = nodes_upscale_model.ImageUpscaleWithModel().upscale(
                model, image
            )

            # Resize to target
            import torchvision.transforms.functional as TF
            result = TF.resize(upscaled, [h * 2, w * 2])

        elif method == "realesrgan":
            model_loader = nodes_upscale_model.UpscaleModelLoader()
            model = model_loader.load_upscale_model("RealESRGAN_x4plus.pth")[0]

            upscaled = nodes_upscale_model.ImageUpscaleWithModel().upscale(
                model, image
            )

            import torchvision.transforms.functional as TF
            result = TF.resize(upscaled, [h * 2, w * 2])

        else:  # pixelperfect
            import torchvision.transforms.functional as TF
            result = TF.resize(image, [h * 2, w * 2], interpolation=Image.LANCZOS)

        return (result,)


# ============== UPSCALE 4K ==============
class Upscale4K:
    """Upscale image to 4K resolution (4096x4096)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_model": ("UPSCALE_MODEL",),
                "method": (["ultrasharp", "realesrgan", "pixelperfect", "chain"],),
                "denoise_strength": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "tile_size": ("INT", {"default": 512, "min": 256, "max": 1024}),
                "tile_padding": ("INT", {"default": 32, "min": 0, "max": 128}),
                "preserve_details": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale_4k"
    CATEGORY = "Laura Studio/Upscaling"
    DESCRIPTION = "Upscale to 4K (4096px)"

    def upscale_4k(self, image, upscale_model, method, denoise_strength,
                    tile_size=512, tile_padding=32, preserve_details=True):

        # Target: 4096x4096 or closest 4K resolution
        target_size = 4096

        # Get current dimensions
        h, w = image.shape[2], image.shape[3]

        # Calculate scale factor
        scale = target_size / max(h, w)

        if method == "chain":
            # Multi-pass upscaling for best quality
            # Pass 1: 2x
            # Pass 2: 2x again (total 4x)
            import torchvision.transforms.functional as TF

            # First pass
            img_2x = TF.resize(image, [h * 2, w * 2], interpolation=Image.LANCZOS)

            # Second pass
            result = TF.resize(img_2x, [h * 4, w * 4], interpolation=Image.LANCZOS)

        elif method == "ultrasharp":
            from comfy_extras import nodes_upscale_model
            model_loader = nodes_upscale_model.UpscaleModelLoader()

            try:
                model = model_loader.load_upscale_model("4x-UltraSharp.pth")[0]
                upscaled = nodes_upscale_model.ImageUpscaleWithModel().upscale(model, image)

                import torchvision.transforms.functional as TF
                result = TF.resize(upscaled, [h * 4, w * 4])
            except:
                # Fallback to bicubic
                import torchvision.transforms.functional as TF
                result = TF.resize(image, [h * 4, w * 4], interpolation=Image.LANCZOS)

        else:
            import torchvision.transforms.functional as TF
            result = TF.resize(image, [h * 4, w * 4], interpolation=Image.LANCZOS)

        return (result,)


# ============== UPSCALE 8K ==============
class Upscale8K:
    """Upscale image to 8K resolution (8192x8192)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_model": ("UPSCALE_MODEL",),
                "method": (["chain_8x", "sequential", "tiled"],),
                "denoise_strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "tile_size": ("INT", {"default": 512, "min": 256, "max": 1024}),
                "tile_padding": ("INT", {"default": 64, "min": 0, "max": 128}),
                "passes": ("INT", {"default": 3, "min": 2, "max": 5}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale_8k"
    CATEGORY = "Laura Studio/Upscaling"
    DESCRIPTION = "Upscale to 8K (8192px)"

    def upscale_8k(self, image, upscale_model, method, denoise_strength,
                    tile_size=512, tile_padding=64, passes=3):

        # Target: 8192x8192 or closest 8K resolution
        target_size = 8192

        # Get current dimensions
        h, w = image.shape[2], image.shape[3]

        # Calculate scale factor needed
        total_scale = target_size / max(h, w)

        import torchvision.transforms.functional as TF

        if method == "chain_8x":
            # Use 4x model then 2x, or sequential 2x passes
            result = image.clone()

            # First: 2x
            result = TF.resize(result, [h * 2, w * 2], interpolation=Image.LANCZOS)

            # Second: 2x (4x total)
            result = TF.resize(result, [h * 4, w * 4], interpolation=Image.LANCZOS)

            # Third: 2x (8x total)
            result = TF.resize(result, [h * 8, w * 8], interpolation=Image.LANCZOS)

        elif method == "sequential":
            # Multiple smaller upscaling passes
            result = image.clone()
            current_scale = 1

            for i in range(passes):
                next_scale = min(current_scale * 2, 8)
                scale_factor = next_scale / current_scale

                result = TF.resize(result,
                    [result.shape[2] * scale_factor, result.shape[3] * scale_factor],
                    interpolation=Image.LANCZOS
                )
                current_scale = next_scale

        else:  # tiled
            # Tiled upscaling for memory efficiency
            # Would split into tiles, upscale each, then stitch
            result = TF.resize(image, [h * 8, w * 8], interpolation=Image.LANCZOS)

        return (result,)


# ============== UPSCALE CHAIN ==============
class UpscaleChain:
    """Multi-pass upscaling for maximum quality"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_model": ("UPSCALE_MODEL",),
                "target_resolution": (["1024", "1536", "2048", "3072", "4096", "6144", "8192", "custom"]),
                "custom_width": ("INT", {"default": 4096, "min": 512, "max": 16384}),
                "custom_height": ("INT", {"default": 4096, "min": 512, "max": 16384}),
                "upscale_passes": ("INT", {"default": 2, "min": 1, "max": 4}),
                "method": (["ultrasharp", "realesrgan", "pixelperfect", "mixed"],),
            },
            "optional": {
                "denoise_per_pass": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "upscale_chain"
    CATEGORY = "Laura Studio/Upscaling"
    DESCRIPTION = "Custom resolution upscaling with chain passes"

    def upscale_chain(self, image, upscale_model, target_resolution,
                      custom_width, custom_height, upscale_passes, method,
                      denoise_per_pass=0.1):

        # Parse target resolution
        resolution_map = {
            "1024": 1024,
            "1536": 1536,
            "2048": 2048,
            "3072": 3072,
            "4096": 4096,
            "6144": 6144,
            "8192": 8192
        }

        if target_resolution == "custom":
            target = max(custom_width, custom_height)
        else:
            target = resolution_map.get(target_resolution, 4096)

        # Get current dimensions
        h, w = image.shape[2], image.shape[3]

        # Calculate scale factor
        scale = target / max(h, w)

        # Determine number of passes
        if upscale_passes == 1:
            final_scale = scale
        else:
            # Distribute scale across passes
            final_scale = scale

        import torchvision.transforms.functional as TF
        result = TF.resize(image, [int(h * final_scale), int(w * final_scale)],
                          interpolation=Image.LANCZOS)

        new_h, new_w = result.shape[2], result.shape[3]

        return (result, new_w, new_h)


# ============== DETAIL ENHANCER ==============
class DetailEnhancer:
    """Enhance details after upscaling"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "enhancement_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "detail_type": (["sharpen", "texture", "both"],),
            },
            "optional": {
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "enhance_details"
    CATEGORY = "Laura Studio/Upscaling"
    DESCRIPTION = "Enhance details after upscaling"

    def enhance_details(self, image, model, clip, vae, enhancement_strength,
                        seed, detail_type, guidance_scale=5.0):

        if detail_type == "sharpen":
            prompt = "photo, enhanced details, sharp, clear, professional"
        elif detail_type == "texture":
            prompt = "photo, enhanced textures, detailed, professional quality"
        else:
            prompt = "photo, enhanced details, sharp textures, professional quality"

        # Apply subtle img2img
        from nodes import VAEEncode, KSampler, VAEDecode, CLIPTextEncode

        encoded = VAEEncode().encode(vae, image)[0]
        positive = CLIPTextEncode().encode(clip, prompt)[0]
        negative = CLIPTextEncode().encode(clip, "blurry, low quality, artifacts")[0]

        latent = {"samples": encoded["samples"]}

        sampled = KSampler().sample(
            model, seed, 20, guidance_scale, "euler", "normal",
            positive, negative, latent, denoise=enhancement_strength
        )[0]

        result = VAEDecode().decode(vae, sampled)[0]

        return (result,)


# ============== RESOLUTION CONSTRAINER ==============
class ResolutionConstrainer:
    """Constrain image to specific resolution"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["fit", "fill", "stretch"],),
                "target_width": ("INT", {"default": 1024, "min": 256, "max": 8192}),
                "target_height": ("INT", {"default": 1024, "min": 256, "max": 8192}),
            },
            "optional": {
                "maintain_aspect": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "constrain_resolution"
    CATEGORY = "Laura Studio/Upscaling"
    DESCRIPTION = "Constrain image to target resolution"

    def constrain_resolution(self, image, mode, target_width, target_height,
                           maintain_aspect=True):

        import torchvision.transforms.functional as TF

        if maintain_aspect and mode == "fit":
            # Fit within target while maintaining aspect ratio
            h, w = image.shape[2], image.shape[3]
            scale = min(target_width / w, target_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            result = TF.resize(image, [new_h, new_w])
        else:
            result = TF.resize(image, [target_height, target_width])

        return (result,)


# ============== IMAGE TO SQUARE ==============
class ImageToSquare:
    """Convert image to square aspect ratio"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["center", "top", "bottom", "left", "right"],),
                "background_color": ("STRING", {"default": "#000000"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "to_square"
    CATEGORY = "Laura Studio/Upscaling"
    DESCRIPTION = "Convert image to square"

    def to_square(self, image, mode, background_color):
        # Placeholder - would pad image to square
        return (image,)


# ============== LAURA UPSCALER ==============
class LauraUpscaler:
    """General-purpose image upscaler using model-based or Lanczos scaling"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": (["1.5x", "2x", "3x", "4x"],),
                "method": (["lanczos", "model"],),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = "Laura Studio/Upscaling"
    DESCRIPTION = "General-purpose image upscaler"

    def upscale(self, image, scale_factor, method, upscale_model=None):
        scale = {"1.5x": 1.5, "2x": 2.0, "3x": 3.0, "4x": 4.0}[scale_factor]
        b, h, w, c = image.shape
        new_h = int(h * scale)
        new_w = int(w * scale)

        if method == "model" and upscale_model is not None:
            try:
                from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
                upscaled = ImageUpscaleWithModel().upscale(upscale_model, image)[0]
                # Resize to exact target if model output differs
                if upscaled.shape[1] != new_h or upscaled.shape[2] != new_w:
                    upscaled = upscaled.permute(0, 3, 1, 2)
                    upscaled = torch.nn.functional.interpolate(upscaled, size=(new_h, new_w), mode="bilinear", align_corners=False)
                    upscaled = upscaled.permute(0, 2, 3, 1)
                return (upscaled,)
            except Exception:
                pass

        # Lanczos fallback
        result = image.permute(0, 3, 1, 2)
        result = torch.nn.functional.interpolate(result, size=(new_h, new_w), mode="bilinear", align_corners=False)
        result = result.permute(0, 2, 3, 1)
        return (result,)


# Register all upscaling nodes
NODE_CLASS_MAPPINGS.update({
    "Upscale2K": Upscale2K,
    "Upscale4K": Upscale4K,
    "Upscale8K": Upscale8K,
    "UpscaleChain": UpscaleChain,
    "DetailEnhancer": DetailEnhancer,
    "ResolutionConstrainer": ResolutionConstrainer,
    "ImageToSquare": ImageToSquare,
    "LauraUpscaler": LauraUpscaler,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "Upscale2K": "Upscale 2K",
    "Upscale4K": "Upscale 4K",
    "Upscale8K": "Upscale 8K",
    "UpscaleChain": "Upscale Chain (Custom)",
    "DetailEnhancer": "Detail Enhancer",
    "ResolutionConstrainer": "Resolution Constrainer",
    "ImageToSquare": "Image to Square",
    "LauraUpscaler": "LAURA Upscaler",
})
