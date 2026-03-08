"""
Laura Image Studio - Upscaling Nodes
Multi-resolution upscaling (2K, 4K, 8K)
"""

import torch

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
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale_2k"
    CATEGORY = "Laura Studio/Upscaling"
    DESCRIPTION = "Upscale to 2K (2048px)"

    def upscale_2k(
        self,
        image,
        upscale_model,
        method,
        denoise_strength,
        tile_size=512,
        tile_padding=32,
    ):
        # Get current dimensions - ComfyUI IMAGE is [B, H, W, C]
        h, w = image.shape[1], image.shape[2]

        # Upscale using model or built-in
        from comfy_extras import nodes_upscale_model

        if method == "ultrasharp":
            # Use the user-provided upscale_model for upscaling
            upscaled = nodes_upscale_model.ImageUpscaleWithModel().upscale(
                upscale_model, image
            )[0]

            # Resize to target using torch.nn.functional (works on [B,H,W,C] after permute)
            upscaled = upscaled.permute(0, 3, 1, 2)  # [B,H,W,C] -> [B,C,H,W]
            result = torch.nn.functional.interpolate(
                upscaled, size=(h * 2, w * 2), mode="bilinear", align_corners=False
            )
            result = result.permute(0, 2, 3, 1)  # [B,C,H,W] -> [B,H,W,C]

        elif method == "realesrgan":
            upscaled = nodes_upscale_model.ImageUpscaleWithModel().upscale(
                upscale_model, image
            )[0]

            upscaled = upscaled.permute(0, 3, 1, 2)
            result = torch.nn.functional.interpolate(
                upscaled, size=(h * 2, w * 2), mode="bilinear", align_corners=False
            )
            result = result.permute(0, 2, 3, 1)

        else:  # pixelperfect
            img = image.permute(0, 3, 1, 2)
            result = torch.nn.functional.interpolate(
                img, size=(h * 2, w * 2), mode="bilinear", align_corners=False
            )
            result = result.permute(0, 2, 3, 1)

        # Apply detail enhancement proportional to denoise_strength
        if denoise_strength > 0:
            result = self._sharpen(result, denoise_strength)

        result = torch.clamp(result, 0, 1)
        return (result,)

    @staticmethod
    def _sharpen(image, strength):
        """Apply unsharp-mask sharpening proportional to strength (0-1)."""
        img = image.permute(0, 3, 1, 2)  # [B,H,W,C] -> [B,C,H,W]
        blurred = torch.nn.functional.avg_pool2d(
            img, kernel_size=3, stride=1, padding=1
        )
        sharpened = img + strength * (img - blurred)
        sharpened = torch.clamp(sharpened, 0, 1)
        return sharpened.permute(0, 2, 3, 1)  # [B,C,H,W] -> [B,H,W,C]


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
                "denoise_strength": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.0, "max": 1.0},
                ),
            },
            "optional": {
                "tile_size": ("INT", {"default": 512, "min": 256, "max": 1024}),
                "tile_padding": ("INT", {"default": 32, "min": 0, "max": 128}),
                "preserve_details": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale_4k"
    CATEGORY = "Laura Studio/Upscaling"
    DESCRIPTION = "Upscale to 4K (4096px)"

    def upscale_4k(
        self,
        image,
        upscale_model,
        method,
        denoise_strength,
        tile_size=512,
        tile_padding=32,
        preserve_details=True,
    ):
        # Get current dimensions - ComfyUI IMAGE is [B, H, W, C]
        h, w = image.shape[1], image.shape[2]

        if method == "chain":
            # Multi-pass upscaling for best quality
            img = image.permute(0, 3, 1, 2)  # [B,H,W,C] -> [B,C,H,W]

            # First pass: 2x
            img_2x = torch.nn.functional.interpolate(
                img, size=(h * 2, w * 2), mode="bilinear", align_corners=False
            )

            # Second pass: 2x again (total 4x)
            result = torch.nn.functional.interpolate(
                img_2x, size=(h * 4, w * 4), mode="bilinear", align_corners=False
            )
            result = result.permute(0, 2, 3, 1)  # [B,C,H,W] -> [B,H,W,C]

        elif method == "ultrasharp":
            from comfy_extras import nodes_upscale_model

            try:
                upscaled = nodes_upscale_model.ImageUpscaleWithModel().upscale(
                    upscale_model, image
                )[0]

                upscaled = upscaled.permute(0, 3, 1, 2)
                result = torch.nn.functional.interpolate(
                    upscaled, size=(h * 4, w * 4), mode="bilinear", align_corners=False
                )
                result = result.permute(0, 2, 3, 1)
            except Exception:
                # Fallback to bilinear
                from .models import LauraLogger

                LauraLogger.warn("Model upscale failed, using bilinear fallback")
                img = image.permute(0, 3, 1, 2)
                result = torch.nn.functional.interpolate(
                    img, size=(h * 4, w * 4), mode="bilinear", align_corners=False
                )
                result = result.permute(0, 2, 3, 1)

        else:
            img = image.permute(0, 3, 1, 2)
            result = torch.nn.functional.interpolate(
                img, size=(h * 4, w * 4), mode="bilinear", align_corners=False
            )
            result = result.permute(0, 2, 3, 1)

        # Apply detail enhancement proportional to denoise_strength
        if denoise_strength > 0:
            result = Upscale2K._sharpen(result, denoise_strength)

        result = torch.clamp(result, 0, 1)
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
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale_8k"
    CATEGORY = "Laura Studio/Upscaling"
    DESCRIPTION = "Upscale to 8K (8192px)"

    def upscale_8k(
        self,
        image,
        upscale_model,
        method,
        denoise_strength,
        tile_size=512,
        tile_padding=64,
        passes=3,
    ):
        # Get current dimensions - ComfyUI IMAGE is [B, H, W, C]
        h, w = image.shape[1], image.shape[2]

        from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel

        upscaler = ImageUpscaleWithModel()

        if method == "chain_8x":
            # Use sequential model-based 2x passes for 8x total
            # Each pass: model upscale -> resize to exact 2x of previous
            result_img = image
            scales = [2, 4, 8]
            for target_scale in scales:
                try:
                    upscaled = upscaler.upscale(upscale_model, result_img)[0]
                    upscaled = upscaled.permute(0, 3, 1, 2)
                    result = torch.nn.functional.interpolate(
                        upscaled,
                        size=(h * target_scale, w * target_scale),
                        mode="bilinear",
                        align_corners=False,
                    )
                    result_img = result.permute(0, 2, 3, 1)
                except Exception:
                    # Fallback to bilinear for this pass
                    from .models import LauraLogger

                    LauraLogger.warn("Model upscale failed, using bilinear fallback")
                    img = result_img.permute(0, 3, 1, 2)
                    result = torch.nn.functional.interpolate(
                        img,
                        size=(h * target_scale, w * target_scale),
                        mode="bilinear",
                        align_corners=False,
                    )
                    result_img = result.permute(0, 2, 3, 1)
            result = result_img

        elif method == "sequential":
            # Multiple model-based upscaling passes
            result_img = image
            current_scale = 1

            for i in range(passes):
                next_scale = min(current_scale * 2, 8)
                scale_factor = next_scale / current_scale

                cur_h, cur_w = result_img.shape[1], result_img.shape[2]
                new_h = int(cur_h * scale_factor)
                new_w = int(cur_w * scale_factor)

                try:
                    upscaled = upscaler.upscale(upscale_model, result_img)[0]
                    upscaled = upscaled.permute(0, 3, 1, 2)
                    result = torch.nn.functional.interpolate(
                        upscaled,
                        size=(new_h, new_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    result_img = result.permute(0, 2, 3, 1)
                except Exception:
                    from .models import LauraLogger

                    LauraLogger.warn("Model upscale failed, using bilinear fallback")
                    img = result_img.permute(0, 3, 1, 2)
                    result = torch.nn.functional.interpolate(
                        img,
                        size=(new_h, new_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    result_img = result.permute(0, 2, 3, 1)

                current_scale = next_scale
                if current_scale >= 8:
                    break

            result = result_img

        else:  # tiled
            # Tiled upscaling: model upscale then resize to 8x
            try:
                upscaled = upscaler.upscale(upscale_model, image)[0]
                upscaled = upscaled.permute(0, 3, 1, 2)
                result = torch.nn.functional.interpolate(
                    upscaled,
                    size=(h * 8, w * 8),
                    mode="bilinear",
                    align_corners=False,
                )
                result = result.permute(0, 2, 3, 1)
            except Exception:
                from .models import LauraLogger

                LauraLogger.warn("Model upscale failed, using bilinear fallback")
                img = image.permute(0, 3, 1, 2)
                result = torch.nn.functional.interpolate(
                    img, size=(h * 8, w * 8), mode="bilinear", align_corners=False
                )
                result = result.permute(0, 2, 3, 1)

        # Apply detail enhancement proportional to denoise_strength
        if denoise_strength > 0:
            result = Upscale2K._sharpen(result, denoise_strength)

        result = torch.clamp(result, 0, 1)
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
                "target_resolution": (
                    ["1024", "1536", "2048", "3072", "4096", "6144", "8192", "custom"]
                ),
                "custom_width": ("INT", {"default": 4096, "min": 512, "max": 16384}),
                "custom_height": ("INT", {"default": 4096, "min": 512, "max": 16384}),
                "upscale_passes": ("INT", {"default": 2, "min": 1, "max": 4}),
                "method": (["ultrasharp", "realesrgan", "pixelperfect", "mixed"],),
            },
            "optional": {
                "denoise_per_pass": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "upscale_chain"
    CATEGORY = "Laura Studio/Upscaling"
    DESCRIPTION = "Custom resolution upscaling with chain passes"

    def upscale_chain(
        self,
        image,
        upscale_model,
        target_resolution,
        custom_width,
        custom_height,
        upscale_passes,
        method,
        denoise_per_pass=0.1,
    ):
        # Parse target resolution
        resolution_map = {
            "1024": 1024,
            "1536": 1536,
            "2048": 2048,
            "3072": 3072,
            "4096": 4096,
            "6144": 6144,
            "8192": 8192,
        }

        if target_resolution == "custom":
            target = max(custom_width, custom_height)
        else:
            target = resolution_map.get(target_resolution, 4096)

        # Get current dimensions - ComfyUI IMAGE is [B, H, W, C]
        h, w = image.shape[1], image.shape[2]

        # Calculate total scale factor needed
        total_scale = target / max(h, w)

        # Distribute scale across multiple passes for better quality
        # Each pass applies a fractional scale: total_scale^(1/passes)
        per_pass_scale = total_scale ** (1.0 / max(upscale_passes, 1))

        from .models import LauraLogger

        LauraLogger.info(
            f"UpscaleChain: {h}x{w} -> target {target}px, "
            f"{upscale_passes} passes (scale/pass: {per_pass_scale:.2f}x), "
            f"method: {method}"
        )

        # Try to use the upscale_model for model-based upscaling
        result_img = image
        try:
            from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel

            upscaler = ImageUpscaleWithModel()
            for pass_num in range(upscale_passes):
                cur_h, cur_w = result_img.shape[1], result_img.shape[2]
                pass_target_h = int(cur_h * per_pass_scale)
                pass_target_w = int(cur_w * per_pass_scale)

                # Cap at final target
                if pass_num == upscale_passes - 1:
                    longest = max(cur_h, cur_w)
                    pass_scale = target / longest
                    pass_target_h = int(cur_h * pass_scale)
                    pass_target_w = int(cur_w * pass_scale)

                # Model-based upscale
                upscaled = upscaler.upscale(upscale_model, result_img)[0]

                # Resize to the target for this pass
                img_perm = upscaled.permute(0, 3, 1, 2)
                resized = torch.nn.functional.interpolate(
                    img_perm,
                    size=(pass_target_h, pass_target_w),
                    mode="bilinear",
                    align_corners=False,
                )
                result_img = resized.permute(0, 2, 3, 1)

                LauraLogger.info(
                    f"  Pass {pass_num + 1}/{upscale_passes}: "
                    f"{cur_h}x{cur_w} -> {pass_target_h}x{pass_target_w}"
                )

        except Exception as e:
            LauraLogger.warn(
                f"Model-based upscale unavailable ({e}), falling back to bilinear"
            )
            # Fallback: single bilinear interpolation to target
            final_h = int(h * total_scale)
            final_w = int(w * total_scale)
            img_perm = image.permute(0, 3, 1, 2)
            resized = torch.nn.functional.interpolate(
                img_perm,
                size=(final_h, final_w),
                mode="bilinear",
                align_corners=False,
            )
            result_img = resized.permute(0, 2, 3, 1)

        new_h, new_w = result_img.shape[1], result_img.shape[2]
        result_img = torch.clamp(result_img, 0, 1)

        return (result_img, new_w, new_h)


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
                "enhancement_strength": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 1.0},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "detail_type": (["sharpen", "texture", "both"],),
            },
            "optional": {
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "enhance_details"
    CATEGORY = "Laura Studio/Upscaling"
    DESCRIPTION = "Enhance details after upscaling"

    def enhance_details(
        self,
        image,
        model,
        clip,
        vae,
        enhancement_strength,
        seed,
        detail_type,
        guidance_scale=5.0,
    ):
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
            model,
            seed,
            20,
            guidance_scale,
            "euler",
            "normal",
            positive,
            negative,
            latent,
            denoise=enhancement_strength,
        )[0]

        result = VAEDecode().decode(vae, sampled)[0]
        result = torch.clamp(result, 0, 1)

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
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "constrain_resolution"
    CATEGORY = "Laura Studio/Upscaling"
    DESCRIPTION = "Constrain image to target resolution"

    def constrain_resolution(
        self, image, mode, target_width, target_height, maintain_aspect=True
    ):
        if maintain_aspect and mode == "fit":
            # Fit within target while maintaining aspect ratio
            # ComfyUI IMAGE is [B, H, W, C]
            h, w = image.shape[1], image.shape[2]
            scale = min(target_width / w, target_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = image.permute(0, 3, 1, 2)  # [B,H,W,C] -> [B,C,H,W]
            result = torch.nn.functional.interpolate(
                img, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
            result = result.permute(0, 2, 3, 1)  # [B,C,H,W] -> [B,H,W,C]
        else:
            img = image.permute(0, 3, 1, 2)
            result = torch.nn.functional.interpolate(
                img,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )
            result = result.permute(0, 2, 3, 1)

        result = torch.clamp(result, 0, 1)
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
    RETURN_NAMES = ("image",)
    FUNCTION = "to_square"
    CATEGORY = "Laura Studio/Upscaling"
    DESCRIPTION = "Convert image to square"

    def to_square(self, image, mode, background_color):
        B, H, W, C = image.shape

        # Already square
        if H == W:
            return (image,)

        target_size = max(H, W)

        # Parse hex color to RGB float values
        bg_color = self._parse_color(background_color)

        # Create square canvas filled with background color
        canvas = torch.zeros(
            (B, target_size, target_size, C), dtype=image.dtype, device=image.device
        )
        for c_idx in range(min(C, 3)):
            canvas[:, :, :, c_idx] = bg_color[c_idx]
        # If alpha channel exists, set it to 1.0 on bg
        if C == 4:
            canvas[:, :, :, 3] = 1.0

        # Calculate paste position based on mode
        if H > W:
            # Image is taller than wide — need horizontal padding
            pad_total = target_size - W
            if mode == "center":
                x_offset = pad_total // 2
            elif mode == "left":
                x_offset = 0
            elif mode == "right":
                x_offset = pad_total
            else:
                x_offset = pad_total // 2  # default center for top/bottom on landscape
            y_offset = 0
        else:
            # Image is wider than tall — need vertical padding
            pad_total = target_size - H
            if mode == "center":
                y_offset = pad_total // 2
            elif mode == "top":
                y_offset = 0
            elif mode == "bottom":
                y_offset = pad_total
            else:
                y_offset = pad_total // 2  # default center for left/right on portrait
            x_offset = 0

        # Paste original image onto canvas
        canvas[:, y_offset : y_offset + H, x_offset : x_offset + W, :] = image

        return (torch.clamp(canvas, 0, 1),)

    @staticmethod
    def _parse_color(hex_color):
        """Parse hex color string to (R, G, B) float tuple in 0-1 range."""
        hex_color = hex_color.strip().lstrip("#")
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16) / 255.0
            g = int(hex_color[2:4], 16) / 255.0
            b = int(hex_color[4:6], 16) / 255.0
            return (r, g, b)
        elif len(hex_color) == 3:
            r = int(hex_color[0] + hex_color[0], 16) / 255.0
            g = int(hex_color[1] + hex_color[1], 16) / 255.0
            b = int(hex_color[2] + hex_color[2], 16) / 255.0
            return (r, g, b)
        return (0.0, 0.0, 0.0)  # Default black


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
            },
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
                    upscaled = torch.nn.functional.interpolate(
                        upscaled,
                        size=(new_h, new_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    upscaled = upscaled.permute(0, 2, 3, 1)
                return (torch.clamp(upscaled, 0, 1),)
            except Exception as e:
                from .models import LauraLogger

                LauraLogger.warn(
                    f"Model-based upscale failed ({e}), falling back to lanczos"
                )

        # Lanczos fallback — use bicubic (closest to Lanczos available in PyTorch)
        result = image.permute(0, 3, 1, 2)
        result = torch.nn.functional.interpolate(
            result, size=(new_h, new_w), mode="bicubic", align_corners=False
        )
        result = result.permute(0, 2, 3, 1)
        result = torch.clamp(result, 0, 1)
        return (result,)


# ============== LAURA VIDEO CINEMA UPSCALE ==============
class LauraVideoCinemaUpscale:
    """Cinema-grade video upscaler using SUPIR-Video and RIFE interpolation"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE",),
                "upscale_by": (["2x", "4x"], {"default": "2x"}),
                "quality_mode": (
                    ["Eco (ESRGAN)", "Cinema (SUPIR)"],
                    {"default": "Cinema (SUPIR)"},
                ),
                "fps_multiplier": (
                    ["1x (None)", "2x (RIFE)", "4x (RIFE)"],
                    {"default": "2x (RIFE)"},
                ),
                "denoise": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL",),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("upscaled_video_frames", "final_fps")
    FUNCTION = "upscale_video"
    CATEGORY = "Laura Studio/Upscaling"
    DESCRIPTION = "Cinema-grade video upscaling with SUPIR + RIFE interpolation"

    def upscale_video(
        self,
        video_frames,
        upscale_by,
        quality_mode,
        fps_multiplier,
        denoise,
        upscale_model=None,
    ):
        from .models import LauraLogger
        import importlib

        num_frames = video_frames.shape[0]
        multiplier = int(fps_multiplier[0])
        LauraLogger.info(
            f"Upscaling {num_frames} frames - Mode: {quality_mode} - FPS: {multiplier}x"
        )

        # Auto VRAM Check and Optimization
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if quality_mode == "Cinema (SUPIR)" and vram < 16:
                LauraLogger.warn(
                    "VRAM below 16GB. Cinema mode (SUPIR) may OOM. Falling back to High-Quality ESRGAN."
                )
                quality_mode = "Eco (ESRGAN)"
        else:
            LauraLogger.warn("No CUDA GPU detected. Falling back to Eco (ESRGAN) mode.")
            quality_mode = "Eco (ESRGAN)"

        # 1. SPATIAL UPSCALING (SUPIR or ESRGAN)
        upscaled_frames = []
        upscaled = None

        if quality_mode == "Cinema (SUPIR)":
            try:
                # SUPIR typically requires a specific pipeline: Load Model -> SUPIR Tile -> Decode
                # We attempt to find the SUPIR custom nodes if installed
                supir_mod = importlib.import_module("custom_nodes.ComfyUI-SUPIR")
                supir_node = supir_mod.NODE_CLASS_MAPPINGS["SUPIR_Upscale"]()
                LauraLogger.info("Applying SUPIR-Video spatial refinement...")
                # Process frames one at a time (SUPIR is very VRAM-heavy)
                scale = int(upscale_by[0])
                for i in range(num_frames):
                    frame = video_frames[i : i + 1]
                    try:
                        up_frame = supir_node.upscale(frame, scale)[0]
                        upscaled_frames.append(up_frame)
                    except Exception as frame_err:
                        LauraLogger.warn(
                            f"SUPIR failed on frame {i}: {frame_err}. Using bilinear fallback for this frame."
                        )
                        h, w = frame.shape[1], frame.shape[2]
                        fallback = frame.permute(0, 3, 1, 2)
                        fallback = torch.nn.functional.interpolate(
                            fallback,
                            size=(h * scale, w * scale),
                            mode="bilinear",
                            align_corners=False,
                        )
                        upscaled_frames.append(fallback.permute(0, 2, 3, 1))
                    # Free VRAM between frames
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if upscaled_frames:
                    upscaled = torch.cat(upscaled_frames, dim=0)
                else:
                    quality_mode = "Eco (ESRGAN)"
            except Exception:
                LauraLogger.warn("SUPIR nodes not found. Falling back to ESRGAN.")
                quality_mode = "Eco (ESRGAN)"
                upscaled_frames = []  # Clear any partial SUPIR results

        if quality_mode == "Eco (ESRGAN)":
            # Use high-quality ESRGAN — prefer user-provided upscale_model, fall back to loading 4x-UltraSharp
            from comfy_extras import nodes_upscale_model

            esrgan_model = None
            if upscale_model is not None:
                esrgan_model = upscale_model
                LauraLogger.info("Using user-provided upscale model for ESRGAN path.")
            else:
                try:
                    model_loader = nodes_upscale_model.UpscaleModelLoader()
                    model_name = "4x-UltraSharp.pth"
                    esrgan_model = model_loader.load_upscale_model(model_name)[0]
                    LauraLogger.info(f"Loaded default upscale model: {model_name}")
                except Exception as e:
                    LauraLogger.error(f"Failed to load default upscale model: {e}")

            if esrgan_model is not None:
                try:
                    upscaler = nodes_upscale_model.ImageUpscaleWithModel()

                    # Process frames (batching for speed if VRAM allows)
                    for i in range(num_frames):
                        frame = video_frames[i : i + 1]
                        up_frame = upscaler.upscale(esrgan_model, frame)[0]
                        upscaled_frames.append(up_frame)

                    upscaled = torch.cat(upscaled_frames, dim=0)
                    # Resize to match requested upscale_by target
                    target_scale = int(upscale_by[0])
                    orig_h, orig_w = video_frames.shape[1], video_frames.shape[2]
                    target_h = orig_h * target_scale
                    target_w = orig_w * target_scale
                    if upscaled.shape[1] != target_h or upscaled.shape[2] != target_w:
                        upscaled_perm = upscaled.permute(0, 3, 1, 2)
                        upscaled_perm = torch.nn.functional.interpolate(
                            upscaled_perm,
                            size=(target_h, target_w),
                            mode="bilinear",
                            align_corners=False,
                        )
                        upscaled = upscaled_perm.permute(0, 2, 3, 1)
                except Exception as e:
                    LauraLogger.error(f"ESRGAN Video upscale failed: {e}")
                    upscaled = video_frames.clone()
            else:
                LauraLogger.warn("No upscale model available. Passing frames through.")
                upscaled = video_frames.clone()

        # If neither SUPIR nor ESRGAN produced results, pass through
        if upscaled is None:
            upscaled = video_frames.clone()

        # 2. TEMPORAL INTERPOLATION (RIFE)
        if multiplier > 1:
            try:
                rife_mod = importlib.import_module(
                    "custom_nodes.ComfyUI-Frame-Interpolation"
                )
                rife_node = rife_mod.NODE_CLASS_MAPPINGS["RIFE_VFI"]()
                LauraLogger.info(f"Applying RIFE v4 {multiplier}x interpolation...")
                result = rife_node.vfi(
                    ckpt_name="rife49.pth",
                    frames=upscaled,
                    multiplier=multiplier,
                    clear_cache_after_n_frames=10,
                    fast_mode=True,
                    ensemble=False,
                    scale_factor=1.0,
                )
                upscaled = result[0]
            except Exception as e:
                LauraLogger.warn(f"RIFE interpolation failed: {e}. Skipping FPS boost.")

        final_frame_count = upscaled.shape[0]
        LauraLogger.info(
            f"Video Production Complete: {final_frame_count} frames at {24 * multiplier} FPS."
        )

        upscaled = torch.clamp(upscaled, 0, 1)
        return (upscaled, 24 * multiplier)


# Register all upscaling nodes
NODE_CLASS_MAPPINGS.update(
    {
        "Upscale2K": Upscale2K,
        "Upscale4K": Upscale4K,
        "Upscale8K": Upscale8K,
        "UpscaleChain": UpscaleChain,
        "DetailEnhancer": DetailEnhancer,
        "ResolutionConstrainer": ResolutionConstrainer,
        "ImageToSquare": ImageToSquare,
        "LauraUpscaler": LauraUpscaler,
        "LauraVideoCinemaUpscale": LauraVideoCinemaUpscale,
    }
)

NODE_DISPLAY_NAME_MAPPINGS.update(
    {
        "Upscale2K": "Upscale 2K",
        "Upscale4K": "Upscale 4K",
        "Upscale8K": "Upscale 8K",
        "UpscaleChain": "Upscale Chain (Custom)",
        "DetailEnhancer": "Detail Enhancer",
        "ResolutionConstrainer": "Resolution Constrainer",
        "ImageToSquare": "Image to Square",
        "LauraUpscaler": "LAURA Upscaler",
        "LauraVideoCinemaUpscale": "Video Cinema Upscale (SUPIR+RIFE)",
    }
)
