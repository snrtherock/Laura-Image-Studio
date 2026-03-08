"""
Laura Image Studio - Background Nodes
Background removal, replacement, and generation
Delegates to RMBG for real background removal
"""

import torch

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


# ============== RMBG IMPORT HELPER ==============
def _try_import_rmbg():
    """Try to import RMBG background removal node"""
    try:
        from custom_nodes import ComfyUI_RMBG

        mappings = ComfyUI_RMBG.NODE_CLASS_MAPPINGS
        if "RMBG" in mappings:
            return mappings["RMBG"]
    except Exception:
        pass
    try:
        import importlib

        mod = importlib.import_module("custom_nodes.ComfyUI-RMBG")
        mappings = mod.NODE_CLASS_MAPPINGS
        if "RMBG" in mappings:
            return mappings["RMBG"]
    except Exception:
        pass
    return None


def _simple_bg_mask_fallback(image):
    """Simple edge-based background estimation when RMBG is not available.
    Assumes subject is in center and background is at edges.
    Returns a rough mask where 1 = foreground, 0 = background.
    """
    B, H, W, C = image.shape
    mask = torch.zeros((B, H, W), dtype=torch.float32, device=image.device)

    # Elliptical foreground region (center 60% of image)
    cy, cx = H / 2, W / 2
    ry, rx = H * 0.35, W * 0.30

    y_coords = torch.arange(H, device=image.device).float()
    x_coords = torch.arange(W, device=image.device).float()
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

    ellipse = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
    fg_mask = (ellipse <= 1.0).float()

    # Smooth the edges
    import torch.nn.functional as F

    fg_mask = fg_mask.unsqueeze(0).unsqueeze(0)
    fg_mask = F.avg_pool2d(fg_mask, kernel_size=31, stride=1, padding=15)
    fg_mask = fg_mask.squeeze(0).squeeze(0)

    mask[:] = fg_mask
    return mask


# ============== BACKGROUND REMOVER ==============
class BackgroundRemover:
    """Remove background using RMBG (supports RMBG-2.0, INSPYRENET, BEN, BiRefNet)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "rmbg_model": (["RMBG-2.0", "INSPYRENET", "BEN", "BiRefNet"],),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "invert_mask": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("cutout", "mask")
    FUNCTION = "remove_background"
    CATEGORY = "Laura Studio/Background"
    DESCRIPTION = "Remove background using RMBG (multiple model support)"

    def remove_background(
        self, image, rmbg_model="RMBG-2.0", threshold=0.5, invert_mask=False
    ):
        RMBGNode = _try_import_rmbg()

        if RMBGNode is not None:
            try:
                rmbg = RMBGNode()
                # RMBG node typically takes image and returns (cutout_image, mask)
                result = rmbg.remove_background(image, model=rmbg_model)
                cutout = result[0]
                mask = result[1]

                # Apply threshold
                mask = (mask > threshold).float()

                if invert_mask:
                    mask = 1.0 - mask

                return (cutout, mask)
            except Exception as e:
                print(f"[Laura Studio] RMBG background removal failed: {e}")
                # Try simpler call signature
                try:
                    result = rmbg.execute(image)
                    return (result[0], result[1])
                except Exception:
                    pass
                print("[Laura Studio] Falling back to simple mask estimation")
                mask = _simple_bg_mask_fallback(image)
                if invert_mask:
                    mask = 1.0 - mask
                cutout = image * mask.unsqueeze(-1)
                return (cutout, mask)
        else:
            print(
                "[Laura Studio] RMBG not installed. Using simple elliptical mask fallback."
            )
            print("[Laura Studio] Install ComfyUI-RMBG for proper background removal.")
            mask = _simple_bg_mask_fallback(image)
            if invert_mask:
                mask = 1.0 - mask
            cutout = image * mask.unsqueeze(-1)
            return (cutout, mask)


# ============== BACKGROUND REPLACER ==============
class BackgroundReplacer:
    """Replace background with new image using foreground mask"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "foreground": ("IMAGE",),
                "mask": ("MASK",),
                "background": ("IMAGE",),
            },
            "optional": {
                "edge_feather": ("INT", {"default": 4, "min": 0, "max": 32}),
                "blend_mode": (["normal", "screen", "multiply"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "replace_background"
    CATEGORY = "Laura Studio/Background"
    DESCRIPTION = "Replace background using foreground mask"

    def replace_background(
        self, foreground, mask, background, edge_feather=4, blend_mode="normal"
    ):
        import torch.nn.functional as F

        # Ensure background matches foreground dimensions
        if background.shape[1:3] != foreground.shape[1:3]:
            bg = F.interpolate(
                background.permute(0, 3, 1, 2),
                size=(foreground.shape[1], foreground.shape[2]),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
        else:
            bg = background

        # Process mask
        m = mask.clone()
        if m.dim() == 2:
            m = m.unsqueeze(0)

        # Feather edges
        if edge_feather > 0:
            k = edge_feather * 2 + 1
            m_4d = m.unsqueeze(1)
            m_4d = F.avg_pool2d(m_4d, kernel_size=k, stride=1, padding=edge_feather)
            m = m_4d.squeeze(1)

        # Expand mask to [B, H, W, C]
        mask_rgb = m.unsqueeze(-1).expand_as(foreground)

        # Composite based on blend mode
        if blend_mode == "normal":
            result = foreground * mask_rgb + bg * (1 - mask_rgb)
        elif blend_mode == "screen":
            blended_bg = 1 - (1 - foreground) * (1 - bg)
            result = foreground * mask_rgb + blended_bg * (1 - mask_rgb)
        elif blend_mode == "multiply":
            blended_bg = foreground * bg
            result = foreground * mask_rgb + blended_bg * (1 - mask_rgb)
        else:
            result = foreground * mask_rgb + bg * (1 - mask_rgb)

        return (result.clamp(0, 1),)


# ============== BACKGROUND GENERATOR ==============
class BackgroundGenerator:
    """Generate background from prompt"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 8}),
                "height": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 4096, "step": 8},
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "professional studio background, gradient, soft lights",
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0}),
            },
            "optional": {
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "person, human, face, deformed, blurry, artifacts",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_background"
    CATEGORY = "Laura Studio/Background"
    DESCRIPTION = "Generate background from prompt"

    def generate_background(
        self,
        model,
        clip,
        vae,
        width,
        height,
        prompt,
        seed,
        steps,
        cfg,
        negative_prompt="person, human, face, deformed, blurry, artifacts",
    ):
        from nodes import EmptyLatentImage, KSampler, VAEDecode, CLIPTextEncode

        latent = EmptyLatentImage().generate(width, height, batch_size=1)[0]
        positive = CLIPTextEncode().encode(clip, prompt)[0]
        negative = CLIPTextEncode().encode(clip, negative_prompt)[0]

        sampled = KSampler().sample(
            model, seed, steps, cfg, "euler", "normal", positive, negative, latent
        )[0]

        result = VAEDecode().decode(vae, sampled)[0]
        return (result,)


# ============== PORTRAIT BOKEH ==============
class PortraitBokeh:
    """Apply professional bokeh effect to background using foreground mask"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "blur_amount": ("INT", {"default": 20, "min": 1, "max": 100}),
            },
            "optional": {
                "bokeh_quality": (["standard", "high"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_bokeh"
    CATEGORY = "Laura Studio/Background"
    DESCRIPTION = "Apply portrait bokeh effect to background"

    def apply_bokeh(self, image, mask, blur_amount, bokeh_quality="standard"):
        import torch.nn.functional as F

        # Process mask
        m = mask.clone()
        if m.dim() == 2:
            m = m.unsqueeze(0)
        mask_rgb = m.unsqueeze(-1).expand_as(image)

        # Apply Gaussian-like blur to background
        # Convert to BCHW for F.avg_pool2d
        img_bchw = image.permute(0, 3, 1, 2)

        # Multi-pass blur for smoother bokeh
        blurred = img_bchw
        passes = 3 if bokeh_quality == "high" else 1
        for _ in range(passes):
            k = blur_amount * 2 + 1
            pad = blur_amount
            blurred = F.avg_pool2d(blurred, kernel_size=k, stride=1, padding=pad)

        blurred = blurred.permute(0, 2, 3, 1)

        # Composite: sharp foreground + blurred background
        result = image * mask_rgb + blurred * (1 - mask_rgb)
        return (result.clamp(0, 1),)


# ============== SEAMLESS TILE ==============
class SeamlessTile:
    """Create seamless tileable background by blending edges"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["horizontal", "vertical", "both"],),
                "blend_width": ("INT", {"default": 128, "min": 16, "max": 512}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "make_seamless"
    CATEGORY = "Laura Studio/Background"
    DESCRIPTION = "Create seamless tileable background"

    def make_seamless(self, image, mode, blend_width):
        result = image.clone()
        B, H, W, C = result.shape

        if mode in ["horizontal", "both"]:
            # Blend left and right edges
            bw = min(blend_width, W // 4)
            left = result[:, :, :bw, :].clone()
            right = result[:, :, -bw:, :].clone()
            # Create linear blend weights
            weights = torch.linspace(0, 1, bw, device=image.device).view(1, 1, bw, 1)
            blended = left * weights + right * (1 - weights)
            result[:, :, :bw, :] = blended
            result[:, :, -bw:, :] = left * (1 - weights) + right * weights

        if mode in ["vertical", "both"]:
            bw = min(blend_width, H // 4)
            top = result[:, :bw, :, :].clone()
            bottom = result[:, -bw:, :, :].clone()
            weights = torch.linspace(0, 1, bw, device=image.device).view(1, bw, 1, 1)
            blended = top * weights + bottom * (1 - weights)
            result[:, :bw, :, :] = blended
            result[:, -bw:, :, :] = top * (1 - weights) + bottom * weights

        return (result,)


# ============== BACKGROUND COLORIZE ==============
class BackgroundColorize:
    """Add color tint to background using mask"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "color": ("STRING", {"default": "#FFE4C4"}),
                "intensity": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "colorize_background"
    CATEGORY = "Laura Studio/Background"
    DESCRIPTION = "Add color tint to background"

    def colorize_background(self, image, mask, color, intensity):
        # Parse hex color
        c = color.lstrip("#")
        if len(c) == 3:
            # Duplicate each character: #FFF -> #FFFFFF, #F0A -> #FF00AA
            c = c[0] + c[0] + c[1] + c[1] + c[2] + c[2]
        elif len(c) < 6:
            c = c + "0" * (6 - len(c))
        r = int(c[0:2], 16) / 255.0
        g = int(c[2:4], 16) / 255.0
        b = int(c[4:6], 16) / 255.0

        color_tensor = torch.tensor([r, g, b], device=image.device).view(1, 1, 1, 3)

        # Process mask - 1 = foreground, 0 = background
        m = mask.clone()
        if m.dim() == 2:
            m = m.unsqueeze(0)
        bg_mask = (1 - m).unsqueeze(-1).expand_as(image)

        # Tint background
        tinted_bg = image * (1 - intensity) + color_tensor * intensity
        result = image * (1 - bg_mask) + tinted_bg * bg_mask

        return (result.clamp(0, 1),)


# ============== PRO LIGHTING ==============
class ProLighting:
    """Apply professional lighting effects to subject"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "lighting_type": (
                    [
                        "rim_light",
                        "soft_box",
                        "hair_light",
                        "butterfly",
                        "split",
                        "rembrandt",
                    ],
                ),
                "intensity": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "color": ("STRING", {"default": "#FFFFFF"}),
                "direction": (["left", "right", "top", "bottom", "center"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_lighting"
    CATEGORY = "Laura Studio/Background"
    DESCRIPTION = "Apply professional lighting effects"

    def apply_lighting(
        self, image, mask, lighting_type, intensity, color="#FFFFFF", direction="right"
    ):
        # Parse color
        c = color.lstrip("#")
        if len(c) == 3:
            c = c[0] + c[0] + c[1] + c[1] + c[2] + c[2]
        elif len(c) < 6:
            c = c + "0" * (6 - len(c))
        lr = int(c[0:2], 16) / 255.0
        lg = int(c[2:4], 16) / 255.0
        lb = int(c[4:6], 16) / 255.0

        B, H, W, C = image.shape
        light_color = torch.tensor([lr, lg, lb], device=image.device).view(1, 1, 1, 3)

        # Create directional light gradient
        if direction == "left":
            grad = torch.linspace(1, 0, W, device=image.device).view(1, 1, W, 1)
        elif direction == "right":
            grad = torch.linspace(0, 1, W, device=image.device).view(1, 1, W, 1)
        elif direction == "top":
            grad = torch.linspace(1, 0, H, device=image.device).view(1, H, 1, 1)
        elif direction == "bottom":
            grad = torch.linspace(0, 1, H, device=image.device).view(1, H, 1, 1)
        else:  # center
            y = torch.linspace(-1, 1, H, device=image.device)
            x = torch.linspace(-1, 1, W, device=image.device)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            grad = (1.0 - (yy**2 + xx**2).sqrt().clamp(0, 1)).unsqueeze(0).unsqueeze(-1)

        # Process mask
        m = mask.clone()
        if m.dim() == 2:
            m = m.unsqueeze(0)
        fg_mask = m.unsqueeze(-1).expand_as(image)

        # Apply lighting based on type
        if lighting_type == "rim_light":
            # Edge detection on mask for rim effect
            import torch.nn.functional as F

            m4d = m.unsqueeze(1)
            eroded = -F.max_pool2d(-m4d, kernel_size=7, stride=1, padding=3)
            rim = (m4d - eroded).squeeze(1).unsqueeze(-1).expand_as(image)
            light = rim * light_color * intensity * grad
        elif lighting_type == "split":
            # Split lighting: one side lit, other dark
            light = grad * light_color * intensity * fg_mask
        elif lighting_type == "butterfly":
            # Top-down lighting
            top_grad = torch.linspace(1, 0.3, H, device=image.device).view(1, H, 1, 1)
            light = top_grad * light_color * intensity * fg_mask
        elif lighting_type == "rembrandt":
            # Triangle light on one side
            light = grad * light_color * intensity * fg_mask * 0.7
        else:
            # soft_box, hair_light - general directional
            light = grad * light_color * intensity * fg_mask

        result = (image + light).clamp(0, 1)
        return (result,)


# Register all background nodes
NODE_CLASS_MAPPINGS.update(
    {
        "BackgroundRemover": BackgroundRemover,
        "BackgroundReplacer": BackgroundReplacer,
        "BackgroundGenerator": BackgroundGenerator,
        "PortraitBokeh": PortraitBokeh,
        "SeamlessTile": SeamlessTile,
        "BackgroundColorize": BackgroundColorize,
        "ProLighting": ProLighting,
    }
)

NODE_DISPLAY_NAME_MAPPINGS.update(
    {
        "BackgroundRemover": "Background Remover (RMBG)",
        "BackgroundReplacer": "Background Replacer",
        "BackgroundGenerator": "Background Generator",
        "PortraitBokeh": "Portrait Bokeh",
        "SeamlessTile": "Seamless Tile",
        "BackgroundColorize": "Background Colorize",
        "ProLighting": "Pro Lighting",
    }
)
