"""
Laura Image Studio - Inpainting & Outpainting Nodes
Smart masking, inpainting, and image expansion
Delegates to SAM2/GroundingDINO for intelligent mask generation
"""

import torch
from PIL import Image
import numpy as np

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


# ============== EXTERNAL NODE IMPORT HELPERS ==============
def _try_import_sam2():
    """Try to import RMBG SAM2Segment node"""
    try:
        from custom_nodes import ComfyUI_RMBG
        mappings = ComfyUI_RMBG.NODE_CLASS_MAPPINGS
        if "SAM2Segment" in mappings:
            return mappings["SAM2Segment"]
    except Exception:
        pass
    try:
        import importlib
        mod = importlib.import_module("custom_nodes.ComfyUI-RMBG")
        mappings = mod.NODE_CLASS_MAPPINGS
        if "SAM2Segment" in mappings:
            return mappings["SAM2Segment"]
    except Exception:
        pass
    return None


def _try_import_grounding_dino():
    """Try to import RMBG GroundingDINO node for text-prompted detection"""
    try:
        from custom_nodes import ComfyUI_RMBG
        mappings = ComfyUI_RMBG.NODE_CLASS_MAPPINGS
        if "GroundingDINO" in mappings:
            return mappings["GroundingDINO"]
    except Exception:
        pass
    try:
        import importlib
        mod = importlib.import_module("custom_nodes.ComfyUI-RMBG")
        mappings = mod.NODE_CLASS_MAPPINGS
        if "GroundingDINO" in mappings:
            return mappings["GroundingDINO"]
    except Exception:
        pass
    return None


def _simple_center_mask(image, mode="subject"):
    """Simple fallback mask when SAM2/GroundingDINO is not available."""
    B, H, W, C = image.shape
    mask = torch.zeros((B, H, W), dtype=torch.float32, device=image.device)

    if mode == "subject":
        # Elliptical center region
        cy, cx = H / 2, W / 2
        ry, rx = H * 0.35, W * 0.30
        y = torch.arange(H, device=image.device).float()
        x = torch.arange(W, device=image.device).float()
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        ellipse = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
        mask[:] = (ellipse <= 1.0).float()
    elif mode == "background":
        # Everything except center
        cy, cx = H / 2, W / 2
        ry, rx = H * 0.35, W * 0.30
        y = torch.arange(H, device=image.device).float()
        x = torch.arange(W, device=image.device).float()
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        ellipse = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
        mask[:] = (ellipse > 1.0).float()
    else:
        # Full mask for "object" or "auto" modes
        mask[:] = 1.0

    return mask


# ============== SMART MASK GENERATOR ==============
class SmartMaskGenerator:
    """AI-powered mask generation using SAM2/GroundingDINO"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "text_prompt": ("STRING", {"multiline": True, "default": "person"}),
            },
            "optional": {
                "mode": (["text_detect", "subject", "background"],),
                "detection_threshold": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 0.9}),
                "expand_mask": ("INT", {"default": 4, "min": 0, "max": 100}),
                "invert": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "preview")
    FUNCTION = "generate_mask"
    CATEGORY = "Laura Studio/Inpainting"
    DESCRIPTION = "Generate masks using SAM2/GroundingDINO text-prompted detection"

    def generate_mask(self, image, text_prompt,
                      mode="text_detect", detection_threshold=0.3,
                      expand_mask=4, invert=False):

        mask = None

        if mode == "text_detect":
            # Try GroundingDINO for text-prompted object detection
            GDNode = _try_import_grounding_dino()
            if GDNode is not None:
                try:
                    gd = GDNode()
                    result = gd.detect(image, text_prompt, threshold=detection_threshold)
                    # Returns bounding boxes; use SAM2 for precise masks
                    boxes = result[0]

                    SAM2Node = _try_import_sam2()
                    if SAM2Node is not None:
                        sam2 = SAM2Node()
                        seg_result = sam2.segment(image, boxes=boxes)
                        mask = seg_result[0]
                    else:
                        # Convert bounding boxes to rectangular mask
                        B, H, W, C = image.shape
                        mask = torch.zeros((B, H, W), dtype=torch.float32, device=image.device)
                        if hasattr(boxes, 'shape') and len(boxes) > 0:
                            for box in boxes:
                                x1, y1, x2, y2 = [int(v) for v in box[:4]]
                                mask[:, y1:y2, x1:x2] = 1.0
                except Exception as e:
                    print(f"[Laura Studio] GroundingDINO detection failed: {e}")

            # Try SAM2 directly if GroundingDINO failed
            if mask is None:
                SAM2Node = _try_import_sam2()
                if SAM2Node is not None:
                    try:
                        sam2 = SAM2Node()
                        seg_result = sam2.segment(image, text_prompt=text_prompt)
                        mask = seg_result[0]
                    except Exception as e:
                        print(f"[Laura Studio] SAM2 segmentation failed: {e}")

        if mask is None:
            # Fallback to simple mode-based mask
            if mode == "subject":
                mask = _simple_center_mask(image, "subject")
            elif mode == "background":
                mask = _simple_center_mask(image, "background")
            else:
                print("[Laura Studio] SAM2/GroundingDINO not available. Using center fallback mask.")
                mask = _simple_center_mask(image, "subject")

        # Expand mask
        if expand_mask > 0:
            import torch.nn.functional as F
            k = expand_mask * 2 + 1
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            mask = F.max_pool2d(mask, kernel_size=k, stride=1, padding=expand_mask)
            mask = mask.squeeze(1)

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        # Invert if requested
        if invert:
            mask = 1.0 - mask

        # Create preview with mask overlay
        mask_rgb = mask.unsqueeze(-1).expand_as(image)
        preview = image * 0.5 + image * mask_rgb * 0.3
        red_tint = torch.zeros_like(image)
        red_tint[..., 0] = 0.4
        preview = preview + red_tint * mask_rgb * 0.3
        preview = preview.clamp(0, 1)

        return (mask, preview)


# ============== MANUAL MASK EDITOR ==============
class ManualMaskEditor:
    """Fine-tune masks with real operations"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "mode": (["invert", "expand", "contract", "feather", "smooth", "threshold"],),
                "amount": ("INT", {"default": 10, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "edit_mask"
    CATEGORY = "Laura Studio/Inpainting"
    DESCRIPTION = "Edit mask with real morphological operations"

    def edit_mask(self, mask, mode, amount):
        import torch.nn.functional as F

        m = mask.clone()
        if m.dim() == 2:
            m = m.unsqueeze(0)

        if mode == "invert":
            return (1.0 - m,)

        # Convert to 4D for pooling operations
        m4d = m.unsqueeze(1)
        k = amount * 2 + 1

        if mode == "expand":
            # Dilation via max_pool
            result = F.max_pool2d(m4d, kernel_size=k, stride=1, padding=amount)
        elif mode == "contract":
            # Erosion via negative max_pool
            result = -F.max_pool2d(-m4d, kernel_size=k, stride=1, padding=amount)
        elif mode == "feather":
            # Gaussian-like blur
            result = F.avg_pool2d(m4d, kernel_size=k, stride=1, padding=amount)
        elif mode == "smooth":
            # Multiple blur passes for smoother result
            result = m4d
            for _ in range(3):
                sk = min(amount, 5) * 2 + 1
                sp = min(amount, 5)
                result = F.avg_pool2d(result, kernel_size=sk, stride=1, padding=sp)
        elif mode == "threshold":
            # Binary threshold at amount/100
            threshold = amount / 100.0
            result = (m4d > threshold).float()
        else:
            result = m4d

        return (result.squeeze(1),)


# ============== INPAINTER ==============
class LauraInpainter:
    """Fill masked areas with AI-generated content using proper mask processing"""

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
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 20.0}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": "deformed, blurry, artifacts, low quality"}),
                "edge_feather": ("INT", {"default": 8, "min": 0, "max": 50}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "inpaint"
    CATEGORY = "Laura Studio/Inpainting"
    DESCRIPTION = "Inpaint masked areas with proper mask-to-latent processing"

    def inpaint(self, model, clip, vae, image, mask, positive_prompt,
                seed, steps, cfg, denoise,
                negative_prompt="deformed, blurry, artifacts, low quality",
                edge_feather=8):

        from nodes import VAEEncode, KSampler, VAEDecode, CLIPTextEncode
        import torch.nn.functional as F

        # Encode image to latent
        encoded = VAEEncode().encode(vae, image)[0]

        # Process mask to latent dimensions
        m = mask.clone()
        if m.dim() == 2:
            m = m.unsqueeze(0).unsqueeze(0)
        elif m.dim() == 3:
            m = m.unsqueeze(1)

        latent_h = encoded["samples"].shape[2]
        latent_w = encoded["samples"].shape[3]
        mask_latent = F.interpolate(m.float(), size=(latent_h, latent_w),
                                     mode="bilinear", align_corners=False)

        # Encode prompts
        positive = CLIPTextEncode().encode(clip, positive_prompt)[0]
        negative = CLIPTextEncode().encode(clip, negative_prompt)[0]

        # Apply noise to masked region
        noise = torch.randn_like(encoded["samples"])
        latent_samples = encoded["samples"] * (1 - mask_latent) + noise * mask_latent

        latent = {"samples": latent_samples, "noise_mask": mask_latent.squeeze(1)}

        # Sample
        sampled = KSampler().sample(
            model, seed, steps, cfg, "dpmpp_2m", "karras",
            positive, negative, latent, denoise=denoise
        )[0]

        # Decode
        result = VAEDecode().decode(vae, sampled)[0]

        # Composite with feathered edge
        comp_mask = mask.clone()
        if comp_mask.dim() == 2:
            comp_mask = comp_mask.unsqueeze(0)

        if edge_feather > 0:
            k = edge_feather * 2 + 1
            cm4d = comp_mask.unsqueeze(1)
            cm4d = F.avg_pool2d(cm4d, kernel_size=k, stride=1, padding=edge_feather)
            comp_mask = cm4d.squeeze(1)

        mask_rgb = comp_mask.unsqueeze(-1).expand_as(image)
        final = result * mask_rgb + image * (1 - mask_rgb)

        return (final,)


# ============== OUTPAINTER ==============
class LauraOutpainter:
    """Expand image boundaries with AI-generated content"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "direction": (["all", "top", "bottom", "left", "right",
                               "top-bottom", "left-right"],),
                "pixels": ("INT", {"default": 256, "min": 64, "max": 1024, "step": 8}),
                "positive_prompt": ("STRING", {"multiline": True, "default": "detailed background, professional photography"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 20.0}),
                "denoise": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": "deformed, blurry, artifacts"}),
                "blend_width": ("INT", {"default": 64, "min": 16, "max": 256}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "new_width", "new_height")
    FUNCTION = "outpaint"
    CATEGORY = "Laura Studio/Inpainting"
    DESCRIPTION = "Expand image boundaries with AI content"

    def outpaint(self, model, clip, vae, image, direction, pixels,
                 positive_prompt, seed, steps, cfg, denoise,
                 negative_prompt="deformed, blurry, artifacts",
                 blend_width=64):

        from nodes import VAEEncode, KSampler, VAEDecode, CLIPTextEncode
        import torch.nn.functional as F

        B, H, W, C = image.shape

        # Calculate padding for each side
        pad_top = pad_bottom = pad_left = pad_right = 0
        if direction == "all":
            pad_top = pad_bottom = pad_left = pad_right = pixels
        elif direction == "top":
            pad_top = pixels
        elif direction == "bottom":
            pad_bottom = pixels
        elif direction == "left":
            pad_left = pixels
        elif direction == "right":
            pad_right = pixels
        elif direction == "top-bottom":
            pad_top = pad_bottom = pixels
        elif direction == "left-right":
            pad_left = pad_right = pixels

        new_H = H + pad_top + pad_bottom
        new_W = W + pad_left + pad_right

        # Round to nearest multiple of 8
        new_H = ((new_H + 7) // 8) * 8
        new_W = ((new_W + 7) // 8) * 8

        # Create expanded canvas (filled with edge-extended pixels)
        expanded = torch.zeros((B, new_H, new_W, C), dtype=image.dtype, device=image.device)

        # Place original image
        expanded[:, pad_top:pad_top + H, pad_left:pad_left + W, :] = image

        # Edge-extend into padding areas
        if pad_top > 0:
            expanded[:, :pad_top, pad_left:pad_left + W, :] = image[:, 0:1, :, :].expand(B, pad_top, W, C)
        if pad_bottom > 0:
            expanded[:, pad_top + H:, pad_left:pad_left + W, :] = image[:, -1:, :, :].expand(B, new_H - pad_top - H, W, C)
        if pad_left > 0:
            expanded[:, pad_top:pad_top + H, :pad_left, :] = image[:, :, 0:1, :].expand(B, H, pad_left, C)
        if pad_right > 0:
            expanded[:, pad_top:pad_top + H, pad_left + W:, :] = image[:, :, -1:, :].expand(B, H, new_W - pad_left - W, C)

        # Create mask: 1 where we need to generate, 0 where original image is
        outpaint_mask = torch.ones((B, new_H, new_W), dtype=torch.float32, device=image.device)
        outpaint_mask[:, pad_top:pad_top + H, pad_left:pad_left + W] = 0.0

        # Add blend zone
        bw = min(blend_width, pixels // 2)
        if pad_top > 0 and bw > 0:
            blend = torch.linspace(1, 0, bw, device=image.device).view(1, bw, 1)
            outpaint_mask[:, pad_top:pad_top + bw, pad_left:pad_left + W] = blend
        if pad_bottom > 0 and bw > 0:
            blend = torch.linspace(0, 1, bw, device=image.device).view(1, bw, 1)
            outpaint_mask[:, pad_top + H - bw:pad_top + H, pad_left:pad_left + W] = blend
        if pad_left > 0 and bw > 0:
            blend = torch.linspace(1, 0, bw, device=image.device).view(1, 1, bw)
            outpaint_mask[:, pad_top:pad_top + H, pad_left:pad_left + bw] = blend
        if pad_right > 0 and bw > 0:
            blend = torch.linspace(0, 1, bw, device=image.device).view(1, 1, bw)
            outpaint_mask[:, pad_top:pad_top + H, pad_left + W - bw:pad_left + W] = blend

        # Encode expanded canvas
        encoded = VAEEncode().encode(vae, expanded)[0]

        # Process mask to latent size
        m4d = outpaint_mask.unsqueeze(1)
        latent_h = encoded["samples"].shape[2]
        latent_w = encoded["samples"].shape[3]
        mask_latent = F.interpolate(m4d, size=(latent_h, latent_w), mode="bilinear", align_corners=False)

        # Encode prompts
        positive = CLIPTextEncode().encode(clip, positive_prompt)[0]
        negative = CLIPTextEncode().encode(clip, negative_prompt)[0]

        # Inpaint the expanded areas
        noise = torch.randn_like(encoded["samples"])
        latent_samples = encoded["samples"] * (1 - mask_latent) + noise * mask_latent
        latent = {"samples": latent_samples, "noise_mask": mask_latent.squeeze(1)}

        sampled = KSampler().sample(
            model, seed, steps, cfg, "dpmpp_2m", "karras",
            positive, negative, latent, denoise=denoise
        )[0]

        result = VAEDecode().decode(vae, sampled)[0]

        return (result, new_W, new_H)


# ============== EDGE BLENDER ==============
class EdgeBlender:
    """Blend edges seamlessly between original and modified images"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "modified_image": ("IMAGE",),
                "mask": ("MASK",),
                "blend_width": ("INT", {"default": 32, "min": 4, "max": 256}),
            },
            "optional": {
                "blend_mode": (["linear", "gaussian", "smooth"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend_edges"
    CATEGORY = "Laura Studio/Inpainting"
    DESCRIPTION = "Blend edges seamlessly between images"

    def blend_edges(self, original_image, modified_image, mask, blend_width,
                    blend_mode="linear"):
        import torch.nn.functional as F

        m = mask.clone()
        if m.dim() == 2:
            m = m.unsqueeze(0)

        # Create feathered blend mask
        m4d = m.unsqueeze(1)
        k = blend_width * 2 + 1

        if blend_mode == "gaussian" or blend_mode == "smooth":
            # Multiple passes for smoother blend
            blended = m4d
            passes = 3 if blend_mode == "smooth" else 2
            for _ in range(passes):
                blended = F.avg_pool2d(blended, kernel_size=k, stride=1, padding=blend_width)
        else:
            # Single pass linear blend
            blended = F.avg_pool2d(m4d, kernel_size=k, stride=1, padding=blend_width)

        blend_mask = blended.squeeze(1).unsqueeze(-1).expand_as(original_image)
        result = modified_image * blend_mask + original_image * (1 - blend_mask)

        return (result,)


# ============== OBJECT REMOVER ==============
class ObjectRemover:
    """Remove objects using text-prompted detection + inpainting"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "object_prompt": ("STRING", {"multiline": True, "default": "car"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 20.0}),
            },
            "optional": {
                "expand_mask": ("INT", {"default": 8, "min": 0, "max": 50}),
                "fill_prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "remove_object"
    CATEGORY = "Laura Studio/Inpainting"
    DESCRIPTION = "Remove objects using text detection + inpainting"

    def remove_object(self, model, clip, vae, image, object_prompt,
                      seed, steps, cfg, expand_mask=8, fill_prompt=""):

        # Step 1: Detect the object with SmartMaskGenerator
        mask_gen = SmartMaskGenerator()
        obj_mask, _ = mask_gen.generate_mask(
            image, text_prompt=object_prompt,
            mode="text_detect", detection_threshold=0.3,
            expand_mask=expand_mask
        )

        # Step 2: Inpaint the masked area
        inpainter = LauraInpainter()
        fill = fill_prompt if fill_prompt else "clean background, seamless, natural"
        result = inpainter.inpaint(
            model, clip, vae, image, obj_mask, fill,
            seed, steps, cfg, denoise=0.95,
            negative_prompt=f"deformed, blurry, {object_prompt}, artifacts",
            edge_feather=8
        )

        return result


# ============== REGION INPAINTER ==============
class RegionInpainter:
    """Inpaint specific regions with different prompts"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "mask_1": ("MASK",),
                "prompt_1": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 20.0}),
            },
            "optional": {
                "mask_2": ("MASK",),
                "prompt_2": ("STRING", {"multiline": True, "default": ""}),
                "mask_3": ("MASK",),
                "prompt_3": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "region_inpaint"
    CATEGORY = "Laura Studio/Inpainting"
    DESCRIPTION = "Inpaint different regions with different prompts"

    def region_inpaint(self, model, clip, vae, image, mask_1, prompt_1,
                       seed, steps, cfg,
                       mask_2=None, prompt_2="",
                       mask_3=None, prompt_3=""):

        inpainter = LauraInpainter()
        result = image

        # Process each region sequentially
        regions = [(mask_1, prompt_1)]
        if mask_2 is not None and prompt_2:
            regions.append((mask_2, prompt_2))
        if mask_3 is not None and prompt_3:
            regions.append((mask_3, prompt_3))

        for i, (region_mask, region_prompt) in enumerate(regions):
            result = inpainter.inpaint(
                model, clip, vae, result, region_mask, region_prompt,
                seed + i, steps, cfg, denoise=0.9
            )[0]

        return (result,)


# Register all inpainting nodes
NODE_CLASS_MAPPINGS.update({
    "SmartMaskGenerator": SmartMaskGenerator,
    "ManualMaskEditor": ManualMaskEditor,
    "LauraInpainter": LauraInpainter,
    "LauraOutpainter": LauraOutpainter,
    "EdgeBlender": EdgeBlender,
    "ObjectRemover": ObjectRemover,
    "RegionInpainter": RegionInpainter,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "SmartMaskGenerator": "Smart Mask Generator (SAM2)",
    "ManualMaskEditor": "Manual Mask Editor",
    "LauraInpainter": "LAURA Inpainter",
    "LauraOutpainter": "LAURA Outpainter",
    "EdgeBlender": "Edge Blender",
    "ObjectRemover": "Object Remover (Auto-Detect)",
    "RegionInpainter": "Region Inpainter",
})
