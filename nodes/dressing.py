"""
Laura Image Studio - Virtual Dressing Nodes
Individual clothing item modification with AI
Delegates to RMBG ClothesSegment for real segmentation
"""

import torch

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


# ============== RMBG INTEGRATION HELPERS ==============
def _try_import_rmbg_clothes():
    """Try to import RMBG ClothesSegment node"""
    try:
        from custom_nodes import ComfyUI_RMBG

        mappings = ComfyUI_RMBG.NODE_CLASS_MAPPINGS
        if "ClothesSegment" in mappings:
            return mappings["ClothesSegment"]
    except Exception:
        pass
    # Try alternative import path
    try:
        import importlib

        mod = importlib.import_module("custom_nodes.ComfyUI-RMBG")
        mappings = mod.NODE_CLASS_MAPPINGS
        if "ClothesSegment" in mappings:
            return mappings["ClothesSegment"]
    except Exception:
        pass
    return None


def _try_import_rmbg_fashion():
    """Try to import RMBG FashionSegmentation node"""
    try:
        from custom_nodes import ComfyUI_RMBG

        mappings = ComfyUI_RMBG.NODE_CLASS_MAPPINGS
        if "FashionSegmentation" in mappings:
            return mappings["FashionSegmentation"]
    except Exception:
        pass
    try:
        import importlib

        mod = importlib.import_module("custom_nodes.ComfyUI-RMBG")
        mappings = mod.NODE_CLASS_MAPPINGS
        if "FashionSegmentation" in mappings:
            return mappings["FashionSegmentation"]
    except Exception:
        pass
    return None


def _try_import_ipadapter():
    """Try to import IPAdapter Plus nodes"""
    try:
        from custom_nodes import ComfyUI_IPAdapter_plus

        mappings = ComfyUI_IPAdapter_plus.NODE_CLASS_MAPPINGS
        loader = mappings.get("IPAdapterUnifiedLoader") or mappings.get(
            "IPAdapterUnifiedLoaderFaceID"
        )
        apply_node = mappings.get("IPAdapterAdvanced") or mappings.get("IPAdapter")
        return loader, apply_node
    except Exception:
        pass
    try:
        import importlib

        mod = importlib.import_module("custom_nodes.ComfyUI_IPAdapter_plus")
        mappings = mod.NODE_CLASS_MAPPINGS
        loader = mappings.get("IPAdapterUnifiedLoader") or mappings.get(
            "IPAdapterUnifiedLoaderFaceID"
        )
        apply_node = mappings.get("IPAdapterAdvanced") or mappings.get("IPAdapter")
        return loader, apply_node
    except Exception:
        pass
    return None, None


# RMBG ClothesSegment category mapping (18 categories)
# 0: Background, 1: Hat, 2: Hair, 3: Sunglasses, 4: Upper-clothes,
# 5: Skirt, 6: Pants, 7: Dress, 8: Belt, 9: Left-shoe, 10: Right-shoe,
# 11: Face, 12: Left-leg, 13: Right-leg, 14: Left-arm, 15: Right-arm,
# 16: Bag, 17: Scarf
RMBG_CATEGORY_MAP = {
    "top": [4],  # Upper-clothes
    "bottom": [5, 6],  # Skirt + Pants
    "dress": [7],  # Dress
    "skirt": [5],  # Skirt only
    "pants": [6],  # Pants only
    "shoes": [9, 10],  # Left-shoe + Right-shoe
    "left_shoe": [9],  # Left shoe only
    "right_shoe": [10],  # Right shoe only
    "hat": [1],  # Hat
    "sunglasses": [3],  # Sunglasses
    "belt": [8],  # Belt
    "bag": [16],  # Bag
    "scarf": [17],  # Scarf
    "hair": [2],  # Hair
    "face": [11],  # Face
    "left_leg": [12],  # Left leg only
    "right_leg": [13],  # Right leg only
    "left_arm": [14],  # Left arm only
    "right_arm": [15],  # Right arm only
    "full_body": [4, 5, 6, 7, 9, 10],  # All clothing
}


def _extract_category_mask(segmentation_output, category):
    """Extract a binary mask for a specific clothing category from RMBG output.

    segmentation_output: tensor from RMBG ClothesSegment [B, H, W] with integer labels
    category: string key from RMBG_CATEGORY_MAP
    Returns: binary mask tensor [B, H, W] as float
    """
    indices = RMBG_CATEGORY_MAP.get(category, [4])  # default to upper-clothes
    mask = torch.zeros_like(segmentation_output, dtype=torch.float32)
    for idx in indices:
        mask = mask + (segmentation_output == idx).float()
    return mask.clamp(0, 1)


def _region_fallback_mask(image, category):
    """Create a rough region-based mask when RMBG is not available.

    Uses simple spatial heuristics based on typical human proportions.
    image: [B, H, W, C] tensor
    Returns: [B, H, W] mask tensor
    """
    B, H, W, C = image.shape
    mask = torch.zeros((B, H, W), dtype=torch.float32, device=image.device)

    region_map = {
        "top": (0.15, 0.50),  # Upper body region
        "bottom": (0.45, 0.80),  # Lower body
        "dress": (0.15, 0.80),  # Full torso
        "skirt": (0.45, 0.70),
        "pants": (0.45, 0.85),
        "shoes": (0.85, 1.00),  # Feet area
        "left_shoe": (0.85, 1.00, "left"),
        "right_shoe": (0.85, 1.00, "right"),
        "hat": (0.00, 0.12),  # Head top
        "sunglasses": (0.10, 0.18),  # Eyes area
        "belt": (0.42, 0.50),
        "bag": (0.30, 0.60),
        "scarf": (0.12, 0.25),
        "hair": (0.00, 0.20),  # Head area
        "face": (0.08, 0.25),  # Face area
        "full_body": (0.15, 0.85),
        "left_leg": (0.65, 0.90, "left"),
        "right_leg": (0.65, 0.90, "right"),
        "left_arm": (0.20, 0.55, "left"),
        "right_arm": (0.20, 0.55, "right"),
        # Hand regions - approximate positions
        "left_hand": (0.30, 0.55, "left_hand"),
        "right_hand": (0.30, 0.55, "right_hand"),
    }

    region = region_map.get(category, (0.15, 0.50))
    y_start_pct, y_end_pct = region[0], region[1]
    y_start = int(H * y_start_pct)
    y_end = int(H * y_end_pct)

    # Handle left/right specific regions
    if len(region) == 3:
        side = region[2]
        if side == "left":
            # Left side of image (viewer's left = subject's right)
            x_margin = int(W * 0.50)
            mask[:, y_start:y_end, x_margin:W] = 1.0
        elif side == "right":
            # Right side of image (viewer's right = subject's left)
            x_margin = int(W * 0.50)
            mask[:, y_start:y_end, 0:x_margin] = 1.0
        elif side == "left_hand":
            # Lower left quadrant - arm extended
            x_margin = int(W * 0.60)
            mask[:, int(H * 0.35) : int(H * 0.55), int(W * 0.40) : x_margin] = 1.0
        elif side == "right_hand":
            # Lower right quadrant - arm extended
            x_margin = int(W * 0.40)
            mask[:, int(H * 0.35) : int(H * 0.55), x_margin : int(W * 0.60)] = 1.0
    else:
        # Center region for most items
        x_margin = int(W * 0.20)
        mask[:, y_start:y_end, x_margin : W - x_margin] = 1.0

    return mask


# ============== CLOTHING SEGMENTOR ==============
class ClothingSegmentor:
    """Detect and segment clothing items using RMBG ClothesSegment"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "category": (
                    [
                        "top",
                        "bottom",
                        "dress",
                        "skirt",
                        "pants",
                        "shoes",
                        "hat",
                        "sunglasses",
                        "belt",
                        "bag",
                        "scarf",
                        "hair",
                        "face",
                        "full_body",
                    ],
                ),
            },
            "optional": {
                "expand_mask": ("INT", {"default": 4, "min": 0, "max": 50}),
                "feather": ("INT", {"default": 2, "min": 0, "max": 20}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "preview")
    FUNCTION = "segment_clothing"
    CATEGORY = "Laura Studio/Dressing"
    DESCRIPTION = "Segment clothing items using RMBG (18 categories)"

    def segment_clothing(self, image, category, expand_mask=4, feather=2):
        ClothesSegmentNode = _try_import_rmbg_clothes()

        if ClothesSegmentNode is not None:
            # Delegate to real RMBG ClothesSegment
            try:
                seg_node = ClothesSegmentNode()
                seg_result = seg_node.segment(image)
                # seg_result is typically (segmentation_map, ...)
                seg_map = seg_result[0]
                mask = _extract_category_mask(seg_map, category)
            except Exception as e:
                print(
                    f"[Laura Studio] RMBG ClothesSegment failed: {e}, using region fallback"
                )
                mask = _region_fallback_mask(image, category)
        else:
            print("[Laura Studio] RMBG not found, using region-based fallback mask")
            mask = _region_fallback_mask(image, category)

        # Expand mask if requested
        if expand_mask > 0:
            import torch.nn.functional as F

            k = expand_mask * 2 + 1
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            mask = F.max_pool2d(mask, kernel_size=k, stride=1, padding=expand_mask)
            mask = mask.squeeze(1)

        # Feather edges
        if feather > 0:
            import torch.nn.functional as F

            k = feather * 2 + 1
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            mask = F.avg_pool2d(mask, kernel_size=k, stride=1, padding=feather)
            mask = mask.squeeze(1)

        # Ensure mask is [B, H, W]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        # Create preview: overlay mask on image
        mask_rgb = mask.unsqueeze(-1).expand_as(image)
        preview = image * 0.5 + image * mask_rgb * 0.5
        # Tint masked area green for visibility
        green_tint = torch.zeros_like(image)
        green_tint[..., 1] = 0.3
        preview = preview + green_tint * mask_rgb * 0.3
        preview = preview.clamp(0, 1)

        return (mask, preview)


# ============== ACCESSORY DETECTOR (NEW) ==============
class AccessoryDetector:
    """Detect accessories using RMBG FashionSegmentation"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "accessory": (
                    [
                        "shoes",
                        "watch",
                        "bag",
                        "belt",
                        "sunglasses",
                        "hat",
                        "scarf",
                        "jewelry",
                    ],
                ),
            },
            "optional": {
                "expand_mask": ("INT", {"default": 4, "min": 0, "max": 50}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "preview")
    FUNCTION = "detect_accessory"
    CATEGORY = "Laura Studio/Dressing"
    DESCRIPTION = "Detect accessories using RMBG FashionSegmentation"

    def detect_accessory(self, image, accessory, expand_mask=4):
        FashionSegNode = _try_import_rmbg_fashion()

        # Map our accessory names to RMBG categories
        # Note: "watch" and "jewelry" have no direct RMBG segment label,
        # so we combine left-arm + right-arm regions as the best proxy.
        accessory_to_rmbg = {
            "shoes": "shoes",
            "watch": "left_arm",
            "bag": "bag",
            "belt": "belt",
            "sunglasses": "sunglasses",
            "hat": "hat",
            "scarf": "scarf",
            "jewelry": "face",
        }
        rmbg_cat = accessory_to_rmbg.get(accessory, "bag")

        if FashionSegNode is not None:
            try:
                seg_node = FashionSegNode()
                seg_result = seg_node.segment(image)
                seg_map = seg_result[0]
                mask = _extract_category_mask(seg_map, rmbg_cat)
            except Exception as e:
                print(
                    f"[Laura Studio] RMBG FashionSegmentation failed: {e}, using region fallback"
                )
                mask = _region_fallback_mask(image, rmbg_cat)
        else:
            print("[Laura Studio] RMBG not found, using region-based fallback mask")
            mask = _region_fallback_mask(image, rmbg_cat)

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

        # Preview
        mask_rgb = mask.unsqueeze(-1).expand_as(image)
        preview = image * 0.5 + image * mask_rgb * 0.5
        blue_tint = torch.zeros_like(image)
        blue_tint[..., 2] = 0.3
        preview = preview + blue_tint * mask_rgb * 0.3
        preview = preview.clamp(0, 1)

        return (mask, preview)


# ============== VIRTUAL DRESSER ==============
class VirtualDresser:
    """Replace clothing items using real mask-based inpainting"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "source_image": ("IMAGE",),
                "clothing_mask": ("MASK",),
                "clothing_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "elegant white blouse, professional",
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 20.0}),
                "denoise": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "deformed, blurry, bad anatomy, wrong clothes, low quality",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "dress"
    CATEGORY = "Laura Studio/Dressing"
    DESCRIPTION = "Replace clothing via mask-based inpainting"

    def dress(
        self,
        model,
        clip,
        vae,
        source_image,
        clothing_mask,
        clothing_prompt,
        seed,
        steps,
        cfg,
        denoise,
        negative_prompt="deformed, blurry, bad anatomy, wrong clothes, low quality",
    ):
        from nodes import VAEEncode, KSampler, VAEDecode, CLIPTextEncode
        import torch.nn.functional as F

        # Encode source image to latent
        encoded = VAEEncode().encode(vae, source_image)[0]

        # Process mask to latent size
        mask = clothing_mask
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)

        latent_h = encoded["samples"].shape[2]
        latent_w = encoded["samples"].shape[3]
        mask_latent = F.interpolate(
            mask.float(),
            size=(latent_h, latent_w),
            mode="bilinear",
            align_corners=False,
        )

        # Build prompt with quality tags
        full_prompt = f"photo of person wearing {clothing_prompt}, same pose, same body, detailed, professional lighting, 8k"

        # Encode prompts
        positive = CLIPTextEncode().encode(clip, full_prompt)[0]
        negative = CLIPTextEncode().encode(clip, negative_prompt)[0]

        # Set up noise mask for inpainting (noise only in masked area)
        noise = torch.randn_like(encoded["samples"])
        latent_samples = encoded["samples"] * (1 - mask_latent) + noise * mask_latent

        latent = {"samples": latent_samples, "noise_mask": mask_latent.squeeze(1)}

        # Sample
        sampled = KSampler().sample(
            model,
            seed,
            steps,
            cfg,
            "dpmpp_2m",
            "karras",
            positive,
            negative,
            latent,
            denoise=denoise,
        )[0]

        # Decode
        result = VAEDecode().decode(vae, sampled)[0]

        # Composite: keep original outside mask, use inpainted inside mask
        mask_full = clothing_mask
        if mask_full.dim() == 2:
            mask_full = mask_full.unsqueeze(0)
        mask_rgb = mask_full.unsqueeze(-1).expand_as(source_image)

        # Feather the compositing edge
        if mask_rgb.shape[1] != result.shape[1] or mask_rgb.shape[2] != result.shape[2]:
            mask_rgb = F.interpolate(
                mask_rgb.permute(0, 3, 1, 2),
                size=(result.shape[1], result.shape[2]),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)

        final = result * mask_rgb + source_image * (1 - mask_rgb)

        return (final,)


# ============== DRESSING ROOM COMPOSITOR (NEW) ==============
class DressingRoomCompositor:
    """Piece-by-piece composite: combine up to 6 individually modified items"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
            },
            "optional": {
                "item_1_image": ("IMAGE",),
                "item_1_mask": ("MASK",),
                "item_2_image": ("IMAGE",),
                "item_2_mask": ("MASK",),
                "item_3_image": ("IMAGE",),
                "item_3_mask": ("MASK",),
                "item_4_image": ("IMAGE",),
                "item_4_mask": ("MASK",),
                "item_5_image": ("IMAGE",),
                "item_5_mask": ("MASK",),
                "item_6_image": ("IMAGE",),
                "item_6_mask": ("MASK",),
                "blend_mode": (["alpha", "feathered"],),
                "feather_amount": ("INT", {"default": 4, "min": 0, "max": 20}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "composite"
    CATEGORY = "Laura Studio/Dressing"
    DESCRIPTION = "Composite up to 6 individually modified items onto base image"

    def composite(
        self,
        base_image,
        blend_mode="alpha",
        feather_amount=4,
        item_1_image=None,
        item_1_mask=None,
        item_2_image=None,
        item_2_mask=None,
        item_3_image=None,
        item_3_mask=None,
        item_4_image=None,
        item_4_mask=None,
        item_5_image=None,
        item_5_mask=None,
        item_6_image=None,
        item_6_mask=None,
    ):
        import torch.nn.functional as F

        result = base_image.clone()

        items = [
            (item_1_image, item_1_mask),
            (item_2_image, item_2_mask),
            (item_3_image, item_3_mask),
            (item_4_image, item_4_mask),
            (item_5_image, item_5_mask),
            (item_6_image, item_6_mask),
        ]

        for item_image, item_mask in items:
            if item_image is None or item_mask is None:
                continue

            mask = item_mask.clone()

            # Feather the mask edges
            if blend_mode == "feathered" and feather_amount > 0:
                k = feather_amount * 2 + 1
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0).unsqueeze(0)
                elif mask.dim() == 3:
                    mask = mask.unsqueeze(1)
                mask = F.avg_pool2d(
                    mask, kernel_size=k, stride=1, padding=feather_amount
                )
                mask = mask.squeeze(1)

            # Ensure mask is [B, H, W]
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)

            # Expand mask to match image channels [B, H, W, C]
            mask_rgb = mask.unsqueeze(-1).expand_as(result)

            # Resize item_image if dimensions don't match
            if item_image.shape[1:3] != result.shape[1:3]:
                item_image = F.interpolate(
                    item_image.permute(0, 3, 1, 2),
                    size=(result.shape[1], result.shape[2]),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)

            # Composite this item
            result = item_image * mask_rgb + result * (1 - mask_rgb)

        return (result,)


# ============== HAIR STYLIST ==============
class HairStylist:
    """Change hairstyle using hair segmentation mask + inpainting"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "source_image": ("IMAGE",),
                "hairstyle": (
                    [
                        "short",
                        "long",
                        "curly",
                        "straight",
                        "wavy",
                        "bob",
                        "ponytail",
                        "braided",
                        "updo",
                        "pixie",
                    ],
                ),
                "hair_color": ("STRING", {"default": "natural brown"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 20.0}),
                "denoise": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "hair_prompt": ("STRING", {"multiline": True, "default": ""}),
                "bangs": (["none", "side bangs", "straight bangs", "curtain bangs"],),
                "highlights": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "style_hair"
    CATEGORY = "Laura Studio/Dressing"
    DESCRIPTION = "Change hairstyle and hair color via segmentation + inpainting"

    def style_hair(
        self,
        model,
        clip,
        vae,
        source_image,
        hairstyle,
        hair_color,
        seed,
        steps,
        cfg,
        denoise,
        hair_prompt="",
        bangs="none",
        highlights="",
    ):
        # First, get hair mask using ClothingSegmentor
        seg = ClothingSegmentor()
        hair_mask, _ = seg.segment_clothing(
            source_image, "hair", expand_mask=8, feather=4
        )

        # Build prompt
        style_desc = f"{hairstyle} hairstyle"
        if bangs != "none":
            style_desc += f", {bangs}"
        color_desc = hair_color
        if highlights:
            color_desc += f" with {highlights}"

        if hair_prompt:
            prompt = hair_prompt
        else:
            prompt = f"{style_desc}, {color_desc} hair"

        # Use VirtualDresser for the actual inpainting
        dresser = VirtualDresser()
        result = dresser.dress(
            model,
            clip,
            vae,
            source_image,
            hair_mask,
            prompt,
            seed,
            steps,
            cfg,
            denoise,
            negative_prompt="deformed, blurry, bad hair, wrong hair color, low quality",
        )

        return result


# ============== ACCESSORY EDITOR ==============
class AccessoryEditor:
    """Edit accessories using segmentation + inpainting"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "source_image": ("IMAGE",),
                "accessory_type": (
                    [
                        "watch",
                        "glasses",
                        "necklace",
                        "earrings",
                        "bracelet",
                        "ring",
                        "bag",
                        "belt",
                    ],
                ),
                "accessory_prompt": (
                    "STRING",
                    {"multiline": True, "default": "luxury watch"},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 20.0}),
                "denoise": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit_accessory"
    CATEGORY = "Laura Studio/Dressing"
    DESCRIPTION = "Add or change accessories via segmentation + inpainting"

    def edit_accessory(
        self,
        model,
        clip,
        vae,
        source_image,
        accessory_type,
        accessory_prompt,
        seed,
        steps,
        cfg,
        denoise,
    ):
        # Map accessory to segmentation category
        acc_to_seg = {
            "watch": "left_arm",  # wrist area proxy
            "glasses": "sunglasses",
            "necklace": "scarf",  # neck area
            "earrings": "face",  # near face
            "bracelet": "left_arm",  # wrist area proxy
            "ring": "left_arm",  # finger area proxy
            "bag": "bag",
            "belt": "belt",
        }
        seg_cat = acc_to_seg.get(accessory_type, "belt")

        # Get mask from AccessoryDetector
        detector = AccessoryDetector()
        mask, _ = detector.detect_accessory(
            source_image,
            seg_cat
            if seg_cat
            in [
                "shoes",
                "watch",
                "bag",
                "belt",
                "sunglasses",
                "hat",
                "scarf",
                "jewelry",
            ]
            else "bag",
            expand_mask=6,
        )

        type_descs = {
            "watch": "luxury watch, timepiece",
            "glasses": "stylish glasses, eyewear",
            "necklace": "elegant necklace, jewelry",
            "earrings": "designer earrings, jewelry",
            "bracelet": "bracelet, jewelry",
            "ring": "ring, fine jewelry",
            "bag": "designer bag, fashion accessory",
            "belt": "leather belt, fashion accessory",
        }
        desc = type_descs.get(accessory_type, "accessory")
        prompt = f"{desc}, {accessory_prompt}"

        # Inpaint the accessory
        dresser = VirtualDresser()
        result = dresser.dress(
            model,
            clip,
            vae,
            source_image,
            mask,
            prompt,
            seed,
            steps,
            cfg,
            denoise,
            negative_prompt="deformed, blurry, bad accessory, low quality",
        )

        return result


# ============== MAKEUP ARTIST ==============
class MakeupArtist:
    """Apply different makeup looks using face segmentation + inpainting"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "source_image": ("IMAGE",),
                "makeup_style": (
                    [
                        "natural",
                        "glam",
                        "dramatic",
                        "nude",
                        "vintage",
                        "bold",
                        "soft",
                        "editorial",
                    ],
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 20.0}),
                "denoise": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "makeup_details": ("STRING", {"multiline": True, "default": ""}),
                "lip_color": ("STRING", {"default": ""}),
                "eye_style": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_makeup"
    CATEGORY = "Laura Studio/Dressing"
    DESCRIPTION = "Apply makeup looks via face segmentation + inpainting"

    def apply_makeup(
        self,
        model,
        clip,
        vae,
        source_image,
        makeup_style,
        seed,
        steps,
        cfg,
        denoise,
        makeup_details="",
        lip_color="",
        eye_style="",
    ):
        # Get face mask for makeup region
        seg = ClothingSegmentor()
        face_mask, _ = seg.segment_clothing(
            source_image, "face", expand_mask=6, feather=4
        )

        style_descs = {
            "natural": "natural makeup, minimal, fresh skin",
            "glam": "glamour makeup, bold, radiant",
            "dramatic": "dramatic makeup, bold colors, smokey eyes",
            "nude": "nude makeup, natural tones, dewy skin",
            "vintage": "vintage makeup, classic red lip",
            "bold": "bold makeup, striking colors",
            "soft": "soft makeup, delicate, pastel",
            "editorial": "editorial makeup, high fashion, avant-garde",
        }
        desc = style_descs.get(makeup_style, "natural makeup")

        prompt_parts = [f"woman with {desc}"]
        if lip_color:
            prompt_parts.append(f"{lip_color} lips")
        if eye_style:
            prompt_parts.append(f"{eye_style} eye makeup")
        if makeup_details:
            prompt_parts.append(makeup_details)

        prompt = ", ".join(prompt_parts) + ", detailed face, professional"

        dresser = VirtualDresser()
        result = dresser.dress(
            model,
            clip,
            vae,
            source_image,
            face_mask,
            prompt,
            seed,
            steps,
            cfg,
            denoise,
            negative_prompt="deformed, blurry, bad makeup, unnatural skin, low quality",
        )

        return result


# ============== OUTFIT COMBINATOR ==============
class OutfitCombinator:
    """Combine multiple clothing items into complete outfit"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "source_image": ("IMAGE",),
                "top_prompt": ("STRING", {"default": "elegant blouse"}),
                "bottom_prompt": ("STRING", {"default": "designer jeans"}),
                "shoes_prompt": ("STRING", {"default": "high heels"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 35, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 20.0}),
                "denoise": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "accessory_prompt": ("STRING", {"default": ""}),
                "style": (
                    [
                        "casual",
                        "formal",
                        "business",
                        "evening",
                        "streetwear",
                        "bohemian",
                    ],
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "combine_outfit"
    CATEGORY = "Laura Studio/Dressing"
    DESCRIPTION = "Create complete outfit by sequentially replacing items"

    def combine_outfit(
        self,
        model,
        clip,
        vae,
        source_image,
        top_prompt,
        bottom_prompt,
        shoes_prompt,
        seed,
        steps,
        cfg,
        denoise,
        accessory_prompt="",
        style="casual",
    ):
        # Sequentially replace each clothing item using real masks
        seg = ClothingSegmentor()
        dresser = VirtualDresser()

        style_prefix = {
            "casual": "casual",
            "formal": "elegant formal",
            "business": "professional business",
            "evening": "sophisticated evening",
            "streetwear": "trendy streetwear",
            "bohemian": "bohemian",
        }.get(style, "stylish")

        # Step 1: Replace top
        top_mask, _ = seg.segment_clothing(
            source_image, "top", expand_mask=4, feather=2
        )
        result = dresser.dress(
            model,
            clip,
            vae,
            source_image,
            top_mask,
            f"{style_prefix} {top_prompt}",
            seed,
            steps,
            cfg,
            denoise,
        )[0]

        # Step 2: Replace bottom
        bottom_mask, _ = seg.segment_clothing(
            result, "bottom", expand_mask=4, feather=2
        )
        result = dresser.dress(
            model,
            clip,
            vae,
            result,
            bottom_mask,
            f"{style_prefix} {bottom_prompt}",
            seed + 1,
            steps,
            cfg,
            denoise,
        )[0]

        # Step 3: Replace shoes
        shoes_mask, _ = seg.segment_clothing(result, "shoes", expand_mask=4, feather=2)
        result = dresser.dress(
            model,
            clip,
            vae,
            result,
            shoes_mask,
            f"{style_prefix} {shoes_prompt}",
            seed + 2,
            steps,
            cfg,
            denoise,
        )[0]

        # Step 4: Accessories (optional)
        if accessory_prompt:
            acc_mask, _ = seg.segment_clothing(result, "belt", expand_mask=4, feather=2)
            result = dresser.dress(
                model,
                clip,
                vae,
                result,
                acc_mask,
                f"{style_prefix} {accessory_prompt}",
                seed + 3,
                steps,
                cfg,
                0.7,
            )[0]

        return (result,)


# ============== IPADAPTER STYLE REFERENCE ==============
class IPAdapterStyleReference:
    """Extract artistic style from reference image using IPAdapter"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "style_image": ("IMAGE",),
                "weight": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0}),
            },
            "optional": {
                "noise": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_style"
    CATEGORY = "Laura Studio/Dressing"
    DESCRIPTION = "Apply style reference using IPAdapter"

    def apply_style(
        self, model, style_image, weight, noise=0.1, start_at=0.0, end_at=1.0
    ):
        IPALoader, IPAApply = _try_import_ipadapter()

        if IPALoader is not None and IPAApply is not None:
            try:
                loader = IPALoader()
                # Standard preset for style
                ipa_model, clip_vision = loader.load_models(
                    model, preset="STANDARD (medium strength)"
                )

                applier = IPAApply()
                result = applier.apply_ipadapter(
                    model=ipa_model,
                    ipadapter=ipa_model,
                    image=style_image,
                    weight=weight,
                    noise=noise,
                    start_at=start_at,
                    end_at=end_at,
                    weight_type="style transfer",
                    combine_embeds="concat",
                )
                return (result[0],)
            except Exception as e:
                print(f"[Laura Studio] IPAdapter style transfer failed: {e}")
                return (model,)
        else:
            print(
                "[Laura Studio] IPAdapter Plus not installed. Returning model unchanged."
            )
            return (model,)


# ============== IPADAPTER CLOTHING REFERENCE ==============
class IPAdapterClothingReference:
    """Extract clothing features from reference image using IPAdapter"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clothing_image": ("IMAGE",),
                "weight": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 2.0}),
            },
            "optional": {
                "noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "end_at": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_clothing"
    CATEGORY = "Laura Studio/Dressing"
    DESCRIPTION = "Apply clothing reference using IPAdapter"

    def apply_clothing(
        self, model, clothing_image, weight, noise=0.0, start_at=0.0, end_at=0.8
    ):
        IPALoader, IPAApply = _try_import_ipadapter()

        if IPALoader is not None and IPAApply is not None:
            try:
                loader = IPALoader()
                # Use composition or standard preset for clothing details
                ipa_model, clip_vision = loader.load_models(
                    model, preset="PLUS (high strength)"
                )

                applier = IPAApply()
                result = applier.apply_ipadapter(
                    model=ipa_model,
                    ipadapter=ipa_model,
                    image=clothing_image,
                    weight=weight,
                    noise=noise,
                    start_at=start_at,
                    end_at=end_at,
                    weight_type="composition",
                    combine_embeds="concat",
                )
                return (result[0],)
            except Exception as e:
                print(f"[Laura Studio] IPAdapter clothing reference failed: {e}")
                return (model,)
        else:
            print(
                "[Laura Studio] IPAdapter Plus not installed. Returning model unchanged."
            )
            return (model,)


# Register all dressing nodes
NODE_CLASS_MAPPINGS.update(
    {
        "ClothingSegmentor": ClothingSegmentor,
        "AccessoryDetector": AccessoryDetector,
        "VirtualDresser": VirtualDresser,
        "DressingRoomCompositor": DressingRoomCompositor,
        "HairStylist": HairStylist,
        "AccessoryEditor": AccessoryEditor,
        "MakeupArtist": MakeupArtist,
        "OutfitCombinator": OutfitCombinator,
        "IPAdapterStyleReference": IPAdapterStyleReference,
        "IPAdapterClothingReference": IPAdapterClothingReference,
    }
)

NODE_DISPLAY_NAME_MAPPINGS.update(
    {
        "ClothingSegmentor": "Clothing Segmentor (RMBG)",
        "AccessoryDetector": "Accessory Detector (RMBG)",
        "VirtualDresser": "Virtual Dresser",
        "DressingRoomCompositor": "Dressing Room Compositor",
        "HairStylist": "Hair Stylist",
        "AccessoryEditor": "Accessory Editor",
        "MakeupArtist": "Makeup Artist",
        "OutfitCombinator": "Outfit Combinator",
        "IPAdapterStyleReference": "Style Reference (IPAdapter)",
        "IPAdapterClothingReference": "Clothing Reference (IPAdapter)",
    }
)
