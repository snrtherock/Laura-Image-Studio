"""
Laura Image Studio - Tile-based Processing Nodes
Handle high-resolution images by splitting them into tiles for generation and upscaling
"""

import torch
import torch.nn.functional as F

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


# ============== TILE SPLITTER ==============
class TileSplitter:
    """Split an image into overlapping tiles for high-res processing"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_size": (
                    "INT",
                    {"default": 512, "min": 256, "max": 2048, "step": 64},
                ),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 256, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE", "TILE_DATA")
    RETURN_NAMES = ("tiles", "tile_data")
    FUNCTION = "split_tiles"
    CATEGORY = "Laura Studio/Tiles"
    DESCRIPTION = "Split high-resolution image into tiles"

    def split_tiles(self, image, tile_size, overlap):
        B, H, W, C = image.shape
        stride = max(1, tile_size - overlap)

        # Calculate padding needed to fit tiles
        pad_h = (
            (stride - (H - tile_size) % stride) % stride
            if H > tile_size
            else (tile_size - H)
        )
        pad_w = (
            (stride - (W - tile_size) % stride) % stride
            if W > tile_size
            else (tile_size - W)
        )

        # Pad the image
        padded_image = F.pad(
            image.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h), mode="reflect"
        ).permute(0, 2, 3, 1)

        tiles = []
        new_h, new_w = padded_image.shape[1], padded_image.shape[2]

        for y in range(0, max(1, new_h - tile_size + 1), stride):
            for x in range(0, max(1, new_w - tile_size + 1), stride):
                tile = padded_image[:, y : y + tile_size, x : x + tile_size, :]
                tiles.append(tile)

        tile_data = {
            "original_shape": (H, W),
            "padded_shape": (new_h, new_w),
            "tile_size": tile_size,
            "overlap": overlap,
            "stride": stride,
            "tiles_x": len(range(0, max(1, new_w - tile_size + 1), stride)),
            "tiles_y": len(range(0, max(1, new_h - tile_size + 1), stride)),
        }

        return (torch.cat(tiles, dim=0), tile_data)


# ============== TILE MERGER ==============
class TileMerger:
    """Merge processed tiles back into a high-resolution image"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "tile_data": ("TILE_DATA",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "merge_tiles"
    CATEGORY = "Laura Studio/Tiles"
    DESCRIPTION = "Merge processed tiles back into one image"

    def merge_tiles(self, tiles, tile_data):
        H, W = tile_data["original_shape"]
        new_h, new_w = tile_data["padded_shape"]
        tile_size = tile_data["tile_size"]
        overlap = tile_data["overlap"]
        stride = tile_data["stride"]

        # Output buffer and weight buffer for blending
        # We assume 1 batch for the merged image
        merged = torch.zeros((1, new_h, new_w, tiles.shape[3]), device=tiles.device)
        weights = torch.zeros((1, new_h, new_w, 1), device=tiles.device)

        # Create a linear fade mask for blending
        mask = torch.ones((1, tile_size, tile_size, 1), device=tiles.device)
        if overlap > 0:
            fade = torch.linspace(0, 1, overlap, device=tiles.device)
            mask[:, :overlap, :, :] *= fade.view(1, overlap, 1, 1)
            mask[:, -overlap:, :, :] *= fade.flip(0).view(1, overlap, 1, 1)
            mask[:, :, :overlap, :] *= fade.view(1, 1, overlap, 1)
            mask[:, :, -overlap:, :] *= fade.flip(0).view(1, 1, overlap, 1)

        tile_idx = 0
        for y in range(0, max(1, new_h - tile_size + 1), stride):
            for x in range(0, max(1, new_w - tile_size + 1), stride):
                tile = tiles[tile_idx : tile_idx + 1]
                merged[:, y : y + tile_size, x : x + tile_size, :] += tile * mask
                weights[:, y : y + tile_size, x : x + tile_size, :] += mask
                tile_idx += 1

        # Normalize by weights
        merged /= torch.clamp(weights, min=1e-5)

        # Crop back to original size
        return (merged[:, :H, :W, :],)


# ============== TILE INPAINTER ==============
class TileInpainter:
    """Specialized inpainting for tiled regions"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "tiles": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("tiles",)
    FUNCTION = "inpaint_tiles"
    CATEGORY = "Laura Studio/Tiles"
    DESCRIPTION = "Refine tiled regions with inpainting at low denoise"

    def inpaint_tiles(self, model, clip, vae, tiles, prompt, denoise):
        from nodes import VAEEncode, KSampler, VAEDecode, CLIPTextEncode
        import random

        num_tiles = tiles.shape[0]

        # If denoise is 0 or prompt is empty, skip processing
        if denoise <= 0.0 or not prompt.strip():
            return (tiles,)

        # Encode prompts once (shared across all tiles)
        positive = CLIPTextEncode().encode(clip, prompt)[0]
        negative = CLIPTextEncode().encode(
            clip, "blurry, low quality, artifacts, noise"
        )[0]

        vae_encoder = VAEEncode()
        vae_decoder = VAEDecode()
        seed = random.randint(0, 0xFFFFFFFFFFFFFFFF)

        processed_tiles = []
        for i in range(num_tiles):
            tile = tiles[i : i + 1]

            # VAEEncode the tile -> latent space
            encoded = vae_encoder.encode(vae, tile)[0]
            latent = {"samples": encoded["samples"]}

            # KSampler with low denoise to refine details without destroying content
            sampled = KSampler().sample(
                model,
                seed + i,
                20,
                7.0,
                "euler",
                "normal",
                positive,
                negative,
                latent,
                denoise=denoise,
            )[0]

            # VAEDecode back to pixel space
            result = vae_decoder.decode(vae, sampled)[0]
            processed_tiles.append(result)

        return (torch.cat(processed_tiles, dim=0),)


NODE_CLASS_MAPPINGS.update(
    {
        "TileSplitter": TileSplitter,
        "TileMerger": TileMerger,
        "TileInpainter": TileInpainter,
    }
)

NODE_DISPLAY_NAME_MAPPINGS.update(
    {
        "TileSplitter": "Tile Splitter",
        "TileMerger": "Tile Merger",
        "TileInpainter": "Tile Inpainter",
    }
)
