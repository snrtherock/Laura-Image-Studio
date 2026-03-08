"""
Laura Image Studio - FLUX.1 Tools Nodes (v0.9)
Specialized nodes for FLUX.1 Fill (inpainting), Depth (depth-guided),
Canny (edge-guided), and Redux (image-variation) pipelines.

Each tool has a dedicated Loader + Generator pair.
All loaders produce a FLUX_TOOL_PIPE dict that generators consume.
"""

import torch
import numpy as np
import folder_paths

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


# ============== SHARED UTILITIES ==============
def _get_logger():
    """Lazy import to avoid circular dependency."""
    from .models import LauraLogger

    return LauraLogger


def _load_flux_checkpoint(model_name):
    """Load a FLUX checkpoint using ComfyUI's standard loader."""
    from nodes import CheckpointLoaderSimple

    result = CheckpointLoaderSimple().load_checkpoint(model_name)
    return result[0], result[1], result[2]  # model, clip, vae


def _load_controlnet(controlnet_name):
    """Load a ControlNet model."""
    import comfy.controlnet

    controlnet_path = folder_paths.get_full_path("controlnet", controlnet_name)
    if controlnet_path is None:
        raise FileNotFoundError(f"ControlNet not found: {controlnet_name}")
    controlnet = comfy.controlnet.load_controlnet(controlnet_path)
    return controlnet


def _encode_prompts(clip, positive_text, negative_text=""):
    """Encode text prompts using CLIP."""
    from nodes import CLIPTextEncode

    encoder = CLIPTextEncode()
    positive = encoder.encode(clip, positive_text)[0]
    # FLUX models typically don't use negative prompts, but we support it for compatibility
    if negative_text:
        negative = encoder.encode(clip, negative_text)[0]
    else:
        negative = encoder.encode(clip, "")[0]
    return positive, negative


# ============== FLUX.1 FILL (INPAINTING) ==============
class FluxFillLoader:
    """Load FLUX.1 Fill model for inpainting/outpainting tasks.
    FLUX.1 Fill is optimized for seamless inpainting with high prompt adherence.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("checkpoints"),),
            },
            "optional": {
                "clip_name": (["auto"] + folder_paths.get_filename_list("clip"),),
                "vae_name": (["auto"] + folder_paths.get_filename_list("vae"),),
            },
        }

    RETURN_TYPES = ("FLUX_TOOL_PIPE",)
    RETURN_NAMES = ("flux_fill_pipe",)
    FUNCTION = "load_fill"
    CATEGORY = "Laura Studio/FLUX Tools"
    DESCRIPTION = "Load FLUX.1 Fill model for high-quality inpainting and outpainting"

    def load_fill(self, model_name, clip_name="auto", vae_name="auto"):
        logger = _get_logger()
        logger.info(f"Loading FLUX.1 Fill: {model_name}")

        model, clip, vae = _load_flux_checkpoint(model_name)

        # Override CLIP/VAE if user specified
        if clip_name != "auto":
            import comfy.sd

            clip_path = folder_paths.get_full_path("clip", clip_name)
            if clip_path:
                clip = comfy.sd.load_clip(ckpt_paths=[clip_path])

        if vae_name != "auto":
            import comfy.sd

            vae_path = folder_paths.get_full_path("vae", vae_name)
            if vae_path:
                vae = comfy.sd.load_vae(vae_path)

        pipe = {
            "model": model,
            "clip": clip,
            "vae": vae,
            "tool_type": "fill",
            "model_name": model_name,
        }

        logger.info("FLUX.1 Fill pipeline ready")
        return (pipe,)


class FluxFillGenerator:
    """Generate inpainted/outpainted images using FLUX.1 Fill.
    Supports masked region filling with text guidance.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flux_fill_pipe": ("FLUX_TOOL_PIPE",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "guidance": (
                    "FLOAT",
                    {"default": 30.0, "min": 0.0, "max": 100.0, "step": 0.5},
                ),
            },
            "optional": {
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "generate_fill"
    CATEGORY = "Laura Studio/FLUX Tools"
    DESCRIPTION = (
        "Inpaint/outpaint using FLUX.1 Fill with masked region and text guidance"
    )

    def generate_fill(
        self,
        flux_fill_pipe,
        image,
        mask,
        prompt,
        seed,
        steps,
        guidance,
        denoise=1.0,
        negative_prompt="",
    ):
        logger = _get_logger()
        model = flux_fill_pipe["model"]
        clip = flux_fill_pipe["clip"]
        vae = flux_fill_pipe["vae"]

        logger.info(f"FLUX.1 Fill generating ({image.shape[2]}x{image.shape[1]})")

        from nodes import VAEEncode, KSampler, VAEDecode
        import torch.nn.functional as F

        # Encode prompts
        positive, negative = _encode_prompts(clip, prompt, negative_prompt)

        # Encode image to latent
        encoded = VAEEncode().encode(vae, image)[0]

        # Process mask for latent space
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
        mask_latent = (mask_latent > 0.5).float()

        # Build latent dict with mask
        latent = {
            "samples": encoded["samples"],
            "noise_mask": mask_latent.squeeze(1),  # KSampler expects [B, H, W]
        }

        # Sample — FLUX Fill uses high guidance
        sampled = KSampler().sample(
            model,
            seed,
            steps,
            guidance,
            "euler",
            "simple",
            positive,
            negative,
            latent,
            denoise=denoise,
        )[0]

        # Decode
        decoded = VAEDecode().decode(vae, sampled)[0]

        return (decoded, sampled)


# ============== FLUX.1 DEPTH (DEPTH-GUIDED) ==============
class FluxDepthLoader:
    """Load FLUX.1 Depth model and ControlNet for depth-guided generation.
    Accepts a depth map to guide the spatial structure of generated images.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("checkpoints"),),
                "controlnet_name": (folder_paths.get_filename_list("controlnet"),),
            },
            "optional": {
                "controlnet_strength": (
                    "FLOAT",
                    {"default": 0.85, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
            },
        }

    RETURN_TYPES = ("FLUX_TOOL_PIPE",)
    RETURN_NAMES = ("flux_depth_pipe",)
    FUNCTION = "load_depth"
    CATEGORY = "Laura Studio/FLUX Tools"
    DESCRIPTION = (
        "Load FLUX.1 Depth model + ControlNet for depth-guided image generation"
    )

    def load_depth(self, model_name, controlnet_name, controlnet_strength=0.85):
        logger = _get_logger()
        logger.info(f"Loading FLUX.1 Depth: {model_name} + CN: {controlnet_name}")

        model, clip, vae = _load_flux_checkpoint(model_name)
        controlnet = _load_controlnet(controlnet_name)

        pipe = {
            "model": model,
            "clip": clip,
            "vae": vae,
            "controlnet": controlnet,
            "controlnet_strength": controlnet_strength,
            "tool_type": "depth",
            "model_name": model_name,
        }

        logger.info("FLUX.1 Depth pipeline ready")
        return (pipe,)


class FluxDepthGenerator:
    """Generate depth-guided images using FLUX.1 Depth.
    Takes a depth map input to control the spatial structure of the output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flux_depth_pipe": ("FLUX_TOOL_PIPE",),
                "depth_map": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "guidance": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.5},
                ),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
                "height": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 2048, "step": 8},
                ),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "controlnet_strength": (
                    "FLOAT",
                    {"default": -1.0, "min": -1.0, "max": 2.0, "step": 0.05},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "generate_depth"
    CATEGORY = "Laura Studio/FLUX Tools"
    DESCRIPTION = "Generate images guided by a depth map using FLUX.1 Depth"

    def generate_depth(
        self,
        flux_depth_pipe,
        depth_map,
        prompt,
        seed,
        steps,
        guidance,
        width,
        height,
        negative_prompt="",
        controlnet_strength=-1.0,
    ):
        logger = _get_logger()
        model = flux_depth_pipe["model"]
        clip = flux_depth_pipe["clip"]
        vae = flux_depth_pipe["vae"]
        controlnet = flux_depth_pipe["controlnet"]
        cn_strength = (
            controlnet_strength
            if controlnet_strength >= 0
            else flux_depth_pipe["controlnet_strength"]
        )

        logger.info(
            f"FLUX.1 Depth generating ({width}x{height}), CN strength: {cn_strength}"
        )

        from nodes import EmptyLatentImage, KSampler, VAEDecode

        # Encode prompts
        positive, negative = _encode_prompts(clip, prompt, negative_prompt)

        # Apply ControlNet to positive conditioning
        try:
            from comfy_extras.nodes_controlnet import ControlNetApplyAdvanced

            result = ControlNetApplyAdvanced().apply_controlnet(
                positive, negative, controlnet, depth_map, cn_strength, 0.0, 1.0
            )
            positive, negative = result[0], result[1]
        except Exception as e:
            logger.warn(
                f"ControlNet apply failed, generating without depth guidance: {e}"
            )

        # Create empty latent
        latent = EmptyLatentImage().generate(width, height, batch_size=1)[0]

        # Sample
        sampled = KSampler().sample(
            model,
            seed,
            steps,
            guidance,
            "euler",
            "simple",
            positive,
            negative,
            latent,
            denoise=1.0,
        )[0]

        # Decode
        decoded = VAEDecode().decode(vae, sampled)[0]

        return (decoded, sampled)


# ============== FLUX.1 CANNY (EDGE-GUIDED) ==============
class FluxCannyLoader:
    """Load FLUX.1 Canny model and ControlNet for edge-guided generation.
    Uses Canny edge detection maps to preserve structure while regenerating content.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("checkpoints"),),
                "controlnet_name": (folder_paths.get_filename_list("controlnet"),),
            },
            "optional": {
                "controlnet_strength": (
                    "FLOAT",
                    {"default": 0.90, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
            },
        }

    RETURN_TYPES = ("FLUX_TOOL_PIPE",)
    RETURN_NAMES = ("flux_canny_pipe",)
    FUNCTION = "load_canny"
    CATEGORY = "Laura Studio/FLUX Tools"
    DESCRIPTION = (
        "Load FLUX.1 Canny model + ControlNet for edge-guided image generation"
    )

    def load_canny(self, model_name, controlnet_name, controlnet_strength=0.90):
        logger = _get_logger()
        logger.info(f"Loading FLUX.1 Canny: {model_name} + CN: {controlnet_name}")

        model, clip, vae = _load_flux_checkpoint(model_name)
        controlnet = _load_controlnet(controlnet_name)

        pipe = {
            "model": model,
            "clip": clip,
            "vae": vae,
            "controlnet": controlnet,
            "controlnet_strength": controlnet_strength,
            "tool_type": "canny",
            "model_name": model_name,
        }

        logger.info("FLUX.1 Canny pipeline ready")
        return (pipe,)


class FluxCannyGenerator:
    """Generate edge-guided images using FLUX.1 Canny.
    Takes a Canny edge map to preserve structural detail while changing content.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flux_canny_pipe": ("FLUX_TOOL_PIPE",),
                "canny_image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "guidance": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.5},
                ),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
                "height": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 2048, "step": 8},
                ),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "controlnet_strength": (
                    "FLOAT",
                    {"default": -1.0, "min": -1.0, "max": 2.0, "step": 0.05},
                ),
                "low_threshold": ("INT", {"default": 100, "min": 0, "max": 500}),
                "high_threshold": ("INT", {"default": 200, "min": 0, "max": 500}),
                "preprocess_canny": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "IMAGE")
    RETURN_NAMES = ("image", "latent", "canny_edges")
    FUNCTION = "generate_canny"
    CATEGORY = "Laura Studio/FLUX Tools"
    DESCRIPTION = "Generate images guided by Canny edges using FLUX.1 Canny. Optionally preprocesses input to extract edges."

    def generate_canny(
        self,
        flux_canny_pipe,
        canny_image,
        prompt,
        seed,
        steps,
        guidance,
        width,
        height,
        negative_prompt="",
        controlnet_strength=-1.0,
        low_threshold=100,
        high_threshold=200,
        preprocess_canny=False,
    ):
        logger = _get_logger()
        model = flux_canny_pipe["model"]
        clip = flux_canny_pipe["clip"]
        vae = flux_canny_pipe["vae"]
        controlnet = flux_canny_pipe["controlnet"]
        cn_strength = (
            controlnet_strength
            if controlnet_strength >= 0
            else flux_canny_pipe["controlnet_strength"]
        )

        # Optional Canny edge preprocessing
        edge_image = canny_image
        if preprocess_canny:
            try:
                import cv2

                # Convert from [B,H,W,C] float to uint8 for OpenCV
                img_np = (canny_image[0].cpu().numpy() * 255).astype(np.uint8)
                if img_np.shape[-1] == 3:
                    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img_np[:, :, 0]
                edges = cv2.Canny(gray, low_threshold, high_threshold)
                # Convert back to [B, H, W, C] float tensor
                edges_rgb = (
                    np.stack([edges, edges, edges], axis=-1).astype(np.float32) / 255.0
                )
                edge_image = torch.from_numpy(edges_rgb).unsqueeze(0)
                logger.info(
                    f"Canny edge preprocessing: thresholds ({low_threshold}, {high_threshold})"
                )
            except ImportError:
                logger.warn(
                    "cv2 not available for Canny preprocessing, using input directly"
                )

        logger.info(
            f"FLUX.1 Canny generating ({width}x{height}), CN strength: {cn_strength}"
        )

        from nodes import EmptyLatentImage, KSampler, VAEDecode

        # Encode prompts
        positive, negative = _encode_prompts(clip, prompt, negative_prompt)

        # Apply ControlNet
        try:
            from comfy_extras.nodes_controlnet import ControlNetApplyAdvanced

            result = ControlNetApplyAdvanced().apply_controlnet(
                positive, negative, controlnet, edge_image, cn_strength, 0.0, 1.0
            )
            positive, negative = result[0], result[1]
        except Exception as e:
            logger.warn(f"ControlNet apply failed: {e}")

        # Create empty latent
        latent = EmptyLatentImage().generate(width, height, batch_size=1)[0]

        # Sample
        sampled = KSampler().sample(
            model,
            seed,
            steps,
            guidance,
            "euler",
            "simple",
            positive,
            negative,
            latent,
            denoise=1.0,
        )[0]

        # Decode
        decoded = VAEDecode().decode(vae, sampled)[0]

        return (decoded, sampled, edge_image)


# ============== FLUX.1 REDUX (IMAGE VARIATION) ==============
class FluxReduxLoader:
    """Load FLUX.1 Redux model for image variation/remixing.
    Redux takes a reference image and generates variations guided by text prompts.
    Uses CLIP Vision for image understanding.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("checkpoints"),),
            },
            "optional": {
                "clip_vision_name": (
                    ["auto"] + folder_paths.get_filename_list("clip_vision"),
                ),
                "redux_strength": (
                    "FLOAT",
                    {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
            },
        }

    RETURN_TYPES = ("FLUX_TOOL_PIPE",)
    RETURN_NAMES = ("flux_redux_pipe",)
    FUNCTION = "load_redux"
    CATEGORY = "Laura Studio/FLUX Tools"
    DESCRIPTION = (
        "Load FLUX.1 Redux for image variation/remixing with CLIP Vision reference"
    )

    def load_redux(self, model_name, clip_vision_name="auto", redux_strength=0.75):
        logger = _get_logger()
        logger.info(f"Loading FLUX.1 Redux: {model_name}")

        model, clip, vae = _load_flux_checkpoint(model_name)

        # Load CLIP Vision if specified
        clip_vision = None
        if clip_vision_name != "auto":
            try:
                import comfy.clip_vision

                cv_path = folder_paths.get_full_path("clip_vision", clip_vision_name)
                if cv_path:
                    clip_vision = comfy.clip_vision.load(cv_path)
                    logger.info(f"CLIP Vision loaded: {clip_vision_name}")
            except Exception as e:
                logger.warn(f"Could not load CLIP Vision: {e}")

        pipe = {
            "model": model,
            "clip": clip,
            "vae": vae,
            "clip_vision": clip_vision,
            "redux_strength": redux_strength,
            "tool_type": "redux",
            "model_name": model_name,
        }

        logger.info("FLUX.1 Redux pipeline ready")
        return (pipe,)


class FluxReduxGenerator:
    """Generate image variations using FLUX.1 Redux.
    Takes a reference image and produces variations guided by text prompts.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flux_redux_pipe": ("FLUX_TOOL_PIPE",),
                "reference_image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "guidance": (
                    "FLOAT",
                    {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.5},
                ),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
                "height": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 2048, "step": 8},
                ),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "redux_strength": (
                    "FLOAT",
                    {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.05},
                ),
                "denoise": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0}),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "generate_redux"
    CATEGORY = "Laura Studio/FLUX Tools"
    DESCRIPTION = "Generate image variations from a reference using FLUX.1 Redux"

    def generate_redux(
        self,
        flux_redux_pipe,
        reference_image,
        prompt,
        seed,
        steps,
        guidance,
        width,
        height,
        negative_prompt="",
        redux_strength=-1.0,
        denoise=0.75,
    ):
        logger = _get_logger()
        model = flux_redux_pipe["model"]
        clip = flux_redux_pipe["clip"]
        vae = flux_redux_pipe["vae"]
        clip_vision = flux_redux_pipe.get("clip_vision")
        strength = (
            redux_strength if redux_strength >= 0 else flux_redux_pipe["redux_strength"]
        )

        logger.info(f"FLUX.1 Redux generating ({width}x{height}), strength: {strength}")

        from nodes import VAEEncode, EmptyLatentImage, KSampler, VAEDecode

        # Encode prompts
        positive, negative = _encode_prompts(clip, prompt, negative_prompt)

        # If CLIP Vision is available, encode reference image for conditioning
        if clip_vision is not None:
            try:
                import comfy.clip_vision

                clip_vision_output = clip_vision.encode_image(reference_image)
                # Merge CLIP Vision embedding into positive conditioning
                if hasattr(clip_vision_output, "image_embeds") or isinstance(
                    clip_vision_output, dict
                ):
                    # Attempt to use unCLIP conditioning if available
                    try:
                        from comfy_extras.nodes_unclip import unCLIPConditioning

                        positive = unCLIPConditioning().apply_adm(
                            positive, clip_vision_output, strength, 0.0
                        )[0]
                        logger.info(
                            "CLIP Vision reference applied via unCLIP conditioning"
                        )
                    except (ImportError, Exception) as e:
                        logger.warn(f"unCLIP conditioning not available: {e}")
            except Exception as e:
                logger.warn(f"CLIP Vision encoding failed: {e}")

        # Encode reference image as starting latent for img2img-style variation
        encoded = VAEEncode().encode(vae, reference_image)[0]

        # If denoise < 1.0, start from encoded reference; otherwise from empty latent
        if denoise < 1.0:
            latent = {"samples": encoded["samples"]}
        else:
            latent = EmptyLatentImage().generate(width, height, batch_size=1)[0]

        # Sample
        sampled = KSampler().sample(
            model,
            seed,
            steps,
            guidance,
            "euler",
            "simple",
            positive,
            negative,
            latent,
            denoise=denoise,
        )[0]

        # Decode
        decoded = VAEDecode().decode(vae, sampled)[0]

        return (decoded, sampled)


# Register all FLUX Tools nodes
NODE_CLASS_MAPPINGS.update(
    {
        "FluxFillLoader": FluxFillLoader,
        "FluxFillGenerator": FluxFillGenerator,
        "FluxDepthLoader": FluxDepthLoader,
        "FluxDepthGenerator": FluxDepthGenerator,
        "FluxCannyLoader": FluxCannyLoader,
        "FluxCannyGenerator": FluxCannyGenerator,
        "FluxReduxLoader": FluxReduxLoader,
        "FluxReduxGenerator": FluxReduxGenerator,
    }
)

NODE_DISPLAY_NAME_MAPPINGS.update(
    {
        "FluxFillLoader": "FLUX.1 Fill Loader",
        "FluxFillGenerator": "FLUX.1 Fill Generator",
        "FluxDepthLoader": "FLUX.1 Depth Loader",
        "FluxDepthGenerator": "FLUX.1 Depth Generator",
        "FluxCannyLoader": "FLUX.1 Canny Loader",
        "FluxCannyGenerator": "FLUX.1 Canny Generator",
        "FluxReduxLoader": "FLUX.1 Redux Loader",
        "FluxReduxGenerator": "FLUX.1 Redux Generator",
    }
)
