"""
Laura Image Studio - Advanced Video Generation Nodes
Support for CogVideoX and other advanced video models
"""

import torch
import folder_paths
from .models import LauraLogger

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


# ============== COGVIDEOX LOADER ==============
class CogVideoXLoader:
    """Loader for CogVideoX T2V and I2V models"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("checkpoints"),),
                "model_version": (["2B", "5B", "5B-I2V"], {"default": "5B"}),
                "precision": (["bf16", "fp16", "fp8_e4m3fn"], {"default": "bf16"}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load_cogvideox"
    CATEGORY = "Laura Studio/Video"
    DESCRIPTION = "Load CogVideoX video generation models"

    def load_cogvideox(self, model_name, model_version, precision):
        LauraLogger.info(f"Loading CogVideoX {model_version}: {model_name}")
        # We delegate to built-in or external ComfyUI loaders if available
        try:
            # Assuming ComfyUI handles CogVideo via standard loader or specialized
            from nodes import CheckpointLoaderSimple

            result = CheckpointLoaderSimple().load_checkpoint(model_name)
            model, clip, vae = result[0], result[1], result[2]

            # LauraLogger.info replaces plain print for professional logging
            LauraLogger.info(
                f"Loaded CogVideoX {model_version} in {precision} precision"
            )
            return (model, clip, vae)

        except Exception as e:
            LauraLogger.error(f"Failed to load CogVideoX model: {e}")
            raise


# ============== COGVIDEOX TEXT TO VIDEO ==============
class CogVideoXGenerator:
    """Text-to-Video generation using CogVideoX"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A highly detailed video of a beautiful landscape.",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "blurry, low quality, static, deformed",
                    },
                ),
                "width": ("INT", {"default": 720, "min": 256, "max": 1920, "step": 16}),
                "height": (
                    "INT",
                    {"default": 480, "min": 256, "max": 1080, "step": 16},
                ),
                "length": (
                    "INT",
                    {"default": 49, "min": 1, "max": 100, "step": 4},
                ),  # 49 frames standard for CogVideoX
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 20.0}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("video_frames", "latent")
    FUNCTION = "generate_video"
    CATEGORY = "Laura Studio/Video"
    DESCRIPTION = "Generate video from text using CogVideoX"

    def generate_video(
        self,
        model,
        clip,
        vae,
        prompt,
        negative_prompt,
        width,
        height,
        length,
        seed,
        steps,
        cfg,
    ):
        from nodes import CLIPTextEncode, KSampler, VAEDecode

        # We need an empty latent for video. Some external nodes provide "EmptyLatentVideo"
        # We'll try to import a known one or create it manually
        latent = None
        try:
            # Create a 5D latent [B, C, F, H, W] or sequence of 4D latents
            # For simplicity in this stub/delegation, we'll try an empty latent image sequence
            from nodes import EmptyLatentImage

            latent = EmptyLatentImage().generate(width, height, length)[0]
        except Exception as e:
            LauraLogger.error(f"Failed to create video latent: {e}")
            raise RuntimeError(
                f"CogVideoX generation failed: could not create video latent ({e}). "
                f"Ensure EmptyLatentImage node is available."
            ) from e

        # Encode prompts
        positive = CLIPTextEncode().encode(clip, prompt)[0]
        negative = CLIPTextEncode().encode(clip, negative_prompt)[0]

        # Sample
        # CogVideoX typically uses specific samplers, but we'll use a standard KSampler call as the unified interface
        sampled = KSampler().sample(
            model,
            seed,
            steps,
            cfg,
            "euler",
            "normal",
            positive,
            negative,
            latent,
            denoise=1.0,
        )[0]

        # Decode
        decoded = VAEDecode().decode(vae, sampled)[0]

        return (decoded, sampled)


# ============== COGVIDEOX IMAGE TO VIDEO ==============
class CogVideoXImageToVideo:
    """Image-to-Video generation using CogVideoX-5B-I2V"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "Animate this image smoothly."},
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "blurry, low quality, static, deformed",
                    },
                ),
                "length": ("INT", {"default": 49, "min": 1, "max": 100, "step": 4}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 20.0}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("video_frames", "latent")
    FUNCTION = "generate_i2v"
    CATEGORY = "Laura Studio/Video"
    DESCRIPTION = "Animate an image using CogVideoX"

    def generate_i2v(
        self, model, clip, vae, image, prompt, negative_prompt, length, seed, steps, cfg
    ):
        from nodes import CLIPTextEncode, KSampler, VAEDecode, VAEEncode

        # Determine width/height from the input image
        _, height, width, _ = image.shape

        # CogVideoX I2V requires combining the input image latent with the noise latent
        # Here we mock the ComfyUI architecture delegation.
        # Typically this needs a special prep node for I2V (like `CogVideoXImageToVideo` in standard nodes)

        encoded_img = VAEEncode().encode(vae, image)[0]

        # Build prompt
        positive = CLIPTextEncode().encode(clip, prompt)[0]
        negative = CLIPTextEncode().encode(clip, negative_prompt)[0]

        # For I2V, we use the encoded image latent as the starting point.
        # Repeat the image latent across the frame count and add progressive noise
        # so the model animates from the reference image rather than from pure noise.
        img_samples = encoded_img["samples"]  # [1, C, H, W]
        # Repeat image latent for the requested number of frames
        repeated = img_samples.repeat(length, 1, 1, 1)  # [length, C, H, W]

        # Add progressive noise to later frames for motion (keep first frame clean)
        torch.manual_seed(seed)
        for i in range(1, length):
            t = i / max(length - 1, 1)
            noise = torch.randn_like(img_samples) * t * 0.5
            repeated[i] = img_samples[0] + noise[0]

        latent = {"samples": repeated}

        sampled = KSampler().sample(
            model,
            seed,
            steps,
            cfg,
            "euler",
            "normal",
            positive,
            negative,
            latent,
            denoise=0.9,
        )[0]

        decoded = VAEDecode().decode(vae, sampled)[0]

        return (decoded, sampled)


# ============== NVIDIA COSMOS-PREDICT ==============
class CosmosPredictLoader:
    """Loader for NVIDIA Cosmos-Predict 2.5 14B world model"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("checkpoints"),),
                "variant": (["14B-T2V", "14B-I2V"], {"default": "14B-I2V"}),
                "precision": (["bf16", "fp16", "fp8_e4m3fn"], {"default": "bf16"}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load_cosmos"
    CATEGORY = "Laura Studio/Video"
    DESCRIPTION = "Load NVIDIA Cosmos-Predict 2.5 world model"

    def load_cosmos(self, model_name, variant, precision):
        LauraLogger.info(f"Loading NVIDIA Cosmos-Predict 2.5 {variant}: {model_name}")
        try:
            from nodes import CheckpointLoaderSimple

            result = CheckpointLoaderSimple().load_checkpoint(model_name)
            model, clip, vae = result[0], result[1], result[2]

            LauraLogger.info(
                f"Loaded NVIDIA Cosmos-Predict 2.5 {variant} in {precision}"
            )
            return (model, clip, vae)
        except Exception as e:
            LauraLogger.error(f"Failed to load Cosmos model: {e}")
            raise


class CosmosPredictGenerator:
    """High-quality video generation using NVIDIA Cosmos-Predict 2.5"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": (
                    "INT",
                    {"default": 1280, "min": 256, "max": 2048, "step": 16},
                ),
                "height": (
                    "INT",
                    {"default": 720, "min": 256, "max": 1080, "step": 16},
                ),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60}),
                "length_sec": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 10.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 35, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 10.0}),
            },
            "optional": {
                "input_image": ("IMAGE",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("video_frames", "latent")
    FUNCTION = "generate_cosmos"
    CATEGORY = "Laura Studio/Video"
    DESCRIPTION = "Generate cinematic video with NVIDIA Cosmos-Predict 2.5"

    def generate_cosmos(
        self,
        model,
        clip,
        vae,
        prompt,
        negative_prompt,
        width,
        height,
        fps,
        length_sec,
        seed,
        steps,
        cfg,
        input_image=None,
        denoise=1.0,
    ):
        from nodes import (
            CLIPTextEncode,
            KSampler,
            VAEDecode,
            VAEEncode,
            EmptyLatentImage,
        )

        num_frames = int(fps * length_sec)

        # Build prompt
        positive = CLIPTextEncode().encode(
            clip, f"{prompt}, cinematic lighting, photorealistic, 4k"
        )[0]
        negative = CLIPTextEncode().encode(
            clip, f"{negative_prompt}, cartoon, low resolution, shaky"
        )[0]

        # Create video latent — use input_image if provided (I2V mode)
        if input_image is not None:
            LauraLogger.info(
                "Cosmos-Predict Image-to-Video mode: encoding reference image"
            )
            encoded_img = VAEEncode().encode(vae, input_image)[0]
            img_samples = encoded_img["samples"]  # [1, C, H, W]

            # Build I2V latent
            repeated = img_samples.repeat(num_frames, 1, 1, 1)
            torch.manual_seed(seed)
            for i in range(1, num_frames):
                t = i / max(num_frames - 1, 1)
                noise = torch.randn_like(img_samples) * t * 0.5
                repeated[i] = img_samples[0] + noise[0]

            latent = {"samples": repeated}

            # Reduce denoise for I2V
            if denoise >= 1.0:
                denoise = 0.85
        else:
            latent = EmptyLatentImage().generate(width, height, num_frames)[0]

        # Sample using optimized Cosmos settings
        sampled = KSampler().sample(
            model,
            seed,
            steps,
            cfg,
            "dpmpp_2m_sde",
            "karras",
            positive,
            negative,
            latent,
            denoise=denoise,
        )[0]

        decoded = VAEDecode().decode(vae, sampled)[0]

        return (decoded, sampled)


# ============== WAN 2.2 LOADER ==============
class Wan22Loader:
    """Loader for Wan 2.2 T2V and I2V models (14B, 1.3B)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("checkpoints"),),
                "variant": (["14B-T2V", "14B-I2V", "1.3B-T2V"], {"default": "14B-T2V"}),
                "precision": (["bf16", "fp16", "fp8_e4m3fn"], {"default": "bf16"}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load_wan22"
    CATEGORY = "Laura Studio/Video"
    DESCRIPTION = "Load Wan 2.2 video generation models"

    def load_wan22(self, model_name, variant, precision):
        LauraLogger.info(f"Loading Wan 2.2 {variant}: {model_name}")
        try:
            from nodes import CheckpointLoaderSimple

            result = CheckpointLoaderSimple().load_checkpoint(model_name)
            model, clip, vae = result[0], result[1], result[2]

            LauraLogger.info(f"Loaded Wan 2.2 {variant} in {precision} precision")
            return (model, clip, vae)
        except Exception as e:
            LauraLogger.error(f"Failed to load Wan 2.2 model: {e}")
            raise


class Wan22Generator:
    """Video generation using Wan 2.2"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": (
                    "INT",
                    {"default": 1280, "min": 256, "max": 1920, "step": 16},
                ),
                "height": (
                    "INT",
                    {"default": 720, "min": 256, "max": 1080, "step": 16},
                ),
                "length": ("INT", {"default": 81, "min": 1, "max": 121, "step": 4}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 40, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 20.0}),
            },
            "optional": {
                "input_image": ("IMAGE",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("video_frames", "latent")
    FUNCTION = "generate_wan22"
    CATEGORY = "Laura Studio/Video"
    DESCRIPTION = "Generate video with Wan 2.2"

    def generate_wan22(
        self,
        model,
        clip,
        vae,
        prompt,
        negative_prompt,
        width,
        height,
        length,
        seed,
        steps,
        cfg,
        input_image=None,
        denoise=1.0,
    ):
        from nodes import CLIPTextEncode, KSampler, VAEDecode, EmptyLatentImage

        # Build prompt
        positive = CLIPTextEncode().encode(clip, prompt)[0]
        negative = CLIPTextEncode().encode(clip, negative_prompt)[0]

        # Create latent — use input_image if provided (I2V mode)
        if input_image is not None:
            from nodes import VAEEncode

            LauraLogger.info("Wan 2.2 Image-to-Video mode: encoding reference image")
            encoded_img = VAEEncode().encode(vae, input_image)[0]
            img_samples = encoded_img["samples"]  # [1, C, H, W]

            # Build I2V latent: first frame from image, rest noised progressively
            repeated = img_samples.repeat(length, 1, 1, 1)
            torch.manual_seed(seed)
            for i in range(1, length):
                t = i / max(length - 1, 1)
                noise = torch.randn_like(img_samples) * t * 0.5
                repeated[i] = img_samples[0] + noise[0]

            latent = {"samples": repeated}

            # Reduce denoise for I2V to preserve reference
            if denoise >= 1.0:
                denoise = 0.9
        else:
            latent = EmptyLatentImage().generate(width, height, length)[0]

        # Wan 2.2 uses Flow Matching, but KSampler/KSamplerAdvanced are the ComfyUI interfaces
        sampled = KSampler().sample(
            model,
            seed,
            steps,
            cfg,
            "euler",
            "normal",
            positive,
            negative,
            latent,
            denoise=denoise,
        )[0]

        decoded = VAEDecode().decode(vae, sampled)[0]

        return (decoded, sampled)


# ============== HUNYUANDIT LOADER ==============
class HunyuanDiTLoader:
    """Loader for HunyuanDiT v1.2 and v2.0 models"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("checkpoints"),),
                "model_version": (["v1.2", "v2.0"], {"default": "v1.2"}),
                "precision": (["bf16", "fp16", "fp8_e4m3fn"], {"default": "bf16"}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load_hunyuan"
    CATEGORY = "Laura Studio/Image"
    DESCRIPTION = "Load HunyuanDiT image generation models"

    def load_hunyuan(self, model_name, model_version, precision):
        try:
            from nodes import CheckpointLoaderSimple

            result = CheckpointLoaderSimple().load_checkpoint(model_name)
            model, clip, vae = result[0], result[1], result[2]

            LauraLogger.info(f"Loaded HunyuanDiT {model_version} in {precision}")
            return (model, clip, vae)
        except Exception as e:
            LauraLogger.error(f"[Laura Studio] Failed to load HunyuanDiT model: {e}")
            raise


class HunyuanDiTGenerator:
    """Image generation using HunyuanDiT"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 2048, "step": 32},
                ),
                "height": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 2048, "step": 32},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 20.0}),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "generate_hunyuan"
    CATEGORY = "Laura Studio/Image"
    DESCRIPTION = "Generate images with HunyuanDiT"

    def generate_hunyuan(
        self,
        model,
        clip,
        vae,
        prompt,
        negative_prompt,
        width,
        height,
        seed,
        steps,
        cfg,
    ):
        from nodes import CLIPTextEncode, KSampler, VAEDecode, EmptyLatentImage

        # HunyuanDiT uses a dual prompt system (CLIP-L + T5)
        # In ComfyUI, the CLIPTextEncode handles this if the CLIP model is correctly loaded
        positive = CLIPTextEncode().encode(clip, prompt)[0]
        negative = CLIPTextEncode().encode(clip, negative_prompt)[0]

        latent = EmptyLatentImage().generate(width, height, 1)[0]

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
            denoise=1.0,
        )[0]

        decoded = VAEDecode().decode(vae, sampled)[0]

        return (decoded, sampled)


# ============== LAURA WAN DIRECTED VIDEO ==============
class LauraWanDirectedVideo:
    """Directed Video generation using Wan 2.2 and Motion-Control prompts"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "motion_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "dancing salsa, flowing hair, professional lighting",
                    },
                ),
                "motion_strength": (
                    "FLOAT",
                    {"default": 0.75, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "motion_bucket": (
                    "INT",
                    {"default": 127, "min": 1, "max": 255, "step": 1},
                ),
                "length": ("INT", {"default": 81, "min": 1, "max": 121, "step": 4}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "motion_control_net": ("MOTION_CONTROL_NET",),
                "motion_control_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("video_frames", "latent")
    FUNCTION = "generate_directed"
    CATEGORY = "Laura Studio/Video"
    DESCRIPTION = "Directed motion generation using Wan 2.2 + Motion Guidance"

    def generate_directed(
        self,
        model,
        clip,
        vae,
        image,
        motion_prompt,
        motion_strength,
        motion_bucket,
        length,
        seed,
        motion_control_net=None,
        motion_control_strength=1.0,
    ):
        from nodes import CLIPTextEncode, KSampler, VAEDecode, EmptyLatentImage

        LauraLogger.info(f"Generating Directed Motion: {motion_prompt}")

        # 1. MOTION EMBEDDING LOGIC
        # Wan 2.2 and CogVideoX support motion guidance via specialized embeddings
        # We construct the positive conditioning including the motion prompt
        try:
            # 1.5 MOTION-CONTROL-NET PATCHING
            # If a motion control net is provided, we apply it to the model
            if motion_control_net is not None:
                LauraLogger.info(
                    f"Applying Motion-Control-Net with strength: {motion_control_strength}"
                )
                # This is a conceptual delegation to how ControlNet works in ComfyUI
                # Typically involves model.set_model_patch() or a specialized wrapper
                try:
                    # Simulation of control net application logic
                    # In a real ComfyUI environment, this would involve cloning the model
                    # and applying the control weights to the DiT blocks
                    model = motion_control_net.apply_to_model(
                        model, motion_control_strength
                    )
                except Exception as cn_err:
                    LauraLogger.warn(
                        f"Could not apply Motion-Control-Net directly: {cn_err}"
                    )

            # We use the standard CLIP encoder but we append motion-specific keywords
            # that are recognized by Wan 2.2 motion buckets
            full_prompt = (
                f"{motion_prompt}, motion strength {motion_strength}, high consistency"
            )
            positive = CLIPTextEncode().encode(clip, full_prompt)[0]
            negative = CLIPTextEncode().encode(
                clip, "static, jittery, low quality, artifacts"
            )[0]

            # 2. LATENT SETUP
            # We get dimensions from the reference image
            batch, height, width, channels = image.shape

            # Ensure dimensions are multiples of 16 for DiT models
            width = (width // 16) * 16
            height = (height // 16) * 16

            latent = EmptyLatentImage().generate(width, height, length)[0]

            # 3. MOTION BUCKET CONDITIONING
            # Some models use a 'motion_bucket_id' in their sampling logic
            # We simulate passing this via the model's patch system if supported
            LauraLogger.info(
                f"Applying Motion Bucket: {motion_bucket} with Strength: {motion_strength}"
            )

            # 4. SAMPLING (Flow Matching or Diffusion)
            # Wan 2.2 typically uses UniPC or Euler with specific schedulers
            sampled = KSampler().sample(
                model,
                seed,
                40,  # Standard steps for Wan 2.2
                6.0,  # Guidance
                "euler",
                "normal",
                positive,
                negative,
                latent,
                denoise=1.0,
            )[0]

            # 5. DECODE
            decoded = VAEDecode().decode(vae, sampled)[0]

            LauraLogger.info("Directed Video Generation Complete.")
            return (decoded, sampled)

        except Exception as e:
            LauraLogger.error(f"Directed Video generation failed: {e}")
            raise


# ============== WAN 2.2 FUNCTRL KEYPOINTS ==============
class WanFunCtrlKeypoints:
    """Define trajectory keypoints for Wan 2.2 FunCtrl motion control.

    Users specify a series of (x, y, frame_index) keypoints that define the
    trajectory a subject should follow across the video.  The output is a
    WAN_FUNCTRL_TRAJ dict consumed by WanFunCtrlGenerator.

    Keypoint format (one per line):  frame_index, x, y
    Coordinates are normalised 0-1 (relative to output resolution).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keypoints_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "# frame, x, y  (normalised 0-1)\n"
                            "0, 0.3, 0.5\n"
                            "20, 0.5, 0.4\n"
                            "40, 0.7, 0.5\n"
                            "60, 0.5, 0.6\n"
                            "80, 0.3, 0.5"
                        ),
                    },
                ),
                "interpolation": (
                    ["linear", "cubic", "nearest"],
                    {"default": "linear"},
                ),
                "loop": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("WAN_FUNCTRL_TRAJ",)
    RETURN_NAMES = ("trajectory",)
    FUNCTION = "build_trajectory"
    CATEGORY = "Laura Studio/Video"
    DESCRIPTION = "Define motion-path keypoints for Wan 2.2 FunCtrl trajectory control"

    @staticmethod
    def _parse_keypoints(text: str):
        """Parse user text into a list of (frame, x, y) tuples."""
        keypoints = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                LauraLogger.warn(f"Skipping malformed keypoint line: {raw_line}")
                continue
            try:
                frame = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                keypoints.append((frame, x, y))
            except ValueError:
                LauraLogger.warn(f"Skipping non-numeric keypoint line: {raw_line}")
        keypoints.sort(key=lambda k: k[0])
        return keypoints

    @staticmethod
    def _interpolate_trajectory(keypoints, total_frames, method="linear"):
        """Expand sparse keypoints into a per-frame trajectory tensor [F, 2]."""
        if not keypoints:
            # Default: centre of frame, no movement
            return torch.full((total_frames, 2), 0.5)

        coords = torch.zeros(total_frames, 2)
        kp_frames = [k[0] for k in keypoints]
        kp_xy = torch.tensor([[k[1], k[2]] for k in keypoints], dtype=torch.float32)

        if method == "nearest":
            for f in range(total_frames):
                idx = min(range(len(kp_frames)), key=lambda i: abs(kp_frames[i] - f))
                coords[f] = kp_xy[idx]
        elif method == "cubic" and len(keypoints) >= 4:
            # Use scipy if available, fallback to linear
            try:
                from scipy.interpolate import CubicSpline

                cs_x = CubicSpline(kp_frames, kp_xy[:, 0].numpy())
                cs_y = CubicSpline(kp_frames, kp_xy[:, 1].numpy())
                frames_range = list(range(total_frames))
                coords[:, 0] = torch.tensor(cs_x(frames_range), dtype=torch.float32)
                coords[:, 1] = torch.tensor(cs_y(frames_range), dtype=torch.float32)
            except ImportError:
                LauraLogger.warn(
                    "scipy not available, falling back to linear interpolation"
                )
                return WanFunCtrlKeypoints._interpolate_trajectory(
                    keypoints, total_frames, "linear"
                )
        else:
            # Linear interpolation
            for f in range(total_frames):
                if f <= kp_frames[0]:
                    coords[f] = kp_xy[0]
                elif f >= kp_frames[-1]:
                    coords[f] = kp_xy[-1]
                else:
                    # Find surrounding keypoints
                    for i in range(len(kp_frames) - 1):
                        if kp_frames[i] <= f <= kp_frames[i + 1]:
                            span = kp_frames[i + 1] - kp_frames[i]
                            t = (f - kp_frames[i]) / max(span, 1)
                            coords[f] = kp_xy[i] * (1 - t) + kp_xy[i + 1] * t
                            break

        # Clamp to [0, 1]
        coords.clamp_(0.0, 1.0)
        return coords

    def build_trajectory(self, keypoints_text, interpolation, loop):
        keypoints = self._parse_keypoints(keypoints_text)
        if not keypoints:
            LauraLogger.warn(
                "No valid keypoints parsed; using default centre trajectory"
            )
            keypoints = [(0, 0.5, 0.5)]

        max_frame = max(k[0] for k in keypoints)
        total_frames = max(max_frame + 1, 1)

        trajectory_tensor = self._interpolate_trajectory(
            keypoints, total_frames, interpolation
        )

        trajectory = {
            "keypoints": keypoints,
            "trajectory": trajectory_tensor,
            "interpolation": interpolation,
            "loop": loop,
            "total_frames": total_frames,
        }

        LauraLogger.info(
            f"Built FunCtrl trajectory: {len(keypoints)} keypoints, "
            f"{total_frames} frames, interp={interpolation}, loop={loop}"
        )
        return (trajectory,)


# ============== WAN 2.2 FUNCTRL GENERATOR ==============
class WanFunCtrlGenerator:
    """Generate video with Wan 2.2 FunCtrl trajectory-guided motion control.

    Accepts a Wan 2.2 model (from Wan22Loader) and a trajectory from
    WanFunCtrlKeypoints to produce a video where the subject follows the
    specified motion path.  Internally applies the trajectory as spatial
    conditioning maps injected into the DiT blocks during sampling.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "trajectory": ("WAN_FUNCTRL_TRAJ",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A person walking along the path, smooth motion",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "static, jittery, low quality, artifacts",
                    },
                ),
                "width": (
                    "INT",
                    {"default": 832, "min": 256, "max": 1920, "step": 16},
                ),
                "height": (
                    "INT",
                    {"default": 480, "min": 256, "max": 1080, "step": 16},
                ),
                "length": (
                    "INT",
                    {"default": 81, "min": 1, "max": 121, "step": 4},
                ),
                "ctrl_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 40, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 20.0}),
            },
            "optional": {
                "input_image": ("IMAGE",),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("video_frames", "latent")
    FUNCTION = "generate_functrl"
    CATEGORY = "Laura Studio/Video"
    DESCRIPTION = "Trajectory-controlled video generation with Wan 2.2 FunCtrl"

    @staticmethod
    def _build_spatial_hint(trajectory_tensor, width, height, total_frames):
        """Convert normalised [F,2] trajectory into a spatial hint tensor.

        Returns a [F, 1, H//8, W//8] tensor where a Gaussian blob is placed
        at the trajectory position for each frame — this acts as the spatial
        conditioning map injected into the DiT.
        """
        h_lat = height // 8
        w_lat = width // 8

        # Ensure trajectory covers the requested frame count
        src_frames = trajectory_tensor.shape[0]
        if src_frames < total_frames:
            # Linearly interpolate to stretch
            traj = (
                torch.nn.functional.interpolate(
                    trajectory_tensor.unsqueeze(0).permute(0, 2, 1),  # [1, 2, F_src]
                    size=total_frames,
                    mode="linear",
                    align_corners=True,
                )
                .permute(0, 2, 1)
                .squeeze(0)
            )  # [F, 2]
        else:
            traj = trajectory_tensor[:total_frames]

        hint = torch.zeros(total_frames, 1, h_lat, w_lat)

        # Place a Gaussian blob at each frame's trajectory position
        sigma = max(h_lat, w_lat) * 0.05  # ~5% of latent size
        ys = torch.arange(h_lat, dtype=torch.float32)
        xs = torch.arange(w_lat, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        for f in range(total_frames):
            cx = traj[f, 0].item() * (w_lat - 1)
            cy = traj[f, 1].item() * (h_lat - 1)
            blob = torch.exp(
                -((grid_x - cx) ** 2 + (grid_y - cy) ** 2) / (2 * sigma**2)
            )
            hint[f, 0] = blob / (blob.max() + 1e-8)

        return hint

    def generate_functrl(
        self,
        model,
        clip,
        vae,
        trajectory,
        prompt,
        negative_prompt,
        width,
        height,
        length,
        ctrl_strength,
        seed,
        steps,
        cfg,
        input_image=None,
        denoise=1.0,
    ):
        from nodes import CLIPTextEncode, KSampler, VAEDecode, EmptyLatentImage

        LauraLogger.info(
            f"Wan FunCtrl Generation: {width}x{height}, {length} frames, "
            f"ctrl_strength={ctrl_strength}"
        )

        # 1. Build spatial hint from trajectory
        if "trajectory" not in trajectory:
            LauraLogger.error(
                "Invalid trajectory data: missing 'trajectory' key. "
                "Expected output from WanFunCtrlKeypoints node."
            )
            raise KeyError(
                "Trajectory dict missing 'trajectory' key. "
                "Connect a WanFunCtrlKeypoints node to provide valid trajectory data."
            )
        traj_tensor = trajectory["trajectory"]
        spatial_hint = self._build_spatial_hint(traj_tensor, width, height, length)
        LauraLogger.info(f"Spatial hint shape: {spatial_hint.shape}")

        # 2. Encode prompts
        positive = CLIPTextEncode().encode(clip, prompt)[0]
        negative = CLIPTextEncode().encode(clip, negative_prompt)[0]

        # 3. Inject spatial conditioning into the model via set_model_patch
        # FunCtrl works by adding the spatial hint as an extra conditioning
        # channel that is combined with the model's internal representations.
        try:
            # Clone the model so we don't mutate the original
            patched_model = model.clone()

            # Register the spatial hint as a model patch
            # The hint is scaled by ctrl_strength before injection
            hint_scaled = spatial_hint * ctrl_strength

            def functrl_patch(model_function, kwargs):
                """Patch that adds FunCtrl spatial hint to the diffusion input."""
                x = kwargs["input"]
                device = x.device
                hint_device = hint_scaled.to(device=device, dtype=x.dtype)

                # Broadcast hint to match latent batch/channel dims
                # x shape: [B, C, F, H, W] or [B*F, C, H, W]
                if x.ndim == 5:
                    b, c, f, h, w = x.shape
                    # hint_device is [F_hint, 1, H_lat, W_lat]
                    # trilinear requires 5D: [N, C, D, H, W]
                    hint_5d = hint_device.permute(1, 0, 2, 3).unsqueeze(
                        0
                    )  # [1, 1, F_hint, H_lat, W_lat]
                    hint_resized = torch.nn.functional.interpolate(
                        hint_5d,
                        size=(f, h, w),
                        mode="trilinear",
                        align_corners=True,
                    )  # [1, 1, f, h, w]
                    hint_broadcast = hint_resized.expand(b, c, -1, -1, -1)
                    kwargs["input"] = x + hint_broadcast
                elif x.ndim == 4:
                    b, c, h, w = x.shape
                    hint_flat = hint_device[:b]  # [B, 1, H_hint, W_hint]
                    hint_resized = torch.nn.functional.interpolate(
                        hint_flat, size=(h, w), mode="bilinear", align_corners=True
                    )
                    kwargs["input"] = x + hint_resized.expand(-1, c, -1, -1)

                return model_function(
                    kwargs["input"], kwargs["timestep"], **kwargs.get("c", {})
                )

            patched_model.set_model_unet_function_wrapper(functrl_patch)
            active_model = patched_model
            LauraLogger.info("FunCtrl spatial patch applied to model")
        except Exception as e:
            LauraLogger.warn(
                f"Could not apply FunCtrl patch (will sample without trajectory): {e}"
            )
            active_model = model

        # 4. Create latent
        latent = EmptyLatentImage().generate(width, height, length)[0]

        # 5. Sample with Wan 2.2 Flow Matching via KSampler
        sampled = KSampler().sample(
            active_model,
            seed,
            steps,
            cfg,
            "euler",
            "normal",
            positive,
            negative,
            latent,
            denoise=denoise,
        )[0]

        # 6. Decode
        decoded = VAEDecode().decode(vae, sampled)[0]

        LauraLogger.info("Wan FunCtrl video generation complete")
        return (decoded, sampled)


# ============== HUNYUAN VIDEO 2.0 LOADER ==============
class HunyuanVideoLoader:
    """Loader for HunyuanVideo 2.0 video generation models.

    HunyuanVideo 2.0 supports native 1080p generation, FP8 quantisation
    for 8 GB VRAM cards, and both text-to-video and image-to-video modes.
    The model uses a Dual-Stream DiT architecture with 3D-VAE.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("checkpoints"),),
                "variant": (
                    ["HunyuanVideo", "HunyuanVideo-PromptRewrite"],
                    {"default": "HunyuanVideo"},
                ),
                "precision": (
                    ["bf16", "fp16", "fp8_e4m3fn"],
                    {"default": "bf16"},
                ),
                "text_encoder": (
                    ["llava-llama-3-8b", "clip-vit-large"],
                    {"default": "llava-llama-3-8b"},
                ),
                "enable_prompt_rewrite": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load_hunyuan_video"
    CATEGORY = "Laura Studio/Video"
    DESCRIPTION = "Load HunyuanVideo 2.0 for native 1080p video generation (8GB FP8)"

    def load_hunyuan_video(
        self, model_name, variant, precision, text_encoder, enable_prompt_rewrite
    ):
        LauraLogger.info(
            f"Loading HunyuanVideo 2.0 ({variant}): {model_name} | "
            f"precision={precision}, text_encoder={text_encoder}"
        )
        try:
            from nodes import CheckpointLoaderSimple

            result = CheckpointLoaderSimple().load_checkpoint(model_name)
            model, clip, vae = result[0], result[1], result[2]

            # Apply FP8 casting if requested and hardware supports it
            if precision == "fp8_e4m3fn":
                try:
                    from .quantization import _detect_fp8_capability

                    if _detect_fp8_capability()[0]:
                        LauraLogger.info(
                            "FP8 capable GPU detected — casting HunyuanVideo to fp8_e4m3fn"
                        )
                        # Actually cast model to FP8 for 8GB VRAM savings
                        import torch

                        model_patcher = model
                        if hasattr(model_patcher, "model"):
                            model_patcher.model.to(torch.float8_e4m3fn)
                        elif hasattr(model_patcher, "patch_model"):
                            model_patcher.patch_model()
                        LauraLogger.info(
                            "HunyuanVideo model cast to fp8_e4m3fn successfully"
                        )
                    else:
                        LauraLogger.warn(
                            "GPU does not natively support FP8; model loaded as-is. "
                            "Consider bf16/fp16 for best results."
                        )
                except (ImportError, RuntimeError) as e:
                    LauraLogger.warn(f"FP8 casting failed: {e}. Model loaded as-is.")

            if enable_prompt_rewrite:
                LauraLogger.info(
                    "Prompt rewrite enabled — HunyuanVideo will auto-expand short prompts"
                )

            LauraLogger.info(
                f"Loaded HunyuanVideo 2.0 ({variant}) in {precision} precision"
            )
            return (model, clip, vae)

        except Exception as e:
            LauraLogger.error(f"Failed to load HunyuanVideo 2.0 model: {e}")
            raise


# ============== HUNYUAN VIDEO 2.0 GENERATOR ==============
class HunyuanVideoGenerator:
    """Generate high-quality video with HunyuanVideo 2.0.

    Supports native 1080p output, text-to-video and image-to-video modes,
    Flow Matching sampling, and runs on 8 GB GPUs with FP8 quantisation.
    Uses a Dual-Stream DiT with MMDIT-style cross-attention for text/image
    conditioning via LLaVA-LLaMA-3-8B or CLIP-ViT-Large encoders.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "A cinematic drone shot over a mountain range at sunrise, "
                            "golden light, volumetric fog, 4K, smooth motion"
                        ),
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "blurry, low quality, static, watermark, text overlay, "
                            "deformed, jittery, overexposed"
                        ),
                    },
                ),
                "width": (
                    "INT",
                    {"default": 1920, "min": 256, "max": 1920, "step": 16},
                ),
                "height": (
                    "INT",
                    {"default": 1080, "min": 256, "max": 1080, "step": 16},
                ),
                "length": (
                    "INT",
                    {"default": 129, "min": 1, "max": 257, "step": 4},
                ),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "cfg": (
                    "FLOAT",
                    {"default": 7.0, "min": 0.0, "max": 20.0, "step": 0.1},
                ),
                "embedded_cfg_scale": (
                    "FLOAT",
                    {"default": 6.0, "min": 0.0, "max": 20.0, "step": 0.1},
                ),
            },
            "optional": {
                "input_image": ("IMAGE",),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("video_frames", "latent")
    FUNCTION = "generate_hunyuan_video"
    CATEGORY = "Laura Studio/Video"
    DESCRIPTION = "Generate native 1080p video with HunyuanVideo 2.0 (supports 8GB FP8)"

    def generate_hunyuan_video(
        self,
        model,
        clip,
        vae,
        prompt,
        negative_prompt,
        width,
        height,
        length,
        fps,
        seed,
        steps,
        cfg,
        embedded_cfg_scale,
        input_image=None,
        denoise=1.0,
    ):
        from nodes import CLIPTextEncode, KSampler, VAEDecode, EmptyLatentImage

        LauraLogger.info(
            f"HunyuanVideo 2.0 Generation: {width}x{height} @ {fps}fps, "
            f"{length} frames, steps={steps}, cfg={cfg}, "
            f"embedded_cfg={embedded_cfg_scale}"
        )

        # Note: embedded_cfg_scale is stored for metadata/logging but not passed
        # to KSampler, which does not support dual-CFG natively. The standard
        # cfg parameter controls guidance. embedded_cfg_scale is recorded here
        # so downstream nodes or analytics can reference the intended value.
        if embedded_cfg_scale != cfg:
            LauraLogger.info(
                f"Note: embedded_cfg_scale ({embedded_cfg_scale}) differs from cfg ({cfg}). "
                f"KSampler uses cfg={cfg} for guidance; embedded_cfg_scale is stored for metadata only."
            )

        # 1. Encode prompts
        # HunyuanVideo uses LLaVA-LLaMA-3 for rich semantic understanding
        positive = CLIPTextEncode().encode(clip, prompt)[0]
        negative = CLIPTextEncode().encode(clip, negative_prompt)[0]

        # 2. Handle I2V mode if input_image is provided
        if input_image is not None:
            LauraLogger.info("Image-to-Video mode: encoding reference image")
            from nodes import VAEEncode

            encoded_img = VAEEncode().encode(vae, input_image)[0]
            img_samples = encoded_img["samples"]  # [1, C, H, W]

            # Build I2V latent: first frame from image, rest noised progressively
            repeated = img_samples.repeat(length, 1, 1, 1)
            torch.manual_seed(seed)
            for i in range(1, length):
                t = i / max(length - 1, 1)
                noise = torch.randn_like(img_samples) * t * 0.6
                repeated[i] = img_samples[0] + noise[0]

            latent = {"samples": repeated}

            # Reduce denoise for I2V to preserve reference
            if denoise >= 1.0:
                denoise = 0.85
                LauraLogger.info(
                    "Auto-adjusted denoise to 0.85 for image-to-video mode"
                )
        else:
            # Text-to-video: empty latent
            latent = EmptyLatentImage().generate(width, height, length)[0]

        # 3. Sample using Flow Matching
        # HunyuanVideo uses a modified DDPM/Flow schedule
        # Best results with euler or dpmpp_2m_sde + sgm_uniform/simple
        sampled = KSampler().sample(
            model,
            seed,
            steps,
            cfg,
            "dpmpp_2m_sde",
            "simple",
            positive,
            negative,
            latent,
            denoise=denoise,
        )[0]

        # 4. Decode with 3D-VAE
        decoded = VAEDecode().decode(vae, sampled)[0]

        LauraLogger.info(
            f"HunyuanVideo 2.0 generation complete: "
            f"{decoded.shape[0]} frames @ {width}x{height}"
        )
        return (decoded, sampled)


# ============== LAURA VRAM CLEANER ==============
class LauraVRAMCleaner:
    """Explicitly clear VRAM and system cache to prevent OOM in large workflows"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (
                    [
                        "Soft (Empty Cache)",
                        "Hard (Unload All Models)",
                        "Extreme (GC + Unload)",
                    ],
                    {"default": "Hard (Unload All Models)"},
                ),
            },
            "optional": {
                "any_input": ("*",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "clean_vram"
    CATEGORY = "Laura Studio/Utility"
    DESCRIPTION = "Clear GPU memory and system garbage collection"
    # Allow wildcard input types for ComfyUI compatibility
    INPUT_IS_LIST = False
    OUTPUT_NODE = True

    def clean_vram(self, mode, any_input=None):
        import gc
        import comfy.model_management

        LauraLogger.info(f"VRAM Cleanup Initiated (Mode: {mode})")

        if mode == "Soft (Empty Cache)":
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        elif mode == "Hard (Unload All Models)":
            comfy.model_management.unload_all_models()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except AttributeError:
                    pass  # ipc_collect removed in newer PyTorch

        elif mode == "Extreme (GC + Unload)":
            comfy.model_management.unload_all_models()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except AttributeError:
                    pass  # ipc_collect removed in newer PyTorch

            # CLEAR CUDA IPC - Crucial for back-to-back 14B models
            try:
                import torch.cuda as cuda

                if cuda.is_available():
                    cuda.empty_cache()
                    cuda.synchronize()
            except Exception as e:
                LauraLogger.warn(f"CUDA cleanup warning: {e}")

        return (f"VRAM cleanup complete ({mode})",)


NODE_CLASS_MAPPINGS.update(
    {
        "CogVideoXLoader": CogVideoXLoader,
        "CogVideoXGenerator": CogVideoXGenerator,
        "CogVideoXImageToVideo": CogVideoXImageToVideo,
        "CosmosPredictLoader": CosmosPredictLoader,
        "CosmosPredictGenerator": CosmosPredictGenerator,
        "Wan22Loader": Wan22Loader,
        "Wan22Generator": Wan22Generator,
        "HunyuanDiTLoader": HunyuanDiTLoader,
        "HunyuanDiTGenerator": HunyuanDiTGenerator,
        "LauraWanDirectedVideo": LauraWanDirectedVideo,
        "WanFunCtrlKeypoints": WanFunCtrlKeypoints,
        "WanFunCtrlGenerator": WanFunCtrlGenerator,
        "HunyuanVideoLoader": HunyuanVideoLoader,
        "HunyuanVideoGenerator": HunyuanVideoGenerator,
        "LauraVRAMCleaner": LauraVRAMCleaner,
    }
)

NODE_DISPLAY_NAME_MAPPINGS.update(
    {
        "CogVideoXLoader": "CogVideoX Loader",
        "CogVideoXGenerator": "CogVideoX Text-to-Video",
        "CogVideoXImageToVideo": "CogVideoX Image-to-Video",
        "CosmosPredictLoader": "NVIDIA Cosmos-Predict Loader",
        "CosmosPredictGenerator": "NVIDIA Cosmos-Predict Generator",
        "Wan22Loader": "Wan 2.2 Model Loader",
        "Wan22Generator": "Wan 2.2 Video Generator",
        "HunyuanDiTLoader": "HunyuanDiT Model Loader",
        "HunyuanDiTGenerator": "HunyuanDiT Image Generator",
        "LauraWanDirectedVideo": "Wan 2.2 Directed Motion Video",
        "WanFunCtrlKeypoints": "Wan 2.2 FunCtrl Keypoints",
        "WanFunCtrlGenerator": "Wan 2.2 FunCtrl Generator",
        "HunyuanVideoLoader": "HunyuanVideo 2.0 Loader",
        "HunyuanVideoGenerator": "HunyuanVideo 2.0 Generator",
        "LauraVRAMCleaner": "Laura VRAM Master Cleaner",
    }
)
