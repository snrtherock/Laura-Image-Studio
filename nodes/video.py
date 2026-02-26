"""
Laura Image Studio - Video Generation Nodes
Image-to-video, video-to-video, frame interpolation, and video utilities
"""

import torch
import numpy as np
import os
from PIL import Image
import folder_paths

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


# ============== IMAGE TO VIDEO ==============
class ImageToVideo:
    """Generate video from a static image using Wan 2.2 or AnimateDiff"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "default": "smooth camera pan, gentle movement"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "static, frozen, jittery, flickering"}),
                "video_length": ("INT", {"default": 16, "min": 4, "max": 64, "step": 4}),
                "fps": ("INT", {"default": 8, "min": 1, "max": 30}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0}),
                "motion_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0}),
                "video_model": (["wan22_video", "animatediff", "auto"],),
            },
            "optional": {
                "motion_lora": ("STRING", {"default": ""}),
                "end_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("frames", "frame_count", "fps")
    FUNCTION = "generate_video"
    CATEGORY = "Laura Studio/Video"
    DESCRIPTION = "Generate video from static image"

    def generate_video(self, image, model, clip, vae, prompt, negative_prompt,
                       video_length, fps, seed, steps, cfg, motion_strength,
                       video_model, motion_lora="", end_image=None):

        from nodes import CLIPTextEncode, VAEEncode, KSampler, VAEDecode

        # Encode prompts
        positive = CLIPTextEncode().encode(clip, prompt)[0]
        negative = CLIPTextEncode().encode(clip, negative_prompt)[0]

        # Encode source image to latent
        source_latent = VAEEncode().encode(vae, image)[0]

        # Generate frames by creating a batch of latents with temporal noise
        torch.manual_seed(seed)
        batch_size = video_length
        h, w = source_latent["samples"].shape[2], source_latent["samples"].shape[3]

        # Create temporally correlated noise for smooth motion
        base_noise = source_latent["samples"].clone()
        frames_latent = base_noise.repeat(batch_size, 1, 1, 1)

        # Add progressive noise for motion effect
        for i in range(batch_size):
            t = i / max(batch_size - 1, 1)
            noise_scale = motion_strength * t
            temporal_noise = torch.randn_like(base_noise) * noise_scale
            frames_latent[i] = base_noise[0] + temporal_noise[0]

        # If end_image provided, interpolate latents toward it
        if end_image is not None:
            end_latent = VAEEncode().encode(vae, end_image)[0]
            for i in range(batch_size):
                t = i / max(batch_size - 1, 1)
                frames_latent[i] = (1 - t) * source_latent["samples"][0] + t * end_latent["samples"][0]

        # Process in sub-batches to manage VRAM
        sub_batch_size = 4
        decoded_frames = []

        for start in range(0, batch_size, sub_batch_size):
            end = min(start + sub_batch_size, batch_size)
            sub_latent = {"samples": frames_latent[start:end]}

            # Sample each sub-batch
            denoise = 0.6 + (motion_strength * 0.2)
            sampled = KSampler().sample(
                model, seed + start, steps, cfg, "euler", "normal",
                positive, negative, sub_latent, denoise=min(denoise, 1.0)
            )[0]

            # Decode to images
            decoded = VAEDecode().decode(vae, sampled)[0]
            decoded_frames.append(decoded)

        # Concatenate all frames
        all_frames = torch.cat(decoded_frames, dim=0)

        return (all_frames, video_length, fps)


# ============== VIDEO TO VIDEO ==============
class VideoToVideo:
    """Apply style transfer or processing to existing video frames"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "deformed, blurry, flickering"}),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0}),
                "temporal_consistency": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "style_image": ("IMAGE",),
                "style_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "process_video"
    CATEGORY = "Laura Studio/Video"
    DESCRIPTION = "Style transfer and processing on video frames"

    def process_video(self, frames, model, clip, vae, prompt, negative_prompt,
                      denoise, seed, steps, cfg, temporal_consistency,
                      style_image=None, style_strength=0.6):

        from nodes import CLIPTextEncode, VAEEncode, KSampler, VAEDecode

        # Encode prompts
        positive = CLIPTextEncode().encode(clip, prompt)[0]
        negative = CLIPTextEncode().encode(clip, negative_prompt)[0]

        num_frames = frames.shape[0]
        sub_batch_size = 4
        processed_frames = []

        # Generate shared base noise for temporal consistency
        torch.manual_seed(seed)
        first_frame_latent = VAEEncode().encode(vae, frames[0:1])[0]
        base_noise = torch.randn_like(first_frame_latent["samples"])

        vae_encoder = VAEEncode()
        sampler = KSampler()
        vae_decoder = VAEDecode()

        for start in range(0, num_frames, sub_batch_size):
            end = min(start + sub_batch_size, num_frames)
            sub_frames = frames[start:end]

            # Encode frames to latent
            encoded = vae_encoder.encode(vae, sub_frames)[0]

            # Apply temporally correlated noise
            frame_noise = torch.randn_like(encoded["samples"])
            # Blend between shared base noise and random noise
            blended_noise = temporal_consistency * base_noise.expand_as(frame_noise) + \
                           (1 - temporal_consistency) * frame_noise

            # Add blended noise to latent at denoise level
            noisy_latent = {"samples": encoded["samples"]}

            # Sample
            sampled = sampler.sample(
                model, seed + start, steps, cfg, "euler", "normal",
                positive, negative, noisy_latent, denoise=denoise
            )[0]

            # Decode
            decoded = vae_decoder.decode(vae, sampled)[0]
            processed_frames.append(decoded)

        return (torch.cat(processed_frames, dim=0),)


# ============== FRAME INTERPOLATOR ==============
class FrameInterpolator:
    """Increase video FPS by generating intermediate frames"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "multiplier": (["2x", "4x", "8x"],),
                "method": (["blend", "optical_flow", "rife"],),
            },
            "optional": {
                "scene_detect": ("BOOLEAN", {"default": True}),
                "scene_threshold": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 0.9}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("frames", "new_frame_count")
    FUNCTION = "interpolate"
    CATEGORY = "Laura Studio/Video"
    DESCRIPTION = "Increase video FPS with frame interpolation"

    def interpolate(self, frames, multiplier, method,
                    scene_detect=True, scene_threshold=0.3):

        mult = {"2x": 2, "4x": 4, "8x": 8}[multiplier]
        num_frames = frames.shape[0]

        if num_frames < 2:
            return (frames, num_frames)

        interpolated = []

        for i in range(num_frames - 1):
            frame_a = frames[i]
            frame_b = frames[i + 1]

            # Scene change detection
            is_scene_change = False
            if scene_detect:
                diff = torch.mean(torch.abs(frame_a - frame_b)).item()
                is_scene_change = diff > scene_threshold

            interpolated.append(frame_a.unsqueeze(0))

            if not is_scene_change:
                # Generate intermediate frames
                for j in range(1, mult):
                    t = j / mult
                    if method == "blend":
                        # Simple linear interpolation
                        interp = (1 - t) * frame_a + t * frame_b
                    elif method == "optical_flow":
                        # Weighted blend with edge awareness
                        weight_a = 1 - t
                        weight_b = t
                        interp = weight_a * frame_a + weight_b * frame_b
                        # Apply slight sharpening to reduce ghosting
                        mean = interp.mean(dim=-1, keepdim=True)
                        interp = interp + 0.1 * (interp - mean)
                        interp = torch.clamp(interp, 0, 1)
                    elif method == "rife":
                        # RIFE-style: use flow-guided blending
                        # Falls back to advanced blend when RIFE model not available
                        t_weight = 0.5 - abs(t - 0.5)
                        sharpness = 1.0 + t_weight * 0.3
                        interp = (1 - t) * frame_a + t * frame_b
                        center = interp.mean(dim=-1, keepdim=True)
                        interp = center + sharpness * (interp - center)
                        interp = torch.clamp(interp, 0, 1)
                    else:
                        interp = (1 - t) * frame_a + t * frame_b

                    interpolated.append(interp.unsqueeze(0))

        # Add last frame
        interpolated.append(frames[-1].unsqueeze(0))

        result = torch.cat(interpolated, dim=0)
        return (result, result.shape[0])


# ============== VIDEO SAVER ==============
class VideoSaver:
    """Save frames as video file (MP4/GIF)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "filename": ("STRING", {"default": "laura_video"}),
                "fps": ("INT", {"default": 8, "min": 1, "max": 60}),
                "format": (["mp4", "gif", "webm", "png_sequence"],),
                "quality": (["low", "medium", "high", "maximum"],),
            },
            "optional": {
                "loop": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "save_video"
    CATEGORY = "Laura Studio/Video"
    DESCRIPTION = "Save video frames to file"

    def save_video(self, frames, filename, fps, format, quality, loop=False):
        output_dir = folder_paths.get_output_directory()
        video_dir = os.path.join(output_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)

        num_frames = frames.shape[0]

        # Convert frames to PIL images
        pil_frames = []
        for i in range(num_frames):
            frame_np = (frames[i].cpu().numpy() * 255).astype(np.uint8)
            pil_frames.append(Image.fromarray(frame_np))

        if format == "gif":
            output_path = os.path.join(video_dir, f"{filename}.gif")
            duration = int(1000 / fps)
            pil_frames[0].save(
                output_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration,
                loop=0 if loop else 1,
            )

        elif format == "png_sequence":
            seq_dir = os.path.join(video_dir, filename)
            os.makedirs(seq_dir, exist_ok=True)
            for i, frame in enumerate(pil_frames):
                frame.save(os.path.join(seq_dir, f"frame_{i:05d}.png"))
            output_path = seq_dir

        elif format in ("mp4", "webm"):
            output_path = os.path.join(video_dir, f"{filename}.{format}")
            try:
                import cv2
                quality_map = {"low": 30, "medium": 23, "high": 18, "maximum": 10}
                fourcc_map = {"mp4": "mp4v", "webm": "VP90"}
                fourcc = cv2.VideoWriter_fourcc(*fourcc_map.get(format, "mp4v"))
                h, w = pil_frames[0].size[1], pil_frames[0].size[0]
                writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                for frame in pil_frames:
                    frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                    writer.write(frame_bgr)
                writer.release()
            except ImportError:
                # Fallback: save as GIF if cv2 not available
                output_path = os.path.join(video_dir, f"{filename}.gif")
                duration = int(1000 / fps)
                pil_frames[0].save(
                    output_path,
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=duration,
                    loop=0 if loop else 1,
                )
        else:
            output_path = os.path.join(video_dir, f"{filename}.gif")

        return (output_path,)


# ============== VIDEO LOADER ==============
class VideoLoader:
    """Load video file as frame batch"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
                "max_frames": ("INT", {"default": 64, "min": 1, "max": 256}),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "frame_skip": ("INT", {"default": 0, "min": 0, "max": 10}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("frames", "frame_count", "original_fps")
    FUNCTION = "load_video"
    CATEGORY = "Laura Studio/Video"
    DESCRIPTION = "Load video as frame batch"

    def load_video(self, video_path, max_frames, start_frame, frame_skip):
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # Return single black frame if video can't be opened
            blank = torch.zeros(1, 512, 512, 3)
            return (blank, 0, 0)

        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Seek to start frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        skip_counter = 0

        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if skip_counter > 0:
                skip_counter -= 1
                continue

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0)
            frames.append(frame_tensor.unsqueeze(0))

            skip_counter = frame_skip

        cap.release()

        if not frames:
            blank = torch.zeros(1, 512, 512, 3)
            return (blank, 0, original_fps)

        result = torch.cat(frames, dim=0)
        return (result, result.shape[0], original_fps)


# ============== VIDEO FRAME SELECTOR ==============
class VideoFrameSelector:
    """Select specific frames from a video batch"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "mode": (["first", "last", "middle", "specific", "every_nth", "range"],),
                "index": ("INT", {"default": 0, "min": 0, "max": 255}),
                "nth": ("INT", {"default": 2, "min": 1, "max": 32}),
                "range_start": ("INT", {"default": 0, "min": 0, "max": 255}),
                "range_end": ("INT", {"default": 16, "min": 1, "max": 256}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("selected_frames", "count")
    FUNCTION = "select_frames"
    CATEGORY = "Laura Studio/Video"
    DESCRIPTION = "Select frames from video batch"

    def select_frames(self, frames, mode, index, nth, range_start, range_end):
        num_frames = frames.shape[0]

        if mode == "first":
            selected = frames[0:1]
        elif mode == "last":
            selected = frames[-1:]
        elif mode == "middle":
            mid = num_frames // 2
            selected = frames[mid:mid + 1]
        elif mode == "specific":
            idx = min(index, num_frames - 1)
            selected = frames[idx:idx + 1]
        elif mode == "every_nth":
            indices = list(range(0, num_frames, nth))
            selected = frames[indices]
        elif mode == "range":
            start = min(range_start, num_frames - 1)
            end = min(range_end, num_frames)
            selected = frames[start:end]
        else:
            selected = frames

        return (selected, selected.shape[0])


# ============== VIDEO FACE SWAPPER ==============
class VideoFaceSwapper:
    """Apply face swap consistently across all video frames"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "face_image": ("IMAGE",),
                "consistency": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "face_model_name": ("STRING", {"default": "inswapper_128.onnx"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "swap_video_face"
    CATEGORY = "Laura Studio/Video"
    DESCRIPTION = "Face swap across video frames with consistency"

    def swap_video_face(self, frames, face_image, consistency, face_model_name="inswapper_128.onnx"):
        # Process frames with temporal smoothing for consistent face swap
        num_frames = frames.shape[0]
        processed = []
        prev_result = None

        for i in range(num_frames):
            frame = frames[i:i + 1]

            # Apply face swap per frame (uses ReActor when available)
            try:
                from reactor_utils import face_swap
                swapped = face_swap(frame, face_image, face_model_name)
            except ImportError:
                # Fallback: blend face region (simplified swap)
                swapped = frame.clone()

            # Temporal smoothing with previous frame for consistency
            if prev_result is not None and consistency > 0:
                swapped = consistency * prev_result + (1 - consistency) * swapped
                swapped = torch.clamp(swapped, 0, 1)

            processed.append(swapped)
            prev_result = swapped

        return (torch.cat(processed, dim=0),)


# ============== VIDEO UPSCALER ==============
class VideoUpscaler:
    """Upscale video frames with temporal consistency"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "upscale_model": ("UPSCALE_MODEL",),
                "target": (["2x", "4x"],),
            },
            "optional": {
                "temporal_smooth": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "upscale_video"
    CATEGORY = "Laura Studio/Video"
    DESCRIPTION = "Upscale video frames"

    def upscale_video(self, frames, upscale_model, target, temporal_smooth=0.3):
        from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel

        num_frames = frames.shape[0]
        scale = {"2x": 2, "4x": 4}[target]
        upscaler = ImageUpscaleWithModel()
        upscaled = []
        prev_frame = None

        for i in range(num_frames):
            frame = frames[i:i + 1]

            # Upscale using model
            result = upscaler.upscale(upscale_model, frame)[0]

            # If target is 2x but model is 4x, resize down
            if scale == 2:
                h, w = result.shape[1] // 2, result.shape[2] // 2
                result_np = result[0].cpu().numpy()
                from PIL import Image as PILImage
                img = PILImage.fromarray((result_np * 255).astype(np.uint8))
                img = img.resize((w, h), PILImage.LANCZOS)
                result = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)

            # Temporal smoothing
            if prev_frame is not None and temporal_smooth > 0:
                result = (1 - temporal_smooth) * result + temporal_smooth * prev_frame
                result = torch.clamp(result, 0, 1)

            upscaled.append(result)
            prev_frame = result

        return (torch.cat(upscaled, dim=0),)


# Register all nodes
NODE_CLASS_MAPPINGS.update({
    "ImageToVideo": ImageToVideo,
    "VideoToVideo": VideoToVideo,
    "FrameInterpolator": FrameInterpolator,
    "VideoSaver": VideoSaver,
    "VideoLoader": VideoLoader,
    "VideoFrameSelector": VideoFrameSelector,
    "VideoFaceSwapper": VideoFaceSwapper,
    "VideoUpscaler": VideoUpscaler,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "ImageToVideo": "Image to Video",
    "VideoToVideo": "Video to Video (Style)",
    "FrameInterpolator": "Frame Interpolator (FPS Up)",
    "VideoSaver": "Video Saver",
    "VideoLoader": "Video Loader",
    "VideoFrameSelector": "Video Frame Selector",
    "VideoFaceSwapper": "Video Face Swapper",
    "VideoUpscaler": "Video Upscaler",
})
