"""
Laura Image Studio - Terminal Save Node
Final output routing node for saving images/videos with metadata
"""

import os
import json
import numpy as np
from datetime import datetime
from PIL import Image
from PIL.PngImagePlugin import PngInfo

try:
    import folder_paths
except ImportError:
    folder_paths = None

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


class LauraTerminalSave:
    """Save images and video frames with generation metadata"""

    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename_prefix": ("STRING", {"default": "Laura"}),
                "output_type": (["image", "video", "both"], {"default": "image"}),
            },
            "optional": {
                "images": ("IMAGE",),
                "video_frames": ("IMAGE",),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
                "format": (["png", "jpg", "webp"], {"default": "png"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_output"
    CATEGORY = "Laura Studio/Core"
    DESCRIPTION = "Terminal node that saves images/videos with metadata"

    def _get_output_dir(self):
        if folder_paths is not None:
            return folder_paths.get_output_directory()
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")

    def _get_next_counter(self, output_dir, prefix):
        counter = 1
        existing = [
            f for f in os.listdir(output_dir)
            if f.startswith(prefix) and os.path.isfile(os.path.join(output_dir, f))
        ]
        if existing:
            for f in existing:
                name = os.path.splitext(f)[0]
                parts = name.rsplit("_", 1)
                if len(parts) == 2:
                    try:
                        counter = max(counter, int(parts[1]) + 1)
                    except ValueError:
                        pass
        return counter

    def _tensor_to_pil(self, tensor):
        arr = tensor.cpu().numpy()
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _build_metadata(self):
        metadata = PngInfo()
        metadata.add_text("Generator", "Laura Image Studio")
        metadata.add_text("Timestamp", datetime.now().isoformat())
        return metadata

    def _save_images(self, images, output_dir, prefix, fmt, quality):
        saved = []
        counter = self._get_next_counter(output_dir, prefix)
        metadata = self._build_metadata()

        for i in range(images.shape[0]):
            pil_img = self._tensor_to_pil(images[i])
            filename = f"{prefix}_{counter:05d}.{fmt}"
            filepath = os.path.join(output_dir, filename)

            save_kwargs = {}
            if fmt == "png":
                save_kwargs["pnginfo"] = metadata
            elif fmt in ("jpg", "webp"):
                save_kwargs["quality"] = quality

            save_fmt = "JPEG" if fmt == "jpg" else fmt.upper()
            pil_img.save(filepath, format=save_fmt, **save_kwargs)
            saved.append({"filename": filename, "subfolder": "", "type": "output"})
            counter += 1

        return saved

    def _save_video_frames(self, frames, output_dir, prefix, fps):
        saved = []
        try:
            import av
            filename = f"{prefix}_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            filepath = os.path.join(output_dir, filename)
            h, w = frames.shape[1], frames.shape[2]

            container = av.open(filepath, mode="w")
            stream = container.add_stream("h264", rate=fps)
            stream.width = w
            stream.height = h
            stream.pix_fmt = "yuv420p"

            for i in range(frames.shape[0]):
                arr = (frames[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                av_frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
                for packet in stream.encode(av_frame):
                    container.mux(packet)

            for packet in stream.encode():
                container.mux(packet)
            container.close()
            saved.append({"filename": filename, "subfolder": "", "type": "output"})
        except ImportError:
            frame_dir = os.path.join(output_dir, f"{prefix}_frames")
            os.makedirs(frame_dir, exist_ok=True)
            for i in range(frames.shape[0]):
                pil_img = self._tensor_to_pil(frames[i])
                fname = f"frame_{i:05d}.png"
                pil_img.save(os.path.join(frame_dir, fname))
                saved.append({"filename": os.path.join(f"{prefix}_frames", fname), "subfolder": "", "type": "output"})
        return saved

    def save_output(self, filename_prefix, output_type, images=None, video_frames=None,
                    fps=24, format="png", quality=95):
        output_dir = self._get_output_dir()
        os.makedirs(output_dir, exist_ok=True)

        saved_images = []
        saved_videos = []

        if output_type in ("image", "both") and images is not None:
            saved_images = self._save_images(images, output_dir, filename_prefix, format, quality)

        if output_type in ("video", "both") and video_frames is not None:
            saved_videos = self._save_video_frames(video_frames, output_dir, filename_prefix, fps)

        all_saved = saved_images + saved_videos
        return {"ui": {"images": all_saved}}


class LauraShowText:
    """Display text in the ComfyUI UI. Replaces dependency on ShowText|pysssss."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute"
    CATEGORY = "Laura Studio/Core"
    OUTPUT_NODE = True

    def execute(self, text=""):
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS["LauraTerminalSave"] = LauraTerminalSave
NODE_CLASS_MAPPINGS["LauraShowText"] = LauraShowText
NODE_DISPLAY_NAME_MAPPINGS["LauraTerminalSave"] = "Laura Terminal Save"
NODE_DISPLAY_NAME_MAPPINGS["LauraShowText"] = "Laura Show Text"
