"""
Laura Image Studio - Checkpoint/Crash Recovery Nodes
Save and restore intermediate pipeline state for crash recovery
"""

import torch
import json
import os
import time
import numpy as np
from PIL import Image
import folder_paths

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


# ============== CHECKPOINT SAVER ==============
class CheckpointSaver:
    """Save intermediate pipeline state to disk for crash recovery"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "checkpoint_name": ("STRING", {"default": "checkpoint_01"}),
                "save_path": ("STRING", {"default": "laura_checkpoints"}),
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "latent": ("LATENT",),
                "mask": ("MASK",),
                "metadata": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "MASK", "STRING")
    RETURN_NAMES = ("image", "latent", "mask", "checkpoint_path")
    FUNCTION = "save_checkpoint"
    CATEGORY = "Laura Studio/Checkpoint"
    DESCRIPTION = "Save intermediate state for crash recovery"

    def save_checkpoint(
        self,
        image,
        checkpoint_name,
        save_path,
        enabled,
        latent=None,
        mask=None,
        metadata="",
    ):
        checkpoint_dir = ""

        if enabled:
            output_dir = folder_paths.get_output_directory()
            checkpoint_dir = os.path.join(output_dir, save_path)
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Save image as PNG
            img_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_image.png")
            img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(img_np).save(img_path)

            # Save latent as torch tensor
            latent_path = ""
            if latent is not None:
                latent_path = os.path.join(
                    checkpoint_dir, f"{checkpoint_name}_latent.pt"
                )
                torch.save(latent, latent_path)

            # Save mask as torch tensor
            mask_path = ""
            if mask is not None:
                mask_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_mask.pt")
                torch.save(mask, mask_path)

            # Write manifest
            manifest = {
                "checkpoint_name": checkpoint_name,
                "timestamp": time.time(),
                "timestamp_human": time.strftime("%Y-%m-%d %H:%M:%S"),
                "image_path": img_path,
                "latent_path": latent_path,
                "mask_path": mask_path,
                "image_shape": list(image.shape),
                "metadata": metadata,
            }
            manifest_path = os.path.join(
                checkpoint_dir, f"{checkpoint_name}_manifest.json"
            )
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

        # Provide safe defaults for optional outputs
        if latent is None:
            latent = {"samples": torch.zeros(1, 4, 8, 8)}
        if mask is None:
            mask = torch.zeros(1, 64, 64)
        return (image, latent, mask, checkpoint_dir)


# ============== PIPELINE CHECKPOINT LOADER ==============
class PipelineCheckpointLoader:
    """Load previously saved checkpoint for resume"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint_name": ("STRING", {"default": "checkpoint_01"}),
                "load_path": ("STRING", {"default": "laura_checkpoints"}),
            },
            "optional": {
                "fallback_image": ("IMAGE",),
                "fallback_latent": ("LATENT",),
                "fallback_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "MASK", "BOOLEAN", "STRING")
    RETURN_NAMES = ("image", "latent", "mask", "checkpoint_found", "info")
    FUNCTION = "load_checkpoint"
    CATEGORY = "Laura Studio/Checkpoint"
    DESCRIPTION = "Load saved checkpoint for resume"

    def load_checkpoint(
        self,
        checkpoint_name,
        load_path,
        fallback_image=None,
        fallback_latent=None,
        fallback_mask=None,
    ):
        output_dir = folder_paths.get_output_directory()
        checkpoint_dir = os.path.join(output_dir, load_path)
        manifest_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_manifest.json")

        if not os.path.exists(manifest_path):
            info = f"Checkpoint '{checkpoint_name}' not found, using fallback"
            # Provide safe defaults when fallbacks are None
            out_image = fallback_image
            out_latent = fallback_latent
            out_mask = fallback_mask
            if out_image is None:
                out_image = torch.zeros(1, 64, 64, 3)
            if out_latent is None:
                out_latent = {"samples": torch.zeros(1, 4, 8, 8)}
            if out_mask is None:
                out_mask = torch.zeros(1, 64, 64)
            return (out_image, out_latent, out_mask, False, info)

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # Load image
        image = fallback_image
        if manifest.get("image_path") and os.path.exists(manifest["image_path"]):
            img = Image.open(manifest["image_path"]).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            image = torch.from_numpy(img_np).unsqueeze(0)

        # Load latent
        latent = fallback_latent
        if manifest.get("latent_path") and os.path.exists(manifest["latent_path"]):
            latent = torch.load(manifest["latent_path"], weights_only=True)

        # Load mask
        mask = fallback_mask
        if manifest.get("mask_path") and os.path.exists(manifest["mask_path"]):
            mask = torch.load(manifest["mask_path"], weights_only=True)

        saved_time = manifest.get("timestamp_human", "unknown")
        info = f"Loaded checkpoint '{checkpoint_name}' saved at {saved_time}"

        # Provide safe defaults if files were missing and fallbacks were None
        if image is None:
            image = torch.zeros(1, 64, 64, 3)
        if latent is None:
            latent = {"samples": torch.zeros(1, 4, 8, 8)}
        if mask is None:
            mask = torch.zeros(1, 64, 64)

        return (image, latent, mask, True, info)


# ============== CHECKPOINT MANAGER ==============
class CheckpointManager:
    """Manage checkpoint lifecycle - list, clean, prune old checkpoints"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["list", "clean_old", "clean_all", "status"],),
                "checkpoint_path": ("STRING", {"default": "laura_checkpoints"}),
                "max_age_hours": ("INT", {"default": 24, "min": 1, "max": 168}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "manage"
    CATEGORY = "Laura Studio/Checkpoint"
    DESCRIPTION = "Manage saved checkpoints"

    def manage(self, action, checkpoint_path, max_age_hours):
        output_dir = folder_paths.get_output_directory()
        checkpoint_dir = os.path.join(output_dir, checkpoint_path)

        if not os.path.exists(checkpoint_dir):
            return (f"Checkpoint directory not found: {checkpoint_dir}",)

        if action == "list":
            manifests = [
                f for f in os.listdir(checkpoint_dir) if f.endswith("_manifest.json")
            ]
            if not manifests:
                return ("No checkpoints found.",)
            lines = []
            for m in sorted(manifests):
                with open(os.path.join(checkpoint_dir, m), "r") as f:
                    data = json.load(f)
                lines.append(
                    f"- {data['checkpoint_name']} ({data.get('timestamp_human', 'unknown')})"
                )
            return ("\n".join(lines),)

        elif action == "clean_old":
            cutoff = time.time() - (max_age_hours * 3600)
            removed = 0
            manifests = [
                f for f in os.listdir(checkpoint_dir) if f.endswith("_manifest.json")
            ]
            for m in manifests:
                mpath = os.path.join(checkpoint_dir, m)
                with open(mpath, "r") as f:
                    data = json.load(f)
                if data.get("timestamp", 0) < cutoff:
                    # Remove associated files
                    for key in ["image_path", "latent_path", "mask_path"]:
                        fpath = data.get(key, "")
                        if fpath and os.path.exists(fpath):
                            os.remove(fpath)
                    os.remove(mpath)
                    removed += 1
            return (f"Removed {removed} checkpoints older than {max_age_hours} hours.",)

        elif action == "clean_all":
            files = os.listdir(checkpoint_dir)
            for f in files:
                fpath = os.path.join(checkpoint_dir, f)
                if os.path.isfile(fpath):
                    os.remove(fpath)
            return (f"Removed all {len(files)} files from checkpoint directory.",)

        elif action == "status":
            total_size = 0
            file_count = 0
            for f in os.listdir(checkpoint_dir):
                fpath = os.path.join(checkpoint_dir, f)
                if os.path.isfile(fpath):
                    total_size += os.path.getsize(fpath)
                    file_count += 1
            size_mb = total_size / (1024 * 1024)
            manifests = [
                f for f in os.listdir(checkpoint_dir) if f.endswith("_manifest.json")
            ]
            return (
                f"Checkpoints: {len(manifests)}, Files: {file_count}, Size: {size_mb:.1f} MB",
            )

        return ("Unknown action.",)


# ============== AUTO CHECKPOINT ==============
class AutoCheckpoint:
    """Automatic checkpoint with stage-based naming"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "stage": (
                    [
                        "post_generation",
                        "post_face",
                        "post_dressing",
                        "post_inpainting",
                        "post_background",
                        "pre_upscale",
                        "post_upscale",
                        "final",
                    ],
                ),
                "enabled": ("BOOLEAN", {"default": True}),
                "workflow_id": ("STRING", {"default": "default"}),
            },
            "optional": {
                "latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "auto_save"
    CATEGORY = "Laura Studio/Checkpoint"
    DESCRIPTION = "Auto-save checkpoint at pipeline stage"

    def auto_save(self, image, stage, enabled, workflow_id, latent=None):
        if enabled:
            timestamp = int(time.time())
            checkpoint_name = f"{workflow_id}_{stage}_{timestamp}"

            output_dir = folder_paths.get_output_directory()
            checkpoint_dir = os.path.join(output_dir, "laura_checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Save image
            img_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_image.png")
            img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(img_np).save(img_path)

            # Save latent
            latent_path = ""
            if latent is not None:
                latent_path = os.path.join(
                    checkpoint_dir, f"{checkpoint_name}_latent.pt"
                )
                torch.save(latent, latent_path)

            # Write manifest
            manifest = {
                "checkpoint_name": checkpoint_name,
                "stage": stage,
                "workflow_id": workflow_id,
                "timestamp": time.time(),
                "timestamp_human": time.strftime("%Y-%m-%d %H:%M:%S"),
                "image_path": img_path,
                "latent_path": latent_path,
                "image_shape": list(image.shape),
            }
            manifest_path = os.path.join(
                checkpoint_dir, f"{checkpoint_name}_manifest.json"
            )
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

        if latent is None:
            latent = {"samples": torch.zeros(1, 4, 8, 8)}
        return (image, latent)


# ============== RESUME FROM CHECKPOINT ==============


class ResumeFromCheckpoint:
    """Resume workflow from last successful checkpoint"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["fresh_start", "resume_latest", "resume_specific"],),
                "fresh_image": ("IMAGE",),
                "checkpoint_path": ("STRING", {"default": "laura_checkpoints"}),
                "workflow_id": ("STRING", {"default": "default"}),
            },
            "optional": {
                "specific_checkpoint": ("STRING", {"default": ""}),
                "fresh_latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "STRING", "BOOLEAN")
    RETURN_NAMES = ("image", "latent", "resumed_from", "is_resumed")
    FUNCTION = "resume"
    CATEGORY = "Laura Studio/Checkpoint"
    DESCRIPTION = "Resume from crash or start fresh"

    def resume(
        self,
        mode,
        fresh_image,
        checkpoint_path,
        workflow_id,
        specific_checkpoint="",
        fresh_latent=None,
    ):
        # Ensure latent is never None for downstream consumers
        if fresh_latent is None:
            fresh_latent = {"samples": torch.zeros(1, 4, 8, 8)}

        if mode == "fresh_start":
            return (fresh_image, fresh_latent, "fresh_start", False)

        output_dir = folder_paths.get_output_directory()
        checkpoint_dir = os.path.join(output_dir, checkpoint_path)

        if mode == "resume_specific" and specific_checkpoint:
            manifest_path = os.path.join(
                checkpoint_dir, f"{specific_checkpoint}_manifest.json"
            )
            if os.path.exists(manifest_path):
                return self._load_from_manifest(
                    manifest_path, fresh_image, fresh_latent
                )

        if mode == "resume_latest":
            if not os.path.exists(checkpoint_dir):
                return (fresh_image, fresh_latent, "no_checkpoints_found", False)

            # Find latest manifest matching workflow_id
            manifests = []
            for f in os.listdir(checkpoint_dir):
                if f.endswith("_manifest.json"):
                    mpath = os.path.join(checkpoint_dir, f)
                    with open(mpath, "r") as fh:
                        data = json.load(fh)
                    if data.get("workflow_id", "") == workflow_id:
                        manifests.append((data.get("timestamp", 0), mpath))

            if not manifests:
                return (fresh_image, fresh_latent, "no_matching_checkpoints", False)

            # Sort by timestamp, load most recent
            manifests.sort(key=lambda x: x[0], reverse=True)
            return self._load_from_manifest(manifests[0][1], fresh_image, fresh_latent)

        return (fresh_image, fresh_latent, "unknown_mode", False)

    def _load_from_manifest(self, manifest_path, fallback_image, fallback_latent):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # Load image
        image = fallback_image
        if manifest.get("image_path") and os.path.exists(manifest["image_path"]):
            img = Image.open(manifest["image_path"]).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            image = torch.from_numpy(img_np).unsqueeze(0)

        # Load latent
        latent = fallback_latent
        if manifest.get("latent_path") and os.path.exists(manifest["latent_path"]):
            latent = torch.load(manifest["latent_path"], weights_only=True)

        checkpoint_name = manifest.get("checkpoint_name", "unknown")
        if latent is None:
            latent = {"samples": torch.zeros(1, 4, 8, 8)}
        return (image, latent, checkpoint_name, True)


# Register all nodes
NODE_CLASS_MAPPINGS.update(
    {
        "CheckpointSaver": CheckpointSaver,
        "PipelineCheckpointLoader": PipelineCheckpointLoader,
        "CheckpointManager": CheckpointManager,
        "AutoCheckpoint": AutoCheckpoint,
        "ResumeFromCheckpoint": ResumeFromCheckpoint,
    }
)

NODE_DISPLAY_NAME_MAPPINGS.update(
    {
        "CheckpointSaver": "Checkpoint Saver",
        "PipelineCheckpointLoader": "Pipeline Checkpoint Loader",
        "CheckpointManager": "Checkpoint Manager",
        "AutoCheckpoint": "Auto Checkpoint (Stage)",
        "ResumeFromCheckpoint": "Resume From Checkpoint",
    }
)
