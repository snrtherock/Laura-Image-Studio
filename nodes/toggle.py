"""
Laura Image Studio - Toggle/Bypass Nodes
Conditional pass-through nodes for enabling/disabling pipeline stages
"""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


# ============== TOGGLE IMAGE SWITCH ==============
class ToggleImageSwitch:
    """Pass through or bypass IMAGE processing"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "processed_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "switch"
    CATEGORY = "Laura Studio/Toggle"
    DESCRIPTION = "Toggle between processed and original image"

    def switch(self, input_image, enabled, processed_image=None):
        if enabled and processed_image is not None:
            return (processed_image,)
        return (input_image,)


# ============== TOGGLE LATENT SWITCH ==============
class ToggleLatentSwitch:
    """Pass through or bypass LATENT processing"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_latent": ("LATENT",),
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "processed_latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "switch"
    CATEGORY = "Laura Studio/Toggle"
    DESCRIPTION = "Toggle between processed and original latent"

    def switch(self, input_latent, enabled, processed_latent=None):
        if enabled and processed_latent is not None:
            return (processed_latent,)
        return (input_latent,)


# ============== TOGGLE MASK SWITCH ==============
class ToggleMaskSwitch:
    """Pass through or bypass MASK processing"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_mask": ("MASK",),
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "processed_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "switch"
    CATEGORY = "Laura Studio/Toggle"
    DESCRIPTION = "Toggle between processed and original mask"

    def switch(self, input_mask, enabled, processed_mask=None):
        if enabled and processed_mask is not None:
            return (processed_mask,)
        return (input_mask,)


# ============== TOGGLE MODEL SWITCH ==============
class ToggleModelSwitch:
    """Pass through or bypass MODEL processing"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "processed_model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "switch"
    CATEGORY = "Laura Studio/Toggle"
    DESCRIPTION = "Toggle between processed and original model"

    def switch(self, input_model, enabled, processed_model=None):
        if enabled and processed_model is not None:
            return (processed_model,)
        return (input_model,)


# ============== TOGGLE CLIP SWITCH ==============
class ToggleClipSwitch:
    """Pass through or bypass CLIP processing"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_clip": ("CLIP",),
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "processed_clip": ("CLIP",),
            },
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "switch"
    CATEGORY = "Laura Studio/Toggle"
    DESCRIPTION = "Toggle between processed and original CLIP"

    def switch(self, input_clip, enabled, processed_clip=None):
        if enabled and processed_clip is not None:
            return (processed_clip,)
        return (input_clip,)


# ============== TOGGLE CONDITIONING SWITCH ==============
class ToggleConditioningSwitch:
    """Pass through or bypass CONDITIONING processing"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_conditioning": ("CONDITIONING",),
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "processed_conditioning": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "switch"
    CATEGORY = "Laura Studio/Toggle"
    DESCRIPTION = "Toggle between processed and original conditioning"

    def switch(self, input_conditioning, enabled, processed_conditioning=None):
        if enabled and processed_conditioning is not None:
            return (processed_conditioning,)
        return (input_conditioning,)


# ============== PIPELINE TOGGLE ==============
class PipelineToggle:
    """Toggle entire pipeline stage (IMAGE + LATENT + MASK)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "stage_name": ("STRING", {"default": "Upscaling"}),
            },
            "optional": {
                "input_image": ("IMAGE",),
                "processed_image": ("IMAGE",),
                "input_latent": ("LATENT",),
                "processed_latent": ("LATENT",),
                "input_mask": ("MASK",),
                "processed_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "MASK", "BOOLEAN", "STRING")
    RETURN_NAMES = ("image", "latent", "mask", "was_enabled", "stage_name")
    FUNCTION = "toggle_pipeline"
    CATEGORY = "Laura Studio/Toggle"
    DESCRIPTION = "Toggle entire pipeline stage on/off"

    def toggle_pipeline(
        self,
        enabled,
        stage_name,
        input_image=None,
        processed_image=None,
        input_latent=None,
        processed_latent=None,
        input_mask=None,
        processed_mask=None,
    ):
        if enabled:
            out_image = processed_image if processed_image is not None else input_image
            out_latent = (
                processed_latent if processed_latent is not None else input_latent
            )
            out_mask = processed_mask if processed_mask is not None else input_mask
        else:
            out_image = input_image
            out_latent = input_latent
            out_mask = input_mask

        # Provide safe defaults when outputs would be None to prevent downstream crashes
        if out_image is None:
            import torch

            out_image = torch.zeros(1, 64, 64, 3)
        if out_latent is None:
            import torch

            out_latent = {"samples": torch.zeros(1, 4, 8, 8)}
        if out_mask is None:
            import torch

            out_mask = torch.zeros(1, 64, 64)

        return (out_image, out_latent, out_mask, enabled, stage_name)


# ============== TOGGLE VAE SWITCH ==============
class ToggleVAESwitch:
    """Pass through or bypass VAE processing"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_vae": ("VAE",),
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "processed_vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "switch"
    CATEGORY = "Laura Studio/Toggle"
    DESCRIPTION = "Toggle between processed and original VAE"

    def switch(self, input_vae, enabled, processed_vae=None):
        if enabled and processed_vae is not None:
            return (processed_vae,)
        return (input_vae,)


# ============== WORKFLOW TOGGLE PANEL ==============
class WorkflowTogglePanel:
    """Central toggle panel for ALL workflow stages - the master control"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable_upscaling": ("BOOLEAN", {"default": True}),
                "enable_face_ops": ("BOOLEAN", {"default": True}),
                "enable_background": ("BOOLEAN", {"default": True}),
                "enable_dressing": ("BOOLEAN", {"default": False}),
                "enable_virtual_tryon": ("BOOLEAN", {"default": False}),
                "enable_dressing_room": ("BOOLEAN", {"default": False}),
                "enable_inpainting": ("BOOLEAN", {"default": False}),
                "enable_outpainting": ("BOOLEAN", {"default": False}),
                "enable_detail_enhance": ("BOOLEAN", {"default": True}),
                "enable_video": ("BOOLEAN", {"default": False}),
                "enable_cinema_video": ("BOOLEAN", {"default": False}),
                "enable_directed_motion": ("BOOLEAN", {"default": False}),
                "enable_face_drive": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (
        "BOOLEAN",
        "BOOLEAN",
        "BOOLEAN",
        "BOOLEAN",
        "BOOLEAN",
        "BOOLEAN",
        "BOOLEAN",
        "BOOLEAN",
        "BOOLEAN",
        "BOOLEAN",
        "BOOLEAN",
        "BOOLEAN",
        "BOOLEAN",
    )
    RETURN_NAMES = (
        "upscaling",
        "face_ops",
        "background",
        "dressing",
        "virtual_tryon",
        "dressing_room",
        "inpainting",
        "outpainting",
        "detail_enhance",
        "video",
        "cinema_video",
        "directed_motion",
        "face_drive",
    )
    FUNCTION = "get_toggles"
    CATEGORY = "Laura Studio/Toggle"
    DESCRIPTION = "Master control panel for all workflow stage toggles"

    def get_toggles(
        self,
        enable_upscaling,
        enable_face_ops,
        enable_background,
        enable_dressing,
        enable_virtual_tryon,
        enable_dressing_room,
        enable_inpainting,
        enable_outpainting,
        enable_detail_enhance,
        enable_video,
        enable_cinema_video,
        enable_directed_motion,
        enable_face_drive,
    ):
        return (
            enable_upscaling,
            enable_face_ops,
            enable_background,
            enable_dressing,
            enable_virtual_tryon,
            enable_dressing_room,
            enable_inpainting,
            enable_outpainting,
            enable_detail_enhance,
            enable_video,
            enable_cinema_video,
            enable_directed_motion,
            enable_face_drive,
        )


# Register all nodes
NODE_CLASS_MAPPINGS.update(
    {
        "ToggleImageSwitch": ToggleImageSwitch,
        "ToggleLatentSwitch": ToggleLatentSwitch,
        "ToggleMaskSwitch": ToggleMaskSwitch,
        "ToggleModelSwitch": ToggleModelSwitch,
        "ToggleClipSwitch": ToggleClipSwitch,
        "ToggleConditioningSwitch": ToggleConditioningSwitch,
        "ToggleVAESwitch": ToggleVAESwitch,
        "PipelineToggle": PipelineToggle,
        "WorkflowTogglePanel": WorkflowTogglePanel,
    }
)

NODE_DISPLAY_NAME_MAPPINGS.update(
    {
        "ToggleImageSwitch": "Toggle Switch (Image)",
        "ToggleLatentSwitch": "Toggle Switch (Latent)",
        "ToggleMaskSwitch": "Toggle Switch (Mask)",
        "ToggleModelSwitch": "Toggle Switch (Model)",
        "ToggleClipSwitch": "Toggle Switch (CLIP)",
        "ToggleConditioningSwitch": "Toggle Switch (Conditioning)",
        "ToggleVAESwitch": "Toggle Switch (VAE)",
        "PipelineToggle": "Pipeline Stage Toggle",
        "WorkflowTogglePanel": "Workflow Toggle Panel (Master)",
    }
)
