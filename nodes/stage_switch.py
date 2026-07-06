"""
Laura Image Studio - Stage Switch Nodes
Lazy-evaluation stage switches that prevent upstream execution when disabled
"""

try:
    from comfy.graph_utils import ExecutionBlocker
except (ImportError, ModuleNotFoundError):
    ExecutionBlocker = None

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _blocker():
    if ExecutionBlocker is not None:
        return ExecutionBlocker()
    return None


class LauraStageSwitch:
    INPUT_IS_LIST = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "stage_name": ("STRING", {"default": "Stage"}),
            },
            "optional": {
                "input_image": ("IMAGE", {"lazy": True}),
                "input_latent": ("LATENT", {"lazy": True}),
                "input_mask": ("MASK", {"lazy": True}),
                "input_model": ("MODEL", {"lazy": True}),
                "input_clip": ("CLIP", {"lazy": True}),
                "input_vae": ("VAE", {"lazy": True}),
                "input_conditioning_pos": ("CONDITIONING", {"lazy": True}),
                "input_conditioning_neg": ("CONDITIONING", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "MASK", "MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("image", "latent", "mask", "model", "clip", "vae", "conditioning_pos", "conditioning_neg", "is_enabled", "stage_name")
    FUNCTION = "execute"
    CATEGORY = "Laura Studio/Core"
    DESCRIPTION = "Stage switch with lazy evaluation — disabled stages skip all upstream computation"

    OPTIONAL_INPUTS = ("input_image", "input_latent", "input_mask", "input_model", "input_clip", "input_vae", "input_conditioning_pos", "input_conditioning_neg")

    def check_lazy_status(self, enabled, stage_name, **kwargs):
        if not enabled:
            return []
        needed = []
        for name in self.OPTIONAL_INPUTS:
            if name in kwargs and isinstance(kwargs[name], ExecutionBlocker if ExecutionBlocker is not None else type(None)):
                needed.append(name)
        return needed

    def execute(self, enabled, stage_name, **kwargs):
        if not enabled:
            b = _blocker()
            return (b, b, b, b, b, b, b, b, False, stage_name)

        return (
            kwargs.get("input_image"),
            kwargs.get("input_latent"),
            kwargs.get("input_mask"),
            kwargs.get("input_model"),
            kwargs.get("input_clip"),
            kwargs.get("input_vae"),
            kwargs.get("input_conditioning_pos"),
            kwargs.get("input_conditioning_neg"),
            True,
            stage_name,
        )


class LauraStageGate:
    INPUT_IS_LIST = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "any_input": ("*", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = "Laura Studio/Core"
    DESCRIPTION = "Simple gate — passes input through when enabled, blocks upstream execution when disabled"

    def check_lazy_status(self, enabled, **kwargs):
        if not enabled:
            return []
        if "any_input" in kwargs and isinstance(kwargs["any_input"], ExecutionBlocker if ExecutionBlocker is not None else type(None)):
            return ["any_input"]
        return []

    def execute(self, enabled, any_input=None):
        if not enabled:
            return (_blocker(),)
        return (any_input,)


NODE_CLASS_MAPPINGS["LauraStageSwitch"] = LauraStageSwitch
NODE_CLASS_MAPPINGS["LauraStageGate"] = LauraStageGate
NODE_DISPLAY_NAME_MAPPINGS["LauraStageSwitch"] = "Laura Stage Switch"
NODE_DISPLAY_NAME_MAPPINGS["LauraStageGate"] = "Laura Stage Gate"
