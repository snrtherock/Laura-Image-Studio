"""
Laura Image Studio - Batch Processing Nodes
Manage image queues and batch generation
"""

import torch

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


# ============== BATCH IMAGE QUEUE ==============
class BatchImageQueue:
    """Queue multiple images for sequential processing"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image_batch", "batch_size")
    FUNCTION = "create_batch"
    CATEGORY = "Laura Studio/Batch"
    DESCRIPTION = "Combine up to 6 images into a single batch"

    def create_batch(
        self,
        image_1,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
    ):
        images = [image_1]
        for img in [image_2, image_3, image_4, image_5, image_6]:
            if img is not None:
                images.append(img)

        # Ensure all images have same dimensions for batching
        # In a real ComfyUI node, we might want to resize them to match image_1
        target_h, target_w = image_1.shape[1], image_1.shape[2]
        processed_images = []

        import torch.nn.functional as F

        for img in images:
            if img.shape[1] != target_h or img.shape[2] != target_w:
                # Resize to match first image
                resized = F.interpolate(
                    img.permute(0, 3, 1, 2), size=(target_h, target_w), mode="bilinear"
                ).permute(0, 2, 3, 1)
                processed_images.append(resized)
            else:
                processed_images.append(img)

        batch = torch.cat(processed_images, dim=0)
        return (batch, len(processed_images))


# ============== BATCH PROMPT LIST ==============
class BatchPromptList:
    """Sequential list of prompts for batch generation"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_1": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "prompt_2": ("STRING", {"multiline": True, "default": ""}),
                "prompt_3": ("STRING", {"multiline": True, "default": ""}),
                "prompt_4": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_list",)
    FUNCTION = "create_list"
    CATEGORY = "Laura Studio/Batch"
    DESCRIPTION = "Create a list of prompts for sequential generation"
    OUTPUT_IS_LIST = (True,)

    def create_list(self, prompt_1, prompt_2="", prompt_3="", prompt_4=""):
        prompts = [prompt_1]
        for p in [prompt_2, prompt_3, prompt_4]:
            if p:
                prompts.append(p)
        return (prompts,)


# ============== BATCH IMAGE SELECTOR ==============
class BatchImageSelector:
    """Select a specific image from a batch"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_batch": ("IMAGE",),
                "index": ("INT", {"default": 0, "min": 0, "max": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "select_image"
    CATEGORY = "Laura Studio/Batch"
    DESCRIPTION = "Select a specific image from a batch by index"

    def select_image(self, image_batch, index):
        if index >= image_batch.shape[0]:
            index = image_batch.shape[0] - 1
        return (image_batch[index : index + 1],)


# ============== BATCH ITERATOR (MOCK) ==============
class BatchIterator:
    """Helper node for iterating over batches (Logic usually handled by ComfyUI core)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_batch": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("current_image", "current_index", "total_count")
    FUNCTION = "iterate"
    CATEGORY = "Laura Studio/Batch"
    DESCRIPTION = "Iterate over a batch, returning one image at a time"

    def iterate(self, image_batch):
        # In a real ComfyUI context, this might interact with the execution engine
        # For now, we provide it as a utility to split batches
        return (image_batch[0:1], 0, image_batch.shape[0])


NODE_CLASS_MAPPINGS.update(
    {
        "BatchImageQueue": BatchImageQueue,
        "BatchPromptList": BatchPromptList,
        "BatchImageSelector": BatchImageSelector,
        "BatchIterator": BatchIterator,
    }
)

NODE_DISPLAY_NAME_MAPPINGS.update(
    {
        "BatchImageQueue": "Batch Image Queue",
        "BatchPromptList": "Batch Prompt List",
        "BatchImageSelector": "Batch Image Selector",
        "BatchIterator": "Batch Iterator",
    }
)
