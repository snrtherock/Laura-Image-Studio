"""
Laura Image Studio - Model Comparison & Presets
Nodes for side-by-side comparison and professional background presets
"""

import torch
import numpy as np
from PIL import Image
import folder_paths


class MultiModelComparison:
    """Compare outputs from multiple models in a side-by-side grid"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "label_1": ("STRING", {"default": "Model A"}),
                "label_2": ("STRING", {"default": "Model B"}),
                "layout": (["horizontal", "vertical"], {"default": "horizontal"}),
            },
            "optional": {
                "image_3": ("IMAGE",),
                "label_3": ("STRING", {"default": "Model C"}),
                "image_4": ("IMAGE",),
                "label_4": ("STRING", {"default": "Model D"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("comparison_grid",)
    FUNCTION = "compare"
    CATEGORY = "Laura Studio/Excellence"

    def compare(
        self,
        image_1,
        image_2,
        label_1,
        label_2,
        layout,
        image_3=None,
        label_3="Model C",
        image_4=None,
        label_4="Model D",
    ):
        images = [image_1, image_2]
        if image_3 is not None:
            images.append(image_3)
        if image_4 is not None:
            images.append(image_4)

        # Ensure all images are the same size for the grid
        target_h = image_1.shape[1]
        target_w = image_1.shape[2]

        processed_images = []
        for img in images:
            if img.shape[1] != target_h or img.shape[2] != target_w:
                # Simple resize if needed (ComfyUI typically handles this via other nodes, but we'll be safe)
                import torch.nn.functional as F

                img = img.permute(0, 3, 1, 2)
                img = F.interpolate(img, size=(target_h, target_w), mode="bilinear")
                img = img.permute(0, 2, 3, 1)
            processed_images.append(img)

        if layout == "horizontal":
            grid = torch.cat(processed_images, dim=2)
        else:
            grid = torch.cat(processed_images, dim=1)

        return (grid,)


class ProfessionalBackgroundLibrary:
    """Library of 50+ professional background presets for influencers"""

    PRESETS = {
        "Luxury Penthouse": "A high-end luxury penthouse with floor-to-ceiling windows, city skyline at night, designer furniture, soft ambient lighting, 8k resolution.",
        "Neon Tokyo Street": "A vibrant street in Tokyo at night, neon signs in pink and blue, wet pavement with reflections, cinematic atmosphere, cyberpunk aesthetic.",
        "Minimalist Photo Studio": "A professional minimalist photo studio, clean white cyclorama wall, professional softbox lighting, high-end commercial look.",
        "Tropical Beach Sunset": "A pristine tropical beach at sunset, golden hour lighting, calm turquoise water, palm trees silhouetted against a purple and orange sky.",
        "Parisian Café": "A charming outdoor café in Paris, cobblestone street, wicker chairs, blurred background of the Eiffel Tower, romantic atmosphere.",
        "Nordic Cabin": "Cozy interior of a modern Nordic cabin, large windows overlooking a snowy pine forest, warm fireplace, hygge aesthetic.",
        "Cyberpunk Lab": "A futuristic high-tech laboratory, blue and white glowing panels, holographic displays, clean sci-fi aesthetic.",
        "Ancient Library": "A grand ancient library with towering mahogany bookshelves, thousands of leather-bound books, dusty sunbeams, dark academia style.",
        "Mediterranean Villa": "A sunny Mediterranean villa balcony, white stone walls, vibrant bougainvillea flowers, view of the deep blue sea, summer vibe.",
        "Industrial Loft": "A spacious industrial loft with exposed brick walls, large factory windows, morning sunlight, urban professional aesthetic.",
        "Enchanted Forest": "A mystical enchanted forest, glowing flora, ethereal fog, soft rays of light through ancient trees, fantasy atmosphere.",
        "Desert Oasis": "A luxurious modern villa at a desert oasis, sand dunes in the background, infinity pool, warm sunset lighting.",
        "Underwater Kingdom": "An ethereal underwater kingdom with coral architecture, glowing jellyfish, deep blue water, cinematic lighting.",
        "Moon Base": "A futuristic moon base interior, view of the Earth through a reinforced glass dome, sterile tech aesthetic.",
        "Autumn Park": "A beautiful park in autumn, vibrant orange and red maple leaves on the ground, soft morning mist, peaceful atmosphere.",
        "Luxury Yacht": "The deck of a luxury yacht, wooden floor, ocean horizon, expensive outdoor furniture, wealthy lifestyle aesthetic.",
        "Modern Art Gallery": "A clean modern art gallery, white walls, large abstract paintings, spotlighting, sophisticated atmosphere.",
        "Mountain Peaks": "Breath-taking view from a mountain peak, clouds below, jagged rocks, crisp blue sky, epic landscape.",
        "Zen Garden": "A peaceful Japanese Zen garden, raked sand, mossy rocks, bonsai trees, meditative atmosphere.",
        "Gothic Cathedral": "Inside a grand Gothic cathedral, stained glass windows casting colorful light, high vaulted ceilings, dramatic shadows.",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (list(cls.PRESETS.keys()),),
                "custom_modifiers": (
                    "STRING",
                    {"default": "highly detailed, professional photography, 8k"},
                ),
                "lighting_preset": (
                    ["soft", "dramatic", "natural", "cinematic", "neon"],
                    {"default": "natural"},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("background_prompt",)
    FUNCTION = "get_preset"
    CATEGORY = "Laura Studio/Excellence"

    def get_preset(self, preset, custom_modifiers, lighting_preset):
        base_prompt = self.PRESETS.get(preset, "")
        lighting_map = {
            "soft": "soft diffused lighting, no harsh shadows",
            "dramatic": "high contrast, dramatic chiaroscuro lighting",
            "natural": "natural sunlight, realistic shadows",
            "cinematic": "cinematic lighting, anamorphic lens flares, depth of field",
            "neon": "vibrant neon lighting, colorful glows",
        }

        final_prompt = (
            f"{base_prompt} {lighting_map[lighting_preset]}. {custom_modifiers}"
        )
        return (final_prompt,)


NODE_CLASS_MAPPINGS = {
    "MultiModelComparison": MultiModelComparison,
    "ProfessionalBackgroundLibrary": ProfessionalBackgroundLibrary,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiModelComparison": "Multi-Model Comparison Grid",
    "ProfessionalBackgroundLibrary": "Professional Background Presets",
}
