"""
Laura Image Studio - Node Package
Export all node modules
"""

from .generation import NODE_CLASS_MAPPINGS as generation_mappings
from .generation import NODE_DISPLAY_NAME_MAPPINGS as generation_names

from .models import NODE_CLASS_MAPPINGS as models_mappings
from .models import NODE_DISPLAY_NAME_MAPPINGS as models_names

from .toggle import NODE_CLASS_MAPPINGS as toggle_mappings
from .toggle import NODE_DISPLAY_NAME_MAPPINGS as toggle_names

from .checkpoint import NODE_CLASS_MAPPINGS as checkpoint_mappings
from .checkpoint import NODE_DISPLAY_NAME_MAPPINGS as checkpoint_names

from .video import NODE_CLASS_MAPPINGS as video_mappings
from .video import NODE_DISPLAY_NAME_MAPPINGS as video_names

from .dressing import NODE_CLASS_MAPPINGS as dressing_mappings
from .dressing import NODE_DISPLAY_NAME_MAPPINGS as dressing_names

from .face import NODE_CLASS_MAPPINGS as face_mappings
from .face import NODE_DISPLAY_NAME_MAPPINGS as face_names

from .inpainting import NODE_CLASS_MAPPINGS as inpainting_mappings
from .inpainting import NODE_DISPLAY_NAME_MAPPINGS as inpainting_names

from .upscaling import NODE_CLASS_MAPPINGS as upscaling_mappings
from .upscaling import NODE_DISPLAY_NAME_MAPPINGS as upscaling_names

from .background import NODE_CLASS_MAPPINGS as background_mappings
from .background import NODE_DISPLAY_NAME_MAPPINGS as background_names

# Combine all mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_CLASS_MAPPINGS.update(generation_mappings)
NODE_CLASS_MAPPINGS.update(models_mappings)
NODE_CLASS_MAPPINGS.update(toggle_mappings)
NODE_CLASS_MAPPINGS.update(checkpoint_mappings)
NODE_CLASS_MAPPINGS.update(video_mappings)
NODE_CLASS_MAPPINGS.update(dressing_mappings)
NODE_CLASS_MAPPINGS.update(face_mappings)
NODE_CLASS_MAPPINGS.update(inpainting_mappings)
NODE_CLASS_MAPPINGS.update(upscaling_mappings)
NODE_CLASS_MAPPINGS.update(background_mappings)

NODE_DISPLAY_NAME_MAPPINGS.update(generation_names)
NODE_DISPLAY_NAME_MAPPINGS.update(models_names)
NODE_DISPLAY_NAME_MAPPINGS.update(toggle_names)
NODE_DISPLAY_NAME_MAPPINGS.update(checkpoint_names)
NODE_DISPLAY_NAME_MAPPINGS.update(video_names)
NODE_DISPLAY_NAME_MAPPINGS.update(dressing_names)
NODE_DISPLAY_NAME_MAPPINGS.update(face_names)
NODE_DISPLAY_NAME_MAPPINGS.update(inpainting_names)
NODE_DISPLAY_NAME_MAPPINGS.update(upscaling_names)
NODE_DISPLAY_NAME_MAPPINGS.update(background_names)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
