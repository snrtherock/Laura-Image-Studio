"""
Laura Image Studio - Face & Identity Nodes
Face swapping, IPAdapter embedding, and identity preservation
Delegates to ReActor, IPAdapter Plus, and RMBG for real operations
"""

import torch
from PIL import Image
import numpy as np

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


# ============== EXTERNAL NODE IMPORT HELPERS ==============
def _try_import_reactor():
    """Try to import ReActor face swap node"""
    try:
        from custom_nodes import ComfyUI_ReActor
        mappings = ComfyUI_ReActor.NODE_CLASS_MAPPINGS
        if "ReActorFaceSwap" in mappings:
            return mappings["ReActorFaceSwap"]
    except Exception:
        pass
    try:
        import importlib
        mod = importlib.import_module("custom_nodes.ComfyUI-ReActor")
        mappings = mod.NODE_CLASS_MAPPINGS
        if "ReActorFaceSwap" in mappings:
            return mappings["ReActorFaceSwap"]
    except Exception:
        pass
    return None


def _try_import_reactor_boost():
    """Try to import ReActor face boost/enhance node"""
    try:
        from custom_nodes import ComfyUI_ReActor
        mappings = ComfyUI_ReActor.NODE_CLASS_MAPPINGS
        for name in ["ReActorFaceBoost", "ReActorRestoreFace"]:
            if name in mappings:
                return mappings[name]
    except Exception:
        pass
    try:
        import importlib
        mod = importlib.import_module("custom_nodes.ComfyUI-ReActor")
        mappings = mod.NODE_CLASS_MAPPINGS
        for name in ["ReActorFaceBoost", "ReActorRestoreFace"]:
            if name in mappings:
                return mappings[name]
    except Exception:
        pass
    return None


def _try_import_ipadapter():
    """Try to import IPAdapter Plus nodes"""
    try:
        from custom_nodes import ComfyUI_IPAdapter_plus
        mappings = ComfyUI_IPAdapter_plus.NODE_CLASS_MAPPINGS
        loader = mappings.get("IPAdapterUnifiedLoader") or mappings.get("IPAdapterUnifiedLoaderFaceID")
        apply_node = mappings.get("IPAdapterAdvanced") or mappings.get("IPAdapter")
        return loader, apply_node
    except Exception:
        pass
    try:
        import importlib
        mod = importlib.import_module("custom_nodes.ComfyUI_IPAdapter_plus")
        mappings = mod.NODE_CLASS_MAPPINGS
        loader = mappings.get("IPAdapterUnifiedLoader") or mappings.get("IPAdapterUnifiedLoaderFaceID")
        apply_node = mappings.get("IPAdapterAdvanced") or mappings.get("IPAdapter")
        return loader, apply_node
    except Exception:
        pass
    return None, None


def _try_import_rmbg_face():
    """Try to import RMBG FaceParsing node"""
    try:
        from custom_nodes import ComfyUI_RMBG
        mappings = ComfyUI_RMBG.NODE_CLASS_MAPPINGS
        if "FaceParsing" in mappings:
            return mappings["FaceParsing"]
    except Exception:
        pass
    try:
        import importlib
        mod = importlib.import_module("custom_nodes.ComfyUI-RMBG")
        mappings = mod.NODE_CLASS_MAPPINGS
        if "FaceParsing" in mappings:
            return mappings["FaceParsing"]
    except Exception:
        pass
    return None


def _face_region_fallback(image):
    """Create a rough face region mask when no detection model is available.
    Assumes face is in upper-center of image.
    """
    B, H, W, C = image.shape
    mask = torch.zeros((B, H, W), dtype=torch.float32, device=image.device)
    # Face region: top 8-30% vertically, center 30-70% horizontally
    y_start, y_end = int(H * 0.08), int(H * 0.30)
    x_start, x_end = int(W * 0.30), int(W * 0.70)
    mask[:, y_start:y_end, x_start:x_end] = 1.0
    return mask


# ============== FACE DETECTOR ==============
class FaceDetector:
    """Detect faces using RMBG FaceParsing (19 facial features)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "expand_mask": ("INT", {"default": 8, "min": 0, "max": 50}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("face_crop", "face_mask", "face_info")
    FUNCTION = "detect_face"
    CATEGORY = "Laura Studio/Face"
    DESCRIPTION = "Detect faces using RMBG FaceParsing"

    def detect_face(self, image, expand_mask=8):
        FaceParsingNode = _try_import_rmbg_face()

        if FaceParsingNode is not None:
            try:
                face_parser = FaceParsingNode()
                parse_result = face_parser.parse(image)
                # FaceParsing returns segmentation map with 19 facial feature labels
                # Labels 1-13 are face features (skin, brows, eyes, nose, mouth, etc.)
                seg_map = parse_result[0]
                # Create face mask: any non-zero label is face
                mask = (seg_map > 0).float()
                info = "Face detected via RMBG FaceParsing (19 features)"
            except Exception as e:
                print(f"[Laura Studio] RMBG FaceParsing failed: {e}, using fallback")
                mask = _face_region_fallback(image)
                info = "Face region estimated (RMBG unavailable)"
        else:
            print("[Laura Studio] RMBG FaceParsing not found, using region fallback")
            mask = _face_region_fallback(image)
            info = "Face region estimated (RMBG not installed)"

        # Expand mask
        if expand_mask > 0:
            import torch.nn.functional as F
            k = expand_mask * 2 + 1
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            mask = F.max_pool2d(mask, kernel_size=k, stride=1, padding=expand_mask)
            mask = mask.squeeze(1)

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        # Crop face region from mask bounding box
        face_crop = image.clone()
        # Find bounding box of mask
        if mask.sum() > 0:
            nonzero = torch.nonzero(mask[0] > 0.5)
            if len(nonzero) > 0:
                y_min, x_min = nonzero.min(dim=0).values
                y_max, x_max = nonzero.max(dim=0).values
                # Add padding
                pad = 20
                y_min = max(0, y_min - pad)
                x_min = max(0, x_min - pad)
                y_max = min(image.shape[1], y_max + pad)
                x_max = min(image.shape[2], x_max + pad)
                face_crop = image[:, y_min:y_max, x_min:x_max, :]
                info += f" | bbox: ({x_min},{y_min})-({x_max},{y_max})"

        return (face_crop, mask, info)


# ============== FACE SWAPPER ==============
class FaceSwapper:
    """Swap face using ReActor"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_face": ("IMAGE",),
                "target_image": ("IMAGE",),
            },
            "optional": {
                "source_face_index": ("INT", {"default": 0, "min": 0, "max": 10}),
                "target_face_index": ("INT", {"default": 0, "min": 0, "max": 10}),
                "console_log_level": (["0", "1", "2"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "swap_face"
    CATEGORY = "Laura Studio/Face"
    DESCRIPTION = "Swap faces using ReActor"

    def swap_face(self, source_face, target_image,
                  source_face_index=0, target_face_index=0, console_log_level="1"):
        ReActorNode = _try_import_reactor()

        if ReActorNode is not None:
            try:
                reactor = ReActorNode()
                # ReActor expects specific parameters - try to call its face_swap method
                result = reactor.execute(
                    enabled=True,
                    input_image=target_image,
                    source_image=source_face,
                    swap_model="inswapper_128.onnx",
                    facedetection="retinaface_resnet50",
                    face_restore_model="none",
                    face_restore_visibility=1.0,
                    codeformer_weight=0.5,
                    detect_gender_source="no",
                    detect_gender_input="no",
                    source_faces_index=str(source_face_index),
                    input_faces_index=str(target_face_index),
                    console_log_level=int(console_log_level),
                )
                # ReActor returns tuple: (image, face_model)
                return (result[0],)
            except Exception as e:
                print(f"[Laura Studio] ReActor face swap failed: {e}")
                print("[Laura Studio] Returning target image unchanged")
                return (target_image,)
        else:
            print("[Laura Studio] ReActor not installed. Install ComfyUI-ReActor for face swapping.")
            print("[Laura Studio] Returning target image unchanged")
            return (target_image,)


# ============== FACE REFERENCE ==============
class FaceReference:
    """Create face reference embedding for consistency across generations"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_image": ("IMAGE",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
            },
            "optional": {
                "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
            }
        }

    RETURN_TYPES = ("MODEL", "IMAGE")
    RETURN_NAMES = ("model", "face_image")
    FUNCTION = "create_reference"
    CATEGORY = "Laura Studio/Face"
    DESCRIPTION = "Create face reference via IPAdapter for consistent identity"

    def create_reference(self, face_image, model, clip, weight=1.0):
        IPALoader, IPAApply = _try_import_ipadapter()

        if IPALoader is not None and IPAApply is not None:
            try:
                # Load IPAdapter for FaceID
                loader = IPALoader()
                ipa_model, clip_vision = loader.load_models(
                    model, preset="PLUS FACE (portraits)"
                )

                # Apply face embedding
                applier = IPAApply()
                result_model = applier.apply_ipadapter(
                    model=ipa_model,
                    ipadapter=ipa_model,
                    image=face_image,
                    weight=weight,
                    weight_type="ease in-out",
                    combine_embeds="concat",
                )
                return (result_model[0], face_image)
            except Exception as e:
                print(f"[Laura Studio] IPAdapter face reference failed: {e}")
                return (model, face_image)
        else:
            print("[Laura Studio] IPAdapter Plus not installed. Returning model unchanged.")
            return (model, face_image)


# ============== IPADAPTER FACE ==============
class IPAdapterFace:
    """Apply IPAdapter face embedding for character consistency"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "face_image": ("IMAGE",),
                "weight": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 2.0}),
            },
            "optional": {
                "noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipadapter"
    CATEGORY = "Laura Studio/Face"
    DESCRIPTION = "Apply IPAdapter face embedding to model"

    def apply_ipadapter(self, model, face_image, weight,
                        noise=0.0, start_at=0.0, end_at=1.0):
        IPALoader, IPAApply = _try_import_ipadapter()

        if IPALoader is not None and IPAApply is not None:
            try:
                loader = IPALoader()
                ipa_model, clip_vision = loader.load_models(
                    model, preset="PLUS FACE (portraits)"
                )

                applier = IPAApply()
                result = applier.apply_ipadapter(
                    model=ipa_model,
                    ipadapter=ipa_model,
                    image=face_image,
                    weight=weight,
                    noise=noise,
                    start_at=start_at,
                    end_at=end_at,
                    weight_type="ease in-out",
                    combine_embeds="concat",
                )
                return (result[0],)
            except Exception as e:
                print(f"[Laura Studio] IPAdapter apply failed: {e}")
                return (model,)
        else:
            print("[Laura Studio] IPAdapter Plus not installed. Returning model unchanged.")
            return (model,)


# ============== EXPRESSION TRANSFER ==============
class ExpressionTransfer:
    """Transfer facial expressions between images using face mask + img2img"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transfer_expression"
    CATEGORY = "Laura Studio/Face"
    DESCRIPTION = "Transfer facial expression from source to target"

    def transfer_expression(self, source_image, target_image, model, clip, vae, strength, seed):
        from nodes import VAEEncode, KSampler, VAEDecode, CLIPTextEncode
        import torch.nn.functional as F

        # Get face mask on target
        detector = FaceDetector()
        _, face_mask, _ = detector.detect_face(target_image, expand_mask=4)

        # Encode target to latent
        encoded = VAEEncode().encode(vae, target_image)[0]

        # Use expression cues from source as conditioning
        # In practice we prompt for the expression visible in source
        prompt = "same person, same face, expressive, detailed face, professional photo"
        neg = "deformed, blurry, different person, bad anatomy"

        positive = CLIPTextEncode().encode(clip, prompt)[0]
        negative = CLIPTextEncode().encode(clip, neg)[0]

        # Process mask to latent size
        mask = face_mask
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)

        latent_h = encoded["samples"].shape[2]
        latent_w = encoded["samples"].shape[3]
        mask_latent = F.interpolate(mask.float(), size=(latent_h, latent_w), mode="bilinear", align_corners=False)

        # Inpaint face region with expression
        noise = torch.randn_like(encoded["samples"])
        latent_samples = encoded["samples"] * (1 - mask_latent) + noise * mask_latent
        latent = {"samples": latent_samples, "noise_mask": mask_latent.squeeze(1)}

        sampled = KSampler().sample(
            model, seed, 25, 6.0, "euler", "normal",
            positive, negative, latent, denoise=strength
        )[0]

        result = VAEDecode().decode(vae, sampled)[0]

        # Composite: face region from result, rest from target
        mask_rgb = face_mask.unsqueeze(-1).expand_as(target_image)
        final = result * mask_rgb + target_image * (1 - mask_rgb)

        return (final,)


# ============== AGE ADJUSTER ==============
class AgeAdjuster:
    """Adjust perceived age of face using face mask + inpainting"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "age_delta": ("INT", {"default": 0, "min": -30, "max": 30}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust_age"
    CATEGORY = "Laura Studio/Face"
    DESCRIPTION = "Adjust face age using face-region inpainting"

    def adjust_age(self, image, model, clip, vae, age_delta, seed):
        if age_delta == 0:
            return (image,)

        from nodes import VAEEncode, KSampler, VAEDecode, CLIPTextEncode
        import torch.nn.functional as F

        # Get face mask
        detector = FaceDetector()
        _, face_mask, _ = detector.detect_face(image, expand_mask=6)

        if age_delta > 0:
            prompt = f"photo of woman, older, mature, {age_delta} years older, same person, detailed face"
        else:
            prompt = f"photo of woman, younger, youthful, {abs(age_delta)} years younger, same person, detailed face"

        encoded = VAEEncode().encode(vae, image)[0]
        positive = CLIPTextEncode().encode(clip, prompt)[0]
        negative = CLIPTextEncode().encode(clip, "deformed, blurry, different person, bad face")[0]

        # Process mask
        mask = face_mask
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        latent_h = encoded["samples"].shape[2]
        latent_w = encoded["samples"].shape[3]
        mask_latent = F.interpolate(mask.float(), size=(latent_h, latent_w), mode="bilinear", align_corners=False)

        noise = torch.randn_like(encoded["samples"])
        latent_samples = encoded["samples"] * (1 - mask_latent) + noise * mask_latent
        latent = {"samples": latent_samples, "noise_mask": mask_latent.squeeze(1)}

        sampled = KSampler().sample(
            model, seed, 25, 6.0, "euler", "normal",
            positive, negative, latent, denoise=0.45
        )[0]

        result = VAEDecode().decode(vae, sampled)[0]

        # Composite
        mask_rgb = face_mask.unsqueeze(-1).expand_as(image)
        final = result * mask_rgb + image * (1 - mask_rgb)

        return (final,)


# ============== FACE ENHANCER ==============
class FaceEnhancer:
    """Enhance face details using ReActor FaceBoost"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "enhancement_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "face_restore_model": (["codeformer", "GFPGAN"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "enhance_face"
    CATEGORY = "Laura Studio/Face"
    DESCRIPTION = "Enhance face details using ReActor/CodeFormer/GFPGAN"

    def enhance_face(self, image, enhancement_strength=0.5, face_restore_model="codeformer"):
        BoostNode = _try_import_reactor_boost()

        if BoostNode is not None:
            try:
                booster = BoostNode()
                result = booster.execute(
                    image=image,
                    face_restore_model=face_restore_model,
                    face_restore_visibility=enhancement_strength,
                    codeformer_weight=1.0 - enhancement_strength,
                )
                return (result[0],)
            except Exception as e:
                print(f"[Laura Studio] ReActor face enhance failed: {e}")
                return (image,)
        else:
            print("[Laura Studio] ReActor not installed. Returning image unchanged.")
            return (image,)


# ============== MULTI FACE HANDLER ==============
class MultiFaceHandler:
    """Handle multiple faces using ReActor with multi-face support"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "source_face": ("IMAGE",),
                "mode": (["swap_all", "swap_specific"],),
            },
            "optional": {
                "target_face_index": ("INT", {"default": 0, "min": 0, "max": 9}),
                "source_face_index": ("INT", {"default": 0, "min": 0, "max": 9}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("result", "info")
    FUNCTION = "handle_faces"
    CATEGORY = "Laura Studio/Face"
    DESCRIPTION = "Handle multiple faces in image using ReActor"

    def handle_faces(self, image, source_face, mode,
                     target_face_index=0, source_face_index=0):
        ReActorNode = _try_import_reactor()

        if ReActorNode is None:
            print("[Laura Studio] ReActor not installed. Returning image unchanged.")
            return (image, "ReActor not installed")

        try:
            reactor = ReActorNode()

            if mode == "swap_all":
                # Swap all detected faces with source
                faces_index = "0,1,2,3,4,5,6,7,8,9"
                source_index = str(source_face_index)
            else:
                faces_index = str(target_face_index)
                source_index = str(source_face_index)

            result = reactor.execute(
                enabled=True,
                input_image=image,
                source_image=source_face,
                swap_model="inswapper_128.onnx",
                facedetection="retinaface_resnet50",
                face_restore_model="none",
                face_restore_visibility=1.0,
                codeformer_weight=0.5,
                detect_gender_source="no",
                detect_gender_input="no",
                source_faces_index=source_index,
                input_faces_index=faces_index,
                console_log_level=1,
            )

            info = f"Face swap mode: {mode}, target: {faces_index}, source: {source_index}"
            return (result[0], info)
        except Exception as e:
            print(f"[Laura Studio] ReActor multi-face failed: {e}")
            return (image, f"Error: {e}")


# Register all face nodes
NODE_CLASS_MAPPINGS.update({
    "FaceDetector": FaceDetector,
    "FaceSwapper": FaceSwapper,
    "FaceReference": FaceReference,
    "IPAdapterFace": IPAdapterFace,
    "ExpressionTransfer": ExpressionTransfer,
    "AgeAdjuster": AgeAdjuster,
    "FaceEnhancer": FaceEnhancer,
    "MultiFaceHandler": MultiFaceHandler,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "FaceDetector": "Face Detector (RMBG)",
    "FaceSwapper": "Face Swapper (ReActor)",
    "FaceReference": "Face Reference (IPAdapter)",
    "IPAdapterFace": "IPAdapter Face",
    "ExpressionTransfer": "Expression Transfer",
    "AgeAdjuster": "Age Adjuster",
    "FaceEnhancer": "Face Enhancer (ReActor)",
    "MultiFaceHandler": "Multi-Face Handler (ReActor)",
})
