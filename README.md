# Laura Image Studio

**Professional-grade ComfyUI custom node suite for AI image generation, virtual dressing, video production, and VRAM optimization.**

107 custom nodes | 15 modules | 30+ model support | 2GB to 80GB+ VRAM

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom_Nodes-blue)](https://github.com/comfyanonymous/ComfyUI)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Version](https://img.shields.io/badge/Version-0.8.1-orange)](https://github.com/snrtherock/Laura-Image-Studio/releases)

---

## Features

### Model Support (30+ Models)
- **Image**: SDXL, SD 1.5/2.1/3.0/3.5, FLUX.1/2, HunyuanDiT, Kolors, Pixart Sigma, Playground v2.5, Lumina, AuraFlow
- **Video**: Wan 2.1/2.2 (1.3B/14B), CogVideoX (2B/5B), Cosmos-Predict2.5, LivePortrait v2, AnimateDiff
- **Upscale**: 4X-UltraSharp, RealESRGAN, ScuNet, SUPIR-Video, RIFE v4
- **Face**: inswapper_128, IPAdapter FaceID, CodeFormer, GFPGAN

### Automatic VRAM Optimization
Detects your GPU VRAM and auto-configures quantization, resolution scaling, and CPU offloading. Works on any hardware from 2GB to 80GB+ VRAM.

| VRAM | Tier | Models Available |
|------|------|-----------------|
| 2-4GB | Ultra Low | SD 1.5, Pixart |
| 4-6GB | Low | SD 3.5 Medium |
| 6-8GB | Medium | SDXL, Kolors |
| 8-12GB | High | FLUX, Wan 1.3B |
| 12-16GB | Very High | Wan 2.1 14B |
| 16-24GB | Ultra | All models |
| 24-80GB | Extreme | Wan 2.2, CogVideoX-5B |
| 80GB+ | HPC | HunyuanImage-3.0, all models full precision |

### Virtual Dressing Room
10-slot clothing system with IPAdapter-based style transfer:
Hat, Sunglasses, Top, Dress, Skirt, Pants, Belt, Scarf, Bag, Shoes

### Video Production Suite
- **Wan 2.2 Directed Motion**: Prompt-based motion control for I2V
- **Cinema Upscale (SUPIR+RIFE)**: Combined 4K/8K upscale + 60fps interpolation in one node
- **Video Face Drive**: LivePortrait v2 temporal-mesh face animation
- **CogVideoX & Cosmos**: Full support for latest video architectures

### Professional Backgrounds
20 curated background presets with AI generation, pro lighting (8 presets), portrait bokeh, and seamless compositing.

### Crash Recovery
Auto-checkpoint system saves pipeline state at each stage. Resume from any checkpoint after crashes without losing progress.

---

## Node Categories (107 Nodes)

| Category | Nodes | Key Features |
|----------|-------|-------------|
| **Models & Loading** | 12 | Universal loader, Multi-LoRA stack, ControlNet, auto-detection |
| **Generation** | 6 | SDXL generator, prompt builder, negative presets, seed control |
| **Video** | 8 | Image-to-video, video-to-video, frame interpolation, saving |
| **Video Advanced** | 11 | Wan 2.2, CogVideoX, Cosmos, HunyuanDiT, VRAM cleaner |
| **Toggle** | 9 | Type switches (Image/Latent/Model/CLIP/VAE/Mask/Conditioning), pipeline toggle, master panel |
| **Upscaling** | 9 | 2K/4K/8K, detail enhance, cinema upscale, resolution constrainer |
| **Face** | 9 | Detect, swap, enhance, expression transfer, age adjust, face drive |
| **Dressing** | 10 | Clothing segmentor, virtual dresser, hair stylist, makeup artist |
| **Inpainting** | 7 | SAM2 smart masking, inpaint, outpaint, object removal |
| **Background** | 7 | Remove, replace, generate, bokeh, pro lighting, colorize |
| **Quantization** | 5 | VRAM detection, quantization, resolution scaling, offloading |
| **Checkpoint** | 5 | Auto-save, resume, manage checkpoints |
| **Batch Processing** | 4 | Image queue, prompt list, batch iterator |
| **Tile Processing** | 3 | Tile split, inpaint, merge with custom TILE_DATA |
| **Comparison** | 2 | Multi-model grid, 20 professional background presets |

---

## Installation

### Via ComfyUI-Manager (Recommended)
1. Open ComfyUI-Manager
2. Search for **Laura Image Studio**
3. Click **Install**
4. Restart ComfyUI

### Manual Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/snrtherock/Laura-Image-Studio.git Laura_Image_Studio
```

Dependencies install automatically on first launch. The auto-installer detects your Python environment (Portable, Conda, or System) and uses the correct pip.

### Manual Dependency Install (if auto-install fails)
```bash
cd ComfyUI/custom_nodes/Laura_Image_Studio
pip install -r requirements.txt
```

---

## Required Models

### Essential (Base Pipeline)
| Model | Location | Download |
|-------|----------|----------|
| Any SDXL checkpoint | `models/checkpoints/` | [CivitAI](https://civitai.com) / [HuggingFace](https://huggingface.co) |
| inswapper_128.onnx | `models/insightface/` | [GitHub](https://github.com/facefusion/facefusion-assets) |
| IPAdapter FaceID Plus | `models/ipadapter/` | [HuggingFace](https://huggingface.co/h94/IP-Adapter-FaceID) |
| RMBG-2.0 | `models/rmbg/` | [HuggingFace](https://huggingface.co/briaai/RMBG-2.0) |
| 4X-UltraSharp | `models/upscale_models/` | [OpenModelDB](https://openmodeldb.info) |

### Video (Optional)
| Model | VRAM | Download |
|-------|------|----------|
| Wan 2.2 T2V-14B | 14GB+ | [HuggingFace](https://huggingface.co/Wan-AI/Wan2.2-T2V-14B) |
| Wan 2.2 I2V-14B | 14GB+ | [HuggingFace](https://huggingface.co/Wan-AI/Wan2.2-I2V-14B-720P) |
| CogVideoX-5B-I2V | 16GB+ | [HuggingFace](https://huggingface.co/THUDM/CogVideoX-5b) |
| LivePortrait | 4GB+ | [HuggingFace](https://huggingface.co/Kijai/LivePortrait_safetensors) |
| SUPIR v0 | 10GB+ | [HuggingFace](https://huggingface.co/Kijai/SUPIR_safetensors) |

---

## Required Community Nodes

Laura Image Studio delegates certain operations to established community nodes:

| Package | Purpose |
|---------|---------|
| [ComfyUI-ReActor](https://github.com/Gourieff/comfyui-reactor-node) | Face swap engine |
| [ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus) | Face/clothing embedding |
| [ComfyUI-RMBG](https://github.com/bdsqlsz/ComfyUI-RMBG) | Background removal, segmentation |
| [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) | Package management (recommended) |

---

## Premium Workflows

While the custom nodes are fully open-source, we offer **expertly tuned production workflows** for supporters. These are pre-configured for professional AI influencer and viral video content.

**Get Premium Workflows:** [Patreon / Buy Me a Coffee] (Links coming soon!)

Three editions available:
- **Community Edition** - Uses only community nodes
- **Studio Edition** - Uses only Laura Image Studio nodes
- **Hybrid Edition** - Best of both worlds (recommended)

---

## Hardware Requirements

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| GPU VRAM | 4GB | 8-12GB | 16-24GB |
| RAM | 8GB | 16GB | 32GB |
| Storage | 10GB | 50GB | 100GB+ |
| GPU | GTX 1060 | RTX 3060/4060 | RTX 4070 Ti / 4090 |

The auto VRAM detection system adapts to any GPU. Lower VRAM GPUs use quantized models and CPU offloading automatically.

---

## Project Structure

```
Laura_Image_Studio/
  __init__.py              # Entry point with auto-dependency installer
  requirements.txt         # Python dependencies
  model-list.json          # Model registry for ComfyUI-Manager
  node_config.json         # Node metadata for ComfyUI-Manager
  nodes/
    __init__.py
    models.py              # 12 nodes - Model loading, LoRA, ControlNet
    generation.py          #  6 nodes - Image generation, prompts, seeds
    video.py               #  8 nodes - Basic video (Wan, AnimateDiff)
    video_advanced.py      # 11 nodes - CogVideoX, Cosmos, Wan 2.2
    toggle.py              #  9 nodes - Type switches, pipeline control
    upscaling.py           #  9 nodes - 2K/4K/8K, cinema upscale
    face.py                #  9 nodes - Detection, swap, enhance, drive
    dressing.py            # 10 nodes - Clothing, accessories, styling
    inpainting.py          #  7 nodes - SAM2 masking, inpaint, outpaint
    background.py          #  7 nodes - Removal, replacement, lighting
    quantization.py        #  5 nodes - VRAM detection, optimization
    checkpoint.py          #  5 nodes - Save/load/resume pipeline
    batch_processing.py    #  4 nodes - Image queues, prompt lists
    tile_processing.py     #  3 nodes - Tile split/inpaint/merge
    comparison.py          #  2 nodes - Multi-model grid, BG presets
```

---

## Changelog

### v0.8.0 - Viral Video Edition (2026-03-03)
- Added Wan 2.2 support (T2V/I2V MoE models)
- Added CogVideoX 2B/5B/5B-I2V support
- Added NVIDIA Cosmos-Predict2.5 support
- Added HunyuanDiT v1.2 support
- Added LauraVideoFaceDrive (LivePortrait v2 temporal-mesh)
- Added LauraVideoCinemaUpscale (SUPIR+RIFE combined)
- Added LauraWanDirectedVideo (motion-prompt control)
- Added VRAM auto-detection system (8 tiers)
- Added crash recovery checkpoint system
- Added tile processing with custom TILE_DATA type
- Added batch processing (image queue + prompt list)
- Added professional background library (20 presets)
- Added multi-model comparison grid
- Added LauraVRAMCleaner (Soft/Hard/Extreme modes)
- Added WorkflowTogglePanel (13 boolean outputs)
- Total: 107 nodes across 15 modules

### v0.5.0 - Foundation
- Initial release with 82 nodes
- SDXL, SD 1.5/2.1, FLUX.1 support
- Virtual dressing room (10 slots)
- Face swap and enhancement
- Background removal and replacement
- Basic upscaling (2K/4K/8K)
- Basic video (Wan 2.1, AnimateDiff)

---

## Contributing

Contributions are welcome! Please open an issue or pull request.

## License

MIT License - Created by **snrtherock**

## Support

- GitHub Issues: [Report a bug](https://github.com/snrtherock/Laura-Image-Studio/issues)
- Premium Workflows: [Patreon / Buy Me a Coffee] (Links coming soon!)
