# 🎀 Laura Image Studio (v0.8)

Professional-grade ComfyUI custom node suite optimized for **Wan 2.2**, **Flux.1**, **NVIDIA Cosmos**, and **LivePortrait**. Developed for high-end AI influencer and viral video production by **snrtherock**.

---

## 🚀 Key Features

### 🧠 Intelligence & VRAM Management
- **Universal Model Loader**: Auto-detects architecture (Flux, Wan, SDXL, etc.) and maps optimal precision (FP8/BF16) and attention backends based on your VRAM tier.
- **Laura VRAM Master Cleaner**: Integrated "Hard Unload" and GC collection nodes to prevent OOM when switching between massive DiT models (e.g., Wan to SUPIR).
- **Model Health Check**: Instantly verify if SOTA 2026 models are correctly installed and receive hardware-specific optimization advice.

### 🎬 Advanced Video Suite
- **Wan 2.2 / CogVideoX Directed Motion**: Generate cinematic video from images with precise motion buckets and strength control.
- **Cinema Upscale (SUPIR + RIFE)**: High-end 4k/8k upscaling with motion interpolation for 60fps results.
- **SOTA Face Drive**: Advanced LivePortrait implementation for perfect character consistency.

### 🎨 Character & Style
- **Multi-LoRA Stack**: Optimized logic for combining Character, Style, and Lighting LoRAs without artifacts.
- **Influencer Presets**: Specialized loaders for Laura and Zoriana identity models.
- **Identity Injector**: Automatically injects character trigger words into prompts based on the model type.

---

## 🛠️ Installation

1.  **ComfyUI-Manager (Recommended)**:
    - Search for `Laura Image Studio` in the node list and click Install.
2.  **Manual**:
    - Clone this repo to `custom_nodes/Laura_Image_Studio`.
    - Run `pip install -r requirements.txt`.

---

## 📦 Included Nodes

| Category | Node Name | Functionality |
| :--- | :--- | :--- |
| **Utility** | `ModelHealthCheck` | Verifies model installation and provides VRAM optimization advice. |
| **Utility** | `LauraVRAMCleaner` | Performs deep VRAM cleaning and garbage collection. |
| **Loader** | `LauraUniversalLoader` | Smart loader for Flux, Wan, and SDXL models with auto-precision. |
| **Video** | `WanVideoGenerator` | Specialized generator for Wan 2.2 with motion bucket support. |
| **Video** | `CogVideoXGenerator` | Advanced CogVideoX support for high-quality video. |
| **Upscaling** | `Upscale2K/4K/8K` | High-fidelity upscaling nodes with tiling support. |
| **Face** | `LauraFaceDrive` | Character consistency and facial animation using LivePortrait. |
| **Identity** | `IdentityInjector` | Automates identity triggers for influencer workflows. |

---

## 💎 Premium Production Workflows

While the nodes are open-source, we provide **expertly tuned production workflows** for our supporters. These workflows are pre-configured with the ideal settings for professional AI influencer content.

👉 **Get Premium Workflows on [Patreon / Buy Me a Coffee] (Links coming soon!)**

---

## ⚖️ License
MIT License - Developed by **snrtherock**. Identity: `[snrtherock/Laura Studio]`
