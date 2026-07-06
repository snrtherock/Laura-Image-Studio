# Workflow Node Audit Report

Generated: 2026-07-05 08:44:42
Laura Studio nodes available: 137
Workflows scanned: 8

## Summary

| Workflow | Laura | Built-in | Community |
|----------|------:|--------:|---------:|
| 00_health_check.json | 5 | 0 | 0 |
| 01_image_basic_beginner.json | 3 | 5 | 0 |
| 02_image_edit.json | 3 | 8 | 0 |
| 03_virtual_tryon.json | 4 | 1 | 0 |
| 04_image_to_video.json | 4 | 6 | 0 |
| 05_audio_voiceover.json | 5 | 2 | 0 |
| 06_enhanced_studio.json | 9 | 6 | 0 |
| 07_virtual_dressing.json | 11 | 3 | 0 |

## Details

### 00_health_check.json

**Laura** (5):
- `LauraHardwareProfiler`
- `LauraModelCatalog`
- `LauraModelRecommender`
- `LauraShowText`
- `LauraWorkflowHealthCheck`

### 01_image_basic_beginner.json

**Laura** (3):
- `LauraHardwareProfiler`
- `LauraModelRecommender`
- `LauraTerminalSave`

**Built-in** (5):
- `CLIPTextEncode`
- `CheckpointLoaderSimple`
- `EmptyLatentImage`
- `KSampler`
- `VAEDecode`

### 02_image_edit.json

**Laura** (3):
- `LauraHardwareProfiler`
- `LauraModelRecommender`
- `LauraTerminalSave`

**Built-in** (8):
- `CLIPLoader`
- `CLIPTextEncode`
- `KSampler`
- `LoadImage`
- `UNETLoader`
- `VAEDecode`
- `VAEEncode`
- `VAELoader`

### 03_virtual_tryon.json

**Laura** (4):
- `LauraHardwareProfiler`
- `LauraModelRecommender`
- `LauraStageSwitch`
- `LauraTerminalSave`

**Built-in** (1):
- `LoadImage`

### 04_image_to_video.json

**Laura** (4):
- `LauraHardwareProfiler`
- `LauraModelRecommender`
- `LauraStageSwitch`
- `LauraTerminalSave`

**Built-in** (6):
- `CLIPTextEncode`
- `CheckpointLoaderSimple`
- `EmptyLatentImage`
- `KSampler`
- `LoadImage`
- `VAEDecode`

### 05_audio_voiceover.json

**Laura** (5):
- `LauraHardwareProfiler`
- `LauraModelRecommender`
- `LauraShowText`
- `LauraStageSwitch`
- `LauraTerminalSave`

**Built-in** (2):
- `LoadImage`
- `Note`

### 06_enhanced_studio.json

**Laura** (9):
- `BackgroundRemover`
- `BackgroundReplacer`
- `FaceEnhancer`
- `FaceSwapper`
- `LauraHardwareProfiler`
- `LauraModelRecommender`
- `LauraShowText`
- `LauraStageSwitch`
- `LauraTerminalSave`

**Built-in** (6):
- `CLIPTextEncode`
- `CheckpointLoaderSimple`
- `EmptyLatentImage`
- `KSampler`
- `LoadImage`
- `VAEDecode`

### 07_virtual_dressing.json

**Laura** (11):
- `BackgroundRemover`
- `BackgroundReplacer`
- `ClothingSegmentor`
- `FaceEnhancer`
- `FaceSwapper`
- `LauraHardwareProfiler`
- `LauraModelRecommender`
- `LauraStageSwitch`
- `LauraTerminalSave`
- `OutfitCombinator`
- `VirtualDresser`

**Built-in** (3):
- `CheckpointLoaderSimple`
- `LoadImage`
- `Note`
