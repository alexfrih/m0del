# AI Image Generation - Complete Toolkit

Generate AI images locally with state-of-the-art 2025 models. One simple command, choose your model, enter a prompt - done!

## Quick Start

```bash
# Install dependencies
pip install torch diffusers transformers accelerate protobuf sentencepiece

# Run interactive generator
python generate.py

# Select model, enter prompt, generate!
```

That's it!

## What This Does

**Problem:** Random pixels are cheap to generate, but meaningful images require 1,000,000× more computation.

**Solution:** Use efficient 2025 models that generate high-quality images in 1-4 steps instead of 20-50.

**Proof:** Includes theoretical analysis showing FLOPs costs from random noise (3 MFLOPs) to diffusion (5 PFLOPs).

## Available Models

### 🚀 Fast (CPU-Friendly)
- **SDXL Turbo** - 1 step, 20-40s, ⭐⭐⭐⭐ quality
- **SD 3 Medium** - 20 steps, 30-50s, ⭐⭐⭐⭐ quality
- **Waifu Diffusion** - 20 steps, 15-30s, ⭐⭐⭐ quality (anime)

### 🎨 High Quality
- **FLUX-schnell** - 4 steps, 60-90s, ⭐⭐⭐⭐⭐ photorealism
- **SD 3.5 Large** - 28 steps, 40-70s, ⭐⭐⭐⭐⭐ quality
- **Qwen-Image** - 28 steps, 120-180s, ⭐⭐⭐⭐⭐ text rendering

### 💾 Quantized (Large Models, Less RAM)
- **Qwen-Image 4-bit** - Needs 12-16GB RAM, best text
- **FLUX-schnell 4-bit** - Needs 8GB RAM, photorealism
- **SD 3.5 Large 4-bit** - Needs 8GB RAM, great quality

### 🔞 NSFW-Capable (Uncensored)
- **Realistic Vision** - Photorealistic uncensored
- **DreamShaper** - Versatile uncensored
- **Anything V5** - Anime uncensored
- **Deliberate** - Artistic uncensored

## Usage

### Interactive Mode (Recommended)

```bash
python generate.py
```

You'll see:
```
[1] SDXL Turbo          ⭐⭐⭐⭐    20-40s    Fast + great quality
[2] SD 3 Medium         ⭐⭐⭐⭐    30-50s    Good text rendering
[3] Waifu Diffusion     ⭐⭐⭐      15-30s    Fastest, anime
[4] FLUX-schnell        ⭐⭐⭐⭐⭐  60-90s    Best photorealism
...
[10] Realistic Vision   ⭐⭐⭐⭐    40-60s    🔞 Photorealistic NSFW
[12] Anything V5        ⭐⭐⭐⭐    40-60s    🔞 Anime NSFW

Select model [1-13]: 1
Enter prompt: a beautiful sunset over mountains
Steps: [press Enter]

✓ Image saved to output/
```

### Command Line Mode

```bash
# Fast generation
python generate_2025_models.py "a cat" --model sdxl-turbo

# Best quality
python generate_2025_models.py "portrait" --model flux-schnell

# Best text rendering
python generate_2025_models.py "poster with text 'HELLO'" --model qwen-image

# NSFW
python generate_2025_models.py "your prompt" --model realistic-vision-nsfw

# Quantized (less RAM)
python generate_quantized.py "your prompt" --model qwen-image --quantize 4bit
```

## Understanding Steps

**Steps = number of refinement iterations**

| Steps | Quality | Time | Use Case |
|-------|---------|------|----------|
| 1-4 | ⭐⭐⭐ | Very fast | Quick previews, fast models |
| 10-20 | ⭐⭐⭐⭐ | Fast | Testing prompts |
| 20-30 | ⭐⭐⭐⭐ | Medium | **Most use cases** ← Start here |
| 40-50 | ⭐⭐⭐⭐⭐ | Slow | Final high-quality images |

**Key Rules:**
- **Fast models (Turbo, Lightning):** Use 1-4 steps (more is WORSE!)
- **Standard models:** Use 20-30 steps
- **Quality models:** Use 28-40 steps
- **When in doubt:** Press Enter for defaults!

## Project Structure

```
image-generation/
├── README.md                    # You are here
├── requirements.txt             # Dependencies
│
├── Scripts (User-facing)
│   ├── generate.py              # Interactive mode ⭐ START HERE
│   ├── generate_2025_models.py  # Command line with 2025 models
│   ├── generate_quantized.py    # Quantized models (less RAM)
│   └── add_uncensored_models.py # NSFW model examples
│
├── Demos (Educational)
│   ├── pixel_map_efficiency.py  # FLOPs calculations
│   ├── pixel_map_visualizer.py  # Generate simple patterns
│   └── why_iteration_costs.py   # Show iteration costs
│
└── output/                      # Generated images (not tracked)
    ├── generated/               # From generate.py
    ├── 2025_models/             # From generate_2025_models.py
    └── quantized/               # From generate_quantized.py
```

## Model Details

### SDXL Turbo (Recommended for CPU)
- **Quality:** ⭐⭐⭐⭐
- **Steps:** 1 (yes, just ONE!)
- **Time:** 20-40s CPU, 0.3s GPU
- **Size:** 7GB
- **Best for:** Quick iterations, real-time generation
- **Why it's fast:** Distilled from SDXL, trained for 1-step

### FLUX-schnell (Best Quality)
- **Quality:** ⭐⭐⭐⭐⭐ SOTA photorealism
- **Steps:** 4
- **Time:** 60-90s CPU, 1-2s GPU
- **Size:** 23GB
- **Best for:** Professional photorealistic images
- **Released:** Aug 2024 by Black Forest Labs

### Qwen-Image (Best Text Rendering)
- **Quality:** ⭐⭐⭐⭐⭐ SOTA text
- **Steps:** 28
- **Time:** 120-180s CPU (80-120s with 4-bit), 3-5s GPU
- **Size:** 40GB (10GB with 4-bit quantization)
- **Best for:** Posters, signs, multilingual text
- **Released:** Aug 2025 by Alibaba

### Realistic Vision (Best NSFW)
- **Quality:** ⭐⭐⭐⭐
- **Steps:** 20-30
- **Time:** 40-60s CPU, 2s GPU
- **Size:** 4GB
- **Best for:** Photorealistic uncensored content
- **Note:** Community fine-tune of SD 1.5

## Performance on Different Hardware

### CPU (Your Setup)
| Model | Time | Quality |
|-------|------|---------|
| Waifu Diffusion | 15-30s | ⭐⭐⭐ |
| SDXL Turbo | 20-40s | ⭐⭐⭐⭐ ← Best CPU choice |
| SD 3 Medium | 30-50s | ⭐⭐⭐⭐ |
| FLUX-schnell | 60-90s | ⭐⭐⭐⭐⭐ |
| Qwen-Image | 120-180s | ⭐⭐⭐⭐⭐ (text) |

### GPU (RTX 3060)
| Model | Time |
|-------|------|
| SDXL Turbo | 0.3s |
| FLUX-schnell | 1-2s |
| Qwen-Image | 3-5s |

**100× faster on GPU!**

## Installation

### Minimal (Demos Only)
```bash
pip install numpy pillow
python pixel_map_visualizer.py  # Generate patterns
python pixel_map_efficiency.py  # Show FLOPs calculations
```

### Full (Real AI Image Generation)
```bash
pip install torch diffusers transformers accelerate protobuf sentencepiece
python generate.py  # Interactive generator
```

### Quantization Support (Optional)
```bash
pip install bitsandbytes
python generate_quantized.py  # Use quantized models
```

## Quantization (Run Large Models on Limited RAM)

**Problem:** FLUX (23GB) and Qwen (40GB) don't fit in RAM

**Solution:** Quantization reduces size with minimal quality loss

| Model | Full Size | 8-bit | 4-bit |
|-------|-----------|-------|-------|
| Qwen-Image | 40GB | 20GB | **10GB** ✓ |
| FLUX-schnell | 23GB | 12GB | **6GB** ✓ |
| SD 3.5 Large | 16GB | 8GB | **4GB** ✓ |

**Usage:**
```bash
pip install bitsandbytes

# Qwen-Image 4-bit (fits in 12-16GB RAM)
python generate_quantized.py "poster with text" --model qwen-image --quantize 4bit

# FLUX 4-bit (fits in 8GB RAM)
python generate_quantized.py "portrait" --model flux-schnell --quantize 4bit
```

**Quality loss:** ~10% (90% quality at 25% size - great tradeoff!)

## NSFW Models

**All official models (SD, FLUX, Qwen) are censored** - trained to refuse NSFW prompts.

**Community uncensored models available:**

| Model | Style | Usage |
|-------|-------|-------|
| Realistic Vision | Photorealistic | `--model realistic-vision-nsfw` |
| DreamShaper | Versatile | `--model dreamshaper-nsfw` |
| Anything V5 | Anime | `--model anything-v5-nsfw` |
| Deliberate | Artistic | `--model deliberate-nsfw` |

**Safety checker:** Already disabled in all scripts

**Legal/Ethical:** Only fictional/artistic content, follow local laws, respect consent

## Theory & Education

This project includes educational demos proving the efficiency difference:

### pixel_map_efficiency.py
Shows FLOPs calculations:
- Random pixels: 3 MFLOPs
- 20-step diffusion: 5 PFLOPs
- **1,000,000× difference!**

```bash
python pixel_map_efficiency.py
```

### pixel_map_visualizer.py
Generates simple patterns showing "no semantic meaning":
- Random noise
- Smooth noise
- Gradients
- Checkerboard

All generated in <100ms, but none look like "a cat" - that's the expensive part!

```bash
python pixel_map_visualizer.py
```

### why_iteration_costs.py
Shows why 50 steps costs 50× more than 1 step with visual proof.

```bash
python why_iteration_costs.py
```

## Key Insights

### 1. Random Pixels Are Trivial
```python
img = np.random.randint(0, 256, (1024, 1024, 3))
# Time: ~10ms, FLOPs: ~3 million
```

### 2. Coherent Images Need Constraints
To generate "a photo of a cat":
- **Semantic:** Cat has fur, eyes, ears (learned from training)
- **Spatial:** Ear above eye, not on tail (long-range dependencies)
- **Photometric:** Consistent shadows (physics simulation)
- **Textual:** "red apple" → red pixels + round shape (cross-modal alignment)

### 3. Diffusion Models Iterate (Expensive)
```
x_T ← random noise
for t in T..1:
    x_{t-1} = Denoise(x_t, t, prompt)  # 20-50 iterations!
```

Each step: ~250 TFLOPs × 20 steps = **5 PFLOPs total**

### 4. 2025 Efficient Models Skip Iteration
**Old way (diffusion):**
- 20-50 steps
- 5 PFLOPs
- ~3-5 seconds

**New way (SDXL Turbo, FLUX-schnell):**
- 1-4 steps
- 50-250 TFLOPs
- ~0.5-2 seconds
- **100× less computation!**

## Troubleshooting

### "protobuf not found" or "sentencepiece not found"
```bash
pip install protobuf sentencepiece
```

### "Out of memory" / Process killed
Try quantized models:
```bash
pip install bitsandbytes
python generate_quantized.py "prompt" --model sd-3.5-large --quantize 4bit
```

Or use smaller models:
- SDXL Turbo (7GB)
- SD 3 Medium (4GB)
- Waifu Diffusion (2GB)

### "Generation is slow"
You're on CPU. Expected times:
- SDXL Turbo: 20-40s
- SD 3 Medium: 30-50s
- FLUX: 60-90s

For faster: Use GPU or cloud (Google Colab, Replicate.com)

### "Model download is slow"
First run downloads 4-40GB depending on model. Subsequent runs are instant (models cached in `~/.cache/huggingface/`).

## Recommendations

### First Time Users
```bash
python generate.py
Select: [1] SDXL Turbo
```
Fast, great quality, works on CPU.

### Need Text in Images
```bash
python generate_quantized.py "poster text" --model qwen-image --quantize 4bit
```
Best text rendering, reasonable RAM usage.

### Want Best Quality
```bash
python generate.py
Select: [4] FLUX-schnell
```
State-of-the-art photorealism.

### NSFW Content
```bash
python generate.py
Select: [10] Realistic Vision (photorealistic)
# or [12] Anything V5 (anime)
```

### Low RAM (8GB)
```bash
python generate.py
Select: [1] SDXL Turbo
# or [8] SD 3.5 Large (4-bit)
```

## Examples

### Generate a Portrait
```bash
python generate.py
Select: 4  # FLUX-schnell
Prompt: portrait of an elderly man, warm lighting, photorealistic
Steps: [press Enter]
```

### Generate a Poster with Text
```bash
python generate_quantized.py "vintage travel poster with text 'Paris 1920'" \
  --model qwen-image --quantize 4bit --steps 28
```

### Generate Anime Art
```bash
python generate.py
Select: 3  # Waifu Diffusion
Prompt: anime girl with blue hair and red eyes
Steps: [press Enter]
```

### Quick Test Multiple Prompts
```bash
python generate.py
Select: 1  # SDXL Turbo (fastest)
Prompt: a cat
Steps: [press Enter]

Generate another? y
Prompt: a dog
...
```

## Credits

- **Stable Diffusion:** Stability AI
- **FLUX:** Black Forest Labs
- **Qwen-Image:** Alibaba Qwen Team
- **SDXL Turbo/Lightning:** Stability AI / ByteDance
- **Community Models:** CivitAI creators
- **Documentation:** Generated with Claude Code

## License

MIT - Use freely for learning, research, or commercial projects.

**Model licenses:**
- SD 1.5: CreativeML Open RAIL-M
- SDXL: SDXL License
- FLUX: Apache 2.0
- Community models: Various (check model cards)

---

## Quick Reference

**Generate an image:**
```bash
python generate.py
```

**Use specific model:**
```bash
python generate_2025_models.py "prompt" --model sdxl-turbo
```

**Less RAM:**
```bash
python generate_quantized.py "prompt" --model qwen-image --quantize 4bit
```

**See theory:**
```bash
python pixel_map_efficiency.py
```

**Need help:** Check the interactive prompts or open an issue!

---

**TL;DR:** One command (`python generate.py`), pick a model, enter prompt, generate high-quality AI images locally. Includes theory, 13+ models, and NSFW support.
