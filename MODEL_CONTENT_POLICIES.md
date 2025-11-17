# Model Content Policies & NSFW Capabilities

## Overview

Different models have different content policies. Here's the technical breakdown:

## Current Models in Your Scripts

**All models in `generate_2025_models.py` have safety checkers DISABLED** (`safety_checker=None`)

However, **disabling the safety checker ≠ NSFW capable!** Here's why:

### Two Types of Filtering

1. **Post-Generation Filter (Safety Checker)**
   - Checks image AFTER generation
   - Blacks out NSFW content
   - **Status in your scripts:** DISABLED ✓

2. **Training-Level Censorship**
   - Model was trained to REFUSE NSFW prompts
   - Baked into the model weights
   - **Can't be disabled without retraining**

## Model Capabilities

### ❌ NSFW-Restricted (Trained to Refuse)

These models have built-in refusal, even with safety checker disabled:

| Model | Policy | What Happens |
|-------|--------|--------------|
| **SD 1.5** | LAION-5B filtered | Distorts/refuses explicit prompts |
| **SD 2.x** | Heavily censored | Trained to refuse NSFW |
| **SDXL** | Filtered training | Refuses or produces poor quality |
| **SD 3.x** | Filtered | Refuses NSFW prompts |
| **FLUX (all)** | Clean training data | Won't generate NSFW well |
| **Qwen-Image** | Corporate/filtered | Refuses NSFW |

**Your safety checker is OFF, but these models were trained on filtered data.**

### ⚠️ Partially Capable

| Model | Notes |
|-------|-------|
| **SD 1.5** | Original can do mild NSFW, later versions more restricted |
| **Waifu Diffusion** | Anime-focused, some NSFW capability (trained on Danbooru) |

### ✅ NSFW-Capable (Community Models)

These are **community fine-tunes** of base models, trained on unfiltered data:

#### Realistic/Photorealistic

| Model | Hugging Face ID | Style | Notes |
|-------|-----------------|-------|-------|
| **Realistic Vision v5.1** | `SG161222/Realistic_Vision_V5.1_noVAE` | Photorealistic | Very popular on CivitAI |
| **DreamShaper** | `Lykon/DreamShaper` | Versatile realistic | Good balance |
| **Deliberate** | `XpucT/Deliberate` | Artistic/Painted | Painterly style |
| **epiCRealism** | `emilianJR/epiCRealism` | Ultra-realistic | Photographic quality |

#### Anime/Stylized

| Model | Hugging Face ID | Style | Notes |
|-------|-----------------|-------|-------|
| **Anything V5** | `stablediffusionapi/anything-v5` | Anime | Popular anime model |
| **AbyssOrangeMix** | `WarriorMama777/OrangeMixs` | Anime | High-quality anime |
| **Counterfeit** | `gsdf/Counterfeit-V3.0` | Anime | Clean anime style |

#### Artistic

| Model | Hugging Face ID | Style |
|-------|-----------------|-------|
| **OpenJourney** | `prompthero/openjourney` | Midjourney-like |
| **Protogen** | `darkstorm2150/Protogen_x3.4` | Sci-fi/fantasy |

## How to Use Uncensored Models

### Option 1: Use the Example Script

I created **`add_uncensored_models.py`** with 4 popular models ready to use:

```bash
python add_uncensored_models.py
```

Edit the script to uncomment and customize.

### Option 2: Add to Your Main Script

Add this to `generate_2025_models.py` model configs:

```python
"realistic-vision": {
    "model_id": "SG161222/Realistic_Vision_V5.1_noVAE",
    "pipeline": StableDiffusionPipeline,
    "quality": "⭐⭐⭐⭐",
    "size": "~4GB",
    "params": "1B",
    "steps": 20,
    "speed": "Medium",
    "cpu_time": "40-60s",
    "gpu_time": "2s",
    "note": "Photorealistic, uncensored community model",
    "width": 512,
    "height": 512,
    "released": "2023 (v5.1)",
    "best_for": "Photorealistic uncensored content",
},
```

### Option 3: Download from CivitAI

**CivitAI** (https://civitai.com) has thousands of uncensored models:

1. Find a model on CivitAI
2. Download the `.safetensors` file
3. Load with:

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_single_file(
    "path/to/model.safetensors",
    safety_checker=None,
)
```

## Popular NSFW Model Sources

### 1. Hugging Face
- Search: "realistic vision", "dreamshaper", "deliberate"
- Filter by "Unconditional Image Generation"
- Check model card for content policy

### 2. CivitAI (Largest Collection)
- https://civitai.com
- Thousands of NSFW-capable models
- Community ratings and examples
- Download .safetensors files

### 3. Model Categories on CivitAI

- **Realistic/Photorealistic:** Realistic Vision, epiCRealism
- **Anime:** Anything V5, AbyssOrangeMix, Counterfeit
- **2.5D (Semi-realistic):** DreamShaper, Rev Animated
- **Artistic:** Deliberate, Protogen

## Important Notes

### Legal & Ethical

⚠️ **Before using NSFW models:**

1. **Age Verification:** Ensure all depicted subjects are fictional/artistic
2. **Local Laws:** Check your jurisdiction's laws on AI-generated content
3. **Consent:** Never create images of real people without consent
4. **Distribution:** Be aware of platform ToS (Discord, Twitter, etc. ban NSFW AI)

### Technical Considerations

1. **No Post-Filter:** Safety checker is already disabled in your scripts
2. **Training Matters:** Model must be trained on uncensored data
3. **Prompting:** Some models still need explicit prompts
4. **Quality:** Uncensored ≠ high quality (check reviews)

### Why Official Models Are Censored

1. **Corporate Liability:** Stability AI, Black Forest Labs, etc. need to comply with laws
2. **Dataset Filtering:** LAION-5B removed NSFW content after 2022
3. **Public Funding:** Many models funded by grants with restrictions
4. **App Store Policies:** Need to work with Apple/Google ToS

### Community Models Are Fine-Tunes

**How they work:**
1. Start with SD 1.5 or SDXL base
2. Fine-tune on unfiltered datasets (Danbooru, etc.)
3. Share on Hugging Face or CivitAI

**Legal:** Fine-tuning is allowed under CreativeML Open RAIL-M license (SD 1.5) and SDXL license.

## Recommended Setup for NSFW

### Best Models by Use Case

**Photorealistic:**
```python
Model: Realistic Vision v5.1
Steps: 20-30
Size: 512×512 or 768×768
```

**Anime:**
```python
Model: Anything V5
Steps: 20-28
Size: 512×512
```

**Artistic/Painted:**
```python
Model: Deliberate
Steps: 30-40
Size: 512×768
```

### Example Workflow

```bash
# 1. Use add_uncensored_models.py
python add_uncensored_models.py

# 2. Or add to generate.py (edit the MODELS dict)

# 3. Or download from CivitAI and load manually
```

## Current Status of Your Scripts

**What's already set up:**
- ✅ Safety checker: DISABLED
- ✅ All infrastructure: Ready
- ❌ NSFW-capable models: NOT included (all are censored base models)

**What you need:**
- Add community fine-tuned models (see examples above)

## Quick Start: Add Realistic Vision

Edit `generate_2025_models.py`, add to `get_model_configs()`:

```python
"realistic-vision": {
    "model_id": "SG161222/Realistic_Vision_V5.1_noVAE",
    "pipeline": StableDiffusionPipeline,
    "quality": "⭐⭐⭐⭐",
    "size": "~4GB",
    "params": "1B",
    "steps": 20,
    "speed": "Medium",
    "cpu_time": "40-60s",
    "gpu_time": "2s",
    "note": "Photorealistic community model, uncensored",
    "width": 512,
    "height": 512,
    "released": "2023",
    "best_for": "Photorealistic content without restrictions",
},
```

Then use:
```bash
python generate_2025_models.py "your prompt" --model realistic-vision
```

## Resources

- **CivitAI:** https://civitai.com (biggest NSFW model collection)
- **Hugging Face:** https://huggingface.co (search "uncensored" or model names)
- **r/StableDiffusion:** Reddit community with model recommendations
- **Model reviews:** CivitAI has user ratings and sample images

## Summary

**Your current models:** All censored (trained on filtered data)

**For NSFW capability:** Use community fine-tuned models like:
- Realistic Vision (photorealistic)
- DreamShaper (versatile)
- Anything V5 (anime)

**How to add:** See `add_uncensored_models.py` example or edit `generate_2025_models.py`

**Legal:** Fine-tuned models are legal under open licenses, but follow local laws and ethical guidelines.
