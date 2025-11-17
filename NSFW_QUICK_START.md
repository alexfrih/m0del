# NSFW Models - Quick Start

## ✅ NSFW Models Added!

I've added **4 uncensored community models** to your scripts, all marked with 🔞.

## Fix Missing Dependencies First

You got an error about protobuf and sentencepiece. Fix it:

```bash
pip install protobuf sentencepiece
```

Or reinstall requirements:

```bash
pip install -r requirements.txt
```

## Available NSFW Models

### In `generate.py` (Interactive Script)

Run:
```bash
python generate.py
```

You'll see a new section:

```
🔞 NSFW-CAPABLE (UNCENSORED COMMUNITY MODELS):
  [10] Realistic Vision v5.1 [NSFW]  ⭐⭐⭐⭐         40-60s
       Photorealistic, uncensored community model

  [11] DreamShaper [NSFW]            ⭐⭐⭐⭐         40-60s
       Versatile realistic, uncensored

  [12] Anything V5 (Anime) [NSFW]    ⭐⭐⭐⭐         40-60s
       Anime specialist, uncensored

  [13] Deliberate [NSFW]             ⭐⭐⭐⭐         40-60s
       Artistic/painted style, uncensored

Select model [1-13]:
```

**Just pick 10, 11, 12, or 13!**

### In `generate_2025_models.py` (Command Line)

Run directly:

```bash
# Photorealistic NSFW
python generate_2025_models.py "your prompt" --model realistic-vision-nsfw

# Versatile NSFW
python generate_2025_models.py "your prompt" --model dreamshaper-nsfw

# Anime NSFW
python generate_2025_models.py "your prompt" --model anything-v5-nsfw

# Artistic NSFW
python generate_2025_models.py "your prompt" --model deliberate-nsfw
```

## Model Details

### [10] Realistic Vision v5.1 [NSFW]

**Best for:** Photorealistic uncensored content
**Model ID:** `SG161222/Realistic_Vision_V5.1_noVAE`
**Size:** ~4GB
**CPU Time:** 40-60 seconds
**Quality:** ⭐⭐⭐⭐
**Optimal Steps:** 20-30

**Example:**
```bash
python generate.py
Select: 10
Prompt: your photorealistic prompt
Steps: [press Enter for default]
```

---

### [11] DreamShaper [NSFW]

**Best for:** Versatile uncensored content (works for various styles)
**Model ID:** `Lykon/DreamShaper`
**Size:** ~4GB
**CPU Time:** 40-60 seconds
**Quality:** ⭐⭐⭐⭐
**Optimal Steps:** 20-30

**Good for:** General purpose, realistic and semi-realistic

---

### [12] Anything V5 (Anime) [NSFW]

**Best for:** Anime/manga uncensored content
**Model ID:** `stablediffusionapi/anything-v5`
**Size:** ~4GB
**CPU Time:** 40-60 seconds
**Quality:** ⭐⭐⭐⭐
**Optimal Steps:** 20-28

**Training:** Trained on Danbooru (anime art database)
**Style:** Clean anime aesthetic

---

### [13] Deliberate [NSFW]

**Best for:** Artistic/painted uncensored content
**Model ID:** `XpucT/Deliberate`
**Size:** ~4GB
**CPU Time:** 50-70 seconds
**Quality:** ⭐⭐⭐⭐
**Optimal Steps:** 30-40

**Style:** Painterly, artistic look

---

## Quick Examples

### Interactive Mode (Easiest)

```bash
python generate.py

# You'll see the menu
Select model [1-13]: 10  # Realistic Vision

Enter prompt: your NSFW prompt here
Steps: [press Enter]

# Done! Image saved to output/
```

### Command Line Mode

```bash
# Photorealistic
python generate_2025_models.py "portrait" --model realistic-vision-nsfw

# Anime
python generate_2025_models.py "anime character" --model anything-v5-nsfw --steps 25

# Artistic
python generate_2025_models.py "fantasy art" --model deliberate-nsfw --steps 35
```

## First Run

**Models will download on first use (~4GB each):**

```
Loading model: realistic-vision-nsfw
Downloading model (first run only)...
[Downloads ~4GB]
✓ Model loaded successfully!
```

**Subsequent runs:** Instant (models are cached)

## Comparison to Your Current Models

| Your Current Models | NSFW Capability | Notes |
|---------------------|-----------------|-------|
| SD 1.5, SDXL, SD 3.x | ❌ Censored | Trained to refuse NSFW |
| FLUX, Qwen-Image | ❌ Censored | Corporate/filtered |
| **Realistic Vision** | ✅ **Uncensored** | **Community fine-tune** |
| **DreamShaper** | ✅ **Uncensored** | **Community fine-tune** |
| **Anything V5** | ✅ **Uncensored** | **Community fine-tune** |
| **Deliberate** | ✅ **Uncensored** | **Community fine-tune** |

## Safety Checker Status

**Already disabled** in all your scripts! (`safety_checker=None`)

The difference is:
- Your old models: Safety checker OFF, but **trained to refuse** NSFW
- New NSFW models: Safety checker OFF, **trained on unfiltered data** ✓

## Recommendations

### Photorealistic → Use [10] Realistic Vision
```bash
python generate.py
Select: 10
```

Most popular photorealistic uncensored model.

### Anime/Manga → Use [12] Anything V5
```bash
python generate.py
Select: 12
```

Best anime uncensored model.

### Versatile → Use [11] DreamShaper
```bash
python generate.py
Select: 11
```

Works for various styles.

### Artistic/Painted → Use [13] Deliberate
```bash
python generate.py
Select: 13
```

Painterly aesthetic.

## Important Notes

⚠️ **Legal/Ethical:**
- All content must be fictional/artistic
- Never depict real people without consent
- Check your local laws on AI-generated content
- Respect platform ToS when sharing

📝 **Technical:**
- Safety checker already disabled in your scripts
- These are fine-tuned versions of SD 1.5
- Legal under CreativeML Open RAIL-M license
- Community-maintained, not official Stability AI

## Troubleshooting

### "protobuf not found" or "sentencepiece not found"

```bash
pip install protobuf sentencepiece
```

### Model download is slow

First run downloads ~4GB per model. Be patient!

### Out of memory

Models are ~4GB each. Close other applications or try a smaller resolution:

```bash
python generate_2025_models.py "prompt" --model realistic-vision-nsfw --width 512 --height 512
```

### Want more NSFW models?

Check:
- **CivitAI:** https://civitai.com (thousands of models)
- **Hugging Face:** Search "uncensored" or model names
- See **MODEL_CONTENT_POLICIES.md** for full list

## Summary

✅ **4 NSFW models added**
✅ **Clearly marked with 🔞**
✅ **Available in both scripts**
✅ **Ready to use now!**

**Start with:**
```bash
python generate.py
Select: 10  # Realistic Vision
```

Or:
```bash
python generate_2025_models.py "your prompt" --model realistic-vision-nsfw
```

Enjoy!
