# Simple Start - One Command!

## The Easiest Way

Just run this:

```bash
python generate.py
```

That's it! Then follow the prompts.

## What Happens

### Step 1: Choose a Model

You'll see a menu:

```
AVAILABLE MODELS
================================================================================

🚀 FAST (CPU-FRIENDLY):
  [1] SDXL Turbo                ⭐⭐⭐⭐          20-40s
      Fast + great quality - BEST FOR CPU

  [2] SD 3 Medium               ⭐⭐⭐⭐          30-50s
      Good text rendering, balanced

  [3] Waifu Diffusion           ⭐⭐⭐            15-30s
      Fastest! Anime/manga style

🎨 HIGH QUALITY (SLOWER ON CPU):
  [4] FLUX-schnell              ⭐⭐⭐⭐⭐        60-90s
      Best photorealism (slow on CPU)

  [5] SD 3.5 Large              ⭐⭐⭐⭐⭐        40-70s
      Excellent typography (slow on CPU)

💾 QUANTIZED (LARGE MODELS, LESS RAM):
  [6] Qwen-Image (4-bit)        ⭐⭐⭐⭐⭐        80-120s
      Best text rendering! (needs 12-16GB RAM)

  [7] FLUX-schnell (4-bit)      ⭐⭐⭐⭐          45-65s
      Photorealism, less RAM (needs 8GB+)

  [8] SD 3.5 Large (4-bit)      ⭐⭐⭐⭐          30-50s
      Great quality, low RAM (needs 8GB)

📊 BASELINE:
  [9] SD 1.5 (Your Old Model)   ⭐⭐             50-60s
      For comparison only (outdated)

💡 RECOMMENDATIONS:
  • First time?           Try [1] SDXL Turbo
  • Need text in image?   Try [6] Qwen-Image or [2] SD 3 Medium
  • Want best quality?    Try [4] FLUX-schnell
  • Want fastest?         Try [3] Waifu Diffusion
  • Low RAM (8GB)?        Try [8] SD 3.5 Large (4-bit)

Select model [1-9, or 'q' to quit]:
```

**Just press 1** for the best CPU-friendly option!

### Step 2: Enter Your Prompt

```
Enter your prompt (or 'change' to switch model, 'quit' to exit): a pig in 3D blender style
```

Type anything you want!

### Step 3: Choose Steps (Optional)

```
Number of steps [default for this model, or enter number]:
```

Just press **Enter** for default, or type a number (more = better quality but slower).

### Step 4: Wait & Done!

The image generates and saves automatically!

```
✓ Image generated successfully!
  Check the output/ folder
```

### Step 5: Generate More (Optional)

```
Generate another image with this model? (y/n):
```

Press **y** to keep going, **n** to quit.

## Complete Example Session

```bash
$ python generate.py

AVAILABLE MODELS
...

Select model [1-9]: 1

✓ Selected: SDXL Turbo
  Quality: ⭐⭐⭐⭐
  Time: 20-40s

Enter your prompt: a cute pig

Number of steps [default]:

GENERATING WITH SDXL Turbo
Estimated time: 20-40s
...

✓ Image generated successfully!

Generate another image? (y/n): y

Enter your prompt: a magical forest

...

✓ Image generated successfully!

Generate another image? (y/n): n

Goodbye!
```

## Quick Commands

### Run Interactive Mode
```bash
python generate.py
```

### Switch Model Mid-Session
When prompted for a prompt, type:
```
change
```

Then select a new model!

### Quit Anytime
When prompted, type:
```
quit
```
or press **Ctrl+C**

## First-Time Recommendations

### Best Starting Point
```
Select model [1-9]: 1
```
**SDXL Turbo** - Fast, great quality, works on CPU

### If You Want Text
```
Select model [1-9]: 6
```
**Qwen-Image (4-bit)** - Best text rendering (needs 12-16GB RAM)

Or if that's too big:
```
Select model [1-9]: 2
```
**SD 3 Medium** - Good text, smaller (needs 8GB RAM)

### If You Want Speed
```
Select model [1-9]: 3
```
**Waifu Diffusion** - Fastest (15-30s), anime style

### If You Want Best Quality
```
Select model [1-9]: 4
```
**FLUX-schnell** - Photorealistic (60-90s on CPU)

## Troubleshooting

### "Script not found"
Make sure you're in the right directory:
```bash
cd /Users/alexandre/Code/experiments/image-generation
python generate.py
```

### "Out of memory" / Process killed
Try a quantized model:
- Select [8] for SD 3.5 Large (4-bit) - needs 8GB RAM
- Or select [7] for FLUX (4-bit) - needs 8GB+ RAM

### "bitsandbytes not found"
If you select a quantized model ([6], [7], [8]):
```bash
pip install bitsandbytes
```

### Want to compare to your old SD 1.5?
```
Select model [1-9]: 9
```
Then generate the same prompt you used before!

## Features

### Change Model Mid-Session
Don't like the current model? Just type `change` when prompted for a prompt.

### Multiple Images in One Session
Keep generating with the same model - no need to restart!

### Custom Steps
Want more quality? Enter a higher number when asked for steps (e.g., 30 or 50).

### Easy to Use
No complex command-line arguments - just follow the prompts!

## Summary

**One command does everything:**

```bash
python generate.py
```

1. Shows you all models
2. Let's you pick one
3. Asks for your prompt
4. Generates the image
5. Saves it automatically
6. Let's you make more!

**No need to remember model names, flags, or arguments!**

---

Ready to try it?

```bash
python generate.py
```

Pick option **[1]** for the best CPU experience!
