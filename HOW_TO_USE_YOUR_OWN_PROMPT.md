# How to Use Your Own Prompt

## Quick Answer

To generate images from **your own text prompts** (like "a cat in a wizard hat"), you need to use **real AI models**.

### Two-Step Process:

### Step 1: Install AI Model Libraries

```bash
pip install torch torchvision diffusers transformers accelerate
```

This downloads ~2GB of software (PyTorch + Stable Diffusion libraries).

### Step 2: Run the Generator

**Interactive mode** (easiest - just type prompts):
```bash
python generate_from_prompt.py --interactive
```

**Command line mode** (one image):
```bash
python generate_from_prompt.py "your prompt here"
```

## Examples

### Example 1: Simple Prompt
```bash
python generate_from_prompt.py "a red sports car"
```

Output:
```
Using device: cuda
Loading model: sd-1.5...
✓ Model loaded: sd-1.5

Generating image...
  Prompt: 'a red sports car'
  Model: sd-1.5
  Steps: 20
  Size: 512×512
  Device: cuda

================================================================================
GENERATION COMPLETE
================================================================================
Time:              3.45 seconds
Est. FLOPs:        500.0 TFLOPs
Est. power draw:   ~320W (GPU)
Est. energy:       ~1104.0 Joules
================================================================================

✓ Saved: output/generated/20250117_143022_sd-1.5_s20_red_sports_car.png
```

### Example 2: Better Quality (More Steps)
```bash
python generate_from_prompt.py "a majestic mountain landscape at sunset" --steps 50
```

More steps = better quality, but slower.

### Example 3: Fast Generation (LCM Model)
```bash
python generate_from_prompt.py "a cyberpunk city at night" --model lcm --steps 4
```

This is the **efficient 2025 approach** - same quality in 4 steps vs 20!

### Example 4: Interactive Mode
```bash
python generate_from_prompt.py --interactive
```

Then just type prompts interactively:
```
Enter prompt: a magical forest with glowing mushrooms
Number of steps [20]: 30

Generating image...
[... generates image ...]

Enter prompt: a steampunk airship
Number of steps [20]: 20

[... generates another image ...]

Enter prompt: quit
Goodbye!
```

### Example 5: Compare Different Step Counts
```bash
python generate_from_prompt.py "a cute puppy" --compare
```

This generates the same image with 5, 20, and 50 steps and shows the performance difference.

## What You Get

Every generation creates two files in `output/generated/`:

1. **Image file** (PNG):
   - `20250117_143022_sd-1.5_s20_cute_puppy.png`

2. **Metadata file** (TXT):
   - `20250117_143022_sd-1.5_s20_cute_puppy.txt`
   - Contains: prompt, settings, timing, FLOPs, etc.

## GPU vs CPU

### With GPU (Recommended)
- **Fast:** 2-5 seconds per image
- **Works on:** NVIDIA cards with 4GB+ VRAM
- Example: RTX 3060, RTX 4060, RTX 3080, RTX 4090

### Without GPU (CPU Only)
- **Slow:** 5-10 minutes per image
- Works, but not recommended
- The script auto-detects and uses CPU if no GPU found

## Model Comparison

| Model | Speed | Quality | Best For | Steps |
|-------|-------|---------|----------|-------|
| **sd-1.5** | Medium | Good | General use | 20-50 |
| **sd-2.1** | Slower | Better | High quality | 20-50 |
| **lcm** | **Fast** | Good | **Speed** | **1-8** |

**Try LCM for the efficiency gains we talked about!**

## Common Prompts to Try

```bash
# Nature
python generate_from_prompt.py "a serene lake surrounded by mountains at dawn"
python generate_from_prompt.py "a close-up of a colorful butterfly on a flower"

# Fantasy
python generate_from_prompt.py "a dragon flying over a medieval castle"
python generate_from_prompt.py "a wizard's study filled with magical books and potions"

# Sci-fi
python generate_from_prompt.py "a futuristic cityscape with flying cars"
python generate_from_prompt.py "an astronaut exploring an alien planet"

# Animals
python generate_from_prompt.py "a photorealistic portrait of a lion"
python generate_from_prompt.py "a cute kitten playing with yarn"

# Abstract
python generate_from_prompt.py "swirling colors in abstract patterns"
python generate_from_prompt.py "geometric shapes in a minimalist composition"
```

## Seeing the Efficiency Difference

1. **Old approach (20 steps):**
   ```bash
   time python generate_from_prompt.py "a sunset" --steps 20
   # Takes ~3-5 seconds
   ```

2. **New efficient approach (LCM, 4 steps):**
   ```bash
   time python generate_from_prompt.py "a sunset" --model lcm --steps 4
   # Takes ~0.5-1 second
   ```

**5-10× faster with similar quality!**

This is exactly the efficiency improvement we demonstrated in theory with `pixel_map_efficiency.py`.

## Troubleshooting

### "No module named 'torch'"
→ Install dependencies: `pip install torch diffusers transformers accelerate`

### "CUDA out of memory"
→ Reduce image size: `--width 512 --height 512`

### "Very slow generation"
→ Check if using GPU: look for "Using device: cuda" in output
→ If it says "cpu", you need CUDA-enabled PyTorch

### "Model download taking forever"
→ First run downloads ~4GB, only happens once
→ Models are cached for future use

## Full Options

```bash
python generate_from_prompt.py --help
```

Key options:
- `--steps N` - Number of denoising steps (default: 20)
- `--model M` - Model choice: sd-1.5, sd-2.1, lcm
- `--width W` - Image width (default: 512)
- `--height H` - Image height (default: 512)
- `--seed N` - Random seed for reproducibility
- `--compare` - Compare different step counts
- `--interactive` - Interactive mode

## Connection to the Demonstrations

The other scripts in this repo (**`pixel_map_efficiency.py`**, **`pixel_map_visualizer.py`**, etc.) show the **theory** - they calculate FLOPs and generate simple patterns.

**`generate_from_prompt.py`** is where you **see it in practice** - actual AI image generation with real computational costs.

Compare:
- **Demos:** Show what "no semantic meaning" looks like (noise, gradients, patterns)
- **Real model:** Generates actual meaningful images from your prompts

That's the 1,000,000× FLOPs difference explained!

---

**Ready to start?**

```bash
# Install
pip install torch torchvision diffusers transformers accelerate

# Generate!
python generate_from_prompt.py --interactive
```

Then type any prompt you want!
