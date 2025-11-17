# Usage Guide: Real Models vs Demonstrations

## What's the Difference?

### Demonstration Scripts (Don't Need GPU)

These show **concepts** and **calculations**, but don't generate actual AI art:

1. **`pixel_map_efficiency.py`** - Calculates theoretical FLOPs
   - Just math - shows you the numbers
   - Doesn't generate images from prompts
   - Runs anywhere (no GPU needed)

2. **`pixel_map_visualizer.py`** - Generates simple patterns
   - Makes noise, gradients, checkerboards
   - Shows what "no semantic meaning" looks like
   - Runs anywhere (no GPU needed)

3. **`why_iteration_costs.py`** - Simulates iteration cost
   - Uses simple blur operations to demonstrate concepts
   - Not real AI image generation
   - Runs anywhere (no GPU needed)

### Real Model Script (Needs GPU Recommended)

4. **`generate_from_prompt.py`** ✨ **NEW**
   - Uses **trained AI models** (Stable Diffusion, LCM)
   - Generates **actual photorealistic images** from text
   - Takes your prompt: "a cat wearing a hat" → actual cat image
   - Requires: ~4-8GB GPU VRAM (or slow on CPU)
   - Downloads models: ~4GB on first run

## Using Your Own Prompts

### Option 1: Quick Install (Just Demonstrations)

```bash
# Install minimal dependencies (just NumPy + Pillow)
pip install -r requirements.txt

# Run demonstrations
python pixel_map_visualizer.py
python pixel_map_efficiency.py
```

This shows you the **theory** but can't generate images from text prompts.

### Option 2: Full Install (Real AI Models)

```bash
# Install AI model dependencies (includes PyTorch, ~2GB download)
pip install torch torchvision diffusers transformers accelerate

# Now you can generate real images!
python generate_from_prompt.py --interactive
```

This lets you **enter prompts and generate actual images**.

## Examples with Real Models

### Interactive Mode (Easiest)

```bash
python generate_from_prompt.py --interactive
```

Then just type prompts:
```
Enter prompt: a photorealistic cat wearing a wizard hat
Number of steps [20]: 20

Generating image...
  Prompt: 'a photorealistic cat wearing a wizard hat'
  Model: sd-1.5
  Steps: 20
  Size: 512×512
  Device: cuda

Time: 3.45 seconds
Est. FLOPs: 500 TFLOPs
Est. energy: ~1500 Joules

✓ Saved: output/generated/20250117_143022_sd-1.5_s20_cat_wizard_hat.png
```

### Command Line Mode

```bash
# Basic usage
python generate_from_prompt.py "a red apple on a wooden table"

# More steps = better quality (but slower)
python generate_from_prompt.py "a sunset over mountains" --steps 50

# High resolution
python generate_from_prompt.py "a futuristic city" --width 768 --height 768

# Use faster 1-step model (LCM)
python generate_from_prompt.py "a dog in space" --model lcm --steps 4

# Reproducible results (same seed = same image)
python generate_from_prompt.py "a flower" --seed 42
```

### Compare Different Step Counts

```bash
python generate_from_prompt.py "a mountain landscape" --compare
```

This generates the same image with 5, 20, and 50 steps, showing:
- How quality improves with more steps
- How time increases linearly
- The FLOPs difference

Example output:
```
PERFORMANCE COMPARISON
================================================================================
Steps      Time            FLOPs                Speedup vs 50
--------------------------------------------------------------------------------
5          0.89 sec        125.0 TFLOPs         10.0×
20         3.45 sec        500.0 TFLOPs         2.6×
50         8.92 sec        1250.0 TFLOPs        1.0×
================================================================================
```

## Model Options

### `sd-1.5` (Stable Diffusion 1.5) - Default
- **Best for:** General use, good quality
- **Size:** ~4GB
- **Speed:** Medium (20 steps = ~3-5 seconds on RTX 3060)
- **Recommended steps:** 20-50

### `sd-2.1` (Stable Diffusion 2.1)
- **Best for:** Higher quality, better prompt following
- **Size:** ~5GB
- **Speed:** Slightly slower than 1.5
- **Recommended steps:** 20-50

### `lcm` (Latent Consistency Model)
- **Best for:** FAST generation (this is the "efficient" approach!)
- **Size:** ~4GB
- **Speed:** Very fast (4 steps = ~0.5 seconds)
- **Recommended steps:** 1-8 (optimized for low step counts)
- **Quality:** Near SD-1.5 quality in 4 steps vs 20!

Example:
```bash
# Old way: 20 steps, ~3 seconds
python generate_from_prompt.py "a cat" --steps 20

# New efficient way: 4 steps, ~0.5 seconds, similar quality!
python generate_from_prompt.py "a cat" --model lcm --steps 4
```

## Hardware Requirements

| Hardware | Can Run? | Speed | Notes |
|----------|----------|-------|-------|
| **CPU only** | ✓ Yes | Very slow (minutes) | Not recommended |
| **4GB GPU** | ✓ Yes | Slow | Use `--width 512 --height 512` |
| **8GB GPU** | ✓✓ Yes | Good | RTX 3060, RTX 4060 - recommended |
| **12GB+ GPU** | ✓✓✓ Yes | Fast | RTX 3080, 4080, 4090 - best |

## Performance on Different Hardware

### RTX 3060 (8GB) - Consumer GPU
```
512×512, 20 steps: ~3-5 seconds
768×768, 20 steps: ~8-12 seconds
```

### RTX 4090 (24GB) - High-end GPU
```
512×512, 20 steps: ~1-2 seconds
1024×1024, 20 steps: ~3-5 seconds
```

### CPU (No GPU)
```
512×512, 20 steps: ~5-10 minutes (not recommended!)
```

## What Gets Generated?

For each image, you get:

1. **PNG file** with the generated image
2. **TXT file** with metadata:
   ```
   GENERATION METADATA
   ============================================================

   prompt              : a red apple on a wooden table
   model               : sd-1.5
   steps               : 20
   width               : 512
   height              : 512
   device              : cuda
   time_sec            : 3.45
   estimated_tflops    : 500.0
   ```

## Connecting to the Efficiency Demonstrations

Now you can **see the difference yourself**:

1. Run the demo scripts first:
   ```bash
   python pixel_map_visualizer.py  # See "no semantic meaning" examples
   ```

2. Then generate a real image:
   ```bash
   python generate_from_prompt.py "a photorealistic cat"
   ```

3. Compare:
   - Demo scripts: <100ms, simple patterns, no meaning
   - Real model: 3+ seconds, actual cat image, semantic understanding

**That's the 1,000,000× FLOPs difference in action!**

## Common Issues

### "CUDA out of memory"
Solution: Reduce image size
```bash
python generate_from_prompt.py "prompt" --width 512 --height 512
```

### "Model download is slow"
First run downloads ~4GB. Subsequent runs are fast (models are cached).

### "Generation is very slow"
- Check you're using GPU: look for "Using device: cuda" in output
- If it says "Using device: cpu", install CUDA-enabled PyTorch
- Try the LCM model for faster generation

### "Too expensive for my GPU"
Use the LCM model with fewer steps:
```bash
python generate_from_prompt.py "prompt" --model lcm --steps 4
```

This is the "2025 efficient approach" in practice!

## Next Steps

After generating some images, try:

1. **Compare step counts** to see the speed/quality tradeoff
2. **Use LCM** to see the efficient 1-step approach
3. **Track energy** with `nvidia-smi` to see real power draw
4. **Experiment** with different prompts to see semantic understanding

## Summary Table

| Script | Purpose | Needs GPU? | Generates Real Images? |
|--------|---------|------------|------------------------|
| `pixel_map_efficiency.py` | Show calculations | No | No (just numbers) |
| `pixel_map_visualizer.py` | Show patterns | No | No (just patterns) |
| `why_iteration_costs.py` | Show iteration cost | No | No (simple blur demo) |
| **`generate_from_prompt.py`** | **Real AI art** | **Yes (recommended)** | **Yes! From your prompts** |

---

**Ready to try it?**

```bash
# Install dependencies
pip install torch diffusers transformers accelerate

# Generate your first image!
python generate_from_prompt.py "a beautiful sunset over ocean waves" --steps 20
```
