"""
Pixel Map Visualizer
====================

Practical demonstration of pixel map generation with visual output.
Shows the difference between:
1. Random noise (trivial)
2. Structured noise (simple constraints)
3. Why coherent images need heavy computation
"""

import numpy as np
from PIL import Image
import time
from pathlib import Path


def generate_random_pixels(width: int, height: int) -> tuple[np.ndarray, float]:
    """
    Generate completely random RGB pixels - the trivial case

    Returns:
        (image_array, time_ms)
    """
    start = time.perf_counter()

    # Pure random noise - no intelligence
    pixels = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    elapsed_ms = (time.perf_counter() - start) * 1000

    return pixels, elapsed_ms


def generate_smooth_noise(width: int, height: int, scale: int = 8) -> tuple[np.ndarray, float]:
    """
    Generate smoothed noise - shows why spatial constraints add cost

    This is still random, but enforces a simple constraint:
    "Nearby pixels should be similar" - requires more computation

    Args:
        scale: How much to downsample/upsample for smoothing

    Returns:
        (image_array, time_ms)
    """
    start = time.perf_counter()

    # Generate low-res noise
    low_res_h = height // scale
    low_res_w = width // scale
    low_res = np.random.randint(0, 256, (low_res_h, low_res_w, 3), dtype=np.uint8)

    # Upsample with bilinear interpolation (simulates spatial constraint)
    img = Image.fromarray(low_res)
    img = img.resize((width, height), Image.Resampling.BILINEAR)
    pixels = np.array(img)

    elapsed_ms = (time.perf_counter() - start) * 1000

    return pixels, elapsed_ms


def generate_gradient(width: int, height: int, colors: list = None) -> tuple[np.ndarray, float]:
    """
    Generate a smooth gradient - simple global constraint

    Constraint: "Color should change smoothly from top to bottom"
    Still trivial computation, but shows the concept of structure

    Returns:
        (image_array, time_ms)
    """
    start = time.perf_counter()

    if colors is None:
        colors = [(255, 0, 0), (0, 0, 255)]  # Red to blue

    # Create vertical gradient
    gradient = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        t = i / height  # 0.0 to 1.0
        # Linear interpolation between colors
        r = int(colors[0][0] * (1 - t) + colors[1][0] * t)
        g = int(colors[0][1] * (1 - t) + colors[1][1] * t)
        b = int(colors[0][2] * (1 - t) + colors[1][2] * t)
        gradient[i, :] = [r, g, b]

    elapsed_ms = (time.perf_counter() - start) * 1000

    return gradient, elapsed_ms


def generate_checkerboard(width: int, height: int, square_size: int = 64) -> tuple[np.ndarray, float]:
    """
    Generate a checkerboard pattern - shows structural constraints

    Constraint: "Alternating colors in a grid pattern"
    Still simple, but demonstrates spatial logic

    Returns:
        (image_array, time_ms)
    """
    start = time.perf_counter()

    pixels = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            # Determine if this square is black or white
            row_square = i // square_size
            col_square = j // square_size
            if (row_square + col_square) % 2 == 0:
                pixels[i, j] = [255, 255, 255]  # White
            else:
                pixels[i, j] = [0, 0, 0]  # Black

    elapsed_ms = (time.perf_counter() - start) * 1000

    return pixels, elapsed_ms


def simulate_semantic_constraint():
    """
    Explain why semantic constraints are expensive

    This function doesn't generate anything - it explains the problem
    """
    constraints = """
SEMANTIC CONSTRAINTS (Why Diffusion Models Are Expensive)
==========================================================

To generate "a photo of a cat", the model must enforce:

1. GLOBAL SEMANTIC STRUCTURE
   - Object identity: "This is a cat, not a dog"
   - Scene composition: "Cat in foreground, background blurred"
   - Requires: Learned priors from millions of training images
   - Cost: Billions of neural network parameters

2. LOCAL TEXTURE DETAILS
   - Fur texture: "Individual hairs, not solid color"
   - Eye structure: "Reflections, pupil, iris details"
   - Whiskers: "Thin, curved lines from face"
   - Requires: Multi-scale pattern matching
   - Cost: Multiple convolutional layers at different resolutions

3. SPATIAL COHERENCE (Long-range dependencies)
   - Ear above eye, not on tail
   - Shadow direction consistent with light source
   - Perspective: Things get smaller with distance
   - Requires: Attention mechanisms across entire image
   - Cost: O(n²) where n = number of pixels/patches

4. CROSS-MODAL ALIGNMENT
   - Text "red apple" → actual red color + round shape
   - Style: "photorealistic" vs "cartoon" vs "oil painting"
   - Requires: Text encoder + image decoder coordination
   - Cost: Additional transformer networks

HOW DIFFUSION MODELS ENFORCE THESE:
------------------------------------
1. Start with random noise (cheap - like generate_random_pixels())
2. For each denoising step (20-1000 iterations):
   a. Pass entire image through U-Net (conv layers)
      → Cost: ~50 TFLOPs per step
   b. Apply transformer attention (check all pixel relationships)
      → Cost: ~200 TFLOPs per step
   c. Adjust pixels based on text prompt embedding
      → Cost: ~10 TFLOPs per step
3. Repeat until image is coherent

Total: 20 steps × 260 TFLOPs = 5.2 PFLOPs
Compare to random pixels: ~3 MFLOPs (1 million× difference!)

WHY CAN'T WE JUST "PREDICT THE RIGHT PIXELS"?
----------------------------------------------
We can! That's exactly what 2025's efficient models do:

- Consistency Models: Train to map noise → image in 1 step
  → Cuts 20× computation, but needs careful training

- RAE (NYU): Predict 4×4 semantic grid, then expand
  → Only predicts 16 "concepts", not 1M pixels

- TokenSet (Tencent): Treat image as set of patches, not sequence
  → Avoids sequential processing overhead

- Optical (UCLA): Use physical light interference
  → No digital computation at all during inference!

The bottleneck is not the pixels themselves.
It's encoding human semantic understanding into those pixels.
"""
    return constraints


def main():
    """Generate and save example pixel maps"""

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    width, height = 512, 512

    print("=" * 80)
    print("PIXEL MAP GENERATION EXAMPLES")
    print("=" * 80)
    print(f"Resolution: {width}×{height} RGB")
    print(f"Total pixels: {width * height:,} × 3 channels = {width * height * 3:,} values\n")

    # 1. Random noise
    print("[1] Random Noise (No Constraints)")
    print("-" * 80)
    random_pixels, random_time = generate_random_pixels(width, height)
    Image.fromarray(random_pixels).save(output_dir / "01_random_noise.png")
    print(f"Time: {random_time:.2f} ms")
    print(f"FLOPs: ~{width * height * 3 / 1e6:.1f} million")
    print(f"Output: output/01_random_noise.png")
    print("Result: Pure chaos - no structure whatsoever\n")

    # 2. Smooth noise
    print("[2] Smooth Noise (Simple Spatial Constraint)")
    print("-" * 80)
    smooth_pixels, smooth_time = generate_smooth_noise(width, height, scale=8)
    Image.fromarray(smooth_pixels).save(output_dir / "02_smooth_noise.png")
    print(f"Time: {smooth_time:.2f} ms")
    print(f"Cost increase: {smooth_time / random_time:.1f}× vs random")
    print(f"Output: output/02_smooth_noise.png")
    print("Result: 'Nearby pixels should be similar' - simple but visible structure\n")

    # 3. Gradient
    print("[3] Gradient (Global Color Constraint)")
    print("-" * 80)
    gradient_pixels, gradient_time = generate_gradient(width, height)
    Image.fromarray(gradient_pixels).save(output_dir / "03_gradient.png")
    print(f"Time: {gradient_time:.2f} ms")
    print(f"Output: output/03_gradient.png")
    print("Result: Smooth color transition - aesthetically pleasing but no 'content'\n")

    # 4. Checkerboard
    print("[4] Checkerboard (Structural Pattern)")
    print("-" * 80)
    checker_pixels, checker_time = generate_checkerboard(width, height, square_size=64)
    Image.fromarray(checker_pixels).save(output_dir / "04_checkerboard.png")
    print(f"Time: {checker_time:.2f} ms")
    print(f"Output: output/04_checkerboard.png")
    print("Result: Recognizable pattern - but still no semantic meaning\n")

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"All 4 examples generated in under {checker_time:.0f} ms total")
    print("All have STRUCTURE, but none have SEMANTIC MEANING\n")

    print("To generate 'a photo of a cat' at this resolution:")
    print("  - Diffusion model: ~2-10 seconds on RTX 4090")
    print("  - Power: 450W")
    print("  - Cost: ~5 PFLOPs (1,000,000× more than random noise)")
    print("\nWhy the difference?")
    print("  → Semantic, spatial, and photometric constraints")
    print("  → Requires learned knowledge from millions of training images")
    print("  → Must coordinate billions of neural network parameters")
    print("=" * 80)

    # Print explanation of semantic constraints
    print("\n" + simulate_semantic_constraint())

    print("\nImages saved to output/ directory:")
    print("  01_random_noise.png    - Pure randomness (3 MFLOPs)")
    print("  02_smooth_noise.png    - Spatial smoothness constraint")
    print("  03_gradient.png        - Global color constraint")
    print("  04_checkerboard.png    - Structural pattern")
    print("\nNone of these have semantic meaning - that's the expensive part!")


if __name__ == "__main__":
    main()
