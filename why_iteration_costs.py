"""
Why Iteration Costs So Much
============================

Demonstrates the computational cost of iterative refinement
vs. direct generation.

This shows a simplified diffusion process and counts actual FLOPs.
"""

import numpy as np
from PIL import Image
import time
from pathlib import Path


def simple_blur_kernel(size: int = 3) -> np.ndarray:
    """Create a simple averaging blur kernel"""
    kernel = np.ones((size, size)) / (size * size)
    return kernel


def convolve_2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Simple 2D convolution (per-channel)

    In real neural networks, each conv layer has:
    - Multiple input channels (e.g., 3 for RGB)
    - Multiple output channels (e.g., 64, 128, 256)
    - Multiple layers stacked (U-Net has ~100 layers)

    This is a toy version to show the basic operation.
    """
    h, w = image.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2

    # Pad image
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

    # Output
    output = np.zeros_like(image)

    # Convolve
    flops = 0
    for i in range(h):
        for j in range(w):
            # Extract patch
            patch = padded[i:i+k_h, j:j+k_w]
            # Element-wise multiply and sum
            output[i, j] = np.sum(patch * kernel)
            # Count FLOPs: k_h * k_w multiplications + (k_h * k_w - 1) additions
            flops += k_h * k_w * 2 - 1

    return output, flops


def iterative_denoising(
    noise: np.ndarray,
    target: np.ndarray,
    num_steps: int,
    learning_rate: float = 0.1
) -> tuple[list[np.ndarray], int]:
    """
    Simplified iterative denoising process

    In real diffusion:
    - Each step runs a full U-Net (millions of parameters)
    - Uses sophisticated noise prediction
    - Has attention mechanisms

    This toy version:
    - Gradually moves from noise toward target
    - Uses simple convolution to simulate network processing
    - Still demonstrates why iteration is expensive

    Returns:
        (list of intermediate images, total FLOPs)
    """
    current = noise.copy()
    kernel = simple_blur_kernel(3)
    intermediates = [current.copy()]
    total_flops = 0

    for step in range(num_steps):
        # Simulate "network processing" with blur
        # (Real networks do far more computation per step)
        processed_r, flops_r = convolve_2d(current[:, :, 0], kernel)
        processed_g, flops_g = convolve_2d(current[:, :, 1], kernel)
        processed_b, flops_b = convolve_2d(current[:, :, 2], kernel)

        processed = np.stack([processed_r, processed_g, processed_b], axis=2)
        step_flops = flops_r + flops_g + flops_b

        # Move toward target (simplified "denoising")
        current = current + learning_rate * (target - current)

        # Mix in processed version (simulates neural network refinement)
        current = 0.7 * current + 0.3 * processed

        # Additional FLOPs for the mixing operations
        mixing_flops = current.size * 4  # multiply + add for each pixel
        step_flops += mixing_flops

        total_flops += step_flops
        intermediates.append(current.copy())

    return intermediates, total_flops


def direct_generation(target: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Direct generation - just return the target

    In real 1-step models:
    - Still runs a full network once
    - But trained to map noise → image directly
    - No iteration loop

    Returns:
        (image, FLOPs)
    """
    # In reality, there's still one network forward pass
    # But we avoid the 20-50× iteration multiplier
    kernel = simple_blur_kernel(3)

    # Simulate one network pass
    processed_r, flops_r = convolve_2d(target[:, :, 0], kernel)
    processed_g, flops_g = convolve_2d(target[:, :, 1], kernel)
    processed_b, flops_b = convolve_2d(target[:, :, 2], kernel)

    result = np.stack([processed_r, processed_g, processed_b], axis=2)
    total_flops = flops_r + flops_g + flops_b

    return result, total_flops


def main():
    """Demonstrate iterative vs direct generation costs"""

    output_dir = Path("output/iteration_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a simple target image (gradient)
    size = 256  # Small for fast demo
    print(f"Image size: {size}×{size} RGB = {size * size * 3:,} values\n")

    # Create target pattern
    target = np.zeros((size, size, 3), dtype=float)
    for i in range(size):
        for j in range(size):
            target[i, j, 0] = (i / size) * 255  # Red gradient
            target[i, j, 1] = (j / size) * 255  # Green gradient
            target[i, j, 2] = 128  # Constant blue

    # Create random noise
    noise = np.random.rand(size, size, 3) * 255

    print("=" * 80)
    print("ITERATIVE VS DIRECT GENERATION")
    print("=" * 80)

    # Test iterative denoising with different step counts
    step_counts = [1, 5, 20, 50]

    for num_steps in step_counts:
        print(f"\n[ITERATIVE] {num_steps} denoising steps")
        print("-" * 80)

        start = time.perf_counter()
        intermediates, flops = iterative_denoising(
            noise, target, num_steps, learning_rate=0.1
        )
        elapsed = time.perf_counter() - start

        # Save first, middle, and last frame
        first_frame = np.clip(intermediates[0], 0, 255).astype(np.uint8)
        mid_idx = len(intermediates) // 2
        mid_frame = np.clip(intermediates[mid_idx], 0, 255).astype(np.uint8)
        last_frame = np.clip(intermediates[-1], 0, 255).astype(np.uint8)

        Image.fromarray(first_frame).save(
            output_dir / f"iterative_{num_steps:02d}_start.png"
        )
        Image.fromarray(mid_frame).save(
            output_dir / f"iterative_{num_steps:02d}_mid.png"
        )
        Image.fromarray(last_frame).save(
            output_dir / f"iterative_{num_steps:02d}_end.png"
        )

        print(f"Time:       {elapsed*1000:.2f} ms")
        print(f"FLOPs:      {flops:,} ({flops/1e6:.2f} MFLOPs)")
        print(f"Saved:      output/iteration_demo/iterative_{num_steps:02d}_*.png")

    # Test direct generation
    print(f"\n[DIRECT] 1-step generation")
    print("-" * 80)

    start = time.perf_counter()
    result, flops = direct_generation(target)
    elapsed = time.perf_counter() - start

    result_img = np.clip(result, 0, 255).astype(np.uint8)
    Image.fromarray(result_img).save(output_dir / "direct_1step.png")

    print(f"Time:       {elapsed*1000:.2f} ms")
    print(f"FLOPs:      {flops:,} ({flops/1e6:.2f} MFLOPs)")
    print(f"Saved:      output/iteration_demo/direct_1step.png")

    # Save target for reference
    target_img = np.clip(target, 0, 255).astype(np.uint8)
    Image.fromarray(target_img).save(output_dir / "target.png")

    # Comparison
    print("\n" + "=" * 80)
    print("COST COMPARISON (This Toy Example)")
    print("=" * 80)
    print(f"{'Method':<30} {'FLOPs':<20} {'Multiplier':<15}")
    print("-" * 80)

    # Get flops for each approach
    _, flops_1 = iterative_denoising(noise, target, 1, learning_rate=0.1)
    _, flops_5 = iterative_denoising(noise, target, 5, learning_rate=0.1)
    _, flops_20 = iterative_denoising(noise, target, 20, learning_rate=0.1)
    _, flops_50 = iterative_denoising(noise, target, 50, learning_rate=0.1)
    _, flops_direct = direct_generation(target)

    print(f"{'Direct (1-step)':<30} {flops_direct/1e6:>8.2f} MFLOPs      1.0×")
    print(f"{'Iterative (1 step)':<30} {flops_1/1e6:>8.2f} MFLOPs      "
          f"{flops_1/flops_direct:.1f}×")
    print(f"{'Iterative (5 steps)':<30} {flops_5/1e6:>8.2f} MFLOPs      "
          f"{flops_5/flops_direct:.1f}×")
    print(f"{'Iterative (20 steps)':<30} {flops_20/1e6:>8.2f} MFLOPs      "
          f"{flops_20/flops_direct:.1f}×")
    print(f"{'Iterative (50 steps)':<30} {flops_50/1e6:>8.2f} MFLOPs      "
          f"{flops_50/flops_direct:.1f}×")
    print("=" * 80)

    print("\nREAL-WORLD SCALING:")
    print("-" * 80)
    print("This toy example uses simple 3×3 convolutions.")
    print("Real diffusion models (FLUX, Stable Diffusion) use:")
    print("  - U-Net with 100+ layers")
    print("  - Conv kernels with 64, 128, 256+ channels")
    print("  - Transformer attention (O(n²) cost)")
    print("  - Multiple resolution levels (downsampling/upsampling)")
    print("\nScaling factor: ~1,000,000×")
    print("  → 20 steps × 1M parameters = ~5 PFLOPs total")
    print("  → That's why it needs a GPU and takes seconds")
    print("=" * 80)

    print("\nKEY INSIGHT:")
    print("-" * 80)
    print("Iterative refinement gives better results BUT:")
    print("  - Each step processes the ENTIRE image")
    print("  - Cost scales linearly with number of steps")
    print("  - 50 steps = 50× the computation of 1 step")
    print("\n2025's efficient models:")
    print("  - Train to get good results in 1-2 steps instead of 20-50")
    print("  - Same network size, but run it fewer times")
    print("  - Result: 20-50× speedup and energy savings")
    print("=" * 80)


if __name__ == "__main__":
    main()
