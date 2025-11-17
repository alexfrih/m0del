#!/usr/bin/env python3
"""
Generate Images from Your Own Prompts
======================================

This script uses real diffusion models to generate images from text prompts.
It compares different approaches and shows the performance difference.

Requirements:
    pip install torch diffusers transformers accelerate pillow

GPU recommended (but CPU will work, just slower).
"""

import argparse
import time
from pathlib import Path
from datetime import datetime
import sys

try:
    import torch
    from diffusers import (
        StableDiffusionPipeline,
        DPMSolverMultistepScheduler,
        LCMScheduler,
    )
    from PIL import Image
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("=" * 80)
    print("WARNING: diffusers not installed")
    print("=" * 80)
    print("To use this script, install the required packages:")
    print("  pip install torch diffusers transformers accelerate")
    print("\nOr install all optional dependencies:")
    print("  pip install -r requirements.txt  # (uncomment the optional section)")
    print("=" * 80)


class ImageGenerator:
    """Handles image generation with performance tracking"""

    def __init__(self, device: str = "auto", use_fp16: bool = True):
        """
        Initialize the generator

        Args:
            device: "cuda", "cpu", or "auto" (auto-detect)
            use_fp16: Use half precision (faster, less memory)
        """
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError("diffusers not installed. See error message above.")

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.dtype = torch.float16 if use_fp16 and self.device == "cuda" else torch.float32

        print(f"Using device: {self.device}")
        print(f"Using dtype: {self.dtype}")

        self.pipes = {}  # Cache loaded models

    def load_model(self, model_type: str = "sd-1.5"):
        """
        Load a specific model

        Args:
            model_type: "sd-1.5", "sd-2.1", "sdxl", "lcm"
        """
        if model_type in self.pipes:
            return self.pipes[model_type]

        print(f"\nLoading model: {model_type}...")
        print("(This may take a minute on first run - models are ~4GB)")

        model_configs = {
            "sd-1.5": {
                "model_id": "runwayml/stable-diffusion-v1-5",
                "scheduler": DPMSolverMultistepScheduler,
            },
            "sd-2.1": {
                "model_id": "stabilityai/stable-diffusion-2-1",
                "scheduler": DPMSolverMultistepScheduler,
            },
            "lcm": {
                "model_id": "SimianLuo/LCM_Dreamshaper_v7",
                "scheduler": LCMScheduler,
                "note": "Latent Consistency Model - optimized for 1-4 steps",
            },
        }

        if model_type not in model_configs:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available: {list(model_configs.keys())}")

        config = model_configs[model_type]

        # Load pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            config["model_id"],
            torch_dtype=self.dtype,
            safety_checker=None,  # Disable for speed
        )

        # Set scheduler
        pipe.scheduler = config["scheduler"].from_config(pipe.scheduler.config)

        # Move to device
        pipe = pipe.to(self.device)

        # Enable optimizations if on CUDA
        if self.device == "cuda":
            try:
                # Enable memory efficient attention if available
                pipe.enable_attention_slicing()
                print("  ✓ Enabled attention slicing (memory optimization)")
            except:
                pass

            try:
                # Try xformers if available (faster)
                pipe.enable_xformers_memory_efficient_attention()
                print("  ✓ Enabled xformers (speed optimization)")
            except:
                pass

        self.pipes[model_type] = pipe

        if "note" in config:
            print(f"  Note: {config['note']}")

        print(f"✓ Model loaded: {model_type}")
        return pipe

    def generate(
        self,
        prompt: str,
        model_type: str = "sd-1.5",
        num_steps: int = 20,
        width: int = 512,
        height: int = 512,
        guidance_scale: float = 7.5,
        seed: int = None,
    ):
        """
        Generate an image from a text prompt

        Args:
            prompt: Text description of desired image
            model_type: Which model to use
            num_steps: Number of denoising steps (more = better quality, slower)
            width, height: Image dimensions
            guidance_scale: How closely to follow prompt (7.5 is typical)
            seed: Random seed for reproducibility

        Returns:
            dict with image, metadata, and performance stats
        """
        pipe = self.load_model(model_type)

        # Set seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        print(f"\nGenerating image...")
        print(f"  Prompt: '{prompt}'")
        print(f"  Model: {model_type}")
        print(f"  Steps: {num_steps}")
        print(f"  Size: {width}×{height}")
        print(f"  Device: {self.device}")

        # Warm up GPU if using CUDA (first run is slower)
        if self.device == "cuda" and not hasattr(self, '_warmed_up'):
            print("  Warming up GPU...")
            _ = pipe(
                "warmup",
                num_inference_steps=1,
                width=width,
                height=height,
                generator=generator,
            )
            torch.cuda.synchronize()
            self._warmed_up = True

        # Measure generation time
        if self.device == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        # Generate
        with torch.inference_mode():
            result = pipe(
                prompt,
                num_inference_steps=num_steps,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                generator=generator,
            )

        if self.device == "cuda":
            torch.cuda.synchronize()

        elapsed_time = time.perf_counter() - start_time

        image = result.images[0]

        # Calculate performance metrics
        total_pixels = width * height * 3
        megapixels = (width * height) / 1e6

        # Rough FLOP estimate (these are approximations)
        # SD 1.5: ~25 TFLOPs per step at 512x512
        # Scales roughly with pixel count
        flops_per_step = 25e12 * (megapixels / 0.262)  # 512x512 = 0.262 MP
        total_flops = flops_per_step * num_steps

        metadata = {
            'prompt': prompt,
            'model': model_type,
            'steps': num_steps,
            'width': width,
            'height': height,
            'guidance_scale': guidance_scale,
            'seed': seed,
            'device': self.device,
            'time_sec': elapsed_time,
            'time_per_step_sec': elapsed_time / num_steps,
            'total_pixels': total_pixels,
            'megapixels': megapixels,
            'estimated_flops': total_flops,
            'estimated_tflops': total_flops / 1e12,
        }

        # Print results
        print(f"\n{'='*80}")
        print(f"GENERATION COMPLETE")
        print(f"{'='*80}")
        print(f"Time:              {elapsed_time:.2f} seconds")
        print(f"Time per step:     {metadata['time_per_step_sec']:.3f} seconds")
        print(f"Est. FLOPs:        {metadata['estimated_tflops']:.1f} TFLOPs")
        print(f"Est. FLOP/s:       {total_flops / elapsed_time / 1e12:.1f} TFLOPs/s")

        if self.device == "cuda":
            print(f"Est. power draw:   ~{self._estimate_power()}W (GPU)")
            print(f"Est. energy:       ~{self._estimate_power() * elapsed_time:.1f} Joules")

        print(f"{'='*80}")

        return {
            'image': image,
            'metadata': metadata,
        }

    def _estimate_power(self):
        """Rough power estimate based on GPU"""
        if self.device != "cuda":
            return 50  # CPU estimate

        try:
            # Get GPU name
            gpu_name = torch.cuda.get_device_name(0).lower()

            # Rough estimates based on common GPUs
            if "4090" in gpu_name:
                return 450
            elif "4080" in gpu_name:
                return 320
            elif "3090" in gpu_name:
                return 350
            elif "3080" in gpu_name:
                return 320
            elif "3060" in gpu_name:
                return 170
            elif "a100" in gpu_name:
                return 400
            elif "h100" in gpu_name:
                return 700
            else:
                return 250  # Generic estimate
        except:
            return 250


def save_image_with_metadata(image: Image.Image, metadata: dict, output_dir: Path):
    """Save image with metadata in filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = metadata['model']
    steps = metadata['steps']
    time_sec = metadata['time_sec']

    # Clean prompt for filename (first 50 chars, safe characters only)
    prompt_clean = "".join(c if c.isalnum() or c in " -_" else "" for c in metadata['prompt'][:50])
    prompt_clean = prompt_clean.replace(" ", "_").strip("_")

    filename = f"{timestamp}_{model}_s{steps}_{prompt_clean}.png"
    filepath = output_dir / filename

    # Save image
    image.save(filepath)

    # Save metadata as text file
    metadata_file = filepath.with_suffix('.txt')
    with open(metadata_file, 'w') as f:
        f.write(f"GENERATION METADATA\n")
        f.write(f"{'='*60}\n\n")
        for key, value in metadata.items():
            f.write(f"{key:20s}: {value}\n")

    print(f"\n✓ Saved: {filename}")
    print(f"✓ Metadata: {metadata_file.name}")

    return filepath


def compare_step_counts(generator: ImageGenerator, prompt: str, model_type: str = "sd-1.5"):
    """Generate same prompt with different step counts to show performance difference"""
    print("\n" + "="*80)
    print("COMPARING DIFFERENT STEP COUNTS")
    print("="*80)
    print(f"Prompt: '{prompt}'")
    print(f"Model: {model_type}")

    step_counts = [1, 5, 20, 50] if model_type == "lcm" else [5, 20, 50]
    results = []

    # Use same seed for fair comparison
    seed = 42

    for steps in step_counts:
        result = generator.generate(
            prompt=prompt,
            model_type=model_type,
            num_steps=steps,
            seed=seed,
        )
        results.append((steps, result))

    # Print comparison table
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    print(f"{'Steps':<10} {'Time':<15} {'FLOPs':<20} {'Speedup vs 50':<15}")
    print("-"*80)

    baseline_time = next(r['metadata']['time_sec'] for s, r in results if s == max(step_counts))

    for steps, result in results:
        meta = result['metadata']
        speedup = baseline_time / meta['time_sec']
        print(f"{steps:<10} {meta['time_sec']:>6.2f} sec     "
              f"{meta['estimated_tflops']:>8.1f} TFLOPs      "
              f"{speedup:>5.1f}×")

    print("="*80)

    return results


def interactive_mode(generator: ImageGenerator):
    """Interactive prompt entry mode"""
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("Enter prompts to generate images.")
    print("Type 'quit' or 'exit' to stop.")
    print("Type 'compare' to run performance comparison.")
    print("="*80)

    output_dir = Path("output/generated")
    output_dir.mkdir(parents=True, exist_ok=True)

    while True:
        print()
        prompt = input("Enter prompt: ").strip()

        if prompt.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not prompt:
            continue

        if prompt.lower() == 'compare':
            print()
            compare_prompt = input("Enter prompt for comparison: ").strip()
            if compare_prompt:
                compare_step_counts(generator, compare_prompt, model_type="sd-1.5")
            continue

        # Ask for parameters
        try:
            steps = input("Number of steps [20]: ").strip()
            steps = int(steps) if steps else 20

            result = generator.generate(
                prompt=prompt,
                model_type="sd-1.5",
                num_steps=steps,
            )

            save_image_with_metadata(result['image'], result['metadata'], output_dir)

        except KeyboardInterrupt:
            print("\nCancelled.")
            continue
        except Exception as e:
            print(f"Error: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Generate images from text prompts with performance tracking"
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Text prompt for image generation"
    )
    parser.add_argument(
        "--model", "-m",
        default="sd-1.5",
        choices=["sd-1.5", "sd-2.1", "lcm"],
        help="Model to use (default: sd-1.5)"
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=20,
        help="Number of denoising steps (default: 20)"
    )
    parser.add_argument(
        "--width", "-w",
        type=int,
        default=512,
        help="Image width (default: 512)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height (default: 512)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare different step counts"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use (default: auto)"
    )

    args = parser.parse_args()

    if not DIFFUSERS_AVAILABLE:
        sys.exit(1)

    # Create generator
    generator = ImageGenerator(device=args.device)

    # Create output directory
    output_dir = Path("output/generated")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Interactive mode
    if args.interactive or not args.prompt:
        interactive_mode(generator)
        return

    # Join prompt
    prompt = " ".join(args.prompt)

    # Comparison mode
    if args.compare:
        results = compare_step_counts(generator, prompt, args.model)
        # Save all results
        for steps, result in results:
            save_image_with_metadata(result['image'], result['metadata'], output_dir)
        return

    # Single generation
    result = generator.generate(
        prompt=prompt,
        model_type=args.model,
        num_steps=args.steps,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )

    save_image_with_metadata(result['image'], result['metadata'], output_dir)


if __name__ == "__main__":
    main()
