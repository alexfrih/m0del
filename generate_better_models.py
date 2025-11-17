#!/usr/bin/env python3
"""
Generate Images with Better Open Source Models
===============================================

Updated version with newer, higher-quality models:
- FLUX.1-schnell (best quality, 2024)
- SDXL Turbo (fast + high quality)
- Stable Diffusion XL (better than SD 1.5)
- LCM models (1-step generation)

Requirements:
    pip install torch diffusers transformers accelerate
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
        StableDiffusionXLPipeline,
        AutoPipelineForText2Image,
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
    print("Install with: pip install torch diffusers transformers accelerate")
    print("=" * 80)


class BetterImageGenerator:
    """Generate images with state-of-the-art open source models"""

    def __init__(self, device: str = "auto", use_fp16: bool = True):
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError("diffusers not installed")

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.dtype = torch.float16 if use_fp16 and self.device == "cuda" else torch.float32

        print(f"Using device: {self.device}")
        if self.device == "cpu":
            print("⚠️  WARNING: Running on CPU will be VERY slow!")
            print("   Recommend using a GPU for models larger than SD 1.5")
        print(f"Using dtype: {self.dtype}")

        self.pipes = {}

    def get_model_configs(self):
        """
        Available models with their configurations

        Quality ratings:
        ⭐⭐⭐⭐⭐ = State-of-the-art (2024-2025)
        ⭐⭐⭐⭐   = Excellent
        ⭐⭐⭐     = Good
        ⭐⭐       = OK (older)
        """
        return {
            # === BEST QUALITY (2024-2025) ===
            "flux-schnell": {
                "model_id": "black-forest-labs/FLUX.1-schnell",
                "pipeline": AutoPipelineForText2Image,
                "quality": "⭐⭐⭐⭐⭐",
                "size": "~23GB",
                "steps": 4,
                "speed": "Fast (4 steps optimized)",
                "note": "SOTA quality, best results, needs 16GB+ GPU",
                "width": 1024,
                "height": 1024,
            },

            # === FAST + HIGH QUALITY ===
            "sdxl-turbo": {
                "model_id": "stabilityai/sdxl-turbo",
                "pipeline": AutoPipelineForText2Image,
                "quality": "⭐⭐⭐⭐",
                "size": "~7GB",
                "steps": 1,
                "speed": "Very fast (1-4 steps)",
                "note": "Distilled SDXL - great quality in 1 step!",
                "guidance_scale": 0.0,  # Turbo models don't use guidance
                "width": 512,
                "height": 512,
            },

            "sd3-medium": {
                "model_id": "stabilityai/stable-diffusion-3-medium-diffusers",
                "pipeline": AutoPipelineForText2Image,
                "quality": "⭐⭐⭐⭐⭐",
                "size": "~10GB",
                "steps": 20,
                "speed": "Medium",
                "note": "SD 3 - excellent prompt following, newer architecture",
                "width": 1024,
                "height": 1024,
            },

            # === GOOD QUALITY ===
            "sdxl": {
                "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
                "pipeline": StableDiffusionXLPipeline,
                "quality": "⭐⭐⭐⭐",
                "size": "~7GB",
                "steps": 20,
                "speed": "Medium (20-50 steps)",
                "note": "Better quality than SD 1.5, industry standard",
                "width": 1024,
                "height": 1024,
            },

            "sdxl-lightning": {
                "model_id": "ByteDance/SDXL-Lightning",
                "pipeline": StableDiffusionXLPipeline,
                "quality": "⭐⭐⭐⭐",
                "size": "~7GB",
                "steps": 4,
                "speed": "Fast (4-8 steps)",
                "note": "SDXL quality in 4 steps - great efficiency!",
                "repo_file": "sdxl_lightning_4step_unet.safetensors",
                "width": 1024,
                "height": 1024,
            },

            # === FAST (1-4 STEPS) ===
            "lcm-dreamshaper": {
                "model_id": "SimianLuo/LCM_Dreamshaper_v7",
                "pipeline": StableDiffusionPipeline,
                "quality": "⭐⭐⭐",
                "size": "~4GB",
                "steps": 4,
                "speed": "Very fast (1-8 steps)",
                "note": "LCM - SD 1.5 quality in 4 steps",
                "scheduler": LCMScheduler,
                "width": 512,
                "height": 512,
            },

            # === BASELINE (YOUR CURRENT MODEL) ===
            "sd-1.5": {
                "model_id": "runwayml/stable-diffusion-v1-5",
                "pipeline": StableDiffusionPipeline,
                "quality": "⭐⭐",
                "size": "~4GB",
                "steps": 20,
                "speed": "Slow (20-50 steps)",
                "note": "Your current model - OK quality, dated (2022)",
                "width": 512,
                "height": 512,
            },
        }

    def list_models(self):
        """Print available models"""
        configs = self.get_model_configs()

        print("\n" + "=" * 90)
        print("AVAILABLE MODELS")
        print("=" * 90)
        print(f"{'Name':<20} {'Quality':<15} {'Size':<10} {'Speed':<20} {'Steps':<8}")
        print("-" * 90)

        for name, config in configs.items():
            print(f"{name:<20} {config['quality']:<15} {config['size']:<10} "
                  f"{config['speed']:<20} {config['steps']:<8}")

        print("=" * 90)
        print("\nRECOMMENDATIONS:")
        print("  • Best quality:       flux-schnell (needs powerful GPU)")
        print("  • Best speed/quality: sdxl-turbo or sdxl-lightning")
        print("  • CPU compatible:     sd-1.5 or lcm-dreamshaper (small, fast)")
        print("  • Balanced:           sdxl (good quality, reasonable size)")
        print("=" * 90)

    def load_model(self, model_name: str):
        """Load a model by name"""
        if model_name in self.pipes:
            return self.pipes[model_name]

        configs = self.get_model_configs()

        if model_name not in configs:
            available = list(configs.keys())
            raise ValueError(f"Unknown model: {model_name}. Available: {available}")

        config = configs[model_name]

        print(f"\n{'='*80}")
        print(f"LOADING MODEL: {model_name}")
        print(f"{'='*80}")
        print(f"Quality:  {config['quality']}")
        print(f"Size:     {config['size']}")
        print(f"Speed:    {config['speed']}")
        print(f"Note:     {config['note']}")
        print(f"{'='*80}")
        print("Downloading model files (first run only, ~1-5 min)...")

        # Load pipeline
        pipeline_class = config["pipeline"]

        try:
            if "scheduler" in config:
                # Custom scheduler
                pipe = pipeline_class.from_pretrained(
                    config["model_id"],
                    torch_dtype=self.dtype,
                    safety_checker=None,
                )
                pipe.scheduler = config["scheduler"].from_config(pipe.scheduler.config)
            else:
                pipe = pipeline_class.from_pretrained(
                    config["model_id"],
                    torch_dtype=self.dtype,
                    safety_checker=None,
                )

            pipe = pipe.to(self.device)

            # Enable optimizations
            if self.device == "cuda":
                try:
                    pipe.enable_attention_slicing()
                    print("  ✓ Memory optimization enabled")
                except:
                    pass

        except Exception as e:
            print(f"\n❌ Error loading model: {e}")
            print("\nTroubleshooting:")
            print("  • Model too large for your system? Try a smaller model")
            print("  • Out of memory? Reduce image size or use CPU")
            print("  • Network issues? Check internet connection")
            raise

        self.pipes[model_name] = pipe
        print(f"✓ Model loaded successfully!")
        return pipe

    def generate(
        self,
        prompt: str,
        model_name: str = "sdxl-turbo",
        num_steps: int = None,
        width: int = None,
        height: int = None,
        guidance_scale: float = None,
        seed: int = None,
    ):
        """Generate an image"""

        pipe = self.load_model(model_name)
        config = self.get_model_configs()[model_name]

        # Use model defaults if not specified
        if num_steps is None:
            num_steps = config["steps"]
        if width is None:
            width = config["width"]
        if height is None:
            height = config["height"]
        if guidance_scale is None:
            guidance_scale = config.get("guidance_scale", 7.5)

        # Set seed
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        print(f"\n{'='*80}")
        print(f"GENERATING IMAGE")
        print(f"{'='*80}")
        print(f"Prompt:   '{prompt}'")
        print(f"Model:    {model_name}")
        print(f"Steps:    {num_steps}")
        print(f"Size:     {width}×{height}")
        print(f"Guidance: {guidance_scale}")
        print(f"{'='*80}")

        # Generate
        if self.device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

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

        elapsed = time.perf_counter() - start

        image = result.images[0]

        # Stats
        print(f"\n{'='*80}")
        print(f"COMPLETE!")
        print(f"{'='*80}")
        print(f"Time:         {elapsed:.2f} seconds")
        print(f"Time/step:    {elapsed/num_steps:.3f} seconds")

        if self.device == "cpu":
            print(f"\n💡 TIP: This would be ~20-50× faster on GPU!")

        print(f"{'='*80}")

        metadata = {
            'prompt': prompt,
            'model': model_name,
            'steps': num_steps,
            'width': width,
            'height': height,
            'guidance_scale': guidance_scale,
            'seed': seed,
            'time_sec': elapsed,
        }

        return {'image': image, 'metadata': metadata}


def save_image(image: Image.Image, metadata: dict, output_dir: Path):
    """Save image with metadata"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = metadata['model']
    prompt_clean = "".join(c if c.isalnum() or c in " -_" else ""
                          for c in metadata['prompt'][:40]).replace(" ", "_")

    filename = f"{timestamp}_{model}_{prompt_clean}.png"
    filepath = output_dir / filename

    image.save(filepath)

    # Save metadata
    meta_file = filepath.with_suffix('.txt')
    with open(meta_file, 'w') as f:
        for k, v in metadata.items():
            f.write(f"{k}: {v}\n")

    print(f"\n✓ Saved: {filename}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Generate images with better open source models")
    parser.add_argument("prompt", nargs="*", help="Text prompt")
    parser.add_argument("--model", "-m", default="sdxl-turbo",
                       help="Model to use (default: sdxl-turbo)")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--steps", "-s", type=int, help="Number of steps (uses model default if not set)")
    parser.add_argument("--width", type=int, help="Image width")
    parser.add_argument("--height", type=int, help="Image height")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    args = parser.parse_args()

    if not DIFFUSERS_AVAILABLE:
        sys.exit(1)

    gen = BetterImageGenerator(device=args.device)

    if args.list:
        gen.list_models()
        return

    if not args.prompt:
        print("Usage: python generate_better_models.py 'your prompt here' [options]")
        print("Or:    python generate_better_models.py --list")
        print("\nQuick examples:")
        print("  python generate_better_models.py 'a cat' --model sdxl-turbo")
        print("  python generate_better_models.py 'a sunset' --model flux-schnell")
        sys.exit(1)

    prompt = " ".join(args.prompt)

    output_dir = Path("output/better_models")
    output_dir.mkdir(parents=True, exist_ok=True)

    result = gen.generate(
        prompt=prompt,
        model_name=args.model,
        num_steps=args.steps,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )

    save_image(result['image'], result['metadata'], output_dir)


if __name__ == "__main__":
    main()
