#!/usr/bin/env python3
"""
Generate Images with 2025 State-of-the-Art Models
==================================================

Updated November 2025 with latest open-source models:
- Qwen-Image (best for text rendering, multilingual)
- FLUX.1.1 (3x faster than FLUX.1)
- SD 3.5 Large (improved typography)
- HiDream-I1 (new SOTA for complexity)
- And more...

All models are Apache/MIT licensed and run locally.

Requirements:
    pip install torch diffusers transformers accelerate
    pip install bitsandbytes  # Optional: for quantization speedup
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
        DiffusionPipeline,
        DPMSolverMultistepScheduler,
        LCMScheduler,
    )
    from PIL import Image
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("=" * 80)
    print("ERROR: Required libraries not installed")
    print("=" * 80)
    print("Install with: pip install torch diffusers transformers accelerate")
    print("=" * 80)


class Model2025Generator:
    """Generate images with November 2025 state-of-the-art models"""

    def __init__(self, device: str = "auto", use_fp16: bool = True):
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError("diffusers not installed")

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.dtype = torch.float16 if use_fp16 and self.device == "cuda" else torch.float32

        print(f"\n{'='*80}")
        print(f"DEVICE CONFIGURATION")
        print(f"{'='*80}")
        print(f"Device: {self.device}")
        print(f"Dtype:  {self.dtype}")

        if self.device == "cpu":
            print(f"\n⚠️  CPU MODE - Generation will be slower")
            print(f"   Estimated times: 20s-180s per image (see model list)")
            print(f"   Recommend: Start with smaller models (Waifu, Neta, SDXL Turbo)")
            print(f"   Or use GPU via: Google Colab, Replicate.com, RunPod")
        else:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU:    {gpu_name}")
            print(f"VRAM:   {gpu_memory:.1f} GB")

            if gpu_memory < 8:
                print(f"\n⚠️  Low VRAM - Use smaller models or lower resolution")

        print(f"{'='*80}\n")

        self.pipes = {}

    def get_model_configs(self):
        """
        All available models as of November 2025

        Organized by use case:
        - Best for text rendering
        - Best quality (SOTA)
        - Best speed
        - Best for CPU
        - Specialized (anime, artistic, etc.)
        """
        return {
            # ========================================
            # BEST FOR TEXT RENDERING
            # ========================================
            "qwen-image": {
                "model_id": "Qwen/Qwen-Image",
                "pipeline": DiffusionPipeline,
                "quality": "⭐⭐⭐⭐⭐",
                "size": "~40GB",
                "params": "20B",
                "steps": 28,
                "speed": "Medium",
                "cpu_time": "120-180s",
                "gpu_time": "3-5s",
                "note": "SOTA text rendering (multilingual), excellent layouts, diverse styles",
                "width": 1024,
                "height": 1024,
                "released": "Aug 2025",
                "best_for": "Text-heavy images, posters, infographics, bilingual content",
            },

            "deepfloyd-if": {
                "model_id": "DeepFloyd/IF-I-XL-v1.0",
                "pipeline": DiffusionPipeline,
                "quality": "⭐⭐⭐⭐",
                "size": "~9GB",
                "params": "4.3B",
                "steps": 20,
                "speed": "Medium",
                "cpu_time": "50-80s",
                "gpu_time": "2-3s",
                "note": "Cascaded pixel diffusion, superior text integration",
                "width": 64,  # Starts at 64x64, then upscales
                "height": 64,
                "released": "Apr 2023 (2025 fine-tunes)",
                "best_for": "Legible in-image text, marketing materials",
            },

            # ========================================
            # BEST QUALITY (SOTA 2025)
            # ========================================
            "flux-1.1-pro": {
                "model_id": "black-forest-labs/FLUX.1.1-pro",
                "pipeline": DiffusionPipeline,
                "quality": "⭐⭐⭐⭐⭐",
                "size": "~23GB",
                "params": "12B",
                "steps": 4,
                "speed": "Fast (3x faster than FLUX.1)",
                "cpu_time": "45-75s",
                "gpu_time": "1-2s",
                "note": "Enhanced FLUX, better editing & consistency, outperforms SD3",
                "width": 1024,
                "height": 1024,
                "released": "Early 2025",
                "best_for": "Professional work, highest quality images",
            },

            "flux-schnell": {
                "model_id": "black-forest-labs/FLUX.1-schnell",
                "pipeline": DiffusionPipeline,
                "quality": "⭐⭐⭐⭐⭐",
                "size": "~23GB",
                "params": "12B",
                "steps": 4,
                "speed": "Fast (4-step optimized)",
                "cpu_time": "60-90s",
                "gpu_time": "1-2s",
                "note": "SOTA photorealism, excellent hands/faces/lighting",
                "width": 1024,
                "height": 1024,
                "released": "Aug 2024",
                "best_for": "Photorealistic portraits, detailed scenes",
            },

            "hidream-i1": {
                "model_id": "HiDream/HiDream-I1",
                "pipeline": DiffusionPipeline,
                "quality": "⭐⭐⭐⭐⭐",
                "size": "~30GB",
                "params": "17B",
                "steps": 20,
                "speed": "Slow (but best quality)",
                "cpu_time": "90-120s",
                "gpu_time": "3-4s",
                "note": "Sparse DiT architecture, outperforms FLUX on artistic benchmarks",
                "width": 1024,
                "height": 1024,
                "released": "Apr 2025",
                "best_for": "Complex artistic compositions, photorealism",
                "requires_ram": "32GB+",
            },

            "sd-3.5-large": {
                "model_id": "stabilityai/stable-diffusion-3.5-large",
                "pipeline": AutoPipelineForText2Image,
                "quality": "⭐⭐⭐⭐⭐",
                "size": "~16GB",
                "params": "8B",
                "steps": 20,
                "speed": "Medium",
                "cpu_time": "40-70s",
                "gpu_time": "2-3s",
                "note": "Accurate typography, high coherence, inpainting/outpainting",
                "width": 1024,
                "height": 1024,
                "released": "Oct 2024",
                "best_for": "General purpose, large community ecosystem",
            },

            # ========================================
            # BEST SPEED (1-4 STEPS)
            # ========================================
            "sdxl-turbo": {
                "model_id": "stabilityai/sdxl-turbo",
                "pipeline": AutoPipelineForText2Image,
                "quality": "⭐⭐⭐⭐",
                "size": "~7GB",
                "params": "3.5B",
                "steps": 1,
                "speed": "Very fast (1 step!)",
                "cpu_time": "20-40s",
                "gpu_time": "0.3-0.5s",
                "note": "SDXL quality in 1 step, speed king for real-time generation",
                "guidance_scale": 0.0,
                "width": 512,
                "height": 512,
                "released": "Nov 2023",
                "best_for": "Quick iterations, real-time applications",
            },

            "sdxl-lightning": {
                "model_id": "ByteDance/SDXL-Lightning",
                "pipeline": StableDiffusionXLPipeline,
                "quality": "⭐⭐⭐⭐",
                "size": "~7GB",
                "params": "3.5B",
                "steps": 4,
                "speed": "Very fast (4-8 steps)",
                "cpu_time": "30-50s",
                "gpu_time": "0.5-1s",
                "note": "Near-instant on GPU, good upscaling",
                "width": 1024,
                "height": 1024,
                "released": "Feb 2024",
                "best_for": "Fast drafts, iterative design",
            },

            # ========================================
            # BEST FOR CPU (LIGHTWEIGHT)
            # ========================================
            "waifu-diffusion": {
                "model_id": "hakurei/waifu-diffusion",
                "pipeline": StableDiffusionPipeline,
                "quality": "⭐⭐⭐",
                "size": "~2GB",
                "params": "1B",
                "steps": 20,
                "speed": "Very fast (lightweight)",
                "cpu_time": "15-30s",
                "gpu_time": "0.5-1s",
                "note": "Anime/manga specialist, consistent characters",
                "width": 512,
                "height": 512,
                "released": "2022 (2025 LoRAs)",
                "best_for": "Anime art, character design, CPU users",
            },

            "neta": {
                "model_id": "Neta-Art/Neta-v1",
                "pipeline": StableDiffusionPipeline,
                "quality": "⭐⭐⭐",
                "size": "~4GB",
                "params": "2B",
                "steps": 20,
                "speed": "Fast (lightweight)",
                "cpu_time": "25-45s",
                "gpu_time": "1-1.5s",
                "note": "Positioning-focused, excels at concept separation (multi-subject)",
                "width": 512,
                "height": 512,
                "released": "Jul 2025",
                "best_for": "Complex layouts, multiple subjects, CPU users",
            },

            "sd-3-medium": {
                "model_id": "stabilityai/stable-diffusion-3-medium-diffusers",
                "pipeline": AutoPipelineForText2Image,
                "quality": "⭐⭐⭐⭐",
                "size": "~4GB",
                "params": "2B",
                "steps": 20,
                "speed": "Fast",
                "cpu_time": "30-50s",
                "gpu_time": "1-2s",
                "note": "Efficient for text-heavy prompts, good mid-res",
                "width": 1024,
                "height": 1024,
                "released": "Jun 2024",
                "best_for": "CPU users wanting quality, text generation",
            },

            # ========================================
            # SPECIALIZED
            # ========================================
            "skyreels-v2": {
                "model_id": "SkyReels/SkyReels-v2",
                "pipeline": StableDiffusionPipeline,
                "quality": "⭐⭐⭐",
                "size": "~6GB",
                "params": "3B",
                "steps": 20,
                "speed": "Medium",
                "cpu_time": "35-55s",
                "gpu_time": "1.5-2s",
                "note": "Cinematic/video-like stills, better diversity than WAN",
                "width": 512,
                "height": 512,
                "released": "2025",
                "best_for": "Dynamic prompts, sunset/landscape scenes",
            },

            "janus-pro": {
                "model_id": "Janus-Pro/Janus-Pro-7B",
                "pipeline": DiffusionPipeline,
                "quality": "⭐⭐⭐",
                "size": "~12GB",
                "params": "7B",
                "steps": 20,
                "speed": "Medium",
                "cpu_time": "40-60s",
                "gpu_time": "2s",
                "note": "Hybrid diffusion, good for artistic variety, prompt separation",
                "width": 512,
                "height": 512,
                "released": "Mid-2025",
                "best_for": "Artistic styles, community-driven fine-tunes",
            },

            # ========================================
            # BASELINE (FOR COMPARISON)
            # ========================================
            "sd-1.5": {
                "model_id": "runwayml/stable-diffusion-v1-5",
                "pipeline": StableDiffusionPipeline,
                "quality": "⭐⭐",
                "size": "~4GB",
                "params": "1B",
                "steps": 20,
                "speed": "Slow (20-50 steps)",
                "cpu_time": "50-60s",
                "gpu_time": "2s",
                "note": "Original SD - dated (2022), OK quality, baseline comparison",
                "width": 512,
                "height": 512,
                "released": "2022",
                "best_for": "Baseline comparison only",
            },

            # ========================================
            # NSFW-CAPABLE (UNCENSORED COMMUNITY MODELS)
            # ========================================
            "realistic-vision-nsfw": {
                "model_id": "SG161222/Realistic_Vision_V5.1_noVAE",
                "pipeline": StableDiffusionPipeline,
                "quality": "⭐⭐⭐⭐",
                "size": "~4GB",
                "params": "1B",
                "steps": 20,
                "speed": "Medium",
                "cpu_time": "40-60s",
                "gpu_time": "2s",
                "note": "🔞 Photorealistic, uncensored community fine-tune",
                "width": 512,
                "height": 512,
                "released": "2023",
                "best_for": "Photorealistic uncensored content",
            },

            "dreamshaper-nsfw": {
                "model_id": "Lykon/DreamShaper",
                "pipeline": StableDiffusionPipeline,
                "quality": "⭐⭐⭐⭐",
                "size": "~4GB",
                "params": "1B",
                "steps": 20,
                "speed": "Medium",
                "cpu_time": "40-60s",
                "gpu_time": "2s",
                "note": "🔞 Versatile realistic, uncensored community model",
                "width": 512,
                "height": 512,
                "released": "2023",
                "best_for": "Versatile uncensored content",
            },

            "anything-v5-nsfw": {
                "model_id": "stablediffusionapi/anything-v5",
                "pipeline": StableDiffusionPipeline,
                "quality": "⭐⭐⭐⭐",
                "size": "~4GB",
                "params": "1B",
                "steps": 20,
                "speed": "Medium",
                "cpu_time": "40-60s",
                "gpu_time": "2s",
                "note": "🔞 Anime specialist, uncensored (trained on Danbooru)",
                "width": 512,
                "height": 512,
                "released": "2023",
                "best_for": "Anime uncensored content",
            },

            "deliberate-nsfw": {
                "model_id": "XpucT/Deliberate",
                "pipeline": StableDiffusionPipeline,
                "quality": "⭐⭐⭐⭐",
                "size": "~4GB",
                "params": "1B",
                "steps": 30,
                "speed": "Medium",
                "cpu_time": "50-70s",
                "gpu_time": "2-3s",
                "note": "🔞 Artistic/painted style, uncensored",
                "width": 512,
                "height": 512,
                "released": "2023",
                "best_for": "Artistic uncensored content",
            },
        }

    def list_models(self, sort_by: str = "quality"):
        """Print available models"""
        configs = self.get_model_configs()

        print("\n" + "=" * 120)
        print("AVAILABLE MODELS (November 2025)")
        print("=" * 120)

        if self.device == "cpu":
            print(f"{'Name':<20} {'Quality':<15} {'Size':<10} {'CPU Time':<15} {'Best For':<40}")
            print("-" * 120)
            for name, cfg in sorted(configs.items(), key=lambda x: x[1]['cpu_time'].split('-')[0]):
                print(f"{name:<20} {cfg['quality']:<15} {cfg['size']:<10} {cfg['cpu_time']:<15} {cfg['best_for']:<40}")
        else:
            print(f"{'Name':<20} {'Quality':<15} {'Size':<10} {'GPU Time':<15} {'Best For':<40}")
            print("-" * 120)
            for name, cfg in sorted(configs.items(), key=lambda x: x[1]['gpu_time'].split('-')[0]):
                print(f"{name:<20} {cfg['quality']:<15} {cfg['size']:<10} {cfg['gpu_time']:<15} {cfg['best_for']:<40}")

        print("=" * 120)
        print("\nRECOMMENDATIONS:")
        if self.device == "cpu":
            print("  🏆 Best for CPU (fast):    waifu-diffusion, neta, sdxl-turbo")
            print("  ⭐ Best quality on CPU:    sd-3-medium, sd-3.5-large")
            print("  📝 Best for text:          qwen-image (slow but worth it!)")
            print("  ⚡ Fastest:                waifu-diffusion (15-30s)")
        else:
            print("  🏆 Best overall quality:   flux-1.1-pro, hidream-i1, qwen-image")
            print("  ⚡ Fastest:                sdxl-turbo (0.3-0.5s)")
            print("  📝 Best for text:          qwen-image, deepfloyd-if")
            print("  💰 Best speed/quality:     sdxl-lightning, flux-schnell")
        print("=" * 120)

    def load_model(self, model_name: str):
        """Load a model by name"""
        if model_name in self.pipes:
            return self.pipes[model_name]

        configs = self.get_model_configs()
        if model_name not in configs:
            available = list(configs.keys())
            raise ValueError(f"Unknown model: {model_name}\nAvailable: {available}")

        cfg = configs[model_name]

        print(f"\n{'='*80}")
        print(f"LOADING: {model_name}")
        print(f"{'='*80}")
        print(f"Quality:     {cfg['quality']}")
        print(f"Size:        {cfg['size']} ({cfg['params']} parameters)")
        print(f"Released:    {cfg['released']}")
        print(f"Best for:    {cfg['best_for']}")
        print(f"Est. time:   {cfg['cpu_time'] if self.device == 'cpu' else cfg['gpu_time']}")
        print(f"Note:        {cfg['note']}")
        print(f"{'='*80}")

        if "requires_ram" in cfg and self.device == "cpu":
            print(f"⚠️  WARNING: This model requires {cfg['requires_ram']} RAM on CPU")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                raise RuntimeError("Model loading cancelled")

        print("Downloading model (first run only)...")

        try:
            pipeline_class = cfg["pipeline"]

            pipe = pipeline_class.from_pretrained(
                cfg["model_id"],
                torch_dtype=self.dtype,
                safety_checker=None,
            )

            if "scheduler" in cfg:
                pipe.scheduler = cfg["scheduler"].from_config(pipe.scheduler.config)

            pipe = pipe.to(self.device)

            if self.device == "cuda":
                try:
                    pipe.enable_attention_slicing()
                    pipe.enable_vae_slicing()
                    print("  ✓ Memory optimizations enabled")
                except:
                    pass

        except Exception as e:
            print(f"\n❌ Error loading model: {e}")
            print("\nTroubleshooting:")
            print("  • Out of memory? Try a smaller model")
            print("  • Network error? Check internet connection")
            print("  • VRAM error? Reduce image size or use CPU")
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
        cfg = self.get_model_configs()[model_name]

        # Use model defaults
        if num_steps is None:
            num_steps = cfg["steps"]
        if width is None:
            width = cfg["width"]
        if height is None:
            height = cfg["height"]
        if guidance_scale is None:
            guidance_scale = cfg.get("guidance_scale", 7.5)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        print(f"\n{'='*80}")
        print(f"GENERATING")
        print(f"{'='*80}")
        print(f"Prompt:   '{prompt}'")
        print(f"Model:    {model_name}")
        print(f"Steps:    {num_steps}")
        print(f"Size:     {width}×{height}")
        print(f"Est. time: {cfg['cpu_time'] if self.device == 'cpu' else cfg['gpu_time']}")
        print(f"{'='*80}")

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

        print(f"\n{'='*80}")
        print(f"✓ COMPLETE")
        print(f"{'='*80}")
        print(f"Actual time:  {elapsed:.2f}s")
        print(f"Time/step:    {elapsed/num_steps:.3f}s")
        print(f"{'='*80}")

        metadata = {
            'prompt': prompt,
            'model': model_name,
            'steps': num_steps,
            'width': width,
            'height': height,
            'time_sec': elapsed,
        }

        return {'image': result.images[0], 'metadata': metadata}


def save_image(image: Image.Image, metadata: dict, output_dir: Path):
    """Save image"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_clean = "".join(c if c.isalnum() or c in " -_" else ""
                          for c in metadata['prompt'][:40]).replace(" ", "_")
    filename = f"{timestamp}_{metadata['model']}_{prompt_clean}.png"
    filepath = output_dir / filename

    image.save(filepath)

    with open(filepath.with_suffix('.txt'), 'w') as f:
        for k, v in metadata.items():
            f.write(f"{k}: {v}\n")

    print(f"\n✓ Saved: {filename}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Generate with 2025 SOTA models")
    parser.add_argument("prompt", nargs="*", help="Text prompt")
    parser.add_argument("--model", "-m", default="sdxl-turbo", help="Model name")
    parser.add_argument("--list", action="store_true", help="List all models")
    parser.add_argument("--steps", "-s", type=int, help="Number of steps")
    parser.add_argument("--width", type=int, help="Width")
    parser.add_argument("--height", type=int, help="Height")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    args = parser.parse_args()

    if not DIFFUSERS_AVAILABLE:
        sys.exit(1)

    gen = Model2025Generator(device=args.device)

    if args.list:
        gen.list_models()
        return

    if not args.prompt:
        print("Usage: python generate_2025_models.py 'your prompt' --model MODEL")
        print("       python generate_2025_models.py --list")
        sys.exit(1)

    prompt = " ".join(args.prompt)
    output_dir = Path("output/2025_models")
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
