#!/usr/bin/env python3
"""
AI Image Generator - One Simple Command
========================================

Just run: python generate.py

Features:
- 13+ models (SDXL Turbo, FLUX, Qwen-Image, NSFW, etc.)
- Quantization support (4-bit/8-bit for limited RAM)
- Interactive menu or CLI mode
- Runs locally on CPU or GPU

Examples:
    # Interactive mode (easiest)
    python generate.py

    # CLI mode
    python generate.py "a cat" --model sdxl-turbo
    python generate.py "poster text" --model qwen-image --quantize 4bit
    python generate.py "portrait" --model realistic-vision-nsfw --steps 25
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

try:
    import torch
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
        AutoPipelineForText2Image,
        DiffusionPipeline,
        BitsAndBytesConfig,
    )
    from PIL import Image
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

def get_all_models():
    """All available models with their configurations"""
    return {
        # CPU-Friendly Fast Models
        "sdxl-turbo": {
            "model_id": "stabilityai/sdxl-turbo",
            "pipeline": AutoPipelineForText2Image,
            "quality": "⭐⭐⭐⭐",
            "size": "~7GB",
            "steps": 1,
            "cpu_time": "20-40s",
            "gpu_time": "0.3s",
            "note": "Fast + great quality - BEST FOR CPU",
            "width": 512,
            "height": 512,
            "category": "fast",
        },
        "sd-3-medium": {
            "model_id": "stabilityai/stable-diffusion-3-medium-diffusers",
            "pipeline": AutoPipelineForText2Image,
            "quality": "⭐⭐⭐⭐",
            "size": "~4GB",
            "steps": 20,
            "cpu_time": "30-50s",
            "gpu_time": "1-2s",
            "note": "Good text rendering, balanced",
            "width": 1024,
            "height": 1024,
            "category": "fast",
        },
        "waifu-diffusion": {
            "model_id": "hakurei/waifu-diffusion",
            "pipeline": StableDiffusionPipeline,
            "quality": "⭐⭐⭐",
            "size": "~2GB",
            "steps": 20,
            "cpu_time": "15-30s",
            "gpu_time": "0.5s",
            "note": "Fastest! Anime/manga style",
            "width": 512,
            "height": 512,
            "category": "fast",
        },

        # High Quality Models
        "flux-schnell": {
            "model_id": "black-forest-labs/FLUX.1-schnell",
            "pipeline": DiffusionPipeline,
            "quality": "⭐⭐⭐⭐⭐",
            "size": "~23GB",
            "steps": 4,
            "cpu_time": "60-90s",
            "gpu_time": "1-2s",
            "note": "Best photorealism (slow on CPU)",
            "width": 1024,
            "height": 1024,
            "category": "quality",
            "supports_quantization": True,
        },
        "sd-3.5-large": {
            "model_id": "stabilityai/stable-diffusion-3.5-large",
            "pipeline": AutoPipelineForText2Image,
            "quality": "⭐⭐⭐⭐⭐",
            "size": "~16GB",
            "steps": 20,
            "cpu_time": "40-70s",
            "gpu_time": "2-3s",
            "note": "Excellent typography (slow on CPU)",
            "width": 1024,
            "height": 1024,
            "category": "quality",
            "supports_quantization": True,
        },
        "qwen-image": {
            "model_id": "Qwen/Qwen-Image",
            "pipeline": DiffusionPipeline,
            "quality": "⭐⭐⭐⭐⭐",
            "size": "~40GB",
            "steps": 28,
            "cpu_time": "120-180s",
            "gpu_time": "3-5s",
            "note": "Best text rendering! (needs quantization)",
            "width": 1024,
            "height": 1024,
            "category": "quality",
            "supports_quantization": True,
            "requires_quantization": True,
        },

        # NSFW Models
        "realistic-vision-nsfw": {
            "model_id": "SG161222/Realistic_Vision_V5.1_noVAE",
            "pipeline": StableDiffusionPipeline,
            "quality": "⭐⭐⭐⭐",
            "size": "~4GB",
            "steps": 20,
            "cpu_time": "40-60s",
            "gpu_time": "2s",
            "note": "🔞 Photorealistic, uncensored",
            "width": 512,
            "height": 512,
            "category": "nsfw",
        },
        "dreamshaper-nsfw": {
            "model_id": "Lykon/DreamShaper",
            "pipeline": StableDiffusionPipeline,
            "quality": "⭐⭐⭐⭐",
            "size": "~4GB",
            "steps": 20,
            "cpu_time": "40-60s",
            "gpu_time": "2s",
            "note": "🔞 Versatile realistic, uncensored",
            "width": 512,
            "height": 512,
            "category": "nsfw",
        },
        "anything-v5-nsfw": {
            "model_id": "stablediffusionapi/anything-v5",
            "pipeline": StableDiffusionPipeline,
            "quality": "⭐⭐⭐⭐",
            "size": "~4GB",
            "steps": 20,
            "cpu_time": "40-60s",
            "gpu_time": "2s",
            "note": "🔞 Anime specialist, uncensored",
            "width": 512,
            "height": 512,
            "category": "nsfw",
        },
        "deliberate-nsfw": {
            "model_id": "XpucT/Deliberate",
            "pipeline": StableDiffusionPipeline,
            "quality": "⭐⭐⭐⭐",
            "size": "~4GB",
            "steps": 30,
            "cpu_time": "50-70s",
            "gpu_time": "2-3s",
            "note": "🔞 Artistic/painted, uncensored",
            "width": 512,
            "height": 512,
            "category": "nsfw",
        },

        # Baseline
        "sd-1.5": {
            "model_id": "runwayml/stable-diffusion-v1-5",
            "pipeline": StableDiffusionPipeline,
            "quality": "⭐⭐",
            "size": "~4GB",
            "steps": 20,
            "cpu_time": "50-60s",
            "gpu_time": "2s",
            "note": "For comparison only (outdated)",
            "width": 512,
            "height": 512,
            "category": "baseline",
        },
    }


# ============================================================================
# IMAGE GENERATOR
# ============================================================================

class ImageGenerator:
    """Unified image generator with all models and features"""

    def __init__(self, device="auto", quantization=None):
        if not DEPS_AVAILABLE:
            raise RuntimeError("Dependencies not installed. Run: pip install torch diffusers transformers accelerate protobuf sentencepiece")

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.quantization = quantization
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipes = {}

        print(f"\n{'='*80}")
        print(f"DEVICE: {self.device.upper()}")
        if quantization:
            print(f"QUANTIZATION: {quantization}")
            memory_savings = {"4bit": "~75%", "8bit": "~50%"}
            print(f"MEMORY SAVED: {memory_savings.get(quantization, 'N/A')}")
        print(f"{'='*80}\n")

    def load_model(self, model_name):
        """Load a model with optional quantization"""
        cache_key = f"{model_name}_{self.quantization or 'none'}"
        if cache_key in self.pipes:
            return self.pipes[cache_key]

        models = get_all_models()
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}\nAvailable: {list(models.keys())}")

        cfg = models[model_name]

        print(f"\n{'='*80}")
        print(f"LOADING: {model_name}")
        print(f"{'='*80}")
        print(f"Quality:  {cfg['quality']}")
        print(f"Size:     {cfg['size']}")
        print(f"Note:     {cfg['note']}")
        print(f"{'='*80}")

        # Check if quantization is required
        if cfg.get("requires_quantization") and not self.quantization:
            print(f"\n⚠️  This model requires quantization!")
            print(f"   Add: --quantize 4bit")
            raise RuntimeError("Quantization required")

        # Setup quantization if requested
        quantization_config = None
        if self.quantization:
            if not cfg.get("supports_quantization"):
                print(f"⚠️  Warning: {model_name} doesn't support quantization")
                print(f"   Running in normal mode...")
            else:
                if self.quantization == "4bit":
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                    )
                    print("  Using 4-bit quantization (75% memory reduction)")
                elif self.quantization == "8bit":
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    print("  Using 8-bit quantization (50% memory reduction)")

        print("Downloading model (first run only)...")

        try:
            # Load pipeline
            if quantization_config:
                pipe = cfg["pipeline"].from_pretrained(
                    cfg["model_id"],
                    torch_dtype=self.dtype,
                    quantization_config=quantization_config,
                    device_map="auto" if self.device == "cuda" else None,
                    safety_checker=None,
                )
            else:
                pipe = cfg["pipeline"].from_pretrained(
                    cfg["model_id"],
                    torch_dtype=self.dtype,
                    safety_checker=None,
                )
                pipe = pipe.to(self.device)

            # Memory optimizations
            if self.device == "cuda":
                try:
                    pipe.enable_attention_slicing()
                    pipe.enable_vae_slicing()
                except:
                    pass

            print(f"✓ Model loaded successfully!")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            if "memory" in str(e).lower():
                print("\n💡 OUT OF MEMORY! Try:")
                print("   • Add --quantize 4bit")
                print("   • Use a smaller model")
                print("   • Close other applications")
            raise

        self.pipes[cache_key] = pipe
        return pipe

    def generate(self, prompt, model_name="sdxl-turbo", steps=None, width=None, height=None, seed=None):
        """Generate an image"""
        pipe = self.load_model(model_name)
        cfg = get_all_models()[model_name]

        # Use defaults if not specified
        if steps is None:
            steps = cfg["steps"]
        if width is None:
            width = cfg["width"]
        if height is None:
            height = cfg["height"]

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        print(f"\n{'='*80}")
        print(f"GENERATING")
        print(f"{'='*80}")
        print(f"Prompt: '{prompt}'")
        print(f"Model:  {model_name}")
        print(f"Steps:  {steps}")
        print(f"Size:   {width}×{height}")
        print(f"{'='*80}")

        start = time.perf_counter()

        with torch.inference_mode():
            result = pipe(
                prompt,
                num_inference_steps=steps,
                width=width,
                height=height,
                generator=generator,
            )

        elapsed = time.perf_counter() - start

        print(f"\n{'='*80}")
        print(f"✓ COMPLETE - {elapsed:.1f}s ({elapsed/steps:.2f}s per step)")
        print(f"{'='*80}")

        return result.images[0], {
            'prompt': prompt,
            'model': model_name,
            'quantization': self.quantization,
            'steps': steps,
            'width': width,
            'height': height,
            'time_sec': elapsed,
        }


def save_image(image, metadata, output_dir):
    """Save image with metadata"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_clean = "".join(c if c.isalnum() or c in " -_" else ""
                          for c in metadata['prompt'][:40]).replace(" ", "_")

    filename = f"{timestamp}_{metadata['model']}_{prompt_clean}.png"
    filepath = output_dir / filename

    image.save(filepath)

    # Save metadata
    with open(filepath.with_suffix('.txt'), 'w') as f:
        for k, v in metadata.items():
            f.write(f"{k}: {v}\n")

    print(f"\n✓ Saved: {filename}")
    return filepath


# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def print_models_menu():
    """Display model selection menu"""
    models = get_all_models()

    print("\n" + "=" * 80)
    print("AVAILABLE MODELS")
    print("=" * 80)

    # Fast models
    print("\n🚀 FAST (CPU-FRIENDLY):")
    print("-" * 80)
    for i, (name, cfg) in enumerate([(k, v) for k, v in models.items() if v['category'] == 'fast'], 1):
        print(f"  [{i}] {name:<25} {cfg['quality']:<15} {cfg['cpu_time']}")
        print(f"      {cfg['note']}")

    # Quality models
    print("\n🎨 HIGH QUALITY:")
    print("-" * 80)
    start = sum(1 for v in models.values() if v['category'] == 'fast')
    for i, (name, cfg) in enumerate([(k, v) for k, v in models.items() if v['category'] == 'quality'], start+1):
        quant_note = " (add --quantize 4bit)" if cfg.get('requires_quantization') else ""
        print(f"  [{i}] {name:<25} {cfg['quality']:<15} {cfg['cpu_time']}{quant_note}")
        print(f"      {cfg['note']}")

    # NSFW models
    print("\n🔞 NSFW (UNCENSORED):")
    print("-" * 80)
    start = sum(1 for v in models.values() if v['category'] in ['fast', 'quality'])
    for i, (name, cfg) in enumerate([(k, v) for k, v in models.items() if v['category'] == 'nsfw'], start+1):
        print(f"  [{i}] {name:<25} {cfg['quality']:<15} {cfg['cpu_time']}")
        print(f"      {cfg['note']}")

    # Baseline
    print("\n📊 BASELINE:")
    print("-" * 80)
    start = sum(1 for v in models.values() if v['category'] in ['fast', 'quality', 'nsfw'])
    for i, (name, cfg) in enumerate([(k, v) for k, v in models.items() if v['category'] == 'baseline'], start+1):
        print(f"  [{i}] {name:<25} {cfg['quality']:<15} {cfg['cpu_time']}")
        print(f"      {cfg['note']}")

    print("\n" + "=" * 80)


def interactive_mode():
    """Run interactive session"""
    print("\n" + "=" * 80)
    print("AI IMAGE GENERATOR")
    print("=" * 80)
    print("Generate AI images from text prompts!")
    print("=" * 80)

    # Show models and let user pick
    print_models_menu()

    print("\n💡 RECOMMENDATIONS:")
    print("  • First time?           Try [1] sdxl-turbo")
    print("  • Need text in image?   Try [2] sd-3-medium or [6] qwen-image")
    print("  • Want best quality?    Try [4] flux-schnell")
    print("  • NSFW photorealistic?  Try [7] realistic-vision-nsfw")
    print("  • NSFW anime?           Try [9] anything-v5-nsfw")

    models_list = list(get_all_models().items())

    while True:
        choice = input("\nSelect model [1-13] or 'q' to quit: ").strip()

        if choice.lower() == 'q':
            print("Goodbye!")
            return

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(models_list):
                model_name, cfg = models_list[idx]
                break
        except ValueError:
            pass

        print("Invalid choice!")

    print(f"\n✓ Selected: {model_name}")
    print(f"  {cfg['note']}")

    # Check if needs quantization
    quantization = None
    if cfg.get('requires_quantization'):
        print(f"\n⚠️  This model requires quantization")
        quantization = "4bit"
        print(f"  Using 4-bit quantization automatically")
    elif cfg.get('supports_quantization'):
        use_quant = input(f"\nUse quantization for less RAM? (y/n): ").strip().lower()
        if use_quant == 'y':
            quantization = "4bit"
            print("  Using 4-bit quantization")

    # Initialize generator
    gen = ImageGenerator(quantization=quantization)

    # Generation loop
    while True:
        prompt = input("\nEnter prompt (or 'quit'): ").strip()

        if not prompt or prompt.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        # Ask for steps with guidance
        print(f"\n💡 Recommended steps for {model_name}: {cfg['steps']}")
        steps_input = input(f"Enter steps (or press Enter for {cfg['steps']}): ").strip()
        steps = int(steps_input) if steps_input.isdigit() else None

        # Generate
        try:
            image, metadata = gen.generate(prompt, model_name, steps=steps)
            save_image(image, metadata, "output/generated")

            print("\n✓ Image saved to output/generated/")

        except KeyboardInterrupt:
            print("\n\n⚠️  Cancelled!")
        except Exception as e:
            print(f"\n❌ Error: {e}")

        # Continue?
        another = input("\nGenerate another? (y/n): ").strip().lower()
        if another != 'y':
            print("Goodbye!")
            break


# ============================================================================
# COMMAND LINE MODE
# ============================================================================

def cli_mode(args):
    """Run in CLI mode"""
    if args.list:
        print_models_menu()
        return

    if not args.prompt:
        print("Error: No prompt provided")
        print("\nUsage:")
        print('  python generate.py "your prompt" --model MODEL')
        print('  python generate.py --list')
        print("\nExamples:")
        print('  python generate.py "a cat"')
        print('  python generate.py "a cat" --model flux-schnell')
        print('  python generate.py "poster text" --model qwen-image --quantize 4bit')
        sys.exit(1)

    gen = ImageGenerator(device=args.device, quantization=args.quantize)

    image, metadata = gen.generate(
        prompt=args.prompt,
        model_name=args.model,
        steps=args.steps,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )

    save_image(image, metadata, "output/generated")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AI Image Generator - One Simple Command",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python generate.py

  # Generate with default model
  python generate.py "a beautiful sunset"

  # Use specific model
  python generate.py "a cat" --model flux-schnell

  # Use quantization for large models
  python generate.py "poster with text" --model qwen-image --quantize 4bit

  # NSFW models
  python generate.py "your prompt" --model realistic-vision-nsfw

  # List all models
  python generate.py --list
        """
    )

    parser.add_argument("prompt", nargs="?", help="Text prompt")
    parser.add_argument("--model", "-m", default="sdxl-turbo", help="Model name")
    parser.add_argument("--steps", "-s", type=int, help="Number of steps")
    parser.add_argument("--width", type=int, help="Image width")
    parser.add_argument("--height", type=int, help="Image height")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--quantize", "-q", choices=["4bit", "8bit"], help="Quantization (for large models)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Device")
    parser.add_argument("--list", action="store_true", help="List all models")

    args = parser.parse_args()

    if not DEPS_AVAILABLE:
        print("=" * 80)
        print("ERROR: Required libraries not installed")
        print("=" * 80)
        print("Install with:")
        print("  pip install torch diffusers transformers accelerate protobuf sentencepiece")
        print("\nFor quantization support:")
        print("  pip install bitsandbytes")
        print("=" * 80)
        sys.exit(1)

    # Interactive mode if no prompt
    if not args.prompt and not args.list:
        try:
            interactive_mode()
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
    else:
        cli_mode(args)


if __name__ == "__main__":
    main()
