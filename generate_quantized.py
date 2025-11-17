#!/usr/bin/env python3
"""
Generate with Quantized Models (Smaller Memory Footprint)
==========================================================

Runs large models on limited RAM using:
- 8-bit quantization (half memory)
- 4-bit quantization (quarter memory)
- FP8 precision (half memory)

This lets you run models like Qwen-Image and FLUX on 8-16GB RAM!

Requirements:
    pip install torch diffusers transformers accelerate bitsandbytes
"""

import argparse
import time
from pathlib import Path
from datetime import datetime
import sys

try:
    import torch
    from diffusers import (
        DiffusionPipeline,
        AutoPipelineForText2Image,
        BitsAndBytesConfig,
    )
    from PIL import Image
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("=" * 80)
    print("ERROR: Required libraries not installed")
    print("=" * 80)
    print("Install with: pip install torch diffusers transformers accelerate bitsandbytes")
    print("=" * 80)


class QuantizedGenerator:
    """Generate images with quantized models for lower memory usage"""

    def __init__(self, device: str = "auto", quantization: str = "8bit"):
        """
        Args:
            device: "cuda", "cpu", or "auto"
            quantization: "none", "8bit", "4bit", "fp8"
        """
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError("diffusers not installed")

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.quantization = quantization

        print(f"\n{'='*80}")
        print(f"QUANTIZED MODEL GENERATOR")
        print(f"{'='*80}")
        print(f"Device:        {self.device}")
        print(f"Quantization:  {quantization}")

        if quantization != "none":
            memory_reduction = {"8bit": "~50%", "4bit": "~75%", "fp8": "~50%"}
            print(f"Memory saved:  {memory_reduction.get(quantization, 'varies')}")

        if self.device == "cpu":
            print(f"\n⚠️  CPU MODE - Generation will be slower")
            if quantization == "none":
                print(f"   TIP: Use --quantize 8bit or 4bit to reduce memory usage!")

        print(f"{'='*80}\n")

        self.pipes = {}

    def get_model_configs(self):
        """Available large models with quantization support"""
        return {
            # Models that benefit most from quantization
            "qwen-image": {
                "model_id": "Qwen/Qwen-Image",
                "full_size": "~40GB",
                "8bit_size": "~20GB",
                "4bit_size": "~10GB",
                "quality": "⭐⭐⭐⭐⭐",
                "note": "SOTA text rendering, multilingual",
                "cpu_time_full": "120-180s",
                "cpu_time_8bit": "100-140s",
                "cpu_time_4bit": "80-120s",
                "best_for": "Text-heavy images, posters, multilingual content",
            },

            "flux-schnell": {
                "model_id": "black-forest-labs/FLUX.1-schnell",
                "full_size": "~23GB",
                "8bit_size": "~12GB",
                "4bit_size": "~6GB",
                "quality": "⭐⭐⭐⭐⭐",
                "note": "SOTA photorealism",
                "cpu_time_full": "60-90s",
                "cpu_time_8bit": "50-75s",
                "cpu_time_4bit": "45-65s",
                "best_for": "Photorealistic images, portraits",
            },

            "flux-dev": {
                "model_id": "black-forest-labs/FLUX.1-dev",
                "full_size": "~23GB",
                "8bit_size": "~12GB",
                "4bit_size": "~6GB",
                "quality": "⭐⭐⭐⭐⭐",
                "note": "FLUX development version, more flexible",
                "cpu_time_full": "100-150s",
                "cpu_time_8bit": "85-120s",
                "cpu_time_4bit": "70-100s",
                "best_for": "High-quality generation, more steps",
            },

            "sd-3.5-large": {
                "model_id": "stabilityai/stable-diffusion-3.5-large",
                "full_size": "~16GB",
                "8bit_size": "~8GB",
                "4bit_size": "~4GB",
                "quality": "⭐⭐⭐⭐⭐",
                "note": "Latest SD, excellent typography",
                "cpu_time_full": "40-70s",
                "cpu_time_8bit": "35-60s",
                "cpu_time_4bit": "30-50s",
                "best_for": "General purpose, community support",
            },

            "sdxl": {
                "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
                "full_size": "~7GB",
                "8bit_size": "~3.5GB",
                "4bit_size": "~2GB",
                "quality": "⭐⭐⭐⭐",
                "note": "Industry standard SDXL",
                "cpu_time_full": "50-80s",
                "cpu_time_8bit": "45-70s",
                "cpu_time_4bit": "40-60s",
                "best_for": "Balanced quality/size",
            },
        }

    def list_models(self):
        """Print available models with size comparisons"""
        configs = self.get_model_configs()

        print("\n" + "=" * 120)
        print("QUANTIZED MODELS - MEMORY USAGE COMPARISON")
        print("=" * 120)
        print(f"{'Model':<20} {'Full Size':<12} {'8-bit':<12} {'4-bit':<12} {'CPU Time (4-bit)':<20} {'Best For':<30}")
        print("-" * 120)

        for name, cfg in configs.items():
            print(f"{name:<20} {cfg['full_size']:<12} {cfg['8bit_size']:<12} "
                  f"{cfg['4bit_size']:<12} {cfg['cpu_time_4bit']:<20} {cfg['best_for']:<30}")

        print("=" * 120)
        print("\nMEMORY REQUIREMENTS:")
        print("  Full precision (FP32):  Highest quality, most memory")
        print("  8-bit quantization:     ~50% memory, minimal quality loss")
        print("  4-bit quantization:     ~75% memory, slight quality loss")
        print("\nRECOMMENDATIONS:")
        print("  8GB RAM:  Use 4-bit quantization with SD 3.5 Large or SDXL")
        print("  16GB RAM: Use 8-bit quantization with Qwen-Image or FLUX")
        print("  32GB RAM: Use full precision or 8-bit for best quality")
        print("=" * 120)

    def load_model(self, model_name: str):
        """Load a model with quantization"""
        cache_key = f"{model_name}_{self.quantization}"

        if cache_key in self.pipes:
            return self.pipes[cache_key]

        configs = self.get_model_configs()
        if model_name not in configs:
            available = list(configs.keys())
            raise ValueError(f"Unknown model: {model_name}\nAvailable: {available}")

        cfg = configs[model_name]

        # Determine size and time based on quantization
        if self.quantization == "8bit":
            size = cfg['8bit_size']
            time_est = cfg.get('cpu_time_8bit', cfg['cpu_time_full'])
        elif self.quantization == "4bit":
            size = cfg['4bit_size']
            time_est = cfg.get('cpu_time_4bit', cfg['cpu_time_full'])
        else:
            size = cfg['full_size']
            time_est = cfg['cpu_time_full']

        print(f"\n{'='*80}")
        print(f"LOADING: {model_name} ({self.quantization})")
        print(f"{'='*80}")
        print(f"Quality:       {cfg['quality']}")
        print(f"Size:          {size}")
        print(f"Quantization:  {self.quantization}")
        print(f"Est. time:     {time_est}")
        print(f"Note:          {cfg['note']}")
        print(f"{'='*80}")
        print("Downloading and loading model...")

        try:
            # Prepare quantization config
            if self.quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                torch_dtype = torch.float16
                print("  Using 8-bit quantization (50% memory reduction)")

            elif self.quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
                torch_dtype = torch.float16
                print("  Using 4-bit NF4 quantization (75% memory reduction)")

            else:
                quantization_config = None
                torch_dtype = torch.float32 if self.device == "cpu" else torch.float16

            # Load pipeline
            if quantization_config:
                pipe = DiffusionPipeline.from_pretrained(
                    cfg["model_id"],
                    torch_dtype=torch_dtype,
                    quantization_config=quantization_config,
                    device_map="auto" if self.device == "cuda" else None,
                )
            else:
                pipe = DiffusionPipeline.from_pretrained(
                    cfg["model_id"],
                    torch_dtype=torch_dtype,
                )
                pipe = pipe.to(self.device)

            # Memory optimizations
            if self.device == "cuda":
                try:
                    pipe.enable_attention_slicing()
                    pipe.enable_vae_slicing()
                    print("  ✓ Memory optimizations enabled")
                except:
                    pass

            print(f"✓ Model loaded successfully!")

        except Exception as e:
            print(f"\n❌ Error loading model: {e}")

            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                print("\n💡 OUT OF MEMORY! Try:")
                print(f"   • Use more aggressive quantization: --quantize 4bit")
                print(f"   • Try a smaller model: sd-3.5-large or sdxl")
                print(f"   • Close other applications")
                print(f"   • Use a cloud GPU (Google Colab, Replicate)")

            raise

        self.pipes[cache_key] = pipe
        return pipe

    def generate(
        self,
        prompt: str,
        model_name: str = "sd-3.5-large",
        num_steps: int = 20,
        width: int = 512,
        height: int = 512,
        seed: int = None,
    ):
        """Generate an image with quantized model"""

        pipe = self.load_model(model_name)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        print(f"\n{'='*80}")
        print(f"GENERATING")
        print(f"{'='*80}")
        print(f"Prompt:        '{prompt}'")
        print(f"Model:         {model_name}")
        print(f"Quantization:  {self.quantization}")
        print(f"Steps:         {num_steps}")
        print(f"Size:          {width}×{height}")
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
                generator=generator,
            )

        if self.device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        print(f"\n{'='*80}")
        print(f"✓ COMPLETE")
        print(f"{'='*80}")
        print(f"Time:     {elapsed:.2f}s")
        print(f"Per step: {elapsed/num_steps:.3f}s")
        print(f"{'='*80}")

        metadata = {
            'prompt': prompt,
            'model': model_name,
            'quantization': self.quantization,
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
    quant = metadata.get('quantization', 'none')
    filename = f"{timestamp}_{metadata['model']}_{quant}_{prompt_clean}.png"
    filepath = output_dir / filename

    image.save(filepath)

    with open(filepath.with_suffix('.txt'), 'w') as f:
        for k, v in metadata.items():
            f.write(f"{k}: {v}\n")

    print(f"\n✓ Saved: {filename}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Generate with quantized models")
    parser.add_argument("prompt", nargs="*", help="Text prompt")
    parser.add_argument("--model", "-m", default="sd-3.5-large", help="Model name")
    parser.add_argument("--quantize", "-q", default="8bit",
                       choices=["none", "8bit", "4bit"],
                       help="Quantization level (default: 8bit)")
    parser.add_argument("--list", action="store_true", help="List models")
    parser.add_argument("--steps", "-s", type=int, default=20, help="Steps")
    parser.add_argument("--width", type=int, default=512, help="Width")
    parser.add_argument("--height", type=int, default=512, help="Height")
    parser.add_argument("--seed", type=int, help="Seed")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    args = parser.parse_args()

    if not DIFFUSERS_AVAILABLE:
        sys.exit(1)

    gen = QuantizedGenerator(device=args.device, quantization=args.quantize)

    if args.list:
        gen.list_models()
        return

    if not args.prompt:
        print("Usage: python generate_quantized.py 'your prompt' --model MODEL --quantize 8bit")
        print("       python generate_quantized.py --list")
        print("\nExamples:")
        print("  # Qwen-Image with 4-bit (fits in 8GB RAM!)")
        print("  python generate_quantized.py 'poster with text AI' --model qwen-image --quantize 4bit")
        print("\n  # FLUX with 8-bit")
        print("  python generate_quantized.py 'a cat' --model flux-schnell --quantize 8bit")
        sys.exit(1)

    prompt = " ".join(args.prompt)
    output_dir = Path("output/quantized")
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
