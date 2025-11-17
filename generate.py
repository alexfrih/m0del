#!/usr/bin/env python3
"""
Simple Interactive Image Generator
===================================

Just run: python generate.py

Then follow the prompts!
"""

import subprocess
import sys
from pathlib import Path

MODELS = {
    # CPU-Friendly (Fast)
    "1": {
        "name": "SDXL Turbo",
        "id": "sdxl-turbo",
        "time": "20-40s",
        "quality": "⭐⭐⭐⭐",
        "note": "Fast + great quality - BEST FOR CPU",
        "script": "generate_2025_models.py",
    },
    "2": {
        "name": "SD 3 Medium",
        "id": "sd-3-medium",
        "time": "30-50s",
        "quality": "⭐⭐⭐⭐",
        "note": "Good text rendering, balanced",
        "script": "generate_2025_models.py",
    },
    "3": {
        "name": "Waifu Diffusion",
        "id": "waifu-diffusion",
        "time": "15-30s",
        "quality": "⭐⭐⭐",
        "note": "Fastest! Anime/manga style",
        "script": "generate_2025_models.py",
    },

    # High Quality (Slower)
    "4": {
        "name": "FLUX-schnell",
        "id": "flux-schnell",
        "time": "60-90s",
        "quality": "⭐⭐⭐⭐⭐",
        "note": "Best photorealism (slow on CPU)",
        "script": "generate_2025_models.py",
    },
    "5": {
        "name": "SD 3.5 Large",
        "id": "sd-3.5-large",
        "time": "40-70s",
        "quality": "⭐⭐⭐⭐⭐",
        "note": "Excellent typography (slow on CPU)",
        "script": "generate_2025_models.py",
    },

    # Quantized (Large models, less RAM)
    "6": {
        "name": "Qwen-Image (4-bit)",
        "id": "qwen-image",
        "time": "80-120s",
        "quality": "⭐⭐⭐⭐⭐",
        "note": "Best text rendering! (needs 12-16GB RAM)",
        "script": "generate_quantized.py",
        "quantize": "4bit",
    },
    "7": {
        "name": "FLUX-schnell (4-bit)",
        "id": "flux-schnell",
        "time": "45-65s",
        "quality": "⭐⭐⭐⭐",
        "note": "Photorealism, less RAM (needs 8GB+)",
        "script": "generate_quantized.py",
        "quantize": "4bit",
    },
    "8": {
        "name": "SD 3.5 Large (4-bit)",
        "id": "sd-3.5-large",
        "time": "30-50s",
        "quality": "⭐⭐⭐⭐",
        "note": "Great quality, low RAM (needs 8GB)",
        "script": "generate_quantized.py",
        "quantize": "4bit",
    },

    # Baseline
    "9": {
        "name": "SD 1.5 (Your Old Model)",
        "id": "sd-1.5",
        "time": "50-60s",
        "quality": "⭐⭐",
        "note": "For comparison only (outdated)",
        "script": "generate_from_prompt.py",
    },

    # ========================================
    # NSFW-CAPABLE (UNCENSORED COMMUNITY MODELS)
    # ========================================
    "10": {
        "name": "Realistic Vision v5.1 [NSFW]",
        "id": "realistic-vision-nsfw",
        "time": "40-60s",
        "quality": "⭐⭐⭐⭐",
        "note": "Photorealistic, uncensored community model",
        "script": "generate_2025_models.py",
        "nsfw": True,
    },
    "11": {
        "name": "DreamShaper [NSFW]",
        "id": "dreamshaper-nsfw",
        "time": "40-60s",
        "quality": "⭐⭐⭐⭐",
        "note": "Versatile realistic, uncensored",
        "script": "generate_2025_models.py",
        "nsfw": True,
    },
    "12": {
        "name": "Anything V5 (Anime) [NSFW]",
        "id": "anything-v5-nsfw",
        "time": "40-60s",
        "quality": "⭐⭐⭐⭐",
        "note": "Anime specialist, uncensored",
        "script": "generate_2025_models.py",
        "nsfw": True,
    },
    "13": {
        "name": "Deliberate [NSFW]",
        "id": "deliberate-nsfw",
        "time": "40-60s",
        "quality": "⭐⭐⭐⭐",
        "note": "Artistic/painted style, uncensored",
        "script": "generate_2025_models.py",
        "nsfw": True,
    },
}


def print_models():
    """Display available models"""
    print("\n" + "=" * 80)
    print("AVAILABLE MODELS")
    print("=" * 80)

    print("\n🚀 FAST (CPU-FRIENDLY):")
    print("-" * 80)
    for key in ["1", "2", "3"]:
        model = MODELS[key]
        print(f"  [{key}] {model['name']:<25} {model['quality']:<15} {model['time']:<12}")
        print(f"      {model['note']}")

    print("\n🎨 HIGH QUALITY (SLOWER ON CPU):")
    print("-" * 80)
    for key in ["4", "5"]:
        model = MODELS[key]
        print(f"  [{key}] {model['name']:<25} {model['quality']:<15} {model['time']:<12}")
        print(f"      {model['note']}")

    print("\n💾 QUANTIZED (LARGE MODELS, LESS RAM):")
    print("-" * 80)
    for key in ["6", "7", "8"]:
        model = MODELS[key]
        print(f"  [{key}] {model['name']:<25} {model['quality']:<15} {model['time']:<12}")
        print(f"      {model['note']}")

    print("\n📊 BASELINE:")
    print("-" * 80)
    model = MODELS["9"]
    print(f"  [9] {model['name']:<25} {model['quality']:<15} {model['time']:<12}")
    print(f"      {model['note']}")

    print("\n🔞 NSFW-CAPABLE (UNCENSORED COMMUNITY MODELS):")
    print("-" * 80)
    for key in ["10", "11", "12", "13"]:
        model = MODELS[key]
        print(f"  [{key}] {model['name']:<25} {model['quality']:<15} {model['time']:<12}")
        print(f"      {model['note']}")

    print("\n" + "=" * 80)


def select_model():
    """Let user select a model"""
    print_models()

    print("\n💡 RECOMMENDATIONS:")
    print("  • First time?           Try [1] SDXL Turbo")
    print("  • Need text in image?   Try [6] Qwen-Image or [2] SD 3 Medium")
    print("  • Want best quality?    Try [4] FLUX-schnell")
    print("  • Want fastest?         Try [3] Waifu Diffusion")
    print("  • Low RAM (8GB)?        Try [8] SD 3.5 Large (4-bit)")
    print("  • NSFW photorealistic?  Try [10] Realistic Vision")
    print("  • NSFW anime?           Try [12] Anything V5")

    while True:
        choice = input("\nSelect model [1-13, or 'q' to quit]: ").strip()

        if choice.lower() == 'q':
            print("Goodbye!")
            sys.exit(0)

        if choice in MODELS:
            return MODELS[choice]

        print("Invalid choice! Please enter 1-9.")


def generate_image(model, prompt, steps=None):
    """Generate an image with the selected model"""
    script = model["script"]
    model_id = model["id"]

    # Build command
    cmd = ["python", script, prompt, "--model", model_id]

    # Add quantization if needed
    if "quantize" in model:
        cmd.extend(["--quantize", model["quantize"]])

    # Add steps if specified
    if steps:
        cmd.extend(["--steps", str(steps)])

    print(f"\n{'='*80}")
    print(f"GENERATING WITH {model['name']}")
    print(f"{'='*80}")
    print(f"Estimated time: {model['time']}")
    print(f"Quality:        {model['quality']}")
    print(f"{'='*80}\n")

    # Check if script exists
    if not Path(script).exists():
        print(f"❌ Error: {script} not found!")
        print(f"   Make sure all scripts are in the current directory.")
        return False

    # Run the command
    try:
        result = subprocess.run(cmd)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation cancelled!")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def interactive_mode():
    """Main interactive loop"""
    print("\n" + "=" * 80)
    print("SIMPLE IMAGE GENERATOR")
    print("=" * 80)
    print("Generate AI images from text prompts!")
    print("Type 'quit' or 'exit' at any time to stop.")
    print("=" * 80)

    # Select model once
    model = select_model()

    print(f"\n✓ Selected: {model['name']}")
    print(f"  Quality: {model['quality']}")
    print(f"  Time: {model['time']}")
    print(f"  Note: {model['note']}")

    # Check if quantized model needs bitsandbytes
    if "quantize" in model:
        try:
            import bitsandbytes
        except ImportError:
            print("\n⚠️  WARNING: This model requires bitsandbytes")
            print("   Install with: pip install bitsandbytes")
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Please install bitsandbytes first.")
                return

    # Generation loop
    while True:
        print("\n" + "=" * 80)
        prompt = input("\nEnter your prompt (or 'change' to switch model, 'quit' to exit): ").strip()

        if not prompt:
            continue

        if prompt.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break

        if prompt.lower() == 'change':
            model = select_model()
            print(f"\n✓ Switched to: {model['name']}")
            continue

        # Ask for steps with helpful guidance
        print("\n" + "-" * 80)
        print("STEPS GUIDE:")
        print("  • Steps = number of refinement iterations")
        print("  • More steps = better quality BUT slower")
        print("  • Each model has an optimal range")
        print("-" * 80)

        # Show model-specific guidance
        if model["id"] in ["sdxl-turbo", "sdxl-lightning"]:
            print(f"💡 {model['name']} is optimized for LOW steps!")
            print(f"   Recommended: 1-4 steps (default is best!)")
            print(f"   ⚠️  Using 20+ steps will make it WORSE!")
        elif model["id"] == "flux-schnell":
            print(f"💡 {model['name']} is optimized for 4 steps")
            print(f"   Recommended: 4 steps (default)")
        elif model["id"] == "qwen-image":
            print(f"💡 {model['name']} works best with 28-40 steps")
            print(f"   Recommended: 28 (default) for good quality")
            print(f"   Optional: 40 for best quality (+30s)")
        else:
            print(f"💡 {model['name']} - Standard model")
            print(f"   Recommended: 20-30 steps (default: 20)")
            print(f"   Quick test: 10-15 steps")
            print(f"   High quality: 40-50 steps (+20-40s)")

        print("-" * 80)
        print("Common values: 1, 4, 10, 20, 30, 40, 50")
        print("OR just press Enter to use optimal default!")
        print("-" * 80)

        steps_input = input(f"Enter steps (or press Enter for default): ").strip()
        steps = int(steps_input) if steps_input.isdigit() else None

        # Show what was chosen
        if steps:
            print(f"\n✓ Using {steps} steps")
            # Warn if potentially suboptimal
            if model["id"] in ["sdxl-turbo", "sdxl-lightning"] and steps > 10:
                print(f"⚠️  WARNING: {model['name']} may perform worse with {steps} steps!")
                print(f"   Consider using 1-4 steps instead.")
            elif model["id"] not in ["sdxl-turbo", "sdxl-lightning", "flux-schnell"] and steps < 10:
                print(f"⚠️  WARNING: {steps} steps is low for {model['name']}")
                print(f"   Image may be blurry. Consider 20+ steps.")
        else:
            print(f"\n✓ Using default steps (optimal for this model)")

        # Generate
        success = generate_image(model, prompt, steps)

        if success:
            print("\n✓ Image generated successfully!")
            print("  Check the output/ folder")
        else:
            print("\n⚠️  Generation failed or was cancelled")

        # Ask to continue
        another = input("\nGenerate another image with this model? (y/n): ").strip()
        if another.lower() not in ['y', 'yes', '']:
            print("\nGoodbye!")
            break


def main():
    """Entry point"""
    try:
        interactive_mode()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
