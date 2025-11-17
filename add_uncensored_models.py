#!/usr/bin/env python3
"""
Example: How to add uncensored models to the generator

This shows how to use community NSFW-capable models.
"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Popular uncensored models from Hugging Face
UNCENSORED_MODELS = {
    "realistic-vision": {
        "id": "SG161222/Realistic_Vision_V5.1_noVAE",
        "note": "Photorealistic, very popular on CivitAI",
        "style": "Photorealistic",
    },
    "dreamshaper": {
        "id": "Lykon/DreamShaper",
        "note": "Versatile, good for various styles",
        "style": "Versatile",
    },
    "deliberate": {
        "id": "XpucT/Deliberate",
        "note": "Artistic, painterly style",
        "style": "Artistic/Painted",
    },
    "anything-v5": {
        "id": "stablediffusionapi/anything-v5",
        "note": "Anime specialist",
        "style": "Anime",
    },
}

def generate_with_uncensored_model(
    model_name: str,
    prompt: str,
    steps: int = 20,
    width: int = 512,
    height: int = 512,
):
    """
    Generate image with uncensored model

    Args:
        model_name: Key from UNCENSORED_MODELS dict
        prompt: Your text prompt
        steps: Denoising steps
        width, height: Image dimensions
    """

    if model_name not in UNCENSORED_MODELS:
        raise ValueError(f"Unknown model. Available: {list(UNCENSORED_MODELS.keys())}")

    model_config = UNCENSORED_MODELS[model_name]
    model_id = model_config["id"]

    print(f"\nLoading: {model_name}")
    print(f"Model ID: {model_id}")
    print(f"Style: {model_config['style']}")
    print(f"Note: {model_config['note']}")
    print(f"\nThis may take a minute on first run (downloads ~4GB)...\n")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,  # No post-generation filtering
    ).to(device)

    # Generate
    print(f"Generating: '{prompt}'")
    print(f"Steps: {steps}, Size: {width}×{height}\n")

    result = pipe(
        prompt,
        num_inference_steps=steps,
        width=width,
        height=height,
    )

    image = result.images[0]

    # Save
    filename = f"output/{model_name}_{prompt[:30].replace(' ', '_')}.png"
    image.save(filename)
    print(f"\n✓ Saved: {filename}")

    return image


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("UNCENSORED MODEL EXAMPLE")
    print("=" * 80)
    print("\nAvailable models:")
    for name, config in UNCENSORED_MODELS.items():
        print(f"  • {name}: {config['style']} - {config['note']}")

    print("\n" + "=" * 80)
    print("\nTo use:")
    print("  1. Edit the script below")
    print("  2. Change model_name and prompt")
    print("  3. Run: python add_uncensored_model_example.py")
    print("\n" + "=" * 80)

    # EDIT THESE:
    model_name = "realistic-vision"  # or "dreamshaper", "deliberate", "anything-v5"
    prompt = "a portrait of a woman"  # Your prompt here

    # Uncomment to run:
    # generate_with_uncensored_model(model_name, prompt, steps=20)

    print("\nScript ready! Uncomment the last line to generate.")
