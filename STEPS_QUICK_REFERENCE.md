# Steps - Quick Reference Card

## TL;DR

**Just press Enter!** Defaults are set optimally for each model.

## What Are Steps?

**Steps = How many times the model refines the image**

```
Start: [Random noise]
  ↓
Step 1-5: [Blurry shapes]
  ↓
Step 10-20: [Clear image]
  ↓
Step 30-50: [Polished details]
```

**More steps = Better quality BUT slower!**

## Quick Values

| Steps | Quality | Time | When to Use |
|-------|---------|------|-------------|
| **1-4** | ⭐⭐⭐ | Very fast | Quick previews, fast models only |
| **10-15** | ⭐⭐⭐ | Fast | Testing prompts |
| **20-30** | ⭐⭐⭐⭐ | Medium | **Normal use ← START HERE** |
| **40-50** | ⭐⭐⭐⭐⭐ | Slow | High quality finals |
| **80+** | ⭐⭐⭐⭐⭐ | Very slow | Rarely worth it |

## By Model Type

### Fast Models (Use LOW steps!)

| Model | Optimal Steps | Why |
|-------|---------------|-----|
| SDXL Turbo | **1** | Trained for 1 step! |
| SDXL Lightning | **4** | Optimized for 4 steps |
| FLUX-schnell | **4** | Distilled for 4 steps |

⚠️ **Using 20+ steps with these makes them WORSE!**

### Standard Models (Use 20-30 steps)

| Model | Quick (10-15) | Recommended | Best (40-50) |
|-------|---------------|-------------|--------------|
| SD 1.5 | Blurry | **20-30** ✓ | Polished |
| SDXL | Basic | **30** ✓ | Excellent |
| SD 3 Medium | Usable | **20** ✓ | Great |
| SD 3.5 Large | Decent | **28** ✓ | Superb |

### Quality Models (Use 28-40 steps)

| Model | Min | Recommended | Max Useful |
|-------|-----|-------------|------------|
| Qwen-Image | 20 | **28-35** ✓ | 50 |
| FLUX-dev | 30 | **40** ✓ | 80 |
| DeepFloyd IF | 30 | **50** ✓ | 100 |

## Time Impact (Your CPU)

**Time = Steps × Time-per-step**

Example (SD 3 Medium on CPU):
- 10 steps = ~15 seconds
- 20 steps = ~30 seconds
- 50 steps = ~75 seconds

**Double steps = double time!**

## What You'll See

### SDXL Turbo Example

**Prompt:** "a red apple"

```
1 step  (2s):  Clear apple, good! ← OPTIMAL ✓
4 steps (8s):  Slightly sharper
20 steps (40s): Over-processed, worse! ← DON'T DO THIS ✗
```

### SD 3 Medium Example

**Prompt:** "a red apple"

```
5 steps  (8s):  Blurry blob ← Too few ✗
10 steps (15s): Recognizable, soft edges
20 steps (30s): Sharp, detailed ← GOOD ✓
30 steps (45s): Excellent, polished ← BETTER ✓
50 steps (75s): Slightly better than 30
100 steps (150s): Barely better than 50 ← Waste of time ✗
```

## Common Mistakes

### ❌ MISTAKE 1: Too many steps with fast models
```
Model: SDXL Turbo
Steps: 50
Result: Worse than 1 step! (over-refined)
```

**Fix:** Use 1-4 steps with Turbo/Lightning/Fast models!

### ❌ MISTAKE 2: Too few steps with standard models
```
Model: SD 3.5 Large
Steps: 5
Result: Blurry mess
```

**Fix:** Use 20-30 steps minimum!

### ❌ MISTAKE 3: Using 100+ steps
```
Model: Any
Steps: 100
Result: Barely better than 50, wastes time
```

**Fix:** Stop at 40-50 steps max!

## Interactive Script Guidance

When you run `python generate.py`, you'll see:

```
STEPS GUIDE:
  • Steps = number of refinement iterations
  • More steps = better quality BUT slower
  • Each model has an optimal range
--------------------------------------------------------------------------------

For SDXL Turbo:
💡 SDXL Turbo is optimized for LOW steps!
   Recommended: 1-4 steps (default is best!)
   ⚠️  Using 20+ steps will make it WORSE!

For SD 3 Medium:
💡 SD 3 Medium - Standard model
   Recommended: 20-30 steps (default: 20)
   Quick test: 10-15 steps
   High quality: 40-50 steps (+20-40s)

For Qwen-Image:
💡 Qwen-Image works best with 28-40 steps
   Recommended: 28 (default) for good quality
   Optional: 40 for best quality (+30s)

Common values: 1, 4, 10, 20, 30, 40, 50
OR just press Enter to use optimal default!
```

## Decision Tree

```
Do you need text in the image?
├─ Yes → Qwen-Image (28 steps) or SD 3 Medium (20 steps)
└─ No
    ├─ Need it fast? → SDXL Turbo (1 step)
    ├─ Best quality? → FLUX-schnell (4 steps) or SD 3.5 Large (28 steps)
    └─ Balanced? → SD 3 Medium (20 steps)
```

## Cheat Sheet - Copy This!

```
MODEL                   OPTIMAL STEPS    TIME (CPU)
================================================
SDXL Turbo              1               ~2s
SDXL Lightning          4               ~8s
FLUX-schnell            4               ~15s
Waifu Diffusion         20              ~20s
SD 3 Medium             20              ~30s
SD 3.5 Large            28              ~45s
Qwen-Image (4-bit)      28              ~110s
FLUX-dev                40              ~120s
```

## Your SD 1.5 Comparison

You generated a pig with SD 1.5:
- Steps: 20
- Time: 54 seconds
- Quality: ⭐⭐ OK

Try SDXL Turbo instead:
- Steps: **1** (not 20!)
- Time: ~2-5 seconds
- Quality: ⭐⭐⭐⭐ Much better!

**10× faster AND better quality!**

## Final Recommendation

**90% of the time:** Just press Enter for defaults!

**The other 10%:**
- Quick test → Use half the default (e.g., 10 instead of 20)
- Best quality → Use 1.5× the default (e.g., 30 instead of 20)
- Never use more than 50 steps

---

**See UNDERSTANDING_STEPS.md for detailed explanation!**
