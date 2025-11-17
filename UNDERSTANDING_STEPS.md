# Understanding Steps in Image Generation

## What Are "Steps"?

**Steps** = Number of denoising iterations the model runs

Think of it like this:
1. Start with **random noise** (TV static)
2. Each **step** refines the noise → closer to your prompt
3. After N steps → final image

## Visual Example

```
Step 1:  [████████████] Random noise
Step 5:  [████▓▓▓▓░░░░] Blurry shapes visible
Step 10: [████▓▓░░    ] Basic forms clear
Step 20: [████░       ] Details emerging
Step 50: [████        ] Fine details, polished
```

More steps = more refinement = better quality (but slower!)

## Common Values & Impact

### Typical Ranges

| Steps | Quality | Use Case | Time (on CPU) |
|-------|---------|----------|---------------|
| **1-4** | ⭐⭐⭐ Basic | Ultra-fast drafts | 1-10s |
| **10-20** | ⭐⭐⭐⭐ Good | Balanced (recommended) | 20-60s |
| **30-50** | ⭐⭐⭐⭐⭐ Great | High quality | 60-120s |
| **50+** | ⭐⭐⭐⭐⭐ Excellent | Diminishing returns | 120s+ |

### The Sweet Spot

**Most models:** **20-30 steps** = best quality/speed tradeoff

## Model-Specific Recommendations

### Fast Models (Optimized for Few Steps)

| Model | Optimal Steps | Why |
|-------|---------------|-----|
| **SDXL Turbo** | **1-4** | Trained for 1-step! More steps = worse! |
| **SDXL Lightning** | **4-8** | Optimized for 4 steps |
| **LCM models** | **4-8** | Latency Consistency Models |
| **FLUX-schnell** | **4** | Distilled for 4 steps |

⚠️ **Important:** Don't use 20+ steps with these! They're trained for low steps.

### Standard Models (Traditional)

| Model | Min Steps | Recommended | Max Useful |
|-------|-----------|-------------|------------|
| **SD 1.5** | 15 | **20-30** | 50 |
| **SDXL** | 20 | **30-40** | 80 |
| **SD 3 Medium** | 15 | **20-30** | 50 |
| **SD 3.5 Large** | 20 | **28-40** | 60 |
| **Qwen-Image** | 20 | **28-35** | 50 |
| **FLUX-dev** | 20 | **30-50** | 100 |

### Quality Models (Need More Steps)

| Model | Min Steps | Recommended | Max Useful |
|-------|-----------|-------------|------------|
| **DeepFloyd IF** | 30 | **50-100** | 150 |
| **HiDream-I1** | 25 | **30-50** | 80 |

## What Happens at Different Step Counts

### 1-4 Steps (Ultra Fast)
**Time:** 1-10s on CPU
**Quality:** Basic shapes, colors correct, details blurry
**Use for:**
- Quick previews
- Concept testing
- Fast models (SDXL Turbo, Lightning)

**Example:**
```
Prompt: "a red apple"
Result: Round red blob, recognizable as apple, but blurry
```

### 10-15 Steps (Fast)
**Time:** 20-40s on CPU
**Quality:** Clear subject, basic details, some artifacts
**Use for:**
- Rapid iteration
- Testing prompts
- When time matters

**Example:**
```
Prompt: "a red apple"
Result: Clear apple, basic highlights, slightly soft edges
```

### 20-30 Steps (Recommended)
**Time:** 40-80s on CPU
**Quality:** Good details, clean edges, professional look
**Use for:**
- Final images
- Most use cases
- Best quality/time balance

**Example:**
```
Prompt: "a red apple"
Result: Sharp apple, realistic shine, stem details, shadows
```

### 40-50 Steps (High Quality)
**Time:** 80-120s on CPU
**Quality:** Excellent details, very polished
**Use for:**
- Professional work
- When quality is critical
- Large prints

**Example:**
```
Prompt: "a red apple"
Result: Photorealistic, skin texture, perfect lighting, tiny imperfections
```

### 80+ Steps (Diminishing Returns)
**Time:** 160s+ on CPU
**Quality:** Marginal improvement over 50 steps
**Use for:**
- Rarely worth it
- Only for specific cases

**Example:**
```
Prompt: "a red apple"
Result: Slightly better than 50 steps, but you might not notice
```

## The Math: Steps vs Time

**Time scales linearly with steps:**

```
Time = Steps × Time-per-step

Example (SDXL on CPU):
- 1 step:  ~2s  (2s × 1)
- 10 steps: ~20s (2s × 10)
- 20 steps: ~40s (2s × 20)
- 50 steps: ~100s (2s × 50)
```

**Double the steps = double the time!**

## Common Mistakes

### ❌ Using 50 steps with SDXL Turbo
```bash
# WRONG - Turbo is trained for 1 step!
python generate.py
Select: [1] SDXL Turbo
Steps: 50  ← BAD! Image will look worse!
```

**Correct:**
```bash
Select: [1] SDXL Turbo
Steps: 1   ← Good! (or just press Enter for default)
```

### ❌ Using 5 steps with SDXL
```bash
# WRONG - SDXL needs 20+ steps
python generate.py
Select: [5] SD 3.5 Large
Steps: 5  ← Too few! Image will be blurry
```

**Correct:**
```bash
Select: [5] SD 3.5 Large
Steps: 28  ← Good! (or press Enter for default)
```

### ❌ Using 100 steps for quick tests
```bash
# WRONG - Wasting time!
python generate.py
Select: [2] SD 3 Medium
Steps: 100  ← Overkill! No visible improvement over 30
```

**Correct:**
```bash
Select: [2] SD 3 Medium
Steps: 20-30  ← Optimal! (or press Enter for default)
```

## Quick Reference Card

### What Steps Should I Use?

**Just press Enter!** The defaults are set optimally for each model.

**Or use this guide:**

| Your Goal | Steps |
|-----------|-------|
| Quick test/preview | 10-15 |
| **Normal use (recommended)** | **20-30** |
| High quality | 40-50 |
| Professional/print | 50-60 |

**Special cases:**
- SDXL Turbo: **1** (or 1-4)
- SDXL Lightning: **4** (or 4-8)
- FLUX-schnell: **4**
- Qwen-Image: **28** (default)

## Performance Impact on Your CPU

Based on your SD 1.5 baseline (54s for 20 steps):

### SDXL Turbo (Fast Model)

| Steps | Time | Quality |
|-------|------|---------|
| 1 | ~2s | ⭐⭐⭐ Good enough! |
| 4 | ~8s | ⭐⭐⭐⭐ Best for this model |
| 20 | ~40s | ⭐⭐ Worse! (over-refined) |

### SD 3 Medium (Standard Model)

| Steps | Time | Quality |
|-------|------|---------|
| 10 | ~15s | ⭐⭐⭐ Usable |
| 20 | ~30s | ⭐⭐⭐⭐ Recommended |
| 30 | ~45s | ⭐⭐⭐⭐⭐ Best |
| 50 | ~75s | ⭐⭐⭐⭐⭐ Slight improvement |

### Qwen-Image (Quality Model)

| Steps | Time | Quality |
|-------|------|---------|
| 20 | ~80s | ⭐⭐⭐⭐ Decent |
| 28 | ~110s | ⭐⭐⭐⭐⭐ Recommended (default) |
| 40 | ~160s | ⭐⭐⭐⭐⭐ Excellent |
| 50 | ~200s | ⭐⭐⭐⭐⭐ Marginal gain |

## Visual Comparison (Same Prompt, Different Steps)

**Prompt:** "a photorealistic cat portrait"

**SD 3 Medium (CPU):**

```
5 steps   (15s):  [Blurry cat, basic shapes]
10 steps  (30s):  [Clear cat, some details, soft]
20 steps  (60s):  [Sharp cat, fur texture visible, good]  ← Sweet spot!
50 steps  (150s): [Very detailed, whiskers sharp, excellent]
100 steps (300s): [Barely better than 50, not worth it]
```

**SDXL Turbo (CPU):**

```
1 step   (2s):   [Good cat, slightly soft but fast!]  ← Optimal!
4 steps  (8s):   [Slightly better than 1]
20 steps (40s):  [Worse than 1 step! Over-processed]  ← Don't do this!
```

## The FLOPs Impact

Remember from `pixel_map_efficiency.py`:

**Total FLOPs = FLOPs-per-step × Number of steps**

Example (SD 1.5, 512×512):
- 1 step:  ~25 TFLOPs
- 20 steps: ~500 TFLOPs (what you experienced)
- 50 steps: ~1.25 PFLOPs

**This is why efficient models matter!**

SDXL Turbo at 1 step (~25 TFLOPs) beats SD 1.5 at 20 steps (~500 TFLOPs)
→ **20× less computation for better results!**

## Recommendations by Use Case

### Quick Experiments
```
Model: SDXL Turbo [1]
Steps: 1 (default)
Time: ~2-5s
Result: Fast, good enough
```

### Daily Use (Best Balance)
```
Model: SD 3 Medium [2] or SDXL Turbo [1]
Steps: 20 (or 1 for Turbo)
Time: 30-40s
Result: Great quality, reasonable time
```

### Professional/Final Images
```
Model: FLUX-schnell [4] or SD 3.5 Large [5]
Steps: 30-40
Time: 60-120s
Result: Excellent quality
```

### Text-Heavy Images
```
Model: Qwen-Image (4-bit) [6]
Steps: 28 (default)
Time: 110-140s
Result: Perfect text rendering
```

## TL;DR - Just Remember This

**3 Simple Rules:**

1. **Most models:** Use **20-30 steps** (or just press Enter for defaults!)
2. **Fast models (Turbo, Lightning):** Use **1-4 steps** (they're trained for it!)
3. **More steps = better quality BUT:**
   - Time increases linearly (2× steps = 2× time)
   - Returns diminish after ~30-40 steps
   - Some models get WORSE with too many steps!

**When in doubt:** **Press Enter** for the default! The defaults are set optimally.

---

## Interactive Example

```bash
python generate.py

Select model [1-9]: 1  # SDXL Turbo

Enter prompt: a cat

Number of steps [default, or enter number]:

What to do:
  • Press Enter   → Uses 1 step (optimal for Turbo!) ✅
  • Type "4"      → Uses 4 steps (still good)
  • Type "20"     → Uses 20 steps (TOO MANY, worse result!) ❌
```

```bash
python generate.py

Select model [1-9]: 2  # SD 3 Medium

Enter prompt: a cat

Number of steps [default, or enter number]:

What to do:
  • Press Enter   → Uses 20 steps (optimal!) ✅
  • Type "30"     → Uses 30 steps (even better!)
  • Type "5"      → Uses 5 steps (TOO FEW, blurry!) ❌
```

**Best advice: Just press Enter!** The defaults are chosen for optimal quality/speed.
