# Taiko Diffusion Pivot Notes

## Why Pivot Away From the Current Autoregressive Transformer?

Current approach:

```text
Audio
→ AudioEncoder
→ Transformer Decoder
→ Token Sequence
→ .osu
```

Current token format:

```text
TIME_5 HIT_DON TIME_2 HIT_KAT ...
```

This behaves like a language model (GPT-style).

### Problems With This Approach

#### 1. Exposure Bias

During training:

* model sees ground truth previous tokens

During inference:

* model sees its own generated tokens

Errors accumulate over long maps.

This causes:

* unstable rhythm
* pattern collapse
* incoherent long-term structure

---

#### 2. Sequence Length Explosion

Taiko maps become extremely long token sequences.

Example:

* 3 minute map
* 3000–10000+ tokens

Transformer attention complexity:

O(n²)

This becomes expensive very quickly.

---

#### 3. Rhythm Is Continuous

Current representation discretizes rhythm into symbolic tokens.

Example:

```text
TIME_13
TIME_14
TIME_15
```

But rhythm is fundamentally continuous.

Diffusion handles continuous structure much better.

---

#### 4. Weak Global Structure

Autoregressive models are good locally but weaker globally.

Hard to maintain:

* phrase structure
* motif consistency
* long-term pattern flow
* musical symmetry

---

# Why MugDiffusion Works Better

MugDiffusion is based on:

# Latent Diffusion Models (LDM)

Similar family to:

* Stable Diffusion
* latent diffusion research
* image diffusion pipelines

Core idea:

```text
Beatmap
→ latent representation
→ diffusion in latent space
→ reconstructed beatmap
```

Instead of:

* predicting next token

It:

* denoises an entire latent beatmap representation

This is fundamentally more suitable for rhythm games.

---

# Major Advantages of Diffusion for Beatmaps

## 1. Global Coherence

Diffusion sees:

* entire map structure simultaneously

This improves:

* consistency
* phrase-level structure
* pattern evolution
* musical flow

---

## 2. Continuous Representation

Diffusion naturally models:

* density
* rhythm flow
* spacing
* phrase textures

instead of symbolic token jumps.

---

## 3. Better Long-Map Stability

Autoregressive:

* errors accumulate

Diffusion:

* iterative refinement

Much more stable for long generation.

---

## 4. Latent Compression

Instead of enormous token sequences:

```text
TIME_5 HIT_DON ...
```

MugDiffusion compresses maps into latent tensors.

Benefits:

* smaller representation
* better memory efficiency
* easier global reasoning
* scalable generation

---

# MOST IMPORTANT INSIGHT

Beatmaps are closer to:

```text
continuous temporal-spatial fields
```

than natural language.

Diffusion models are naturally better suited for this kind of structure.

---

# Recommended New Representation

## OLD Representation

```text
TIME_5 HIT_DON TIME_2 HIT_KAT
```

## NEW Representation

Tensor-based representation:

```python
shape = [channels, time]
```

Example channels:

| Channel | Meaning |
| ------- | ------- |
| don     |         |
| kat     |         |
| big_don |         |
| big_kat |         |
| roll    |         |
| denden  |         |

Example:

* 10ms resolution
* 100 seconds song

```text
100000ms / 10ms = 10000 timesteps
```

Tensor shape:

```python
[6, 10000]
```

This is diffusion-friendly.

---

# Core New Pipeline

## Final Target Architecture

```text
Audio
→ Mel Spectrogram
→ Audio Encoder

Beatmap
→ Beatmap Tensor
→ Autoencoder
→ Latent z

Latent z + Audio Conditioning
→ Diffusion UNet
→ Generated Latent

Generated Latent
→ Decoder
→ Beatmap Tensor
→ .osu Serializer
```

---

# WHAT TO KEEP FROM CURRENT PROJECT

Keep:

* osu_parser.py
* serializer
* metadata extraction
* dataset crawling
* taiko semantics
* conditioning ideas
* audio preprocessing
* CFG concepts

Do NOT throw these away.

---

# WHAT TO REMOVE

Remove:

* autoregressive token decoder
* GPT-style generation loop
* symbolic time-token representation

---

# THE REAL ROADMAP

# Phase 1 — Beatmap Tensor Representation

Build:

```python
beatmap_to_tensor()
tensor_to_beatmap()
```

Goal:

```text
.osu
→ tensor
→ reconstructed .osu
```

with minimal loss.

THIS IS THE MOST IMPORTANT FIRST STEP.

---

# Phase 2 — Audio Pipeline

Build:

* mel spectrogram extraction
* precise audio-map alignment

Create:

```text
taiko/data/audio.py
```

Functions:

* audio_to_mel()
* align_audio_and_map()

---

# Phase 3 — Dataset Pipeline

Training samples should look like:

```python
{
    "mel": mel_tensor,
    "beatmap": beatmap_tensor,
    "metadata": ...
}
```

Use chunk training:

* 8-second windows
* 16-second windows

Do NOT start with full songs.

---

# Phase 4 — Beatmap Autoencoder

Goal:

```text
Beatmap Tensor
→ latent
→ reconstructed Beatmap Tensor
```

This is required BEFORE diffusion.

Recommended architecture:

* 1D CNN
* ResNet blocks
* downsampling encoder
* upsampling decoder

Possible losses:

* BCE loss
* focal loss
* temporal consistency losses later

Target:

* 95%+ reconstruction accuracy

---

# Phase 5 — Latent Space Inspection

Visualize latent space:

* PCA
* t-SNE
* UMAP

Check:

* similar maps cluster together
* difficulties cluster together
* styles cluster together

If latent space is bad:
diffusion will fail.

---

# Phase 6 — Audio Conditioning Encoder

Input:

* mel spectrogram

Output:

* audio embeddings

Possible architectures:

* CNN
* Transformer
* Conformer

Output shape:

```python
[B, T, D]
```

---

# Phase 7 — Diffusion Model

Build:

* latent diffusion
* UNet denoiser

Input:

* noisy latent z_t
* audio conditioning

Output:

* predicted noise
  or
* denoised latent

UNet is recommended over transformers initially.

---

# Phase 8 — Diffusion Training

Train:

```text
clean latent
→ add noise
→ predict denoising
```

Learn:

* DDPM
* DDIM
* cosine schedules
* CFG guidance

---

# Phase 9 — CFG (Classifier-Free Guidance)

Condition on:

* difficulty
* density
* map style later

This improves controllability.

---

# Phase 10 — Decode Back Into .osu

Pipeline:

```text
generated latent
→ decoder
→ beatmap tensor
→ .osu serializer
```

---

# Phase 11 — Postprocessing

Very important.

Need cleanup systems for:

* impossible overlaps
* timing jitter
* malformed rolls
* density spikes
* invalid note states

---

# Phase 12 — Dataset Quality

Dataset quality matters enormously.

Prefer:

* ranked maps
* loved maps
* high-quality taiko maps

Avoid:

* joke maps
* low-quality timing
* gimmick-heavy maps initially

Dataset quality heavily determines final output quality.

---

# Phase 13 — Evaluation System

Create metrics:

* note density
* timing alignment
* rhythm entropy
* repetition statistics
* variance metrics

Human evaluation is also critical.

---

# Important Long-Term Truths

## 1. Representation Is Everything

The representation choice determines:

* trainability
* coherence
* scalability
* quality ceiling

This is the most important architectural decision.

---

## 2. Autoencoder Quality Determines Diffusion Quality

Bad latent space = bad diffusion results.

The autoencoder stage is foundational.

---

## 3. Dataset Quality Eventually Dominates Architecture

Even perfect architecture struggles with poor maps.

---

## 4. Diffusion Is Much Harder Engineering

Expect:

* higher VRAM usage
* slower training
* harder debugging
* more infrastructure complexity

But:

* much higher quality ceiling

---

# Immediate Next Step

FIRST TASK:

Build:

```python
beatmap_to_tensor()
tensor_to_beatmap()
```

Milestone:

```text
.osu
→ tensor
→ reconstructed .osu
```

with:

* minimal timing loss
* stable note reconstruction
* stable rolls/denden
* reversible conversion

Once this works:
the MugDiffusion pivot officially begins.