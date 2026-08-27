# tAIkoMapper — Development Direction

**Goal:** MuG-Diffusion-class osu!taiko chart generation with controllable difficulty and
style, trained on Kaggle 2×T4.

**Status assessment:** the architecture is the right one. The wiring between its parts is
not. There are three defects that each independently prevent the model from learning
audio-to-chart alignment, and they are all in the seams between modules that were written
at different times. None of them show up as a crash or as a bad training loss — the loss
will go down happily while the model learns nothing useful. Fix the seams first; do not
spend Kaggle quota on a long run until Gate A below passes.

---

## 1. What exists

**Live path** (latent diffusion, MuG-shaped):

```
.osu ──osu_parser──> TaikoBeatmap ──tensor_repr──> [7, T] chart tensor @20ms
audio ──audio.py───> mel [128, T_mel]

chart ──BeatmapAutoencoder (KL-VAE, 16× down)──> z [16, T/16]
mel   ──MelEncoder1D (multi-scale)────────────> audio_features[]
z + t + audio_features + cond ──TaikoDiffusionUNet──> ε̂
cond = difficulty(SR) + avg_nps + peak_nps + style(4-class) + motif(16-dim)
sampling: DDIM 50 steps + CFG ──> z ──VAE decode──> [7,T] ──> notes ──timing_refine──> .osz
```

**Dead path** (pre-redesign autoregressive transformer, ~1,900 lines):
`taiko/model/model.py`, `taiko/model/decoder.py`, `taiko/data/dataset.py`,
most of `taiko/data/tokenizer.py`, `scripts/train.py`, `scripts/process_dataset.py`,
`tests/test_model.py`, `tests/test_pipeline.py`. Only `OsuTaikoSerializer` is still used.

**Data:** 13,018 filtered taiko difficulties (`taiko_files_filtered.json`), with SR /
ranked status from osu! API v1, plus derived NPS, snap ratios and a 4-class style label.
That is a good corpus — comparable in order of magnitude to what MuG trained on.

---

## 2. Blocking defects (P0 — fix before any real training run)

### 2.1 Audio and chart are misaligned during training

`taiko/data/audio.py:27` — `HOP_LENGTH = 441` @ 22050 Hz = **20 ms per mel frame**
(the module docstring claiming 10 ms is wrong; `tensor_repr.py` correctly documents 20 ms
and asserts mel frame *i* == chart frame *i*).

But `WindowedDataset` assumes mel runs at **2× the chart rate**:

- `preprocessed_dataset.py:66` — `MEL_FRAMES = 36_000  # 2× tensor`
- `preprocessed_dataset.py:297-298` — `mel_start = start * 2`, `mel_end = end * 2`
- returns `mel [128, W*2]` against `tensor [7, W]`

So for a chart window starting at frame `s`, the model is shown the audio starting at
frame `2s` and running twice as long. Every training sample except `start == 0` pairs a
chart with the wrong part of the song. **The model cannot learn onset alignment, and its
loss will still decrease** — it just learns the marginal distribution of taiko rhythms.

`scripts/generate.py:121` already carries the comment *"mel frames == beatmap frames at
20 ms hop (do not divide by 2)"*, so inference and training also disagree with each other.

**Fix:** pick one frame rate and enforce it in one place.
Recommended: keep 20 ms, make the dataset 1:1, and add a startup assertion
`mel.shape[1] ≈ tensor.shape[1]` that fails loudly. (A 10 ms mel is defensible for onset
precision, but it doubles the mel corpus to ~20 GB and doubles the audio-encoder cost —
treat it as a later experiment, not the unblock.)

### 2.2 The audio encoder's resolutions do not match the U-Net's

`AudioConcatBlock` (`unet.py:433`) reconciles the two with
`F.interpolate(audio, size=x.shape[2], mode="linear")`.

At U-Net level 0 the latent is `W/16` frames (93 for a 30 s window) while
`audio_features[0]` is at full mel resolution (1500 frames). Linear interpolation from
1500 → 93 **point-samples** the audio: roughly 15 of every 16 mel frames are discarded,
and the surviving ones are chosen by an arbitrary phase. Onset energy — the single most
important signal for chart generation — is thrown away at exactly the level where the
U-Net decides where notes go.

In MuG the audio encoder is built so that each of its levels lands on the corresponding
latent resolution. Here it starts at mel resolution and never gets there.

**Fix:** give `MelEncoder1D` a stem that strides down to latent resolution before level 0,
then one level per U-Net level:

```
mel [128, T]                          T frames @ 20 ms
  └─ stem: 4 × strided conv  ──────>  [C, T/16]   ← latent resolution, learned pooling
       ├─ level 0 blocks ──────────>  f0 [C0, T/16 ]   → U-Net level 0
       ├─ down → level 1 ──────────>  f1 [C1, T/32 ]   → U-Net level 1
       ├─ down → level 2 ──────────>  f2 [C2, T/64 ]   → U-Net level 2
       └─ down → level 3 ──────────>  f3 [C3, T/128]   → U-Net level 3
```

Keep `AudioConcatBlock` as a ±1-frame safety net, but it should become a no-op in the
normal case. This is the highest-leverage change in the whole plan after 2.1.

### 2.3 Drumroll durations are nonsense

`osu_parser.py:288` reads slider `parts[7]` as an end time in ms. In the osu! file format
a slider line is
`x,y,time,type,hitSound,curveType|points,slides,length,edgeSounds,edgeSets,hitSample` —
so `parts[7]` is **`length`, in osu! pixels**. The correct duration is:

```
duration_ms = length / (SliderMultiplier * 100 * SV) * beatLength * slides
SV = -100 / green_line_beatLength   (1.0 when no active green line)
```

Every drumroll in the corpus therefore has a fabricated end time, `CH_ROLL` supervision is
garbage, and `TaikoBeatmap.duration_ms` (which keys off the last note's `end_time`) is
wrong whenever a map ends on a roll. Spinner/denden parsing (`parts[5]`) is correct.

**Fix:** compute roll duration properly, which requires walking timing points to find the
active red line + green line at each slider's time. Then regenerate all tensors.

---

## 3. Structural gaps vs. MuG-Diffusion

These are not bugs; they are the difference between "it produces notes" and "it produces
charts someone would play".

### 3.1 The beat grid is an output; it should be an input

`CH_BEAT` is channel 6 of the generated tensor, weighted 0.5 in the VAE loss. At inference
the model must hallucinate a tempo from noise, and `timing_refine` then tries to reverse
engineer BPM and offset from the notes it produced.

Tempo is *known* at generation time — the user has it, or `beat_snap.detect_bpm` finds it.
Make it conditioning:

- Remove `CH_BEAT` from the generated channels (VAE models **6** chart channels).
- Add a 3-channel timing stream at chart resolution: `sin(beat_phase)`, `cos(beat_phase)`,
  `downbeat`. Continuous phase gives the network exact sub-frame timing, which a binary
  pulse channel cannot.
- Concatenate it into the U-Net at every level, alongside the audio features.

Payoff: notes land on-grid by construction; `timing_refine` becomes cleanup instead of
rescue; and the user can force a specific BPM/offset.

**This also unlocks the cleanest difficulty control you have.** Condition additionally on a
*permitted-subdivision mask* (`1/1, 1/2, 1/4, 1/6, 1/8, 1/12, 1/16`). A Kantan chart is
literally a chart restricted to 1/1–1/2; an Oni is one that uses 1/4 freely. This is a
control knob MuG does not have and taiko specifically wants.

### 3.2 The motif vector leaks the answer

`_compute_motif` (`preprocessed_dataset.py:104`) computes the 16-dim motif **from the
ground-truth chart window the model is being asked to generate** — IOI histogram, don/kat
split, big-note fraction, local density, all measured on the target.

The model will learn to lean on it hard (it is nearly a summary of the answer), training
loss will look excellent, and at inference — where the user supplies a rough,
hand-specified vector — the conditioning is off-distribution and quality collapses. This is
the classic self-conditioning trap.

Mitigations, in order of importance:

1. **Quantize** each dim to 4–8 buckets, so it is a coarse control, not a copy.
2. **Per-dimension dropout** (~0.3 each) on top of the existing all-or-nothing CFG drop, so
   no single dim can be relied on.
3. Add noise to continuous dims.
4. Verify with an ablation: train briefly, then sample with a *shuffled* motif from another
   map. If quality craters, the model is leaking.

There is also a live inconsistency: `scripts/analyze_motifs.py` defines a **different**
16-dim motif (map-level, different semantics per index) and writes it to the index, where
`WindowedDataset` never reads it. Pick one canonical spec, put it in `taiko/data/`, and use
it in preprocessing, training and inference.

### 3.3 Classifier-free guidance is misconfigured

- `cfg_dropout = 0.5` (`model_config.py:50`). Standard is 0.1–0.2. At 0.5 half of every
  batch carries no conditioning; the conditional branch gets half the gradient it should.
- `unet.py:265` — the "null" condition sets `style = 0`, but **style 0 is a valid class
  ("standard")**. So the unconditional branch *is* the standard-style branch, and CFG at
  inference pushes samples *away from standard style* rather than away from
  unconditionality. Style control will not work correctly until this is fixed.

**Fix:** `n_styles + 1` with a dedicated null index; a learned null embedding for the whole
conditioning vector; `cfg_dropout = 0.15`; then re-tune CFG scale (expect 3–5, not 1.5).

### 3.4 No EMA, no latent scale calibration

Both are cheap and both are near-mandatory for diffusion sample quality.

- **EMA** of U-Net + audio-encoder weights (decay 0.9995), sampled from instead of the raw
  weights. Absent entirely. Typically the single largest visible quality jump per line of
  code in a diffusion codebase.
- **Latent scale.** `AutoencoderConfig.scale = 1.0` (`autoencoder.py:332`) and is never
  calibrated. Stable-Diffusion-family models set `scale = 1/std(z)` measured over the
  training set so the diffusion input has unit variance. If the VAE's latent std is, say,
  3.5, the noise schedule's effective SNR is wrong at every timestep. Measure it after VAE
  training and bake it into the checkpoint.

Also worth taking while you are in there: **cosine beta schedule** and **v-prediction**
instead of linear + ε-prediction. Both are small diffs and both reliably help on sparse
signals.

### 3.5 Inference does not match training

`generate.py` runs the whole song in one pass (a 3-minute song is 9,000 chart frames vs.
the 1,500 seen in training), through an audio encoder containing global
`nn.MultiheadAttention` — 36× the attention cost of training, with different statistics,
and an OOM risk.

**Fix:** generate in overlapping 30 s windows using **MultiDiffusion** — at each DDIM step,
run the U-Net on each window and average the ε-predictions in the overlap regions before
stepping. ~20 lines, exactly matches training conditions, seamless across window
boundaries, and bounded memory for any song length.

---

## 4. Throughput (P1 — this is what makes the schedule feasible)

### 4.1 The second T4 is idle

`train_diffusion.py:270` wraps the model in `nn.DataParallel`, then
`train_diffusion.py:151` and `:320` call `unwrap(model).training_loss(...)` — reaching
*through* the wrapper. DataParallel only scatters on `forward()`, so it never runs. All
training is on `cuda:0`; the second GPU has been idle for every run to date.

**Fix:** move the loss into `TaikoDiffusion.forward()` and call `model(...)`, or (better)
switch to DDP via `mp.spawn`. Expect ~1.7× from DP, ~1.9× from DDP.

### 4.2 fp16 is switched off

`train_diffusion.py:73` — `USE_FP16 = False`. The `GradScaler`/`autocast` plumbing is
already in place. T4s have tensor cores and no bf16; fp16 AMP is the intended path.
**~1.8× on the conv-dominated workload.** (Keep GroupNorm in fp32 — `autocast` does this.)

### 4.3 Profile `p2` computes six levels it then discards

`MelEncoder1D` returns all 10 levels for `p2`, but the U-Net reads
`audio_features[min(lvl, len-1)]` for `lvl` in 0..3 — i.e. levels 0–3 only. **Levels 4–9
receive no gradient and contribute nothing**; they are pure wasted FLOPs.

Worse, `p2`'s `channel_mult = (1,1,1,1,2,2,2,4,4,4)` means its *used* levels are
`[128,128,128,128]`, while `p1`'s are `[128,128,256,512]`:

| profile | encoder levels | delivered to U-Net | dead levels |
|---|---|---|---|
| p1 | 4 | `[128, 128, 256, 512]` | 0 |
| p2 | 10 | `[128, 128, 128, 128]` | **6** |

So `p2` costs strictly more and delivers strictly less audio capacity — and it is the
**default** (`--profile p2`). Switch the default to `p1` immediately; fold the intent
behind `p2` into the redesigned encoder from §2.2.

### 4.4 Data loading will bottleneck the GPU

13k+ individually-compressed `.npz` files, decompressed per sample, with `num_workers=4`,
on Kaggle's disk. Once the GPU work is 4× faster this becomes the wall.

**Fix:** pack once into a few large shards —
mel as an `fp16` memmap (halves size *and* I/O), charts as sparse `int32` `(frame, channel)`
event lists (taiko charts are ~99% zeros; dense storage is pure waste), plus a single
offsets index. Expect a large win and smaller Kaggle dataset uploads.

### 4.5 Two scripts are currently broken on launch

- `train_autoencoder.py:178,185` pass `samples_per_epoch=` to `WindowedDataset`, which has
  no such parameter → `TypeError`. (The concept is worth implementing: `__len__` is
  currently `len(records)`, so one "epoch" is a single random window per map.)
- `generate.py:91` passes `audio_base_channels=` and friends to `TaikoDiffusion.__init__`,
  which now takes `profile=` → `TypeError`. It also never passes `avg_nps`, `peak_nps` or
  `motif`, so inference runs with **zero conditioning** — the null embedding. Difficulty and
  style control are wired end-to-end nowhere right now.

Minor: `python-dotenv` is imported by `preprocess_for_colab.py` but absent from
`requirements.txt`. There is no README.

---

## 5. Evaluation — define "quality" before optimising for it

Nothing in the repo measures chart quality, so "MuG-level" is currently unfalsifiable.
Build this in Phase 0; it is what every later decision is judged against.

Held-out set: ~200 ranked maps with their real audio, never trained on.

| Metric | What it catches | Target |
|---|---|---|
| Onset F1 vs. ground truth (±25 ms) | audio alignment | > 0.55 |
| Snap validity — % notes within 5 ms of a 1/1…1/16 grid | on-grid-ness | > 0.95 |
| SR controllability — Pearson(requested SR, realised SR) | difficulty control | > 0.85 |
| NPS error — \|requested − realised\| avg_nps | density control | < 1.0 |
| Pattern 4-gram KL (don/kat) vs. real maps of same SR | idiom | as low as possible |
| Unplayability rate — sub-30 ms gaps, illegal big-note doubles | basic playability | < 0.5% |
| Don/kat balance, big-note rate | style sanity | in-distribution |

Onset F1 is the one that catches defect 2.1: with misaligned training it will sit near
chance no matter how good the loss curve looks.

Plus a manual gate that no metric replaces: **import 10 generated `.osz` files into osu!
and play them.** Do this at every phase boundary.

---

## 6. Phased plan with gates

Each gate is a hard stop. Do not spend Kaggle quota on the next phase until it passes.

### Phase 0 — Unblock and measure (~1 week, mostly CPU)
1. Delete the dead transformer path; keep `OsuTaikoSerializer`. Write a README.
2. Fix the mel/chart frame-rate contract (§2.1) + add the startup assertion.
3. Fix drumroll durations (§2.3); regenerate all tensors.
4. Fix `train_autoencoder.py` and `generate.py` launch errors (§4.5).
5. Build the evaluation harness (§5) and the data packing (§4.4).
6. Drop maps whose SR is unknown — `preprocess_for_colab.py:557` currently falls back to
   `overall_difficulty`, which is an *accuracy* parameter, not a star rating, and poisons
   the difficulty conditioning.

**Gate 0:** round-trip test passes; a sanity script confirms a chart window and its mel
window cover the same milliseconds of the same song.

### Phase 1 — VAE (~10 GPU-hours)
1. Drop `CH_BEAT` from the modelled channels → 6 chart channels (§3.1).
2. Train, then measure reconstruction at ±1 frame.
3. Calibrate and store the latent scale (§3.4).

**Gate A: VAE onset F1 ≥ 0.98 at ±1 frame, roll start/end within 2 frames.**
If it fails at 16× compression, drop to 8× (`channel_mult = [1,1,2,2]`) and retrain. A
weak VAE puts a hard ceiling on everything downstream — there is no recovering from this
later, so do not proceed on 0.95.

### Phase 2 — Diffusion, correctness (~30 GPU-hours)
1. Redesign the audio encoder pyramid (§2.2). **The most important change in the plan.**
2. Add the timing/beat conditioning stream (§3.1).
3. Fix CFG: null style token, learned null embedding, `cfg_dropout = 0.15` (§3.3).
4. Add EMA; switch to cosine schedule + v-prediction (§3.4).
5. Fix DataParallel/DDP, enable fp16, default to `p1` (§4.1–4.3).
6. Motif de-leaking: quantize + per-dim dropout (§3.2).

**Gate B: onset F1 > 0.40 on held-out audio.** This is the alignment gate — it proves the
model is listening to the music rather than emitting plausible-looking rhythms. A model
that fails here will never be fixed by more steps.

### Phase 3 — Scale up (~150–250 GPU-hours, 5–8 weeks of Kaggle quota)
1. Long run on ranked + loved maps. Bulletproof resume-from-checkpoint — 12 h session
   caps mean roughly 20 restarts over this phase.
2. Add a `quality` conditioning scalar (ranked vs. unranked), so the VAE can still learn
   chart syntax from everything while the sampler can be asked for ranked-grade output.
   Same trick as aesthetic-score conditioning in Stable Diffusion.
3. MultiDiffusion windowed inference (§3.5).
4. Tune CFG scale and DDIM steps against the §5 metrics.

**Gate C: all §5 targets met, and the 10 hand-played maps are playable.**

### Phase 4 — Controllability (the actual product)
1. **Subdivision-mask conditioning** — the honest way to express Kantan→Inner Oni (§3.1).
2. **Style presets** — a named library of motif vectors ("1/4 stream", "big-note heavy",
   "1/6 tech", "simple") so users never hand-author 16 floats.
3. **Style transfer from a reference chart** — extract the motif from any `.osu` and
   generate in that style. Nearly free given the encoder, and the most compelling feature
   in the whole plan.
4. **Difficulty envelope** — a per-section density curve instead of one global number, so
   kiai sections get denser. The window-local conditioning already supports this.
5. Gradio UI (already in `requirements.txt`, never built).

---

## 7. Recommended decisions, condensed

| Decision | Recommendation | Why |
|---|---|---|
| Frame rate | Keep 20 ms, enforce 1:1 mel↔chart | Unblocks today; 10 ms doubles a 20 GB corpus |
| VAE compression | Start 16×, drop to 8× if Gate A fails | 8× = 160 ms latent stride, under one beat |
| Beat grid | Conditioning **input**, 3 channels (sin/cos phase + downbeat) | Tempo is known at generation time |
| Audio encoder | Stem strides to latent resolution; one level per U-Net level | Stops discarding 15/16 of onset frames |
| Profile default | `p1`, not `p2` | `p2` has 6 dead levels and less delivered capacity |
| Parameterization | v-prediction + cosine schedule | Better on sparse signals |
| EMA | 0.9995, sample from EMA | Largest quality/effort ratio available |
| CFG | null style token, dropout 0.15, scale 3–5 | Current setup guides away from "standard style" |
| Long-range | Keep the depthwise-conv S4 stand-in; add self-attention at the coarsest latent level | T ≈ 93 there — global context, negligible cost |
| Multi-GPU | Fix DP now, DDP if time permits | The second T4 has never been used |
| Inference | MultiDiffusion over 30 s windows | Matches training; any song length |
| Style control | Motif vector + presets + reference-chart extraction | The 4-class label is too coarse to be a product |

---

## 8. Realistic expectations

MuG-Diffusion's quality rests on three things this project can match — latent diffusion
over a chart array, tight multi-scale audio conditioning, and a large ranked corpus — and
one it currently cannot: a mature, debugged data pipeline. 13k taiko maps is a good corpus.
2×T4 is slow but sufficient *if* §4 is fixed; it is not sufficient at the current
one-GPU-fp32-with-dead-FLOPs throughput.

The honest risk ranking:

1. **§2.1 + §2.2 (alignment)** — if these are not fixed, no amount of training produces a
   chart that follows the music. They are also the two most likely to be dismissed as
   "cosmetic" because nothing crashes and the loss looks fine.
2. **§3.2 (motif leakage)** — will look excellent in training and disappoint at inference.
3. **Compute budget** — Phase 3 is 5–8 weeks of Kaggle quota. Plan around interruption
   from day one; do not discover the resume path is broken at hour 11 of a session.

Everything else on this list is a straightforward fix.
