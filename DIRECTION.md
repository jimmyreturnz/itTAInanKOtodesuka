# tAIkoMapper — Project Direction

Locked decisions, agreed 2026-08-30. This is the *commitments* document.
`DEVELOPMENT_PLAN.md` is archaeology — why the architecture is shaped the way
it is and which bugs were found getting here. `README.md` is the architecture,
`docs/MANUAL.md` is what to run. This file is what we decided to do next and
why, so a future session does not relitigate it.

---

## The product

A desktop app an osu! player downloads and runs. It serves a GUI on localhost:
drag an audio file in, pick a difficulty and a style, adjust the map's
metadata, press Generate, get a `.osz` to drop into osu!. Under it, a latent
diffusion model in the shape of MuG-Diffusion, trained on ranked taiko maps.

Not a hosted service. Not a research artifact. The thing being optimised is
"a player generates a playable map for their own song in under two minutes".

---

## Locked decisions

| # | Decision | Why |
|---|---|---|
| **D1** | **Model first, GUI second.** Next concrete action is packing the real dataset. | Every knob the product needs already exists as a `generate.py` argument. The GUI is the cheap end; weights are the uncertain end. |
| **D2** | **Ranked maps only, both training stages** -- osu! API `approved == 1`, nothing else. Drop approved, qualified, loved, graveyard, wip, pending. ~11,100 charts over ~2,800 songs. | Unranked contains rate-ups — *the same chart against different audio*, which is anti-signal for exactly the alignment Gate B measures. Plus shitmaps and gimmick maps, and loved has gimmicks too, so star rating cannot filter them. Ranked status is the only reliable quality signal available. Loved is a popularity vote, not a quality bar: the first pack's top map was a loved chart named "Too Bad, dont play" at 18.16* on 8.81 nps -- SV abuse inflating the very number you would try to filter it with. |
| **D3** | **Kaggle free tier only.** 2xT4, 30 GPU-h/week, 12 h sessions. No rented GPU. | Rented compute is cheaper in wall-clock but not worth the setup and workaround cost. Consequence: the speed work below is mandatory, not optional, and checkpoint-resume is the most safety-critical path in the repo. |
| **D4** | **Full `tiny` rehearsal on real data before committing to `p1`.** Pack → stage 1 → stage 2 (~20k steps) → Gate B → generate a `.osz` → play it. | Everything verified so far ran on a *synthetic* corpus. Real data brings mel cache misses, charts longer than audio, memmap behaviour on Kaggle disk, DataParallel at real batch sizes, and the resume loop under a real 12 h cutoff. Finding any of those in week three costs far more than a weekend now. Gate A is meaningful at `tiny`; Gate B is advisory only, since capacity may fail it for its own reasons. |
| **D5** | **Keep `--window-frames 1536`. Use `--grad-accum 16`** → effective batch 64. | Effective batch is currently **4** (2/GPU × 2 GPUs), the smallest in any working diffusion recipe; typical is 32–256. Diffusion gradients are unusually noisy because each sample draws a random timestep, so a tiny batch makes every step worth less — which is how a run silently becomes a 200-hour run. Shortening the window was considered and rejected: half the window is half the compute *and* half the audio, so throughput in audio-seconds/sec is a wash, and it would cost the 30 s structural horizon that long-song coherence needs. |
| **D6** | **Timing = pretrained beat tracker + mandatory manual override.** Not a trained model. | osu! timing needs single-digit-ms accuracy to be rankable; SOTA trackers (`beat_this`, madmom DBN, BeatNet) reach ~±20–30 ms. That ceiling is the task, not the architecture, so training our own lands in the same place after months. Three input paths, in order of preference: import from an existing `.osu`; automatic tracker seed; manual entry. **Timing is not the AI's job** — the product is "I have a timed song, write me the notes". |
| **D7** | **Distributable desktop app serving a local web GUI**, like TaikoEditor. Auto-updater for app code and model weights, versioned independently. | Local inference means no GPU bill, no upload path, no auth, and no abuse surface — the entire security story collapses to nothing. A retrained model ships as a ~150 MB asset bump, not a reinstall. |
| **D8** | **Shipped app runs ONNX Runtime with the DirectML execution provider.** PyTorch stays for training and local development. | ~400 MB installer versus ~3 GB, and **any DX12 GPU** works — NVIDIA, AMD, Intel Arc, integrated. Under the PyTorch path every non-NVIDIA user falls back to CPU. Nothing in the model resists export: `s4_block.py` is depthwise conv + SiLU gating, no FFT, no complex tensors. This is a coverage decision, not a speed one. |
| **D9** | **Stay on `p1` (35.4M params). Bank schedule slack as more training runs, not more parameters.** | Nobody has trained this on real data even once; the highest-value use of spare quota is a second and third run informed by what the first got wrong. `p2` is a bad deal regardless — 55M params but it delivers *less* audio capacity to the U-Net than `p1` (`DEVELOPMENT_PLAN.md` §4.3). **Contingency:** if Gate B passes clearly but maps read *bland* rather than mistimed, that is a capacity symptom — define `p3` (192 base channels, ~70M, `p1`'s encoder shape) as a flag, not a redesign. |
| **D10** | **Derive motif presets from the ranked corpus. Delete the hand-authored constants.** | `_preset()` leaves unnamed dimensions at `0.0` and `generate.py:111` passes `ones_like` as the mask, asserting those zeros as confident values. Real maps are never exactly zero there, so every preset lands in a region of motif space training never visits — the exact off-distribution collapse `DEVELOPMENT_PLAN.md` ranks as risk #2, with the per-dimension dropout built to prevent it switched off. Measured centroids are in-distribution in every dimension at once, so the problem stops existing rather than being worked around. |

### Also settled, without argument

- **Green lines are never modelled.** `_red_line_segments` filters on `tp.uninherited`, and `tensor_to_beatmap` only ever constructs `uninherited=True` points. Nothing in the codebase can write a green line. Generated maps ship at SV 1.0. Green lines are still *read* at parse time — correctly, since SV determines a drumroll's true duration from its pixel length.
- **Ranked red lines are ground truth.** This also makes the corpus a free benchmark for choosing a beat tracker: 10,642 maps with correct timing and their audio, no labelling required.
- **Arbitrary song length is already solved.** MultiDiffusion in `taiko/model/sampling.py` — `plan_windows` covers any length, the final window is pulled back to land exactly on the end, raised-cosine tapers blend the overlaps, and all windows denoise one shared latent rather than being stitched afterwards. 10 min = 30,000 frames, under `MAX_FRAMES = 45_000`. Cost is linear in length.
- **Multi-BPM is supported end to end except at inference.** `build_timing_stream` restarts phase at each red line exactly as osu! does, and `osu_writer` emits every timing point. Only three call sites take a scalar BPM (below).
- **The GTX 1650 4 GB is sufficient for inference.** 35.4M params is 71 MB in fp16; a 10-min mel is 15 MB.
- **Gates are unchanged.** Gate A: autoencoder onset F1 ≥ 0.98 at ±1 frame. Gate B: diffusion onset F1 > 0.40 on held-out audio, checked early (~20k steps).

---

## Schedule

**Measured 2026-08-31**, `p1`, 1536-frame window, 2×T4, fp16, no gradient
checkpointing. Throughput is set almost entirely by batch size, because at small
batch these models are launch-overhead bound rather than compute bound — step
time barely moves while the batch grows 16×:

| batch/GPU | step time | samples/s | GPU util | memory |
|---|---|---|---|---|
| 2 (profile default) | 2.22 s | 1.8 | 22–27% | 0.9 GiB |
| 8 | 2.4 s | 6.7 | — | 0.9 GiB |
| **32** | 2.7 s | **23.7** | 40–54% | 1.3 GiB |

At effective batch 64 that is **75 hours for 100k steps, 225 for 300k** — 2.5 to
7 weeks of quota. The profile's own `per_gpu_batch = 2` was costing 13×, and the
"~9 GB per T4" note beside it was an unmeasured guess that overstated memory by
about 10×. Both are corrected in `model_config.py`; treat `per_gpu_batch` as a
floor that fits anywhere, not a recommendation.

40–54% utilisation says there is still headroom at 64/GPU. It was not taken,
because that doubles the effective batch to 128 and D5's reasoning is about
effective batch, not throughput — worth revisiting with an lr adjustment if the
schedule gets tight.

`tiny` measures 29 samples/s under the same settings, so `p1` costs only ~20%
more per sample than the rehearsal profile. That removes most of the argument
for rehearsing on `tiny` at all (D4): the plumbing it was meant to shake out —
mel cache, memmap, DataParallel, alignment on real data — is already proven, and
a `p1` run keeps its weights.

---

## Work order

Packing is the only step whose clock cannot be compressed, and it needs no code
beyond step 1. Start it, then write the rest while it runs.

### Now — unblocks packing
1. **`pack_dataset.py --ranked-only`.** It already stores the flag at line 448; it just cannot filter on it. (D2)
2. **Start packing.** 2–5 h unattended CPU.

### Done: the shards are on Kaggle

`jimmyreturnz/taiko-shards` holds all three files byte-exact (`mels.dat`
7,178,432,256 B, `charts.npz`, `index.json`), verified against the local pack by
`scripts/upload_to_kaggle.py --verify-only`. Kaggle's dataset listing reports a
smaller *stored* total (6.34 GB) because it compresses; that is not a short
upload, and the verifier no longer flags it.

Items 3–5 below are done too: the notebook clones `main`, both training scripts
set `cudnn.benchmark`, and `--grad-accum` is wired.

### Two sessions lost, and what actually took them

Stage 2 died 46 minutes into an 11-hour session at step 1,025, with the log
reporting `ram 30.7/32.1G`. It has now happened twice. What each part
contributed, separated by measurement rather than by reading:

**Not the training code.** 2,018 steps of stage 2 against a synthetic corpus,
four workers, batch 64 at the same 1,536-frame window: memory dead flat, start
to finish. The dataset, the loader, the EMA, validation, and the checkpoint
path do not leak. `--mel-io read` also does what it claims — 300 batches, zero
RSS growth.

**The memory figure was wrong, and that is why this cost two sessions rather
than one.** `ram` was the sum of RSS over the process and its dataloader
workers. RSS counts a shared page in full in every process mapping it, so
everything a fork shares — the CUDA context, torch's shared objects, the model,
the packed index — was multiplied by the worker count. Measured here: parent
825 MB, RSS sum 2,618 MB, real footprint about 1 GB. On Kaggle, where every
worker also inherits a CUDA context, the overstatement is larger still. "30.7
GB" was never a number anyone could act on. It is now PSS, split between this
process and its workers, next to the container's own usage and limit and the
page cache called out separately.

**The notebook is what turned a 46-minute problem into two lost sessions.**
Three separate faults, all in `kaggle_train.ipynb`:

1. Stage 2 launched with `!python ...`, which discards the exit code. The
   trainer was killed; the notebook zipped, evaluated and finished green with
   ten paid GPU-hours idle. Nothing restarted it, though `last.pt` at step
   1,000 was intact and only ~2 minutes of training had been lost.
2. **Two cells could not be parsed at all** — the one that restores checkpoints
   from `/kaggle/input` and the one that sets `RESUMABLE`. A `\n` written for
   the notebook was expanded when `build_notebook.py` was imported and split
   the string literal it sat inside. A `SyntaxError` is raised at compile time,
   so neither cell ever ran a statement: nothing was restored and nothing
   resumed, in a notebook whose whole design is resuming twenty times. The diff
   read perfectly, because nothing had ever parsed the generated notebook.
3. Saving a version is not attaching a dataset. Section 8 writes the zip to the
   session output; section 4 reads `/kaggle/input`. Without the attach step the
   next session legitimately starts at step 0 — which is the "it resets and
   starts from the very beginning" symptom exactly.

**Fixed.** Section 5 supervises both stages: restart on an out-of-memory death,
resume from the existing checkpoint, budget from what is left of the session,
one fewer worker each attempt, stop on any other error, raise if the stage
never finished. `--min-free-gb` makes a stage save and exit with code 17 while
there is still room to write 541 MB, instead of taking a SIGKILL. The build
refuses to write a notebook containing a cell Python cannot parse, and
`tests/test_notebook.py` keeps it that way.

**Still open.** With the data path cleared by measurement and the metric
corrected, the residual growth — if there is any — has not been attributed.
The next session's first log line now prints an honest breakdown; that is what
to read before changing anything else.

### Next: the `tiny` rehearsal (D4)

Attach the dataset to `notebooks/kaggle_train.ipynb` on a T4 x2 session and run
it top to bottom. **Record the measured steps/sec from stage 2's first ten
minutes and replace the Schedule paragraph above with it.**

### After the rehearsal — before the GUI
6. **Batch the sampler.** `sampling.py:196` loops windows sequentially at batch 1 and runs CFG as a second separate forward: a 5-min song is **1,900 batch-1 U-Net forwards**, entirely launch-overhead bound. Stack all windows *and* both CFG branches into one batch → 50 forwards at batch ~38. Estimated 3–8 min → 10–30 s. Cap the batch (`--batch-windows`) so a 15-min song still fits 4 GB. ~15 lines.
7. **Multi-BPM at inference — three call sites**, not a redesign:
   - `generate.py:196` — `timing_stream_from_bpm(bpm, offset, ...)` takes one global librosa BPM; must take a segment list.
   - `tensor_to_beatmap` (`tensor_repr.py:288`) — constructs a single red line; must emit all supplied red lines.
   - `apply_timing_refinement` — takes a scalar `audio_bpm`; must go segment-aware.
8. **Corpus presets.** Cluster window motifs over the ranked shards (k≈8), name the clusters, ship centroids as `PRESETS`. (D10)
9. **Timing input paths:** `--timing-from map.osu` first — it covers most real use, since popular songs are already timed — then a tracker seed. (D6)

### v2 — the GUI
10. Gradio on localhost wrapping `generate.py`'s existing arguments: difficulty, style, CFG, seed, **N charts**, metadata fields, Generate, download `.osz`.

### v3 — the shipped app
11. ONNX export; DDIM + MultiDiffusion loop in numpy (~100 lines, logic exists in `sampling.py`).
12. Packaged installer. The updater checks a `version.json` on GitHub Releases, downloads the changed asset, verifies SHA-256, swaps. Weights versioned separately from code. ~40 lines, no update framework.

---

## Deferred, deliberately

- **Hosted web version.** Would need a GPU bill (D3 rules it out) or Hugging Face Spaces + ZeroGPU. The Gradio app ports unchanged if this is ever wanted, so deferring costs nothing.
- **Per-dimension motif masking in the GUI** ("streams, don't care about big notes"). A real feature that centroids cannot express, but a refinement — not the fix for the preset bug.
- **Subdivision-mask conditioning and the difficulty envelope** (`DEVELOPMENT_PLAN.md` Phase 4). Still the honest way to express Kantan→Inner Oni.
- **`torch.compile`.** 1.2–1.4×, against a 2–5 min recompile on *every* Kaggle session restart. Revisit if step time disappoints.
- **DDP instead of DataParallel.** 1.7× → 1.9×, fiddly inside a notebook. Revisit only if the rehearsal shows DP overhead dominating at batch 2.

---

## What would reopen these

- **Gate A fails at `tiny`** → a data or plumbing bug, not capacity. Stop and find it; `python tests/test_dataset.py` first.
- **Gate B fails at `p1` on real data** → alignment, not capacity. Do not add parameters or steps. Re-check `test_dataset.py`, the audio encoder resolutions, and the motif leakage probe (`evaluate.py --use-reference-motif`).
- **Gate B passes but maps read bland** → capacity. This is what `p3` (D9) exists for.
- **Measured step time far above 1.5 s** → the schedule reverts toward the old estimate, and D9 should be revisited in the *other* direction: a smaller model, not a larger one.
