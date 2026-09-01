# Manual

Everything you actually have to do, in order.

The split is: **your PC** handles the maps and audio (they live there, and the
step is slow but needs no GPU). **Kaggle** handles training (two T4s). You move
one folder between them.

```
   your PC                          Kaggle
   -------                          ------
   osu! Songs folder
        |
        | pack_dataset.py           (hours, CPU only, resumable)
        v
   data/processed/shards/  ---->    upload as a Kaggle Dataset
                                         |
                                         | train_autoencoder.py    ~8 h
                                         | train_diffusion.py      150-250 h
                                         v
                                    checkpoints  <---- download
        |                                              
        v
   generate.py  ---->  a .osz you drop into osu!
```

---

## 0. One-time setup

On your PC:

```bash
git clone https://github.com/jimmyreturnz/itTAInanKOtodesuka.git
cd itTAInanKOtodesuka
git checkout main
pip install -r requirements.txt
```

Check it works:

```bash
python tests/test_dataset.py
```

That one matters more than it looks. It encodes each frame's own index into a
synthetic mel and reads it back out of a training window, which is the only
way to prove audio and charts describe the same milliseconds. If it fails, stop
and fix that before anything else.

### An osu! API key (recommended)

Star ratings come from the osu! API. Without them maps get dropped, because the
alternative is using `OverallDifficulty` as a star rating — it is an accuracy
parameter with no relation to difficulty, and feeding it in poisons the
conditioning control you will reach for most.

Get a key at <https://osu.ppy.sh/home/account/edit> (bottom of the page, "Legacy
API"), then:

```bash
# Windows PowerShell
$env:OSU_API_KEY = "your_key_here"

# Linux / macOS
export OSU_API_KEY=your_key_here
```

---

## 1. Pack the dataset (your PC)

```bash
python scripts/pack_dataset.py --scan "D:/osu!/Songs" --ranked-only
```

`--ranked-only` is the corpus decision from `DIRECTION.md` (D2): unranked maps
include rate-ups, which pair one chart with differently-timed audio and are
anti-signal for exactly the alignment Gate B measures, plus gimmick maps that
star rating cannot filter out. Folders holding no ranked map are skipped before
mel extraction, which is about a fifth of the packing time.

Try a small run first — it takes a minute and shows you whether anything is
wrong before you commit to a few hours:

```bash
python scripts/pack_dataset.py --scan "D:/osu!/Songs" --ranked-only --limit 200
```

What it does:

1. Finds every taiko-mode `.osu` under your Songs folder
2. Extracts one mel spectrogram per song folder into `data/processed/mel_cache/`
3. Parses every difficulty, derives star rating, NPS, style and snap ratios
4. Writes three files to `data/processed/shards/`

**It is resumable.** The mel cache is per song, so if you interrupt it (or your
PC sleeps), re-running picks up where it stopped. Re-running with no `--scan`
reuses the cached file list:

```bash
python scripts/pack_dataset.py --ranked-only
```

Expect roughly 2-5 hours for 13k maps, almost all of it in mel extraction.

### Reading the output

```
Packed  : 11,482 maps over 3,914 songs
Audio   : 27,183,442 frames (151.0 hours, 6.48 GB)
Skipped :
      812  no star rating
      401  too few notes
       97  no mel
```

Skips are normal. `no star rating` means the API had nothing for that map —
usually unsubmitted or very old. Large numbers of `no mel` mean audio files
that failed to decode; check `--limit` output first if you see thousands.

If `chart longer than audio` appears in the skip list, those maps have audio
that does not match their charts (a common problem with re-uploaded sets), and
dropping them is correct.

### Upload

The `data/processed/shards/` folder is what you upload. Nothing else.

1. <https://www.kaggle.com/datasets> → **New Dataset**
2. Drag in the three files: `mels.dat`, `charts.npz`, `index.json`
3. Name it **taiko-shards**
4. Set it Private, create

Expect 5-10 GB. The upload takes a while; Kaggle's browser uploader is more
reliable than the CLI for files this size.

---

## 2. Train (Kaggle)

Open `notebooks/kaggle_train.ipynb` in Kaggle (File → Import Notebook, or
create a notebook and paste the cells).

**Notebook settings**, right-hand panel:

| Setting | Value |
|---|---|
| Accelerator | **GPU T4 x2** |
| Internet | **On** |
| Persistence | Files only |

Attach your `taiko-shards` dataset (**Add Data** → your datasets).

Then run the cells in order — all of them, every session. Section 4 restores
checkpoints before anything trains, so a resumed session that skips it starts
over from scratch without saying so.

Section 5 has an alignment sanity check: it reports how much louder the audio
is at note positions than elsewhere. Above 1.0 means notes land on audio
events. Near 1.0 means they do not, and you should stop rather than train on
it.

Section 3 prints how mels will be read. A 6.5 GB `mels.dat` is most of a
Kaggle notebook's RAM once it is resident, which is why `--mel-io read` is the
default — see the memory entry under Troubleshooting.

### Stage 1: autoencoder (~8 hours, usually one session)

```
val loss 0.0142 | onset F1 0.9831 (P 0.981 R 0.985 @ threshold 0.90) | Gate A PASS
```

**Gate A is onset F1 >= 0.98.** Validation loss is not the gate. The script
refuses to hand off a checkpoint that has not cleared it, and it is right to:
whatever the autoencoder loses, the diffusion model can never recover.

If it plateaus below 0.98, drop one entry from `--channel-mult`:

```python
"--channel-mult", "1", "1", "2", "2",    # 8x instead of 16x
```

That halves the compression, doubles the latent length, and makes stage 2
slower — worth it. Do not proceed on 0.95.

### Stage 2: diffusion (150-250 hours, 15-25 sessions)

This is the long haul. Each session: run every cell in order. Section 4
restores the checkpoints *before* anything trains, section 7 trains until its
share of the 11-hour budget runs out, and section 8 packages the result.

Section 8 is the one that bites. Kaggle deletes `/kaggle/working` when a
session ends. If you forget, you lose that session's work entirely.

Saving checkpoints between sessions:

- Run section 8 to produce `checkpoints.zip`
- Download it from the Output panel
- Upload it as a Dataset named **taiko-checkpoints** (or update the existing one)
- Attach it next session; section 4 restores from it

**Upload both `best.pt` and `last.pt`, for both stages.** They do different
jobs and neither substitutes for the other:

| file | carries | used by |
|---|---|---|
| `last.pt` | weights, optimiser, GradScaler, EMA, step, epoch, position in the epoch | `--resume` |
| `best.pt` | the same, at the best validation score so far | `evaluate.py`, `generate.py`, and as the fallback if `last.pt` is damaged |

`best.pt` cannot resume a run properly: it is a snapshot from whenever
validation last improved, so resuming from it silently discards every step
since. And with only `last.pt` you have nothing to generate from if the most
recent weights are worse than the best ones.

### What a session writes, and when

`last.pt` is written every 250 steps, every 10 minutes, and at the end of every
epoch — whichever comes first. The clock is the one that matters. Steps are not
a unit of risk: at 2.6 s/step, saving every 500 steps means an OOM kill or a
session cut-off costs 22 minutes, and the previous run lost exactly that when
it died at step 550 with `last.pt` still at step 500.

Writes are atomic — a temporary file, then a rename — so a kill during a save
cannot leave a `last.pt` the next session refuses to open. If one is damaged
anyway, `--resume` falls back to `best.pt` and says so rather than ending the
session.

SIGTERM and the notebook's interrupt button now save before exiting. `SIGKILL`
from the OOM killer cannot be caught by anything, which is why the periodic
save above is the real defence.

### Resuming lands where it stopped

The checkpoint records the position within the epoch, and `--resume` runs only
the remaining batches of it. The old loop restarted the epoch from batch zero,
repeating up to a full epoch on every resume — across twenty resumes, days.

A healthy loss curve falls fast for the first few thousand steps and then
improves slowly. Do not read much into small val-loss movements — the number
that matters is Gate B.

### Gate B (section 9)

```
onset_f1           0.5231   target > 0.550    below target
snap_validity      0.9764   target > 0.950    PASS
GATE B  onset F1 > 0.4: PASSED  (0.5231)
```

**Gate B is onset F1 > 0.40.** It is the alignment gate: a model that ignores
the audio and emits plausible taiko rhythms scores near zero here no matter how
good its loss looks. Check it once early — around 20k steps — because a model
that fails Gate B will not be fixed by more steps, and finding that out at
200k steps costs weeks.

If Gate B fails early:

- Run `python tests/test_dataset.py` — audio/chart alignment
- Check the section 5 sanity check reported well above 1.0
- Confirm the audio encoder levels match the U-Net levels (`profile.summary()`)

Once Gate B passes, keep training; the remaining metrics improve with steps.

---

## 3. Generate maps

Download `checkpoints.zip`, unzip into `checkpoints/`, then:

```bash
python scripts/generate.py --audio "song.mp3" --difficulty 5.5
```

Output lands in `outputs/` as a `.osz`. Double-click to import into osu!.

### Getting the tempo right

This matters more than any other flag. The model generates *against* a beat
grid you supply, so a wrong tempo produces a chart that is internally
consistent and completely off the music.

```bash
python scripts/generate.py --audio song.mp3 --difficulty 5.5 --bpm 174 --offset 812
```

Without `--bpm` the tempo is detected and printed. Detection is decent but not
reliable — if the result feels off-beat, open the audio in the osu! editor, use
its timing panel to find the real BPM and offset, and pass them.

### Controlling difficulty and style

| Flag | What it does |
|---|---|
| `--difficulty 5.5` | target star rating |
| `--preset stream` | named style: `simple`, `standard`, `stream`, `speed`, `tech`, `big_heavy` |
| `--style stream` | the coarse 4-class label from training |
| `--reference "map.osu"` | copy the style of an existing map |
| `--avg-nps 7` | target notes per second |
| `--cfg-scale 4.0` | how hard to push toward what you asked for |

`--reference` is the interesting one:

```bash
python scripts/generate.py --audio song.mp3 --difficulty 6 \
    --reference "Songs/12345 Some Song/Some Song [Oni].osu"
```

It extracts the rhythmic fingerprint of that map — its snap distribution, colour
change rate, burst density, big-note rate — and asks for a chart with the same
character.

### When output is wrong

| Symptom | Try |
|---|---|
| Too many notes | `--threshold 0.95` (higher = fewer) |
| Too few notes | `--threshold 0.7` |
| Empty map | lower `--threshold`, or the model is undertrained |
| Ignores your settings | raise `--cfg-scale` to 6-8 |
| Repetitive, mechanical | lower `--cfg-scale` to 2-3 |
| Off-beat | supply `--bpm` and `--offset` |
| Notes slightly off-grid | drop `--no-refine` so snapping runs |

`--seed 42` makes a run reproducible; changing it gives a different map for the
same settings.

---

## 4. Measuring quality

```bash
python scripts/evaluate.py --n-maps 40
```

| Metric | Target | What a bad value means |
|---|---|---|
| onset_f1 | > 0.55 | the chart does not follow the music |
| snap_validity | > 0.95 | notes sit off the beat grid |
| sr_correlation | > 0.85 | asking for a difficulty does nothing |
| nps_error | < 1.0 | density control does not work |
| unplayability | < 0.005 | it produces patterns humans cannot hit |

The leakage probe is worth running once:

```bash
python scripts/evaluate.py --n-maps 20 --use-reference-motif
```

This conditions on the reference chart's own motif, which is an oracle. Compare
against the normal run: a small gap is fine, a large one means the model has
learned to read the answer off its conditioning vector rather than the audio,
and real generation will disappoint in a way the training loss never showed.

No metric replaces playing them. Import ten maps into osu! and play them at
each milestone.

---

## Troubleshooting

**`No packed dataset found under /kaggle/input`**
The dataset is not attached, or you uploaded the parent folder. Kaggle must see
`mels.dat`, `charts.npz` and `index.json`.

**`window_frames must be a multiple of 64`**
Windows must be whole numbers of latent frames. 1536 works; 1500 does not — it
decodes back to 1488 and loses the tail into the loss.

**`checkpoint is profile 'p1', you asked for 'tiny'`**
Resume with the profile that checkpoint was trained with.

**`chart runs past the end of the audio`**
A map claims to be longer than its audio file. Usually a re-uploaded set whose
audio was replaced. `pack_dataset.py` already drops these.

**CUDA out of memory**
Lower `--batch-size` to 1, or switch to `--profile tiny` to confirm the
pipeline first.

**Training is slow / GPUs idle**
Check both T4s are visible in section 1. Raise `--num-workers` to 4 and
`--prefetch-factor` to 4 — but watch the `ram` figure in the log, since each
queued batch costs `batch x window x 128` floats of host memory. If the GPU
sits below 80% the dataloader is the bottleneck, not the model.

**Kaggle session died and I lost work**
At most ten minutes of it, plus whatever was in `/kaggle/working` and never
packaged. Always run section 8 before the session ends, and leave `--max-hours`
set from the session budget so training stops with time to spare.

**"Your notebook tried to allocate more memory than is available"**
Host RAM, not GPU. The usual cause was reading mels through a memmap: every
page it touches becomes a resident page, so random window sampling walks RSS up
by the full size of `mels.dat` over an hour or two and then the session dies —
a leak with nothing in the Python heap to blame for it. Both training scripts
now default to `--mel-io read`, which preads one window at a time and holds no
pages (measured on a 732 MB file: 20k random windows cost 732 MB of RSS mapped,
3 MB pread).

If it still happens, the log line carries the numbers:

```
epoch 2 step 550  loss 0.79  mae 0.71  |g| 0.37  lr 2.75e-05  2.4min  ram 6.2/28.9G
```

`ram` is the whole process tree, dataloader workers included, over the total the
machine has. If it climbs steadily, the next things to cut are
`--prefetch-factor` (each queued batch is `batch x window x 128` floats),
`--num-workers`, and `--batch-size`.

**Resuming started from an earlier step than I expected**
The run was killed before its next save. Check the `Resumed from ...: step N,
epoch E, batch B/M` line — that is exactly where it will continue from. If it is
older than you expect, lower `--save-every-min`.

---

## Where things live

```
taiko/data/frames.py       the time grid; nothing else may define one
taiko/data/tensor_repr.py  chart <-> tensor, and the beat-grid conditioning
taiko/data/shards.py       the packed dataset format
taiko/data/motif.py        the style vector and its presets
taiko/train/session.py     atomic checkpoints, save triggers, memory reporting
taiko/model/               autoencoder, audio encoder, U-Net, diffusion
taiko/eval/metrics.py      chart quality measures
scripts/                   pack, train, evaluate, generate
notebooks/                 the Kaggle notebook (built by build_notebook.py)
```

`DEVELOPMENT_PLAN.md` explains why the architecture is what it is, and what is
still outstanding.
