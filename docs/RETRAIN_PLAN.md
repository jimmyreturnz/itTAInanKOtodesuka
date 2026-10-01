# Retrain plan and status

The working plan for the scratch-or-continue retrain: every decision Jimmy
made, the measurements behind them, and what is done. The status section
comes first; the full plan, as decided, follows it.

## Working rules (Jimmy's, standing)
- **Measure first.** Write the harness in `scripts/`, check it against ranked
  maps before believing it, and put the numbers in the commit message.
- **Never touch unranked maps.** Iterate the shard index records, which are
  ranked by construction. Use the Songs scan only to find a record's own
  .osu, matched by folder and version. Never loop over the scan itself.
- **No AI attribution in commits:** no Co-Authored-By, no "Generated with".
- The ranking checks in `taiko/eval/criteria.py` and `mapset.py` were
  written from the osu! wiki with an external checker as reference. Cite
  the wiki; **never name that checker anywhere in the repo.**
- Blind A/B rounds always use new songs. Jimmy marks which chart is the
  human one.
- osu! stores a snapped note **truncated** to whole ms, not rounded
  (`decode.osu_ms`), and a red line's offset can be fractional
  (`TimingPoint.exact_time`).

## Harness commands
```
python scripts/generate.py --audio song.mp3 --timing-from map.osu --level Oni
python scripts/evaluate.py --per-band 10 --seeds 3 --probs-cache outputs/eval_cache_v2
python scripts/check_criteria.py --probs-cache outputs/eval_cache_v2 --enforce
python scripts/check_mapset.py --probs-cache outputs/eval_cache_v2
python scripts/select_checkpoint.py outputs/eval/*.json --incumbent outputs/eval_baseline_phase2.json
python scripts/blind_ab.py make --round N        # then: blind_ab.py score --round N
python scripts/measure_repetition.py --probs-cache outputs/eval_cache_v2
python scripts/sweep_family_switch.py --probs-cache outputs/eval_cache_v2
```

## Status at 2026-10-02 — start here

### Done (newest last)
| Commit | What |
|---|---|
| fc2de4a | Peak-first grid decoder. Default-pool F1 0.613 -> 0.655 |
| 7c24720 | Sampler edge fix: the song's first/last latent frame had blend weight 0 (31 -> 5 notes in the first 320 ms) |
| 05ea38e | Audio offset measured over 2820 songs: one mode, no decoder split. 8 songs had a rate-edited audio file; interim fix in `find_audio` |
| 4ed6b53 | 20 ms frames are not the limit: a perfect model gets 98.06% of 7*+ hits on the exact ms, so no autoencoder retrain |
| 73ae5aa | `note_times.npz` (real .osu ms) + `exact_snap` metric + `--per-band` stratified eval pool |
| f07a6eb | Feel metrics (stream / snap / colour / density JS) against ranked maps of the same SR band |
| ae7f77b | `select_checkpoint.py` gates (silence, too-fast, per-band F1), `--snapshot-every`, silence metric |
| a028900 | `blind_ab.py` blind rounds + `taiko/data/osu_db.py` (osu!.db: NM/DT/HT taiko SR, unplayed flag, ranked status) |
| 07123aa..c85088b | `taiko/eval/criteria.py`: per-difficulty ranking checks (problem / warning / minor), tempo folded into 130-270 BPM, level from exact name or SR |
| ee12b3b | `generate.py --level Kantan..Hell Oni`: ranked median SR, OD and HP, plus `criteria.enforce` |
| b36285c | `taiko/eval/mapset.py`: set-level ranking checks. Decoder now truncates to whole ms like osu! does, Grid uses exact fractional red-line offsets, no notes before the first red line, `fix_chart` |
| d877d24 | Two low-difficulty tells from round 3 are now checks: Kantan 1/1 runs in the song's own beat, triplet-only notes in Kantan/Futsuu |

### Baseline (best.pt, step 58968; `evaluate.py --per-band 10 --seeds 3`, 60 maps)
- Onset F1 0.686. Exact snap 0.955, against a ceiling of 0.998.
- Exact snap by the mapper's divisor: 1/1 0.962, 1/2 0.945, 1/3 0.512,
  1/4 0.826, 1/6 0.751, 1/8 0.610.
- Exact snap at 7*+: 0.851.
- Notes on silence: 2.5% against ranked 2.3%.
- Before `criteria.enforce`, the model breaks a problem- or warning-level
  check on 52-92% of charts per level. After it, 0%, with 0.2% of notes
  dropped.
- Bar repetition, share of bars whose rhythm recurs (ranked / model): <2*
  0.82 / 0.47, 7*+ 0.54 / 0.08.

### Blind A/B (Jimmy picks which chart is human)
- Round 1: 6/6.
- Round 2, criteria enforced: 6/6.
- Round 3, criteria plus set-level fixes: 6/6, judged in the editor
  without playing. His notes:
  - The Kantan had too many consecutive notes. Fixed, d877d24.
  - The Futsuu had 1/6 out of nowhere. Fixed, d877d24.
  - Kat and finisher placement don't match the song.
  - At Oni and above, the snaps follow noise.
- Every round uses 6 new never-played songs (`outputs/blind_ab/used.json`).

**What is left is in the model, not the decoder:**
- Snap flicker: 13.9 fast-snap changes per 100 hits at 5.5*+, against
  ranked 2.0. The family-switch penalty cannot fix it
  (`scripts/sweep_family_switch.py`). Matched notes alone change 5.8 per
  100, and the 30% of extra notes make the rest.
- Finisher and colour placement.
- No pattern repetition, a 30.7 s window limit.

### Decision pending: warm-start or scratch
Recommended: **warm-start**.
- The new conditioning (difficulty profile, ADR 0001) only touches the
  conditioning embedding, 0.08M of 35.6M params (0.2%).
  `taiko/train/warm_start.py` already widened this model once, with an
  exact reproduction check.
- Scratch to today's 62.4k steps costs about 47 Kaggle GPU-h.
- Settle it with a 10k-step scratch control (about 7.5 GPU-h), compared on
  the eval pool and the criteria checks.

### Next, in order (Phase 3 below)
1. **Re-pack (item 10h)**, which also covers items 9, 10e and the audio fix:
   - ranked = API ranked and the md5 matches;
   - SR from osu!.db by md5;
   - one record per beatmap ID;
   - audio = the AudioFilename of the md5-verified ranked .osu;
   - split by beatmapset ID;
   - every skipped map reported with its reason;
   - outros included in training;
   - `pack_dataset.py` writes `note_times.npz` itself.
2. **Rate augmentation (item 10):** a window's SR interpolated from the
   map's own HT/NM/DT ratings, which `osu_db` already reads.
3. **Difficulty profile (item 10b, ADR 0001):** it replaces the style label
   and the per-window motif. Add the criteria level to it. Presets per the
   ADR.
4. **SR-band sampling (item 12),** then the **warm-start plus scratch
   control (item 13)**. Judge by the eval, the criteria checks,
   `select_checkpoint.py` and a blind round.

### What needs Jimmy's machine
- These are machine-local and gitignored: the osu! Songs folder,
  `D:/osu!/osu!.db`, `data/processed/shards` (7 GB), `checkpoints/` and the
  eval sample cache.
- **Needs the local machine:** packing, evaluation, `check_criteria.py` /
  `check_mapset.py` runs, blind rounds, generation.
- **Fine in a cloud session:** code changes and their tests. Every test
  runs on synthetic data:
  ```
  for t in tests/test_*.py; do python "$t"; done
  ```
  `test_shards.py` and `test_checkpointing.py` have Linux-only memory checks.

---

## The plan, as decided: where the model can learn the wrong thing, and what to do

## Context

Jimmy asked for a review of itTAInanKOtodesuka from the point of view of an ML
engineer who also plays taiko: how to improve it, and whether the model can
learn the wrong thing even though training uses only ranked maps. Everything
below was read from the code or measured in this session. Items marked
**unverified** are hypotheses that come with the measurement that would settle
them.

The pipeline's split is sound. Validation is **split by song**
(`split_indices` in `preprocessed_dataset.py`), so a song's Oni can't leak
into training while its Muzukashii is in validation. The main risks are
elsewhere: labels that are wrong for some samples, and evaluation that rewards
the wrong thing.

## Findings, ranked by how much they can mislead training or evaluation

### 1. Evaluation rewards wrong snaps (measured today)
The shards store charts as 20 ms frames, so the eval's reference notes are
frame-quantised, not the mapper's real milliseconds. Proof: sweeping the
decoder's neighbour weight raised hard-map F1 from 0.722 to 0.758, while the
synthetic test showed notes moving onto the **wrong line** (a 1/8 where the
mapper wrote 1/3). F1's ±25 ms window can't tell 1/4, 1/6 and 1/8 apart at
high BPM. So F1 and snap validity can both improve while the rhythm gets
worse.
- **Fix:** store each chart's real note times (ms) in the shard index next to
  the frames, and score the eval against those: exact-line accuracy per snap,
  not ±25 ms onset F1.

### 2. The training target can't represent dense high-BPM rhythm
One frame is 20 ms. 1/8 at 200 BPM is 37.5 ms, under 2 frames, and 1/6 and
1/4 lines after a beat are 22 ms apart at 220 BPM, about 1 frame. Two notes
inside one frame collapse into one. For the deathstream and speed charts you
care about, the chart tensor itself loses information; the sin/cos timing
input carries sub-frame phase, but the *target* does not.
- **Measure first:** the share of ranked 6★+ notes that share a frame with
  another note or land 1 frame apart, per BPM band.
- **If it's material:** add a sub-frame offset channel (the note's position
  within its frame). This changes `tensor_repr` and needs the autoencoder and
  diffusion retrained, so it's the costly option.

### 3. Rate augmentation mislabels difficulty
`_rate_augment` plays 20% of windows at 0.9–1.1×. It scales `avg_nps` and
`peak_nps` by the rate, but **not `difficulty`**. A 1.1× window is harder, yet
it's labelled with the 1.0× SR. That teaches the model that SR and density
are partly unrelated, which fights the SR control you want.
- **Fix:** scale SR with the rate. Taiko SR is roughly linear in rate over
  ±10%; check that against the SR calculator on a few maps. Alternatively
  drop the SR label (difficulty dropout) on augmented windows.

### 4. Audio timing may be inconsistent between songs (**unverified**, and possibly large)
`audio.py` decodes with torchaudio and falls back to librosa on failure.
MP3 decoders disagree about encoder delay/padding (about 25 ms at 44.1 kHz for
LAME). If some songs decode through a path that keeps the delay, their audio
is shifted by about 1 frame against the charts. That is per-song label noise
at exactly the scale that separates 1/4 from 1/6. Today on Decoherence, onsets
detected from the audio sat about 20 ms before the ranked map's beat lines, and
the auto-timing came out 22 ms early.
- **Measure:** for every training song, cross-correlate the onset strength
  with the chart's onsets and histogram the per-song lag. A spike at 0 means
  fine. Two modes (0 and ~25 ms), or a smear, means this is real. Split the
  result by which decoder was used.
- **Fix if real:** decode everything through one gapless-aware path, and
  re-pack.

### 5. Hard maps are underrepresented
Training time by SR: <2★ 16.7%, 2–4★ 35.8%, 4–5.5★ 22.7%, 5.5–7★ 15.9%,
**7★+ 8.8%** (508 songs, 201 mappers). The hard end, where you play and where
the density ceiling showed up today (the model had no peaks left for a 7.5★
request), gets less than a tenth of the gradient.
- **Fix:** weight window sampling by SR band, e.g. flatten toward uniform, and
  watch the easy buckets for regressions in the eval.

### 6. Conditioning that shows the model part of the answer
- **Motif** is computed from the target window, with 30% per-dimension
  dropout. The leakage probe exists (`evaluate.py --use-reference-motif`) but
  I found no record of it being run on this checkpoint. Run it: a large gap
  means the model reads its conditioning instead of the audio. Hand-made
  motifs (like the deathstream one) may also be far from anything seen in
  training.
- **Style** is `infer_style_and_snaps(bm)`, a rule applied to the chart
  itself. The label is a function of the answer, and it's skewed (class 0
  6312, 3 2495, 1 1776, 2 523). It works as a snap-mix hint, not an
  independent style signal.

### 7. The best checkpoint is picked by the wrong number
`best.pt` is chosen by diffusion val MSE (`train_diffusion.py:466`). That loss
is averaged over noise levels and barely tracks chart quality. `best.pt` has
been stuck at step 58968 while training reached 62400, and the kept checkpoint
may not be the best one for charting.
- **Fix:** every N steps, score a fixed cached set of hard held-out maps
  (`--probs-cache` makes this cheap after sampling once) and keep the
  checkpoint with the best chart metric. Keep MSE for monitoring only.

### 8. The eval is too small to decide things
30 maps with one sample each, and one 7★+ map in the default pool. Differences
of 0.01 F1 are inside seed noise. For any decision, use the 99-map ≥5.5★ pool
and `--seeds 3`.

### 9. Nothing measures whether a chart is good to play
Unplayability only checks physical limits (30 ms gaps). Pattern KL is one
number against one reference. Things a player would feel:
- stream length distribution
- 1/4 against 1/6 mix per section
- colour-pattern n-grams (dddk, kkdd, …)
- how density moves between sections
- whether repeated sections (a chorus coming back) get consistent patterns

The last one also needs context wider than the 30.7 s window.
- **Fix:** compare those distributions with ranked maps in the same SR band.
  Also run a small blind A/B with you playing: AI chart against a ranked chart
  of a similar song, scored by feel.

### Smaller points
- **SR labels:** `beatmapset_cache.json` stores whatever SR the API returned
  when each map was fetched. If taiko SR was recalculated between fetches, the
  labels are mixed versions. Check when they were fetched.
- **Kaggle-only code:** `test_shards.py` and `test_checkpointing.py` fail on
  Windows, and `os.preadv` crashed local eval. Mark Linux-only tests and skip
  them elsewhere, so a real failure isn't hidden among expected ones.

## Decisions (Jimmy, 2026-10-01)
- **Retraining:** anything, including the autoencoder, if a measurement says
  it's needed.
- **Target:** all difficulties. Flatten SR sampling somewhat, but easy
  buckets must not regress.
- **Blind A/B:** yes, regularly. His ratings are the final judge alongside
  the metrics.
- **SR labels:** move off the API to `osu!.db`, as taiko_arranger does
  (`taiko_arranger/osu_db.py`). This gives one SR version (the local client's)
  for every map, and md5 matching guarantees the rated `.osu` is the one that
  was packed.

## Plan, in order

### Phase 1 — measurements that decide the expensive work (no retrain)
Scripts go in `scripts/` as harnesses, with numbers in the commit messages.
1. `scripts/measure_audio_offset.py`: per-song lag between mel onset strength
   and the chart's onsets over the whole corpus, histogram split by decoder
   (torchaudio or librosa fallback). Settles finding 4.
2. `scripts/measure_frame_collisions.py`: the share of ranked notes 0–1 frames
   from their neighbour, per BPM band and SR band. Settles finding 2 (whether
   sub-frame offsets are worth an autoencoder retrain).
3. Run `evaluate.py --use-reference-motif` against the default run on the
   cached pool. Settles finding 6.

### Phase 2 — evaluation that rewards the right thing
4. **Real note times:** `pack_dataset.py` stores each chart's hit times (ms)
   in `index.json` (small: roughly 1k ints per chart). `evaluate.py` scores
   against them: exact-line accuracy per snap divisor, alongside F1.
5. **Pool:** a fixed eval set of held-out maps stratified across all SR bands
   (e.g. 10 per band where the pool allows) with `--seeds 3`, samples kept via
   `--probs-cache`.
6. **Feel metrics** against ranked maps in the same SR band: stream length
   distribution, colour n-grams, snap mix per section, density transitions.
   Goes into `taiko/eval/metrics.py`.
7. **Checkpoint selection (Q21):** a checkpoint must pass three gates:
   - notes on silence at most 1.3× ranked
   - raw too-fast pairs at most 2× ranked
   - no SR band's F1 more than 0.02 worse than the previous best

   Among those that pass, it is ranked by exact-snap accuracy. Snap truth is
   the line a note sits on within 2 ms; unsnapped notes are left out (Q3).
   Val MSE is for logs only.
8. **Blind A/B pack (Q4, Q9):** `scripts/blind_ab.py`:
   - 6 held-out songs, one per SR band, that Jimmy has never played
     (filtered via `scores.db`)
   - each song gets an AI chart and its own ranked map at a matched SR,
     under neutral names in shuffled order, plus a sealed key
   - the same 6 songs every round, replaced once they're remembered
   - forced choice plus one line of "why"; `--score` reads the ratings back
     against the key
   - a round happens whenever a checkpoint passes item 7, and Jimmy is
     prompted each time

**Phase 2 status (2026-10-02): items 4-8 done.** Commits 73ae5aa, f07a6eb,
ae7f77b, a028900 in itTAInanKOtodesuka. Baseline, best.pt step 58968,
`--per-band 10 --seeds 3`: F1 0.686, exact snap 0.955 (ceiling 0.998); by
divisor 1/4 0.826, 1/3 0.512, 1/6 0.751, 1/8 0.610; 7*+ exact 0.851; silence
2.5% vs ranked 2.3%. Result: `outputs/eval_baseline_phase2.json`, which is the
incumbent for `select_checkpoint.py`. Blind A/B round 1 is built at
`outputs/blind_ab/round_1` (best.pt); waiting on Jimmy to play it.
osu_db.py is already ported, with NM/DT/HT taiko SR and the unplayed flag,
so item 9 only has to wire it into the packer.

**Blind A/B round 1 (2026-10-02): Jimmy picked the human chart 6/6.** Two
measured reasons (commits 07123aa, fb97f03):
- **Ranking criteria.** `taiko/eval/criteria.py` encodes the wiki's
  Kantan-Inner Oni rules, with the 180 BPM beat cap from Scaling_BPM.
  Ranked maps break a rule 0-1.7% of the time. The model breaks one in 61%
  of Kantans, 37% of Futsuus, 60% of Muzukashiis and 38% of Onis.
- **Repetition.** Ranked charts repeat bar rhythms when the music repeats
  (rhythm-only recurrence 0.44-0.82 by band); the model's are 0.06-0.47.
- SR by difficulty name, median: Kantan 1.36, Futsuu 2.25, Muzukashii 3.23,
  Oni 4.19, Inner 5.41, Ura 5.95, Hell 6.90.
- Blind A/B now copies the ranked map's green lines into both charts.

### Phase 2b — the criteria in chart creation (A done 2026-10-02: ee12b3b, 22e0efd)
- A. No retrain: `generate.py --level Kantan|...|Inner Oni`. The SR target
  comes from the name's median. The decoder enforces the level's minimum
  gap. A criteria repair pass fixes the rest: single-colour 1/2 in Kantan,
  finisher rules, splitting runs that are too long. Gate: check_criteria
  rule breaks at most 2% on the eval pool.
- B. Retrain: the difficulty level joins the profile (ADR 0001), so the
  model learns it instead of being repaired into it.
- C. Repetition: find repeated sections from the audio's self-similarity
  and map them consistently. Research; needs a design.

**Blind A/B round 3 (2026-10-02, criteria + mapset fixes applied): 6/6,
judged from the editor without playing.** Jimmy's notes on the AI charts:
- Kantan: too many consecutive notes. 10 in a row against 6 ranked; at 128
  BPM the tempo fold reads a 1/1 as 1.5 beats, so the 7-note limit never
  applied.
- Futsuu: 1/6 out of nowhere. 4% of gaps triplet-type, ranked 0%. Legal on
  Futsuu's 1/3 grid.
- Muzukashii: kat not matching the song, finishers misplaced. 75% of
  finishers on the beat, ranked 97%.
- Oni and above: inconsistent snaps that follow noise. Fast-snap changes
  62/129/421 against ranked 21/10/24.
Measured (scripts/sweep_family_switch.py): the decoder's family-switch
penalty cannot fix the snap flicker (best 13.9 -> 9.9 changes per 100 hits
at 5.5*+, ranked 2.0). Matched notes alone change snap 5.8 per 100, and the
30% unmatched extra notes interleave to make the rest. **The snap flicker is
in the model's probabilities, which is evidence for the retrain:** the
profile (ADR 0001) carries tech share and 1/8 share, which tell the model
when a song has no ternary at all.

### Phase 3 — data and label fixes, then retrain
9. **osu!.db SR:** port `taiko_arranger/osu_db.py` (no new dependency) and
   extend it to keep the taiko SR for NM, **DT and HT**, not just `mods == 0`.
   `pack_dataset.py` uses it instead of `beatmapset_cache.json`, matched by
   md5.
10. **Rate augmentation:** set the augmented window's SR from that map's own
    HT/NM/DT ratings (interpolate over 0.75, 1.0, 1.5 at the sampled rate),
    so SR and density move together. Test: a 1.1× window's SR is above its
    1.0× one and within the map's measured curve.
10b. **Difficulty profile replaces the style label and the per-window motif**
    (grilling Q2, Q10, Q12, Q14, Q22). Profile = the motif computed over the
    *whole map* + avg/peak NPS + stable SR from osu!.db. No lazer skill port.
    - The 4-class `style` input and the per-window motif are removed from
      conditioning. Local shape comes from the audio, plus per-window density
      measured from the audio (`--window-density auto`).
    - The profile carries **tech share** (ternary + unusual snaps) and the
      1/8 share (speed) as separate numbers.
    - **Presets** are profiles averaged over reference maps Jimmy chose:
      - deathstream: YaniFR's ranked single-difficulty sets (the 12 in the
        corpus plus Tenkai e no Kippu [Shining Blade], which the packer
        missed), plus all 3 of reisen91937's sets. Ranked maps only:
        YaniFR's Quite Contrary diff is not ranked and is out.
      - speed: SUNKiSS 3 DROP [Love Oni] and [Sunlight Love], POWA OF DA
        WILDANES (Whulf)
      - tech: Adcar (paz08) above 6★, Stigma (Billain Remix)
      - stream: I shortlist 5 corpus maps (long 1/4 runs, low tech share,
        5–7★) and Jimmy picks
      - standard: no preset (profile unspecified)
    - `--reference map.osu` gives any map's profile directly.
    - The phase-2 eval and blind A/B also bucket by tech share and speed
      share, not only by SR band.
10c. **Decoder snaps:** 1/1, 1/2, 1/3, 1/4, 1/6, 1/8. **1/12 only when a
    tech preset or reference asks for it** (Q17; 29% of ranked 7★+ maps use
    some). 1/5 and 1/7 stay out (0.13% of 7★+ notes). Maps with unusual
    snaps stay in training (Q6).
    **The deathstream preset caps its snaps at 1/6** (Jimmy, 2026-10-02): it
    uses 1/1, 1/2, 1/3, 1/4 and 1/6. It never uses 1/8, 1/12 or 1/5, even if
    the profile averaged from its reference maps contains them.
10d. **Finisher rule in `repair`:** a finisher with both neighbours within
    1/4 beat (+2 ms) becomes a small note of the same colour. Ranked maps put
    a finisher there at most 0.10% of the time in any SR band (Q18). Count how
    often the current model does it on the cached samples.
10e. **Outros in training:** a chart's training length = the song's length,
    not last note + 1 s, so empty outros are in the loss and windows can start
    there (Q15, Q19). Intros are already in the loss; nothing classifies them.
10f. **Sampler edge bug (no retrain needed, do first):** `_blend_weights`
    tapers to exactly 0 at the song's first and last latent frame, where no
    neighbour window exists. The first 320 ms is left as raw noise, which is
    the source of the notes at 0.01–0.30 s. Taper only where windows overlap.
    Measure first and last note against audible start and end, before and
    after.
10g. **Silence as a metric and a gate:** notes on silence, against ranked
    (currently 2.23% vs 1.10% on the 59 cached samples; 1.6× mid-song, 3.5×
    at the edges). Measure whether it rises with guidance scale and with NPS
    conditioning (Q20). A generation-time silence guard exists only as an
    opt-in flag.
10h. **Re-pack: ranked means "the API says ranked AND the file's md5 is the
    ranked file's md5"** (Jimmy, 2026-10-01). Local folders mix rate edits
    and cut versions made with external tools that keep the ranked map's
    BeatmapID. The current index has **43 beatmap IDs twice (45 extra
    records)**: BLUE ARMY in two folders, "o'er the flood (Cut Ver.)",
    "PUNISHMENT DEVIL (arra…)". None crossed the train/val split this time,
    but nothing prevented it.
    - Ranked status and `file_md5` come from the osu! API. A local `.osu`
      counts only if its md5 matches.
    - SR comes from `osu!.db` looked up by that md5: the stable SR of exactly
      that file (item 9).
    - One record per beatmap ID. A second folder holding the same ranked
      file is packed once.
    - Every ranked taiko map known to the API or `osu!.db` is accounted for.
      The pack reports each one it skips, with the reason. Today 124 are
      silently missing: 88 from sets old enough to have been packed
      (Tenkai e no Kippu [Shining Blade] among them), 36 ranked since Aug 30
      (SUNKiSS 3 DROP).
    - The split uses the beatmapset ID, not the folder name, so two folders
      of one set can never land on both sides.
    - **Audio = the AudioFilename of the md5-verified ranked .osu**, one mel
      per audio file. `measure_audio_offset.py` (2026-10-02) found 8 songs
      packed against a rate-edited copy in their folder (Euphoria, Hope Fate,
      northern_limit, Terminal 11, Architecture, ...). The interim vote in
      `find_audio` fixes 6; Euphoria and Hope Fate have more rate-edit diffs
      than ranked ones, so the vote picks the edit there and the pass-2 guard
      skips their ranked diffs instead.
11. ~~**Audio:** one gapless-aware decode path~~ — not needed. Measured
    2026-10-02 over 2820 songs: one mode at +25 ms (IQR +22..+27), mp3 and
    ogg agree, and torchaudio/librosa match ffmpeg to the sample on 150.
12. **Sampling:** window sampling weighted toward a flatter SR distribution.
    The weight is set from the phase-2 eval: hard buckets improve while easy
    ones hold.
13. **Retrain:** diffusion only. Phase 1 item 2 (2026-10-02): a perfect
    model decodes 98.06% of 7*+ hits to the exact ms through the 20 ms frame
    (99.8%+ under 5.5*), so sub-frame offsets are not worth the autoencoder
    retrain yet. Re-run `measure_frame_collisions.py` once the model nears
    that ceiling. Judged by the phase-2 eval and blind
    A/B, not val loss.

### Housekeeping
- Mark `test_shards.py` / `test_checkpointing.py` memory checks Linux-only
  (skip elsewhere).
- Commit the decoder work from the previous session first, with its numbers,
  so phase 1 starts from a clean baseline.

## Verification
Every item names the measurement that settles it. No fix ships without
before/after numbers on the fixed stratified pool. Phase 2 lands before any
retrain so the retrain can be judged. The blind A/B has the final word on
whether a checkpoint is better to play.
