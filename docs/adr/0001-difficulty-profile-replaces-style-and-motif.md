# 1. A map-wide difficulty profile replaces the style label and the per-window motif

Status: accepted, 2026-10-02

## Context

The diffusion model is conditioned on two inputs that are computed from the
answer it is asked to produce:

- **Style**: `infer_style_and_snaps(bm)` is a rule applied to the target chart
  itself. Its 4 classes are skewed (6312 / 1776 / 523 / 2495), so it works as
  a snap-mix hint, not an independent style signal.
- **Motif** is computed from the target *window*, with 30% per-dimension
  dropout. It hands the model the local shape of the window it should be
  reading from the audio. A hand-written motif, like the deathstream one, may
  also sit far from anything the model saw in training.

## Decision

Conditioning becomes one **difficulty profile** per map:

- the motif computed over the whole map
- avg and peak NPS
- stable SR from `osu!.db`
- tech share (ternary and unusual snaps) and 1/8 share (speed), as separate
  numbers

Local shape comes from the audio, plus per-window density measured from the
audio (`--window-density auto`). Style and the per-window motif are removed.

A **preset** is a profile averaged over reference maps that Jimmy chose:

- deathstream: YaniFR's ranked single-difficulty sets and reisen91937's sets.
  Snaps capped at 1/6: no 1/8, 1/12 or 1/5.
- speed: SUNKiSS 3 DROP [Love Oni] and [Sunlight Love], POWA OF DA WILDANES
  (Whulf).
- tech: Adcar (paz08) above 6★, Stigma (Billain Remix).
- stream: 5 shortlisted corpus maps, Jimmy picks.
- standard: no preset.

`--reference map.osu` gives any map's profile directly.

## Consequences

- The diffusion model needs a retrain. The autoencoder does not, because the
  chart representation is unchanged.
- Presets and `--reference` use the same input, so a preset is just a
  reference averaged over several maps.
- The eval and the blind A/B bucket results by tech share and speed share as
  well as SR band, so a preset can be checked against the maps it came from.
