"""
taiko/data/profile.py

The motif part of a map's difficulty profile (ADR 0001): the same 16-dim motif
the model always read, measured over the whole map instead of one window.

It spans first note to last, not the song: the pack runs charts to the end of
the audio, but `--reference map.osu` has only the .osu, and both must measure
the same thing.
"""

from __future__ import annotations

import numpy as np

from taiko.data.motif import MOTIF_DIM, beat_frames_from_bpm, compute_motif


SNAP_DENOMS   = (1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16)
SNAP_MATCH_MS = 3.0     # 1 ms truncation + a fractional red line, with margin


def active_red_line(timing_points, time_ms):
    active = None
    for tp in timing_points:
        if tp.time > time_ms:
            break
        if tp.uninherited and tp.beat_length > 0:
            active = tp
    return active


def snap_of(note_time: int, timing_points) -> int | None:
    """
    The coarsest snap the note sits on, within SNAP_MATCH_MS. Not the closest:
    osu! truncates a snapped note to whole ms, so a 1/4 note lands up to 1 ms
    off its line and a finer grid (1/12, 1/16) can fit it better, which read
    clean 1/4 streams as 13% tech (Tenkai e no Kippu, 447 notes).
    """
    tp = active_red_line(timing_points, note_time)
    if tp is None:
        return None
    offset = (note_time - tp.exact_time) % tp.beat_length
    for denom in SNAP_DENOMS:
        grid = tp.beat_length / denom
        if abs(offset - round(offset / grid) * grid) <= SNAP_MATCH_MS:
            return denom
    return None


def note_divisor(note_time: int, timing_points) -> int:
    """Index into conditioning.DIVISORS of the note's folded snap; -1 if none."""
    from taiko.data.conditioning import DIVISOR_FOLD, DIVISORS
    d = snap_of(note_time, timing_points)
    return DIVISORS.index(DIVISOR_FOLD[d]) if d in DIVISOR_FOLD else -1


def map_motif(chart: np.ndarray, bpm: float) -> np.ndarray:
    """[16] quantised motif of a whole [6, T] chart, over its notes' span."""
    active = np.flatnonzero(chart.max(axis=0) > 0.5)
    if active.size == 0:
        return np.zeros(MOTIF_DIM, dtype=np.float32)
    return compute_motif(chart[:, active[0]:active[-1] + 1], beat_frames_from_bpm(bpm))


# --------------------------------------------------------------------------- #
# Presets: a profile averaged over reference maps Jimmy chose (ADR 0001, and
# the reference chart study in RETRAIN_PLAN.md). Matched to shard index records
# by beatmapset ID (the folder's leading number) and difficulty name, so a
# preset is only ever ranked, md5-verified maps, whichever folder was packed.
# --------------------------------------------------------------------------- #

PRESET_REFS: dict[str, list[tuple[str, str]]] = {
    # YaniFR's single-difficulty sets and reisen91937's. IRON ROSE is out: 290
    # BPM, longest run 21, a burst map.
    "deathstream": [
        ("2283239 Dragon Guardian - Tenkai e no Kippu", "Shining Blade"),
        ("1977770 Dragon Guardian - Sei Juuji Gakuen", "Arrow of Destiny"),
        ("2009040 Dragon Guardian - Ankoku Butoukai", "Dancing in the Night under the Black Moon"),
        ("2111683 SKYWINGS - TONIGHT", "I Will Meet You Under The Sky"),
        ("2111842 Dragon Guardian - Tabidachi no Asa", "Rise From This Nightmare and Save Our Lands!"),
        ("2139170 Dragon Guardian - Mahou no Sho", "Magical Obsession"),
        ("2159934 ANCIENT MYTH - Chaos to Infinity", "Frontier of Eternity"),
        ("2180835 CADAVER  REVADAC", "YaniFR & Eltigant's Miracle"),
        ("2191269 ANCIENT MYTH - Zenith", "Sunlight Symphony"),
        ("2217406 Dragon Guardian - Eien no Tsurugi", "Ancient Miracle"),
        ("2287038 SKYWINGS", "Swords Lead to Forever Free"),
        ("2422005 Dragon Guardian - Kikaijikake no Kami", "Dragon God"),
        ("2415871 ryu5150 - My spirit lives on", "Tatsujin"),
        ("2447178 ryu5150 - Like a Universe", "Galaxy"),
        ("2559071 ryu5150 - Mighty Red", "Scarlet Fury"),
    ],
    # Speed split in two by the study: one continuous mixed-snap stream at
    # ~185 BPM (SUNKiSS, with Stigma's 1/6 and 1/8), and short bursts at 300.
    "speed": [
        ("2514119 jun with Alison - SUNKiSS _3 DROP ~jun side~", "Love Oni"),
        ("2514119 jun with Alison - SUNKiSS _3 DROP ~jun side~", "Sunlight Love"),
        ("2170880 Noisia - Stigma (Billain Remix)", "Contaminated Escutcheon"),
    ],
    "burst": [
        ("1938286 Camellia - POWA OF DA WILDANES", "ONI"),
    ],
    # Adcar above 6*: triplet tech, 24-27% 1/3 in every difficulty.
    "tech": [
        ("2132229 Terminal 11 - Adcar", "Inner Oni"),
        ("2132229 Terminal 11 - Adcar", "Hell Oni"),
        ("2132229 Terminal 11 - Adcar", "Illegal"),
        ("2132229 Terminal 11 - Adcar", "ulko's Nervousness"),
    ],
    # Jimmy's pick (2026-10-10) of the shortlist from scripts/shortlist_stream.py:
    # 5-7*, 63-69% of notes in 1/4 runs of 16+, tech under 1%, no run over 231.
    "stream": [
        ("2200737 SHK - Flight of the Bumblebee (RST Classic)", "Symphonic"),
        ("2124033 Mage - The Words I Never Said in D&B", "Remorse"),
        ("2095104 Imperfect Cherry Blossom", "Lunatic"),
        ("2437756 Folktales Princess", "Mythic Adventure"),
        ("2242259 Rise Of The Chaos Wizards", "Return Of The Warfare"),
    ],
    # "standard": no preset; leave the profile unspecified.
}

PROFILE_FIELDS = ("tech_share", "snap_1_8", "avg_nps", "peak_nps", "difficulty")

# A preset's divisor mask is the union of its references' unless Jimmy fixed
# it: deathstream uses 1/1, 1/2, 1/3, 1/4 and 1/6, never 1/8 (2026-10-02).
PRESET_DIVISORS: dict[str, set[int]] = {"deathstream": {1, 2, 3, 4, 6}}


def preset_profile(name: str, records: list[dict]) -> dict:
    """The preset's references averaged: motif (re-quantised) plus PROFILE_FIELDS."""
    from taiko.data.motif import quantise_motif
    wanted = {(int(folder.split()[0]), version) for folder, version in PRESET_REFS[name]}
    refs = [r for r in records if (r["beatmapset_id"], r["version"]) in wanted]
    found = {(r["beatmapset_id"], r["version"]) for r in refs}
    if found != wanted:
        raise LookupError(f"preset {name!r}: not in the shards: {sorted(wanted - found)}")
    out = {k: float(np.mean([r[k] for r in refs])) for k in PROFILE_FIELDS}
    out["profile_motif"] = quantise_motif(np.mean([r["profile_motif"] for r in refs], axis=0))
    from taiko.data.conditioning import divisors_used
    out["divisors"] = PRESET_DIVISORS.get(name) or set().union(
        *(divisors_used(r["snap_counts"]) for r in refs))
    return out


def record_conditioning(record: dict, features: int) -> dict:
    """
    How the eval and the blind A/B condition on a held-out map: its own style
    label before features 3, its own profile from features 3 on, the same way
    they already pass its own SR and NPS.
    """
    from taiko.data.conditioning import STYLE_NULL, record_profile
    if features < 3:
        return {"style": int(record.get("style", STYLE_NULL))}
    motif = np.asarray(record["profile_motif"], dtype=np.float32)
    return {"style": STYLE_NULL, "motif": motif, "motif_mask": np.ones_like(motif),
            "profile": record_profile(record)}
