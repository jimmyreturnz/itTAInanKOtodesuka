"""
taiko/data/tokenizer.py  (v2 — beat-relative tokens)

Converts TaikoBeatmap <-> integer token sequences.

KEY CHANGE FROM v1:
    Time is now encoded as beat-relative subdivisions, not ms deltas.
    The model learns rhythm patterns in musical terms (1/4, 1/8 notes)
    rather than arbitrary millisecond values.

Vocabulary design:
    BEAT tokens:  BEAT_0 ... BEAT_N  (cumulative subdivision steps from song start)
    Note tokens:  HIT_DON, HIT_KAT, BIG_DON, BIG_KAT, ROLL_START, ROLL_END, ...
    Special:      PAD, SOS, EOS, UNK
    Conditioning: DIFF_*, DENSITY_*, DON_RATIO_*, BIG_RATIO_*, OD_*

Beat subdivision grid:
    Base unit = 1/48 of a beat (LCM of 1,2,3,4,6,8,12,16,24,48)
    This covers all common taiko snaps:
        1/1  = 48 units
        1/2  = 24 units
        1/3  = 16 units
        1/4  = 12 units
        1/6  =  8 units
        1/8  =  6 units
        1/12 =  4 units
        1/16 =  3 units
        1/48 =  1 unit  (finest grid)

    BEAT tokens represent DELTA steps (not absolute position).
    Max gap = 192 units = 4 beats (long enough for any musical pause).
    Gaps longer than 4 beats use multiple BEAT tokens or a SILENCE token.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math

from taiko.data.osu_parser import TaikoBeatmap, TaikoNote, TimingPoint


# ---------------------------------------------------------------------------
# Beat grid constants
# ---------------------------------------------------------------------------

SUBDIVISIONS_PER_BEAT = 48          # LCM of all common snap divisors
MAX_BEAT_DELTA        = 192         # max delta token = 4 beats
SILENCE_THRESHOLD     = 192         # gaps > 4 beats → SILENCE token


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

PAD_TOKEN  = "<PAD>"
SOS_TOKEN  = "<SOS>"
EOS_TOKEN  = "<EOS>"
UNK_TOKEN  = "<UNK>"
SILENCE_TOKEN = "SILENCE"           # represents a gap > 4 beats

NOTE_TOKENS = [
    "HIT_DON", "HIT_KAT", "BIG_DON", "BIG_KAT",
    "ROLL_START", "ROLL_END", "DENDEN_START", "DENDEN_END",
]

# Conditioning tokens
DIFF_TOKENS      = [f"DIFF_{i}"      for i in range(31)]   # 0-15* in 0.5 steps
DENSITY_TOKENS   = [f"DENSITY_{i}"   for i in range(21)]   # 0-20 nps
DON_RATIO_TOKENS = [f"DON_RATIO_{i}" for i in range(11)]   # 0-100%
BIG_RATIO_TOKENS = [f"BIG_RATIO_{i}" for i in range(11)]
OD_TOKENS        = [f"OD_{i}"        for i in range(11)]
UNK_COND_TOKEN   = "UNK_COND"

# Beat delta tokens: BEAT_0 (no movement) ... BEAT_192 (4 beats)
BEAT_TOKENS = [f"BEAT_{i}" for i in range(MAX_BEAT_DELTA + 1)]


class TaikoVocabulary:
    """Maps token strings <-> integer IDs."""

    def __init__(self):
        all_tokens = (
            [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, SILENCE_TOKEN]
            + NOTE_TOKENS
            + DIFF_TOKENS
            + DENSITY_TOKENS
            + DON_RATIO_TOKENS
            + BIG_RATIO_TOKENS
            + OD_TOKENS
            + [UNK_COND_TOKEN]
            + BEAT_TOKENS
        )
        self._tok2id = {tok: i for i, tok in enumerate(all_tokens)}
        self._id2tok = {i: tok for tok, i in self._tok2id.items()}

        self.PAD_ID     = self._tok2id[PAD_TOKEN]
        self.SOS_ID     = self._tok2id[SOS_TOKEN]
        self.EOS_ID     = self._tok2id[EOS_TOKEN]
        self.UNK_ID     = self._tok2id[UNK_TOKEN]
        self.SILENCE_ID = self._tok2id[SILENCE_TOKEN]

        self.BEAT_START_ID = self._tok2id["BEAT_0"]
        self.BEAT_END_ID   = self._tok2id[f"BEAT_{MAX_BEAT_DELTA}"]

    def __len__(self):
        return len(self._tok2id)

    def encode(self, token: str) -> int:
        return self._tok2id.get(token, self.UNK_ID)

    def decode(self, token_id: int) -> str:
        return self._id2tok.get(token_id, UNK_TOKEN)

    def is_beat_token(self, token_id: int) -> bool:
        return self.BEAT_START_ID <= token_id <= self.BEAT_END_ID

    def beat_token_to_steps(self, token_id: int) -> int:
        return token_id - self.BEAT_START_ID

    def steps_to_beat_token(self, steps: int) -> int:
        steps = max(0, min(steps, MAX_BEAT_DELTA))
        return self.BEAT_START_ID + steps

    def conditioning_ids(
        self,
        star_rating: float,
        notes_per_second: float,
        don_ratio: float,
        big_ratio: float,
        overall_difficulty: float,
        use_unknown: bool = False,
    ) -> list[int]:
        if use_unknown:
            return [self.encode(UNK_COND_TOKEN)] * 5
        return [
            self.encode(f"DIFF_{max(0, min(int(star_rating * 2), 30))}"),
            self.encode(f"DENSITY_{max(0, min(int(notes_per_second), 20))}"),
            self.encode(f"DON_RATIO_{max(0, min(int(don_ratio * 10), 10))}"),
            self.encode(f"BIG_RATIO_{max(0, min(int(big_ratio * 10), 10))}"),
            self.encode(f"OD_{max(0, min(int(overall_difficulty), 10))}"),
        ]


# ---------------------------------------------------------------------------
# Beat grid helpers
# ---------------------------------------------------------------------------

def ms_to_steps(ms: float, beat_length_ms: float, offset_ms: float = 0.0) -> int:
    """Convert absolute ms position to subdivision steps from song start."""
    relative_ms = ms - offset_ms
    steps = relative_ms / beat_length_ms * SUBDIVISIONS_PER_BEAT
    return int(round(steps))


def steps_to_ms(steps: int, beat_length_ms: float, offset_ms: float = 0.0) -> float:
    """Convert subdivision steps back to absolute ms."""
    return offset_ms + steps * beat_length_ms / SUBDIVISIONS_PER_BEAT


def snap_ms_to_grid(ms: float, beat_length_ms: float, offset_ms: float = 0.0) -> tuple[int, float]:
    """
    Snap a ms time to nearest subdivision grid point.
    Returns (steps, snapped_ms).
    """
    steps = ms_to_steps(ms, beat_length_ms, offset_ms)
    snapped_ms = steps_to_ms(steps, beat_length_ms, offset_ms)
    return steps, snapped_ms


def get_beat_length(bm: TaikoBeatmap, time_ms: float) -> float:
    """Get the beat length (ms/beat) at a given time from timing points."""
    beat_length = 500.0  # default 120 BPM
    for tp in bm.timing_points:
        if tp.uninherited and tp.time <= time_ms:
            beat_length = tp.beat_length
    return beat_length


def get_offset(bm: TaikoBeatmap) -> float:
    """Get the first timing point offset."""
    for tp in bm.timing_points:
        if tp.uninherited:
            return float(tp.time)
    return 0.0


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

@dataclass
class TokenizedBeatmap:
    conditioning_ids: list[int]
    token_ids: list[int]
    beat_length_ms: float          # stored for decoding
    offset_ms: float               # first beat offset
    metadata: dict


class TaikoTokenizer:
    """
    Converts TaikoBeatmap <-> beat-relative token sequences.

    Token sequence layout:
        [COND_1..COND_N, SOS, BEAT_d1, NOTE_1, BEAT_d2, NOTE_2, ..., EOS]

    BEAT_d = delta in subdivision steps from previous note.
    For gaps > MAX_BEAT_DELTA steps, a SILENCE token is emitted instead,
    and the delta resets.
    """

    def __init__(self, vocab: Optional[TaikoVocabulary] = None):
        self.vocab = vocab or TaikoVocabulary()

    def encode(self, bm: TaikoBeatmap,
               use_unknown_conditioning: bool = False) -> TokenizedBeatmap:
        vocab = self.vocab

        # Get timing info (use first uninherited timing point)
        beat_length_ms = get_beat_length(bm, 0)
        offset_ms      = get_offset(bm)

        # Conditioning
        cond_ids = vocab.conditioning_ids(
            star_rating=bm.star_rating,
            notes_per_second=bm.notes_per_second,
            don_ratio=bm.don_ratio,
            big_ratio=bm.big_ratio,
            overall_difficulty=bm.overall_difficulty,
            use_unknown=use_unknown_conditioning,
        )

        # Build event list: (abs_steps, note_token_str)
        events: list[tuple[int, str]] = []
        for note in bm.notes:
            steps, _ = snap_ms_to_grid(note.time, beat_length_ms, offset_ms)
            if note.note_type == "don":
                events.append((steps, "HIT_DON"))
            elif note.note_type == "kat":
                events.append((steps, "HIT_KAT"))
            elif note.note_type == "big_don":
                events.append((steps, "BIG_DON"))
            elif note.note_type == "big_kat":
                events.append((steps, "BIG_KAT"))
            elif note.note_type == "roll":
                events.append((steps, "ROLL_START"))
                if note.end_time > note.time:
                    end_steps, _ = snap_ms_to_grid(note.end_time, beat_length_ms, offset_ms)
                    events.append((end_steps, "ROLL_END"))
            elif note.note_type == "denden":
                events.append((steps, "DENDEN_START"))
                if note.end_time > note.time:
                    end_steps, _ = snap_ms_to_grid(note.end_time, beat_length_ms, offset_ms)
                    events.append((end_steps, "DENDEN_END"))

        events.sort(key=lambda e: e[0])

        # Encode as delta beat tokens
        token_ids  = [vocab.SOS_ID]
        prev_steps = 0

        for abs_steps, note_tok in events:
            delta = abs_steps - prev_steps
            if delta < 0:
                delta = 0

            # Large gap → emit SILENCE + reset, then small delta
            if delta > MAX_BEAT_DELTA:
                token_ids.append(vocab.SILENCE_ID)
                # Emit remaining delta clamped
                delta = min(delta % MAX_BEAT_DELTA, MAX_BEAT_DELTA)

            token_ids.append(vocab.steps_to_beat_token(delta))
            token_ids.append(vocab.encode(note_tok))
            prev_steps = abs_steps

        token_ids.append(vocab.EOS_ID)

        return TokenizedBeatmap(
            conditioning_ids=cond_ids,
            token_ids=token_ids,
            beat_length_ms=beat_length_ms,
            offset_ms=offset_ms,
            metadata={
                "title":       bm.title,
                "version":     bm.version,
                "star_rating": bm.star_rating,
                "note_count":  bm.note_count,
                "bpm":         60000.0 / beat_length_ms,
                "offset_ms":   offset_ms,
            },
        )

    def decode(self, token_ids: list[int],
               beat_length_ms: float,
               offset_ms: float,
               bm_template: Optional[TaikoBeatmap] = None) -> TaikoBeatmap:
        vocab = self.vocab
        bm    = TaikoBeatmap()

        if bm_template:
            bm.title          = bm_template.title
            bm.artist         = bm_template.artist
            bm.creator        = bm_template.creator
            bm.audio_filename = bm_template.audio_filename
            bm.timing_points  = bm_template.timing_points
            bm.overall_difficulty = bm_template.overall_difficulty
            bm.slider_multiplier  = bm_template.slider_multiplier

        notes      = []
        abs_steps  = 0
        open_roll  = None
        open_denden = None

        i = 0
        while i < len(token_ids):
            tid = token_ids[i]

            if tid in (vocab.SOS_ID, vocab.PAD_ID):
                i += 1; continue
            if tid == vocab.EOS_ID:
                break
            if tid == vocab.SILENCE_ID:
                i += 1; continue  # gap handled implicitly by next BEAT delta

            if vocab.is_beat_token(tid):
                delta     = vocab.beat_token_to_steps(tid)
                abs_steps += delta
                abs_ms    = steps_to_ms(abs_steps, beat_length_ms, offset_ms)

                if i + 1 < len(token_ids):
                    note_tok = vocab.decode(token_ids[i + 1])
                    note = self._make_note(note_tok, int(abs_ms), beat_length_ms)
                    if note:
                        notes.append(note)
                        if note.note_type == "roll":
                            open_roll = note
                        elif note.note_type == "denden":
                            open_denden = note
                    elif note_tok == "ROLL_END" and open_roll:
                        open_roll.end_time = int(abs_ms)
                        open_roll = None
                    elif note_tok == "DENDEN_END" and open_denden:
                        open_denden.end_time = int(abs_ms)
                        open_denden = None
                    i += 2; continue
            i += 1

        bm.notes = notes
        bm.compute_stats()
        return bm

    def _make_note(self, tok: str, time_ms: int,
                   beat_length_ms: float) -> Optional[TaikoNote]:
        simple = {
            "HIT_DON": "don", "HIT_KAT": "kat",
            "BIG_DON": "big_don", "BIG_KAT": "big_kat",
        }
        if tok in simple:
            return TaikoNote(time=time_ms, note_type=simple[tok])
        elif tok == "ROLL_START":
            return TaikoNote(time=time_ms, note_type="roll",
                             end_time=time_ms + int(beat_length_ms * 2))
        elif tok == "DENDEN_START":
            return TaikoNote(time=time_ms, note_type="denden",
                             end_time=time_ms + int(beat_length_ms * 4))
        return None


# ---------------------------------------------------------------------------
# Serializer
# ---------------------------------------------------------------------------

class OsuTaikoSerializer:
    def serialize(self, bm: TaikoBeatmap, audio_filename: str = "") -> str:
        af = audio_filename or bm.audio_filename or "audio.mp3"
        lines = [
            "osu file format v14", "",
            "[General]",
            f"AudioFilename: {af}",
            "AudioLeadIn: 0",
            "Mode: 1",
            "LetterboxInBreaks: 0", "",
            "[Metadata]",
            f"Title:{bm.title}",
            f"TitleUnicode:{bm.title_unicode or bm.title}",
            f"Artist:{bm.artist}",
            f"Creator:{bm.creator}",
            f"Version:{bm.version or 'AI Generated'}",
            f"BeatmapID:{bm.beatmap_id}",
            f"BeatmapSetID:{bm.beatmap_set_id}", "",
            "[Difficulty]",
            f"HPDrainRate:{bm.hp_drain}",
            "CircleSize:5",
            f"OverallDifficulty:{bm.overall_difficulty}",
            f"ApproachRate:{bm.approach_rate}",
            f"SliderMultiplier:{bm.slider_multiplier}",
            f"SliderTickRate:{bm.slider_tick_rate}", "",
            "[TimingPoints]",
        ]
        for tp in bm.timing_points:
            flag = 1 if tp.uninherited else 0
            lines.append(f"{tp.time},{tp.beat_length:.6f},{tp.meter},1,0,100,{flag},0")

        lines += ["", "[HitObjects]"]
        for note in sorted(bm.notes, key=lambda n: n.time):
            lines.append(self._note_to_line(note))
        return "\n".join(lines)

    def _note_to_line(self, note: TaikoNote) -> str:
        x, y = 256, 192
        if note.note_type in ("don", "big_don"):
            hs = 4 if note.note_type == "big_don" else 0
            return f"{x},{y},{note.time},1,{hs},0:0:0:0:"
        elif note.note_type in ("kat", "big_kat"):
            hs = 10 if note.note_type == "big_kat" else 8
            return f"{x},{y},{note.time},1,{hs},0:0:0:0:"
        elif note.note_type == "roll":
            length = max(1, note.end_time - note.time)
            return f"{x},{y},{note.time},2,0,L|{x+1}:{y},1,{length}"
        elif note.note_type == "denden":
            return f"{x},{y},{note.time},8,0,{note.end_time},0:0:0:0:"
        return ""
