"""
taiko/data/osu_parser.py

Parses osu! beatmap files (Mode:1 taiko) into structured Python objects.
Handles hit objects, timing points, and metadata extraction.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from enum import IntFlag, auto


# ---------------------------------------------------------------------------
# Hit object type flags (osu! bit flags)
# ---------------------------------------------------------------------------

class HitType(IntFlag):
    CIRCLE   = 1
    SLIDER   = 2   # drumroll in taiko
    SPINNER  = 8   # denden in taiko
    NEW_COMBO = 4
    # taiko doesn't use combo colors, but the flag still appears

class HitSound(IntFlag):
    NORMAL  = 0
    WHISTLE = 2   # kat
    FINISH  = 4   # big note
    CLAP    = 8   # kat

    @property
    def is_kat(self) -> bool:
        return bool(self & (HitSound.WHISTLE | HitSound.CLAP))

    @property
    def is_big(self) -> bool:
        return bool(self & HitSound.FINISH)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TimingPoint:
    time: int           # ms
    beat_length: float  # ms per beat (positive) or SV multiplier (negative)
    meter: int          # beats per measure
    uninherited: bool   # True = BPM point, False = inherited (SV)

    @property
    def bpm(self) -> Optional[float]:
        if self.uninherited and self.beat_length > 0:
            return 60_000.0 / self.beat_length
        return None

    @property
    def ms_per_beat(self) -> float:
        """Returns ms per beat for uninherited points only."""
        if self.uninherited:
            return self.beat_length
        raise ValueError("Inherited timing point has no absolute BPM.")


@dataclass
class TaikoNote:
    time: int           # ms offset
    note_type: str      # "don", "kat", "big_don", "big_kat", "roll", "denden"
    end_time: int = 0   # only for roll/denden

    # Raw slider geometry, kept only between parsing a drumroll and resolving
    # its duration against the timing points. `resolve_roll_durations()` reads
    # these and then they are dead weight -- nothing downstream should use them.
    px_length: float = 0.0   # osu! pixel length of the slider path
    slides:    int   = 1     # number of times the drumroll repeats

    @property
    def duration(self) -> int:
        return max(0, self.end_time - self.time)

    @property
    def is_long(self) -> bool:
        return self.note_type in ("roll", "denden")


@dataclass
class TaikoBeatmap:
    # Metadata
    title: str = ""
    title_unicode: str = ""
    artist: str = ""
    creator: str = ""
    version: str = ""        # difficulty name
    audio_filename: str = ""
    beatmap_id: int = 0
    beatmap_set_id: int = 0

    # Difficulty params
    hp_drain: float = 5.0
    overall_difficulty: float = 5.0
    approach_rate: float = 5.0  # unused in taiko but present
    slider_multiplier: float = 1.4
    slider_tick_rate: float = 1.0

    # Star rating (if available from osu! API or .osu extended fields)
    star_rating: float = 0.0

    # Content
    timing_points: list[TimingPoint] = field(default_factory=list)
    notes: list[TaikoNote] = field(default_factory=list)

    # Derived stats (filled after parsing)
    duration_ms: int = 0
    note_count: int = 0
    don_count: int = 0
    kat_count: int = 0
    big_count: int = 0
    roll_count: int = 0
    denden_count: int = 0

    def compute_stats(self):
        self.note_count = len(self.notes)
        self.don_count    = sum(1 for n in self.notes if n.note_type == "don")
        self.kat_count    = sum(1 for n in self.notes if n.note_type == "kat")
        self.big_count    = sum(1 for n in self.notes if n.note_type in ("big_don", "big_kat"))
        self.roll_count   = sum(1 for n in self.notes if n.note_type == "roll")
        self.denden_count = sum(1 for n in self.notes if n.note_type == "denden")
        if self.notes:
            # max(), not notes[-1]: a long drumroll can start before the final
            # hit and still end after it.
            self.duration_ms = max(
                (n.end_time if n.is_long else n.time) for n in self.notes
            )

    @property
    def don_ratio(self) -> float:
        hits = self.don_count + self.kat_count
        return self.don_count / hits if hits > 0 else 0.5

    @property
    def big_ratio(self) -> float:
        return self.big_count / self.note_count if self.note_count > 0 else 0.0

    @property
    def notes_per_second(self) -> float:
        if self.duration_ms <= 0:
            return 0.0
        return self.note_count / (self.duration_ms / 1000.0)


# ---------------------------------------------------------------------------
# Drumroll duration resolution
# ---------------------------------------------------------------------------

# osu! defines one "slider velocity unit" as 100 pixels per beat, scaled by the
# beatmap's SliderMultiplier and by the active green line's velocity.
OSU_PIXELS_PER_BEAT = 100.0

# Guard rails. A drumroll longer than this is always a broken green line or a
# corrupt file, never a real chart.
MAX_ROLL_MS = 60_000
MIN_ROLL_MS = 10


def _timing_at(timing_points: list["TimingPoint"], time_ms: int) -> tuple[float, float]:
    """
    Resolve the timing state in force at `time_ms`.

    Returns:
        (ms_per_beat, sv_multiplier)

        ms_per_beat    from the most recent uninherited (red) line
        sv_multiplier  from the most recent inherited (green) line, where
                       osu! encodes velocity as a negative number:
                       sv = -100 / beat_length, so -100 -> 1.0x, -50 -> 2.0x

    A green line's effect ends when a later red line resets it, which is why
    both are tracked in a single forward pass rather than searched separately.
    """
    ms_per_beat = 500.0     # osu!'s own fallback: 120 BPM
    sv          = 1.0

    for tp in timing_points:
        if tp.time > time_ms:
            break
        if tp.uninherited:
            if tp.beat_length > 0:
                ms_per_beat = tp.beat_length
            sv = 1.0        # a red line clears any active green line
        else:
            if tp.beat_length < 0:
                sv = -100.0 / tp.beat_length
            # A non-negative inherited value is malformed; osu! ignores it.

    return ms_per_beat, sv


def resolve_roll_durations(bm: "TaikoBeatmap") -> None:
    """
    Turn each drumroll's raw pixel length into a real end time, in place.

        beats    = px_length / (SliderMultiplier * 100 * sv)
        duration = beats * ms_per_beat * slides

    Must run after timing points are parsed and sorted. Without it every
    drumroll in the corpus carries a fabricated end time, which poisons the
    roll channel of the chart tensor and any duration derived from it.
    """
    if not bm.notes:
        return

    slider_mult = bm.slider_multiplier if bm.slider_multiplier > 0 else 1.4

    for note in bm.notes:
        if note.note_type != "roll":
            continue

        if note.px_length <= 0:
            # No usable geometry. One beat is the least-wrong assumption, and
            # it keeps the note representable rather than dropping it.
            ms_per_beat, _ = _timing_at(bm.timing_points, note.time)
            note.end_time = note.time + int(round(ms_per_beat))
            continue

        ms_per_beat, sv = _timing_at(bm.timing_points, note.time)

        velocity = slider_mult * OSU_PIXELS_PER_BEAT * sv
        if velocity <= 0:
            note.end_time = note.time + int(round(ms_per_beat))
            continue

        beats    = note.px_length / velocity
        duration = beats * ms_per_beat * max(1, note.slides)

        duration = min(max(duration, MIN_ROLL_MS), MAX_ROLL_MS)
        note.end_time = note.time + int(round(duration))


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class OsuTaikoParser:
    """Parse a .osu file in taiko mode (Mode:1) into a TaikoBeatmap."""

    SECTION_RE = re.compile(r"^\[(\w+)\]$")

    def parse_file(self, path: str | Path) -> TaikoBeatmap:
        path = Path(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        return self.parse_text(text)

    def parse_text(self, text: str) -> TaikoBeatmap:
        bm = TaikoBeatmap()
        section = None

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("//"):
                continue

            m = self.SECTION_RE.match(line)
            if m:
                section = m.group(1)
                continue

            if section == "General":
                self._parse_general(line, bm)
            elif section == "Metadata":
                self._parse_metadata(line, bm)
            elif section == "Difficulty":
                self._parse_difficulty(line, bm)
            elif section == "TimingPoints":
                tp = self._parse_timing_point(line)
                if tp:
                    bm.timing_points.append(tp)
            elif section == "HitObjects":
                note = self._parse_hit_object(line)
                if note:
                    bm.notes.append(note)

        # Sort by time (some maps have minor ordering issues)
        bm.timing_points.sort(key=lambda t: t.time)
        bm.notes.sort(key=lambda n: n.time)

        # Drumrolls were parsed with raw pixel geometry; only now that the
        # timing points are known and sorted can they become real durations.
        resolve_roll_durations(bm)

        bm.compute_stats()
        return bm

    # -- Section parsers -----------------------------------------------------

    def _parse_general(self, line: str, bm: TaikoBeatmap):
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip()
        if k == "AudioFilename":
            bm.audio_filename = v
        elif k == "Mode":
            if v != "1":
                raise ValueError(f"Expected taiko mode (1), got mode {v}")

    def _parse_metadata(self, line: str, bm: TaikoBeatmap):
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip()
        mapping = {
            "Title": "title", "TitleUnicode": "title_unicode",
            "Artist": "artist", "Creator": "creator",
            "Version": "version", "BeatmapID": None, "BeatmapSetID": None,
        }
        if k == "BeatmapID":
            bm.beatmap_id = int(v) if v.isdigit() else 0
        elif k == "BeatmapSetID":
            bm.beatmap_set_id = int(v) if v.isdigit() else 0
        elif k in mapping and mapping[k]:
            setattr(bm, mapping[k], v)

    def _parse_difficulty(self, line: str, bm: TaikoBeatmap):
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip()
        try:
            fv = float(v)
        except ValueError:
            return
        mapping = {
            "HPDrainRate": "hp_drain",
            "OverallDifficulty": "overall_difficulty",
            "ApproachRate": "approach_rate",
            "SliderMultiplier": "slider_multiplier",
            "SliderTickRate": "slider_tick_rate",
        }
        if k in mapping:
            setattr(bm, mapping[k], fv)

    def _parse_timing_point(self, line: str) -> Optional[TimingPoint]:
        parts = line.split(",")
        if len(parts) < 2:
            return None
        try:
            time        = int(float(parts[0]))
            beat_length = float(parts[1])
            meter       = int(parts[2]) if len(parts) > 2 else 4
            uninherited = bool(int(parts[6])) if len(parts) > 6 else True
            return TimingPoint(time, beat_length, meter, uninherited)
        except (ValueError, IndexError):
            return None

    def _parse_hit_object(self, line: str) -> Optional[TaikoNote]:
        """
        osu! hit object format:
          x,y,time,type,hitSound[,objectParams][,hitSample]

        In taiko:
          - circles  → don/kat depending on hitSound flags
          - sliders  → drumroll (has end time in objectParams)
          - spinners → denden   (has end time as 5th param)
        """
        parts = line.split(",")
        if len(parts) < 5:
            return None
        try:
            time      = int(parts[2])
            obj_type  = int(parts[3])
            hit_sound = HitSound(int(parts[4]))
        except (ValueError, IndexError):
            return None

        is_circle  = bool(obj_type & HitType.CIRCLE)
        is_slider  = bool(obj_type & HitType.SLIDER)
        is_spinner = bool(obj_type & HitType.SPINNER)

        if is_circle:
            is_big = hit_sound.is_big
            is_kat = hit_sound.is_kat
            if is_big:
                note_type = "big_kat" if is_kat else "big_don"
            else:
                note_type = "kat" if is_kat else "don"
            return TaikoNote(time=time, note_type=note_type)

        elif is_slider:
            # Drumroll. Format from index 5 on:
            #   curveType|curvePoints , slides , length , edgeSounds , ...
            #
            # parts[7] is `length` in osu! PIXELS, not a time. Converting it
            # needs the active red line (ms per beat) and green line (slider
            # velocity) at this instant, which the caller resolves afterwards
            # in resolve_roll_durations(). Leave end_time unset until then.
            slides    = 1
            px_length = 0.0
            try:
                slides = max(1, int(parts[6]))
            except (ValueError, IndexError):
                pass
            try:
                px_length = max(0.0, float(parts[7]))
            except (ValueError, IndexError):
                pass
            return TaikoNote(
                time=time,
                note_type="roll",
                end_time=time,
                px_length=px_length,
                slides=slides,
            )

        elif is_spinner:
            end_time = int(parts[5]) if len(parts) > 5 else time
            return TaikoNote(time=time, note_type="denden", end_time=end_time)

        return None


# ---------------------------------------------------------------------------
# Utility: find all .osu files in taiko mode
# ---------------------------------------------------------------------------

def find_taiko_osu_files(root: str | Path) -> list[Path]:
    """
    Walk a directory tree and return .osu files that are taiko mode.
    Checks for 'Mode: 1' in the file quickly before full parsing.
    """
    root = Path(root)
    results = []
    for osu_file in root.rglob("*.osu"):
        try:
            text = osu_file.read_text(encoding="utf-8", errors="replace")
            if "Mode: 1" in text or "Mode:1" in text:
                results.append(osu_file)
        except OSError:
            continue
    return results
