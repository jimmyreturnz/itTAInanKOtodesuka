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
            self.duration_ms = self.notes[-1].end_time if self.notes[-1].is_long else self.notes[-1].time

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
            # drumroll: params are "curve_type|...,slides,length"
            # end_time derived from length and current SV — approximate here
            # accurate end time needs timing context; store raw for now
            end_time = time  # will be filled by BPM-aware post-process
            if len(parts) >= 8:
                try:
                    end_time = int(float(parts[7]))  # some formats include end time
                except (ValueError, IndexError):
                    pass
            return TaikoNote(time=time, note_type="roll", end_time=end_time)

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
