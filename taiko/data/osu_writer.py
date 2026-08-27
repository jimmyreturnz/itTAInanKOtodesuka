"""
taiko/data/osu_writer.py

TaikoBeatmap -> .osu text.

Everything the model produces leaves through here, so a bug in this file
degrades every output regardless of model quality -- and degrades it silently,
because the file still opens in osu!. Two such bugs lived here:

  * a big kat was written as hitsound 10 (WHISTLE|CLAP), which carries no
    FINISH bit, so osu! read it back as an ordinary kat and every big kat in a
    generated map was quietly downgraded. It is FINISH|CLAP = 12.

  * a drumroll's extent was written as its duration in milliseconds, but osu!
    stores a PIXEL length and derives duration from the slider velocity in
    force. That is the same units confusion that made every *parsed* drumroll
    wrong, inverted.

Split out of the retired tokenizer module, which existed for an autoregressive
token model that the latent-diffusion design replaced.
"""

from __future__ import annotations

from taiko.data.osu_parser import TaikoBeatmap, TaikoNote, _timing_at


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
            lines.append(self._note_to_line(note, bm))
        return "\n".join(lines)

    def _note_to_line(self, note: TaikoNote, bm: TaikoBeatmap) -> str:
        """
        One hit object line.

        Hitsounds are bit flags: FINISH(4) makes a note big, and WHISTLE(2) or
        CLAP(8) makes it a kat. A big kat is therefore FINISH|CLAP = 12. Writing
        10 (WHISTLE|CLAP) produces a note that is a kat but carries no FINISH
        bit, so osu! -- and this repo's own parser -- read it back as an
        ordinary kat, silently dropping every big kat in a generated map.
        """
        x, y = 256, 192

        if note.note_type in ("don", "big_don"):
            hs = 4 if note.note_type == "big_don" else 0
            return f"{x},{y},{note.time},1,{hs},0:0:0:0:"

        if note.note_type in ("kat", "big_kat"):
            hs = 12 if note.note_type == "big_kat" else 8
            return f"{x},{y},{note.time},1,{hs},0:0:0:0:"

        if note.note_type == "roll":
            # osu! stores a drumroll's extent as a PIXEL length, and derives its
            # duration from the slider velocity in force. Writing the duration
            # in milliseconds here would produce a roll whose real length is
            # off by whatever ms_per_beat happens to be -- the same confusion,
            # inverted, that made every parsed drumroll wrong.
            #
            #   duration = px_length / (SliderMultiplier * 100 * SV) * ms_per_beat
            #   px_length = duration / ms_per_beat * SliderMultiplier * 100 * SV
            ms_per_beat, sv = _timing_at(bm.timing_points, note.time)
            duration = max(1, note.end_time - note.time)
            px_length = duration / ms_per_beat * bm.slider_multiplier * 100.0 * sv
            return f"{x},{y},{note.time},2,0,L|{x + 1}:{y},1,{px_length:.3f}"

        if note.note_type == "denden":
            return f"{x},{y},{note.time},8,0,{note.end_time},0:0:0:0:"

        return ""
