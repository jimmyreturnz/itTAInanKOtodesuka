from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from taiko.data.osu_parser import OsuTaikoParser
from pathlib import Path

# Path to a .osu file


OSU_FILE = Path("taiko/data/beatmap/Nile - Kem Khefa Kheshef (Genjuro) [Lord of Chaos].osu")

parser = OsuTaikoParser()

beatmap = parser.parse_file(OSU_FILE)

print("=== METADATA ===")
print("Title:", beatmap.title)
print("Artist:", beatmap.artist)
print("Creator:", beatmap.creator)
print("Difficulty:", beatmap.version)

print("\n=== STATS ===")
print("Notes:", beatmap.note_count)
print("Don:", beatmap.don_count)
print("Kat:", beatmap.kat_count)
print("Big:", beatmap.big_count)
print("Rolls:", beatmap.roll_count)
print("Denden:", beatmap.denden_count)
print("Duration:", beatmap.duration_ms, "ms")
print("NPS:", round(beatmap.notes_per_second, 2))
print("SR:", beatmap.star_rating)

print("\n=== FIRST 10 NOTES ===")
for note in beatmap.notes[:10]:
    print(note)

print("\n=== FIRST 5 TIMING POINTS ===")
for tp in beatmap.timing_points[:5]:
    print(tp)
