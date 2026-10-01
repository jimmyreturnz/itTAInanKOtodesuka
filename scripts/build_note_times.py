"""
scripts/build_note_times.py

Write note_times.npz beside the shards: every packed map's real hit times in
ms, read back from its .osu. The charts store 20 ms frames, so scoring a
generated chart against them rewards the wrong line (finding 1); exact_snap
needs the mapper's own milliseconds. Matched by folder and difficulty name,
like the measurement harnesses. The re-pack writes the same file itself.

    python scripts/build_note_times.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from pack_dataset import DEFAULT_SCAN_CACHE, safe_name
from taiko.data.osu_parser import OsuTaikoParser
from taiko.data.shards import ShardReader, write_note_times


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, default=Path("data/processed/shards"))
    ap.add_argument("--scan-cache", type=Path, default=DEFAULT_SCAN_CACHE)
    args = ap.parse_args()

    reader = ShardReader(args.shards)
    osu_by_key: dict[str, list[Path]] = defaultdict(list)
    for p in map(Path, json.loads(args.scan_cache.read_text(encoding="utf-8"))):
        osu_by_key[safe_name(p.parent.name)].append(p)

    parser = OsuTaikoParser()
    times: list[list[int]] = []
    missing = []
    for n, rec in enumerate(reader.records):
        if n % 1000 == 0:
            print(f"  {n}/{len(reader.records)}", flush=True)
        for p in osu_by_key.get(rec["mel_key"], []):
            try:
                bm = parser.parse_file(p)
            except Exception:                           # noqa: BLE001
                continue
            if bm.version == rec.get("version"):
                times.append(sorted(x.time for x in bm.notes if not x.is_long))
                break
        else:
            times.append([])
            missing.append(f"{rec['mel_key']} [{rec.get('version')}]")

    path = write_note_times(args.shards, reader.records, times)
    print(f"{len(reader.records) - len(missing)}/{len(reader.records)} maps -> {path} "
          f"({path.stat().st_size / 1e6:.1f} MB)")
    for m in missing[:20]:
        print(f"  no .osu found: {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
