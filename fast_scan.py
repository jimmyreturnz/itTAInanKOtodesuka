"""
fast_scan.py

First run:  scans everything, saves cache to taiko_files_cache.json
Next runs:  loads cache instantly (~0.1s)

Run from project root:
    python fast_scan.py
"""

import json
import time
from pathlib import Path

OSU_SONGS_DIR = r"D:\osu!\Songs"
CACHE_FILE    = "taiko_files_cache.json"


def scan_and_cache(root: Path, cache_path: Path) -> list:
    print(f"First run — scanning {root}")
    print("This will take a few minutes. Won't happen again.")

    results = []
    checked = 0

    for osu_file in root.rglob("*.osu"):
        checked += 1
        if checked % 500 == 0:
            print(f"  Checked {checked} files, found {len(results)} taiko so far...")
        try:
            with open(osu_file, "r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f):
                    if i > 30:
                        break
                    if line.strip() in ("Mode: 1", "Mode:1"):
                        results.append(str(osu_file))
                        break
        except OSError:
            continue

    cache_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Cached {len(results)} taiko files to {cache_path}")
    return results


def load_or_scan(root: Path, cache_path: Path) -> list:
    if cache_path.exists():
        print(f"Loading cache from {cache_path} ...")
        t0 = time.time()
        paths = [Path(p) for p in json.loads(cache_path.read_text())]
        print(f"Loaded {len(paths)} taiko files in {time.time()-t0:.2f}s")

        missing = sum(1 for p in paths if not p.exists())
        if missing:
            print(f"Warning: {missing} cached paths no longer exist (maps deleted/moved)")
            print("Delete taiko_files_cache.json to re-scan.")
        return paths
    else:
        paths_str = scan_and_cache(root, cache_path)
        return [Path(p) for p in paths_str]


if __name__ == "__main__":
    root       = Path(OSU_SONGS_DIR)
    cache_path = Path(CACHE_FILE)

    t0 = time.time()
    files = load_or_scan(root, cache_path)
    print(f"Total time: {time.time()-t0:.2f}s")
    print(f"\nFirst 5 files:")
    for f in files[:5]:
        print(f"  {f.parent.name[:60]} / {f.name}")
