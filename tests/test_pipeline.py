"""
test_pipeline.py

Full pipeline test: parse -> mel -> dataset -> DataLoader
Point this at your osu! Songs folder and it will:
  1. Find all taiko .osu files
  2. Parse a few to confirm the parser works
  3. Build a small dataset index (first 50 beatmapsets only)
  4. Extract one mel spectrogram
  5. Run a DataLoader batch

Run from project root:
    python test_pipeline.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

# ---- Config ----------------------------------------------------------------
OSU_SONGS_DIR = r"D:\osu!\Songs"   # your Songs folder
TEST_OUTPUT   = r"data\test_run"   # temp output for this test
MAX_SETS      = 50                 # only process 50 beatmapsets for speed
# ---------------------------------------------------------------------------


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def main():
    songs_path = Path(OSU_SONGS_DIR)
    output_path = Path(TEST_OUTPUT)
    output_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # STEP 1: Load taiko .osu files from cache
    # ------------------------------------------------------------------ #
    separator("STEP 1 — Loading taiko .osu files from cache")

    import json
    cache_path = Path("taiko_files_cache.json")

    if not cache_path.exists():
        print("ERROR: Cache not found. Run fast_scan.py first:")
        print("  python fast_scan.py")
        return

    t0 = time.time()
    taiko_files = [Path(p) for p in json.loads(cache_path.read_text())]
    elapsed = time.time() - t0

    print(f"Loaded {len(taiko_files)} taiko .osu files in {elapsed:.2f}s")
    if not taiko_files:
        print("ERROR: Cache is empty. Delete it and re-run fast_scan.py")
        return

    print("Sample files:")
    for f in taiko_files[:5]:
        print(f"  {f.parent.name[:60]} / {f.name}")


    # ------------------------------------------------------------------ #
    # STEP 2: Parse a few beatmaps
    # ------------------------------------------------------------------ #
    separator("STEP 2 — Parsing sample beatmaps")

    from taiko.data.osu_parser import OsuTaikoParser

    parser = OsuTaikoParser()
    parse_errors = 0

    for osu_file in taiko_files[:10]:
        try:
            bm = parser.parse_file(osu_file)
            print(
                f"  OK  [{bm.star_rating:.1f}*] {bm.title[:40]:<40} "
                f"| {bm.version[:20]:<20} "
                f"| {bm.note_count} notes "
                f"| {bm.notes_per_second:.1f} nps "
                f"| don:{bm.don_ratio:.0%}"
            )
        except Exception as e:
            print(f"  ERR {osu_file.name}: {e}")
            parse_errors += 1

    print(f"\nParsed 10 files — {parse_errors} errors")


    # ------------------------------------------------------------------ #
    # STEP 3: Tokenize one beatmap
    # ------------------------------------------------------------------ #
    separator("STEP 3 — Tokenizer")

    from taiko.data.tokenizer import TaikoTokenizer, TaikoVocabulary

    vocab     = TaikoVocabulary()
    tokenizer = TaikoTokenizer(vocab)

    bm = parser.parse_file(taiko_files[0])
    tok = tokenizer.encode(bm)

    print(f"Vocabulary size:     {len(vocab)}")
    print(f"Conditioning tokens: {tok.conditioning_ids}")
    print(f"  decoded:           {[vocab.decode(i) for i in tok.conditioning_ids]}")
    print(f"Sequence length:     {len(tok.token_ids)} tokens")
    print(f"First 12 tokens:     {tok.token_ids[:12]}")
    print(f"  decoded:           {[vocab.decode(i) for i in tok.token_ids[:12]]}")


    # ------------------------------------------------------------------ #
    # STEP 4: Mel spectrogram extraction
    # ------------------------------------------------------------------ #
    separator("STEP 4 — Mel spectrogram")

    try:
        import torchaudio
        backend = "torchaudio"
    except ImportError:
        try:
            import librosa
            backend = "librosa"
        except ImportError:
            print("ERROR: Install torchaudio or librosa:")
            print("  pip install torchaudio   (recommended)")
            print("  pip install librosa soundfile")
            return

    print(f"Audio backend: {backend}")

    from taiko.data.audio import MelExtractor

    extractor   = MelExtractor()
    audio_path  = taiko_files[0].parent / bm.audio_filename

    # Find audio file (case-insensitive, handle missing)
    if not audio_path.exists():
        candidates = list(taiko_files[0].parent.glob("*.mp3")) + \
                     list(taiko_files[0].parent.glob("*.ogg"))
        if candidates:
            audio_path = candidates[0]
        else:
            print(f"WARNING: No audio found in {taiko_files[0].parent}, skipping mel test")
            audio_path = None

    if audio_path:
        print(f"Extracting mel from: {audio_path.name}")
        t0 = time.time()
        mel = extractor.extract(audio_path)
        elapsed = time.time() - t0
        print(f"Mel shape:     {mel.shape}   (expected: [128, ~T])")
        print(f"Mel range:     [{mel.min():.3f}, {mel.max():.3f}]  (expected: [-1, 1])")
        print(f"Ms per frame:  {extractor.ms_per_frame:.2f}ms  (expected: ~10ms)")
        print(f"Extraction time: {elapsed:.2f}s")

        # Verify alignment
        from taiko.data.audio import verify_alignment
        align = verify_alignment(mel, bm.duration_ms)
        status = "OK" if align["aligned"] else "WARNING — large offset"
        print(f"Alignment:     {status} (diff={align['diff_ms']:.0f}ms)")


    # ------------------------------------------------------------------ #
    # STEP 5: Build dataset index (first MAX_SETS beatmapsets only)
    # ------------------------------------------------------------------ #
    separator(f"STEP 5 — Building dataset index (first {MAX_SETS} beatmapsets)")

    from taiko.data.dataset import DatasetBuilder

    builder = DatasetBuilder(
        raw_dir=songs_path,
        processed_dir=output_path,
        mel_extractor=MelExtractor(),
        tokenizer=TaikoTokenizer(),
    )

    # Monkey-patch to only process MAX_SETS directories
    original_iterdir = Path.iterdir
    _dirs_seen = [0]
    def limited_iterdir(self):
        items = list(original_iterdir(self))
        dirs  = [i for i in items if i.is_dir()]
        files = [i for i in items if not i.is_dir()]
        return iter(files + dirs[:MAX_SETS])

    # Simpler: just limit the beatmapset list manually
    all_bms_dirs = sorted([d for d in songs_path.iterdir() if d.is_dir()])
    test_dirs    = all_bms_dirs[:MAX_SETS]
    print(f"Processing {len(test_dirs)} beatmapset folders...")

    # Temporarily symlink / process only the test subset by building manually
    import json
    from taiko.data.audio import save_mel, load_mel
    from taiko.data.dataset import BeatmapRecord

    mel_dir    = output_path / "mels"
    index_path = output_path / "index.jsonl"
    mel_dir.mkdir(exist_ok=True)

    records = []
    errors  = []

    for bms_dir in test_dirs:
        # Find audio
        audio = None
        for ext in (".mp3", ".ogg", ".wav", ".flac"):
            found = list(bms_dir.glob(f"*{ext}"))
            if found:
                audio = found[0]
                break
        if audio is None:
            continue

        # Mel (skip if already cached)
        mel_path = mel_dir / f"{bms_dir.name[:80]}.npz"  # truncate long names
        if mel_path.exists():
            mel = load_mel(mel_path)
        else:
            try:
                mel = MelExtractor().extract(audio)
                save_mel(mel, mel_path)
            except Exception as e:
                errors.append(f"{bms_dir.name}: {e}")
                continue

        # Parse each .osu in this folder
        for osu_path in bms_dir.glob("*.osu"):
            try:
                bm = parser.parse_file(osu_path)
            except Exception:
                continue
            if bm.note_count < 30:
                continue

            try:
                tok = TaikoTokenizer().encode(bm)
            except Exception:
                continue

            records.append(BeatmapRecord(
                beatmap_id=bm.beatmap_id,
                beatmap_set_id=bm.beatmap_set_id,
                audio_path=str(audio),
                mel_path=str(mel_path),
                conditioning_ids=tok.conditioning_ids,
                token_ids=tok.token_ids,
                duration_ms=bm.duration_ms,
                star_rating=bm.star_rating,
                title=bm.title,
                version=bm.version,
                note_count=bm.note_count,
            ))

    with open(index_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict()) + "\n")

    print(f"Index built: {len(records)} beatmaps | {len(errors)} errors")
    if errors:
        print(f"First 3 errors: {errors[:3]}")


    # ------------------------------------------------------------------ #
    # STEP 6: TaikoDataset + DataLoader
    # ------------------------------------------------------------------ #
    separator("STEP 6 — TaikoDataset + DataLoader")

    if len(records) == 0:
        print("No records — skipping DataLoader test")
        return

    try:
        import torch
        from torch.utils.data import DataLoader
        from taiko.data.dataset import TaikoDataset, taiko_collate_fn
    except ImportError:
        print("ERROR: torch not installed. pip install torch")
        return

    ds = TaikoDataset(index_path, training=True)
    print(f"Dataset size: {len(ds)} samples")

    # Single sample
    sample = ds[0]
    print(f"\nSingle sample:")
    print(f"  mel shape:       {sample['mel'].shape}")
    print(f"  conditioning:    {sample['conditioning'].tolist()}")
    print(f"  token_ids[:8]:   {sample['token_ids'][:8].tolist()}")
    print(f"  real tokens:     {sample['token_mask'].sum().item()} / {len(sample['token_mask'])}")
    print(f"  star rating:     {sample['star_rating'].item():.2f}")

    # DataLoader batch
    loader = DataLoader(
        ds,
        batch_size=4,
        shuffle=True,
        num_workers=0,   # 0 for Windows (avoids multiprocessing issues)
        collate_fn=taiko_collate_fn,
    )

    print(f"\nDataLoader batch (batch_size=4):")
    t0 = time.time()
    batch = next(iter(loader))
    elapsed = time.time() - t0

    print(f"  mel:          {batch['mel'].shape}       dtype={batch['mel'].dtype}")
    print(f"  conditioning: {batch['conditioning'].shape}  dtype={batch['conditioning'].dtype}")
    print(f"  token_ids:    {batch['token_ids'].shape}   dtype={batch['token_ids'].dtype}")
    print(f"  token_mask:   {batch['token_mask'].shape}   dtype={batch['token_mask'].dtype}")
    print(f"  Batch load time: {elapsed:.3f}s")

    separator("ALL STEPS PASSED — Pipeline is working!")
    print(f"  Output written to: {output_path}")
    print(f"  You can now run the full dataset build with:")
    print(f'  python scripts/process_dataset.py --input "{OSU_SONGS_DIR}" --output data/processed')


if __name__ == "__main__":
    main()
