import json
from pathlib import Path

INDEX_PATH = Path("data/processed/colab_index.jsonl")
BASE       = Path("data/processed")

records = [json.loads(l) for l in INDEX_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
before  = len(records)

valid = [r for r in records
        if (BASE / r["mel_path"]).exists()
        and (BASE / r["tensor_path"]).exists()]

INDEX_PATH.write_text("\n".join(json.dumps(r) for r in valid) + "\n")
print(f"Removed {before - len(valid)} incomplete records, {len(valid)} remain")