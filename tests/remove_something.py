import json
from pathlib import Path

index  = Path("data/processed/colab_index.jsonl")
base   = Path("data/processed")
records = [json.loads(l) for l in index.read_text(encoding="utf-8").splitlines() if l.strip()]

# Exact maps to remove — (title, version)
to_remove = {
    ("Party Without Me", "AGTS Aces"),
    ("Unwelcome School", "Sans"),
    ("Fastest Crash", "Hyper Speed Decay"),
    ("Acorn", "Quercus"),
    ("Atarashii MEME ga FOOL MOON Nanode Kuyashiiwa", "dkdk Oni"),
    ("Atarashii MEME ga FOOL MOON Nanode Kuyashiiwa", "kdkd Oni"),
    ("ANOMALY", "?"),
}

keep    = []
removed = 0

for r in records:
    key = (r.get("title", ""), r.get("version", ""))
    if key in to_remove:
        for path_key in ["mel_path", "tensor_path"]:
            p = base / r[path_key].replace("\\", "/")
            if p.exists():
                p.unlink()
                print(f"Deleted: {p.name}")
        removed += 1
    else:
        keep.append(r)

index.write_text("\n".join(json.dumps(r) for r in keep) + "\n", encoding="utf-8")
print(f"\nRemoved {removed} maps, {len(keep)} remain")