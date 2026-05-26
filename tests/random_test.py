import json
from pathlib import Path

index = Path("data/processed/colab_index.jsonl")
records = [json.loads(l) for l in index.read_text(encoding="utf-8").splitlines() if l.strip()]

high_sr = [r for r in records if r.get("difficulty", 0) > 10]
print(f"Maps with SR > 10: {len(high_sr)}")
for r in sorted(high_sr, key=lambda x: x["difficulty"], reverse=True):
    print(f"  SR={r['difficulty']:.1f} | {r['title']} [{r['version']}] | {r['style_name']}")
    
import torch

checkpoint_path = 'D:\Jimmy\CodingProject\itTAInanKOtodesuka\checkpoints\diffusion\step_0166000.pt' # adjust name if it varies

try:
    print("Opening checkpoint metadata...")
    # map_location='cpu' prevents GPU memory overhead, weights_only=True peaks at keys safely
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("\n📦 Top-Level Keys inside your checkpoint:")
    for key in checkpoint.keys():
        # Check the data type of the key
        data_type = type(checkpoint[key])
        print(f" ->  {key} ({data_type})")
        
        # If it's an integer or float, let's print its value directly (e.g., current step)
        if data_type in [int, float, str]:
            print(f"     Value: {checkpoint[key]}")
            
except Exception as e:
    print(f"❌ Failed to parse checkpoint file: {e}")