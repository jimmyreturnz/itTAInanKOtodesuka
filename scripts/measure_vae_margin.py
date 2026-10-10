"""
Does the frozen chart VAE lose colour/finisher, and how thin is snap identity
in its latent?  Ranked validation records only (shard index, split_indices).

For each record: chart -> encode (mode, scaled) -> + sigma * N(0,1) -> decode
-> decode_on_grid.  Reference = decode_on_grid of the ground-truth chart, so
the decoder's own ceiling cancels out.  Reports, per SR band and sigma:
exact-ms note match, colour (don/kat) and size (big/small) agreement on
matched notes, fast-snap switches per 100 hits (ref vs recon).
Run from the itTAInanKOtodesuka root.
"""
import sys, random
import numpy as np, torch
sys.path.insert(0, ".")
from taiko.data.shards import ShardReader, decode_timing_points
from taiko.data.preprocessed_dataset import split_indices
from taiko.data.decode import decode_on_grid
from taiko.data.grid import Grid
from taiko.eval.metrics import snap_switch_rate
from taiko.model.autoencoder import ChartAutoencoder, AutoencoderConfig

SIGMAS = (0.0, 0.1, 0.2, 0.3, 0.5)
PER_BAND = int(sys.argv[1]) if len(sys.argv) > 1 else 20
BANDS = {"<4*": (0, 4.0), "5.5*+": (5.5, 99)}

reader = ShardReader("data/processed/shards")
_, val = split_indices(reader, ranked_only=True)
ck = torch.load("checkpoints/autoencoder/best.pt", map_location="cpu", weights_only=False)
vae = ChartAutoencoder(AutoencoderConfig(**ck["config"]))
vae.load_state_dict(ck["model"]); vae = vae.cuda().eval()

def hits(bm):
    return {n.time: n.note_type for n in bm.notes if not n.is_long}

rng = random.Random(0)
for band, (lo, hi) in BANDS.items():
    pool = [i for i in val if lo <= reader.records[i]["difficulty"] < hi]
    rng.shuffle(pool)
    acc = {s: dict(m=0, n=0, r=0, col=0, size=0, sw=[], sw_ref=[]) for s in SIGMAS}
    for i in pool[:PER_BAND]:
        rec = reader.records[i]
        T = reader.chart_length(i)
        chart = reader.chart_window(i, 0, T).astype(np.float32)
        pad = (-T) % 16
        x = torch.from_numpy(np.pad(chart, ((0, 0), (0, pad))))[None].cuda()
        pts = decode_timing_points(rec["timing_points"])
        grid = Grid(pts)
        ref_bm = decode_on_grid(chart, pts)
        ref = hits(ref_bm)
        with torch.no_grad():
            z = vae.encode(x).mode()
            g = torch.Generator(device="cuda").manual_seed(i)
            eps = torch.randn(z.shape, generator=g, device="cuda")
            for s in SIGMAS:
                p = torch.sigmoid(vae.decode(z + s * eps, target_length=x.shape[-1]))[0, :, :T].cpu().numpy()
                bm = decode_on_grid(p, pts)
                got = hits(bm)
                a = acc[s]
                both = ref.keys() & got.keys()
                a["m"] += len(both); a["n"] += len(got); a["r"] += len(ref)
                a["col"] += sum(("kat" in ref[t]) == ("kat" in got[t]) for t in both)
                a["size"] += sum(("big" in ref[t]) == ("big" in got[t]) for t in both)
                a["sw"].append(snap_switch_rate(bm.notes, grid))
                a["sw_ref"].append(snap_switch_rate(ref_bm.notes, grid))
    print(f"\n{band}  ({min(PER_BAND, len(pool))} val maps)")
    print("sigma  exact_recall  exact_prec  colour_agree  size_agree  switch/100 (ref)")
    for s in SIGMAS:
        a = acc[s]
        print(f"{s:4.1f}   {a['m']/max(a['r'],1):.4f}       {a['m']/max(a['n'],1):.4f}     "
              f"{a['col']/max(a['m'],1):.4f}        {a['size']/max(a['m'],1):.4f}     "
              f"{np.mean(a['sw']):.2f} ({np.mean(a['sw_ref']):.2f})")
