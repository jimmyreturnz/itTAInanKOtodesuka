"""
Train-step cost of a whole-song beat-token transformer against the p1 U-Net,
on the same GPU, so the U-Net's known Kaggle cost calibrates the token model's.
Random inputs; nothing is learned.  Run from the itTAInanKOtodesuka root.
"""
import sys, time
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, ".")
from taiko.model.diffusion import TaikoDiffusion

dev = "cuda"

def bench(step, n=12, warm=3):
    for _ in range(warm): step()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): step()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n

# --- p1 U-Net, 1536-frame window, frozen VAE as in training ---------------
B_U = 8
m = TaikoDiffusion(autoencoder_ckpt="checkpoints/autoencoder/best.pt", profile="p1", verbose=False).to(dev)
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], 1e-4)
sc = torch.amp.GradScaler()
W = 1536
batch = dict(chart=torch.rand(B_U, 6, W, device=dev).round(), mel=torch.randn(B_U, 128, W, device=dev),
             timing=torch.randn(B_U, 3, W, device=dev), difficulty=torch.rand(B_U, device=dev) * 7,
             style=torch.zeros(B_U, dtype=torch.long, device=dev), valid_mask=torch.ones(B_U, W, device=dev),
             avg_nps=torch.rand(B_U, device=dev) * 8, peak_nps=torch.rand(B_U, device=dev) * 12)
def unet_step():
    with torch.autocast("cuda", dtype=torch.float16):
        loss, _ = m(**batch)
    opt.zero_grad(); sc.scale(loss).backward(); sc.step(opt); sc.update()
tu = bench(unet_step)
print(f"p1 U-Net: {tu*1000:.0f} ms/step at batch {B_U} -> {tu/B_U*1000:.1f} ms per 30.7 s window")
del m, opt; torch.cuda.empty_cache()

# --- beat-token model: whole song, audio stem over the full mel ------------
class TokenModel(nn.Module):
    def __init__(self, d=512, layers=10, vocab=1024 + 8, n_col=4 * 48):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(128, 256, 5, padding=2), nn.SiLU(),
                                  nn.Conv1d(256, 256, 5, padding=2), nn.SiLU(),
                                  nn.Conv1d(256, d, 5, padding=2))
        self.tok = nn.Embedding(vocab + 1, d)          # +1 = [MASK]
        enc = nn.TransformerEncoderLayer(d, 8, 4 * d, batch_first=True, norm_first=True, dropout=0.0)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d, vocab)
        self.col = nn.Linear(d, n_col)
    def forward(self, mel, beat_frames, tokens):
        a = self.stem(mel)                                     # [B, d, T]
        # per beat: mean of the audio frames the beat spans (stand-in for 48-tick sampling)
        idx = beat_frames.clamp(max=a.shape[-1] - 1)           # [B, N, K]
        feats = torch.gather(a, 2, idx.flatten(1)[:, None].expand(-1, a.shape[1], -1))
        feats = feats.view(a.shape[0], a.shape[1], *idx.shape[1:]).mean(-1).transpose(1, 2)
        h = self.tr(self.tok(tokens) + feats)
        return self.head(h), self.col(h)

N, T, K = 1536, 24000, 16          # 1536 beats ~ 8 min at 190 BPM; 16 frames sampled per beat
tm = TokenModel().to(dev)
print(f"token model params: {sum(p.numel() for p in tm.parameters())/1e6:.1f}M")
opt = torch.optim.AdamW(tm.parameters(), 1e-4)
for B_T in (1, 2):
    mel = torch.randn(B_T, 128, T, device=dev)
    bf = (torch.arange(N, device=dev)[:, None] * (T // N) + torch.arange(K, device=dev))[None].expand(B_T, -1, -1).contiguous()
    tok = torch.randint(0, 1033, (B_T, N), device=dev)
    def token_step():
        with torch.autocast("cuda", dtype=torch.float16):
            lg, cl = tm(mel, bf, tok)
            loss = F.cross_entropy(lg.float().flatten(0, 1), tok.clamp(max=1031).flatten()) + cl.float().mean()
        opt.zero_grad(); sc.scale(loss).backward(); sc.step(opt); sc.update()
    tt = bench(token_step, n=6, warm=2)
    print(f"token: {tt*1000:.0f} ms/step at batch {B_T} songs -> {tt/B_T*1000:.0f} ms per whole song "
          f"= {tt/B_T/tu*B_U:.1f} U-Net windows; peak mem {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
