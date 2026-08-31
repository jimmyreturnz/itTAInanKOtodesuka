# tAIkoMapper — diagrams

Mermaid views of the pipeline and the model. Numbers are the `p1` profile at
`--window-frames 1536` (30.7 s) unless noted. Source of truth is the code:
`taiko/model/`, `taiko/data/frames.py`, `taiko/model/model_config.py`.

---

## 1. End to end — audio in, `.osz` out

```mermaid
flowchart LR
    subgraph offline["Offline, once"]
        songs["osu! Songs/<br/>ranked .osu + audio"]
        pack["scripts/pack_dataset.py<br/>--ranked-only"]
        shards["data/processed/shards/<br/>mels.dat 6.7 GB<br/>charts.npz, index.json<br/>10,566 + 540 maps"]
        kaggle["Kaggle dataset<br/>taiko-shards"]
        songs --> pack --> shards --> kaggle
    end

    subgraph train["Training, Kaggle 2x T4"]
        s1["Stage 1<br/>train_autoencoder.py<br/>Gate A: onset F1 >= 0.98"]
        s2["Stage 2<br/>train_diffusion.py<br/>Gate B: onset F1 > 0.40"]
        kaggle --> s1 --> s2
    end

    subgraph infer["Inference, local GTX 1650"]
        audio["song.mp3"]
        gen["scripts/generate.py"]
        osz["outputs/*.osz"]
        audio --> gen --> osz
    end

    s1 -- "frozen VAE" --> gen
    s2 -- "U-Net + audio encoder" --> gen
```

Gate A is a permanent ceiling: whatever the autoencoder loses, no amount of
stage-2 training recovers.

---

## 2. The three tensors, and the one time grid

```mermaid
flowchart TD
    subgraph grid["frames.py — one frame is 20 ms, 50 fps, hop 441 @ 22050 Hz"]
        mel["mel [128, T]<br/>log-mel, referenced to clip peak"]
        chart["chart [6, T]<br/>don, kat, big_don, big_kat, roll, denden"]
        timing["timing [3, T]<br/>sin, cos, downbeat"]
    end

    audio["audio"] --> mel
    osu[".osu"] --> chart
    bpm["red lines / BPM"] --> timing

    mel --- note1["mel frame i and chart frame i<br/>describe the same millisecond.<br/>assert_aligned at every boundary."]
    chart --- note1

    style note1 fill:#fff3cd,stroke:#856404,color:#856404
```

Timing is an **input, not an output** — tempo is known at generation time, and
sin/cos phase carries the sub-frame precision that 1/4 and 1/6 snaps need.
A pulse train cannot.

---

## 3. Stage 1 — the chart autoencoder (frozen afterwards)

```mermaid
flowchart LR
    c["chart<br/>[6, 1536]"] --> enc["Encoder<br/>channel_mult 1,1,2,2,4<br/>4 downsamples = 16x"]
    enc --> dist["DiagonalGaussian<br/>mean, logvar"]
    dist -- "mode() x latent_scale" --> z["z [16, 96]"]
    z --> dec["Decoder<br/>nearest-upsample + conv"]
    dec --> logits["logits [6, 1536]"]
    logits -- sigmoid --> out["chart"]

    z -.-> stage2["to stage 2<br/>(encoder and decoder frozen,<br/>requires_grad = False)"]

    style stage2 fill:#e7f3ff,stroke:#0366d6,color:#0366d6
```

5.33M params. `latent_scale` is a registered buffer, not a config value — a
diffusion model trained against the wrong latent scale fails at every timestep
in a way the loss curve does not show.

---

## 4. Stage 2 — latent diffusion, one training step

```mermaid
flowchart TD
    chart["chart [6, 1536]"] --> vae["frozen VAE encode"] --> z0["z0 [16, 96]"]
    t["t ~ U(0, 1000)"] --> addn
    noise["eps ~ N(0, I)"] --> addn["scheduler.add_noise<br/>cosine beta schedule"]
    z0 --> addn --> zt["z_t [16, 96]"]

    mel["mel [128, 1536]"] --> wave["MelEncoder1D"] --> feats["audio features<br/>one per U-Net level"]
    timing["timing [3, 1536]"] --> pool["adaptive_avg_pool1d"] --> tl["timing [3, 96]"]

    cond["difficulty, avg_nps, peak_nps,<br/>style (4), motif (16) + mask"] --> ce["ConditionEmbedding<br/>-> 256-d"]
    t --> te["TimestepEmbedding<br/>-> 256-d"]
    drop["CFG drop_mask<br/>stratified, exactly floor(B*p)"] --> ce

    zt --> unet["TaikoDiffusionUNet"]
    feats --> unet
    tl --> unet
    ce --> unet
    te --> unet

    unet --> v["v-prediction [16, 96]"]
    target["target_for(z0, eps, t)"] --> mse
    v --> mse["masked MSE<br/>padding frames excluded"]
    mse --> loss["loss"]

    style drop fill:#fff3cd,stroke:#856404,color:#856404
```

The motif vector is measured on the same window the model must generate, so it
is nearly a summary of the answer. Per-dimension dropout is what stops the model
reading the answer off its own conditioning; `evaluate.py --use-reference-motif`
is the leakage probe.

---

## 5. The audio encoder — why it has a strided stem

```mermaid
flowchart LR
    mel["mel [128, 1536]"] --> stem["strided conv stem<br/>learned pooling to 96 frames"]
    mel --> flux["SpectralFlux<br/>sum max(0, mel[b,t] - mel[b,t-1])<br/>per-clip peak normalised"]
    flux --> mp["max_pool1d /16"]
    flux --> ap["avg_pool1d /16"]
    mp --> onset["OnsetStem conv"]
    ap --> onset
    stem --> merge["1x1 merge"]
    onset --> merge

    merge --> l0["level 0<br/>128 ch, 96 frames"]
    l0 --> l1["level 1<br/>128 ch, 48"]
    l1 --> l2["level 2<br/>256 ch, 24"]
    l2 --> l3["level 3<br/>512 ch, 12"]

    style flux fill:#e7f3ff,stroke:#0366d6,color:#0366d6
```

Max-pooling the flux is the point: a 16x reduction by strided convolution alone
smooths away exactly the transients that decide where notes go. The alternative
— interpolating audio across the resolution gap inside the U-Net — point-samples
most of the onset information into nothing.

---

## 6. Inside the U-Net (`p1`: base 128, mult 1,2,3,4, 2 res blocks/level)

```mermaid
flowchart TD
    zin["z_t [16, 96]"] --> ci["conv_in -> 128 ch"]

    ci --> e0["L0  128 ch, 96 fr"]
    e0 --> e1["L1  256 ch, 48 fr"]
    e1 --> e2["L2  384 ch, 24 fr"]
    e2 --> e3["L3  512 ch, 12 fr"]

    e3 --> m1["ResBlock"] --> att["self-attention<br/>12 frames, whole window<br/>for almost nothing"] --> m2["ResBlock"]

    m2 --> d3["L3 up"]
    d3 --> d2["L2 up"]
    d2 --> d1["L1 up"]
    d1 --> d0["L0"]
    d0 --> co["conv_out -> [16, 96]"]

    e3 -. skip .-> d3
    e2 -. skip .-> d2
    e1 -. skip .-> d1
    e0 -. skip .-> d0

    aud["audio features<br/>per level"] --> e0 & e1 & e2 & e3
    aud --> d3 & d2 & d1 & d0
    tim["timing, 3 ch<br/>per level"] --> e0 & e1 & e2 & e3
    tim --> d3 & d2 & d1 & d0
    emb["time + condition emb<br/>256-d, into every ResBlock"] --> e0 & e1 & e2 & e3 & m1 & m2 & d3 & d2 & d1 & d0

    style att fill:#e7f3ff,stroke:#0366d6,color:#0366d6
```

Each level is: concat `[h, audio_lvl, timing]` -> 1x1 proj -> N ResBlocks ->
S4 block (long-range, depthwise conv + SiLU gating) -> down/up. Audio and
timing are re-injected at **every** level rather than added once at the input.

---

## 7. Generation — arbitrary song length

```mermaid
flowchart TD
    mp3["song.mp3"] --> melg["mel [128, T]"]
    bpmin["BPM + offset<br/>(or --timing-from map.osu)"] --> timg["timing [3, T]"]
    ui["difficulty, style,<br/>nps, motif preset"] --> condg["conditioning"]

    melg --> plan["plan_windows(T, 1536, overlap)<br/>final window pulled back<br/>to land on the end"]
    plan --> w["windows"]

    subgraph ddim["for each of ~50 DDIM steps"]
        direction TB
        pred["U-Net forward per window<br/>cond and uncond branches"]
        cfg["real CFG:<br/>uncond + s * (cond - uncond)"]
        blend["raised-cosine taper,<br/>averaged into ONE shared latent"]
        pred --> cfg --> blend
    end

    w --> ddim
    timg --> ddim
    condg --> ddim

    ddim --> zf["z [16, T/16]"]
    zf --> dec["frozen VAE decode + sigmoid"]
    dec --> thr["threshold + beat snap<br/>+ timing refine"]
    thr --> writer["osu_writer<br/>SV 1.0, red lines only"]
    writer --> osz[".osz"]

    style blend fill:#e7f3ff,stroke:#0366d6,color:#0366d6
```

MultiDiffusion, not stitching: overlapping windows denoise **one shared latent**
and are averaged at every step, so there is no seam to blend afterwards. 10 min
= 30,000 frames, under `MAX_FRAMES = 45,000`; cost is linear in length.

Known slow path: `sampling.py` runs those windows sequentially at batch 1, and
CFG as a second separate forward — a 5-min song is ~1,900 batch-1 U-Net
forwards, entirely launch-overhead bound. See work item 6 in `DIRECTION.md`.
