# 🥁 Taiko Diffusion

> AI-powered osu! taiko beatmap generator — audio in, `.osu` file out.

## Architecture

```
Audio (.mp3/.ogg)
      │
      ▼
 Mel Spectrogram                   Conditioning tokens
 [T × 128 mel bins]          [difficulty, style, don_ratio, ...]
      │                                    │
      ▼                                    ▼
 Whisper Encoder ──────────── Decoder (Autoregressive Transformer)
                                           │
                                           ▼
                              Token sequence:
                              TIME_450 HIT_DON
                              TIME_600 HIT_KAT
                              TIME_750 BIG_DON
                              TIME_900 ROLL_START
                              TIME_1200 ROLL_END
                                           │
                                           ▼
                                   .osu file (Mode:1)
```

## Vocabulary

| Token class | Examples | Count |
|---|---|---|
| TIME_* | TIME_0 … TIME_N (10ms quantized) | dynamic |
| Hit types | HIT_DON, HIT_KAT, BIG_DON, BIG_KAT | 4 |
| Long notes | ROLL_START, ROLL_END, DENDEN_START, DENDEN_END | 4 |
| Special | SOS, EOS, PAD, UNK | 4 |
| Conditioning | DIFF_*, DENSITY_*, STYLE_* | ~50 |

## Project Phases

- [x] **Phase 1** — Data pipeline (`.osu` parser, tokenizer, mel spectrogram, dataset)
- [ ] **Phase 2** — Model (Whisper encoder + autoregressive decoder)
- [ ] **Phase 3** — Training loop (conditioning, scheduler, checkpointing)
- [ ] **Phase 4** — Inference (generate + decode to `.osu`)
- [ ] **Phase 5** — WebUI (Gradio)

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Process your beatmap dataset
python scripts/process_dataset.py --input data/raw --output data/processed

# 3. Train
python scripts/train.py --config configs/base.yaml

# 4. Generate
python scripts/generate.py --audio my_song.mp3 --difficulty 4.5 --output out/
```

## Hardware

Designed to train on a single consumer GPU (RTX 3060–4080, 10–16GB VRAM).
Target model size: ~100M parameters.
Estimated training time on RTX 3080: ~2–4 days on 10k maps.
