"""
scripts/train_diffusion.py

Stage 2. Trains the latent diffusion model against a frozen autoencoder.

    python scripts/train_diffusion.py --ae checkpoints/autoencoder/best.pt
    python scripts/train_diffusion.py --resume checkpoints/diffusion/last.pt

Built for interruption. A Kaggle session caps at about 12 hours and this stage
wants 150-250 GPU-hours, so it will be resumed roughly twenty times; every
checkpoint carries the optimiser, the scaler, the EMA, the step count and the
position within the epoch, and --resume genuinely picks up there.

Surviving the kill
------------------
Three things go wrong on a preemptible machine, and each has its own defence:

  The session ends. --max-hours stops early and saves; SIGTERM and the
  notebook's interrupt button now do the same rather than discarding the work
  since the last save.

  The OOM killer arrives. SIGKILL cannot be caught, so the only defence is
  having saved recently: --save-every-min writes last.pt on a clock. Step
  counts are not a unit of risk -- at 2.6 s/step, "--save-every 500" is a
  22-minute blast radius, and a kill at step 550 discards 50 steps of work
  while the log says the checkpoint is current.

  The kill lands mid-write. Checkpoints are written to a temporary file and
  renamed, so last.pt is always either the previous checkpoint or the new one.
  A truncated last.pt used to end the *next* session before it began.

Host memory
-----------
--mel-io read is the default here. A memmap's touched pages are resident pages,
so sampling random windows walks RSS up by the whole size of mels.dat over an
hour or two and the run dies of a leak that is not in the Python heap. See
taiko/data/shards.py.

Two things were then still missing, and between them they cost two Kaggle
sessions. The first is a memory figure worth reading: the log used to print the
sum of RSS over this process and its dataloader workers, which counts every
page they share once per worker and reported a run at 30 GB whose real
footprint was a fraction of that. It now prints PSS, split between this process
and its workers, alongside the cgroup's own usage and limit -- the number the
container is actually killed on. The second is doing something about it:
--min-free-gb saves and exits with EXIT_LOW_MEMORY while there is still room to
write the checkpoint, so a supervisor restarts with --resume and the run loses
two minutes instead of ten hours.

Gate B: onset F1 above 0.40 against held-out audio. That is the alignment gate
-- it is what distinguishes a model listening to the music from one emitting
plausible rhythms. Run scripts/evaluate.py to measure it; a diffusion model
that fails Gate B is not fixed by more steps.

Multi-GPU: nn.DataParallel, which works inside a Kaggle notebook where
torchrun does not. The loss lives in TaikoDiffusion.forward(), so the wrapper
actually scatters -- the previous loop called unwrap(model).training_loss()
and reached through it, leaving the second T4 idle for every run ever done.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from taiko.data.frames import describe
from taiko.data.preprocessed_dataset import (
    WINDOW_FRAMES_DEFAULT, WindowedDataset, print_split_stats, split_indices,
)
from taiko.data.shards import MEL_IO_MODES, ShardReader
from taiko.model.diffusion import EMA, TaikoDiffusion
from taiko.model.model_config import PROFILES, get_profile
from taiko.train import (
    EXIT_LOW_MEMORY, CheckpointSaver, MemoryTrend, SaveTrigger, headroom_gb,
    install_stop_handlers, load_checkpoint, memory_line, memory_mb, memory_report,
)


def unwrap(model: nn.Module) -> TaikoDiffusion:
    return model.module if isinstance(model, nn.DataParallel) else model


def lr_at(step: int, warmup: int, total: int, peak: float, floor: float = 1e-6) -> float:
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    if total is None or step >= total:
        return floor
    progress = (step - warmup) / max(total - warmup, 1)
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def batch_to(batch: dict, device: torch.device) -> dict:
    """Only the tensors the model consumes, moved once."""
    keys = ("chart", "mel", "timing", "difficulty", "style", "valid_mask",
            "avg_nps", "peak_nps", "motif", "motif_mask")
    return {k: batch[k].to(device, non_blocking=True) for k in keys}


@torch.no_grad()
def validate(model, loader, device, max_batches: int, use_fp16: bool) -> float:
    """
    Mean loss over the first `max_batches` validation batches.

    The iterator is dropped explicitly rather than left to fall out of scope.
    Abandoning a half-consumed DataLoader keeps its prefetch queue -- and, with
    persistent workers, the workers themselves -- alive until the next garbage
    collection, which on a machine already short of RAM is the wrong moment to
    be holding twenty windows nobody will read.
    """
    model.eval()
    total, seen = 0.0, 0
    iterator = iter(loader)
    try:
        for _ in range(max_batches):
            try:
                batch = next(iterator)
            except StopIteration:
                break
            with torch.amp.autocast("cuda", enabled=use_fp16):
                loss, _ = model(**batch_to(batch, device))
            total += float(loss.mean())
            seen += 1
    finally:
        del iterator
        model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return total / max(seen, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae", type=Path, default=Path("checkpoints/autoencoder/best.pt"),
                    help="frozen autoencoder checkpoint from stage 1")
    ap.add_argument("--shards", type=Path, default=Path("data/processed/shards"))
    ap.add_argument("--out", type=Path, default=Path("checkpoints/diffusion"))
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--profile", default="p1", choices=sorted(PROFILES))
    ap.add_argument("--window-frames", type=int, default=WINDOW_FRAMES_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=None, help="per GPU; default from profile")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--samples-per-epoch", type=int, default=20_000)
    ap.add_argument("--lr", type=float, default=None, help="default from profile")
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--ema-decay", type=float, default=0.9995)
    ap.add_argument("--prediction-type", default="v", choices=["v", "epsilon"])
    ap.add_argument("--ranked-only", action="store_true",
                    help="train only on ranked maps (recommended for the final run)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--val-workers", type=int, default=0,
                    help="workers for the validation loader. 0 keeps two fewer "
                         "processes -- and two fewer prefetch queues -- resident "
                         "for a loader used once every --val-every steps")
    ap.add_argument("--prefetch-factor", type=int, default=2,
                    help="batches queued per worker; each one costs "
                         "batch x window x 128 floats of host RAM")
    ap.add_argument("--mel-io", default="read", choices=list(MEL_IO_MODES),
                    help="'read' preads each window and holds no pages; 'mmap' is "
                         "faster but its resident set grows to the size of mels.dat")
    ap.add_argument("--no-pin-memory", dest="pin_memory", action="store_false",
                    default=True,
                    help="do not page-lock loader batches. Pinned memory cannot "
                         "be swapped or reclaimed, and at this window a batch is "
                         "tens of MB; turn it off on a host short of RAM")
    ap.add_argument("--min-free-gb", type=float, default=2.0,
                    help="save and exit cleanly when this little host memory is "
                         "left, rather than waiting to be killed. Exits with "
                         f"code {EXIT_LOW_MEMORY} so a supervisor can restart "
                         "with --resume. 0 to disable")
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=1000,
                    help="save last.pt every N optimiser steps (0 to disable)")
    ap.add_argument("--save-every-min", type=float, default=10.0,
                    help="save last.pt every N minutes. This is the one that "
                         "bounds what an OOM kill costs; 0 to disable")
    ap.add_argument("--no-epoch-save", dest="epoch_save", action="store_false",
                    default=True, help="do not save at the end of every epoch")
    ap.add_argument("--no-epoch-val", dest="epoch_val", action="store_false",
                    default=True, help="do not validate at the end of every epoch")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--total-steps", type=int, default=None,
                    help="horizon for the cosine schedule. Defaults to "
                         "epochs x batches/epoch, and is carried in the checkpoint "
                         "so a resume cannot silently reshape the schedule")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop cleanly before a session limit, saving first")
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--fp32", dest="fp16", action="store_false")
    ap.add_argument("--single-gpu", action="store_true")
    ap.add_argument("--no-grad-checkpoint", action="store_true",
                    help="disable gradient checkpointing. It buys memory with "
                         "recomputation, which is a bad trade when the card is "
                         "empty: p1 at 8/GPU uses under 1 GiB of 15.")
    args = ap.parse_args()

    profile = get_profile(args.profile)
    if args.no_grad_checkpoint:
        profile = replace(profile, use_checkpoint=False)
    batch_per_gpu = args.batch_size or profile.per_gpu_batch
    lr = args.lr or profile.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Every batch is the same shape, so let cuDNN pick its fastest kernels
    # once instead of re-deciding per call. 5-15% on a conv-dominated run.
    torch.backends.cudnn.benchmark = True
    n_gpus = 0 if device.type == "cpu" else (1 if args.single_gpu else torch.cuda.device_count())
    use_fp16 = args.fp16 and device.type == "cuda"

    print(describe())
    print(f"Device: {device}  GPUs: {n_gpus}  fp16: {use_fp16}")
    for i in range(n_gpus):
        p = torch.cuda.get_device_properties(i)
        print(f"  cuda:{i}  {p.name}  {p.total_memory / 1024**3:.1f} GB")
    print(f"Profile: {profile.summary()}")

    if not args.ae.exists():
        print(f"ERROR: no autoencoder checkpoint at {args.ae}")
        print("  Run scripts/train_autoencoder.py first and clear Gate A.")
        return 1

    ae_ckpt = torch.load(args.ae, map_location="cpu", weights_only=False)
    gate_a = ae_ckpt.get("best_f1", 0.0)
    print(f"Autoencoder Gate A: onset F1 {gate_a:.4f}")
    if gate_a < 0.98:
        print("  WARNING: this autoencoder did not clear Gate A (0.98). Whatever")
        print("  it loses is a ceiling on the diffusion model, and no amount of")
        print("  diffusion training recovers it.")

    # ---- data ------------------------------------------------------------ #
    reader = ShardReader(args.shards, mel_io=args.mel_io)
    print(reader.describe_mel_io())
    train_idx, val_idx = split_indices(reader, val_ratio=0.05, ranked_only=args.ranked_only)
    print_split_stats(reader, train_idx, "Train")
    print_split_stats(reader, val_idx, "Val")

    train_ds = WindowedDataset(
        reader, train_idx, window_frames=args.window_frames,
        random_window=True, augment=True, samples_per_epoch=args.samples_per_epoch,
    )
    val_ds = WindowedDataset(
        reader, val_idx, window_frames=args.window_frames,
        random_window=False, augment=False,
    )

    total_batch = batch_per_gpu * max(n_gpus, 1)

    def loader_kwargs(workers: int, persistent: bool) -> dict:
        return dict(
            num_workers=workers,
            pin_memory=(device.type == "cuda" and args.pin_memory),
            persistent_workers=persistent and workers > 0,
            prefetch_factor=args.prefetch_factor if workers > 0 else None,
        )

    train_loader = DataLoader(train_ds, batch_size=total_batch, shuffle=True,
                              drop_last=True, **loader_kwargs(args.num_workers, True))
    # The validation loader is not persistent: it runs once every --val-every
    # steps, and keeping its workers (and their prefetch queues) alive in
    # between spends host RAM continuously to save a few seconds occasionally.
    val_loader = DataLoader(val_ds, batch_size=total_batch, shuffle=False,
                            **loader_kwargs(args.val_workers, False))

    sample_mb = total_batch * (128 + 6 + 3 + 1) * args.window_frames * 4 / 1024 ** 2
    print(f"Loader: {args.num_workers} train workers x {args.prefetch_factor} prefetch "
          f"= {sample_mb * args.num_workers * args.prefetch_factor:.0f} MB queued "
          f"({sample_mb:.0f} MB/batch), {args.val_workers} val workers, "
          f"pin_memory={args.pin_memory and device.type == 'cuda'}")

    # ---- model ------------------------------------------------------------ #
    model = TaikoDiffusion(
        autoencoder_ckpt=str(args.ae),
        profile=profile,          # the object, not the name -- --no-grad-checkpoint
                                  # modifies it and passing args.profile would
                                  # silently re-fetch the unmodified original
        prediction_type=args.prediction_type,
    )
    model.first_stage.check_window(args.window_frames)
    model = model.to(device)

    inner = unwrap(model)
    ema = EMA(inner.trainable_parameters(), decay=args.ema_decay)

    if n_gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(n_gpus)))
        print(f"DataParallel across {n_gpus} GPUs")

    trainable = sum(p.numel() for p in inner.trainable_parameters())
    print(f"Trainable: {trainable / 1e6:.1f}M | "
          f"batch {total_batch} ({batch_per_gpu}/GPU x {max(n_gpus, 1)}) "
          f"x {args.grad_accum} accum = {total_batch * args.grad_accum} effective")

    optimizer = torch.optim.AdamW(inner.trainable_parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    batches_per_epoch = len(train_loader)
    default_total = args.epochs * batches_per_epoch // args.grad_accum
    total_steps = args.total_steps or default_total

    # `batch_offset` is how far into `start_epoch` the previous session got.
    # Windows are drawn i.i.d., so resuming means running the remaining
    # batches of that epoch, not re-running all of them: the old loop restarted
    # the epoch from zero and quietly repeated up to an epoch of work on every
    # single resume, which over twenty resumes is days.
    step, start_epoch, best_val, batch_offset = 0, 0, float("inf"), 0
    if args.resume:
        ckpt, source = load_checkpoint(args.resume, map_location=device)
        if ckpt is None:
            print(f"No checkpoint at {args.resume}; starting from scratch.")
        else:
            if ckpt.get("profile") != args.profile:
                print(f"ERROR: checkpoint is profile {ckpt.get('profile')!r}, "
                      f"you asked for {args.profile!r}. Shapes will not match.")
                return 1
            inner.unet_model.load_state_dict(ckpt["unet"])
            inner.wave_model.load_state_dict(ckpt["wave"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scaler.load_state_dict(ckpt["scaler"])
            if ckpt.get("ema"):
                ema.load_state_dict(ckpt["ema"])
            step = ckpt["step"]
            start_epoch = ckpt["epoch"]
            best_val = ckpt["best_val"]
            batch_offset = int(ckpt.get("batch_in_epoch", 0))
            if batch_offset >= batches_per_epoch:
                # The epoch length changed (a different --samples-per-epoch or
                # batch size); the offset no longer means anything.
                start_epoch, batch_offset = start_epoch + 1, 0

            saved_total = ckpt.get("total_steps")
            if saved_total and args.total_steps is None and saved_total != total_steps:
                # The cosine schedule is defined by its horizon. Silently
                # changing that horizon mid-run rewrites the learning rate for
                # every remaining step, so the saved one wins unless overridden.
                print(f"  keeping the saved schedule horizon ({saved_total} steps); "
                      f"this session's arguments imply {total_steps}. "
                      f"Pass --total-steps to change it deliberately.")
                total_steps = saved_total

            best_text = ("no validation yet" if best_val == float("inf")
                         else f"best val {best_val:.5f}")
            print(f"Resumed from {source}: step {step}, epoch {start_epoch + 1}, "
                  f"batch {batch_offset}/{batches_per_epoch}, {best_text}")

    state = {"step": step, "epoch": start_epoch, "batch_in_epoch": batch_offset,
             "best_val": best_val}

    def payload() -> dict:
        node = unwrap(model)
        return {
            "unet": node.unet_model.state_dict(),
            "wave": node.wave_model.state_dict(),
            "ema": ema.state_dict() if ema else None,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": state["step"],
            "epoch": state["epoch"],
            "batch_in_epoch": state["batch_in_epoch"],
            "best_val": state["best_val"],
            "total_steps": total_steps,
            "profile": args.profile,
            "window_frames": args.window_frames,
            "prediction_type": args.prediction_type,
            "autoencoder_ckpt": str(args.ae),
        }

    trigger = SaveTrigger(args.save_every, args.save_every_min)
    saver = CheckpointSaver(args.out, payload, trigger)
    stop_signal = install_stop_handlers()

    print(f"\n{batches_per_epoch} batches/epoch, ~{total_steps} optimiser steps total")
    print(f"Checkpointing to {args.out}: {trigger.summary()}")
    if args.max_hours:
        print(f"Will stop cleanly after {args.max_hours:.1f} hours")
    if args.min_free_gb:
        print(f"Will stop cleanly if free host memory falls below "
              f"{args.min_free_gb:.1f} GB")
    print("Host memory before the first batch:")
    print(memory_report())
    print()

    t0 = time.time()
    trend = MemoryTrend()
    model.train()
    stop = False

    def run_validation(tag: str) -> None:
        """Validate on the EMA weights and keep best.pt if they improved."""
        ema.store(inner.trainable_parameters())
        ema.copy_to(inner.trainable_parameters())
        try:
            val_loss = validate(model, val_loader, device, args.val_batches, use_fp16)
        finally:
            ema.restore(inner.trainable_parameters())

        marker = ""
        if val_loss < state["best_val"]:
            state["best_val"] = val_loss
            saver.save("best.pt")
            marker = "  <- best"
        print(f"  val {val_loss:.5f} (best {state['best_val']:.5f}){marker}  [{tag}]")
        # Validation is the natural place for the fuller picture: it is rare
        # enough not to be noise and frequent enough to be current.
        print(trend.report(), flush=True)

    epoch = start_epoch
    reason = "finished"

    for epoch in range(start_epoch, args.epochs):
        if stop:
            break

        # Only the resumed epoch is short; every later one is full.
        done_in_epoch, batch_offset = batch_offset, 0
        budget = batches_per_epoch - done_in_epoch
        if budget <= 0:
            continue
        state["epoch"] = epoch

        for i, batch in enumerate(train_loader):
            if i >= budget:
                break
            done_in_epoch += 1

            lr_now = lr_at(step, args.warmup, total_steps, lr)
            for group in optimizer.param_groups:
                group["lr"] = lr_now

            with torch.amp.autocast("cuda", enabled=use_fp16):
                loss, metrics = model(**batch_to(batch, device))
                # DataParallel returns one row per replica; averaging is what
                # makes the reported numbers mean the same thing on 1 GPU and 2.
                loss = loss.mean()
                metrics = metrics.reshape(-1, 4).mean(0)
                scaled = loss / args.grad_accum

            scaler.scale(scaled).backward()

            if (i + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(inner.trainable_parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(inner.trainable_parameters())
                step += 1

                state["step"] = step
                state["batch_in_epoch"] = done_in_epoch

                if step % args.log_every == 0:
                    elapsed = time.time() - t0
                    snapshot = memory_mb()
                    trend.observe(step, snapshot)
                    growth = trend.compact()
                    print(f"epoch {epoch + 1:3d} step {step:7d}  "
                          f"loss {float(metrics[0]):.4f}  mae {float(metrics[1]):.4f}  "
                          f"|g| {float(grad_norm):.2f}  lr {lr_now:.2e}  "
                          f"{elapsed / 60:.1f}min  {memory_line(snapshot)}"
                          + (f" | {growth}" if growth else ""), flush=True)
                    alarm = trend.first_warning()
                    if alarm:
                        print(alarm, flush=True)

                if args.val_every and step % args.val_every == 0:
                    run_validation(f"step {step}")

                saver.maybe_save(step)

                if stop_signal:
                    reason = f"{stop_signal.reason} at step {step}"
                    stop = True
                    break

                if args.max_hours and (time.time() - t0) / 3600 >= args.max_hours:
                    print(f"\nReached {args.max_hours}h; saving and stopping.")
                    reason = f"--max-hours at step {step}"
                    stop = True
                    break

                # Checked every step, not every log line: the last run climbed a
                # gigabyte in twenty-five steps. Saving here is the difference
                # between losing two minutes and losing the session, because
                # SIGKILL arrives without warning and cannot be caught.
                if args.min_free_gb and headroom_gb() < args.min_free_gb:
                    print(f"\nOnly {headroom_gb():.1f} GB of host memory left "
                          f"(--min-free-gb {args.min_free_gb:.1f}). Saving and "
                          f"stopping before the kernel does it for us.")
                    print(memory_report())
                    print(trend.report())
                    saver.save("last.pt", f"low memory at step {step}")
                    print(f"\nStopped at step {step} with a current checkpoint. "
                          f"Start again with:")
                    print(f"  --resume {args.out / 'last.pt'}")
                    return EXIT_LOW_MEMORY

        # ---- end of epoch ------------------------------------------------- #
        if done_in_epoch >= batches_per_epoch:
            # A completed epoch starts the next one at batch zero.
            state["epoch"] = epoch + 1
            state["batch_in_epoch"] = 0
        if not stop:
            # Validating here is what guarantees a best.pt exists even in a
            # session too short to reach --val-every. A run that dies with
            # best_val still at infinity has produced nothing generation can use.
            if args.epoch_val:
                run_validation(f"end of epoch {epoch + 1}")
            if args.epoch_save:
                saver.save("last.pt", f"end of epoch {epoch + 1}")

    if args.epoch_val and stop and state["best_val"] == float("inf"):
        # Same argument, at the other exit: never end a session with no best.pt.
        run_validation("before stopping")
    saver.save("last.pt", reason)

    print(f"\n{'=' * 60}")
    best_text = ("never validated" if state["best_val"] == float("inf")
                 else f"{state['best_val']:.5f}")
    print(f"Stopped at step {step} ({reason}), best val {best_text}")
    print(f"Epoch {state['epoch'] + 1}, batch {state['batch_in_epoch']}/{batches_per_epoch} "
          f"-- --resume picks up exactly here")
    print(f"Checkpoints: {args.out / 'best.pt'}  {args.out / 'last.pt'}")
    print("Host memory at exit:")
    print(memory_report())
    print(trend.report())
    print("\nNext, measure Gate B (onset F1 > 0.40 on held-out audio):")
    print(f"  python scripts/evaluate.py --diffusion {args.out / 'best.pt'} --ae {args.ae}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
