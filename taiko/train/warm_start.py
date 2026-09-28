"""
taiko/train/warm_start.py

Carry a trained checkpoint into a model that has grown new inputs.

The upgrade from features 1 to 2 (taiko/model/diffusion.py) adds inputs in
three places, always *appended* to what a layer already reads:

    U-Net enc_proj / dec_proj   + TimingEncoder features per level
    OnsetStem.proj              + song-level flux and loudness channels
    ConditionEmbedding.out_proj + the density-context vector

plus one wholly new module, the TimingEncoder itself.

Widening rule: an old weight is copied into the leading slice of the new one
and the new input columns are **zero**. A 1x1 conv or linear layer with zero
columns ignores those inputs exactly, so the widened model computes the old
model's function bit for bit at step 0 -- it resumes where it stopped, and
the new inputs start contributing only as gradient teaches them to. The
TimingEncoder keeps its fresh initialisation; nothing reads it until the zero
columns move.

Optimiser and EMA state are matched by *name*, not position: the parameter
order changes when modules are added, and a positional match would hand one
tensor's Adam moments to another.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class WarmStartReport:
    copied:  list[str] = field(default_factory=list)
    widened: list[str] = field(default_factory=list)
    fresh:   list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"  copied unchanged   {len(self.copied)} tensors",
                 f"  widened (zero-init new columns)  {len(self.widened)}"]
        lines += [f"      {name}" for name in self.widened]
        lines.append(f"  new, freshly initialised  {len(self.fresh)}")
        prefixes = sorted({name.rsplit(".", 2)[0] for name in self.fresh})
        lines += [f"      {p}.*" for p in prefixes[:12]]
        if len(prefixes) > 12:
            lines.append(f"      ... {len(prefixes) - 12} more")
        return "\n".join(lines)


def widen_tensor(new: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
    """
    `old` into the leading input columns of a tensor shaped like `new`.

    Only growth along dim 1 -- the input-channel axis of Conv1d and Linear
    weights -- is allowed. Anything else means the architecture changed in a
    way this cannot carry across, and guessing would silently corrupt it.
    """
    if new.shape == old.shape:
        return old.clone()
    if (new.dim() != old.dim() or new.dim() < 2
            or new.shape[0] != old.shape[0] or new.shape[2:] != old.shape[2:]
            or new.shape[1] < old.shape[1]):
        raise ValueError(f"cannot widen {tuple(old.shape)} into {tuple(new.shape)}")
    out = torch.zeros_like(new)
    out[:, :old.shape[1]] = old.to(new.dtype)
    return out


def widen_state_dict(module: nn.Module, old_state: dict, prefix: str,
                     report: WarmStartReport) -> None:
    """Load `old_state` into `module`, widening where the module grew."""
    target = module.state_dict()
    unexpected = sorted(set(old_state) - set(target))
    if unexpected:
        raise ValueError(
            f"{prefix}: checkpoint has tensors this model does not: "
            f"{unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}"
        )

    merged = {}
    for key, value in target.items():
        name = f"{prefix}.{key}"
        if key not in old_state:
            merged[key] = value
            report.fresh.append(name)
        elif old_state[key].shape == value.shape:
            merged[key] = old_state[key]
            report.copied.append(name)
        else:
            merged[key] = widen_tensor(value, old_state[key])
            report.widened.append(name)
    module.load_state_dict(merged)


def warm_start(model, ckpt: dict, optimizer: torch.optim.Optimizer | None = None,
               ema=None) -> WarmStartReport:
    """
    Load a checkpoint from an older model generation into `model`.

    Args:
        model:     a TaikoDiffusion built at the new generation
        ckpt:      the old checkpoint dict (as written by train_diffusion.py)
        optimizer: if given, its state is rebuilt from the checkpoint's for
                   every parameter whose shape did not change. Widened and new
                   parameters start with empty Adam moments.
        ema:       if given, its shadow is rebuilt the same way: old averages
                   where they exist, widened like the weights, and the live
                   (zero or freshly initialised) values everywhere else.
    """
    report = WarmStartReport()
    widen_state_dict(model.unet_model, ckpt["unet"], "unet", report)
    widen_state_dict(model.wave_model, ckpt["wave"], "wave", report)
    if model.timing_model is not None:
        if ckpt.get("timing"):
            widen_state_dict(model.timing_model, ckpt["timing"], "timing", report)
        else:
            report.fresh += [f"timing.{k}" for k in model.timing_model.state_dict()]

    named = model.named_trainable_parameters()
    old_keys = {f"unet.{k}" for k in ckpt["unet"]} | {f"wave.{k}" for k in ckpt["wave"]}
    old_keys |= {f"timing.{k}" for k in (ckpt.get("timing") or {})}
    # The old trainable order is the new order with the new tensors removed:
    # modules were only ever added, never reordered.
    old_order = [name for name, _ in named if name in old_keys]
    old_index = {name: i for i, name in enumerate(old_order)}

    if ema is not None and ckpt.get("ema"):
        shadow_old = ckpt["ema"]["shadow"]
        if len(shadow_old) != len(old_order):
            raise ValueError(
                f"EMA holds {len(shadow_old)} tensors but the checkpoint has "
                f"{len(old_order)} trainable parameters; cannot align them"
            )
        shadow = []
        for name, param in named:
            live = param.detach().float().clone()
            if name in old_index:
                shadow.append(widen_tensor(live, shadow_old[old_index[name]].float()))
            else:
                shadow.append(live)
        ema.decay = ckpt["ema"]["decay"]
        ema.warmup = ckpt["ema"].get("warmup", 0)
        ema.step = ckpt["ema"]["step"]
        ema.shadow = shadow

    if optimizer is not None and ckpt.get("optimizer"):
        old_opt = ckpt["optimizer"]
        new_opt = optimizer.state_dict()
        old_state = old_opt["state"]
        state = {}
        for j, (name, param) in enumerate(named):
            i = old_index.get(name)
            if i is None or i not in old_state:
                continue
            moments = old_state[i]
            exp_avg = moments.get("exp_avg")
            if exp_avg is not None and exp_avg.shape != param.shape:
                continue                  # widened: its moments no longer fit
            state[j] = moments
        new_opt["state"] = state
        # Keep this optimiser's own groups (it knows the new parameter list)
        # but carry the old hyperparameters across.
        for group, old_group in zip(new_opt["param_groups"], old_opt["param_groups"]):
            for key, value in old_group.items():
                if key != "params":
                    group[key] = value
        optimizer.load_state_dict(new_opt)

    return report
