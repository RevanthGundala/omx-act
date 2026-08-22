"""Inspect whether a trained policy routes information from its advantage token.

This is a read-only diagnostic. It compares identical observations with the
POSITIVE token visible versus attention-masked. It also summarizes how much
encoder self-attention and decoder cross-attention point at the advantage slot.

Attention is descriptive, not causal; the output delta from masking is the
causal measurement. The two are reported together.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.recap.act_recap import AdvantageCondition
from models.recap.act_recap import ACTPolicy as LegacyACTPolicy
from models.recap.act_recap_masked import (
    ACTPolicy,
    AdvantageCondition as MaskedAdvantageCondition,
)
from models.recap.analysis.check_token_sensitivity import observation_scale
from models.recap.train_policy import config_from_checkpoint

# Decoder memory layout, from act_recap_masked.Decoder.forward: the advantage
# token is prepended to [z, joints, image patches]. Asserted in the tests, since
# a reordering here would silently mislabel every slot in the report.
ADVANTAGE_SLOT = 0
LATENT_SLOT = 1
JOINTS_SLOT = 2


def convert_legacy_state_dict(state_dict: dict) -> dict:
    """Port an ``act_recap.py`` checkpoint onto the maskable model.

    This lives here rather than in act_recap_masked.py because it is a property
    of reading *old* checkpoints, not of the model: nothing that trains or runs
    the masked model needs it.

    The baseline advantage token is ``nn.Linear(3, d_model)`` applied to a
    one-hot row, and ``nn.Linear`` computes ``x @ W.T + b``. A one-hot for class
    ``c`` therefore selects column ``c``: the token is ``W[:, c] + b``. Each
    embedding row is set to exactly that, so a converted checkpoint reproduces
    the baseline's outputs wherever the token is visible.

    The baseline's UNCONDITIONED column is dropped -- this model omits the token
    instead of embedding a third class. ``adv_pos_emb`` is untouched.
    """
    converted = dict(state_dict)
    weight = converted.pop("decoder.adv_proj.weight", None)
    bias = converted.pop("decoder.adv_proj.bias", None)
    if weight is None or bias is None:
        raise KeyError(
            "state dict has no decoder.adv_proj.{weight,bias}; it does not look "
            "like an act_recap.py checkpoint"
        )
    embedded = len(MaskedAdvantageCondition)
    if weight.shape[1] < embedded:
        raise ValueError(
            f"checkpoint adv_proj has {weight.shape[1]} input classes, need at least {embedded}"
        )
    converted["decoder.advantage_embedding.weight"] = (
        weight[:, :embedded].transpose(0, 1) + bias.unsqueeze(0)
    )
    return converted


def load_policy(checkpoint_path: Path, device: torch.device):
    """Load either a native masked checkpoint or a legacy RECAP one.

    A checkpoint trained by train_masked_sft.py is already this architecture and
    loads directly. A legacy act_recap.py checkpoint stores the advantage token
    as a Linear over one-hots, so its state dict is converted; that conversion is
    only meaningful if it is exact, so a legacy model is returned alongside for
    ``assert_conversion_is_faithful`` to check against. Native checkpoints have
    nothing to convert and return ``None`` there.
    """
    cfg = config_from_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"]

    model = ACTPolicy(cfg, pretrained_backbone=False).to(device)
    is_legacy = "decoder.adv_proj.weight" in state
    model.load_state_dict(
        convert_legacy_state_dict(state) if is_legacy else state,
        strict=True,
    )
    model.eval()

    if not is_legacy:
        return model, None, cfg

    legacy = LegacyACTPolicy(cfg, pretrained_backbone=False).to(device)
    legacy.load_state_dict(state, strict=True)
    legacy.eval()
    return model, legacy, cfg


def assert_conversion_is_faithful(model, legacy, images, qpos, *, tolerance=1e-5):
    """Fail loudly if the converted model does not reproduce the legacy one.

    Every number this script reports is a difference between a visible-token
    and a masked forward pass. If the visible-token pass does not match the
    checkpoint it came from, that difference measures the conversion rather
    than the checkpoint.
    """
    batch = qpos.shape[0]
    worst = 0.0
    for condition in (AdvantageCondition.NEGATIVE, AdvantageCondition.POSITIVE):
        ids = torch.full((batch,), int(condition), dtype=torch.long, device=qpos.device)
        one_hot = torch.nn.functional.one_hot(
            ids, num_classes=len(AdvantageCondition)
        ).to(dtype=qpos.dtype)
        with torch.no_grad():
            converted_actions, _, _ = model(images, qpos, advantage=ids)
            legacy_actions, _, _ = legacy(images, qpos, advantage=one_hot)
        worst = max(worst, (converted_actions - legacy_actions).abs().max().item())
    if worst > tolerance:
        raise RuntimeError(
            f"converted model diverges from the legacy checkpoint by {worst:.3e}; "
            "masked-vs-visible differences would be meaningless"
        )
    return worst


def attention_snapshot(model: ACTPolicy) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Copy attention paid to memory slot 0 after one visible-token forward."""
    encoder = []
    for block in model.decoder.transformer_encoder:
        weights = block.self_attn.last_attn_weights
        # Other memory tokens querying the advantage key.
        encoder.append(weights[:, :, 1:, 0].mean(dim=2).cpu())

    decoder = []
    for block in model.decoder.transformer_decoder:
        weights = block.cross_attn.last_attn_weights
        # One value per batch, head, and action-query offset.
        decoder.append(weights[:, :, :, 0].cpu())
    return encoder, decoder


def memory_slot_ranking(model: ACTPolicy) -> torch.Tensor:
    """Attention received by every memory slot in the last decoder layer.

    Attention on the advantage slot only means something relative to the other
    slots. Slot 2 (joints) is the useful contrast: if the model concentrates
    heavily there, then "the architecture cannot single out one slot" is not
    available as an explanation for a low advantage number.
    """
    weights = model.decoder.transformer_decoder[-1].cross_attn.last_attn_weights
    return weights.mean(dim=(0, 1, 2)).cpu()


def masked_attention_max(model: ACTPolicy) -> float:
    values = []
    for block in model.decoder.transformer_encoder:
        values.append(block.self_attn.last_attn_weights[:, :, :, 0].abs().max().item())
    for block in model.decoder.transformer_decoder:
        values.append(block.cross_attn.last_attn_weights[:, :, :, 0].abs().max().item())
    return max(values, default=0.0)


@torch.inference_mode()
def inspect(
    model: ACTPolicy,
    loader,
    device: torch.device,
    max_batches: int,
    legacy=None,
) -> dict:
    conversion_error = None
    output_effects = []
    observation_effects = []
    encoder_batches: list[list[torch.Tensor]] = []
    decoder_batches: list[list[torch.Tensor]] = []
    masked_max = 0.0
    slot_batches: list[torch.Tensor] = []
    samples = 0

    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        images = batch["images"].to(device)
        qpos = batch["qpos"].to(device)
        if legacy is not None and conversion_error is None:
            conversion_error = assert_conversion_is_faithful(
                model, legacy, images, qpos
            )
        positive = torch.full(
            (qpos.shape[0],),
            int(AdvantageCondition.POSITIVE),
            dtype=torch.long,
            device=device,
        )
        visible_actions, _, _ = model(
            images,
            qpos,
            advantage=positive,
        )
        encoder, decoder = attention_snapshot(model)
        encoder_batches.append(encoder)
        decoder_batches.append(decoder)
        slot_batches.append(memory_slot_ranking(model))

        masked_actions, _, _ = model(
            images,
            qpos,
            advantage=None,
        )
        masked_max = max(masked_max, masked_attention_max(model))

        output_effects.append((visible_actions - masked_actions).abs().mean().item())
        observation_effects.append(observation_scale(visible_actions))
        samples += qpos.shape[0]

    if not output_effects:
        raise ValueError("No batches were inspected")

    encoder_layers = []
    for layer in range(len(encoder_batches[0])):
        encoder_layers.append(
            torch.cat([batch[layer] for batch in encoder_batches], dim=0)
            .mean(dim=0)
            .numpy()
        )
    decoder_layers = []
    for layer in range(len(decoder_batches[0])):
        decoder_layers.append(
            torch.cat([batch[layer] for batch in decoder_batches], dim=0)
            .mean(dim=0)
            .numpy()
        )

    output_effect = float(np.mean(output_effects))
    observation_effect = float(np.mean(observation_effects))
    memory_tokens = model.decoder.transformer_encoder[0].self_attn.last_attn_weights.shape[-1]

    slots = torch.stack(slot_batches).mean(dim=0)
    order = torch.argsort(slots, descending=True)
    rank_of = lambda slot: int((order == slot).nonzero()[0].item()) + 1
    return {
        "samples": samples,
        "batches": len(output_effects),
        "output_effect_sd": output_effect,
        "observation_effect_sd": observation_effect,
        "relative_output_effect": output_effect / observation_effect,
        "uniform_attention_baseline": 1.0 / memory_tokens,
        "masked_advantage_attention_max": masked_max,
        "legacy_conversion_max_error": conversion_error,
        "encoder_attention_to_advantage": [layer.tolist() for layer in encoder_layers],
        "decoder_attention_to_advantage": [layer.tolist() for layer in decoder_layers],
        "memory_slots": int(slots.shape[0]),
        "advantage_slot_rank": rank_of(ADVANTAGE_SLOT),
        "advantage_slot_attention": float(slots[ADVANTAGE_SLOT]),
        "joints_slot_rank": rank_of(JOINTS_SLOT),
        "joints_slot_attention": float(slots[JOINTS_SLOT]),
        "latent_slot_rank": rank_of(LATENT_SLOT),
        "latent_slot_attention": float(slots[LATENT_SLOT]),
        "median_slot_attention": float(slots[order[slots.shape[0] // 2]]),
        "top_slot_attention": float(slots[order[0]]),
    }


def plot_report(report: dict, output_path: Path, title: str) -> None:
    encoder = np.asarray(report["encoder_attention_to_advantage"])
    decoder = np.asarray(report["decoder_attention_to_advantage"])
    # Average decoder layers only for plotting; the JSON preserves every layer.
    decoder_map = decoder.mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    enc = axes[0].imshow(encoder * 100, aspect="auto", cmap="magma")
    axes[0].set(
        title="Encoder attention to advantage slot (%)",
        xlabel="Head",
        ylabel="Layer",
    )
    fig.colorbar(enc, ax=axes[0])

    dec = axes[1].imshow(decoder_map * 100, aspect="auto", cmap="magma")
    axes[1].set(
        title="Decoder cross-attention to advantage slot (%)",
        xlabel="Action offset",
        ylabel="Head",
    )
    fig.colorbar(dec, ax=axes[1])

    fig.suptitle(
        f"{title}\nvisible-vs-masked output effect "
        f"{report['relative_output_effect']:.2%} of observation effect"
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", default=None)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--on-train",
        action="store_true",
        help="Measure on the training split even when a holdout exists.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    from models.act.helper import make_dataloaders

    args = parse_args(argv)
    device = torch.device(
        args.device
        or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )
    model, legacy, cfg = load_policy(args.checkpoint, device)
    if args.dataset_repo_id:
        cfg = replace(cfg, dataset_repo_id=args.dataset_repo_id)
    train_loader, val_loader = make_dataloaders(cfg)
    loader = val_loader if (val_loader is not None and not args.on_train) else train_loader
    if loader is train_loader:
        # Worth shouting about: on data the model trained on, the observation
        # alone largely determines the action, which is the regime most likely to
        # make a conditioning token look redundant. The bias runs toward finding
        # the token inert, so a number measured here overstates the case.
        print(
            "WARNING: measuring on TRAINING data"
            + (
                " (--on-train)"
                if args.on_train
                else f" (train_split={cfg.train_split} leaves no holdout)"
            )
            + "; the token will look more redundant here than on held-out episodes",
            flush=True,
        )
    else:
        print(f"measuring on {len(loader.dataset)} held-out frames", flush=True)

    report = inspect(model, loader, device, args.batches, legacy=legacy)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.checkpoint.stem
    report_path = args.output_dir / f"{stem}_attention.json"
    plot_path = args.output_dir / f"{stem}_attention.png"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    plot_report(report, plot_path, stem)

    uniform = report["uniform_attention_baseline"]
    print(f"checkpoint: {args.checkpoint}")
    print(f"samples: {report['samples']}")
    if report["legacy_conversion_max_error"] is not None:
        print(
            "legacy conversion max error: "
            f"{report['legacy_conversion_max_error']:.2e} (0 means the converted "
            "model reproduces the checkpoint exactly)"
        )
    else:
        print("checkpoint is natively maskable; no conversion needed")
    print(
        "visible vs masked output effect: "
        f"{report['output_effect_sd']:.6f} "
        f"({report['relative_output_effect']:.2%} of observation effect)"
    )
    print(f"uniform attention baseline: {uniform:.4%}")
    print(
        "masked attention max (must be 0): "
        f"{report['masked_advantage_attention_max']:.3e}"
    )

    slots = report["memory_slots"]
    print(f"\nattention received, last decoder layer ({slots} memory slots):")
    print(f"  {'slot':<22} {'attention':>11} {'vs uniform':>11} {'rank':>12}")
    for label, key in (
        ("2 joints", "joints"),
        ("1 z (CVAE)", "latent"),
        ("0 advantage", "advantage"),
    ):
        value = report[f"{key}_slot_attention"]
        print(
            f"  {label:<22} {value:11.3e} {value / uniform:10.2f}x "
            f"{report[f'{key}_slot_rank']:6d}/{slots}"
        )
    print(f"  {'median slot':<22} {report['median_slot_attention']:11.3e}")
    print(
        "\n  a low advantage rank next to a high joints rank means the model can "
        "\n  concentrate on one slot and chose not to concentrate on this one."
    )
    print(f"\nreport: {report_path}")
    print(f"plot:   {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
