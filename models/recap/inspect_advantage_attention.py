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
from models.recap.check_token_sensitivity import observation_scale
from models.recap.train_policy import config_from_checkpoint


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
    """Load a legacy RECAP checkpoint into the maskable architecture.

    The two architectures store the advantage token differently, so the state
    dict is converted rather than loaded directly. The conversion is only
    meaningful if it is exact, so the caller should run
    ``assert_conversion_is_faithful`` before reading anything into the masked
    outputs.
    """
    cfg = config_from_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    legacy_state = checkpoint["model_state_dict"]

    model = ACTPolicy(cfg, pretrained_backbone=False).to(device)
    model.load_state_dict(convert_legacy_state_dict(legacy_state), strict=True)
    model.eval()

    legacy = LegacyACTPolicy(cfg, pretrained_backbone=False).to(device)
    legacy.load_state_dict(legacy_state, strict=True)
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
    loader, _ = make_dataloaders(cfg)

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
    print(
        "legacy conversion max error: "
        f"{report['legacy_conversion_max_error']:.2e} (0 means the converted "
        "model reproduces the checkpoint exactly)"
    )
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
    print(f"report: {report_path}")
    print(f"plot:   {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
