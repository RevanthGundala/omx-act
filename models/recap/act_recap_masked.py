"""RECAP ACT variant with a real, maskable advantage embedding.

The legacy architecture in ``act_recap.py`` feeds a one-hot vector through
``nn.Linear(3, d_model)`` and always contributes a token: "no advantage" is
spelled as a third learned class (``UNCONDITIONED``), which the encoder still
attends to. That file and its behaviour are left untouched.

This variant instead does what PI's conditioning dropout describes -- the token
is *absent*, not a third value:

* ``nn.Embedding(2, d_model)`` over NEGATIVE and POSITIVE only. There is no
  learned "unconditioned" vector to attend to.
* ``MASKED_ADVANTAGE`` (-1) marks a sample whose token is missing;
  ``advantage=None`` marks the whole batch.
* Missing entries are clamped to index 0 purely so the embedding lookup has a
  valid index. That placeholder row is then masked out of encoder
  self-attention and decoder cross-attention, so it cannot reach the output.
  The clamp target is arbitrary and unobservable; a test pins that.

The parameter layout therefore differs from the legacy model on purpose, and a
legacy checkpoint does *not* load directly. Use ``convert_legacy_state_dict``
to port one, which reproduces the legacy outputs exactly for visible tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange, repeat

from .act_recap import ACTPolicy as LegacyACTPolicy, AdvantageCondition  # noqa: F401

# Sentinel condition id meaning "this sample has no advantage token". Distinct
# from AdvantageCondition.UNCONDITIONED, which is a learned third class in the
# legacy architecture and deliberately has no counterpart here.
MASKED_ADVANTAGE = -1

# Only NEGATIVE (0) and POSITIVE (1) are embedded.
NUM_ADVANTAGE_CLASSES = 2


class ACTPolicy(LegacyACTPolicy):
    """RECAP ACT whose advantage token can be genuinely absent."""

    def __init__(self, config, pretrained_backbone: bool = False):
        super().__init__(config, pretrained_backbone=pretrained_backbone)
        # Replace the legacy one-hot projection with a two-row embedding.
        del self.decoder.adv_proj
        self.decoder.advantage_embedding = nn.Embedding(
            NUM_ADVANTAGE_CLASSES, config.d_model
        )

    def forward(
        self,
        images,
        qpos,
        future_actions=None,
        action_mask=None,
        advantage=None,
    ):
        if future_actions is None or not self.training:
            z = torch.zeros(
                qpos.shape[0],
                self.d_z,
                dtype=qpos.dtype,
                device=qpos.device,
            )
            mu, log_var = None, None
        else:
            z, mu, log_var = self.cvae_encoder(qpos, future_actions, action_mask)
        actions = _masked_decoder_forward(self.decoder, images, qpos, z, advantage)
        return actions, mu, log_var

    @torch.no_grad()
    def select_action(
        self,
        images,
        qpos,
        use_temporal_ensemble=True,
        *,
        advantage=None,
    ):
        return super().select_action(
            images,
            qpos,
            use_temporal_ensemble=use_temporal_ensemble,
            advantage=advantage,
        )


def convert_legacy_state_dict(state_dict: dict) -> dict:
    """Port a legacy RECAP checkpoint onto this architecture.

    The legacy advantage token is ``nn.Linear(3, d_model)`` applied to a one-hot
    row, and ``nn.Linear`` computes ``x @ W.T + b``. A one-hot for class ``c``
    therefore selects column ``c``: the token is ``W[:, c] + b``. Each embedding
    row is set to exactly that, so visible tokens reproduce the legacy outputs.

    The legacy UNCONDITIONED column is dropped -- this architecture omits the
    token rather than embedding a third class. ``adv_pos_emb`` is untouched.
    """
    converted = dict(state_dict)
    weight = converted.pop("decoder.adv_proj.weight", None)
    bias = converted.pop("decoder.adv_proj.bias", None)
    if weight is None or bias is None:
        raise KeyError(
            "state dict has no decoder.adv_proj.{weight,bias}; it does not look "
            "like a legacy RECAP checkpoint"
        )
    if weight.shape[1] < NUM_ADVANTAGE_CLASSES:
        raise ValueError(
            f"legacy adv_proj has {weight.shape[1]} input classes, need at least "
            f"{NUM_ADVANTAGE_CLASSES}"
        )
    columns = weight[:, :NUM_ADVANTAGE_CLASSES].transpose(0, 1)
    converted["decoder.advantage_embedding.weight"] = columns + bias.unsqueeze(0)
    return converted


def _prepare_advantage(advantage, *, batch_size, device):
    """Return (embedding indices, per-sample present mask).

    Indices for missing samples are clamped to a valid row so the lookup is
    defined; the mask is what actually removes them.
    """
    if advantage is None:
        return (
            torch.zeros(batch_size, dtype=torch.long, device=device),
            torch.zeros(batch_size, dtype=torch.bool, device=device),
        )

    advantage = torch.as_tensor(advantage, device=device)
    if advantage.ndim == 0:
        advantage = advantage.expand(batch_size)
    if advantage.is_floating_point():
        if not torch.equal(advantage, advantage.round()):
            raise ValueError("advantage class IDs must be integers")
    advantage = advantage.long()
    if advantage.shape != (batch_size,):
        raise ValueError(
            "advantage must contain one class ID per batch item, got "
            f"{tuple(advantage.shape)} for batch size {batch_size}"
        )

    present = advantage != MASKED_ADVANTAGE
    invalid = present & ((advantage < 0) | (advantage >= NUM_ADVANTAGE_CLASSES))
    if invalid.any():
        raise ValueError(
            "advantage class IDs must be NEGATIVE (0), POSITIVE (1), or "
            f"MASKED_ADVANTAGE ({MASKED_ADVANTAGE}); got "
            f"{sorted(set(advantage[invalid].tolist()))}"
        )
    # Clamp only so the lookup is defined; masked rows are removed by attention.
    return advantage.clamp_min(0), present


def _block_forward(
    block,
    x,
    *,
    encoder_out=None,
    self_key_mask=None,
    encoder_key_mask=None,
    pos=None,
    query_pos=None,
):
    """TransformerBlock forward with separate self- and cross-attention masks."""
    self_pos = query_pos if block.cross_attn is not None else pos
    skip = x
    x = block.self_attn(x, key_padding_mask=self_key_mask, pos=self_pos)
    x = block.self_norm(skip + block.self_residual_dropout(x))

    if block.cross_attn is not None and encoder_out is not None:
        skip = x
        x = block.cross_attn(
            x,
            encoder_out=encoder_out,
            key_padding_mask=encoder_key_mask,
            pos=pos,
            query_pos=query_pos,
        )
        x = block.cross_norm(skip + block.cross_residual_dropout(x))

    return block.mlp_norm(x + block.mlp(x))


def _masked_decoder_forward(decoder, images, joints, z, advantage):
    images = decoder._prepare_images(images)
    batch_size, num_cameras = images.shape[:2]
    if num_cameras != decoder.num_cameras:
        raise ValueError(f"Expected {decoder.num_cameras} cameras, got {num_cameras}")

    image_features = decoder.backbone(rearrange(images, "b m c h w -> (b m) c h w"))
    image_pos = repeat(
        decoder.image_pos_emb(image_features),
        "1 d h w -> 1 (m h w) d",
        m=num_cameras,
    ).to(dtype=image_features.dtype)
    image_tokens = rearrange(
        image_features,
        "(b m) c h w -> (b m) (h w) c",
        b=batch_size,
        m=num_cameras,
    )
    image_tokens = decoder.image_proj(image_tokens)
    image_tokens = rearrange(
        image_tokens,
        "(b m) n d -> b (m n) d",
        b=batch_size,
        m=num_cameras,
    )

    qpos_token = decoder.joint_proj(joints).unsqueeze(1)
    z_token = decoder.z_proj(z).unsqueeze(1)
    memory = torch.cat([z_token, qpos_token, image_tokens], dim=1)
    joint_z_pos = decoder.joint_z_pos_emb.weight.unsqueeze(0)
    pos = torch.cat([joint_z_pos, image_pos], dim=1)

    indices, present = _prepare_advantage(
        advantage,
        batch_size=batch_size,
        device=joints.device,
    )
    advantage_token = decoder.advantage_embedding(indices).unsqueeze(1)
    advantage_token = advantage_token.to(dtype=memory.dtype)
    memory = torch.cat([advantage_token, memory], dim=1)
    advantage_pos = decoder.adv_pos_emb.weight.unsqueeze(0).to(dtype=pos.dtype)
    pos = torch.cat([advantage_pos, pos], dim=1)

    # Slot 0 is the advantage token. Masking it here is what makes a missing
    # token genuinely absent rather than a third learned value.
    memory_key_mask = torch.ones(
        (batch_size, 1, 1, memory.shape[1]),
        dtype=torch.bool,
        device=memory.device,
    )
    memory_key_mask[:, :, :, 0] = present.view(batch_size, 1, 1)

    for block in decoder.transformer_encoder:
        memory = _block_forward(block, memory, self_key_mask=memory_key_mask, pos=pos)

    query_tokens = decoder.q_emb.expand(batch_size, -1, -1)
    output = torch.zeros_like(query_tokens)
    for block in decoder.transformer_decoder:
        output = _block_forward(
            block,
            output,
            encoder_out=memory,
            encoder_key_mask=memory_key_mask,
            pos=pos,
            query_pos=query_tokens,
        )
    return decoder.act_proj(decoder.final_norm(output))
