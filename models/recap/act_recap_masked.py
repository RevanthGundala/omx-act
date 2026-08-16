"""RECAP ACT whose advantage token can be genuinely absent.

The legacy architecture in ``act_recap.py`` feeds a one-hot vector through
``nn.Linear(3, d_model)``, so it always contributes a token: "no advantage" is
spelled as a third learned class (``UNCONDITIONED``) that the encoder still
attends to. That file is left untouched.

This variant implements the same intent the way PI's conditioning dropout
describes it -- the token is *absent* rather than set to a third value:

* an embedding over the real conditions only, so there is no learned
  "unconditioned" vector to attend to;
* ``MASKED_ADVANTAGE`` marks a single sample whose token is missing, and
  ``advantage=None`` or ``UNCONDITIONED`` marks the whole batch;
* the advantage slot is masked out of encoder self-attention and decoder
  cross-attention, so a missing token cannot reach the output.

The parameter layout therefore differs from the legacy model on purpose. A
legacy checkpoint does not load directly; ``convert_legacy_state_dict`` ports
one and reproduces its outputs exactly wherever the token is visible.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from itertools import chain

import torch.nn.init as init
from einops import rearrange, repeat

from .act_recap import ACTPolicy as LegacyACTPolicy, AdvantageCondition, Decoder

# Condition id meaning "this sample has no advantage token". Deliberately
# outside the embedding's index range so it cannot name a learned vector.
MASKED_ADVANTAGE = -1

# The conditions that get a learned vector. UNCONDITIONED is excluded because
# this architecture omits the token rather than embedding a third value for it.
EMBEDDED_CONDITIONS = tuple(
    condition
    for condition in AdvantageCondition
    if condition is not AdvantageCondition.UNCONDITIONED
)

# Embedding rows are indexed by the condition's own value, and
# convert_legacy_state_dict maps legacy column c onto row c, so those values
# have to be 0..n-1. Reordering or extending the enum must fail here rather
# than silently mapping one condition onto another's row.
if tuple(int(condition) for condition in EMBEDDED_CONDITIONS) != tuple(
    range(len(EMBEDDED_CONDITIONS))
):
    raise RuntimeError(
        "AdvantageCondition values must start at 0 and be contiguous with "
        f"UNCONDITIONED last; got {[(c.name, int(c)) for c in AdvantageCondition]}"
    )


class MaskedDecoder(Decoder):
    """The legacy decoder with a maskable advantage embedding."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.adv_proj
        self.advantage_embedding = nn.Embedding(len(EMBEDDED_CONDITIONS), self.d_model)

    def forward(self, images, joints, z, advantage=MASKED_ADVANTAGE):
        images = self._prepare_images(images)
        bs, num_cameras = images.shape[:2]
        if num_cameras != self.num_cameras:
            raise ValueError(f"Expected {self.num_cameras} cameras, got {num_cameras}")

        image_features = self.backbone(rearrange(images, "b m c h w -> (b m) c h w"))
        image_pos = repeat(
            self.image_pos_emb(image_features),
            "1 d h w -> 1 (m h w) d",
            m=num_cameras,
        ).to(dtype=image_features.dtype)
        image_tokens = rearrange(
            image_features,
            "(b m) c h w -> (b m) (h w) c",
            b=bs,
            m=num_cameras,
        )
        image_tokens = self.image_proj(image_tokens)
        image_tokens = rearrange(image_tokens, "(b m) n d -> b (m n) d", b=bs, m=num_cameras)

        qpos_token = self.joint_proj(joints).unsqueeze(1)
        z_token = self.z_proj(z).unsqueeze(1)
        memory = torch.cat([z_token, qpos_token, image_tokens], dim=1)
        joint_z_pos = self.joint_z_pos_emb.weight.unsqueeze(0)
        pos = torch.cat([joint_z_pos, image_pos], dim=1)

        indices, present = self._prepare_advantage(
            advantage,
            batch_size=bs,
            device=joints.device,
        )
        adv_token = self.advantage_embedding(indices).unsqueeze(1).to(dtype=memory.dtype)
        memory = torch.cat([adv_token, memory], dim=1)
        adv_pos = self.adv_pos_emb.weight.unsqueeze(0).to(dtype=pos.dtype)
        pos = torch.cat([adv_pos, pos], dim=1)

        # Slot 0 is the advantage token. Masking it is what makes a missing
        # token genuinely absent rather than a third learned value.
        memory_key_mask = torch.ones(
            (bs, 1, 1, memory.shape[1]),
            dtype=torch.bool,
            device=memory.device,
        )
        memory_key_mask[:, :, :, 0] = present.view(bs, 1, 1)

        for block in self.transformer_encoder:
            memory = self._run_block(block, memory, self_key_mask=memory_key_mask, pos=pos)

        q_tok = self.q_emb.expand(bs, -1, -1)
        out = torch.zeros_like(q_tok)
        for block in self.transformer_decoder:
            out = self._run_block(
                block,
                out,
                encoder_out=memory,
                encoder_key_mask=memory_key_mask,
                pos=pos,
                query_pos=q_tok,
            )
        return self.act_proj(self.final_norm(out))

    @staticmethod
    def _prepare_advantage(advantage, *, batch_size, device):
        """Return (embedding indices, per-sample present mask).

        ``None`` and ``UNCONDITIONED`` mean the whole batch is missing;
        ``MASKED_ADVANTAGE`` means a single sample is. Indices for missing
        samples are clamped to a valid row only so the lookup is defined -- the
        mask is what removes them.
        """
        if advantage is None:
            advantage = MASKED_ADVANTAGE

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

        # UNCONDITIONED has no learned vector here; it means the same thing the
        # sentinel does, so the legacy call signature keeps working.
        advantage = torch.where(
            advantage == int(AdvantageCondition.UNCONDITIONED),
            torch.full_like(advantage, MASKED_ADVANTAGE),
            advantage,
        )
        present = advantage != MASKED_ADVANTAGE
        invalid = present & ((advantage < 0) | (advantage >= len(EMBEDDED_CONDITIONS)))
        if invalid.any():
            raise ValueError(
                "advantage class IDs must name an embedded condition "
                f"{[c.name for c in EMBEDDED_CONDITIONS]}, MASKED_ADVANTAGE "
                f"({MASKED_ADVANTAGE}), or UNCONDITIONED; got "
                f"{sorted(set(advantage[invalid].tolist()))}"
            )
        return advantage.clamp_min(0), present

    @staticmethod
    def _run_block(
        block,
        x,
        *,
        encoder_out=None,
        self_key_mask=None,
        encoder_key_mask=None,
        pos=None,
        query_pos=None,
    ):
        """One TransformerBlock with separate self- and cross-attention masks.

        The stock block applies a single mask to both, which cannot express
        "mask the advantage key in cross-attention but not the queries".
        """
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


class ACTPolicy(LegacyACTPolicy):
    """Legacy RECAP ACT wired to the maskable decoder."""

    def __init__(self, config, pretrained_backbone: bool = False):
        # The base class builds a legacy Decoder that is replaced below, so it
        # is built without the pretrained backbone: loading those weights twice
        # would be pure waste.
        super().__init__(config, pretrained_backbone=False)
        self.decoder = MaskedDecoder(
            d_model=config.d_model,
            joint_dim=config.joint_dim,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            d_z=config.d_z,
            chunk_size=config.chunk_size,
            num_cameras=len(config.camera_names),
            num_heads=config.num_heads,
            mlp_dim=config.mlp_dim,
            dropout=config.dropout,
            pretrained_backbone=pretrained_backbone,
        )
        # The base class already ran this over the decoder it built, and that
        # decoder is gone.
        for parameter in chain(
            self.decoder.transformer_encoder.parameters(),
            self.decoder.transformer_decoder.parameters(),
        ):
            if parameter.dim() > 1:
                init.xavier_uniform_(parameter)


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
    embedded = len(EMBEDDED_CONDITIONS)
    if weight.shape[1] < embedded:
        raise ValueError(
            f"legacy adv_proj has {weight.shape[1]} input classes, need at least {embedded}"
        )
    columns = weight[:, :embedded].transpose(0, 1)
    converted["decoder.advantage_embedding.weight"] = columns + bias.unsqueeze(0)
    return converted
