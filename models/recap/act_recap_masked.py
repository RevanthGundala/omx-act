"""ACT with an advantage token that can be genuinely absent.

A standalone model, deliberately not a subclass of ``act_recap.py``. That file
is the RECAP baseline this one is compared against, and the two are meant to be
readable and changeable independently -- the same reason ``attention.py`` is a
copy rather than an import.

The difference is how "no advantage" is represented:

* ``act_recap.py`` projects a one-hot through ``nn.Linear(3, d_model)``, so
  "unconditioned" is a third *learned* class that the encoder still attends to.
* here the token is an ``nn.Embedding`` over NEGATIVE and POSITIVE only, and a
  missing token is genuinely missing: its memory slot is masked out of encoder
  self-attention and decoder cross-attention.

Pass ``MASKED_ADVANTAGE`` for a single sample or ``advantage=None`` for a whole
batch. The parameter layout therefore differs from the baseline on purpose, so a
baseline checkpoint does not load directly. Porting one is a concern of the tool
that reads old checkpoints, not of this model: see ``convert_legacy_state_dict``
in ``inspect_advantage_attention.py``.
"""

from enum import IntEnum
from itertools import chain

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torchvision.models.resnet import ResNet18_Weights, resnet18
from torchvision.ops import FrozenBatchNorm2d

from configs.act.base import ACTConfig
from .attention import TransformerBlock
from ..act.pos_emb import ACTSinusoidalPositionEmbedding2d, create_sinusoidal_pos_embedding


class AdvantageCondition(IntEnum):
    """The conditions that have a learned vector.

    There is no UNCONDITIONED member: this model omits the token instead of
    embedding a third value for "no advantage".
    """

    NEGATIVE = 0
    POSITIVE = 1


# Condition id meaning "this sample has no advantage token". Outside the
# embedding's index range so it can never name a learned vector.
MASKED_ADVANTAGE = -1


class ACTPolicy(nn.Module):
    """ACT whose advantage token can be omitted per example."""

    def __init__(self, config: ACTConfig, pretrained_backbone: bool = False):
        super().__init__()
        self.d_z = config.d_z
        self.cvae_encoder = CVAE_Encoder(
            joint_dim=config.joint_dim,
            d_model=config.d_model,
            d_z=config.d_z,
            num_encoder_layers=config.num_vae_encoder_layers,
            chunk_size=config.chunk_size,
            num_heads=config.num_heads,
            mlp_dim=config.mlp_dim,
            dropout=config.dropout,
        )
        self.decoder = Decoder(
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
        for parameter in chain(
            self.decoder.transformer_encoder.parameters(),
            self.decoder.transformer_decoder.parameters(),
        ):
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        temporal_weights = torch.exp(
            -config.exp_weight * torch.arange(0, config.chunk_size, dtype=torch.float32)
        )
        self.register_buffer("temporal_weights", temporal_weights, persistent=False)
        self.register_buffer(
            "temporal_weights_cumsum", torch.cumsum(temporal_weights, dim=0), persistent=False
        )
        self.action_buffer = None
        self.actions_number = None

    def forward(
        self,
        images,
        qpos,
        future_actions=None,
        action_mask=None,
        advantage=None,
    ):
        if future_actions is None or not self.training:
            z = torch.zeros(qpos.shape[0], self.d_z, dtype=qpos.dtype, device=qpos.device)
            mu, log_var = None, None
        else:
            z, mu, log_var = self.cvae_encoder(qpos, future_actions, action_mask)
        act_pred = self.decoder(images, qpos, z, advantage)
        return act_pred, mu, log_var

    @torch.no_grad()
    def select_action(self, images, qpos, use_temporal_ensemble=True, *, advantage=None):
        act_pred, _, _ = self.forward(images, qpos, advantage=advantage)  # [B, K, A]

        if use_temporal_ensemble is False:
            return act_pred[:, 0]

        chunk_size = act_pred.shape[1]
        device = act_pred.device

        if self.action_buffer is None:
            self.action_buffer = act_pred.clone()
            self.actions_number = torch.ones((chunk_size,), device=device, dtype=torch.long)
        else:
            old_len = self.action_buffer.shape[1]
            old_count = self.actions_number[:old_len]

            old_weight_sum = (
                self.temporal_weights_cumsum[old_count - 1]
                .to(device=device, dtype=act_pred.dtype)
                .view(1, old_len, 1)
            )
            new_weight = (
                self.temporal_weights[old_count]
                .to(device=device, dtype=act_pred.dtype)
                .view(1, old_len, 1)
            )
            overlap = (
                self.action_buffer * old_weight_sum + act_pred[:, :old_len] * new_weight
            ) / (old_weight_sum + new_weight)

            self.action_buffer = torch.cat(
                [overlap, act_pred[:, old_len : old_len + 1]],
                dim=1,
            )
            self.actions_number = torch.cat(
                [old_count + 1, torch.ones((1,), device=device, dtype=torch.long)],
                dim=0,
            )

        action = self.action_buffer[:, 0]
        self.action_buffer = self.action_buffer[:, 1:]
        self.actions_number = self.actions_number[1:]
        return action

    def reset_action_selection(self):
        self.action_buffer = None
        self.actions_number = None


class CVAE_Encoder(nn.Module):
    def __init__(
        self,
        joint_dim,
        d_model,
        d_z,
        num_encoder_layers,
        chunk_size,
        num_heads=8,
        mlp_dim=2048,
        dropout=0.1,
    ):
        super().__init__()
        # 2 separate projections for future actions and current joint pos
        self.joint_proj = nn.Linear(joint_dim, d_model)
        self.act_proj = nn.Linear(joint_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn((1, 1, d_model)))
        self.out_proj = nn.Linear(d_model, d_z * 2)
        self.transformer_encoder = nn.ModuleList([
            TransformerBlock(
                dim=d_model,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                use_cross_attention=False,
            ) for _ in range(num_encoder_layers)
        ])
        self.chunk_size = chunk_size
        self.register_buffer(
            "encoder_pos_table",
            create_sinusoidal_pos_embedding(1 + 1 + chunk_size, d_model),
            persistent=False,
        )

    def forward(self, qpos, future_actions, action_mask=None):
        bs = future_actions.shape[0]
        qpos_token = self.joint_proj(qpos).unsqueeze(1)
        cls_token = self.cls_token.expand(bs, -1, -1)
        act_tokens = self.act_proj(future_actions)
        pos = self.encoder_pos_table.unsqueeze(0)
        x = torch.cat([cls_token, qpos_token, act_tokens], dim=1)
        key_padding_mask = None
        if action_mask is not None:
            cls_joint_is_pad = torch.ones((bs, 2), dtype=torch.bool, device=action_mask.device)
            key_padding_mask = torch.cat([cls_joint_is_pad, action_mask.bool()], dim=1)
            key_padding_mask = key_padding_mask[:, None, None, :]
        for block in self.transformer_encoder:
            x = block(x, key_padding_mask=key_padding_mask, pos=pos)
        # reparameterize
        cls = x[:, 0]
        mu, log_var = self.out_proj(cls).chunk(2, dim=-1)
        std = torch.exp(0.5 * log_var)
        z = mu + std * torch.randn_like(mu)
        return z, mu, log_var


class Decoder(nn.Module):
    def __init__(
        self,
        d_model,
        joint_dim,
        num_encoder_layers,
        num_decoder_layers,
        d_z,
        chunk_size,
        num_cameras=1,
        num_heads=8,
        mlp_dim=2048,
        dropout=0.1,
        pretrained_backbone=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_cameras = num_cameras
        weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
        backbone = resnet18(weights=weights, norm_layer=FrozenBatchNorm2d)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.image_proj = nn.Linear(512, d_model)
        self.image_pos_emb = ACTSinusoidalPositionEmbedding2d(d_model // 2)
        self.z_proj = nn.Linear(d_z, d_model)
        self.joint_proj = nn.Linear(joint_dim, d_model)
        self.transformer_encoder = nn.ModuleList([
            TransformerBlock(
                dim=d_model,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                use_cross_attention=False,
            ) for _ in range(num_encoder_layers)
        ])
        self.transformer_decoder = nn.ModuleList([
            TransformerBlock(
                dim=d_model,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                use_cross_attention=True,
            ) for _ in range(num_decoder_layers)
        ])
        self.joint_z_pos_emb = nn.Embedding(2, d_model)
        self.q_emb = nn.Parameter(torch.randn(1, chunk_size, d_model))
        self.act_proj = nn.Linear(d_model, joint_dim)
        self.final_norm = nn.LayerNorm(d_model)

        self.advantage_embedding = nn.Embedding(len(AdvantageCondition), d_model)
        self.adv_pos_emb = nn.Embedding(1, d_model)

    def forward(self, images, joints, z, advantage=None):
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
        image_tokens = rearrange(image_features, "(b m) c h w -> (b m) (h w) c", b=bs, m=num_cameras)
        image_tokens = self.image_proj(image_tokens)
        image_tokens = rearrange(image_tokens, "(b m) n d -> b (m n) d", b=bs, m=num_cameras)

        qpos_token = self.joint_proj(joints).unsqueeze(1)
        z_token = self.z_proj(z).unsqueeze(1)
        memory = torch.cat([z_token, qpos_token, image_tokens], dim=1)
        joint_z_pos = self.joint_z_pos_emb.weight.unsqueeze(0)
        pos = torch.cat([joint_z_pos, image_pos], dim=1)

        # One class id per sample; MASKED_ADVANTAGE (or advantage=None) omits it.
        if advantage is None:
            advantage = MASKED_ADVANTAGE
        advantage = torch.as_tensor(advantage, device=joints.device)
        if advantage.ndim == 0:
            advantage = advantage.expand(bs)
        if advantage.shape != (bs,) or advantage.is_floating_point():
            raise ValueError(
                "advantage must be one integer class ID per batch item, got "
                f"{tuple(advantage.shape)} of {advantage.dtype} for batch size {bs}"
            )
        advantage = advantage.long()
        omitted = advantage == MASKED_ADVANTAGE
        if (~omitted & ((advantage < 0) | (advantage >= len(AdvantageCondition)))).any():
            raise ValueError(
                "advantage class IDs must be "
                f"{[f'{c.name}={int(c)}' for c in AdvantageCondition]} or "
                f"MASKED_ADVANTAGE={MASKED_ADVANTAGE}; got "
                f"{sorted(set(advantage[~omitted].tolist()))}"
            )

        # Omitted samples still need a valid row for the lookup to be defined.
        # Which row is irrelevant: the mask below removes the slot entirely, so
        # nothing about this vector can reach the output.
        adv_token = self.advantage_embedding(advantage.clamp_min(0)).unsqueeze(1)
        memory = torch.cat([adv_token.to(dtype=memory.dtype), memory], dim=1)
        adv_pos = self.adv_pos_emb.weight.unsqueeze(0).to(dtype=pos.dtype)
        pos = torch.cat([adv_pos, pos], dim=1)

        # The advantage token is memory slot 0. Masking it is what makes an
        # omitted token genuinely absent rather than a third learned value.
        key_mask = torch.ones((bs, 1, 1, memory.shape[1]), dtype=torch.bool, device=memory.device)
        key_mask[:, :, :, 0] = ~omitted.view(bs, 1, 1)

        for block in self.transformer_encoder:
            memory = self._run_block(block, memory, self_key_mask=key_mask, pos=pos)

        q_tok = self.q_emb.expand(bs, -1, -1)
        out = torch.zeros_like(q_tok)
        for block in self.transformer_decoder:
            out = self._run_block(
                block,
                out,
                encoder_out=memory,
                encoder_key_mask=key_mask,
                pos=pos,
                query_pos=q_tok,
            )
        out = self.final_norm(out)
        act_pred = self.act_proj(out)
        return act_pred

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

        ``TransformerBlock.forward`` applies one mask to both, which cannot say
        "mask the advantage key when the decoder reads memory".
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

    def _prepare_images(self, images):
        if images.dim() == 4:
            if images.shape[1] in (1, 3):
                images = images.unsqueeze(1)
            elif images.shape[-1] in (1, 3):
                images = rearrange(images, "b h w c -> b 1 c h w")
            else:
                raise ValueError(
                    f"Expected image shape (B,C,H,W) or (B,H,W,C), got {tuple(images.shape)}"
                )
        elif images.dim() == 5:
            if images.shape[2] not in (1, 3) and images.shape[-1] in (1, 3):
                images = rearrange(images, "b m h w c -> b m c h w")
        else:
            raise ValueError(f"Expected images with 4 or 5 dimensions, got {tuple(images.shape)}")
        # run real normalization in preprocessor
        if not images.is_floating_point():
            images = images.float() / 255.0
        return images
