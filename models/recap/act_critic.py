from itertools import chain

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torchvision.models.resnet import ResNet18_Weights, resnet18
from torchvision.ops import FrozenBatchNorm2d

from configs.act.base import ACTConfig
from .attention import TransformerBlock
from ..act.pos_emb import ACTSinusoidalPositionEmbedding2d


class ACTCritic(nn.Module):
    """Observation-only distributional value function for RECAP."""

    def __init__(
        self,
        config: ACTConfig,
        pretrained_backbone: bool = False,
        num_bins: int = 201,
    ):
        super().__init__()
        if num_bins < 2:
            raise ValueError("num_bins must be at least 2")
        self.d_z = config.d_z
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
            num_bins=num_bins,
        )
        for parameter in chain(
            self.decoder.transformer_encoder.parameters(),
            self.decoder.transformer_decoder.parameters(),
        ):
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def _zero_latent(self, qpos):
        return torch.zeros(
            qpos.shape[0],
            self.d_z,
            dtype=qpos.dtype,
            device=qpos.device,
        )

    def encode(self, images, qpos):
        """Pooled observation embedding used by the value head. No advantage token."""
        return self.decoder.features(images, qpos, self._zero_latent(qpos))

    def forward(self, images, qpos):
        return self.decoder(images, qpos, self._zero_latent(qpos))


class Decoder(nn.Module):
    """ACT observation encoder with a distributional value head."""

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
        num_bins=201,
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
        self.transformer_encoder = nn.ModuleList(
            [
                TransformerBlock(
                    dim=d_model,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    use_cross_attention=False,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.transformer_decoder = nn.ModuleList(
            [
                TransformerBlock(
                    dim=d_model,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    use_cross_attention=True,
                )
                for _ in range(num_decoder_layers)
            ]
        )
        self.joint_z_pos_emb = nn.Embedding(2, d_model)
        self.q_emb = nn.Parameter(torch.randn(1, chunk_size, d_model))
        self.final_norm = nn.LayerNorm(d_model)
        self.value_head = nn.Linear(d_model, num_bins)

    def features(self, images, joints, z):
        """Pooled ACT observation tokens; the value head is not applied."""
        images = self._prepare_images(images)
        batch_size, num_cameras = images.shape[:2]
        if num_cameras != self.num_cameras:
            raise ValueError(f"Expected {self.num_cameras} cameras, got {num_cameras}")

        image_features = self.backbone(
            rearrange(images, "b m c h w -> (b m) c h w")
        )
        image_pos = repeat(
            self.image_pos_emb(image_features),
            "1 d h w -> 1 (m h w) d",
            m=num_cameras,
        ).to(dtype=image_features.dtype)

        image_tokens = rearrange(
            image_features,
            "(b m) c h w -> (b m) (h w) c",
            b=batch_size,
            m=num_cameras,
        )
        image_tokens = self.image_proj(image_tokens)
        image_tokens = rearrange(
            image_tokens,
            "(b m) n d -> b (m n) d",
            b=batch_size,
            m=num_cameras,
        )

        qpos_token = self.joint_proj(joints).unsqueeze(1)
        z_token = self.z_proj(z).unsqueeze(1)
        memory = torch.cat([z_token, qpos_token, image_tokens], dim=1)
        joint_z_pos = self.joint_z_pos_emb.weight.unsqueeze(0)
        pos = torch.cat([joint_z_pos, image_pos], dim=1)
        for block in self.transformer_encoder:
            memory = block(memory, pos=pos)

        query_pos = self.q_emb.expand(batch_size, -1, -1)
        out = torch.zeros_like(query_pos)
        for block in self.transformer_decoder:
            out = block(
                out,
                encoder_out=memory,
                pos=pos,
                query_pos=query_pos,
            )
        return self.final_norm(out).mean(dim=1)

    def forward(self, images, joints, z):
        logits = self.value_head(self.features(images, joints, z))
        return torch.distributions.Categorical(logits=logits)

    @staticmethod
    def _prepare_images(images):
        if images.dim() == 4:
            if images.shape[1] in (1, 3):
                images = images.unsqueeze(1)
            elif images.shape[-1] in (1, 3):
                images = rearrange(images, "b h w c -> b 1 c h w")
            else:
                raise ValueError(
                    "Expected image shape (B,C,H,W) or (B,H,W,C), "
                    f"got {tuple(images.shape)}"
                )
        elif images.dim() == 5:
            if images.shape[2] not in (1, 3) and images.shape[-1] in (1, 3):
                images = rearrange(images, "b m h w c -> b m c h w")
        else:
            raise ValueError(
                f"Expected images with 4 or 5 dimensions, got {tuple(images.shape)}"
            )
        if not images.is_floating_point():
            images = images.float() / 255.0
        return images
