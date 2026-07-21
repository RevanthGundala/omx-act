from dataclasses import replace

import torch
import torch.nn as nn
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig as LeRobotACTConfig
from lerobot.policies.act.modeling_act import ACT as LeRobotACT
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from configs.act.base import ACTConfig
from models.act.act import ACTPolicy


def _copy_linear(target, source):
    target.weight.data.copy_(source.weight.data)
    target.bias.data.copy_(source.bias.data)


def _copy_attention(target, source):
    target.qkv_proj.weight.data.copy_(source.in_proj_weight.data)
    target.qkv_proj.bias.data.copy_(source.in_proj_bias.data)
    _copy_linear(target.out_proj, source.out_proj)


def _copy_encoder_block(target, source):
    _copy_attention(target.self_attn, source.self_attn)
    _copy_linear(target.mlp[0], source.linear1)
    _copy_linear(target.mlp[3], source.linear2)
    target.self_norm.load_state_dict(source.norm1.state_dict())
    target.mlp_norm.load_state_dict(source.norm2.state_dict())


def _copy_decoder_block(target, source):
    _copy_attention(target.self_attn, source.self_attn)
    query_weight, key_weight, value_weight = source.multihead_attn.in_proj_weight.chunk(3)
    query_bias, key_bias, value_bias = source.multihead_attn.in_proj_bias.chunk(3)
    target.cross_attn.cross_q_proj.weight.data.copy_(query_weight)
    target.cross_attn.cross_q_proj.bias.data.copy_(query_bias)
    target.cross_attn.cross_k_proj.weight.data.copy_(key_weight)
    target.cross_attn.cross_k_proj.bias.data.copy_(key_bias)
    target.cross_attn.cross_v_proj.weight.data.copy_(value_weight)
    target.cross_attn.cross_v_proj.bias.data.copy_(value_bias)
    _copy_linear(target.cross_attn.out_proj, source.multihead_attn.out_proj)
    _copy_linear(target.mlp[0], source.linear1)
    _copy_linear(target.mlp[3], source.linear2)
    target.self_norm.load_state_dict(source.norm1.state_dict())
    target.cross_norm.load_state_dict(source.norm2.state_dict())
    target.mlp_norm.load_state_dict(source.norm3.state_dict())


def _lerobot_config():
    return LeRobotACTConfig(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (6,)),
            "observation.images.top": PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))},
        device="cpu",
        chunk_size=4,
        n_action_steps=1,
        temporal_ensemble_coeff=None,
        pretrained_backbone_weights=None,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        latent_dim=8,
        n_vae_encoder_layers=1,
        dropout=0.0,
        pre_norm=False,
        feedforward_activation="relu",
    )


def _custom_config():
    return ACTConfig(
        dataset_repo_id="owner/dataset",
        camera_names=("top",),
        chunk_size=4,
        d_model=32,
        joint_dim=6,
        d_z=8,
        num_vae_encoder_layers=1,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        mlp_dim=64,
        dropout=0.0,
        stock_parity=True,
    )


def _copy_inference_weights(custom: ACTPolicy, official: LeRobotACT):
    backbone_names = {
        0: "conv1",
        1: "bn1",
        4: "layer1",
        5: "layer2",
        6: "layer3",
        7: "layer4",
    }
    for custom_index, official_name in backbone_names.items():
        custom.decoder.backbone[custom_index].load_state_dict(
            official.backbone._modules[official_name].state_dict()
        )

    _copy_linear(custom.decoder.z_proj, official.encoder_latent_input_proj)
    _copy_linear(custom.decoder.joint_proj, official.encoder_robot_state_input_proj)
    custom.decoder.image_proj.weight.data.copy_(
        official.encoder_img_feat_input_proj.weight.data.squeeze(-1).squeeze(-1)
    )
    custom.decoder.image_proj.bias.data.copy_(
        official.encoder_img_feat_input_proj.bias.data
    )
    custom.decoder.joint_z_pos_emb.weight.data.copy_(
        official.encoder_1d_feature_pos_embed.weight.data
    )
    custom.decoder.q_emb.data.copy_(
        official.decoder_pos_embed.weight.data.unsqueeze(0)
    )
    _copy_encoder_block(
        custom.decoder.transformer_encoder[0],
        official.encoder.layers[0],
    )
    _copy_decoder_block(
        custom.decoder.transformer_decoder[0],
        official.decoder.layers[0],
    )
    custom.decoder.final_norm.load_state_dict(official.decoder.norm.state_dict())
    _copy_linear(custom.decoder.act_proj, official.action_head)


def _copy_vae_weights(custom: ACTPolicy, official: LeRobotACT):
    custom.cvae_encoder.cls_token.data.copy_(
        official.vae_encoder_cls_embed.weight.data.unsqueeze(0)
    )
    _copy_linear(
        custom.cvae_encoder.joint_proj,
        official.vae_encoder_robot_state_input_proj,
    )
    _copy_linear(
        custom.cvae_encoder.act_proj,
        official.vae_encoder_action_input_proj,
    )
    _copy_linear(
        custom.cvae_encoder.out_proj,
        official.vae_encoder_latent_output_proj,
    )
    _copy_encoder_block(
        custom.cvae_encoder.transformer_encoder[0],
        official.vae_encoder.layers[0],
    )


def test_stock_parity_mode_matches_lerobot_inference_with_mapped_weights():
    torch.manual_seed(0)
    official = LeRobotACT(_lerobot_config()).eval()
    custom = ACTPolicy(_custom_config(), pretrained_backbone=False).eval()
    _copy_inference_weights(custom, official)
    custom.decoder.camera_emb.data.fill_(100)

    state = torch.randn(2, 6)
    image = torch.randn(2, 3, 64, 64)
    with torch.inference_mode():
        expected = official({OBS_STATE: state, OBS_IMAGES: [image]})[0]
        actual = custom(image.unsqueeze(1), state)[0]

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_stock_parity_mode_uses_stock_transformer_topology():
    config = replace(_custom_config(), dropout=0.1)
    custom = ACTPolicy(config, pretrained_backbone=False)

    vae_block = custom.cvae_encoder.transformer_encoder[0]
    encoder_block = custom.decoder.transformer_encoder[0]
    decoder_block = custom.decoder.transformer_decoder[0]
    for block in (vae_block, encoder_block, decoder_block):
        assert isinstance(block.mlp[1], nn.ReLU)
        assert isinstance(block.self_residual_dropout, nn.Dropout)
        assert block.self_residual_dropout.p == 0.1
    assert isinstance(decoder_block.cross_residual_dropout, nn.Dropout)
    assert decoder_block.cross_residual_dropout.p == 0.1
    assert isinstance(custom.decoder.final_norm, nn.LayerNorm)


def test_stock_parity_mode_matches_lerobot_training_forward_with_mapped_weights():
    torch.manual_seed(0)
    official = LeRobotACT(_lerobot_config()).train()
    custom = ACTPolicy(_custom_config(), pretrained_backbone=False).train()
    _copy_inference_weights(custom, official)
    _copy_vae_weights(custom, official)

    state = torch.randn(2, 6)
    image = torch.randn(2, 3, 64, 64)
    actions = torch.randn(2, 4, 6)
    action_is_pad = torch.tensor(
        [[False, False, False, False], [False, False, True, True]]
    )

    torch.manual_seed(123)
    expected_actions, (expected_mu, expected_log_var) = official(
        {
            OBS_STATE: state,
            OBS_IMAGES: [image],
            ACTION: actions,
            "action_is_pad": action_is_pad,
        }
    )
    torch.manual_seed(123)
    actual_actions, actual_mu, actual_log_var = custom(
        image.unsqueeze(1),
        state,
        actions,
        ~action_is_pad,
    )

    assert torch.allclose(actual_actions, expected_actions, atol=1e-5, rtol=1e-5)
    assert torch.allclose(actual_mu, expected_mu, atol=1e-5, rtol=1e-5)
    assert torch.allclose(actual_log_var, expected_log_var, atol=1e-5, rtol=1e-5)
