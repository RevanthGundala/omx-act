"""LeRobot policy adapter, exporter, and async-server bootstrap for custom ACT.

Do not move this file. omx-training loads it by exact path through
--policy-code-root (see its evaluation/recap_policy.py, RECAP_ADAPTER_PATH =
"models/act/lerobot_policy.py"), so a rename here breaks robot evaluation in the
other repo with no failure visible from this one.
"""

from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field, fields
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import torch
import torch.nn.functional as F
from lerobot.configs import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies.act.configuration_act import ACTConfig as LeRobotACTConfig
from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE
from torch import Tensor

from configs.act.base import ACTConfig
from models.act.act import ACTPolicy as BaseACTPolicy
from models.recap.act_recap import (
    ACTPolicy as RecapACTPolicy,
    AdvantageCondition,
)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
# eval_async launches the policy server in its own process, so the requested
# conditioning token crosses the boundary as an environment variable.
ADVANTAGE_ENV_VAR = "RECAP_EVAL_ADVANTAGE"


def resolve_advantage_condition(condition: AdvantageCondition | int | str) -> AdvantageCondition:
    if isinstance(condition, str):
        try:
            return AdvantageCondition[condition.upper()]
        except KeyError as exc:
            raise ValueError(f"Unknown advantage condition: {condition!r}") from exc
    try:
        return AdvantageCondition(condition)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid advantage condition: {condition!r}") from exc


def default_advantage_condition(recap: bool) -> AdvantageCondition:
    """RECAP policies are trained to be asked for POSITIVE at inference.

    Defaulting to UNCONDITIONED would silently evaluate the marginal policy
    averaged over good and bad experience, which reads as "RECAP did not help"
    when the conditioning was simply never switched on.
    """
    if not recap:
        return AdvantageCondition.UNCONDITIONED
    override = os.environ.get(ADVANTAGE_ENV_VAR)
    return resolve_advantage_condition(override) if override else AdvantageCondition.POSITIVE


@PreTrainedConfig.register_subclass("custom_act")
@dataclass
class CustomACTConfig(LeRobotACTConfig):
    source_model_config: dict[str, Any] = field(default_factory=dict)
    dataset_repo_id: str = ""
    dataset_revision: str = "main"
    action_space: str = "absolute"
    recap: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.source_model_config or not self.dataset_repo_id:
            raise ValueError("source_model_config and dataset_repo_id are required.")
        if self.action_space != "absolute":
            raise ValueError("The custom ACT LeRobot adapter supports only absolute actions.")


class CustomACTPolicy(PreTrainedPolicy):
    config_class = CustomACTConfig
    name = "custom_act"

    def __init__(self, config: CustomACTConfig):
        super().__init__(config)
        allowed = {item.name for item in fields(ACTConfig)}
        source = {key: value for key, value in config.source_model_config.items() if key in allowed}
        source["camera_names"] = tuple(source["camera_names"])
        self.source_config = ACTConfig(**source)
        model_class = RecapACTPolicy if config.recap else BaseACTPolicy
        self.model = model_class(self.source_config)
        self._advantage_condition = default_advantage_condition(config.recap)
        self._action_queue: deque[Tensor] = deque(maxlen=config.n_action_steps)
        self.temporal_ensembler = (
            ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)
            if config.temporal_ensemble_coeff is not None
            else None
        )
        self._validate_config()

    @property
    def camera_names(self) -> tuple[str, ...]:
        return tuple(self.source_config.camera_names)

    def _validate_config(self) -> None:
        if self.config.chunk_size != self.source_config.chunk_size:
            raise ValueError("Adapter and source chunk sizes differ.")
        if self.config.action_feature is None or self.config.action_feature.shape != (
            self.source_config.joint_dim,
        ):
            raise ValueError("Adapter action feature does not match the source model.")
        missing = {
            f"observation.images.{camera}"
            for camera in self.camera_names
        } - set(self.config.image_features)
        if missing:
            raise ValueError(f"Adapter is missing source cameras: {sorted(missing)}")

    def get_optim_params(self) -> list[dict]:
        return [{"params": [parameter for parameter in self.parameters() if parameter.requires_grad]}]

    def reset(self) -> None:
        self._action_queue.clear()
        if self.temporal_ensembler is not None:
            self.temporal_ensembler.reset()

    @property
    def advantage_condition(self) -> AdvantageCondition:
        return self._advantage_condition

    def set_advantage_condition(
        self,
        condition: AdvantageCondition | int | str,
    ) -> None:
        if not self.config.recap:
            raise RuntimeError(
                "Advantage conditioning requires a RECAP-exported custom ACT policy"
            )
        self._advantage_condition = resolve_advantage_condition(condition)
        self.reset()

    def _inputs(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        images = torch.stack(
            [batch[f"observation.images.{camera}"] for camera in self.camera_names],
            dim=1,
        )
        return images, batch[OBS_STATE]

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **_kwargs) -> Tensor:
        self.eval()
        if self.config.recap:
            return self.model(
                *self._inputs(batch),
                advantage=self._advantage_condition,
            )[0]
        return self.model(*self._inputs(batch))[0]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **_kwargs) -> Tensor:
        self.eval()
        if self.temporal_ensembler is not None:
            return self.temporal_ensembler.update(self.predict_action_chunk(batch))
        if not self._action_queue:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        if ACTION not in batch:
            raise KeyError(f"missing required training target {ACTION!r}")
        action_mask = None if "action_is_pad" not in batch else ~batch["action_is_pad"].bool()
        model_kwargs = {}
        if self.config.recap:
            model_kwargs["advantage"] = batch.get(
                "advantage_condition",
                AdvantageCondition.UNCONDITIONED,
            )
        actions_hat, mu, log_var = self.model(
            *self._inputs(batch),
            batch[ACTION],
            action_mask,
            **model_kwargs,
        )
        error = F.l1_loss(actions_hat, batch[ACTION], reduction="none")
        if action_mask is None:
            action_loss = error.mean()
        else:
            valid_values = (action_mask.sum() * error.shape[-1]).clamp_min(1)
            action_loss = (error * action_mask.unsqueeze(-1)).sum() / valid_values
        kl_loss = (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp())).sum(dim=1).mean()
        loss = action_loss + self.source_config.kl_weight * kl_loss
        return loss, {"l1_loss": action_loss.item(), "kld_loss": kl_loss.item()}


def export_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    device: str,
    action_space: str,
    overwrite: bool,
    recap: bool = False,
) -> Path:
    if action_space != "absolute":
        raise NotImplementedError(
            "Delta export is unsafe because RobotClient would treat deltas as absolute targets."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source_dict = checkpoint["config"]
    source = ACTConfig(
        **{
            item.name: (
                tuple(source_dict[item.name])
                if item.name == "camera_names"
                else source_dict[item.name]
            )
            for item in fields(ACTConfig)
            if item.name in source_dict
        }
    )
    config = _adapter_config(
        source,
        source_dict,
        device,
        action_space,
        recap=recap,
    )
    metadata = LeRobotDatasetMetadata(source.dataset_repo_id, revision=source.dataset_revision)
    policy = CustomACTPolicy(config)
    try:
        policy.model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        mode = "RECAP" if recap else "base ACT"
        raise ValueError(
            f"Checkpoint is incompatible with requested {mode} export"
        ) from exc
    policy.to(device).eval()
    preprocessor, postprocessor = make_act_pre_post_processors(
        config,
        dataset_stats=_processor_stats(metadata, source.camera_names),
    )

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    policy.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir, config_filename="policy_preprocessor.json")
    postprocessor.save_pretrained(output_dir, config_filename="policy_postprocessor.json")
    (output_dir / "train_config.json").write_text(
        json.dumps(
            {"dataset": {"repo_id": source.dataset_repo_id, "revision": source.dataset_revision}},
            indent=2,
        )
        + "\n"
    )
    (output_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "source_checkpoint": str(checkpoint_path),
                "source_step": checkpoint.get("step"),
                "policy_type": "custom_act",
                "recap": recap,
                "action_space": action_space,
                "source_model_config": source_dict,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return output_dir


def _adapter_config(
    source: ACTConfig,
    source_dict: dict,
    device: str,
    action_space: str,
    recap: bool = False,
) -> CustomACTConfig:
    input_features = {"observation.state": PolicyFeature(FeatureType.STATE, (source.joint_dim,))}
    input_features.update(
        {
            f"observation.images.{camera}": PolicyFeature(FeatureType.VISUAL, (3, 480, 640))
            for camera in source.camera_names
        }
    )
    return CustomACTConfig(
        input_features=input_features,
        output_features={"action": PolicyFeature(FeatureType.ACTION, (source.joint_dim,))},
        device=device,
        chunk_size=source.chunk_size,
        n_action_steps=1,
        temporal_ensemble_coeff=source.exp_weight,
        dim_model=source.d_model,
        n_heads=source.num_heads,
        dim_feedforward=source.mlp_dim,
        n_encoder_layers=source.num_encoder_layers,
        n_decoder_layers=source.num_decoder_layers,
        latent_dim=source.d_z,
        n_vae_encoder_layers=source.num_vae_encoder_layers,
        dropout=source.dropout,
        kl_weight=source.kl_weight,
        source_model_config=source_dict,
        dataset_repo_id=source.dataset_repo_id,
        dataset_revision=source.dataset_revision,
        action_space=action_space,
        recap=recap,
    )


def _processor_stats(metadata, camera_names) -> dict:
    stats = deepcopy(metadata.stats)
    for camera in camera_names:
        stats[f"observation.images.{camera}"]["mean"] = IMAGENET_MEAN.clone()
        stats[f"observation.images.{camera}"]["std"] = IMAGENET_STD.clone()
    return stats


def serve() -> None:
    from lerobot.async_inference import policy_server

    if "custom_act" not in policy_server.SUPPORTED_POLICIES:
        policy_server.SUPPORTED_POLICIES.append("custom_act")
    policy_server.serve()


def main() -> None:
    if sys.argv[1:2] == ["serve"]:
        del sys.argv[1]
        serve()
        return
    if sys.argv[1:2] == ["export"]:
        del sys.argv[1]
    parser = argparse.ArgumentParser(description="Export custom ACT as a LeRobot policy.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--action-space", choices=["absolute", "delta"], required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--recap",
        action="store_true",
        help="Export a models.recap.act_recap checkpoint.",
    )
    args = parser.parse_args()
    output = export_checkpoint(
        Path(args.checkpoint),
        Path(args.output_dir),
        device=args.device,
        action_space=args.action_space,
        overwrite=args.overwrite,
        recap=args.recap,
    )
    print(f"Exported custom ACT LeRobot policy to {output}")


if __name__ == "__main__":
    main()
