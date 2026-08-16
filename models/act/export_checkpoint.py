"""Export this repo's LeRobot .pt checkpoint to pretrained_model format.

The live LeRobot eval path expects a directory consumable by
``Policy.from_pretrained`` plus saved pre/postprocessors. Our lightweight
trainers store raw ``.pt`` checkpoints, so this script converts them.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import shutil
import sys

import torch
from lerobot.configs import NormalizationMode
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.act.base import ACTConfig
from models.act.train_diffusion import make_diffusion_policy_config
from models.act.train_lerobot import make_lerobot_policy_config


def policy_feature_from_dict(value: dict) -> PolicyFeature:
    return PolicyFeature(FeatureType(value["type"]), tuple(value["shape"]))


def restore_policy_config_values(policy_config: dict) -> dict:
    restored = dict(policy_config)
    restored["input_features"] = {
        key: policy_feature_from_dict(value) for key, value in policy_config["input_features"].items()
    }
    restored["output_features"] = {
        key: policy_feature_from_dict(value) for key, value in policy_config["output_features"].items()
    }
    if "normalization_mapping" in policy_config:
        restored["normalization_mapping"] = {
            key: NormalizationMode(value) for key, value in policy_config["normalization_mapping"].items()
        }
    return restored


def infer_policy_type(policy_config: dict, requested_type: str) -> str:
    if requested_type != "auto":
        return requested_type
    if "down_dims" in policy_config or "horizon" in policy_config:
        return "diffusion"
    return "act"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LeRobot .pt checkpoint to pretrained_model directory.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--policy-type", choices=["auto", "act", "diffusion"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    run_config = checkpoint["config"]["run_config"]
    checkpoint_policy_config = checkpoint["config"]["policy_config"]
    cfg = ACTConfig(**{f.name: run_config[f.name] for f in fields(ACTConfig) if f.name in run_config})
    policy_type = infer_policy_type(checkpoint_policy_config, args.policy_type)
    if policy_type == "diffusion":
        policy_cfg = make_diffusion_policy_config(cfg, args.device)
        policy_cfg = type(policy_cfg)(**restore_policy_config_values({**checkpoint_policy_config, "device": args.device}))
    else:
        policy_cfg = make_lerobot_policy_config(cfg, args.device)
    metadata = LeRobotDatasetMetadata(cfg.dataset_repo_id, revision=cfg.dataset_revision)

    policy = make_policy(policy_cfg, ds_meta=metadata)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.config.device = args.device
    policy.to(args.device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(policy_cfg=policy_cfg, dataset_stats=metadata.stats)

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.with_suffix("") / "pretrained_model"
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    policy.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir, config_filename="policy_preprocessor.json")
    postprocessor.save_pretrained(output_dir, config_filename="policy_postprocessor.json")
    (output_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "run_config": run_config,
                "policy_type": policy_type,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Exported pretrained_model to {output_dir}")


if __name__ == "__main__":
    main()
