"""Shared schema handling for builders that merge LeRobot datasets.

Both DAgger builders combine several source datasets into one, so both have to
answer the same question first: can these actually be merged? The rule is that
they agree on frame rate, camera set, and joint ordering. Joint ordering is the
one that matters most and is the least visible -- two datasets can carry the
same six joints in a different order, merge without complaint, and produce a
dataset whose actions are silently permuted partway through.
"""

from __future__ import annotations

from dataclasses import dataclass

IMAGE_PREFIX = "observation.images."


@dataclass(frozen=True)
class SourceSchema:
    """The parts of a source dataset a merged output has to reproduce."""

    fps: int
    camera_keys: list[str]
    state_feature: dict
    action_feature: dict
    height: int
    width: int
    robot_type: str | None

    @property
    def state_dim(self) -> int:
        return int(self.state_feature["shape"][0])

    @property
    def action_dim(self) -> int:
        return int(self.action_feature["shape"][0])


def camera_keys_of(dataset) -> list[str]:
    return sorted(key for key in dataset.meta.features if key.startswith(IMAGE_PREFIX))


def assert_mergeable(sources: dict[str, object]) -> SourceSchema:
    """Check that every source can merge into one dataset, and describe it.

    ``sources`` maps a label used in error messages to a LeRobotDataset. The
    first entry is the reference every other source is compared against.
    """
    if not sources:
        raise ValueError("At least one source dataset is required")

    labels = list(sources)
    reference_label, reference = labels[0], sources[labels[0]]
    fps = int(round(reference.fps))
    camera_keys = camera_keys_of(reference)
    if not camera_keys:
        raise ValueError(f"{reference_label} has no {IMAGE_PREFIX}* features")
    state_feature = reference.meta.features["observation.state"]
    action_feature = reference.meta.features["action"]

    for label in labels[1:]:
        dataset = sources[label]
        if int(round(dataset.fps)) != fps:
            raise ValueError(
                f"fps mismatch: {reference_label}={fps} {label}={int(round(dataset.fps))}"
            )
        if camera_keys_of(dataset) != camera_keys:
            raise ValueError(
                f"camera keys differ: {reference_label}={camera_keys} "
                f"{label}={camera_keys_of(dataset)}"
            )
        if dataset.meta.features["observation.state"].get("names") != state_feature.get(
            "names"
        ):
            raise ValueError(
                f"observation.state joint names differ between {reference_label} and {label}"
            )
        if dataset.meta.features["action"].get("names") != action_feature.get("names"):
            raise ValueError(
                f"action joint names differ between {reference_label} and {label}"
            )

    height, width = reference[0][camera_keys[0]].shape[-2:]
    return SourceSchema(
        fps=fps,
        camera_keys=camera_keys,
        state_feature=state_feature,
        action_feature=action_feature,
        height=int(height),
        width=int(width),
        robot_type=reference.meta.robot_type,
    )


def merged_features(schema: SourceSchema) -> dict:
    """Feature spec for a merged dataset that keeps per-frame provenance."""
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (schema.state_dim,),
            "names": schema.state_feature.get("names"),
        },
        "action": {
            "dtype": "float32",
            "shape": (schema.action_dim,),
            "names": schema.action_feature.get("names"),
        },
        # Carries demonstration vs correction into the merged dataset. RECAP
        # advantage labelling reads it to force corrections positive.
        "intervention": {"dtype": "bool", "shape": (1,), "names": None},
    }
    for key in schema.camera_keys:
        features[key] = {
            "dtype": "video",
            "shape": (3, schema.height, schema.width),
            "names": ["channels", "height", "width"],
        }
    return features
