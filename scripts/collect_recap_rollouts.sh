#!/bin/bash
# Collect autonomous RECAP rollouts with the SFT policy (no corrections).
#
# Runs evaluation/collect_recap_act.py in the sibling omx-training checkout,
# pointed at this repo's exported RECAP policy. Each rollout is saved as one
# LeRobot episode with an outcome sidecar.
#
# Keys during a rollout:
#   S    save as success
#   F    save as failure
#   ESC  discard
#
# Vary the sponge and bowl placement between rollouts. The critic has to
# generalize across scenes, and the previous collection deliberately covered
# 6/6 position buckets.
set -euo pipefail

ACT_SIM="$(cd "$(dirname "$0")/.." && pwd)"
OMX="${OMX_TRAINING_ROOT:-$HOME/projects/omx-training}"

POLICY="${POLICY:-$ACT_SIM/outputs/recap_sft_10k/pretrained_model}"
TASK="${TASK:-Place the sponge on the hot pad}"
REPO_ID="${REPO_ID:-RevanthGundala/rollout_032-recap-gen2-autonomous}"
DATASET_ROOT="${DATASET_ROOT:-$HOME/.cache/huggingface/lerobot/rollout_032-recap-gen2-autonomous}"

if [ ! -d "$OMX" ]; then
  echo "omx-training checkout not found at $OMX (set OMX_TRAINING_ROOT)" >&2
  exit 1
fi
if [ ! -d "$POLICY" ]; then
  echo "Exported RECAP policy not found at $POLICY" >&2
  exit 1
fi

echo "policy       = $POLICY"
echo "dataset      = $REPO_ID"
echo "dataset root = $DATASET_ROOT"
echo "task         = $TASK"
echo
echo "Autonomous collection: do NOT press SPACE. Label each rollout S or F."
echo "Episodes this session: ${EPISODES:-1}  (set EPISODES=n)"
echo

EPISODES="${EPISODES:-1}"
cd "$OMX"
for i in $(seq 1 "$EPISODES"); do
echo "--- episode $i / $EPISODES ---"
uv run python evaluation/collect_recap_act.py \
  --policy-path "$POLICY" \
  --dataset-repo-id "$REPO_ID" \
  --dataset-root "$DATASET_ROOT" \
  --task "$TASK" \
  --actions-per-chunk 30 \
  --action-smoothing 0.7 \
  --chunk-size-threshold 0.4 \
  "$@"
done
