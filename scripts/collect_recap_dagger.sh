#!/bin/bash
# Collect RECAP DAgger episodes with the SFT policy (autonomous + corrections).
#
# Same tool as collect_recap_rollouts.sh, but here you take over mid-episode.
# The autonomous segment and the correction are stored in ONE episode with
# frame-level `intervention` flags, which is what RECAP needs: the autonomous
# part supplies negative-advantage examples, and the correction supplies the
# expert action. DAgger's own correction-only episodes cannot serve this role
# because they contain no autonomous frames.
#
# Keys during a rollout:
#   SPACE  start/stop a correction (safe leader handoff)
#   S      save as success
#   F      save as failure
#   ESC    discard
#
# Take over when the policy is clearly failing, not pre-emptively; the
# autonomous prefix is the training signal.
set -euo pipefail

ACT_SIM="$(cd "$(dirname "$0")/.." && pwd)"
OMX="${OMX_TRAINING_ROOT:-$HOME/projects/omx-training}"

POLICY="${POLICY:-$ACT_SIM/outputs/recap_sft_10k/pretrained_model}"
TASK="${TASK:-Place the sponge on the hot pad}"
REPO_ID="${REPO_ID:-RevanthGundala/rollout_032-recap-gen2-dagger}"
DATASET_ROOT="${DATASET_ROOT:-$HOME/.cache/huggingface/lerobot/rollout_032-recap-gen2-dagger}"

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
echo "DAgger collection: SPACE to take over, then S or F to label."
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
