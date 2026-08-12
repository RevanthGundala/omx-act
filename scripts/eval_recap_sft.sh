#!/bin/bash
# Roll out the RECAP SFT policy (10k steps, advantage fixed to POSITIVE).
#
# The RECAP export conditions on POSITIVE by default. Override with
# --advantage for the ablation arms described at the bottom of this file.
set -euo pipefail

cd "$(dirname "$0")/.."

POLICY="outputs/recap_sft_10k/pretrained_model"
TASK="Place the sponge on the hot pad"
TRIALS="${TRIALS:-10}"
ADVANTAGE="${ADVANTAGE:-positive}"

if [ ! -d "$POLICY" ]; then
  echo "Missing exported policy at $POLICY" >&2
  echo "Export it with:" >&2
  echo "  uv run python -m models.act.lerobot_policy export \\" >&2
  echo "    --checkpoint outputs/recap_sft_10k/step_010000.pt \\" >&2
  echo "    --output-dir $POLICY --action-space absolute --recap" >&2
  exit 1
fi

echo "policy=$POLICY"
echo "advantage=$ADVANTAGE  trials=$TRIALS"
echo

PYTHONPATH=. uv run python -m models.act.eval_async \
  --policy-path "$POLICY" \
  --policy-type custom_act \
  --advantage "$ADVANTAGE" \
  --task "$TASK" \
  --trials "$TRIALS" \
  --actions-per-chunk 30 \
  --policy-device mps \
  --record-rollout \
  --rollout-output-root "outputs/eval_rollouts_recap_${ADVANTAGE}"

# Ablation arms, same weights and same compute, only the token differs:
#   ADVANTAGE=unconditioned ./scripts/eval_recap_sft.sh
#   ADVANTAGE=negative      ./scripts/eval_recap_sft.sh
#
# After SFT these should look similar, because SFT trains on all-POSITIVE data
# and the policy has never seen NEGATIVE. The arms only separate once the
# improvement loop trains on mixed labels.
