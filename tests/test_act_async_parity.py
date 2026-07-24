import numpy as np
import pytest

from models.act.eval_async_custom import limit_action_chunk


def test_native_async_limits_actions_like_lerobot_policy_server():
    chunk = np.arange(24, dtype=np.float32).reshape(4, 6)

    limited = limit_action_chunk(chunk, actions_per_chunk=2)

    assert np.array_equal(limited, chunk[:2])


@pytest.mark.parametrize("actions_per_chunk", [0, 5])
def test_native_async_rejects_invalid_actions_per_chunk(actions_per_chunk):
    chunk = np.zeros((4, 6), dtype=np.float32)

    with pytest.raises(ValueError, match="actions_per_chunk"):
        limit_action_chunk(chunk, actions_per_chunk)
