from queue import Queue
import threading

import numpy as np
import pytest

from models.act.eval_async import ActionStreamWatchdog
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


def test_lerobot_async_watchdog_aborts_when_first_chunk_never_arrives():
    class FakeClient:
        def __init__(self):
            self.shutdown_event = threading.Event()
            self.latest_action_lock = threading.Lock()
            self.latest_action = -1
            self.action_queue_lock = threading.Lock()
            self.action_queue = Queue()

    client = FakeClient()
    watchdog = ActionStreamWatchdog(
        client,
        first_action_timeout=0.01,
        starvation_timeout=0,
        poll_interval=0.001,
    )

    watchdog.run()

    assert client.shutdown_event.is_set()
    assert watchdog.reason == "no first action chunk received within 0.01s"
