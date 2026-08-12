from collections import deque
from queue import Queue
import threading

import numpy as np
import pytest
import torch
from lerobot.async_inference.helpers import TimedAction

from models.act.eval_async import ActionStreamWatchdog, LockedRobotClient
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


def test_lerobot_async_aggregation_holds_queue_lock_while_iterating():
    iteration_started = threading.Event()
    release_iteration = threading.Event()

    class BlockingDeque(deque):
        def __iter__(self):
            iteration_started.set()
            assert release_iteration.wait(timeout=1)
            return super().__iter__()

    client = LockedRobotClient.__new__(LockedRobotClient)
    client.action_queue_lock = threading.RLock()
    client._replan_lock = threading.Lock()
    client._replan_in_flight = False
    client._replan_requested_at = 0.0
    client.action_queue = Queue()
    client.action_queue.queue = BlockingDeque(
        [TimedAction(timestamp=0.0, timestep=1, action=torch.tensor([1.0]))]
    )
    client.latest_action_lock = threading.Lock()
    client.latest_action = 0
    incoming = [
        TimedAction(timestamp=0.1, timestep=1, action=torch.tensor([2.0]))
    ]

    worker = threading.Thread(
        target=client._aggregate_action_queues,
        args=(incoming, lambda old, new: (old + new) / 2),
    )
    worker.start()
    assert iteration_started.wait(timeout=1)

    acquired_during_iteration = client.action_queue_lock.acquire(timeout=0.01)
    if acquired_during_iteration:
        client.action_queue_lock.release()
    release_iteration.set()
    worker.join(timeout=1)

    assert not acquired_during_iteration
    assert not worker.is_alive()
