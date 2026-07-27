from queue import Queue
import threading

import torch
from lerobot.async_inference.helpers import TimedAction

from models.act.eval_async import LockedRobotClient


class _FakeStub:
    def __init__(self):
        self.ready_calls = 0
        self.policy_setup_calls = 0

    def Ready(self, _request):
        self.ready_calls += 1

    def SendPolicyInstructions(self, _request):
        self.policy_setup_calls += 1


def _client_with_trial_state() -> LockedRobotClient:
    client = LockedRobotClient.__new__(LockedRobotClient)
    client.action_queue_lock = threading.RLock()
    client.action_queue = Queue()
    client.action_queue.put(
        TimedAction(timestamp=0.0, timestep=7, action=torch.zeros(2))
    )
    client.latest_action_lock = threading.Lock()
    client.latest_action = 6
    client.action_chunk_size = 30
    client.must_go = threading.Event()
    client.trial_active = threading.Event()
    client.stub = _FakeStub()
    return client


def test_prepare_trial_resets_async_state_without_reloading_policy():
    client = _client_with_trial_state()

    client.prepare_trial()

    assert client.stub.ready_calls == 1
    assert client.stub.policy_setup_calls == 0
    assert client.action_queue.empty()
    assert client.latest_action == -1
    assert client.action_chunk_size == -1
    assert client.must_go.is_set()
    assert client.trial_active.is_set()
