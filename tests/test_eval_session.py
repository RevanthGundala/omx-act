from queue import Queue
import io
import threading
from types import SimpleNamespace

import torch
from lerobot.async_inference.helpers import TimedAction

from models.act.eval_async import (
    LockedRobotClient,
    diagnostic_dataset_reference,
    load_session_diagnostics,
    prompt_for_trial_start,
    run_trial_sequence,
    trial_stop_instructions,
)


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


def test_control_trial_stops_without_global_shutdown_and_reuses_receiver_barrier():
    client = _client_with_trial_state()
    client.shutdown_event = threading.Event()
    client.config = SimpleNamespace(environment_dt=0.0)
    client.start_barrier = SimpleNamespace(wait_calls=0)
    client.start_barrier.wait = lambda: setattr(
        client.start_barrier,
        "wait_calls",
        client.start_barrier.wait_calls + 1,
    )
    client._control_loop_started = False
    client.actions_available = lambda: False
    client._ready_to_send_observation = lambda: False
    client.trial_active.set()

    first_reason = client.control_trial(
        task="test",
        duration_s=60.0,
        stop_requested=lambda: True,
        sleep=lambda _seconds: None,
    )
    client.trial_active.set()
    second_reason = client.control_trial(
        task="test",
        duration_s=60.0,
        stop_requested=lambda: True,
        sleep=lambda _seconds: None,
    )

    assert first_reason == "operator"
    assert second_reason == "operator"
    assert client.start_barrier.wait_calls == 1
    assert not client.shutdown_event.is_set()


def test_trial_sequence_reuses_client_and_runs_reset_callbacks():
    events = []

    class FakeClient:
        def prepare_trial(self):
            events.append("prepare")

        def control_trial(self, **kwargs):
            events.append(("control", kwargs["task"], kwargs["duration_s"]))
            return "operator"

    reasons = run_trial_sequence(
        FakeClient(),
        trials=2,
        task="pick",
        duration_s=60.0,
        stop_requested_factory=lambda _trial: (lambda: False),
        before_trial=lambda trial: events.append(("before", trial)),
        after_trial=lambda trial, reason: events.append(("after", trial, reason)),
    )

    assert reasons == ["operator", "operator"]
    assert events == [
        ("before", 1),
        "prepare",
        ("control", "pick", 60.0),
        ("after", 1, "operator"),
        ("before", 2),
        "prepare",
        ("control", "pick", 60.0),
        ("after", 2, "operator"),
    ]


def test_diagnostic_reference_can_use_seed_dataset_for_dagger_policy(tmp_path):
    policy_path = tmp_path / "pretrained_model"
    policy_path.mkdir()
    (policy_path / "train_config.json").write_text(
        '{"dataset": {"repo_id": "owner/dagger", "revision": "dagger-rev"}}'
    )

    assert diagnostic_dataset_reference(
        policy_path,
        repo_id_override="owner/seed",
        revision_override=None,
    ) == ("owner/seed", "main")


def test_trial_start_prompt_does_not_crash_without_interactive_stdin():
    stream = io.StringIO("")

    prompted = prompt_for_trial_start(
        trial=1,
        total_trials=2,
        stream=stream,
    )

    assert not prompted


def test_new_trial_rejects_entire_chunk_from_prior_observation():
    client = _client_with_trial_state()
    client.trial_active.set()
    client.trial_started_at = 100.25
    client.latest_action = -1
    while not client.action_queue.empty():
        client.action_queue.get_nowait()
    old_chunk = [
        TimedAction(
            timestamp=100.0 + index / 30,
            timestep=index,
            action=torch.zeros(2),
        )
        for index in range(30)
    ]

    client._aggregate_action_queues(old_chunk)

    assert client.action_queue.empty()


def test_trial_sequence_stops_after_watchdog_shutdown():
    events = []

    class FakeClient:
        running = False

        def prepare_trial(self):
            events.append("prepare")

        def control_trial(self, **_kwargs):
            return "shutdown"

    reasons = run_trial_sequence(
        FakeClient(),
        trials=3,
        task="pick",
        duration_s=60,
        stop_requested_factory=lambda _trial: (lambda: False),
        before_trial=lambda trial: events.append(("before", trial)),
        after_trial=lambda trial, reason: events.append(("after", trial, reason)),
    )

    assert reasons == ["shutdown"]
    assert events == [
        ("before", 1),
        "prepare",
        ("after", 1, "shutdown"),
    ]


def test_trial_stop_instructions_only_advertise_available_controls():
    interactive = trial_stop_instructions(
        duration_s=60,
        enter_enabled=True,
        stdin_interactive=True,
    )
    noninteractive = trial_stop_instructions(
        duration_s=60,
        enter_enabled=True,
        stdin_interactive=False,
    )
    enter_disabled = trial_stop_instructions(
        duration_s=0,
        enter_enabled=False,
        stdin_interactive=True,
    )

    assert "Enter" in interactive
    assert "Enter" not in noninteractive
    assert "60s" in noninteractive
    assert "Enter" not in enter_disabled
    assert "Ctrl-C" in enter_disabled


def test_session_diagnostics_respect_camera_disable_and_dataset_override(tmp_path):
    policy_path = tmp_path / "pretrained_model"
    policy_path.mkdir()
    (policy_path / "train_config.json").write_text(
        '{"dataset": {"repo_id": "owner/dagger", "revision": "dagger-rev"}}'
    )
    calls = []

    def prepare(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return "checker", "coverage"

    checker, coverage = load_session_diagnostics(
        policy_path,
        repo_id_override="owner/seed",
        revision_override=None,
        camera_view_check=False,
        camera_reference_count=8,
        prepare=prepare,
    )

    assert (checker, coverage) == ("checker", "coverage")
    assert calls == [
        (
            "owner/seed",
            {
                "revision": "main",
                "camera_name": "top",
                "max_alignment_references": 8,
                "enable_alignment": False,
            },
        )
    ]
