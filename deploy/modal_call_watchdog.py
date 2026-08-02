"""Monitor and hard-stop one exact Modal training call."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import modal
from modal.call_graph import InputStatus


TERMINAL_STATUSES = {
    InputStatus.SUCCESS,
    InputStatus.FAILURE,
    InputStatus.INIT_FAILURE,
    InputStatus.TERMINATED,
    InputStatus.TIMEOUT,
}
ACTIVE_APP_STATES = {
    "initializing",
    "ephemeral",
    "ephemeral (detached)",
    "running",
    "deployed",
    "stopping...",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_json(command: list[str]):
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def record_value(record: dict, label: str):
    return record.get(label, record.get(label.lower().replace(" ", "_")))


def exact_app_record(app_id: str):
    return next(
        (
            record
            for record in run_json(["modal", "app", "list", "--json"])
            if record_value(record, "App ID") == app_id
        ),
        None,
    )


def exact_containers(app_id: str):
    return [
        record
        for record in run_json(["modal", "container", "list", "--json"])
        if record_value(record, "App ID") == app_id
    ]


def stop_exact_resource(function_call, app_id: str, log) -> None:
    log("Cancelling exact Modal function call and terminating its containers.")
    try:
        function_call.cancel(terminate_containers=True)
    except Exception as exc:
        log(f"Function-call cancellation raised {exc!r}; continuing with exact app stop.")
    for attempt in range(1, 6):
        result = subprocess.run(
            ["modal", "app", "stop", app_id],
            capture_output=True,
            text=True,
        )
        log(
            f"Exact app stop attempt {attempt}/5 exit={result.returncode} "
            f"stdout={result.stdout.strip()!r} stderr={result.stderr.strip()!r}"
        )
        time.sleep(10)
        app_record = exact_app_record(app_id)
        containers = exact_containers(app_id)
        state = None if app_record is None else str(record_value(app_record, "State") or "").lower()
        if not containers and state not in ACTIVE_APP_STATES:
            return
    raise RuntimeError(f"Could not verify exact Modal app {app_id} stopped.")


def download_outputs(
    *,
    volume: str,
    remote_path: str,
    local_path: Path,
    output_kind: str,
    expected_final_step: int | None,
    log,
) -> bool:
    if output_kind == "lerobot":
        checkpoint_root = f"{remote_path.rstrip('/')}/checkpoints"
        try:
            entries = run_json(["modal", "volume", "ls", volume, checkpoint_root, "--json"])
        except Exception as exc:
            log(f"Could not list LeRobot checkpoints before download: {exc!r}")
            return False
        checkpoint_names = sorted(
            Path(record_value(entry, "Filename")).name
            for entry in entries
            if record_value(entry, "Type") == "dir"
            and Path(record_value(entry, "Filename")).name.isdigit()
        )
        if not checkpoint_names:
            log(f"No LeRobot checkpoint found under {checkpoint_root}.")
            return False
        preferred = (
            f"{expected_final_step:06d}"
            if expected_final_step is not None
            else checkpoint_names[-1]
        )
        checkpoint_name = preferred if preferred in checkpoint_names else checkpoint_names[-1]
        remote_checkpoint = f"{checkpoint_root}/{checkpoint_name}"
        local_checkpoint = local_path / "checkpoints" / checkpoint_name
        local_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, 4):
            result = subprocess.run(
                [
                    "modal",
                    "volume",
                    "get",
                    "--force",
                    volume,
                    remote_checkpoint,
                    str(local_checkpoint),
                ],
                capture_output=True,
                text=True,
            )
            log(
                f"Download LeRobot checkpoint {checkpoint_name} attempt {attempt}/3 "
                f"exit={result.returncode} stdout={result.stdout.strip()!r} "
                f"stderr={result.stderr.strip()!r}"
            )
            if result.returncode == 0:
                return True
            time.sleep(20)
        return False

    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        entries = run_json(["modal", "volume", "ls", volume, remote_path, "--json"])
    except Exception as exc:
        log(f"Could not list output runs before download: {exc!r}")
        return False
    run_dirs = sorted(
        record_value(entry, "Filename")
        for entry in entries
        if record_value(entry, "Type") == "dir"
    )
    if not run_dirs:
        log(f"No run directory found under {remote_path}.")
        return False
    run_path = run_dirs[-1]
    required_files = ("config.json", "metrics.jsonl", "checkpoints/last.pt")
    for relative_path in required_files:
        destination = local_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, 4):
            result = subprocess.run(
                [
                    "modal",
                    "volume",
                    "get",
                    "--force",
                    volume,
                    f"{run_path}/{relative_path}",
                    str(destination),
                ],
                capture_output=True,
                text=True,
            )
            log(
                f"Download {relative_path} attempt {attempt}/3 exit={result.returncode} "
                f"stdout={result.stdout.strip()!r} stderr={result.stderr.strip()!r}"
            )
            if result.returncode == 0:
                break
            time.sleep(20)
        else:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--call-id", required=True)
    parser.add_argument("--app-id", required=True)
    parser.add_argument("--max-runtime-minutes", type=int, default=180)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--volume", required=True)
    parser.add_argument("--remote-output-path", required=True)
    parser.add_argument("--training-status-path", required=True)
    parser.add_argument("--local-output-path", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--estimated-hourly-cost", type=float, required=True)
    parser.add_argument(
        "--output-kind",
        choices=("custom", "lerobot"),
        default="custom",
    )
    parser.add_argument("--expected-final-step", type=int, default=None)
    args = parser.parse_args()

    args.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        line = f"{utc_now()} {message}"
        print(line, flush=True)
        with args.log_path.open("a") as file:
            file.write(line + "\n")

    function_call = modal.FunctionCall.from_id(args.call_id)
    monitor_started = time.monotonic()
    deadline = time.monotonic() + args.max_runtime_minutes * 60
    log(
        f"Monitoring call={args.call_id} app={args.app_id} "
        f"hard_runtime_minutes={args.max_runtime_minutes}."
    )
    terminal_status = None
    forced_stop_reason = None
    last_training_step = None
    stalled_checks = 0
    while time.monotonic() < deadline:
        elapsed_seconds = time.monotonic() - monitor_started
        remaining_seconds = max(0.0, deadline - time.monotonic())
        estimated_cost = elapsed_seconds / 3600 * args.estimated_hourly_cost
        graph = function_call.get_call_graph()
        statuses = [node.status for node in graph]
        terminal_status = statuses[0] if statuses else None
        app_record = exact_app_record(args.app_id)
        containers = exact_containers(args.app_id)
        log(
            f"provider_state call_statuses={[status.name for status in statuses]} "
            f"app={app_record} containers={containers} "
            f"elapsed_minutes={elapsed_seconds / 60:.1f} "
            f"remaining_minutes={remaining_seconds / 60:.1f} "
            f"estimated_cost_usd={estimated_cost:.3f}"
        )

        for container in containers:
            container_id = record_value(container, "Container ID")
            if not container_id:
                continue
            telemetry = subprocess.run(
                [
                    "modal",
                    "container",
                    "exec",
                    container_id,
                    "--",
                    "nvidia-smi",
                    "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            log(
                f"gpu_telemetry container={container_id} exit={telemetry.returncode} "
                f"stdout={telemetry.stdout.strip()!r} stderr={telemetry.stderr.strip()!r}"
            )

        volume_state = subprocess.run(
            [
                "modal",
                "volume",
                "ls",
                "--json",
                args.volume,
                args.remote_output_path,
            ],
            capture_output=True,
            text=True,
        )
        log(
            f"checkpoint_state exit={volume_state.returncode} "
            f"stdout={volume_state.stdout.strip()!r} stderr={volume_state.stderr.strip()!r}"
        )
        training_state = subprocess.run(
            [
                "modal",
                "volume",
                "get",
                args.volume,
                args.training_status_path,
                "-",
            ],
            capture_output=True,
            text=True,
        )
        log(
            f"training_progress exit={training_state.returncode} "
            f"stdout={training_state.stdout.strip()!r} stderr={training_state.stderr.strip()!r}"
        )
        if training_state.returncode == 0 and training_state.stdout.strip():
            try:
                progress = json.loads(training_state.stdout)
            except json.JSONDecodeError:
                progress = None
            if isinstance(progress, dict) and progress.get("status") == "running":
                current_step = progress.get("step")
                if isinstance(current_step, int):
                    if current_step == last_training_step:
                        stalled_checks += 1
                    else:
                        stalled_checks = 0
                        last_training_step = current_step
                    if stalled_checks >= 2:
                        forced_stop_reason = (
                            f"Training stalled at step {current_step} for two monitor checks."
                        )
                        log(forced_stop_reason)
                        stop_exact_resource(function_call, args.app_id, log)
                        break

        if terminal_status in TERMINAL_STATUSES:
            break
        time.sleep(args.poll_seconds)

    timed_out = terminal_status not in TERMINAL_STATUSES and forced_stop_reason is None
    if timed_out:
        log("Hard runtime reached.")
        stop_exact_resource(function_call, args.app_id, log)

    downloaded = download_outputs(
        volume=args.volume,
        remote_path=args.remote_output_path,
        local_path=args.local_output_path,
        output_kind=args.output_kind,
        expected_final_step=args.expected_final_step,
        log=log,
    )
    log(f"downloaded={downloaded}")
    if not timed_out and forced_stop_reason is None:
        stop_exact_resource(function_call, args.app_id, log)

    app_record = exact_app_record(args.app_id)
    containers = exact_containers(args.app_id)
    state = None if app_record is None else str(record_value(app_record, "State") or "").lower()
    if containers or state in ACTIVE_APP_STATES:
        raise RuntimeError(
            f"Fresh Modal audit still shows billable resource: app={app_record}, containers={containers}"
        )
    log(
        f"Fresh Modal audit confirms exact app is non-active and has no containers: "
        f"app={app_record}, containers={containers}"
    )
    if not downloaded:
        raise RuntimeError("Modal call ended but output download failed after retries.")
    if timed_out:
        raise TimeoutError(f"Modal call {args.call_id} hit the hard runtime.")
    if forced_stop_reason is not None:
        raise RuntimeError(forced_stop_reason)
    if terminal_status is not InputStatus.SUCCESS:
        raise RuntimeError(f"Modal call ended with {terminal_status}.")


if __name__ == "__main__":
    main()
