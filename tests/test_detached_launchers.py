from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = (
    ROOT / "run_nemotron35_tinylora_sweep.sh",
    ROOT / "run_qwen25_7b_tinylora_canary.sh",
)


class DetachedLauncherTests(unittest.TestCase):
    def test_shell_syntax(self) -> None:
        subprocess.run(
            ["bash", "-n", *(str(path) for path in LAUNCHERS)],
            check=True,
            cwd=ROOT,
        )

    def test_training_uses_a_detached_owned_container(self) -> None:
        for path in LAUNCHERS:
            with self.subTest(path=path.name):
                source = path.read_text()
                self.assertIn("docker create --gpus all --ipc=host", source)
                self.assertIn('docker start "$TRAIN_CONTAINER"', source)
                self.assertIn('docker logs -f "$TRAIN_CONTAINER"', source)
                self.assertIn("ai.tinylora.launcher=", source)
                self.assertNotIn("docker run --rm --gpus all", source)
                self.assertLess(
                    source.index("docker create --gpus all --ipc=host"),
                    source.index('docker start "$TRAIN_CONTAINER"'),
                )

    def test_signal_cleanup_is_state_guarded(self) -> None:
        for path in LAUNCHERS:
            with self.subTest(path=path.name):
                source = path.read_text()
                self.assertIn("trap 'handle_signal INT 130' INT", source)
                self.assertIn("trap 'handle_signal TERM 143' TERM", source)
                self.assertIn("trap 'handle_signal HUP 129' HUP", source)
                self.assertIn(
                    '[[ "$training_may_be_running" == 1 && "$state" != stopped ]]',
                    source,
                )
                self.assertIn("Do not start %s until Docker reports", source)
                self.assertIn('docker inspect "$TRAIN_CONTAINER"', source)

    def test_qwen_detached_command_retains_canary_protocol(self) -> None:
        source = LAUNCHERS[1].read_text()
        for expected in (
            "--model Qwen/Qwen2.5-7B-Instruct",
            "--rollout-sync lora",
            "--dataset-split 'train[:-512]'",
            "--prompts-per-step 16",
            "--generations 4",
            "--modules-per-group 16",
            "--parameter-dtype bfloat16",
            '--factor-cache-sha256 "$factor_sha"',
            "--loss-reduction token_mean",
            "--prompt-style verl",
            "--reward-mode strict",
            'if [[ "$PREFLIGHT_ONLY" == 1 ]]',
            "assert_no_competing_gpu_owners || exit 1",
            'OPTIONAL_TRAIN_ARGS+=(--target-layer-indices "$TARGET_LAYER_INDICES")',
            'OPTIONAL_TRAIN_ARGS+=(--profile-memory --profile-sample-interval',
        ):
            with self.subTest(argument=expected):
                self.assertIn(expected, source)

    def test_sigint_does_not_stop_training_or_restore_server(self) -> None:
        fake_docker = r"""#!/usr/bin/env bash
set -eu
printf '%s' "$1" >> "$FAKE_DOCKER_STATE/commands"
shift
for argument in "$@"; do
  printf ' %s' "$argument" >> "$FAKE_DOCKER_STATE/commands"
done
printf '\n' >> "$FAKE_DOCKER_STATE/commands"

command_name="$(tail -n 1 "$FAKE_DOCKER_STATE/commands" | awk '{print $1}')"
case "$command_name" in
  inspect)
    name="${!#}"
    state_file="$FAKE_DOCKER_STATE/$name"
    [[ -f "$state_file" ]] || exit 1
    state="$(<"$state_file")"
    if [[ "$*" == *'DeviceRequests'* ]]; then
      printf '[{"Driver":"nvidia","Capabilities":[["gpu"]]}]\n'
    elif [[ "$*" == *'.Id'* ]]; then
      printf '%s-id\n' "$name"
    elif [[ "$*" == *'.State.ExitCode'* ]]; then
      printf '0\n'
    elif [[ "$*" == *'if .State.Running'* ]]; then
      printf '%s\n' "$state"
    elif [[ "$*" == *'.State.Running'* ]]; then
      [[ "$state" == running ]] && printf 'true\n' || printf 'false\n'
    else
      printf '{}\n'
    fi
    ;;
  ps)
    for state_file in "$FAKE_DOCKER_STATE"/*; do
      name="$(basename "$state_file")"
      case "$name" in commands|training-ready) continue ;; esac
      [[ "$(<"$state_file")" == running ]] && printf '%s\n' "$name"
    done
    ;;
  create)
    name=
    while (($#)); do
      if [[ "$1" == --name ]]; then
        name="$2"
        break
      fi
      shift
    done
    [[ -n "$name" ]]
    printf 'stopped\n' > "$FAKE_DOCKER_STATE/$name"
    printf 'fake-container-id\n'
    ;;
  start)
    name="${!#}"
    printf 'running\n' > "$FAKE_DOCKER_STATE/$name"
    if [[ "$name" != nemotron35_lightning_vllm ]]; then
      : > "$FAKE_DOCKER_STATE/training-ready"
    fi
    ;;
  stop)
    name="${!#}"
    printf 'stopped\n' > "$FAKE_DOCKER_STATE/$name"
    if [[ "${FAKE_STOP_FAIL:-0}" == 1 ]]; then
      exit 44
    fi
    ;;
  logs)
    if [[ "${FAKE_LOGS_MODE:-follow}" == complete ]]; then
      name="${!#}"
      printf 'stopped\n' > "$FAKE_DOCKER_STATE/$name"
      exit 0
    fi
    trap 'exit 130' INT TERM HUP
    while :; do sleep 0.05; done
    ;;
  run)
    printf '%s  /workspace/factor-cache\n' "$FAKE_FACTOR_SHA"
    ;;
  rm)
    name="${!#}"
    rm -f "$FAKE_DOCKER_STATE/$name"
    ;;
  *)
    printf 'unexpected fake docker command: %s\n' "$command_name" >&2
    exit 2
    ;;
esac
"""
        for launcher in LAUNCHERS:
            with self.subTest(launcher=launcher.name), tempfile.TemporaryDirectory() as tmp:
                temporary = Path(tmp)
                fake_bin = temporary / "bin"
                fake_bin.mkdir()
                docker = fake_bin / "docker"
                docker.write_text(fake_docker)
                docker.chmod(0o755)
                curl = fake_bin / "curl"
                curl.write_text("#!/usr/bin/env bash\nexit 0\n")
                curl.chmod(0o755)
                nvidia_smi = fake_bin / "nvidia-smi"
                nvidia_smi.write_text("#!/usr/bin/env bash\nexit 0\n")
                nvidia_smi.chmod(0o755)
                state = temporary / "state"
                state.mkdir()
                (state / "commands").touch()
                (state / "nemotron35_lightning_vllm").write_text("running\n")

                environment = os.environ.copy()
                environment.update(
                    {
                        "PATH": f"{fake_bin}:{environment['PATH']}",
                        "FAKE_DOCKER_STATE": str(state),
                        "FAKE_FACTOR_SHA": "test-factor-sha",
                        "EXPECTED_FACTOR_SHA256": "test-factor-sha",
                        "REPO": str(temporary),
                        "RUN_SLUG": "signal-test",
                    }
                )
                if launcher == LAUNCHERS[1]:
                    preflight_environment = environment.copy()
                    preflight_environment["PREFLIGHT_ONLY"] = "1"
                    preflight = subprocess.run(
                        ["bash", str(launcher)],
                        cwd=temporary,
                        env=preflight_environment,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    self.assertEqual(
                        preflight.returncode,
                        0,
                        (preflight.stdout, preflight.stderr),
                    )
                    self.assertIn(
                        "Preflight passed; GPU training was not launched.",
                        preflight.stdout,
                    )
                    command_names = [
                        line.split()[0]
                        for line in (state / "commands").read_text().splitlines()
                        if line
                    ]
                    for mutating_command in ("create", "start", "stop", "rm"):
                        self.assertNotIn(mutating_command, command_names)
                    (state / "commands").write_text("")
                process = subprocess.Popen(
                    ["bash", str(launcher)],
                    cwd=temporary,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
                ready = state / "training-ready"
                deadline = time.monotonic() + 5
                while not ready.exists() and process.poll() is None:
                    if time.monotonic() >= deadline:
                        self.fail("launcher did not reach detached training state")
                    time.sleep(0.01)

                # Target only the supervisor PID.  This is stricter than a
                # terminal process-group interrupt: its log follower must be
                # reaped without ever signalling the detached container.
                os.kill(process.pid, signal.SIGINT)
                stdout, stderr = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 130, (stdout, stderr))
                self.assertEqual(
                    (state / "nemotron35_lightning_vllm").read_text().strip(),
                    "stopped",
                )
                training_name = (
                    "nemotron35-tinylora-sweep"
                    if "nemotron35" in launcher.name
                    else "spark-tinylora-rl"
                )
                self.assertEqual((state / training_name).read_text().strip(), "running")
                self.assertNotIn(
                    "start nemotron35_lightning_vllm",
                    (state / "commands").read_text(),
                )
                self.assertIn("was left untouched", stderr)

                # Simulate the documented manual recovery, then verify that a
                # fresh normally completing run restores inference and cleans
                # up its stopped training container.
                (state / training_name).unlink()
                (state / "training-ready").unlink()
                (state / "nemotron35_lightning_vllm").write_text("running\n")
                (state / "commands").write_text("")
                environment["FAKE_LOGS_MODE"] = "complete"
                completed = subprocess.run(
                    ["bash", str(launcher)],
                    cwd=temporary,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    (completed.stdout, completed.stderr),
                )
                self.assertIn("exited with status 0", completed.stdout)
                self.assertIn("health check passed", completed.stdout)
                self.assertEqual(
                    (state / "nemotron35_lightning_vllm").read_text().strip(),
                    "running",
                )
                self.assertFalse((state / training_name).exists())

                # A failure while stopping inference happens after `docker
                # create` but before training starts.  That stopped container
                # is launcher-owned and must not block the next invocation.
                (state / "commands").write_text("")
                environment["FAKE_STOP_FAIL"] = "1"
                stopped_early = subprocess.run(
                    ["bash", str(launcher)],
                    cwd=temporary,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.assertEqual(
                    stopped_early.returncode,
                    44,
                    (stopped_early.stdout, stopped_early.stderr),
                )
                self.assertEqual(
                    (state / "nemotron35_lightning_vllm").read_text().strip(),
                    "running",
                )
                self.assertFalse((state / training_name).exists())
                self.assertIn(
                    f"rm {training_name}",
                    (state / "commands").read_text(),
                )


if __name__ == "__main__":
    unittest.main()
