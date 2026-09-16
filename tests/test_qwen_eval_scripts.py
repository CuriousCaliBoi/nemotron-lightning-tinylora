from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "run_qwen25_7b_tinylora_eval.sh"
PROMOTE = ROOT / "run_qwen25_7b_tinylora_promote.sh"


def embedded_python(source: str) -> list[str]:
    blocks = []
    lines = source.splitlines()
    index = 0
    while index < len(lines):
        if "<<'PY'" not in lines[index]:
            index += 1
            continue
        start = index + 1
        index = start
        while index < len(lines) and lines[index] != "PY":
            index += 1
        if index == len(lines):
            raise AssertionError("unterminated embedded Python heredoc")
        blocks.append("\n".join(lines[start:index]) + "\n")
        index += 1
    return blocks


class QwenEvaluationScriptsTest(unittest.TestCase):
    def test_shell_and_embedded_python_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(EVAL), str(PROMOTE)], cwd=ROOT, check=True)
        for path in (EVAL, PROMOTE):
            blocks = embedded_python(path.read_text())
            self.assertTrue(blocks, path.name)
            for number, block in enumerate(blocks, 1):
                with self.subTest(path=path.name, block=number):
                    compile(block, f"{path.name}:heredoc-{number}", "exec")

    def test_eval_is_full_test_zero_control_protocol(self) -> None:
        source = EVAL.read_text()
        for expected in (
            'RUN_SLUG="${RUN_SLUG:-}"',
            'PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"',
            'steps=(16 32 48 64)',
            '--adapter "zero=$zero_container"',
            '--adapter "zero_repeat=$zero_container"',
            '--comparison-baseline zero',
            '--holm-label "step$step"',
            '--split test',
            '--samples "$SAMPLES"',
            '--require-exact-samples',
            '--prompt-style verl',
            '--score-mode strict',
            '--temperature 0',
            '--bootstrap-samples 10000',
            '--include-text',
            'assert_no_competing_gpu_owners "$SERVER_CONTAINER"',
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, source)
        self.assertNotIn("--overwrite", source)

    def test_step64_final_dedup_and_canary_labels(self) -> None:
        eval_source = EVAL.read_text()
        promote_source = PROMOTE.read_text()
        self.assertIn(
            'fail(f"checkpoint-64 and final_adapter differ at tensor {name}")',
            eval_source,
        )
        self.assertIn(
            'tinylora13-verl-strict-canary-lr1e-4-s42',
            promote_source,
        )
        self.assertIn('step_labels+=("$label_prefix/step$step")', promote_source)
        self.assertIn(
            "only checkpoint-64 was registered",
            promote_source,
        )
        self.assertNotIn('--label "$label_prefix/final', promote_source)

    def test_promotion_dry_run_is_content_addressed(self) -> None:
        source = PROMOTE.read_text()
        for expected in (
            'register_all "$temporary_registry"',
            'check_registry_state 0',
            'if [[ "$PREFLIGHT_ONLY" == 1 ]]',
            'register_all "$REGISTRY"',
            '--peft "$artifact_root/step$step"',
            '--candidate "step$step"',
            '--candidate zero',
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, source)
        self.assertNotIn("--replace-label", source)
        self.assertNotIn("--replace ", source)


if __name__ == "__main__":
    unittest.main()
