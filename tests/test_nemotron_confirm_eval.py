from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from evaluate_gsm8k_adapters import (
    EVALUATOR_PROGRAM,
    disjoint_evaluation_record,
    evaluator_source_provenance,
    repeated_contrast_statistics,
)


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "run_nemotron35_tinylora_eval.sh"
CONFIRM = ROOT / "run_nemotron35_tinylora_confirm_eval.sh"


class NemotronConfirmEvaluationTest(unittest.TestCase):
    def test_shell_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(EVAL), str(CONFIRM)], cwd=ROOT, check=True)

    def test_wrapper_pins_the_untouched_confirmatory_protocol(self) -> None:
        source = CONFIRM.read_text()
        for expected in (
            "EVAL_PROTOCOL=confirmatory-untouched384-after-screen-v1",
            "EVAL_CONTAINER=nemotron35-tinylora-confirm-eval",
            "OUTPUT_ROOT_REL=outputs/nemotron35-tinylora-lr-scale-replay-canary-20260916",
            "CANDIDATES=lr1e-5_s1,lr5e-5_s1,lr1e-4_s1,lr1e-4_s0p5",
            "SELECTED_CANDIDATE=lr1e-4_s0p5",
            "VLLM_ENABLE_V1_MULTIPROCESSING=0",
            "VLLM_BATCH_INVARIANT=0",
            "'TRAIN_SPLIT=train[:-512]'",
            "'EVAL_SPLIT=train[-384:]'",
            "SAMPLES=384",
            "MAX_TOKENS=1024",
            "MAX_MODEL_LENGTH=1280",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, source)

    def test_shared_runner_enforces_abba_contrast_and_provenance(self) -> None:
        source = EVAL.read_text()
        for expected in (
            "screen-consumed128-max1024.json",
            'screen_sha256=$(sha256sum "$screen_host"',
            '"trained_b": (trained_path, fingerprint(trained_path))',
            "current selected adapter hashes do not match the screen",
            'disjoint_args=(--disjoint-evaluation "selection_screen=$screen_container")',
            '--adapter "zero_a=$zero_container"',
            '--adapter "trained_a=$container_root/$SELECTED_CANDIDATE/peft_adapter"',
            '--adapter "trained_b=$container_root/$SELECTED_CANDIDATE/peft_adapter"',
            '--adapter "zero_b=$zero_container"',
            "comparison_baseline=zero_a",
            "--no-holm",
            "--repeated-contrast zero_a,trained_a,trained_b,zero_b",
            "--selection-rationale \"$SELECTION_RATIONALE\"",
            "execution_args=(--reuse-lora-slot --enforce-eager)",
            "--require-environment VLLM_ENABLE_V1_MULTIPROCESSING=0",
            "--require-environment VLLM_BATCH_INVARIANT=0",
            '--require-exact-samples',
            '--lora-dtype bfloat16',
            '--prompt-style concise',
            '--score-mode strict',
            '--temperature 0',
            'confirmatory-untouched384-lr1e-4-s0p5-abba-bf16.json',
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, source)
        self.assertNotIn("--overwrite", source)

        zero_a = source.index('--adapter "zero_a=$zero_container"')
        trained_a = source.index('--adapter "trained_a=$container_root/')
        trained_b = source.index('--adapter "trained_b=$container_root/')
        zero_b = source.index('--adapter "zero_b=$zero_container"')
        self.assertLess(zero_a, trained_a)
        self.assertLess(trained_a, trained_b)
        self.assertLess(trained_b, zero_b)

    def test_disjoint_reference_is_hashed_and_overlap_is_rejected(self) -> None:
        prior_question = hashlib.sha256(b"screen question").hexdigest()
        new_question = hashlib.sha256(b"confirmation question").hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "screen.json"
            path.write_text(
                json.dumps(
                    {
                        "question_sha256": [prior_question],
                        "split": "train[-512:-384]",
                        "selected_dataset_rows": 1,
                        "dataset_revision": "revision",
                        "model": "model",
                        "requested_revision": "model-revision",
                    }
                )
            )
            record = disjoint_evaluation_record(path, [new_question])
            self.assertEqual(record["question_overlap_count"], 0)
            self.assertEqual(record["question_count"], 1)
            self.assertEqual(record["artifact"]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

            with self.assertRaisesRegex(ValueError, "1 shared hashes"):
                disjoint_evaluation_record(path, [prior_question])

    def test_evaluator_provenance_is_content_addressed_not_path_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.py"
            second = Path(directory) / "renamed.py"
            first.write_bytes(b"print('same evaluator bytes')\n")
            second.write_bytes(first.read_bytes())

            first_record = evaluator_source_provenance(first, ["--samples", "384"])
            second_record = evaluator_source_provenance(second, ["--samples", "384"])

            self.assertEqual(first_record, second_record)
            self.assertEqual(first_record["schema_version"], 1)
            self.assertEqual(first_record["program"], EVALUATOR_PROGRAM)
            self.assertEqual(first_record["source_size_bytes"], len(first.read_bytes()))
            self.assertEqual(first_record["arguments"], ["--samples", "384"])
            self.assertNotIn("path", first_record)

    def test_evaluator_source_is_captured_at_main_entry(self) -> None:
        source = (ROOT / "evaluate_gsm8k_adapters.py").read_text()
        main = source.index("def main() -> None:")
        capture = source.index("evaluator_source_provenance(", main)
        parse = source.index("args = parse_args()", main)
        model = source.index("engine = LLM(", main)
        result = source.index('"evaluator_provenance": evaluator_provenance', main)
        self.assertLess(capture, parse)
        self.assertLess(parse, model)
        self.assertLess(model, result)

    def test_repeated_contrast_clusters_rows_and_applies_strict_gate(self) -> None:
        labels = ("zero_a", "trained_a", "trained_b", "zero_b")
        correctness = {
            "zero_a": {"strict": [False] * 4, "flexible": [False] * 4},
            "trained_a": {"strict": [True] * 4, "flexible": [True] * 4},
            "trained_b": {"strict": [True] * 4, "flexible": [True] * 4},
            "zero_b": {"strict": [False] * 4, "flexible": [False] * 4},
        }

        def summary(prefix: str, values: dict[str, list[bool]]) -> dict[str, object]:
            return {
                "details": [
                    {
                        "index": index,
                        "completion_sha256": f"{prefix}-text-{index}",
                        "completion_token_ids_sha256": f"{prefix}-tokens-{index}",
                        "finish_reason": "stop",
                        "strict_correct": values["strict"][index],
                        "flexible_correct": values["flexible"][index],
                        "strict_prediction": int(values["strict"][index]),
                        "flexible_prediction": int(values["flexible"][index]),
                    }
                    for index in range(4)
                ]
            }

        summaries = {
            "zero_a": summary("zero", correctness["zero_a"]),
            "trained_a": summary("trained", correctness["trained_a"]),
            "trained_b": summary("trained", correctness["trained_b"]),
            "zero_b": summary("zero", correctness["zero_b"]),
        }
        zero_artifact = {"weights": {"sha256": "zero"}}
        trained_artifact = {"weights": {"sha256": "trained"}}
        artifacts = {
            "zero_a": zero_artifact,
            "trained_a": trained_artifact,
            "trained_b": trained_artifact,
            "zero_b": zero_artifact,
        }
        paths = {
            "zero_a": Path("/zero"),
            "trained_a": Path("/trained"),
            "trained_b": Path("/trained"),
            "zero_b": Path("/zero"),
        }
        result = repeated_contrast_statistics(
            correctness,
            summaries,
            artifacts,
            paths,
            labels,
            bootstrap_samples=100,
            seed=42,
            selection_rationale="screen winner",
        )
        strict = result["by_score_mode"]["strict"]
        self.assertEqual(strict["mean_effect"], 1.0)
        self.assertEqual(strict["row_effects"], [1.0] * 4)
        self.assertEqual(strict["row_cluster_bootstrap_95_ci"], [1.0, 1.0])
        self.assertEqual(
            strict["repeat_specific"]["a_trained_minus_zero"]["accuracy_delta"],
            1.0,
        )
        self.assertTrue(result["primary_efficacy_gate"]["passed"])
        self.assertEqual(result["predeclared_hypotheses"], 1)
        self.assertEqual(
            result["within_arm_repeatability"]["zero"]["exact_text_token_finish"]["count"],
            4,
        )
        self.assertTrue(result["within_arm_identity"]["trained"]["same_artifact_hashes"])

        correctness["trained_b"] = {
            "strict": [False] * 4,
            "flexible": [False] * 4,
        }
        summaries["trained_b"] = summary("trained", correctness["trained_b"])
        one_sided = repeated_contrast_statistics(
            correctness,
            summaries,
            artifacts,
            paths,
            labels,
            bootstrap_samples=100,
            seed=42,
            selection_rationale="screen winner",
        )
        one_sided_strict = one_sided["by_score_mode"]["strict"]
        self.assertEqual(one_sided_strict["mean_effect"], 0.5)
        self.assertEqual(
            one_sided_strict["row_cluster_bootstrap_95_ci"], [0.5, 0.5]
        )
        self.assertFalse(one_sided["primary_efficacy_gate"]["passed"])
        self.assertFalse(
            one_sided["primary_efficacy_gate"]["conditions"][
                "repeat_b_strict_estimate_gt_zero"
            ]
        )


if __name__ == "__main__":
    unittest.main()
