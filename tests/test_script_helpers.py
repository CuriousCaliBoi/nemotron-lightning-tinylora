import unittest

import torch

from analyze_adapter_spectra import resolve_model_dtype
from evaluate_gsm8k_adapters import add_holm_adjustment, compare_correctness
from train_nemotron35_tinylora_sweep import (
    FirstStepReplayRollout,
    supported_max_lora_rank,
    trajectory_design_sha256,
    trajectory_step_sha256,
)


class FakeRolloutBackend:
    def __init__(self) -> None:
        self.generate_calls = 0
        self.sync_calls = 0

    def generate(self, prompt_ids, **kwargs):
        self.generate_calls += 1
        return [(0, [101, self.generate_calls], [-0.1, -0.2], "#### 7")]

    def sync_tinylora(self, model):
        self.sync_calls += 1
        return 24


class ScriptHelpersTest(unittest.TestCase):
    def test_first_step_rollout_is_exactly_replayed_and_mutation_isolated(self) -> None:
        backend = FakeRolloutBackend()
        rollout = FirstStepReplayRollout(backend)
        request = {
            "num_generations": 2,
            "max_tokens": 32,
            "temperature": 1.2,
            "top_p": 1.0,
            "seed": 43,
        }

        rollout.begin_candidate("reference")
        first = rollout.generate([[1, 2, 3]], **request)
        first[0][1].append(999)
        rollout.begin_candidate("replay")
        replayed = rollout.generate([[1, 2, 3]], **request)

        self.assertEqual(backend.generate_calls, 1)
        self.assertEqual(replayed, [(0, [101, 1], [-0.1, -0.2], "#### 7")])
        self.assertTrue(rollout.first_step_replayed)
        self.assertEqual(len(rollout.first_step_request_sha256), 64)

        second_step = rollout.generate([[4, 5]], **{**request, "seed": 44})
        self.assertEqual(backend.generate_calls, 2)
        self.assertEqual(second_step[0][1], [101, 2])

    def test_first_step_replay_rejects_request_mismatch_and_delegates_sync(self) -> None:
        backend = FakeRolloutBackend()
        rollout = FirstStepReplayRollout(backend)
        request = {
            "num_generations": 2,
            "max_tokens": 32,
            "temperature": 1.2,
            "top_p": 1.0,
            "seed": 43,
        }
        rollout.begin_candidate("reference")
        rollout.generate([[1, 2, 3]], **request)
        rollout.begin_candidate("mismatch")
        with self.assertRaises(RuntimeError):
            rollout.generate([[1, 2, 4]], **request)
        self.assertEqual(rollout.sync_tinylora(object()), 24)
        self.assertEqual(backend.sync_calls, 1)

    def test_vllm_rank_is_rounded_up_to_supported_bucket(self) -> None:
        self.assertEqual(supported_max_lora_rank(1), 1)
        self.assertEqual(supported_max_lora_rank(2), 8)
        self.assertEqual(supported_max_lora_rank(8), 8)
        self.assertEqual(supported_max_lora_rank(9), 16)
        with self.assertRaises(ValueError):
            supported_max_lora_rank(513)

    def test_model_storage_dtype_resolution(self) -> None:
        self.assertIs(resolve_model_dtype("float32"), torch.float32)
        self.assertIs(resolve_model_dtype("bfloat16"), torch.bfloat16)
        with self.assertRaises(ValueError):
            resolve_model_dtype("float64")

    def test_explicit_zero_adapter_baseline_comparison(self) -> None:
        reference = {
            "flexible": [True, False, True, False],
            "strict": [True, False, False, False],
        }
        candidate = {
            "flexible": [True, True, False, False],
            "strict": [True, True, False, True],
        }
        result = compare_correctness(
            reference,
            candidate,
            score_mode="strict",
            bootstrap_samples=0,
        )
        self.assertEqual(result["accuracy_delta"], 0.5)
        self.assertEqual(result["wrong_to_right"], 2)
        self.assertEqual(result["right_to_wrong"], 0)
        self.assertEqual(
            result["paired_by_score_mode"]["flexible"]["accuracy_delta"],
            0.0,
        )

    def test_holm_adjustment_is_monotone_and_bounded(self) -> None:
        comparisons = {
            "a": {"mcnemar_exact_p": 0.01},
            "b": {"mcnemar_exact_p": 0.03},
            "c": {"mcnemar_exact_p": 0.8},
        }
        add_holm_adjustment(comparisons)
        self.assertEqual(comparisons["a"]["mcnemar_holm_p"], 0.03)
        self.assertEqual(comparisons["b"]["mcnemar_holm_p"], 0.06)
        self.assertEqual(comparisons["c"]["mcnemar_holm_p"], 0.8)
        self.assertEqual(comparisons["a"]["multiple_comparison_family_size"], 3)

    def test_sweep_design_hash_ignores_stochastic_outputs(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.jsonl"
            second = Path(directory) / "second.jsonl"
            common = {
                "step": 1,
                "group_id": 0,
                "question": "q",
                "gold_answer": "#### 1",
            }
            first.write_text(json.dumps({**common, "completion": "a", "reward": 0}) + "\n")
            second.write_text(json.dumps({**common, "completion": "b", "reward": 1}) + "\n")
            self.assertEqual(
                trajectory_design_sha256(first),
                trajectory_design_sha256(second),
            )
            self.assertNotEqual(trajectory_step_sha256(first), trajectory_step_sha256(second))


if __name__ == "__main__":
    unittest.main()
