import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from torch import nn

from evaluate_tinylora_retention import (
    evaluate_blocks,
    load_candidate_state,
    temporary_tinylora_state,
    validate_candidate_states,
)
from tinylora_rl.adapters import TinyLoRALinear, TinyLoRAParameterBank


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        bank = TinyLoRAParameterBank(1, 2, torch.float32)
        self.add_module("tinylora_bank", bank)
        base = nn.Linear(3, 3, bias=False)
        self.projection = TinyLoRALinear(
            base,
            left=torch.eye(3, 2),
            right=torch.eye(2, 3),
            projection=torch.ones(2, 2, 2),
            bank=bank,
            group_id=0,
            original_name="projection",
            scaling=1.25,
        )
        with torch.no_grad():
            bank.v.copy_(torch.tensor([[1.0, 2.0]]))


class TemporaryTinyLoRAStateTest(unittest.TestCase):
    def test_swaps_and_restores_bank_and_scaling(self) -> None:
        model = TinyModel()
        original_bank = model.tinylora_bank.v.detach().clone()

        with temporary_tinylora_state(
            model,
            torch.tensor([[3.0, 4.0]]),
            {"projection": 2.5},
        ):
            torch.testing.assert_close(
                model.tinylora_bank.v,
                torch.tensor([[3.0, 4.0]]),
            )
            self.assertEqual(model.projection.scaling, 2.5)

        torch.testing.assert_close(model.tinylora_bank.v, original_bank)
        self.assertEqual(model.projection.scaling, 1.25)

    def test_restores_state_when_body_raises(self) -> None:
        model = TinyModel()
        original_bank = model.tinylora_bank.v.detach().clone()

        with self.assertRaisesRegex(RuntimeError, "planned"):
            with temporary_tinylora_state(
                model,
                torch.tensor([[5.0, 6.0]]),
                {"projection": 3.0},
            ):
                raise RuntimeError("planned failure")

        torch.testing.assert_close(model.tinylora_bank.v, original_bank)
        self.assertEqual(model.projection.scaling, 1.25)

    def test_validation_failure_does_not_mutate_state(self) -> None:
        model = TinyModel()
        original_bank = model.tinylora_bank.v.detach().clone()

        with self.assertRaisesRegex(ValueError, "layer set differs"):
            with temporary_tinylora_state(
                model,
                torch.tensor([[7.0, 8.0]]),
                {"wrong_name": 4.0},
            ):
                self.fail("invalid state should not enter the context")

        torch.testing.assert_close(model.tinylora_bank.v, original_bank)
        self.assertEqual(model.projection.scaling, 1.25)


class CandidateCompatibilityTest(unittest.TestCase):
    @staticmethod
    def write_adapter(
        path: Path,
        *,
        bank_value: float,
        scaling: float,
        factor_value: float = 1.0,
    ) -> None:
        path.mkdir()
        metadata = {
            "config": {
                "rank": 1,
                "projection_dim": 1,
                "parameter_dtype": "float32",
                "scaling": scaling,
            },
            "modules": [
                {"name": "projection", "group_id": 0, "scaling": scaling}
            ],
        }
        (path / "adapter_config.json").write_text(json.dumps(metadata))
        save_file(
            {
                "bank.v": torch.tensor([[bank_value]]),
                "layers.0.left": torch.tensor([[factor_value], [0.0]]),
                "layers.0.right": torch.tensor([[1.0, 0.0]]),
                "layers.0.projection": torch.ones(1, 1, 1),
            },
            str(path / "adapter.safetensors"),
        )

    def test_bank_and_scaling_may_differ_but_factors_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_adapter(root / "one", bank_value=1.0, scaling=1.0)
            self.write_adapter(root / "two", bank_value=2.0, scaling=2.0)
            self.write_adapter(
                root / "different",
                bank_value=3.0,
                scaling=3.0,
                factor_value=2.0,
            )

            first = load_candidate_state(root / "one")
            second = load_candidate_state(root / "two")
            different = load_candidate_state(root / "different")
            self.assertEqual(validate_candidate_states([first, second]), [first, second])
            with self.assertRaisesRegex(ValueError, "does not share"):
                validate_candidate_states([first, different])


class EvaluateBlocksTest(unittest.TestCase):
    def test_counts_shifted_tokens_and_computes_global_nll(self) -> None:
        class UniformModel(nn.Module):
            def forward(self, input_ids: torch.Tensor, **_: object) -> object:
                return SimpleNamespace(
                    logits=torch.zeros(*input_ids.shape, 5, device=input_ids.device)
                )

        result = evaluate_blocks(
            UniformModel(),
            torch.tensor([[0, 1, 2, 3], [4, 3, 2, 1]]),
            batch_size=1,
            device=torch.device("cpu"),
        )
        self.assertEqual(result["tokens"], 6)
        self.assertEqual(result["windows"], 2)
        self.assertAlmostEqual(float(result["nll"]), math.log(5), places=6)
        self.assertEqual(len(result["per_window_nll"]), 2)


if __name__ == "__main__":
    unittest.main()
