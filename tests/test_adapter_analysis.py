import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from tinylora_rl.adapter_analysis import (
    LowRankUpdate,
    apply_updates_,
    export_peft_adapter,
    find_intruder_dimensions,
    load_adapter_updates,
)


class NestedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.proj = nn.Linear(3, 4, bias=False)


class AdapterAnalysisTest(unittest.TestCase):
    def test_identical_weights_have_no_intruders(self) -> None:
        torch.manual_seed(1)
        weight = torch.randn(7, 5)
        result = find_intruder_dimensions(weight, weight.clone(), top_k=5, threshold=0.5)
        self.assertEqual(result.count, 0)
        self.assertEqual(result.examined, 5)
        self.assertTrue(all(value > 0.999 for value in result.max_similarities))

    def test_known_rotated_leading_vector_is_an_intruder(self) -> None:
        base = torch.diag(torch.tensor([5.0, 4.0, 3.0, 2.0]))
        direction = torch.full((4,), 0.5)
        tuned = 20.0 * torch.outer(direction, direction) + base
        result = find_intruder_dimensions(base, tuned, top_k=1, threshold=0.6)
        self.assertEqual(result.count, 1)
        self.assertLess(result.max_similarities[0], 0.6)

    def test_load_standard_peft_and_apply(self) -> None:
        left = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 10
        right = torch.arange(6, dtype=torch.float32).reshape(2, 3) / 10
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            save_file(
                {
                    "base_model.model.model.proj.lora_A.weight": right,
                    "base_model.model.model.proj.lora_B.weight": left,
                },
                str(path / "adapter_model.safetensors"),
            )
            (path / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "r": 2,
                        "lora_alpha": 4,
                        "use_rslora": False,
                        "use_dora": False,
                    }
                )
            )
            updates = load_adapter_updates(path)
            self.assertEqual(len(updates), 1)
            self.assertEqual(updates[0].name, "model.proj")
            self.assertEqual(updates[0].scaling, 2.0)
            model = NestedModel()
            before = model.model.proj.weight.detach().clone()
            self.assertEqual(apply_updates_(model, updates), 1)
            torch.testing.assert_close(model.model.proj.weight, before + 2.0 * left @ right)

    def test_load_tinylora_folds_middle_matrix(self) -> None:
        left = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 10
        right = torch.arange(6, dtype=torch.float32).reshape(2, 3) / 10
        projection = torch.stack((torch.eye(2), torch.ones(2, 2)))
        bank = torch.tensor([[0.25, -0.5]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            save_file(
                {
                    "bank.v": bank,
                    "layers.0.left": left,
                    "layers.0.right": right,
                    "layers.0.projection": projection,
                },
                str(path / "adapter.safetensors"),
            )
            (path / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "config": {"rank": 2},
                        "modules": [
                            {"name": "model.proj", "group_id": 0, "scaling": 1.5}
                        ],
                    }
                )
            )
            update = load_adapter_updates(path)[0]
            middle = torch.einsum("u,urs->rs", bank[0], projection)
            torch.testing.assert_close(update.materialize(), 1.5 * left @ middle @ right)

    def test_export_round_trip_preserves_delta(self) -> None:
        update = LowRankUpdate(
            name="model.proj",
            left=torch.randn(4, 2),
            right=torch.randn(2, 3),
            scaling=0.75,
            source="test",
        )
        with tempfile.TemporaryDirectory() as directory:
            export_peft_adapter([update], directory, base_model_name="test/base")
            restored = load_adapter_updates(directory)[0]
            torch.testing.assert_close(restored.materialize(), update.materialize())

    def test_export_can_store_bfloat16_factors(self) -> None:
        update = LowRankUpdate(
            name="model.proj",
            left=torch.randn(4, 2),
            right=torch.randn(2, 3),
            scaling=0.0,
            source="test",
        )
        with tempfile.TemporaryDirectory() as directory:
            export_peft_adapter(
                [update],
                directory,
                base_model_name="test/base",
                tensor_dtype=torch.bfloat16,
            )
            tensors = load_file(str(Path(directory) / "adapter_model.safetensors"))
            self.assertEqual({tensor.dtype for tensor in tensors.values()}, {torch.bfloat16})
            b_tensor = next(
                tensor for name, tensor in tensors.items() if ".lora_B." in name
            )
            self.assertEqual(torch.count_nonzero(b_tensor).item(), 0)


if __name__ == "__main__":
    unittest.main()
