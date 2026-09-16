import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

from tinylora_rl.adapters import (
    TinyLoRAConfig,
    TinyLoRALinear,
    apply_tinylora,
    export_vllm_lora,
    iter_tinylora_layers,
    load_tinylora,
    save_tinylora,
    trainable_parameter_count,
)
from tinylora_rl.rollout import VLLMRolloutBackend


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 6, bias=True)
        self.block = nn.ModuleDict(
            {
                "v_proj": nn.Linear(6, 5, bias=False),
                "down_proj": nn.Linear(5, 4, bias=False),
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block["down_proj"](self.block["v_proj"](self.q_proj(x)))


class LayeredToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.ModuleDict({"q_proj": nn.Linear(4, 4, bias=False)}) for _ in range(3)]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            inputs = layer["q_proj"](inputs)
        return inputs


class TinyLoRATest(unittest.TestCase):
    def test_vllm_lora_targets_are_forwarded_to_engine(self) -> None:
        captured = {}

        class FakeLLM:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

        with patch.dict(sys.modules, {"vllm": SimpleNamespace(LLM=FakeLLM)}):
            VLLMRolloutBackend(
                "test/model",
                gpu_memory_utilization=0.2,
                max_model_len=32,
                seed=7,
                enable_lora=True,
                max_lora_rank=8,
                lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            )
        self.assertEqual(
            captured["lora_target_modules"],
            ["q_proj", "k_proj", "v_proj", "o_proj"],
        )

    def test_vllm_packed_projection_name_mapping(self) -> None:
        map_name = VLLMRolloutBackend._vllm_destination_name
        self.assertEqual(
            map_name("model.layers.0.self_attn.q_proj.weight"),
            "model.layers.0.self_attn.qkv_proj.weight",
        )
        self.assertEqual(
            map_name("model.layers.0.mlp.up_proj.weight"),
            "model.layers.0.mlp.gate_up_proj.weight",
        )
        self.assertEqual(
            map_name("model.layers.0.self_attn.o_proj.weight"),
            "model.layers.0.self_attn.o_proj.weight",
        )

    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = ToyModel()
        self.inputs = torch.randn(3, 8)
        self.base_output = self.model(self.inputs).detach()
        self.config = TinyLoRAConfig(
            rank=2,
            projection_dim=2,
            modules_per_group=2,
            projection_seed=11,
        )
        apply_tinylora(self.model, self.config)

    def test_zero_initialization_preserves_base_policy(self) -> None:
        torch.testing.assert_close(self.model(self.inputs), self.base_output)

    def test_only_parameter_bank_is_trainable_and_tied(self) -> None:
        layers = list(iter_tinylora_layers(self.model))
        self.assertEqual(len(layers), 3)
        self.assertEqual(self.model.tinylora_bank.v.shape, (2, 2))
        self.assertEqual(trainable_parameter_count(self.model), 4)
        self.assertEqual([layer.group_id for _, layer in layers], [0, 0, 1])
        names = [name for name, parameter in self.model.named_parameters() if parameter.requires_grad]
        self.assertEqual(names, ["tinylora_bank.v"])

    def test_explicit_group_count_balances_all_targets(self) -> None:
        model = ToyModel()
        config = TinyLoRAConfig(
            rank=2,
            projection_dim=1,
            num_groups=2,
        )
        apply_tinylora(model, config)
        layers = [layer for _, layer in iter_tinylora_layers(model)]
        self.assertEqual(model.tinylora_bank.v.shape, (2, 1))
        self.assertEqual([layer.group_id for layer in layers], [0, 0, 1])
        self.assertEqual(trainable_parameter_count(model), 2)

    def test_layer_index_filter_restricts_backward_targets(self) -> None:
        model = LayeredToyModel()
        config = TinyLoRAConfig(
            rank=2,
            projection_dim=1,
            target_modules=("q_proj",),
            target_layer_indices=(1, 2),
            num_groups=2,
        )
        apply_tinylora(model, config)
        layers = [layer.original_name for _, layer in iter_tinylora_layers(model)]
        self.assertEqual(layers, ["layers.1.q_proj", "layers.2.q_proj"])
        self.assertIsInstance(model.layers[0]["q_proj"], nn.Linear)
        self.assertEqual(trainable_parameter_count(model), 2)

    def test_factor_cache_is_extended_for_new_layer_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "factors.safetensors"
            partial = LayeredToyModel()
            apply_tinylora(
                partial,
                TinyLoRAConfig(
                    rank=2,
                    target_modules=("q_proj",),
                    target_layer_indices=(2,),
                ),
                factor_cache=cache,
            )
            self.assertEqual(len(load_file(str(cache))), 2)

            complete = LayeredToyModel()
            apply_tinylora(
                complete,
                TinyLoRAConfig(rank=2, target_modules=("q_proj",)),
                factor_cache=cache,
            )
            self.assertEqual(len(load_file(str(cache))), 6)

    def test_factorized_forward_equals_materialized_weight(self) -> None:
        with torch.no_grad():
            self.model.tinylora_bank.v.normal_()
        for _, layer in iter_tinylora_layers(self.model):
            x = torch.randn(2, layer.base_layer.in_features)
            expected = F.linear(x, layer.effective_weight(), layer.base_layer.bias)
            torch.testing.assert_close(layer(x), expected, rtol=1e-5, atol=1e-5)

    def test_gradients_reach_only_tiny_vector(self) -> None:
        loss = self.model(self.inputs).square().mean()
        loss.backward()
        self.assertIsNotNone(self.model.tinylora_bank.v.grad)
        self.assertGreater(float(self.model.tinylora_bank.v.grad.norm()), 0.0)
        for _, layer in iter_tinylora_layers(self.model):
            self.assertIsNone(layer.base_layer.weight.grad)

    def test_small_checkpoint_excludes_base_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            save_tinylora(self.model, directory)
            self.assertTrue((Path(directory) / "adapter.safetensors").exists())
            config = (Path(directory) / "adapter_config.json").read_text()
            self.assertNotIn("base_layer.weight", config)

    def test_saved_adapter_reloads_exactly(self) -> None:
        base = ToyModel()
        base_state = {name: value.clone() for name, value in base.state_dict().items()}
        apply_tinylora(base, self.config)
        with torch.no_grad():
            base.tinylora_bank.v.normal_()
        expected = base(self.inputs)
        with tempfile.TemporaryDirectory() as directory:
            save_tinylora(base, directory)
            restored = ToyModel()
            restored.load_state_dict(base_state)
            load_tinylora(restored, directory)
            torch.testing.assert_close(restored(self.inputs), expected)
            self.assertEqual(trainable_parameter_count(restored), 4)

    def test_vllm_lora_export_exactly_materializes_delta(self) -> None:
        with torch.no_grad():
            self.model.tinylora_bank.v.normal_()
        with tempfile.TemporaryDirectory() as directory:
            output = export_vllm_lora(
                self.model,
                directory,
                base_model_name="test/toy",
            )
            tensors = load_file(str(output / "adapter_model.safetensors"))
            for _, layer in iter_tinylora_layers(self.model):
                prefix = f"base_model.model.{layer.original_name}"
                a = tensors[f"{prefix}.lora_A.weight"]
                b = tensors[f"{prefix}.lora_B.weight"]
                expected = layer.delta_weight().cpu()
                torch.testing.assert_close(b @ a, expected, rtol=1e-5, atol=1e-5)
            config = (output / "adapter_config.json").read_text()
            self.assertIn('"lora_alpha": 2', config)
            self.assertIn('"r": 2', config)


if __name__ == "__main__":
    unittest.main()
