import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from tinylora_rl.adapters import TinyLoRAConfig
from tinylora_rl.trainer import TinyLoRAGRPOTrainer, TrainConfig
from train_tinylora_rl import (
    build_run_provenance,
    summarize_quantization_config,
    verify_factor_cache,
)


class FakeConfig:
    def __init__(self, values, *, commit_hash):
        self.values = values
        self._commit_hash = commit_hash
        for key, value in values.items():
            setattr(self, key, value)

    def to_dict(self):
        return self.values


class MinimalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.trainable = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(
            _name_or_path="learner/model",
            _commit_hash="learner-sha",
        )
        self.tinylora_config = TinyLoRAConfig()


class EmptyDataset:
    _fingerprint = "dataset-fingerprint"

    def __len__(self):
        return 0


class TrainerProvenanceTest(unittest.TestCase):
    def test_factor_cache_identity_is_verified_and_recordable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "factors.safetensors"
            path.write_bytes(b"fixed-factor-cache")
            record = verify_factor_cache(path)
            self.assertEqual(record["path"], str(path))
            self.assertEqual(record["bytes"], len(b"fixed-factor-cache"))
            self.assertEqual(len(record["sha256"]), 64)
            self.assertEqual(
                verify_factor_cache(path, expected_sha256=record["sha256"]),
                record,
            )
            with self.assertRaises(RuntimeError):
                verify_factor_cache(path, expected_sha256="0" * 64)

    def test_large_modelopt_config_is_compact_but_identifiable(self) -> None:
        quantization = {
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "producer": {"name": "modelopt", "version": "0.44.0rc5"},
            "config_groups": {
                "group_0": {
                    "weights": {"num_bits": 4, "type": "float", "group_size": 16},
                    "targets": [f"layers.{index}.weight" for index in range(2_000)],
                }
            },
            "quantized_layers": {
                "layers.0": {"quant_algo": "W4A16_NVFP4"},
                "layers.1": {"quant_algo": "W4A16_NVFP4"},
                "layers.2": {"quant_algo": "FP8"},
            },
            "ignore": ["embed_tokens"],
        }
        summary = summarize_quantization_config(quantization)
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertTrue(summary["configuration_omitted"])
        self.assertEqual(summary["quant_method"], "modelopt")
        self.assertEqual(summary["config_groups"]["group_0"]["target_count"], 2_000)
        self.assertEqual(
            summary["quantized_layer_algorithms"],
            {"FP8": 1, "W4A16_NVFP4": 2},
        )
        self.assertEqual(len(summary["sha256"]), 64)
        self.assertNotIn("targets", json.dumps(summary))

    def test_two_policy_provenance_records_bf16_learner_and_nvfp4_actor(self) -> None:
        learner_config = FakeConfig(
            {
                "architectures": ["NemotronHForCausalLM"],
                "model_type": "nemotron_h",
                "dtype": "bfloat16",
                "num_hidden_layers": 52,
            },
            commit_hash="learner-sha",
        )
        actor_config = FakeConfig(
            {
                "architectures": ["NemotronHForCausalLM"],
                "model_type": "nemotron_h",
                "dtype": "bfloat16",
                "quantization_config": {
                    "quant_method": "modelopt",
                    "quant_algo": "W4A16_NVFP4",
                },
            },
            commit_hash="actor-sha",
        )
        learner = SimpleNamespace(config=learner_config, dtype=torch.bfloat16)
        engine_model_config = SimpleNamespace(
            hf_config=actor_config,
            dtype=torch.bfloat16,
            quantization="modelopt",
            max_model_len=768,
            revision=None,
        )
        rollout = SimpleNamespace(llm=SimpleNamespace(model_config=engine_model_config))
        args = argparse.Namespace(
            model="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
            rollout_model="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
            rollout_sync="lora",
            trust_remote_code=True,
            gradient_checkpointing=True,
            vllm_gpu_memory_utilization=0.23,
            max_model_length=768,
            vllm_kv_cache_dtype="fp8",
            vllm_kv_cache_memory_bytes=2_147_483_648,
            vllm_moe_backend="marlin",
            vllm_mamba_backend="flashinfer",
            vllm_mamba_cache_mode="align",
            max_lora_rank=8,
            target_modules="q_proj,k_proj,v_proj,o_proj",
        )

        provenance = build_run_provenance(args, learner, rollout)

        self.assertEqual(provenance["learner"]["resolved_revision"], "learner-sha")
        self.assertEqual(provenance["learner"]["load"]["dtype"], "bfloat16")
        self.assertEqual(provenance["rollout"]["resolved_revision"], "actor-sha")
        self.assertEqual(provenance["rollout"]["sync_mode"], "lora")
        self.assertEqual(
            provenance["rollout"]["model_config"]["quantization_config"]
            ["configuration"]["quant_algo"],
            "W4A16_NVFP4",
        )
        self.assertEqual(
            provenance["rollout"]["engine"]["cache"]["kv_cache_memory_bytes"],
            2_147_483_648,
        )
        self.assertEqual(
            provenance["rollout"]["engine"]["lora"]["target_modules"],
            ["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.assertIn("torch", provenance["environment"]["libraries"])

    def test_manifest_keeps_reward_semantics_and_additive_provenance(self) -> None:
        provenance = {
            "learner": {"model_id": "learner/model"},
            "rollout": {"model_id": "actor/model", "sync_mode": "lora"},
            "environment": {"libraries": {"torch": "test"}},
        }
        with tempfile.TemporaryDirectory() as directory:
            trainer = TinyLoRAGRPOTrainer(
                model=MinimalModel(),
                tokenizer=object(),
                rollout=object(),
                config=TrainConfig(
                    steps=0,
                    prompt_style="verl",
                    reward_mode="strict",
                ),
                output_dir=directory,
                run_provenance=provenance,
                dataset_split="train[:-512]",
                dataset_revision="dataset-sha",
            )
            # Verify constructor copying: later caller mutation cannot alter the run record.
            provenance["rollout"]["model_id"] = "mutated/model"
            with patch("tinylora_rl.trainer.save_tinylora"):
                trainer.train(EmptyDataset())
            manifest = json.loads((Path(directory) / "run_manifest.json").read_text())

        self.assertEqual(manifest["schema_version"], 3)
        self.assertEqual(manifest["prompt"], {"style": "verl"})
        self.assertEqual(manifest["reward"]["mode"], "strict")
        self.assertEqual(
            manifest["reward"]["implementation"],
            "verl_gsm8k_strict_last_300_chars_v1",
        )
        self.assertEqual(manifest["rollout"]["model_id"], "actor/model")
        self.assertEqual(manifest["base_model_revision"], "learner-sha")
        self.assertEqual(manifest["dataset"]["split"], "train[:-512]")
        self.assertEqual(manifest["dataset"]["revision"], "dataset-sha")


if __name__ == "__main__":
    unittest.main()
