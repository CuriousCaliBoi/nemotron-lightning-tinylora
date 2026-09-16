import hashlib
import json
import stat
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from tinylora_rl.adapter_analysis import export_peft_adapter, load_adapter_updates
from tinylora_rl.registry import (
    RegistryError,
    add_adapter,
    add_evaluation,
    adapter_path,
    list_adapters,
    show_adapter,
    verify_adapter,
)


def make_run(root: Path, *, bank_value: float = 0.25) -> Path:
    run = root / "run"
    adapter = run / "final_adapter"
    adapter.mkdir(parents=True)
    left = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 10
    right = torch.arange(6, dtype=torch.float32).reshape(2, 3) / 10
    projection = torch.eye(2, dtype=torch.float32).unsqueeze(0)
    save_file(
        {
            "bank.v": torch.tensor([[bank_value]], dtype=torch.float32),
            "layers.0.left": left,
            "layers.0.right": right,
            "layers.0.projection": projection,
        },
        str(adapter / "adapter.safetensors"),
    )
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "config": {
                    "rank": 2,
                    "projection_dim": 1,
                    "num_groups": 1,
                    "target_modules": ["proj"],
                },
                "modules": [
                    {"name": "model.proj", "group_id": 0, "scaling": 1.0}
                ],
            }
        )
    )
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "base_model": "test/base-model",
                "trainable_parameters": 1,
                "train_config": {"steps": 1, "seed": 42},
            }
        )
    )
    (run / "metrics.jsonl").write_text(
        json.dumps({"step": 1, "reward_mean": 0.5}) + "\n"
    )
    return run


def adapter_artifacts(adapter: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        path = adapter / filename
        result[filename] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
    return result


class AdapterRegistryTest(unittest.TestCase):
    def test_add_export_list_show_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = make_run(root / "source")
            registry = root / "registry"
            label = "qwen2.5-7b/gsm8k/paper-rep.v1"
            manifest = add_adapter(
                registry,
                run,
                label=label,
                export_peft=True,
                base_revision="abc123",
            )

            self.assertEqual(manifest["adapter"]["trainable_parameters"], 1)
            self.assertEqual(manifest["formats"], ["peft-lora", "tinylora-v1"])
            self.assertEqual(
                manifest["provenance"]["training_summary"]["final"]["reward_mean"],
                0.5,
            )
            object_path = (
                registry
                / "objects"
                / "sha256"
                / manifest["object_id"].removeprefix("sha256:")
            )
            self.assertTrue((object_path / "native" / "adapter.safetensors").is_file())
            self.assertTrue((object_path / "peft" / "adapter_model.safetensors").is_file())
            self.assertEqual(
                stat.S_IMODE((object_path / "native" / "adapter.safetensors").stat().st_mode),
                0o444,
            )
            # A dot in the final label segment must not be mistaken for a suffix.
            self.assertTrue(
                (registry / "refs" / "qwen2.5-7b" / "gsm8k" / "paper-rep.v1.json").is_file()
            )

            repeated = add_adapter(
                registry,
                run,
                label=label,
                export_peft=True,
                base_revision="abc123",
            )
            self.assertEqual(repeated["object_id"], manifest["object_id"])
            entries = list_adapters(registry)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["label"], label)
            shown = show_adapter(registry, label)
            self.assertEqual(shown["labels"], [label])
            self.assertEqual(shown["evaluations"], [])
            verified = verify_adapter(registry, label)
            self.assertEqual(verified["artifacts_verified"], 6)
            self.assertEqual(verified["evaluations_verified"], 0)
            self.assertEqual(
                adapter_path(registry, label, adapter_format="peft"),
                object_path / "peft",
            )

    def test_candidate_run_copies_parent_sweep_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sweep = root / "sweep"
            run = make_run(sweep / "candidate")
            for filename in (
                "sweep_manifest.json",
                "sweep_results.json",
                "sweep_status.json",
            ):
                (run.parent / filename).write_text(json.dumps({"source": filename}))
            registry = root / "registry"
            manifest = add_adapter(
                registry,
                run,
                label="qwen2.5-7b/gsm8k/sweep-candidate",
            )
            copied = manifest["provenance"]["copied_files"]
            self.assertIn("parent_sweep_manifest.json", copied)
            self.assertIn("parent_sweep_results.json", copied)
            self.assertIn("parent_sweep_status.json", copied)
            roles = {artifact["role"] for artifact in manifest["artifacts"]}
            self.assertIn("provenance_parent_sweep_manifest", roles)
            self.assertIn("provenance_parent_sweep_results", roles)
            self.assertIn("provenance_parent_sweep_status", roles)

    def test_derived_control_copies_parent_sweep_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sweep = root / "sweep"
            native = make_run(root / "source") / "final_adapter"
            control = sweep / "zero_control"
            export_peft_adapter(
                load_adapter_updates(native),
                control,
                base_model_name="test/base-model",
            )
            for filename in (
                "sweep_manifest.json",
                "sweep_results.json",
                "sweep_status.json",
            ):
                (sweep / filename).write_text(json.dumps({"source": filename}))

            manifest = add_adapter(
                root / "registry",
                control,
                label="test-model/gsm8k/zero-control",
            )
            self.assertEqual(
                set(manifest["provenance"]["copied_files"]),
                {
                    "parent_sweep_manifest.json",
                    "parent_sweep_results.json",
                    "parent_sweep_status.json",
                },
            )

    def test_declared_peft_base_cannot_be_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = make_run(root / "source") / "final_adapter"
            peft = root / "peft"
            export_peft_adapter(
                load_adapter_updates(native),
                peft,
                base_model_name="test/base-model",
            )
            with self.assertRaisesRegex(RegistryError, "conflicts with the adapter"):
                add_adapter(
                    root / "registry",
                    peft,
                    label="test-model/gsm8k/wrong-base",
                    base_model="test/a-different-model",
                )

    def test_peft_declared_actor_base_overrides_enclosing_learner_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = make_run(root / "source")
            peft = run / "peft_adapter"
            export_peft_adapter(
                load_adapter_updates(run / "final_adapter"),
                peft,
                base_model_name="test/quantized-actor",
            )
            manifest = add_adapter(
                root / "registry",
                peft,
                label="test-actor/gsm8k/deployment-adapter",
                base_revision="actor-sha",
            )
            self.assertEqual(
                manifest["base_model"],
                {"id": "test/quantized-actor", "revision": "actor-sha"},
            )
            self.assertEqual(
                manifest["provenance"]["run_manifest"]["base_model"],
                "test/base-model",
            )

    def test_label_move_is_explicit_and_audited(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "registry"
            first = make_run(root / "first", bank_value=0.1)
            second = make_run(root / "second", bank_value=0.2)
            label = "qwen2.5-7b/gsm8k/candidate"
            old = add_adapter(registry, first, label=label)
            with self.assertRaisesRegex(RegistryError, "already points"):
                add_adapter(registry, second, label=label)
            new = add_adapter(
                registry,
                second,
                label=label,
                replace_label=True,
                reason="higher held-out accuracy",
            )
            self.assertNotEqual(old["object_id"], new["object_id"])
            ref = json.loads(
                (registry / "refs" / "qwen2.5-7b" / "gsm8k" / "candidate.json").read_text()
            )
            self.assertEqual(ref["previous_object_id"], old["object_id"])
            self.assertEqual(ref["reason"], "higher held-out accuracy")

    def test_add_evaluation_keeps_adapter_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "registry"
            run = make_run(root / "source")
            label = "qwen2.5-7b/gsm8k/tinylora-1p"
            adapter = add_adapter(registry, run, label=label, export_peft=True)
            registered_peft = (
                registry
                / "objects"
                / "sha256"
                / adapter["object_id"].removeprefix("sha256:")
                / "peft"
            )
            object_manifest = (
                registry
                / "objects"
                / "sha256"
                / adapter["object_id"].removeprefix("sha256:")
                / "manifest.json"
            )
            before = object_manifest.read_bytes()
            results = root / "gsm8k-eval.json"
            results.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "metric": "gsm8k_greedy_exact_match",
                        "model": "test/base-model",
                        "base": {"samples": 100, "accuracy": 0.2},
                        "comparison_baseline": "base",
                        "candidates": {
                            "tiny": {
                                "samples": 100,
                                "accuracy": 0.3,
                                "accuracy_delta": 0.1,
                                "adapter_artifacts": adapter_artifacts(registered_peft),
                                "details": [{"large": "payload"}],
                            }
                        },
                        "comparisons": {
                            "tiny": {
                                "accuracy_delta": 0.1,
                                "paired_bootstrap_95_ci": [0.01, 0.19],
                                "mcnemar_exact_p": 0.02,
                            }
                        },
                    }
                )
            )
            evaluation = add_evaluation(
                registry,
                label,
                results,
                name="gsm8k/test-n100-greedy",
            )
            self.assertEqual(evaluation["candidate"], "tiny")
            self.assertEqual(evaluation["summary"]["adapted"]["accuracy"], 0.3)
            self.assertNotIn("details", evaluation["summary"]["adapted"])
            self.assertEqual(evaluation["summary"]["comparison_baseline"], "base")
            self.assertEqual(
                evaluation["summary"]["comparisons"]["tiny"]["mcnemar_exact_p"],
                0.02,
            )
            self.assertEqual(object_manifest.read_bytes(), before)
            shown = show_adapter(registry, label)
            self.assertEqual(len(shown["evaluations"]), 1)
            self.assertEqual(shown["evaluations"][0]["evaluation_id"], evaluation["evaluation_id"])
            verified = verify_adapter(registry, label)
            self.assertEqual(verified["evaluations_verified"], 1)

    def test_evaluation_rejects_artifacts_from_another_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "registry"
            first = add_adapter(
                registry,
                make_run(root / "first", bank_value=0.1),
                label="qwen2.5-7b/gsm8k/first",
                export_peft=True,
            )
            add_adapter(
                registry,
                make_run(root / "second", bank_value=0.2),
                label="qwen2.5-7b/gsm8k/second",
                export_peft=True,
            )
            first_peft = (
                registry
                / "objects"
                / "sha256"
                / first["object_id"].removeprefix("sha256:")
                / "peft"
            )
            results = root / "wrong-adapter-eval.json"
            results.write_text(
                json.dumps(
                    {
                        "candidates": {
                            "first": {
                                "accuracy": 0.5,
                                "adapter_artifacts": adapter_artifacts(first_peft),
                            }
                        }
                    }
                )
            )
            with self.assertRaisesRegex(RegistryError, "do not match"):
                add_evaluation(
                    registry,
                    "qwen2.5-7b/gsm8k/second",
                    results,
                    name="gsm8k/cross-adapter",
                )

    def test_verify_checks_evaluation_reference_identity_and_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "registry"
            label = "qwen2.5-7b/gsm8k/ref-integrity"
            adapter = add_adapter(
                registry,
                make_run(root / "source"),
                label=label,
                export_peft=True,
            )
            peft = (
                registry
                / "objects"
                / "sha256"
                / adapter["object_id"].removeprefix("sha256:")
                / "peft"
            )
            results = root / "eval.json"
            results.write_text(
                json.dumps(
                    {
                        "candidates": {
                            "tiny": {"adapter_artifacts": adapter_artifacts(peft)}
                        }
                    }
                )
            )
            add_evaluation(registry, label, results, name="gsm8k/ref-check")
            ref_path = (
                registry
                / "evaluation_refs"
                / adapter["object_id"].removeprefix("sha256:")
                / "gsm8k"
                / "ref-check.json"
            )
            ref = json.loads(ref_path.read_text())
            ref["adapter_object_id"] = "sha256:" + "0" * 64
            ref_path.write_text(json.dumps(ref))
            with self.assertRaisesRegex(RegistryError, "names adapter"):
                verify_adapter(registry, label)

            ref["adapter_object_id"] = adapter["object_id"]
            ref["name"] = "gsm8k/a-different-name"
            ref_path.write_text(json.dumps(ref))
            with self.assertRaisesRegex(RegistryError, "does not match its name"):
                verify_adapter(registry, label)

    def test_verify_checks_adapter_reference_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "registry"
            label = "qwen2.5-7b/gsm8k/ref-integrity"
            add_adapter(registry, make_run(root / "source"), label=label)
            ref_path = registry / "refs" / "qwen2.5-7b" / "gsm8k" / "ref-integrity.json"
            ref = json.loads(ref_path.read_text())
            ref["label"] = "qwen2.5-7b/gsm8k/a-different-label"
            ref_path.write_text(json.dumps(ref))
            with self.assertRaisesRegex(RegistryError, "declares label"):
                verify_adapter(registry, label)

    def test_peft_companion_must_materialize_native_delta(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = make_run(root / "first", bank_value=0.1)
            first_native = first / "final_adapter"
            peft = root / "first-peft"
            export_peft_adapter(
                load_adapter_updates(first_native),
                peft,
                base_model_name="test/base-model",
            )
            registry = root / "registry"
            valid = add_adapter(
                registry,
                first,
                label="qwen2.5-7b/gsm8k/valid-companion",
                peft=peft,
            )
            self.assertEqual(valid["formats"], ["peft-lora", "tinylora-v1"])

            second = make_run(root / "second", bank_value=0.2)
            with self.assertRaisesRegex(RegistryError, "does not materialize native delta"):
                add_adapter(
                    registry,
                    second,
                    label="qwen2.5-7b/gsm8k/wrong-companion",
                    peft=peft,
                )

    def test_verify_detects_artifact_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "registry"
            run = make_run(root / "source")
            label = "qwen2.5-7b/gsm8k/tamper-test"
            manifest = add_adapter(registry, run, label=label)
            weights = (
                registry
                / "objects"
                / "sha256"
                / manifest["object_id"].removeprefix("sha256:")
                / "native"
                / "adapter.safetensors"
            )
            weights.chmod(0o644)
            with weights.open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(RegistryError, "size mismatch"):
                verify_adapter(registry, label)

    def test_multiple_eval_candidates_require_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "registry"
            label = "qwen2.5-7b/gsm8k/multi-eval"
            add_adapter(registry, make_run(root / "source"), label=label)
            results = root / "eval.json"
            results.write_text(json.dumps({"candidates": {"a": {}, "b": {}}}))
            with self.assertRaisesRegex(RegistryError, "pass --candidate"):
                add_evaluation(registry, label, results, name="gsm8k/test")

    def test_rejects_path_traversal_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RegistryError, "invalid label"):
                add_adapter(
                    root / "registry",
                    make_run(root / "source"),
                    label="../outside",
                )


if __name__ == "__main__":
    unittest.main()
