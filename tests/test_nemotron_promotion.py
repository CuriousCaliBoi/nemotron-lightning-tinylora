import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_nemotron35_tinylora_promote.sh"


def embedded_python(source: str) -> list[str]:
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", source, flags=re.DOTALL)
    if not blocks:
        raise AssertionError("promotion script contains no embedded Python")
    return blocks


class NemotronPromotionTest(unittest.TestCase):
    def test_shell_and_embedded_python_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
        source = SCRIPT.read_text()
        for index, block in enumerate(embedded_python(source)):
            compile(block, f"{SCRIPT.name}:heredoc-{index}", "exec")

    def test_labels_and_content_addressed_preflight_are_fixed(self) -> None:
        source = SCRIPT.read_text()
        self.assertIn(
            'OUTPUT_ROOT_REL="outputs/nemotron35-tinylora-lr-scale-replay-canary-20260916"',
            source,
        )
        self.assertIn('screen-consumed128-max1024.json', source)
        self.assertIn('confirmatory-untouched384-max1024-canonical-zero-control.json', source)
        self.assertIn(
            'protocol_label="tinylora13-concise-verl-strict-exact-step1-replay-v1"',
            source,
        )
        self.assertIn('zero_label="$label_prefix/zero-lora-r2-attn24-s42-nvfp4"', source)
        self.assertIn('peft_labels+=("$stem-nvfp4")', source)
        self.assertIn('native_labels+=("$stem-bf16-native")', source)
        self.assertIn('register_all "$temporary_registry"', source)
        self.assertIn('check_registry_state 0', source)
        self.assertIn('check_registry_state 1', source)
        self.assertNotIn("--replace-label", source)
        self.assertNotIn('      --replace \\', source)

    def test_zero_repeat_order_and_four_way_holm_are_guarded(self) -> None:
        source = SCRIPT.read_text()
        self.assertIn(
            'expected_eval_candidates = ["zero", *candidates, "zero_repeat"]',
            source,
        )
        self.assertIn(
            'evaluation.get("holm_family_labels") != list(candidates)',
            source,
        )
        self.assertIn('comparison.get("multiple_comparison_family_size") != 4', source)
        self.assertIn('Repeatability is a measured diagnostic, not a promotion gate', source)
        self.assertIn('"require_exact_samples": True', source)
        self.assertIn('require_explicit_bf16=True', source)
        self.assertIn('confirmation is not content-bound to the exact validated selection screen', source)

    def test_zero_dtype_mismatch_is_explicitly_validated(self) -> None:
        source = SCRIPT.read_text()
        for expected in (
            'len(zero_tensors) != 48',
            'sum(tensor.numel() for tensor in zero_tensors.values()) != 233472',
            '{tensor.dtype for tensor in zero_tensors.values()} != {torch.float32}',
            '{tensor.dtype for tensor in learned_tensors.values()} != {torch.bfloat16}',
            'zero_tensors[key].to(torch.bfloat16)',
            'zero control contains a nonzero PEFT B tensor',
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, source)


if __name__ == "__main__":
    unittest.main()
