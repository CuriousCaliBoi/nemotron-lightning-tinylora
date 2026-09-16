import unittest

from tinylora_rl.prompts import gsm8k_messages


class PromptsTest(unittest.TestCase):
    def test_concise_prompt_preserves_system_instruction(self) -> None:
        messages = gsm8k_messages("What is 2+2?", "concise")
        self.assertEqual([item["role"] for item in messages], ["system", "user"])
        self.assertIn("####", messages[0]["content"])

    def test_verl_prompt_is_single_user_turn(self) -> None:
        messages = gsm8k_messages("What is 2+2?", "verl")
        self.assertEqual([item["role"] for item in messages], ["user"])
        self.assertIn('after "####"', messages[0]["content"])

    def test_unknown_style_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            gsm8k_messages("question", "unknown")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
