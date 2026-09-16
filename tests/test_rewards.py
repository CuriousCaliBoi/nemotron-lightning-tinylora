import unittest

from tinylora_rl.rewards import (
    gsm8k_reward,
    predicted_answer,
    reference_answer,
    strict_predicted_answer,
    strict_reference_answer,
)


class RewardsTest(unittest.TestCase):
    def test_gsm8k_formats(self) -> None:
        self.assertEqual(reference_answer("work\n#### 1,234"), "1234")
        self.assertEqual(predicted_answer("therefore \\boxed{1234}"), "1234")
        self.assertEqual(predicted_answer("last value is 1,234"), "1234")
        self.assertEqual(predicted_answer("#### 6\nrevised #### 7"), "7")
        self.assertEqual(gsm8k_reward("#### 7", "solution #### 7"), 1.0)
        self.assertEqual(gsm8k_reward("#### 8", "solution #### 7"), 0.0)

    def test_strict_reward_requires_hash_answer(self) -> None:
        self.assertIsNone(strict_predicted_answer("therefore \\boxed{7}"))
        self.assertEqual(strict_predicted_answer("#### 6\nrevised #### 7"), "7")
        self.assertEqual(
            gsm8k_reward("therefore \\boxed{7}", "solution #### 7", mode="flexible"),
            1.0,
        )
        self.assertEqual(
            gsm8k_reward("therefore \\boxed{7}", "solution #### 7", mode="strict"),
            0.0,
        )

    def test_strict_reward_matches_verl_extraction(self) -> None:
        self.assertEqual(strict_reference_answer("work\n#### 1,234"), "1234")
        self.assertEqual(strict_predicted_answer("work\n#### 1,234"), "1234")
        self.assertIsNone(strict_predicted_answer("####7"))
        self.assertIsNone(strict_predicted_answer("####  7"))
        self.assertIsNone(strict_predicted_answer("#### +7"))
        self.assertEqual(
            strict_predicted_answer("#### 7" + "x" * 301 + "#### 8"),
            "8",
        )
        self.assertEqual(
            gsm8k_reward("#### 1.0", "solution #### 1", mode="strict"),
            0.0,
        )
        self.assertEqual(gsm8k_reward("no answer", "malformed", mode="strict"), 0.0)


if __name__ == "__main__":
    unittest.main()
