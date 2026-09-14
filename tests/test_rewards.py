import unittest

from tinylora_rl.rewards import gsm8k_reward, predicted_answer, reference_answer


class RewardsTest(unittest.TestCase):
    def test_gsm8k_formats(self) -> None:
        self.assertEqual(reference_answer("work\n#### 1,234"), "1234")
        self.assertEqual(predicted_answer("therefore \\boxed{1234}"), "1234")
        self.assertEqual(predicted_answer("last value is 1,234"), "1234")
        self.assertEqual(gsm8k_reward("#### 7", "solution #### 7"), 1.0)
        self.assertEqual(gsm8k_reward("#### 8", "solution #### 7"), 0.0)


if __name__ == "__main__":
    unittest.main()
