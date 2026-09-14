import unittest

import torch

from tinylora_rl.objectives import (
    GRPOBatch,
    compute_group_advantages,
    grpo_policy_loss,
    selected_completion_logprobs,
    truncated_importance_weights,
)


class ObjectivesTest(unittest.TestCase):
    def test_group_advantages_are_independently_normalized(self) -> None:
        rewards = torch.tensor([0.0, 1.0, 2.0, 2.0])
        groups = torch.tensor([0, 0, 1, 1])
        advantages = compute_group_advantages(rewards, groups)
        expected = torch.tensor([-0.7071058, 0.7071058, 0.0, 0.0])
        torch.testing.assert_close(advantages, expected, rtol=1e-5, atol=1e-5)

    def test_tis_clip_and_mask(self) -> None:
        trainer = torch.log(torch.tensor([[0.01, 1.0, 100.0]]))
        rollout = torch.zeros_like(trainer)
        mask = torch.ones_like(trainer)
        clipped = truncated_importance_weights(
            trainer, rollout, mask, mode="token_clip", minimum=0.1, maximum=10.0
        )
        torch.testing.assert_close(clipped, torch.tensor([[0.1, 1.0, 10.0]]))
        masked = truncated_importance_weights(
            trainer, rollout, mask, mode="token_mask", minimum=0.1, maximum=10.0
        )
        torch.testing.assert_close(masked, torch.tensor([[0.0, 1.0, 0.0]]))

    def test_selected_logprobs_align_next_tokens(self) -> None:
        logits = torch.tensor(
            [[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0], [1.0, 1.0, 1.0]]]
        )
        ids = torch.tensor([[0, 0, 1, 2]])
        mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        actual = selected_completion_logprobs(logits, ids, mask)
        expected_token_1 = torch.log_softmax(logits[0, 1], -1)[1]
        expected_token_2 = torch.log_softmax(logits[0, 2], -1)[2]
        torch.testing.assert_close(actual[0, 2], expected_token_1)
        torch.testing.assert_close(actual[0, 3], expected_token_2)
        self.assertEqual(float(actual[0, :2].abs().sum()), 0.0)

    def test_grpo_loss_has_policy_gradient_at_ratio_one(self) -> None:
        current = torch.tensor([[-0.5, -0.7], [-0.5, -0.7]], requires_grad=True)
        old = current.detach().clone()
        mask = torch.ones_like(current)
        batch = GRPOBatch(
            rewards=torch.tensor([0.0, 1.0]),
            group_ids=torch.tensor([0, 0]),
            old_logprobs=old,
            rollout_logprobs=old,
            response_mask=mask,
        )
        loss, metrics = grpo_policy_loss(
            current,
            torch.tensor([-1.0, 1.0]),
            batch,
            tis_mode="none",
        )
        loss.backward()
        self.assertGreater(float(current.grad.abs().sum()), 0.0)
        torch.testing.assert_close(metrics["policy_ratio_mean"], torch.tensor(1.0))


if __name__ == "__main__":
    unittest.main()
