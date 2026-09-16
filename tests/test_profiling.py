import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from tinylora_rl.profiling import (
    UnifiedMemoryProfiler,
    module_tensor_inventory,
    optimizer_state_inventory,
)


class ProfilingTest(unittest.TestCase):
    def test_tensor_and_optimizer_inventories_are_storage_aware(self) -> None:
        model = nn.Linear(4, 3, bias=False)
        inventory = module_tensor_inventory(model)
        self.assertEqual(inventory["parameters"]["bytes"], 4 * 3 * 4)
        self.assertEqual(inventory["trainable_parameter_elements"], 12)
        self.assertEqual(inventory["gradients"]["bytes"], 0)

        optimizer = torch.optim.AdamW(model.parameters())
        model(torch.ones(2, 4)).sum().backward()
        optimizer.step()
        state = optimizer_state_inventory(optimizer)
        # AdamW stores a scalar step and two fp32 moments for this one tensor.
        self.assertEqual(state["bytes"], 4 + 2 * (4 * 3 * 4))

    def test_profiler_writes_phase_and_saved_tensor_measurements(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "memory_profile.json"
            profiler = UnifiedMemoryProfiler(output, enabled=True, sample_interval=0.01)
            model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 1))
            optimizer = torch.optim.AdamW(model.parameters())
            profiler.start()
            profiler.record_module("model", model)
            with profiler.phase("one_step"):
                with profiler.track_saved_tensors("microbatch", model):
                    loss = model(torch.randn(3, 4)).square().mean()
                    loss.backward()
                optimizer.step()
            profiler.record_training_state("after_step", model, optimizer)
            profiler.finish()

            payload = json.loads(output.read_text())
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(payload["phases"][0]["name"], "one_step")
            self.assertGreater(payload["saved_tensor_profiles"][0]["pack_calls"], 0)
            self.assertGreater(
                payload["saved_tensor_profiles"][0][
                    "estimated_peak_live_non_parameter_storage_by_device"
                ]["cpu"],
                0,
            )
            self.assertGreater(
                payload["tensor_inventories"][-1]["optimizer_state"]["bytes"],
                0,
            )


if __name__ == "__main__":
    unittest.main()
