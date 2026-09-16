# Local adapter registry

This directory is the local, content-addressed store for TinyLoRA and PEFT
adapters produced by this workspace. Generated objects, labels, and evaluation
records are intentionally ignored by Git; this README is the only tracked
file.

Bundles are immutable and addressed by SHA-256. Human-readable labels under
`refs/` point to those bundles, and held-out evaluations are stored separately
so measuring an adapter never mutates it.

For a candidate recovered from an interrupted sweep, the bundle also retains
the parent `recovery_manifest.json` when present. This keeps the reason for the
interruption, recovery checks, and regenerated artifacts alongside the normal
sweep provenance rather than silently presenting the checkpoint as a clean
completed sweep.

Use descriptive labels that encode the model, task, adapter size, protocol,
seed, and checkpoint, for example
`nemotron3.5-lightning/gsm8k/tinylora13-concise-strict-s42-step2`.

The completed 2026-09-16 Lightning replay sweep has a guarded, idempotent
promotion script. It requires both the 128-row selection screen and the
disjoint 384-row confirmation, and attaches both under distinct evaluation
names. Run its preflight and promotion inside the research image so the
root-owned checkpoints and registry objects are readable:

```bash
docker run --rm -v /home/nimitz/projects/RLtests:/workspace -w /workspace \
  -e PREFLIGHT_ONLY=1 spark-vllm-tinylora:0.1 \
  bash /workspace/run_nemotron35_tinylora_promote.sh

docker run --rm -v /home/nimitz/projects/RLtests:/workspace -w /workspace \
  spark-vllm-tinylora:0.1 \
  bash /workspace/run_nemotron35_tinylora_promote.sh
```

Candidate labels follow
`nemotron3.5-lightning/gsm8k/tinylora13-concise-verl-strict-exact-step1-replay-v1-lr{lr}-scale{scale}-s42-step2-{nvfp4|bf16-native}`.
The script registers four deployment PEFT adapters, four separate native BF16
checkpoints, and the NVFP4 zero-LoRA control. It attaches the shared held-out
evaluation only to the five PEFT objects, using each object's matching
candidate selector. Existing labels or evaluation names are accepted only
when their content IDs are identical; they are never moved or overwritten.

The existing Lightning zero control is FP32 on disk because it was exported
from the native FP32 factors, while the learned rollout PEFT files are BF16.
Promotion verifies all 48 keys and 233,472 elements, proves every zero-control
LoRA B tensor is zero, and proves each LoRA A tensor becomes bit-identical to
all four learned adapters after a BF16 cast. vLLM 0.27.1 casts adapters into
its configured LoRA buffers at load time; the confirmation pins those buffers
to BF16 explicitly. A future exporter should save zero controls directly in
BF16 as well, reducing the file from about 940 KB to 473 KB and making this
equivalence visible without a load-time cast.

The Qwen2.5-7B 64-step canary has the same guarded workflow after its full
test-set evaluation. Supply its exact `RUN_SLUG` to
`run_qwen25_7b_tinylora_promote.sh`. Labels are grouped under
`qwen2.5-7b/gsm8k/tinylora13-verl-strict-canary-lr1e-4-s42/RUN_TAG/` and end in
`zero-lora-r2-all196`, `step16`, `step32`, `step48`, or `step64`. Each trained
object bundles the native checkpoint and its hash-verified PEFT companion.
`final_adapter` is not a sixth label: promotion first proves it is tensor-wise
identical to `checkpoint-64`. The word `canary` is intentional; these labels do
not claim the paper's complete training protocol.

Typical use inside the research container:

```bash
python3 -m tinylora_rl.registry --registry /workspace/adapter_registry \
  add /workspace/outputs/RUN \
  --label qwen2.5-7b/gsm8k/tinylora13-verl-strict-s42-step64 \
  --export-peft --base-revision REVISION

python3 -m tinylora_rl.registry --registry /workspace/adapter_registry list
python3 -m tinylora_rl.registry --registry /workspace/adapter_registry verify LABEL
python3 -m tinylora_rl.registry --registry /workspace/adapter_registry \
  add-eval LABEL /workspace/outputs/evaluation.json \
  --name gsm8k/test-n1319-verl-strict-greedy --candidate step64
```

Container-created source checkpoints are often root-owned with mode `0600`,
so promotion should be run in the container. Registered artifact copies are
host-readable and read-only. Only the learned Qwen TinyLoRA bank is 26 bytes;
the portable native bundle also includes frozen SVD/projection factors, and a
PEFT export materializes inference factors, so the files are much larger.

If learner and rollout use different base checkpoints (for example Lightning
BF16 for gradients and NVFP4 for deployment), register the native learner
checkpoint and PEFT actor adapter under separate labels. A native/PEFT pair is
only bundled together when their declared base model and materialized deltas
match exactly.
