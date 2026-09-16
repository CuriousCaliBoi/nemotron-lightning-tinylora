# Nemotron 3.5 Lightning RLVR memory analysis

Measured on the local DGX Spark on 2026-09-15. The raw profile is
[`outputs/nemotron35-tinylora-memory-profile-20260915/memory_profile.json`](outputs/nemotron35-tinylora-memory-profile-20260915/memory_profile.json), and the
step metrics are in the adjacent `metrics.jsonl`.

## Bottom line

The current setup already performs real on-device RLVR optimization of
NVIDIA Nemotron 3.5 Lightning. The measured run used a BF16 differentiable
learner, an NVFP4 vLLM actor, 13 trainable FP32 TinyLoRA values, 32 generated
trajectories, microbatch size 1, and one AdamW update. It produced a nonzero
gradient, changed the adapter, and synchronized all 24 adapted weights back to
vLLM.

The adapter and optimizer are not the capacity limit. The learner plus rollout
copies dominate static memory, and the differentiable forward/backward through
47 of 52 blocks dominates dynamic memory and time.

The measured run preceded a post-run fix that restricts vLLM LoRA slots to the
actual Q/K/V/O targets. Therefore its roughly 423 MiB unrestricted LoRA-slot
allocation describes the earlier successful runs, not the corrected code.

## What carries over from Song Han's work

The relevant paper is *On-Device Training Under 256KB Memory*. Its enduring
lesson is to account for the lifetime of weights, activations, gradients, and
optimizer state separately, then choose the update graph under a memory budget.
Its key mechanisms were real INT8 training with quantization-aware gradient
scaling, sparse layer/tensor updates selected by contribution analysis, and a
compiler that pruned dead backward nodes and reordered in-place updates.

The closest LLM follow-up is *PockEngine*. Its important result for this setup
is that LoRA lowers gradient and optimizer storage but barely shortens an
iteration if an adapter near the bottom of the network still forces a deep
backward traversal. PockEngine's Llama-7B table reports PyTorch full tuning at
7.7 s/45.1 GB, PyTorch rank-8 LoRA at 7.3 s/30.9 GB, and its late-layer sparse
update at 0.9 s/31.2 GB.

Applied here:

- Song Han's QAS is not our mechanism. Backward uses a BF16 learner. The NVFP4
  model is an inference actor, and truncated importance sampling (TIS) corrects
  actor/learner policy drift rather than quantized gradient scale.
- TinyLoRA is our sparse *parameter* update, but it is not yet a shallow
  backward update. The earliest target is layer 5, so gradients traverse the
  suffix from layers 5 through 51.
- Native vLLM LoRA overlays avoid the MCU paper's low-rank-update failure mode:
  the NVFP4 base is never made mutable, merged, or requantized.
- Gradient checkpointing provides generic graph-memory savings, but this is not
  the compile-time dead-code elimination and operator scheduling of TTE or
  PockEngine.

Sources: [On-Device Training Under 256KB Memory](https://arxiv.org/html/2206.15472),
[PockEngine](https://arxiv.org/html/2310.17752).

## Exact live-state inventory

All byte counts below come from unique underlying tensor storages.

| Object | Exact bytes | Binary size |
|---|---:|---:|
| Frozen BF16 learner parameters | 63,155,874,688 | 58.818492 GiB |
| Learner's base FP32 buffers | 11,776 | 11.50 KiB |
| Frozen TinyLoRA factors and projections | 467,136 | 456.19 KiB |
| Trainable bank, 13 FP32 values | 52 | 52 B |
| Bank gradient after backward | 52 | 52 B |
| Adam state after first update | 108 | 108 B |
| Registered NVFP4 rollout tensors | 18,890,070,924 | 17.592749 GiB |
| Registered learner + rollout tensors | 82,046,424,576 | 76.411687 GiB |

Adam's 108 bytes are two 52-byte FP32 moment tensors plus one four-byte step
scalar. The complete current TinyLoRA-specific learner state after backward and
Adam is 467,348 bytes. The exported rank-2 vLLM adapter tensor payload is
466,944 bytes.

After vLLM initialization, PyTorch had 83.366 GiB allocated. The roughly
6.95 GiB above registered model tensors includes KV/cache storage, workspaces,
and engine state. vLLM itself reported 18.04 GiB for weights, 6.35 GiB for its
54,374-token KV cache, and 0.57 GiB peak initialization activation.

## One measured step

The 32 trajectories had four prompts, eight generations per prompt, completion
lengths up to 256, and an actual padded learner sequence length of 363. The
whole update portion took 236.02 seconds.

The profiled process took 807.34 seconds end to end: loading the BF16 learner
took 367.42 seconds and initializing vLLM took 200.03 seconds. Multi-step runs
therefore amortize about 9.5 minutes of startup; launching a fresh process for
every optimizer step would be very inefficient. The managed wrapper also makes
the production Lightning endpoint unavailable during this interval because the
two workloads do not fit concurrently.

| Phase | Time | CUDA peak above phase start | Minimum host `MemAvailable` |
|---|---:|---:|---:|
| vLLM rollout generation | 39.88 s | 0.472 GiB | 24.54 GiB |
| Frozen old-policy log-probs | 41.27 s | 1.576 GiB | 24.28 GiB |
| Differentiable forward/backward | 154.63 s | 3.243 GiB | 20.38 GiB |
| Gradient clipping + AdamW | 0.090 s | 2 KiB allocator granularity | 20.68 GiB |
| Updated-adapter sync | 0.095 s | 32 KiB allocator granularity | 20.62 GiB |

Absolute PyTorch allocation peaked at 87.010 GiB during forward/backward. Host
used memory rose by 4.270 GiB over that phase. These views overlap on coherent
unified memory and must not be added.

The saved-tensor hook saw 282.014 MiB of peak non-parameter storage for the
first representative microbatch. With non-reentrant checkpointing, this means
checkpoint-boundary and outside-checkpoint autograd state, not all temporary
recomputation workspaces. The allocator's 3.243 GiB phase peak is the capacity
authority.

Full-vocabulary logits explain most of the persistent saved-tensor number. At
sequence length 363 and vocabulary size 131,072, BF16 logits occupy 90.75 MiB;
the current FP32 next-token-logit conversion occupies another 181.00 MiB.

Startup temporarily produced the lowest host headroom: `MemAvailable` reached
10.575 GiB while loading the BF16 shards. This is staging/page-cache pressure,
not optimizer pressure. On this unified-memory machine, `cudaMemGetInfo` is
reclaim-unaware and reported less than 1.5 GiB free immediately before a
successful 3.243 GiB training allocation. Treat it as advisory.

## What larger adapters cost

There are six attention blocks at layers 5, 12, 19, 26, 33, and 42. Across
their 24 Q/K/V/O matrices, a conventional rank-r LoRA has `116,736 * r`
trainable values.

| Rank | Standard LoRA values | Tiny frozen factors, current 13-value bank | Standard FP32 params + grads + Adam | BF16 vLLM payload |
|---:|---:|---:|---:|---:|
| 2 | 233,472 | 0.446 MiB | 3.563 MiB | 0.445 MiB |
| 4 | 466,944 | 0.892 MiB | 7.125 MiB | 0.891 MiB |
| 8 | 933,888 | 1.784 MiB | 14.250 MiB | 1.781 MiB |
| 16 | 1,867,776 | 3.574 MiB | 28.500 MiB | 3.563 MiB |

Even conventional rank-8 LoRA state is tiny relative to the measured 20.38 GiB
host headroom. The safer way to add TinyLoRA capacity is not to raise rank:

| Configuration | Independent trained values | Post-step Tiny state | Meaning |
|---|---:|---:|---|
| `r=2, groups=13, u=1` | 13 | 467,348 B | Current extreme-compression point |
| `r=2, groups=24, u=1` | 24 | 467,524 B | One direction per target matrix |
| `r=2, groups=13, u=4` | 52 | 468,548 B | Full four-direction basis per group |
| `r=2, groups=24, u=4` | 96 | 469,252 B | Full rank-2 LoRA-XS middle per matrix |
| `r=4, groups=24, u=16` | 384 | 952,324 B | Full rank-4 LoRA-XS middle per matrix |

At fixed rank 2, increasing groups or projection dimension does not change the
exported adapter shape, vLLM slot size, or serving-time LoRA compute. Raising
rank increases factor storage, transport size, and residual matmul work but
does not add trained dimensions while the bank remains 13 by 1. The TinyLoRA
paper also reports that ranks above 2 hurt its small-budget experiments. The
managed wrapper now derives a rank-specific factor-cache path so a rank change
does not silently recompute mismatched rank-2 factors on every launch.

Standard LoRA is still a useful high-capacity control because it can learn the
left and right subspaces, while TinyLoRA/LoRA-XS stays within frozen SVD
subspaces. It may retain several MiB more target inputs for gradients, but that
is not a fit risk here.

Expanding beyond the 24 attention matrices is a different risk class. The
other 46 blocks are Mamba or MoE; their expert parameters are not necessarily
ordinary `nn.Linear` modules, their vLLM packing differs, and adding an earlier
target can lengthen the backward suffix. Mamba/MoE adaptation therefore needs
wrapper, export, and kernel validation rather than only a larger memory budget.

## vLLM slot correction

Before the post-run fix, `max_lora_rank=8` and an unspecified target list made
vLLM reserve logical slots for supported expert, shared-expert, Mamba, embedding,
and output modules. The derived BF16 slot size was 423.027 MiB. Restricting
`lora_target_modules` to Q/K/V/O reduces this to 1.781 MiB, a 421.246 MiB
reduction. The measured rollout's persistent 410.273 MiB allocation growth
after initial adapter installation independently corroborates that diagnosis.

At fixed `gpu_memory_utilization=0.23`, vLLM may convert freed slot space into a
larger KV cache rather than return it as system headroom. To reclaim memory for
training, test the target filter together with lower vLLM utilization (0.20 is
the next canary) or an explicit KV-cache budget. Do not lower it blindly: the
32 simultaneous generations need enough pages.

## How much quality TinyLoRA loses

No credible Nemotron number exists yet; one optimizer step is an execution and
memory proof, not an evaluation. The Qwen results are useful priors only.

The TinyLoRA paper reports a GSM8K baseline of 76%, about 91% at 13 values,
about 95% around 100--200 values, and a plotted full-finetuning result near 97%.
That is roughly a six-point gap at 13 and a two-point gap around 100--200. On
its harder six-benchmark Qwen-7B table, the exact averages are 40.3 base, 50.1
at 13, 53.2 at 196, 54.4 at 392, and 55.2 for full tuning: gaps of 5.1, 2.0,
and 0.8 points respectively.

The arXiv v1 prose and figures contain inconsistent values, and it omits the
exact tying map and several optimizer/evaluation details. Results should be
described as a protocol-level replication unless author configs become
available. The paper's evidence is also math-specific and may reflect eliciting
capability already present in the base model; it cannot predict agentic-research
transfer. Source: [Learning to Reason in 13 Parameters](https://arxiv.org/html/2602.04118).

This canary's mean sampled reward was already 0.9375 and half of prompt groups
had zero reward variance, so easy GSM8K produces a weak learning signal for
Lightning. The actor/learner mismatch was well behaved: TIS mean 0.9972, ratio
range 0.203--1.806, and only 0.061% of tokens truncated.

## Recommended experiment order

1. Add a held-out pass@1 evaluation before making quality claims.
2. Run the corrected `13 -> 24 -> 52 -> 96` rank-2 capacity ladder with matched
   prompts, rollouts, token budgets, learning-rate sweeps, and at least three
   seeds.
3. Compare all six attention targets with the last four (layers 19, 26, 33,
   42). This shortens the differentiable suffix from 47 to 33 blocks and is more
   likely to change memory/time than adapter size.
4. Add conventional rank-2 LoRA as the subspace-learning control.
5. Move Lightning from easy GSM8K to a harder verifiable math/code or agentic
   task only after the capacity/depth ablation is understood.

The optimizer core is ready for agentic RLVR, but the current rollout loop is
still a one-turn text generator. A real research-agent experiment additionally
needs a multi-turn tool environment, action-token log-prob capture, loss masks
that exclude tool observations, an episode schema, and a deterministic verifier
for claims/citations or task completion. Longer tool traces will stress KV
cache and full-vocabulary learner logits much sooner than adapter state, so the
depth and logit-memory optimizations should precede long-horizon runs.

For a paper-faithful TinyLoRA replication, use Qwen2.5-7B-Instruct, full GSM8K
for three epochs, four samples per problem, batch size 64, 4,096-token maximum
generations, no KL penalty, the seven stated learning rates, three seeds, and
base/full/LoRA/LoRA-XS controls. A Lightning run is a transfer study, not the
paper replication. The paper does not state whether batch size 64 counts
prompts or generated sequences, so both the assumption and effective trajectory
batch must be reported.
