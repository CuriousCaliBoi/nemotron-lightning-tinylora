# TinyLoRA RL from scratch on DGX Spark

This is an independent, protocol-level implementation of the core method in
[Learning to Reason in 13 Parameters](https://arxiv.org/abs/2602.04118).
It uses plain PyTorch/Transformers for gradients and colocated vLLM for
rollouts. The implementation does **not** import TRL or PEFT.

For every frozen linear weight, it computes rank-`r` SVD factors and applies

```text
delta_W = U Sigma (sum_i v_i P_i) V^T
```

`P_i` and the SVD factors are fixed. Only `v` is optimized, and its entries
are tied across projection modules. Qwen2.5-7B has 196 selected projections;
with rank 2, projection dimension 1, and 16 modules per group, the complete
policy has exactly 13 trainable scalars.

The RL stack is deliberately explicit and replaceable:

- exact GSM8K numeric reward;
- group-relative advantages and clipped GRPO in PyTorch;
- token- or sequence-level truncated importance sampling (TIS) to correct
  PyTorch/vLLM numerical drift;
- vLLM generation with recorded chosen-token log probabilities;
- merged-weight refresh after every optimizer step, including Qwen's packed
  `qkv_proj` and `gate_up_proj` layout;
- native vLLM LoRA hot-reload for a quantized rollout model, without merging
  into or requantizing its base weights;
- factor caching and self-contained adapter checkpoints.

## Build and test

The active Nemotron server and training cannot share the Spark GPU. Stop it,
build the research image, and run the CPU tests:

```bash
docker stop --timeout 30 nemotron35_lightning_vllm
docker build -f Dockerfile.research -t spark-vllm-tinylora:0.1 .
docker run --rm -v "$PWD:/workspace" spark-vllm-tinylora:0.1 \
  python3 -m unittest discover -s /workspace/tests -v
```

Run the fast end-to-end GSM8K smoke test (0.5B base, still exactly 13 trainable
scalars because it uses 13 modules per tied group):

```bash
MODEL=Qwen/Qwen2.5-0.5B-Instruct \
MODULES_PER_GROUP=13 STEPS=1 SAMPLES=32 \
OUTPUT_DIR=/workspace/outputs/tinylora-scratch-smoke \
./run_tinylora_rl.sh
```

Run the paper's 7B adapter geometry:

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct \
MODEL_REVISION=a09a35458c702b33eeacc393d103063234e8bc28 \
ROLLOUT_REVISION=a09a35458c702b33eeacc393d103063234e8bc28 \
DATASET_REVISION=740312add88f781978c0658806c59bc2815b9866 \
MODULES_PER_GROUP=16 TINYLORA_RANK=2 PROJECTION_DIM=1 \
PARAMETER_DTYPE=bfloat16 \
PROMPTS_PER_STEP=16 GENERATIONS=4 \
PROMPT_STYLE=verl REWARD_MODE=strict LOSS_REDUCTION=token_mean \
MAX_COMPLETION_LENGTH=4096 MAX_MODEL_LENGTH=4608 \
SAMPLES=7473 STEPS=1402 \
OUTPUT_DIR=/workspace/outputs/qwen2.5-7b-tinylora-gsm8k \
./run_tinylora_rl.sh
```

For the bounded, pinned 64-update effectiveness canary used in this workspace,
run `./run_qwen25_7b_tinylora_canary.sh`. It verifies the frozen-factor hash,
uses the leakage-safe training split, and safely stops/restores the Lightning
serving container.

The Qwen canary and Nemotron sweep launch training as detached, named Docker
containers and only follow their logs from the calling terminal. Consequently,
Ctrl-C, SIGHUP, or a lost SSH session ends the supervisor without signalling
the training process or restarting the inference server on top of it. The
interrupted launcher prints `docker inspect`/`docker logs` recovery commands;
after Docker reports that training has stopped, inspect its exit code, remove
the retained container, and restart the server if its container label
`ai.tinylora.restore-server` is `1`. Normal completion reports the training
exit status, restores the previously running server, and removes the stopped
training container automatically.

`PROMPT_STYLE=verl` uses the canonical single-user GSM8K instruction and
`REWARD_MODE=strict` mirrors VERL's GSM8K string verifier: it requires the
literal `#### ` marker in the final 300 characters and compares the last
extracted answer string. The original repository prompt is still available as
`PROMPT_STYLE=concise`, and normalized last-number scoring remains available
as `flexible` for diagnostics. Do not compare scores across prompt/reward
protocols as if they were the same benchmark.

When a BF16 learner and BF16 vLLM actor are colocated, vLLM's percentage-based
memory profiler may count the learner twice. Set an explicit cache budget to
bypass that heuristic, for example:

```bash
VLLM_GPU_MEMORY_UTILIZATION=0.35 \
VLLM_KV_CACHE_MEMORY_BYTES=2147483648 \
./run_tinylora_rl.sh
```

The percentage must still be below the free-memory fraction at vLLM startup;
the explicit byte value controls the actual KV cache allocation.

The first run computes and caches the truncated SVD factors. Later runs load
them directly. A custom `FACTOR_CACHE=/workspace/cache/name.safetensors` can be
provided; otherwise the filename is derived from the model and rank.

Important research controls are environment variables accepted by
`run_tinylora_rl.sh`: `LEARNING_RATE`, `PPO_EPOCHS`, `CLIP_EPSILON`,
`TIS_MODE`, `TIS_MINIMUM`, `TIS_MAXIMUM`, `LOSS_REDUCTION`, `TEMPERATURE`,
`TOP_P`, `MICRO_BATCH_SIZE`, `VLLM_GPU_MEMORY_UTILIZATION`, `SAVE_EVERY`, and
`GRADIENT_CHECKPOINTING`. `TARGET_LAYER_INDICES` can restrict adaptation and
shorten the differentiable suffix. Set `PROFILE_MEMORY=1` to write phase-level CUDA,
NVML, Linux unified-memory, exact tensor-inventory, and saved-autograd-tensor
measurements to `memory_profile.json`. The Python entrypoint exposes the same
settings as command-line flags.

Use `MODEL_REVISION`, `ROLLOUT_REVISION`, and `DATASET_REVISION` for immutable
runs. `DATASET_SPLIT=train[:-512]` reserves the final 512 GSM8K training rows
for canary selection without touching the official test set.

## Nemotron 3.5 Lightning: BF16 learner + NVFP4 rollouts

The Nemotron path uses the instruction-tuned BF16 checkpoint for exact
gradients and the NVFP4 checkpoint already used by the Spark's vLLM service
for generation. It adapts only the six attention blocks' Q/K/V/O projections
(24 matrices). Those projections are explicitly excluded from quantization in
the released NVFP4 config. `NUM_GROUPS=13` balances those 24 updates across
exactly 13 shared scalar parameters.

After every optimizer step, the code algebraically converts each TinyLoRA
update into standard rank-2 A/B tensors and hot-loads them as a vLLM LoRA.
This conversion is an inference transport only: optimization remains the
paper-style SVD/random-projection method in `tinylora_rl/adapters.py`.

Run the one-step GSM8K proof:

```bash
./run_nemotron35_tinylora_rl.sh
```

The wrapper stops `nemotron35_lightning_vllm`, runs the experiment, and
restores the service (including an HTTP health check) on success or failure.
It defaults to short sequences and one update so both the 63.16 GB BF16 learner
and the NVFP4 vLLM engine fit in the Spark's unified memory. Every setting can
be overridden, for example:

```bash
STEPS=20 PROMPTS_PER_STEP=4 GENERATIONS=8 \
MAX_COMPLETION_LENGTH=256 MAX_MODEL_LENGTH=640 \
OUTPUT_DIR=/workspace/outputs/nemotron35-tinylora-20step \
./run_nemotron35_tinylora_rl.sh
```

Profile that proof run with a fresh output directory:

```bash
PROFILE_MEMORY=1 \
OUTPUT_DIR=/workspace/outputs/nemotron35-tinylora-memory-profile \
./run_nemotron35_tinylora_rl.sh
```

For the proposed later-four-block depth canary, use
`TARGET_LAYER_INDICES=19,26,33,42`; the same rank-2 factor cache can be reused.

The measured September 2026 run and a Song Han-style live-tensor analysis are
documented in [NEMOTRON35_RLVR_MEMORY_ANALYSIS.md](NEMOTRON35_RLVR_MEMORY_ANALYSIS.md).

This is a mixed-precision actor/learner experiment: NVFP4 rollouts and BF16
learner logits are not numerically identical. The trainer records and clips
token-level importance ratios (`TIS_MODE=token_clip`) so that drift is visible
and bounded. Treat the one-step run as an execution proof; scale sequence
length and PPO epochs only after inspecting `rollout_ratio_*`,
`tis_truncated_fraction`, clip fraction, and memory headroom in `metrics.jsonl`.

Each run writes `metrics.jsonl`, `run_manifest.json`, and
`final_adapter/{adapter.safetensors,adapter_config.json}`. Load a checkpoint
onto an identical base model with `tinylora_rl.load_tinylora`.

## Effectiveness evaluation and adapter registry

Training reward and a nonzero gradient only prove that the loop executes. Use
`evaluate_gsm8k_adapters.py` for a paired base-versus-adapter evaluation. It
reports strict and flexible exact match, wrong-to-right/right-to-wrong flips,
a paired bootstrap interval, exact McNemar p-value, format/no-answer rates,
clipping, and completion length. Base and adapters run in the same vLLM
process on identical prompts. For very small updates, include a zero-valued
PEFT adapter and pass `--comparison-baseline zero`: this controls for output
changes caused solely by switching from vLLM's raw-base path to its LoRA
kernel. The result retains both raw-base comparisons and the guarded
trained-versus-zero comparisons. Evaluation files are not overwritten unless
`--overwrite` is explicitly supplied.

For the four-candidate Lightning canary, `run_nemotron35_tinylora_eval.sh`
checks that the sweep completed, derives an exact zero control from the same
frozen factors, stops/restores the serving container, and evaluates all
candidates on the reserved `train[-512:]` rows.

After the 128-row Lightning selection screen in
`screen-consumed128-max1024.json`, use
`run_nemotron35_tinylora_confirm_eval.sh` for the confirmatory pass. It fixes
the sample to the untouched `train[-384:]` rows, requires exactly 384 rows,
and keeps the screen's 1,024-token generation / 1,280-token model limits. The
screen artifact is validated and content-hashed into the result, and the
evaluator refuses to run if any question hash overlaps. Zero and zero-repeat
are replaced by a predeclared ABBA repeat: `zero_a`, `trained_a`, `trained_b`,
`zero_b`. Both trained arms use the selected `lr1e-4_s0p5` artifact and both
zero arms use the exact same zero artifact. The selection record states that
this candidate had the best trained strict score in the screen (99/128), was
+2/128 versus the mean of the two zero repeats, and retains the conservative
0.5 deployment scale.

The four arms run sequentially through one unloaded/reloaded LoRA slot with
BF16 LoRA runtime, eager execution, `VLLM_ENABLE_V1_MULTIPROCESSING=0`, and
an explicitly recorded `VLLM_BATCH_INVARIANT=0` because the Lightning Mamba
backend in this image rejects vLLM's batch-invariant mode. The one predeclared
efficacy contrast is the mean
row effect `((trained_a + trained_b) - (zero_a + zero_b)) / 2`; its 95% CI
uses 10,000 row-cluster bootstrap draws. Strict correctness is primary and
flexible correctness is secondary. The gate requires a positive strict mean,
a positive strict CI lower bound, and positive estimates in both repeats.
Ordinary pairwise comparisons remain diagnostics, but there is no Holm family
because the ABBA average is the single hypothesis.

New evaluation processes also snapshot the evaluator source at `main()` entry
and embed its path-independent SHA-256, byte size, provenance-schema version,
and invocation arguments in the output. Runs that started before this feature
do not retroactively acquire a source attestation.

```bash
PREFLIGHT_ONLY=1 ./run_nemotron35_tinylora_confirm_eval.sh
./run_nemotron35_tinylora_confirm_eval.sh
```

For a completed 64-step Qwen2.5-7B canary,
`run_qwen25_7b_tinylora_eval.sh` validates the pinned training protocol and
factor cache, exports checkpoints 16/32/48/64, proves that checkpoint 64 and
`final_adapter` are duplicates, and evaluates the four distinct checkpoints
on all 1,319 official GSM8K test examples. It uses the VERL prompt and strict
scorer with greedy decoding, an exact zero-LoRA baseline, and a second request
ID for that same zero artifact (`zero_repeat`). Only the four learned
checkpoints enter the Holm family; raw base and zero-repeat are diagnostics.
The script refuses competing GPU owners and restores the serving container.

Set the completed output directory explicitly and preflight before launching:

```bash
RUN_SLUG=qwen2.5-7b-tinylora13-verl-strict-canary-s42-TIMESTAMP \
  PREFLIGHT_ONLY=1 ./run_qwen25_7b_tinylora_eval.sh
RUN_SLUG=qwen2.5-7b-tinylora13-verl-strict-canary-s42-TIMESTAMP \
  ./run_qwen25_7b_tinylora_eval.sh
```

After evaluation, `run_qwen25_7b_tinylora_promote.sh` performs an independent
CPU-only validation and promotes an exact-zero control plus the four native +
PEFT checkpoint bundles. Run it inside the research image. Its labels contain
`canary` deliberately: this 64-step run is not presented as a full TinyLoRA
paper replication. `PREFLIGHT_ONLY=1` computes the exact content IDs and
checks collisions without mutating the registry.

```bash
docker run --rm -v /home/nimitz/projects/RLtests:/workspace -w /workspace \
  -e RUN_SLUG=qwen2.5-7b-tinylora13-verl-strict-canary-s42-TIMESTAMP \
  -e PREFLIGHT_ONLY=1 spark-vllm-tinylora:0.1 \
  bash /workspace/run_qwen25_7b_tinylora_promote.sh
```

After the selection screen and disjoint confirmation complete,
`run_nemotron35_tinylora_promote.sh` performs a CPU-only provenance, artifact,
and registry-collision preflight, then promotes the zero control plus all four
NVFP4 PEFT adapters and four separately labelled BF16 native checkpoints. Run
it inside the research image; both evaluation artifacts are attached under
distinct names, and `PREFLIGHT_ONLY=1` leaves the registry unchanged.

The interrupted 2026-09-15 Lightning canary has a separate, run-specific
recovery path: `run_nemotron35_tinylora_recovered_eval.sh`. It accepts only the
hash-pinned two-step `lr1e-5_s1` checkpoint and one-step `lr5e-5_s1` rollout
export, records the external SIGINT and discarded step-2 rollouts in an
additive `recovery_manifest.json`, and never rewrites the original sweep
status. It evaluates the zero control twice (`zero` and `zero_repeat`) through
distinct LoRA IDs; paired flips for `zero_repeat` quantify evaluator/kernel
repeatability while `zero` remains the effectiveness baseline. Run with
`PREFLIGHT_ONLY=1` to execute every artifact/provenance check without starting
the GPU evaluator.

Adapters can be promoted into the local content-addressed registry:

```bash
python3 -m tinylora_rl.registry --registry /workspace/adapter_registry \
  add /workspace/outputs/RUN --label qwen2.5-7b/gsm8k/LABEL \
  --export-peft --base-revision MODEL_COMMIT

python3 -m tinylora_rl.registry --registry /workspace/adapter_registry list
python3 -m tinylora_rl.registry --registry /workspace/adapter_registry verify LABEL
python3 -m tinylora_rl.registry --registry /workspace/adapter_registry \
  path LABEL --format peft
```

Bundles are immutable and hash-verified; human labels and evaluation records
are separate references. Generated registry contents are local and ignored by
Git. See `adapter_registry/README.md` for the layout and evaluation attachment
command.

Restore the serving container when training is finished:

```bash
docker start nemotron35_lightning_vllm
curl -f http://127.0.0.1:30000/health
```

## Layout

- `tinylora_rl/adapters.py`: SVD adapters, parameter tying, save/load.
- `tinylora_rl/objectives.py`: GRPO, completion log-probs, and TIS.
- `tinylora_rl/rollout.py`: vLLM sampling and merged-weight synchronization.
- `run_nemotron35_tinylora_rl.sh`: managed BF16/NVFP4 Spark experiment.
- `run_nemotron35_tinylora_eval.sh`: guarded held-out evaluation with a zero-LoRA control.
- `run_nemotron35_tinylora_recovered_eval.sh`: hash-pinned interrupted-sweep recovery/evaluation.
- `run_qwen25_7b_tinylora_canary.sh`: pinned 13-BF16-scalar paper-style canary.
- `tinylora_rl/rewards.py`: deterministic GSM8K verifier.
- `tinylora_rl/trainer.py`: raw PyTorch training loop.
- `tinylora_rl/profiling.py`: unified-memory and live-tensor accounting.
- `tinylora_rl/prompts.py`: shared concise and VERL-style GSM8K prompts.
- `tinylora_rl/registry.py`: immutable local adapter/evaluation registry.
- `train_tinylora_rl.py`: executable experiment entrypoint.

The older `train_gsm8k_grpo.py` harness remains only as a historical TRL
baseline; it is not used by this implementation.
