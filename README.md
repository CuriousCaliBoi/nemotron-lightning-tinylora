# TinyLoRA RL from scratch on DGX Spark

This is an independent implementation of the core method in
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
MODULES_PER_GROUP=16 TINYLORA_RANK=2 PROJECTION_DIM=1 \
PARAMETER_DTYPE=bfloat16 \
PROMPTS_PER_STEP=16 GENERATIONS=4 \
MAX_COMPLETION_LENGTH=4096 MAX_MODEL_LENGTH=4608 \
SAMPLES=7473 STEPS=1402 \
OUTPUT_DIR=/workspace/outputs/qwen2.5-7b-tinylora-gsm8k \
./run_tinylora_rl.sh
```

The first run computes and caches the truncated SVD factors. Later runs load
them directly. A custom `FACTOR_CACHE=/workspace/cache/name.safetensors` can be
provided; otherwise the filename is derived from the model and rank.

Important research controls are environment variables accepted by
`run_tinylora_rl.sh`: `LEARNING_RATE`, `PPO_EPOCHS`, `CLIP_EPSILON`,
`TIS_MODE`, `TIS_MINIMUM`, `TIS_MAXIMUM`, `LOSS_REDUCTION`, `TEMPERATURE`,
`TOP_P`, `MICRO_BATCH_SIZE`, `VLLM_GPU_MEMORY_UTILIZATION`, `SAVE_EVERY`, and
`GRADIENT_CHECKPOINTING`. The Python entrypoint exposes the same settings as
command-line flags.

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
It defaults to short sequences and one update so both the 65.8 GB BF16 learner
and the NVFP4 vLLM engine fit in the Spark's unified memory. Every setting can
be overridden, for example:

```bash
STEPS=20 PROMPTS_PER_STEP=4 GENERATIONS=8 \
MAX_COMPLETION_LENGTH=256 MAX_MODEL_LENGTH=640 \
OUTPUT_DIR=/workspace/outputs/nemotron35-tinylora-20step \
./run_nemotron35_tinylora_rl.sh
```

This is a mixed-precision actor/learner experiment: NVFP4 rollouts and BF16
learner logits are not numerically identical. The trainer records and clips
token-level importance ratios (`TIS_MODE=token_clip`) so that drift is visible
and bounded. Treat the one-step run as an execution proof; scale sequence
length and PPO epochs only after inspecting `rollout_ratio_*`,
`tis_truncated_fraction`, clip fraction, and memory headroom in `metrics.jsonl`.

Each run writes `metrics.jsonl`, `run_manifest.json`, and
`final_adapter/{adapter.safetensors,adapter_config.json}`. Load a checkpoint
onto an identical base model with `tinylora_rl.load_tinylora`.

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
- `tinylora_rl/rewards.py`: deterministic GSM8K verifier.
- `tinylora_rl/trainer.py`: raw PyTorch training loop.
- `train_tinylora_rl.py`: executable experiment entrypoint.

The older `train_gsm8k_grpo.py` harness remains only as a historical TRL
baseline; it is not used by this implementation.
