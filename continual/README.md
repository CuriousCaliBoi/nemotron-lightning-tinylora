# Continual learning loop (TinyLoRA on Nemotron 3.5 Lightning)

One process owns a resident BF16 learner and drives the cycle against the
NVFP4 checkpoint served by vLLM:

1. **Sample** `PROMPTS_PER_STEP × GENERATIONS` completions from the served
   adapter over HTTP (`/v1/completions`, token-id prompts, processed logprobs).
2. **Score** each completion with the exact GSM8K verifier (`REWARD_MODE`).
3. **Update** with the unchanged RLtests GRPO trainer (group advantages,
   clipped surrogate, truncated importance sampling) on the 13 TinyLoRA values.
4. **Publish**: save the native adapter and optimizer state, export the rank-2
   PEFT transport, load it into vLLM at runtime, and commit a release.
5. **Record**: rollouts with receipts under `records/`, the commit under
   `history.jsonl`, live state in `status.json`.
6. Every `EVAL_EVERY` steps, evaluate the incumbent greedily on the first
   `EVAL_ROWS` rows of `train[-512:]`, keep a `BEST` pointer, and roll back
   (as a new release) if accuracy falls more than `ROLLBACK_DROP` below best.

The loop always resumes from `HEAD`: the native adapter's 13 values and the
optimizer moments. It never zeroes the adapter. The last 384 rows of the
train split are never read, so the confirmation protocol stays untouched.

## Layout

```
/home/nimitz/cl-state/nemotron35-gsm8k/
  HEAD, BEST                 current and best release ids
  history.jsonl              append-only commits (seed, training, rollback)
  evals.jsonl                periodic evaluation summaries
  releases/r000NNN/
    final_adapter/           native TinyLoRA checkpoint (resume point)
    optimizer.pt             AdamW state
    peft_adapter/            what vLLM serves as lora name cl-r000NNN
    manifest.json            parent, step, metrics, content id
  records/step-NNNNNN.jsonl  rollouts with receipts, rewards, logprobs
  evals/r000NNN.json         per-row evaluation results
  status.json                live phase, step, last metrics
  STOP                       create this file to stop after the current step
```

## Operate

```bash
# serving container with runtime LoRA (stop the previous server first)
continual/serve_lora.sh                       # UTIL=0.30 SPEC=1 by default
# the loop
continual/run_loop.sh                         # env vars override defaults
continual/status.sh                           # pointers, commits, evals
docker logs -f cl-loop                        # JSON log lines
touch /home/nimitz/cl-state/nemotron35-gsm8k/STOP   # graceful stop
```

Restore the original server: stop `nemotron35_lightning_vllm_lora`, then
`docker start nemotron35_lightning_vllm` (same port, original flags).

## Adding a method

The trainer's `_update` is the method. To try another objective, subclass
`TinyLoRAGRPOTrainer` (or write a class with the same `_sample_trajectories`
and `_update` signatures) and point `loop.py` at it; rollouts, scoring,
publication, the chain and evaluation stay as they are.
