"""Continual learning loop for TinyLoRA on the DGX Spark.

The served model is the NVFP4 Nemotron 3.5 Lightning checkpoint in vLLM.
Rollouts are sampled from it over HTTP, scored by the exact GSM8K verifier,
turned into one GRPO update by the RLtests trainer on a resident BF16
learner, exported as a rank-2 LoRA, published into vLLM at runtime, and
recorded as a release in an append-only chain that always resumes from the
incumbent native adapter and its optimizer state.
"""
