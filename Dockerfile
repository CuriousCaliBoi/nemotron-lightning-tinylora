FROM vllm/vllm-openai:v0.27.1

# Keep the known-good Spark CUDA/PyTorch/vLLM stack intact and add only the
# libraries that own the optimizer, LoRA adapters, and training dataset.
RUN python3 -m pip install --no-cache-dir \
    trl==1.12.0 \
    peft==0.20.0

WORKDIR /workspace
ENTRYPOINT []
CMD ["bash"]
