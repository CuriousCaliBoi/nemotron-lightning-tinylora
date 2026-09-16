"""Adapter loading and spectral analysis for LoRA-style updates.

The intruder-dimension calculation follows the implementation released with
"LoRA vs Full Fine-tuning: An Illusion of Equivalence": for a base weight and
its tuned counterpart, compute exact thin SVDs, compare the tuned left
singular vectors with every base left singular vector, and count a top-k
vector as an intruder when its largest absolute cosine similarity is below a
threshold.

This module deliberately has no dependency on PEFT.  It can read both the
custom TinyLoRA checkpoints produced by this repository and standard PEFT
LoRA safetensors checkpoints.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn


_LORA_A_RE = re.compile(r"\.lora_A(?:\.[^.]+)?\.weight$")
_LORA_B_RE = re.compile(r"\.lora_B(?:\.[^.]+)?\.weight$")


@dataclass(frozen=True)
class LowRankUpdate:
    """A named update represented as ``scaling * left @ right``."""

    name: str
    left: Tensor
    right: Tensor
    scaling: float = 1.0
    source: str = "unknown"

    def __post_init__(self) -> None:
        if self.left.ndim != 2 or self.right.ndim != 2:
            raise ValueError("low-rank update factors must both be matrices")
        if self.left.shape[1] != self.right.shape[0]:
            raise ValueError(
                f"incompatible update factors for {self.name}: "
                f"{tuple(self.left.shape)} and {tuple(self.right.shape)}"
            )

    @property
    def rank(self) -> int:
        return int(self.right.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.left.shape[0]), int(self.right.shape[1])

    def materialize(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        left = self.left.to(device=device, dtype=dtype)
        right = self.right.to(device=device, dtype=dtype)
        return (left @ right).mul_(self.scaling)


@dataclass(frozen=True)
class IntruderResult:
    """Spectral comparison for one weight matrix."""

    count: int
    examined: int
    threshold: float
    max_similarities: tuple[float, ...]
    tuned_singular_values: tuple[float, ...]
    base_singular_values: tuple[float, ...]
    update_frobenius_ratio: float

    @property
    def fraction(self) -> float:
        return self.count / self.examined if self.examined else 0.0


@dataclass(frozen=True)
class SpectralReference:
    """Reusable exact SVD state for one base matrix."""

    weight: Tensor
    left_singular_vectors: Tensor
    singular_values: Tensor
    frobenius_norm: Tensor


def _normalise_peft_name(name: str) -> str:
    """Convert a PEFT state-dict prefix to a Hugging Face module name."""

    for prefix in ("base_model.model.", "base_model."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _pattern_value(pattern: dict[str, object], name: str, default: float) -> float:
    matches = [key for key in pattern if name == key or name.endswith(f".{key}")]
    if not matches:
        return default
    key = max(matches, key=len)
    return float(pattern[key])


def _load_tinylora_updates(adapter_dir: Path, device: str) -> list[LowRankUpdate]:
    metadata = json.loads((adapter_dir / "adapter_config.json").read_text())
    modules = metadata.get("modules")
    if not isinstance(modules, list) or not modules:
        raise ValueError(f"{adapter_dir} does not contain TinyLoRA module metadata")
    tensors = load_file(str(adapter_dir / "adapter.safetensors"), device=device)
    bank = tensors["bank.v"].float()
    updates: list[LowRankUpdate] = []
    for index, module in enumerate(modules):
        prefix = f"layers.{index}"
        group_id = int(module["group_id"])
        projection = tensors[f"{prefix}.projection"].float()
        if group_id < 0 or group_id >= bank.shape[0]:
            raise ValueError(f"invalid TinyLoRA group {group_id} for {module['name']}")
        middle = torch.einsum("u,urs->rs", bank[group_id], projection)
        # Folding the small middle matrix into the left factor preserves the
        # low-rank representation and avoids materialising a full delta here.
        left = tensors[f"{prefix}.left"].float() @ middle
        right = tensors[f"{prefix}.right"].float()
        updates.append(
            LowRankUpdate(
                name=str(module["name"]),
                left=left,
                right=right,
                scaling=float(module.get("scaling", 1.0)),
                source="tinylora",
            )
        )
    return updates


def _load_peft_updates(adapter_dir: Path, device: str) -> list[LowRankUpdate]:
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    if bool(config.get("use_dora", False)):
        raise NotImplementedError("DoRA magnitude vectors are not supported")
    if bool(config.get("fan_in_fan_out", False)):
        raise NotImplementedError("fan_in_fan_out LoRA checkpoints are not supported")
    if config.get("lora_bias") not in (None, False):
        raise NotImplementedError("LoRA bias updates are not supported")
    tensor_path = adapter_dir / "adapter_model.safetensors"
    if not tensor_path.exists():
        raise FileNotFoundError(f"missing {tensor_path}")
    tensors = load_file(str(tensor_path), device=device)
    rank_pattern = config.get("rank_pattern") or {}
    alpha_pattern = config.get("alpha_pattern") or {}
    default_rank = float(config.get("r", 0))
    default_alpha = float(config.get("lora_alpha", default_rank))
    use_rslora = bool(config.get("use_rslora", False))

    updates: list[LowRankUpdate] = []
    for a_key in sorted(key for key in tensors if _LORA_A_RE.search(key)):
        prefix = _LORA_A_RE.sub("", a_key)
        b_candidates = [
            key for key in tensors if _LORA_B_RE.sub("", key) == prefix and _LORA_B_RE.search(key)
        ]
        if len(b_candidates) != 1:
            raise ValueError(f"expected one LoRA B tensor for {a_key}, found {b_candidates}")
        name = _normalise_peft_name(prefix)
        right = tensors[a_key].float()
        left = tensors[b_candidates[0]].float()
        observed_rank = float(right.shape[0])
        configured_rank = _pattern_value(rank_pattern, name, default_rank or observed_rank)
        if int(configured_rank) != int(observed_rank):
            raise ValueError(
                f"rank metadata for {name} is {configured_rank}, tensor rank is {observed_rank}"
            )
        alpha = _pattern_value(alpha_pattern, name, default_alpha or observed_rank)
        denominator = math.sqrt(observed_rank) if use_rslora else observed_rank
        updates.append(
            LowRankUpdate(
                name=name,
                left=left,
                right=right,
                scaling=alpha / denominator,
                source="peft",
            )
        )
    if not updates:
        raise ValueError(f"no LoRA A/B tensors found in {tensor_path}")
    return updates


def load_adapter_updates(
    adapter_dir: str | Path,
    *,
    device: str = "cpu",
) -> list[LowRankUpdate]:
    """Load low-rank updates from a TinyLoRA or standard PEFT checkpoint."""

    path = Path(adapter_dir)
    if (path / "adapter.safetensors").exists():
        return _load_tinylora_updates(path, device)
    if (path / "adapter_model.safetensors").exists():
        return _load_peft_updates(path, device)
    raise FileNotFoundError(
        f"{path} contains neither adapter.safetensors nor adapter_model.safetensors"
    )


@torch.no_grad()
def apply_updates_(
    model: nn.Module,
    updates: Iterable[LowRankUpdate],
    *,
    multiplier: float = 1.0,
) -> int:
    """Merge scaled updates into a model in place and return the number applied."""

    applied = 0
    for update in updates:
        module = model.get_submodule(update.name)
        weight = getattr(module, "weight", None)
        if not isinstance(weight, Tensor):
            raise TypeError(f"{update.name} does not expose a weight tensor")
        if tuple(weight.shape) != update.shape:
            raise ValueError(
                f"shape mismatch for {update.name}: model {tuple(weight.shape)}, "
                f"adapter {update.shape}"
            )
        weight.add_(
            update.materialize(device=weight.device, dtype=weight.dtype),
            alpha=multiplier,
        )
        applied += 1
    return applied


@torch.no_grad()
def prepare_spectral_reference(
    base_weight: Tensor,
    *,
    device: torch.device | str | None = None,
) -> SpectralReference:
    """Compute the exact thin SVD state reusable across tuned candidates."""

    if base_weight.ndim != 2:
        raise ValueError("intruder analysis requires a matrix")
    base = base_weight.to(device=device, dtype=torch.float32)
    base_u, base_s, _ = torch.linalg.svd(base, full_matrices=False)
    return SpectralReference(
        weight=base,
        left_singular_vectors=base_u,
        singular_values=base_s,
        frobenius_norm=torch.linalg.vector_norm(base),
    )


@torch.no_grad()
def find_intruder_dimensions_from_reference(
    reference: SpectralReference,
    tuned_weight: Tensor,
    *,
    threshold: float = 0.5,
    top_k: int = 10,
) -> IntruderResult:
    """Compare a tuned matrix against a precomputed exact base SVD."""

    if tuned_weight.ndim != 2 or tuned_weight.shape != reference.weight.shape:
        raise ValueError("base and tuned matrices must have the same two-dimensional shape")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")

    tuned = tuned_weight.to(device=reference.weight.device, dtype=torch.float32)
    tuned_u, tuned_s, _ = torch.linalg.svd(tuned, full_matrices=False)
    examined = min(top_k, tuned_u.shape[1])
    similarities = (
        tuned_u[:, :examined].transpose(0, 1) @ reference.left_singular_vectors
    ).abs()
    maxima = similarities.max(dim=1).values
    count = int((maxima < threshold).sum().item())
    update_norm = torch.linalg.vector_norm(tuned - reference.weight)
    ratio = float(
        (
            update_norm
            / reference.frobenius_norm.clamp_min(torch.finfo(reference.weight.dtype).tiny)
        ).item()
    )
    return IntruderResult(
        count=count,
        examined=examined,
        threshold=float(threshold),
        max_similarities=tuple(float(value) for value in maxima.cpu()),
        tuned_singular_values=tuple(float(value) for value in tuned_s[:examined].cpu()),
        base_singular_values=tuple(
            float(value) for value in reference.singular_values[:examined].cpu()
        ),
        update_frobenius_ratio=ratio,
    )


@torch.no_grad()
def find_intruder_dimensions(
    base_weight: Tensor,
    tuned_weight: Tensor,
    *,
    threshold: float = 0.5,
    top_k: int = 10,
    device: torch.device | str | None = None,
) -> IntruderResult:
    """Apply the paper's exact per-matrix intruder-dimension definition."""

    reference = prepare_spectral_reference(base_weight, device=device)
    return find_intruder_dimensions_from_reference(
        reference,
        tuned_weight,
        threshold=threshold,
        top_k=top_k,
    )


def export_peft_adapter(
    updates: Iterable[LowRankUpdate],
    output_dir: str | Path,
    *,
    base_model_name: str,
    tensor_dtype: torch.dtype | None = None,
) -> Path:
    """Export arbitrary low-rank updates as an inference-only PEFT adapter."""

    update_list = list(updates)
    if not update_list:
        raise ValueError("cannot export an empty adapter")
    ranks = {update.rank for update in update_list}
    if len(ranks) != 1:
        raise ValueError(f"PEFT export requires one rank, found {sorted(ranks)}")
    if tensor_dtype not in (None, torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            "PEFT export tensor_dtype must be float16, bfloat16, float32, or None"
        )
    rank = ranks.pop()
    tensors: dict[str, Tensor] = {}
    target_modules: set[str] = set()
    for update in update_list:
        prefix = f"base_model.model.{update.name}"
        right = update.right.detach().cpu()
        left = update.left.detach().cpu() * update.scaling
        if tensor_dtype is not None:
            right = right.to(dtype=tensor_dtype)
            left = left.to(dtype=tensor_dtype)
        tensors[f"{prefix}.lora_A.weight"] = right.contiguous()
        tensors[f"{prefix}.lora_B.weight"] = left.contiguous()
        target_modules.add(update.name.rsplit(".", 1)[-1])

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output / "adapter_model.safetensors"))
    config = {
        "base_model_name_or_path": base_model_name,
        "bias": "none",
        "inference_mode": True,
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": rank,
        "revision": None,
        "target_modules": sorted(target_modules),
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }
    (output / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    return output
