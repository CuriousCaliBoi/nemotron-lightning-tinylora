"""Paper-faithful TinyLoRA layers implemented directly in PyTorch.

For a frozen linear weight W, TinyLoRA learns

    delta_W = U diag(S) (sum_i v_i P_i) Vh

where U, S, and Vh are a rank-r SVD of W, P is a fixed random projection,
and only the small vector v is trainable. Multiple linear layers may share v.
"""

from __future__ import annotations

import json
import math
import weakref
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Literal

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.nn import functional as F


DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class TinyLoRAConfig:
    rank: int = 2
    projection_dim: int = 1
    modules_per_group: int = 16
    # When set, distribute all target modules over exactly this many groups.
    # This is useful when the paper's parameter budget matters more than a
    # fixed number of modules per group (for example, 24 projections / 13 v's).
    num_groups: int | None = None
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES
    target_layer_indices: tuple[int, ...] | None = None
    grouping: Literal["tiled", "structured"] = "tiled"
    projection_seed: int = 42
    projection_std: float | None = None
    scaling: float = 1.0
    parameter_dtype: Literal["float32", "bfloat16"] = "float32"
    svd_niter: int = 2

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must be positive")
        if self.projection_dim < 1:
            raise ValueError("projection_dim must be positive")
        if self.modules_per_group < 1:
            raise ValueError("modules_per_group must be positive")
        if self.num_groups is not None and self.num_groups < 1:
            raise ValueError("num_groups must be positive when provided")
        if not self.target_modules:
            raise ValueError("target_modules cannot be empty")
        if self.target_layer_indices is not None:
            if not self.target_layer_indices:
                raise ValueError("target_layer_indices cannot be empty when provided")
            if any(index < 0 for index in self.target_layer_indices):
                raise ValueError("target_layer_indices must be non-negative")
            if len(set(self.target_layer_indices)) != len(self.target_layer_indices):
                raise ValueError("target_layer_indices cannot contain duplicates")


class TinyLoRAParameterBank(nn.Module):
    """The complete trainable state of a TinyLoRA adapter."""

    def __init__(self, num_groups: int, projection_dim: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.projection_dim = projection_dim
        # A zero initialization preserves the base policy exactly while retaining
        # nonzero gradients through the fixed projection.
        self.v = nn.Parameter(torch.zeros(num_groups, projection_dim, dtype=dtype))

    def extra_repr(self) -> str:
        return f"groups={self.num_groups}, projection_dim={self.projection_dim}"


class TinyLoRALinear(nn.Module):
    """A frozen linear layer with a TinyLoRA residual update."""

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        left: Tensor,
        right: Tensor,
        projection: Tensor,
        bank: TinyLoRAParameterBank,
        group_id: int,
        original_name: str,
        scaling: float = 1.0,
    ) -> None:
        super().__init__()
        if left.ndim != 2 or right.ndim != 2 or projection.ndim != 3:
            raise ValueError("left/right/projection must have ranks 2/2/3")
        rank = left.shape[1]
        expected = (projection.shape[0], rank, rank)
        if tuple(projection.shape) != expected:
            raise ValueError(f"projection has shape {tuple(projection.shape)}, expected {expected}")
        if right.shape[0] != rank:
            raise ValueError("left and right ranks differ")
        if left.shape[0] != base_layer.out_features or right.shape[1] != base_layer.in_features:
            raise ValueError("SVD factors do not match base layer")
        if group_id < 0 or group_id >= bank.num_groups:
            raise ValueError("invalid parameter-bank group")

        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)
        self.register_buffer("left", left.contiguous(), persistent=True)
        self.register_buffer("right", right.contiguous(), persistent=True)
        self.register_buffer("projection", projection.contiguous(), persistent=True)
        # Avoid registering the same bank as a child of every wrapped layer.
        object.__setattr__(self, "_bank_ref", weakref.ref(bank))
        self.group_id = group_id
        self.original_name = original_name
        self.scaling = float(scaling)

    @property
    def bank(self) -> TinyLoRAParameterBank:
        bank = self._bank_ref()
        if bank is None:
            raise RuntimeError("TinyLoRA parameter bank was destroyed")
        return bank

    def middle(self) -> Tensor:
        v = self.bank.v[self.group_id].to(dtype=self.projection.dtype)
        return torch.einsum("u,urs->rs", v, self.projection)

    def delta_weight(self) -> Tensor:
        return (self.left @ self.middle() @ self.right) * self.scaling

    def effective_weight(self) -> Tensor:
        return self.base_layer.weight + self.delta_weight().to(self.base_layer.weight.dtype)

    def forward(self, inputs: Tensor) -> Tensor:
        result = self.base_layer(inputs)
        adapter_inputs = F.linear(inputs.to(self.right.dtype), self.right)
        adapter_inputs = F.linear(adapter_inputs, self.middle())
        adapter_output = F.linear(adapter_inputs, self.left)
        return result + adapter_output.to(result.dtype) * self.scaling


def _get_submodule_parent(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parent_name, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    return parent, child_name


def _target_linears(model: nn.Module, config: TinyLoRAConfig) -> list[tuple[str, nn.Linear]]:
    selected_layers = (
        set(config.target_layer_indices)
        if config.target_layer_indices is not None
        else None
    )

    def layer_is_selected(name: str) -> bool:
        if selected_layers is None:
            return True
        fields = name.split(".")
        for position, field in enumerate(fields[:-1]):
            if field == "layers" and fields[position + 1].isdigit():
                return int(fields[position + 1]) in selected_layers
        return False

    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and name.rsplit(".", 1)[-1] in config.target_modules
        and layer_is_selected(name)
    ]
    if config.grouping == "structured":
        target_order = {name: index for index, name in enumerate(config.target_modules)}
        targets.sort(key=lambda item: (target_order[item[0].rsplit(".", 1)[-1]], item[0]))
    if not targets:
        raise ValueError(f"no linear modules matched {config.target_modules}")
    return targets


def _parameter_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[name]


@torch.no_grad()
def _truncated_svd(weight: Tensor, rank: int, niter: int, seed: int) -> tuple[Tensor, Tensor]:
    """Return U*Sigma and Vh using randomized truncated SVD."""
    if rank > min(weight.shape):
        raise ValueError(f"rank {rank} exceeds matrix shape {tuple(weight.shape)}")
    device = weight.device
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        matrix = weight.float()
        u, s, v = torch.svd_lowrank(matrix, q=rank, niter=niter)
        left = u * s.unsqueeze(0)
        right = v.transpose(0, 1)
    return left.to(dtype=weight.dtype), right.to(dtype=weight.dtype)


def apply_tinylora(
    model: nn.Module,
    config: TinyLoRAConfig,
    *,
    factor_cache: str | Path | None = None,
) -> nn.Module:
    """Freeze ``model`` and replace matching linears with TinyLoRA wrappers.

    The returned object is the same model instance. Its only trainable tensor is
    ``model.tinylora_bank.v``.
    """
    if hasattr(model, "tinylora_bank"):
        raise ValueError("model already has a TinyLoRA adapter")
    model.requires_grad_(False)
    targets = _target_linears(model, config)
    if config.num_groups is not None and config.num_groups > len(targets):
        raise ValueError(
            f"num_groups ({config.num_groups}) exceeds matched target count ({len(targets)})"
        )
    num_groups = config.num_groups or math.ceil(len(targets) / config.modules_per_group)
    first_weight = targets[0][1].weight
    bank = TinyLoRAParameterBank(
        num_groups,
        config.projection_dim,
        _parameter_dtype(config.parameter_dtype),
    ).to(first_weight.device)
    model.add_module("tinylora_bank", bank)
    setattr(model, "tinylora_config", config)

    cache_path = Path(factor_cache) if factor_cache is not None else None
    cached = load_file(str(cache_path), device=str(first_weight.device)) if cache_path and cache_path.exists() else {}
    new_cache: dict[str, Tensor] = {}
    projection_std = config.projection_std or (1.0 / math.sqrt(config.rank))

    for index, (name, linear) in enumerate(targets):
        key = name.replace(".", "__")
        left_key, right_key = f"{key}.left", f"{key}.right"
        cached_shapes_match = (
            left_key in cached
            and right_key in cached
            and tuple(cached[left_key].shape) == (linear.out_features, config.rank)
            and tuple(cached[right_key].shape) == (config.rank, linear.in_features)
        )
        if cached_shapes_match:
            left = cached[left_key].to(device=linear.weight.device, dtype=linear.weight.dtype)
            right = cached[right_key].to(device=linear.weight.device, dtype=linear.weight.dtype)
        else:
            left, right = _truncated_svd(
                linear.weight,
                rank=config.rank,
                niter=config.svd_niter,
                seed=config.projection_seed + index,
            )
        if cache_path is not None and not cached_shapes_match:
            new_cache[left_key] = left.detach().cpu().contiguous()
            new_cache[right_key] = right.detach().cpu().contiguous()

        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.projection_seed + 1_000_003 * (index + 1))
        projection = torch.randn(
            config.projection_dim,
            config.rank,
            config.rank,
            generator=generator,
            dtype=torch.float32,
        ).mul_(projection_std).to(device=linear.weight.device, dtype=linear.weight.dtype)
        # Explicit group counts use a balanced contiguous assignment.  The
        # first and last target always land in the first and last group and no
        # group is empty when num_groups <= len(targets).
        group_id = (
            (index * num_groups) // len(targets)
            if config.num_groups is not None
            else index // config.modules_per_group
        )
        wrapper = TinyLoRALinear(
            linear,
            left=left,
            right=right,
            projection=projection,
            bank=bank,
            group_id=group_id,
            original_name=name,
            scaling=config.scaling,
        )
        parent, child_name = _get_submodule_parent(model, name)
        setattr(parent, child_name, wrapper)

    if new_cache and cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        combined_cache = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in cached.items()
        }
        combined_cache.update(new_cache)
        save_file(combined_cache, str(cache_path))
    return model


@torch.no_grad()
def export_vllm_lora(
    model: nn.Module,
    output_dir: str | Path,
    *,
    base_model_name: str,
) -> Path:
    """Materialize TinyLoRA deltas as a standard inference-only PEFT LoRA.

    This is only a transport format for vLLM. Training remains TinyLoRA: the
    SVD factors and random projections stay frozen and only ``bank.v`` is
    optimized.  Choosing ``lora_alpha == rank`` makes vLLM apply ``B @ A``
    without an additional scale factor.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    layers = list(iter_tinylora_layers(model))
    if not layers:
        raise ValueError("model has no TinyLoRA layers")

    ranks = {layer.right.shape[0] for _, layer in layers}
    if len(ranks) != 1:
        raise ValueError(f"vLLM export requires a single LoRA rank, found {sorted(ranks)}")
    rank = ranks.pop()
    tensors: dict[str, Tensor] = {}
    target_modules: set[str] = set()
    for _, layer in layers:
        prefix = f"base_model.model.{layer.original_name}"
        # TinyLoRA: left @ middle @ right * scaling.  Standard LoRA: B @ A.
        tensors[f"{prefix}.lora_A.weight"] = layer.right.detach().cpu().contiguous()
        tensors[f"{prefix}.lora_B.weight"] = (
            layer.left @ layer.middle() * layer.scaling
        ).detach().cpu().contiguous()
        target_modules.add(layer.original_name.rsplit(".", 1)[-1])

    save_file(tensors, str(output / "adapter_model.safetensors"))
    peft_config = {
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
    (output / "adapter_config.json").write_text(json.dumps(peft_config, indent=2) + "\n")
    return output


def iter_tinylora_layers(model: nn.Module) -> Iterator[tuple[str, TinyLoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, TinyLoRALinear):
            yield name, module


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


@torch.no_grad()
def save_tinylora(model: nn.Module, output_dir: str | Path) -> None:
    """Save the small adapter and its frozen factors without base-model weights."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config: TinyLoRAConfig = getattr(model, "tinylora_config")
    tensors: dict[str, Tensor] = {"bank.v": model.tinylora_bank.v.detach().cpu().contiguous()}
    module_metadata = []
    for index, (_, layer) in enumerate(iter_tinylora_layers(model)):
        prefix = f"layers.{index}"
        tensors[f"{prefix}.left"] = layer.left.detach().cpu().contiguous()
        tensors[f"{prefix}.right"] = layer.right.detach().cpu().contiguous()
        tensors[f"{prefix}.projection"] = layer.projection.detach().cpu().contiguous()
        module_metadata.append(
            {
                "name": layer.original_name,
                "group_id": layer.group_id,
                "scaling": layer.scaling,
            }
        )
    save_file(tensors, str(output / "adapter.safetensors"))
    metadata = {"config": asdict(config), "modules": module_metadata}
    (output / "adapter_config.json").write_text(json.dumps(metadata, indent=2) + "\n")


@torch.no_grad()
def load_tinylora(model: nn.Module, adapter_dir: str | Path) -> nn.Module:
    """Attach a saved TinyLoRA adapter to the corresponding frozen base model."""
    if hasattr(model, "tinylora_bank"):
        raise ValueError("model already has a TinyLoRA adapter")
    adapter_path = Path(adapter_dir)
    metadata = json.loads((adapter_path / "adapter_config.json").read_text())
    raw_config = dict(metadata["config"])
    raw_config["target_modules"] = tuple(raw_config["target_modules"])
    if raw_config.get("target_layer_indices") is not None:
        raw_config["target_layer_indices"] = tuple(raw_config["target_layer_indices"])
    config = TinyLoRAConfig(**raw_config)
    modules = metadata["modules"]
    if not modules:
        raise ValueError("adapter contains no modules")

    first_layer = model.get_submodule(modules[0]["name"])
    if not isinstance(first_layer, nn.Linear):
        raise TypeError(f"{modules[0]['name']} is not a base nn.Linear")
    tensors = load_file(str(adapter_path / "adapter.safetensors"), device=str(first_layer.weight.device))
    bank_values = tensors["bank.v"]
    if bank_values.ndim != 2 or bank_values.shape[1] != config.projection_dim:
        raise ValueError("saved parameter bank does not match adapter config")

    model.requires_grad_(False)
    bank = TinyLoRAParameterBank(
        num_groups=bank_values.shape[0],
        projection_dim=bank_values.shape[1],
        dtype=_parameter_dtype(config.parameter_dtype),
    ).to(first_layer.weight.device)
    bank.v.copy_(bank_values.to(device=bank.v.device, dtype=bank.v.dtype))
    model.add_module("tinylora_bank", bank)
    setattr(model, "tinylora_config", config)

    seen: set[str] = set()
    for index, module_metadata in enumerate(modules):
        name = module_metadata["name"]
        if name in seen:
            raise ValueError(f"duplicate adapter module: {name}")
        seen.add(name)
        linear = model.get_submodule(name)
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"{name} is not a base nn.Linear")
        prefix = f"layers.{index}"
        wrapper = TinyLoRALinear(
            linear,
            left=tensors[f"{prefix}.left"].to(linear.weight),
            right=tensors[f"{prefix}.right"].to(linear.weight),
            projection=tensors[f"{prefix}.projection"].to(linear.weight),
            bank=bank,
            group_id=int(module_metadata["group_id"]),
            original_name=name,
            scaling=float(module_metadata["scaling"]),
        )
        parent, child_name = _get_submodule_parent(model, name)
        setattr(parent, child_name, wrapper)
    return model
