"""Low-overhead memory accounting for single-device training experiments.

DGX Spark uses unified CPU/GPU memory, so neither CUDA allocator counters nor
host RSS tells the whole story on its own.  This module records both views and
keeps the instrumentation light enough to use during a near-capacity run.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


def _tensor_bytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _storage_key(tensor: Tensor) -> tuple[str, int, int] | None:
    """Identify the allocation retained by a tensor, including tensor views."""
    try:
        storage = tensor.untyped_storage()
        return str(tensor.device), storage.data_ptr(), storage.nbytes()
    except (NotImplementedError, RuntimeError):
        return None


def _iter_tensors(value: Any) -> Iterator[Tensor]:
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item)


def _unique_tensor_bytes(
    tensors: Iterator[Tensor],
) -> tuple[int, Counter[str], Counter[str], int]:
    seen: set[tuple[str, int, int] | tuple[str, int]] = set()
    total = 0
    count = 0
    by_dtype: Counter[str] = Counter()
    by_device: Counter[str] = Counter()
    for tensor in tensors:
        storage = _storage_key(tensor)
        key: tuple[str, int, int] | tuple[str, int]
        key = storage if storage is not None else ("object", id(tensor))
        if key in seen:
            continue
        seen.add(key)
        size = storage[2] if storage is not None else _tensor_bytes(tensor)
        total += size
        count += 1
        by_dtype[str(tensor.dtype)] += size
        by_device[str(tensor.device)] += size
    return total, by_dtype, by_device, count


def module_tensor_inventory(module: nn.Module) -> dict[str, Any]:
    """Return exact logical tensor sizes for a PyTorch module."""
    parameters = list(module.parameters())
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    frozen = [parameter for parameter in parameters if not parameter.requires_grad]
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    buffers = list(module.buffers())

    def summarize(tensors: list[Tensor]) -> dict[str, Any]:
        total, by_dtype, by_device, count = _unique_tensor_bytes(iter(tensors))
        return {
            "bytes": total,
            "tensor_count": count,
            "by_dtype_bytes": dict(sorted(by_dtype.items())),
            "by_device_bytes": dict(sorted(by_device.items())),
        }

    return {
        "parameters": summarize(parameters),
        "trainable_parameters": summarize(trainable),
        "frozen_parameters": summarize(frozen),
        "gradients": summarize(gradients),
        "buffers": summarize(buffers),
        "parameter_elements": sum(parameter.numel() for parameter in parameters),
        "trainable_parameter_elements": sum(parameter.numel() for parameter in trainable),
    }


def optimizer_state_inventory(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    tensors = list(_iter_tensors(optimizer.state))
    total, by_dtype, by_device, count = _unique_tensor_bytes(iter(tensors))
    return {
        "bytes": total,
        "tensor_count": count,
        "by_dtype_bytes": dict(sorted(by_dtype.items())),
        "by_device_bytes": dict(sorted(by_device.items())),
    }


def _read_key_values(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in path.read_text().splitlines():
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            fields = raw.strip().split()
            if not fields:
                continue
            try:
                value = int(fields[0])
            except ValueError:
                continue
            if len(fields) > 1 and fields[1] == "kB":
                value *= 1024
            values[key] = value
    except (FileNotFoundError, PermissionError):
        pass
    return values


def _read_int(path: Path) -> int | None:
    try:
        raw = path.read_text().strip()
        return None if raw == "max" else int(raw)
    except (FileNotFoundError, PermissionError, ValueError):
        return None


def _read_whitespace_key_values(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in path.read_text().splitlines():
            fields = line.split()
            if len(fields) == 2:
                try:
                    values[fields[0]] = int(fields[1])
                except ValueError:
                    continue
    except (FileNotFoundError, PermissionError):
        pass
    return values


class _SavedTensorTracker:
    """Estimate peak non-parameter storage retained for autograd."""

    def __init__(self, excluded: set[tuple[str, int, int]]) -> None:
        self.excluded = excluded
        self.refcounts: Counter[tuple[str, int, int] | tuple[str, int]] = Counter()
        self.current_by_device: Counter[str] = Counter()
        self.peak_by_device: Counter[str] = Counter()
        self.total_pack_calls = 0
        self.total_logical_bytes = 0
        self.non_parameter_logical_bytes = 0
        self.unique_storages: set[tuple[str, int, int] | tuple[str, int]] = set()
        self.lock = threading.Lock()

    def pack(self, tensor: Tensor) -> "_SavedTensorHandle":
        storage = _storage_key(tensor)
        excluded = storage is not None and storage in self.excluded
        key: tuple[str, int, int] | tuple[str, int]
        key = storage if storage is not None else ("object", id(tensor))
        size = storage[2] if storage is not None else _tensor_bytes(tensor)
        device = str(tensor.device)
        with self.lock:
            self.total_pack_calls += 1
            self.total_logical_bytes += _tensor_bytes(tensor)
            if not excluded:
                self.non_parameter_logical_bytes += _tensor_bytes(tensor)
                self.unique_storages.add(key)
                if self.refcounts[key] == 0:
                    self.current_by_device[device] += size
                    self.peak_by_device[device] = max(
                        self.peak_by_device[device], self.current_by_device[device]
                    )
                self.refcounts[key] += 1
        return _SavedTensorHandle(tensor, self, key, size, device, excluded)

    def release(
        self,
        key: tuple[str, int, int] | tuple[str, int],
        size: int,
        device: str,
        excluded: bool,
    ) -> None:
        if excluded:
            return
        with self.lock:
            if self.refcounts[key] <= 1:
                self.refcounts.pop(key, None)
                self.current_by_device[device] -= size
            else:
                self.refcounts[key] -= 1

    def summary(self) -> dict[str, Any]:
        with self.lock:
            return {
                "pack_calls": self.total_pack_calls,
                "logical_bytes_all_pack_calls": self.total_logical_bytes,
                "logical_non_parameter_bytes_all_pack_calls": self.non_parameter_logical_bytes,
                "unique_non_parameter_storages": len(self.unique_storages),
                "measurement_scope": (
                    "autograd-retained non-parameter storage visible outside "
                    "non-reentrant checkpoint recomputation"
                ),
                "estimated_peak_live_non_parameter_storage_by_device": dict(
                    sorted(self.peak_by_device.items())
                ),
            }


class _SavedTensorHandle:
    def __init__(
        self,
        tensor: Tensor,
        tracker: _SavedTensorTracker,
        key: tuple[str, int, int] | tuple[str, int],
        size: int,
        device: str,
        excluded: bool,
    ) -> None:
        self.tensor = tensor
        self.tracker = tracker
        self.key = key
        self.size = size
        self.device = device
        self.excluded = excluded

    def release(self) -> None:
        if self.tracker is not None:
            self.tracker.release(
                self.key,
                self.size,
                self.device,
                self.excluded,
            )
            self.tracker = None  # type: ignore[assignment]

    def __del__(self) -> None:
        self.release()


class UnifiedMemoryProfiler:
    """Aggregate CUDA allocator and Linux unified-memory measurements by phase."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        enabled: bool = False,
        sample_interval: float = 0.20,
    ) -> None:
        if sample_interval <= 0:
            raise ValueError("sample_interval must be positive")
        self.output_path = Path(output_path)
        self.enabled = enabled
        self.sample_interval = sample_interval
        self.data: dict[str, Any] = {
            "schema_version": 1,
            "sample_interval_seconds": sample_interval,
            "marks": [],
            "phases": [],
            "tensor_inventories": [],
            "saved_tensor_profiles": [],
        }
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._active_phase: dict[str, Any] | None = None
        self._nvml: Any | None = None
        self._nvml_handle: Any | None = None
        self._next_nvml_sample_at = 0.0

    @property
    def active(self) -> bool:
        return self.enabled

    def _linux_sample(self) -> dict[str, int | None]:
        memory = _read_key_values(Path("/proc/meminfo"))
        process = _read_key_values(Path("/proc/self/status"))
        cgroup = _read_whitespace_key_values(Path("/sys/fs/cgroup/memory.stat"))
        total = memory.get("MemTotal")
        available = memory.get("MemAvailable")
        return {
            "system_total_bytes": total,
            "system_available_bytes": available,
            "system_free_bytes": memory.get("MemFree"),
            "swap_free_bytes": memory.get("SwapFree"),
            "system_used_bytes": (
                total - available if total is not None and available is not None else None
            ),
            "process_rss_bytes": process.get("VmRSS"),
            "process_peak_rss_bytes": process.get("VmHWM"),
            "cgroup_current_bytes": _read_int(Path("/sys/fs/cgroup/memory.current")),
            "cgroup_peak_bytes": _read_int(Path("/sys/fs/cgroup/memory.peak")),
            "cgroup_swap_current_bytes": _read_int(
                Path("/sys/fs/cgroup/memory.swap.current")
            ),
            "cgroup_anon_bytes": cgroup.get("anon"),
            "cgroup_file_bytes": cgroup.get("file"),
            "cgroup_shmem_bytes": cgroup.get("shmem"),
            "cgroup_pgmajfault": cgroup.get("pgmajfault"),
            "cgroup_pgscan_direct": cgroup.get("pgscan_direct"),
        }

    def _initialize_nvml(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            device_index = torch.cuda.current_device() if torch.cuda.is_available() else 0
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception:  # Import, driver, and pynvml binding errors vary by host.
            self._nvml = None
            self._nvml_handle = None

    def _nvml_sample(self) -> dict[str, Any]:
        if self._nvml is None or self._nvml_handle is None:
            return {
                "nvml_processes": None,
                "nvml_total_process_bytes": None,
                "nvml_self_process_bytes": None,
                "nvml_other_process_bytes": None,
            }
        try:
            unavailable = getattr(self._nvml, "NVML_VALUE_NOT_AVAILABLE", None)
            processes = []
            for process in self._nvml.nvmlDeviceGetComputeRunningProcesses(
                self._nvml_handle
            ):
                raw = getattr(process, "usedGpuMemory", None)
                used = None if raw is None or raw == unavailable else int(raw)
                processes.append({"pid": int(process.pid), "bytes": used})
            total = sum(item["bytes"] or 0 for item in processes)
            own = sum(
                item["bytes"] or 0
                for item in processes
                if item["pid"] == os.getpid()
            )
            return {
                "nvml_processes": processes,
                "nvml_total_process_bytes": total,
                "nvml_self_process_bytes": own,
                "nvml_other_process_bytes": total - own,
            }
        except Exception as exc:  # NVML errors vary across driver bindings.
            return {
                "nvml_processes": None,
                "nvml_total_process_bytes": None,
                "nvml_self_process_bytes": None,
                "nvml_other_process_bytes": None,
                "nvml_error": f"{type(exc).__name__}: {exc}",
            }

    def _cuda_sample(self) -> dict[str, int | list[int] | None]:
        if not torch.cuda.is_available():
            return {
                "cuda_allocated_bytes": None,
                "cuda_reserved_bytes": None,
                "cuda_max_allocated_bytes": None,
                "cuda_max_reserved_bytes": None,
                "cuda_mem_get_info_bytes": None,
            }
        free, total = torch.cuda.mem_get_info()
        return {
            "cuda_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_reserved_bytes": torch.cuda.memory_reserved(),
            "cuda_max_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_max_reserved_bytes": torch.cuda.max_memory_reserved(),
            "cuda_mem_get_info_bytes": [free, total],
        }

    def _sample(self, *, include_cuda: bool, include_nvml: bool = False) -> dict[str, Any]:
        sample: dict[str, Any] = {
            "elapsed_seconds": time.monotonic() - self.data.get("monotonic_start", time.monotonic()),
            **self._linux_sample(),
        }
        if include_cuda:
            sample.update(self._cuda_sample())
        if include_nvml:
            sample.update(self._nvml_sample())
        return sample

    @staticmethod
    def _update_peak(aggregate: dict[str, Any], sample: dict[str, Any]) -> None:
        for key in (
            "system_used_bytes",
            "process_rss_bytes",
            "cgroup_current_bytes",
            "nvml_total_process_bytes",
            "nvml_self_process_bytes",
            "nvml_other_process_bytes",
        ):
            value = sample.get(key)
            if value is not None:
                peak_key = f"peak_{key}"
                aggregate[peak_key] = max(aggregate.get(peak_key, value), value)
        available = sample.get("system_available_bytes")
        if available is not None:
            aggregate["minimum_system_available_bytes"] = min(
                aggregate.get("minimum_system_available_bytes", available), available
            )

    def _sampling_loop(self) -> None:
        while not self._stop.wait(self.sample_interval):
            now = time.monotonic()
            include_nvml = now >= self._next_nvml_sample_at
            if include_nvml:
                self._next_nvml_sample_at = now + 1.0
            sample = self._sample(include_cuda=False, include_nvml=include_nvml)
            with self._lock:
                if self._active_phase is not None:
                    self._update_peak(self._active_phase, sample)

    def _flush(self) -> None:
        if not self.enabled:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self.data)
        payload.pop("monotonic_start", None)
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, self.output_path)

    def start(self) -> None:
        if not self.enabled:
            return
        self.data["started_at"] = datetime.now(timezone.utc).isoformat()
        self.data["monotonic_start"] = time.monotonic()
        self.data["pid"] = os.getpid()
        self.data["accounting_notes"] = [
            "DGX Spark uses unified memory; CUDA, NVML, RSS/cgroup, and host memory views overlap and must not be summed.",
            "cuda_mem_get_info is advisory on Spark because reclaimable DRAM is not fully represented.",
            "Saved-tensor peaks exclude registered parameters/buffers and do not see internals hidden by non-reentrant checkpoint hooks.",
        ]
        self.data["device"] = (
            {
                "name": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
            }
            if torch.cuda.is_available()
            else None
        )
        self._initialize_nvml()
        self._thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._thread.start()
        self.mark("profiler_started")

    def mark(self, name: str, *, synchronize: bool = True, **metadata: Any) -> None:
        if not self.enabled:
            return
        if synchronize and torch.cuda.is_available():
            torch.cuda.synchronize()
        entry = {
            "name": name,
            **self._sample(include_cuda=True, include_nvml=True),
            **metadata,
        }
        self.data["marks"].append(entry)
        self._flush()

    @contextmanager
    def phase(self, name: str, **metadata: Any) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started_at = time.monotonic()
        start = self._sample(include_cuda=True, include_nvml=True)
        aggregate: dict[str, Any] = {}
        self._update_peak(aggregate, start)
        with self._lock:
            if self._active_phase is not None:
                raise RuntimeError("memory profiling phases cannot be nested")
            self._active_phase = aggregate
        error: str | None = None
        try:
            yield
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = self._sample(include_cuda=True, include_nvml=True)
            with self._lock:
                self._update_peak(aggregate, end)
                self._active_phase = None
            phase = {
                "name": name,
                "duration_seconds": time.monotonic() - started_at,
                "start": start,
                "end": end,
                **aggregate,
                **metadata,
            }
            if error is not None:
                phase["error"] = error
            start_allocated = start.get("cuda_allocated_bytes")
            max_allocated = end.get("cuda_max_allocated_bytes")
            if start_allocated is not None and max_allocated is not None:
                phase["cuda_peak_increment_bytes"] = max(0, max_allocated - start_allocated)
            start_reserved = start.get("cuda_reserved_bytes")
            max_reserved = end.get("cuda_max_reserved_bytes")
            if start_reserved is not None and max_reserved is not None:
                phase["cuda_reserved_peak_increment_bytes"] = max(
                    0, max_reserved - start_reserved
                )
            start_used = start.get("system_used_bytes")
            peak_used = aggregate.get("peak_system_used_bytes")
            if start_used is not None and peak_used is not None:
                phase["system_used_peak_increment_bytes"] = max(0, peak_used - start_used)
            self.data["phases"].append(phase)
            self._flush()

    def record_module(self, name: str, module: nn.Module) -> None:
        if not self.enabled:
            return
        self.data["tensor_inventories"].append(
            {"name": name, "module": module_tensor_inventory(module)}
        )
        self._flush()

    def record_training_state(
        self,
        name: str,
        module: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        if not self.enabled:
            return
        self.data["tensor_inventories"].append(
            {
                "name": name,
                "module": module_tensor_inventory(module),
                "optimizer_state": optimizer_state_inventory(optimizer),
            }
        )
        self._flush()

    @contextmanager
    def track_saved_tensors(self, name: str, module: nn.Module) -> Iterator[None]:
        """Profile one representative autograd microbatch.

        Parameter and registered-buffer storage is excluded so the reported
        peak approximates activations and other dynamic tensors retained by
        autograd rather than the frozen model itself.
        """
        if not self.enabled:
            yield
            return
        excluded = {
            key
            for tensor in (*module.parameters(), *module.buffers())
            if (key := _storage_key(tensor)) is not None
        }
        tracker = _SavedTensorTracker(excluded)

        def unpack(handle: _SavedTensorHandle) -> Tensor:
            return handle.tensor

        with torch.autograd.graph.saved_tensors_hooks(tracker.pack, unpack):
            yield
        self.data["saved_tensor_profiles"].append(
            {"name": name, **tracker.summary()}
        )
        self._flush()

    def finish(self, *, status: str = "complete", error: str | None = None) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.sample_interval * 2))
        self.mark("profiler_finished", status=status, error=error)
        self.data["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.data["status"] = status
        if error is not None:
            self.data["error"] = error
        self._flush()
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
