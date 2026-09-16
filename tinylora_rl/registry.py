"""A small, local, content-addressed registry for adapter experiments.

Adapter bundles are immutable.  Human-readable labels are JSON references to
those bundles, and evaluations are separate immutable records so that adding a
new measurement never mutates the adapter it describes.

Registry operations copy and hash files without loading model weights.  Only a
requested TinyLoRA-to-PEFT export loads the training dependencies.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


SCHEMA_VERSION = 1
DEFAULT_REGISTRY = "adapter_registry"
_LABEL_SEGMENT = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_PROVENANCE_FILES = (
    "run_manifest.json",
    "metrics.jsonl",
    "memory_profile.json",
    "final_metrics.json",
    "trainer_state.json",
)
_PARENT_PROVENANCE_FILES = (
    "sweep_manifest.json",
    "sweep_results.json",
    "sweep_status.json",
    "recovery_manifest.json",
)


class RegistryError(RuntimeError):
    """A user-facing registry validation or I/O error."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except PermissionError as exc:
        raise RegistryError(_permission_message(path)) from exc
    except FileNotFoundError as exc:
        raise RegistryError(f"missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RegistryError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RegistryError(f"expected a JSON object in {path}")
    return value


def _permission_message(path: Path) -> str:
    return (
        f"cannot read {path}. Container-created safetensors may be owned by root "
        "with mode 0600; run the registry command inside the research container "
        "or make the source file host-readable first"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except PermissionError as exc:
        raise RegistryError(_permission_message(path)) from exc
    return digest.hexdigest()


def _validate_label(label: str) -> str:
    path = PurePosixPath(label)
    if path.is_absolute() or not label or label.endswith("/"):
        raise RegistryError(f"invalid label: {label!r}")
    parts = path.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise RegistryError(f"invalid label: {label!r}")
    if any(_LABEL_SEGMENT.fullmatch(part) is None for part in parts):
        raise RegistryError(
            "labels must be slash-separated lowercase segments containing only "
            "letters, digits, '.', '_' and '-'"
        )
    return "/".join(parts)


def _safe_relative_path(value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RegistryError(f"unsafe artifact path in manifest: {value!r}")
    return Path(*path.parts)


def _ensure_registry(root: Path) -> None:
    for path in (
        root,
        root / "objects" / "sha256",
        root / "refs",
        root / "evaluations" / "sha256",
        root / "evaluation_refs",
        root / ".tmp",
    ):
        path.mkdir(parents=True, exist_ok=True)
        try:
            path.chmod(0o755)
        except PermissionError:
            # Existing shared registries may be writable through group ACLs.
            pass


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True).encode("utf-8"))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o644)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source.open("rb") as input_handle, destination.open("wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
    except PermissionError as exc:
        raise RegistryError(_permission_message(source)) from exc
    destination.chmod(0o644)


def _artifact(path: Path, root: Path, role: str) -> dict[str, object]:
    relative = path.relative_to(root).as_posix()
    return {
        "path": relative,
        "role": role,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _freeze_tree(path: Path) -> None:
    for child in path.rglob("*"):
        if child.is_symlink():
            raise RegistryError(f"registry objects cannot contain symlinks: {child}")
        if child.is_file():
            child.chmod(0o444)
    directories = sorted(
        (child for child in path.rglob("*") if child.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    )
    for directory in directories:
        directory.chmod(0o555)
    path.chmod(0o555)


def _discard_staging(path: Path) -> None:
    """Remove only a registry-owned temporary directory."""

    if not path.exists():
        return
    for child in path.rglob("*"):
        if child.is_dir():
            try:
                child.chmod(0o755)
            except OSError:
                pass
        elif child.is_file():
            try:
                child.chmod(0o644)
            except OSError:
                pass
    try:
        path.chmod(0o755)
    except OSError:
        pass
    shutil.rmtree(path)


def _content_id(manifest: dict[str, Any], id_field: str) -> str:
    content = dict(manifest)
    content.pop(id_field, None)
    return hashlib.sha256(_canonical_json(content)).hexdigest()


def _locate_source(source: Path) -> tuple[Path, Path | None]:
    """Return ``(adapter_dir, run_dir)`` for a run or direct adapter path."""

    source = source.resolve()
    if not source.exists():
        raise RegistryError(f"source does not exist: {source}")
    final_adapter = source / "final_adapter"
    if final_adapter.is_dir():
        return final_adapter, source
    if (source / "adapter_config.json").is_file():
        run_dir = source.parent
        if not any((run_dir / name).is_file() for name in _PROVENANCE_FILES):
            # Derived controls such as ``SWEEP/zero_control`` have no
            # candidate-local run manifest, but their parent is still the
            # completed sweep whose provenance must travel with the object.
            # Treat the adapter directory as the run directory in this one
            # layout so the normal parent-provenance copy below finds those
            # sweep records.
            run_dir = (
                source
                if any((source.parent / name).is_file() for name in _PARENT_PROVENANCE_FILES)
                else None
            )
        return source, run_dir
    raise RegistryError(
        f"{source} is neither a run containing final_adapter nor an adapter directory"
    )


def _adapter_format(path: Path) -> str:
    native = path / "adapter.safetensors"
    peft = path / "adapter_model.safetensors"
    if native.is_file() and peft.is_file():
        raise RegistryError(f"ambiguous adapter directory contains both formats: {path}")
    if native.is_file():
        return "tinylora-v1"
    if peft.is_file():
        return "peft-lora"
    raise RegistryError(f"no adapter weights found in {path}")


def _base_model_from(
    run_manifest: dict[str, Any] | None,
    adapter_config: dict[str, Any],
) -> str | None:
    # The adapter declaration is authoritative for a deployment artifact.  In
    # actor/learner setups, the enclosing run manifest can name a BF16 learner
    # while a PEFT export intentionally targets a quantized rollout checkpoint.
    value = adapter_config.get("base_model_name_or_path")
    if isinstance(value, str) and value:
        return value
    if run_manifest is not None:
        for key in ("base_model", "model"):
            value = run_manifest.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _base_revision_from(run_manifest: dict[str, Any] | None) -> str | None:
    if run_manifest is None:
        return None
    for key in ("base_model_revision", "model_revision", "resolved_revision"):
        value = run_manifest.get(key)
        if isinstance(value, str) and value:
            return value
    model = run_manifest.get("base_model")
    if isinstance(model, dict):
        revision = model.get("revision")
        if isinstance(revision, str) and revision:
            return revision
    return None


def _training_summary(run_dir: Path | None) -> dict[str, Any] | None:
    if run_dir is None:
        return None
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.is_file():
        records: list[dict[str, Any]] = []
        try:
            with metrics_path.open() as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise RegistryError(
                            f"invalid JSON on line {line_number} of {metrics_path}: {exc}"
                        ) from exc
                    if not isinstance(value, dict):
                        raise RegistryError(
                            f"expected a JSON object on line {line_number} of {metrics_path}"
                        )
                    records.append(value)
        except PermissionError as exc:
            raise RegistryError(_permission_message(metrics_path)) from exc
        if not records:
            return {"records": 0}
        summary: dict[str, Any] = {
            "records": len(records),
            "final": _scalar_fields(records[-1]),
        }
        rewards = [
            float(record["reward_mean"])
            for record in records
            if isinstance(record.get("reward_mean"), (int, float))
        ]
        if rewards:
            summary["maximum_reward_mean"] = max(rewards)
        return summary
    final_metrics_path = run_dir / "final_metrics.json"
    if final_metrics_path.is_file():
        return {"records": 1, "final": _scalar_fields(_read_json(final_metrics_path))}
    return None


def _adapter_summary(
    adapter_format: str,
    adapter_config: dict[str, Any],
) -> dict[str, Any]:
    if adapter_format == "tinylora-v1":
        config = adapter_config.get("config")
        modules = adapter_config.get("modules")
        if not isinstance(config, dict) or not isinstance(modules, list) or not modules:
            raise RegistryError("TinyLoRA adapter_config.json needs non-empty config and modules")
        groups = config.get("num_groups")
        if groups is None:
            try:
                groups = max(int(module["group_id"]) for module in modules) + 1
            except (KeyError, TypeError, ValueError) as exc:
                raise RegistryError("invalid TinyLoRA module group metadata") from exc
        projection_dim = int(config.get("projection_dim", 1))
        return {
            "format": adapter_format,
            "rank": int(config.get("rank", 0)),
            "projection_dim": projection_dim,
            "groups": int(groups),
            "trainable_parameters": int(groups) * projection_dim,
            "target_count": len(modules),
            "target_modules": [str(module.get("name", "")) for module in modules],
            "config": config,
        }
    rank = int(adapter_config.get("r", 0))
    return {
        "format": adapter_format,
        "rank": rank,
        "target_modules": adapter_config.get("target_modules", []),
        "config": adapter_config,
    }


def _copy_adapter(
    source: Path,
    destination: Path,
    *,
    expected_format: str,
    artifact_root: Path,
    artifacts: list[dict[str, object]],
) -> None:
    observed = _adapter_format(source)
    if observed != expected_format:
        raise RegistryError(
            f"expected {expected_format} at {source}, found {observed}"
        )
    _read_json(source / "adapter_config.json")
    weight_name = (
        "adapter.safetensors" if observed == "tinylora-v1" else "adapter_model.safetensors"
    )
    prefix = "native" if observed == "tinylora-v1" else "peft"
    for filename, role_suffix in (
        (weight_name, "weights"),
        ("adapter_config.json", "config"),
    ):
        target = destination / filename
        _copy_file(source / filename, target)
        artifacts.append(_artifact(target, artifact_root, f"{prefix}_{role_suffix}"))


def _verify_peft_companion(
    native: Path,
    peft: Path,
    *,
    base_model: str,
) -> None:
    """Prove that a PEFT companion represents the native TinyLoRA deltas.

    A matching filename or base-model declaration is not sufficient: attaching
    an unrelated PEFT directory would make evaluations ambiguous.  Compare the
    effective (scaled) low-rank update for every module in bounded row chunks so
    this check remains practical for large projection matrices.
    """

    native_config = _read_json(native / "adapter_config.json")
    peft_config = _read_json(peft / "adapter_config.json")
    declared_base = peft_config.get("base_model_name_or_path")
    if declared_base != base_model:
        raise RegistryError(
            "PEFT companion base model does not match the registered adapter: "
            f"{declared_base!r} != {base_model!r}"
        )

    tiny_config = native_config.get("config")
    if not isinstance(tiny_config, dict):
        raise RegistryError("TinyLoRA adapter_config.json is missing config metadata")
    try:
        native_rank = int(tiny_config["rank"])
        peft_rank = int(peft_config["r"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RegistryError("native and PEFT adapters must declare an integer rank") from exc
    if native_rank != peft_rank:
        raise RegistryError(
            f"PEFT companion rank {peft_rank} does not match native rank {native_rank}"
        )

    try:
        import torch

        from .adapter_analysis import load_adapter_updates

        native_updates = load_adapter_updates(native)
        peft_updates = load_adapter_updates(peft)
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RegistryError(f"could not validate PEFT companion: {exc}") from exc

    def by_name(updates: Sequence[Any], format_name: str) -> dict[str, Any]:
        indexed: dict[str, Any] = {}
        for update in updates:
            if update.name in indexed:
                raise RegistryError(
                    f"{format_name} adapter contains duplicate module {update.name!r}"
                )
            indexed[update.name] = update
        return indexed

    native_by_name = by_name(native_updates, "native")
    peft_by_name = by_name(peft_updates, "PEFT")
    if native_by_name.keys() != peft_by_name.keys():
        missing = sorted(native_by_name.keys() - peft_by_name.keys())
        extra = sorted(peft_by_name.keys() - native_by_name.keys())
        raise RegistryError(
            "PEFT companion module set does not match native adapter; "
            f"missing={missing}, extra={extra}"
        )

    # Keep each pair of materialized chunks near 32 MiB in float32.  The
    # factors themselves are small compared with the base model and are loaded
    # once by safetensors.
    elements_per_chunk = 4 * 1024 * 1024
    for name in sorted(native_by_name):
        expected = native_by_name[name]
        observed = peft_by_name[name]
        if expected.rank != native_rank or observed.rank != native_rank:
            raise RegistryError(
                f"PEFT companion rank mismatch for {name}: "
                f"native={expected.rank}, PEFT={observed.rank}, declared={native_rank}"
            )
        if expected.shape != observed.shape:
            raise RegistryError(
                f"PEFT companion shape mismatch for {name}: "
                f"native={expected.shape}, PEFT={observed.shape}"
            )

        output_columns = max(1, expected.shape[1])
        rows_per_chunk = max(1, elements_per_chunk // output_columns)
        expected_right = expected.right.float()
        observed_right = observed.right.float()
        for start in range(0, expected.shape[0], rows_per_chunk):
            stop = min(start + rows_per_chunk, expected.shape[0])
            expected_delta = expected.left[start:stop].float() @ expected_right
            observed_delta = observed.left[start:stop].float() @ observed_right
            expected_delta.mul_(float(expected.scaling))
            observed_delta.mul_(float(observed.scaling))
            if not torch.allclose(
                expected_delta,
                observed_delta,
                rtol=1e-5,
                atol=1e-7,
                equal_nan=False,
            ):
                maximum_error = float((expected_delta - observed_delta).abs().max())
                raise RegistryError(
                    f"PEFT companion does not materialize native delta for {name}; "
                    f"maximum absolute error={maximum_error:.6g}"
                )


def _object_path(root: Path, object_id: str) -> Path:
    digest = object_id.removeprefix("sha256:")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise RegistryError(f"invalid object id: {object_id}")
    return root / "objects" / "sha256" / digest


def _evaluation_path(root: Path, evaluation_id: str) -> Path:
    digest = evaluation_id.removeprefix("sha256:")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise RegistryError(f"invalid evaluation id: {evaluation_id}")
    return root / "evaluations" / "sha256" / digest


def _ref_path(root: Path, label: str) -> Path:
    label = _validate_label(label)
    parts = PurePosixPath(label).parts
    return root / "refs" / Path(*parts[:-1], f"{parts[-1]}.json")


def _evaluation_ref_path(root: Path, object_id: str, name: str) -> Path:
    digest = object_id.removeprefix("sha256:")
    name = _validate_label(name)
    parts = PurePosixPath(name).parts
    return (
        root
        / "evaluation_refs"
        / digest
        / Path(*parts[:-1], f"{parts[-1]}.json")
    )


def _write_ref(
    root: Path,
    *,
    label: str,
    object_id: str,
    source: str,
    replace: bool,
    reason: str | None,
) -> None:
    path = _ref_path(root, label)
    existing = _read_json(path) if path.exists() else None
    if existing is not None and (
        existing.get("schema_version") != SCHEMA_VERSION
        or existing.get("label") != label
        or not isinstance(existing.get("object_id"), str)
    ):
        raise RegistryError(f"invalid existing reference for label {label!r}: {path}")
    if existing is not None:
        _object_path(root, str(existing["object_id"]))
    if existing is not None and existing.get("object_id") == object_id:
        return
    if existing is not None and not replace:
        raise RegistryError(
            f"label {label!r} already points to {existing.get('object_id')}; "
            "use --replace-label with --reason to move it"
        )
    if existing is not None and not reason:
        raise RegistryError("moving an existing label requires --reason")
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "label": label,
        "object_id": object_id,
        "source": source,
        "updated_at": _utc_now(),
    }
    if existing is not None:
        value["previous_object_id"] = existing.get("object_id")
        value["reason"] = reason
    _atomic_json(path, value)


def add_adapter(
    registry: str | Path,
    source: str | Path,
    *,
    label: str,
    peft: str | Path | None = None,
    export_peft: bool = False,
    base_model: str | None = None,
    base_revision: str | None = None,
    replace_label: bool = False,
    reason: str | None = None,
) -> dict[str, Any]:
    """Register an adapter run and return its immutable manifest."""

    if peft is not None and export_peft:
        raise RegistryError("use either --peft or --export-peft, not both")
    label = _validate_label(label)
    root = Path(registry).resolve()
    _ensure_registry(root)
    adapter_dir, run_dir = _locate_source(Path(source))
    primary_format = _adapter_format(adapter_dir)
    if primary_format == "peft-lora" and peft is not None:
        raise RegistryError("the primary adapter is already PEFT; omit --peft")
    primary_config = _read_json(adapter_dir / "adapter_config.json")
    run_manifest_path = run_dir / "run_manifest.json" if run_dir is not None else None
    run_manifest = (
        _read_json(run_manifest_path)
        if run_manifest_path is not None and run_manifest_path.is_file()
        else None
    )
    declared_base_model = primary_config.get("base_model_name_or_path")
    if (
        isinstance(declared_base_model, str)
        and declared_base_model
        and base_model is not None
        and base_model != declared_base_model
    ):
        raise RegistryError(
            "--base-model conflicts with the adapter declaration: "
            f"{base_model!r} != {declared_base_model!r}"
        )
    resolved_base_model = base_model or _base_model_from(run_manifest, primary_config)
    if not resolved_base_model:
        raise RegistryError(
            "the base model could not be inferred; pass --base-model so the adapter "
            "cannot be applied to an ambiguous checkpoint"
        )
    peft_dir = Path(peft).resolve() if peft is not None else None
    if peft_dir is not None:
        if primary_format != "tinylora-v1":
            raise RegistryError("a PEFT companion requires a native TinyLoRA primary adapter")
        observed_peft_format = _adapter_format(peft_dir)
        if observed_peft_format != "peft-lora":
            raise RegistryError(
                f"expected peft-lora at {peft_dir}, found {observed_peft_format}"
            )
        _verify_peft_companion(
            adapter_dir,
            peft_dir,
            base_model=resolved_base_model,
        )

    staging = Path(tempfile.mkdtemp(prefix="adapter-", dir=root / ".tmp"))
    artifacts: list[dict[str, object]] = []
    try:
        if primary_format == "tinylora-v1":
            _copy_adapter(
                adapter_dir,
                staging / "native",
                expected_format=primary_format,
                artifact_root=staging,
                artifacts=artifacts,
            )
        else:
            _copy_adapter(
                adapter_dir,
                staging / "peft",
                expected_format=primary_format,
                artifact_root=staging,
                artifacts=artifacts,
            )

        if peft is not None:
            _copy_adapter(
                peft_dir,
                staging / "peft",
                expected_format="peft-lora",
                artifact_root=staging,
                artifacts=artifacts,
            )
        elif export_peft:
            if primary_format != "tinylora-v1":
                raise RegistryError("--export-peft requires a native TinyLoRA adapter")
            try:
                from .adapter_analysis import export_peft_adapter, load_adapter_updates
            except ImportError as exc:
                raise RegistryError(
                    "TinyLoRA export dependencies are unavailable; run this command "
                    "inside the research container"
                ) from exc
            export_peft_adapter(
                load_adapter_updates(adapter_dir),
                staging / "peft",
                base_model_name=resolved_base_model,
            )
            _verify_peft_companion(
                adapter_dir,
                staging / "peft",
                base_model=resolved_base_model,
            )
            for filename, role in (
                ("adapter_model.safetensors", "peft_weights"),
                ("adapter_config.json", "peft_config"),
            ):
                target = staging / "peft" / filename
                target.chmod(0o644)
                artifacts.append(_artifact(target, staging, role))

        provenance_names: list[str] = []
        if run_dir is not None:
            for filename in _PROVENANCE_FILES:
                source_file = run_dir / filename
                if not source_file.is_file():
                    continue
                target = staging / "provenance" / filename
                _copy_file(source_file, target)
                artifacts.append(_artifact(target, staging, f"provenance_{source_file.stem}"))
                provenance_names.append(filename)
            for filename in _PARENT_PROVENANCE_FILES:
                source_file = run_dir.parent / filename
                if not source_file.is_file():
                    continue
                copied_name = f"parent_{filename}"
                target = staging / "provenance" / copied_name
                _copy_file(source_file, target)
                artifacts.append(
                    _artifact(
                        target,
                        staging,
                        f"provenance_parent_{source_file.stem}",
                    )
                )
                provenance_names.append(copied_name)

        artifacts.sort(key=lambda item: str(item["path"]))
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": "adapter_bundle",
            "base_model": {
                "id": resolved_base_model,
                "revision": base_revision or _base_revision_from(run_manifest),
            },
            "adapter": _adapter_summary(primary_format, primary_config),
            "checkpoint": adapter_dir.name,
            "formats": sorted(
                {
                    "tinylora-v1" if item["role"] == "native_weights" else "peft-lora"
                    for item in artifacts
                    if item["role"] in {"native_weights", "peft_weights"}
                }
            ),
            "provenance": {
                "copied_files": provenance_names,
                "run_manifest": run_manifest,
                "training_summary": _training_summary(run_dir),
            },
            "artifacts": artifacts,
        }
        digest = _content_id(manifest, "object_id")
        object_id = f"sha256:{digest}"
        manifest["object_id"] = object_id
        _atomic_json(staging / "manifest.json", manifest)
        destination = _object_path(root, object_id)
        if destination.exists():
            verify_object_path(destination)
            _discard_staging(staging)
        else:
            _freeze_tree(staging)
            try:
                os.rename(staging, destination)
            except OSError as exc:
                if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise
                _discard_staging(staging)
                verify_object_path(destination)
        _write_ref(
            root,
            label=label,
            object_id=object_id,
            source=str(Path(source).resolve()),
            replace=replace_label,
            reason=reason,
        )
        return manifest
    except BaseException:
        _discard_staging(staging)
        raise


def _manifest_artifact_paths(manifest: dict[str, Any]) -> set[Path]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise RegistryError("manifest artifacts must be a list")
    paths: set[Path] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            raise RegistryError("invalid artifact entry in manifest")
        path = _safe_relative_path(artifact["path"])
        if path in paths:
            raise RegistryError(f"duplicate artifact path in manifest: {path}")
        paths.add(path)
    return paths


def _verify_record_path(path: Path, *, id_field: str) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    manifest = _read_json(manifest_path)
    observed_id = manifest.get(id_field)
    expected_digest = _content_id(manifest, id_field)
    expected_id = f"sha256:{expected_digest}"
    if observed_id != expected_id:
        raise RegistryError(
            f"manifest identity mismatch: recorded {observed_id}, computed {expected_id}"
        )
    if path.name != expected_digest:
        raise RegistryError(
            f"object directory {path.name} does not match manifest {expected_digest}"
        )
    expected_paths = _manifest_artifact_paths(manifest)
    children = list(path.rglob("*"))
    symlinks = [child for child in children if child.is_symlink()]
    if symlinks:
        raise RegistryError(f"registry records cannot contain symlinks: {symlinks[0]}")
    observed_paths = {
        child.relative_to(path)
        for child in children
        if child.is_file() and child.name != "manifest.json"
    }
    if observed_paths != expected_paths:
        missing = sorted(str(item) for item in expected_paths - observed_paths)
        extra = sorted(str(item) for item in observed_paths - expected_paths)
        raise RegistryError(f"artifact set mismatch; missing={missing}, extra={extra}")
    artifacts = manifest["artifacts"]
    for artifact in artifacts:
        artifact_path = path / _safe_relative_path(str(artifact["path"]))
        size = artifact_path.stat().st_size
        if size != int(artifact.get("size_bytes", -1)):
            raise RegistryError(f"size mismatch for {artifact_path}")
        digest = _sha256(artifact_path)
        if digest != artifact.get("sha256"):
            raise RegistryError(f"SHA-256 mismatch for {artifact_path}")
    return manifest


def verify_object_path(path: str | Path) -> dict[str, Any]:
    return _verify_record_path(Path(path), id_field="object_id")


def _resolve_object_id(root: Path, identifier: str) -> str:
    if identifier.startswith("sha256:") or re.fullmatch(r"[0-9a-f]{64}", identifier):
        object_id = identifier if identifier.startswith("sha256:") else f"sha256:{identifier}"
        if not _object_path(root, object_id).is_dir():
            raise RegistryError(f"unknown adapter object: {object_id}")
        return object_id
    ref_path = _ref_path(root, identifier)
    ref = _read_json(ref_path)
    if ref.get("schema_version") != SCHEMA_VERSION:
        raise RegistryError(f"invalid reference schema for label {identifier!r}")
    if ref.get("label") != identifier:
        raise RegistryError(
            f"reference at {ref_path} declares label {ref.get('label')!r}, "
            f"expected {identifier!r}"
        )
    object_id = ref.get("object_id")
    if not isinstance(object_id, str):
        raise RegistryError(f"invalid reference for label {identifier!r}")
    object_path = _object_path(root, object_id)
    if not object_path.is_dir():
        raise RegistryError(f"label {identifier!r} points to missing object {object_id}")
    return object_id


def _labels_for(root: Path, object_id: str) -> list[str]:
    labels = []
    refs_root = root / "refs"
    if not refs_root.exists():
        return labels
    for path in refs_root.rglob("*.json"):
        ref = _read_json(path)
        label = ref.get("label")
        if not isinstance(label, str):
            raise RegistryError(f"invalid registry reference: {path}")
        if ref.get("schema_version") != SCHEMA_VERSION or _ref_path(root, label) != path:
            raise RegistryError(f"registry reference path/identity mismatch: {path}")
        referenced_id = ref.get("object_id")
        if not isinstance(referenced_id, str):
            raise RegistryError(f"invalid registry reference: {path}")
        _object_path(root, referenced_id)
        if referenced_id == object_id:
            labels.append(label)
    return sorted(labels)


def _attached_evaluations(root: Path, object_id: str) -> list[dict[str, Any]]:
    directory = root / "evaluation_refs" / object_id.removeprefix("sha256:")
    if not directory.exists():
        return []
    attached = []
    for path in sorted(directory.rglob("*.json")):
        ref = _read_json(path)
        if ref.get("adapter_object_id") != object_id:
            raise RegistryError(
                f"evaluation reference {path} names adapter "
                f"{ref.get('adapter_object_id')!r}, expected {object_id!r}"
            )
        name = ref.get("name")
        if not isinstance(name, str):
            raise RegistryError(f"invalid evaluation reference name: {path}")
        if _evaluation_ref_path(root, object_id, name) != path:
            raise RegistryError(
                f"evaluation reference path {path} does not match its name {name!r}"
            )
        evaluation_id = ref.get("evaluation_id")
        if not isinstance(evaluation_id, str):
            raise RegistryError(f"invalid evaluation reference: {path}")
        manifest = _verify_record_path(
            _evaluation_path(root, evaluation_id),
            id_field="evaluation_id",
        )
        if manifest.get("adapter_object_id") != object_id:
            raise RegistryError(
                f"evaluation {evaluation_id} belongs to adapter "
                f"{manifest.get('adapter_object_id')!r}, expected {object_id!r}"
            )
        if manifest.get("name") != name:
            raise RegistryError(
                f"evaluation {evaluation_id} is named {manifest.get('name')!r}, "
                f"but its reference is named {name!r}"
            )
        binding = manifest.get("adapter_binding")
        if binding is not None:
            if not isinstance(binding, dict) or binding.get("format") != "peft-lora":
                raise RegistryError(f"evaluation {evaluation_id} has an invalid adapter binding")
            recorded = _normalise_peft_binding(
                binding.get("artifacts"),
                context=f"evaluation {evaluation_id}",
            )
            registered = _registered_peft_binding(
                verify_object_path(_object_path(root, object_id))
            )
            if recorded != registered:
                raise RegistryError(
                    f"evaluation {evaluation_id} adapter binding does not match {object_id}"
                )
        attached.append(manifest)
    return attached


def show_adapter(registry: str | Path, identifier: str) -> dict[str, Any]:
    root = Path(registry).resolve()
    object_id = _resolve_object_id(root, identifier)
    manifest = verify_object_path(_object_path(root, object_id))
    result = dict(manifest)
    result["labels"] = _labels_for(root, object_id)
    result["evaluations"] = _attached_evaluations(root, object_id)
    return result


def list_adapters(registry: str | Path) -> list[dict[str, Any]]:
    root = Path(registry).resolve()
    refs_root = root / "refs"
    if not refs_root.exists():
        return []
    entries = []
    for path in sorted(refs_root.rglob("*.json")):
        ref = _read_json(path)
        object_id = ref.get("object_id")
        label = ref.get("label")
        if not isinstance(object_id, str) or not isinstance(label, str):
            raise RegistryError(f"invalid registry reference: {path}")
        if ref.get("schema_version") != SCHEMA_VERSION or _ref_path(root, label) != path:
            raise RegistryError(f"registry reference path/identity mismatch: {path}")
        manifest = verify_object_path(_object_path(root, object_id))
        entries.append(
            {
                "label": label,
                "object_id": object_id,
                "base_model": manifest.get("base_model"),
                "adapter": manifest.get("adapter"),
                "formats": manifest.get("formats", []),
                "evaluation_count": len(_attached_evaluations(root, object_id)),
            }
        )
    return entries


def verify_adapter(registry: str | Path, identifier: str) -> dict[str, Any]:
    root = Path(registry).resolve()
    object_id = _resolve_object_id(root, identifier)
    manifest = verify_object_path(_object_path(root, object_id))
    evaluations = _attached_evaluations(root, object_id)
    return {
        "object_id": object_id,
        "artifacts_verified": len(manifest["artifacts"]),
        "evaluations_verified": len(evaluations),
    }


def adapter_path(
    registry: str | Path,
    identifier: str,
    *,
    adapter_format: str,
) -> Path:
    """Resolve and verify an adapter's native or PEFT directory."""

    if adapter_format not in {"native", "peft"}:
        raise RegistryError("adapter format must be 'native' or 'peft'")
    root = Path(registry).resolve()
    object_id = _resolve_object_id(root, identifier)
    object_path = _object_path(root, object_id)
    manifest = verify_object_path(object_path)
    required = "tinylora-v1" if adapter_format == "native" else "peft-lora"
    if required not in manifest.get("formats", []):
        raise RegistryError(f"adapter {identifier!r} has no {adapter_format} format")
    return object_path / adapter_format


def _scalar_fields(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if isinstance(item, (str, int, float, bool)) or item is None
    }


_PEFT_BINDING_FILES = {
    "adapter_config.json": "peft_config",
    "adapter_model.safetensors": "peft_weights",
}


def _normalise_peft_binding(value: object, *, context: str) -> dict[str, dict[str, object]]:
    if not isinstance(value, dict):
        raise RegistryError(
            f"{context} is missing adapter_artifacts; rerun the evaluation with "
            "an evaluator that records PEFT config and weight hashes"
        )
    normalised: dict[str, dict[str, object]] = {}
    for filename in _PEFT_BINDING_FILES:
        metadata = value.get(filename)
        if not isinstance(metadata, dict):
            raise RegistryError(f"{context} is missing hash metadata for {filename}")
        digest = metadata.get("sha256")
        size = metadata.get("size_bytes")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RegistryError(f"{context} has an invalid SHA-256 for {filename}")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise RegistryError(f"{context} has an invalid size for {filename}")
        normalised[filename] = {"sha256": digest, "size_bytes": size}
    return normalised


def _registered_peft_binding(manifest: dict[str, Any]) -> dict[str, dict[str, object]]:
    if "peft-lora" not in manifest.get("formats", []):
        raise RegistryError(
            "the registered adapter has no PEFT representation to bind to this evaluation"
        )
    by_role: dict[str, dict[str, object]] = {}
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise RegistryError("adapter manifest artifacts must be a list")
    for artifact in artifacts:
        if isinstance(artifact, dict) and isinstance(artifact.get("role"), str):
            role = str(artifact["role"])
            if role in _PEFT_BINDING_FILES.values():
                if role in by_role:
                    raise RegistryError(f"adapter manifest contains duplicate {role} artifacts")
                by_role[role] = artifact

    binding: dict[str, dict[str, object]] = {}
    for filename, role in _PEFT_BINDING_FILES.items():
        artifact = by_role.get(role)
        if artifact is None:
            raise RegistryError(f"adapter manifest is missing its {role} artifact")
        artifact_path = artifact.get("path")
        if artifact_path != f"peft/{filename}":
            raise RegistryError(
                f"adapter manifest {role} path is {artifact_path!r}, expected "
                f"{'peft/' + filename!r}"
            )
        digest = artifact.get("sha256")
        size = artifact.get("size_bytes")
        binding[filename] = {"sha256": digest, "size_bytes": size}
    return _normalise_peft_binding(binding, context="registered PEFT adapter")


def _evaluation_summary(
    result: dict[str, Any],
    candidate: str | None,
) -> tuple[dict[str, Any], str | None, dict[str, Any] | None]:
    candidates = result.get("candidates")
    selected: dict[str, Any] | None = None
    selected_name = candidate
    if isinstance(candidates, dict):
        if selected_name is None:
            if len(candidates) != 1:
                raise RegistryError(
                    "evaluation contains multiple candidates; pass --candidate"
                )
            selected_name = next(iter(candidates))
        value = candidates.get(selected_name)
        if not isinstance(value, dict):
            raise RegistryError(f"evaluation has no candidate named {selected_name!r}")
        selected = value
    summary: dict[str, Any] = {
        key: result[key]
        for key in (
            "schema_version",
            "metric",
            "model",
            "dataset",
            "dataset_config",
            "split",
            "samples",
            "base_accuracy",
            "adapter_accuracy",
        )
        if key in result
    }
    if isinstance(result.get("base"), dict):
        summary["base"] = _scalar_fields(result["base"])
    if selected is not None:
        adapted = _scalar_fields(selected)
        if isinstance(selected.get("metrics"), dict):
            adapted["metrics"] = _scalar_fields(selected["metrics"])
        summary["candidate"] = selected_name
        summary["adapted"] = adapted
        comparison_baseline = result.get("comparison_baseline")
        if isinstance(comparison_baseline, str):
            summary["comparison_baseline"] = comparison_baseline
        comparisons = result.get("comparisons")
        if (
            isinstance(comparisons, dict)
            and isinstance(selected_name, str)
            and isinstance(comparisons.get(selected_name), dict)
        ):
            # Paired intervals and p-values are small but essential for deciding
            # whether an adapter helped, so retain the selected comparison in
            # the compact manifest as well as in the copied full results file.
            summary["comparisons"] = {selected_name: comparisons[selected_name]}
    return summary, selected_name, selected


def add_evaluation(
    registry: str | Path,
    identifier: str,
    results: str | Path,
    *,
    name: str,
    candidate: str | None = None,
    replace: bool = False,
    reason: str | None = None,
) -> dict[str, Any]:
    """Attach an immutable evaluation result to a registered adapter."""

    name = _validate_label(name)
    root = Path(registry).resolve()
    _ensure_registry(root)
    object_id = _resolve_object_id(root, identifier)
    object_manifest = verify_object_path(_object_path(root, object_id))
    results_path = Path(results).resolve()
    result = _read_json(results_path)
    summary, selected_name, selected = _evaluation_summary(result, candidate)
    if selected is None or selected_name is None:
        raise RegistryError(
            "evaluation result must contain a selected candidate with adapter artifacts"
        )
    evaluated_binding = _normalise_peft_binding(
        selected.get("adapter_artifacts"),
        context=f"evaluation candidate {selected_name!r}",
    )
    registered_binding = _registered_peft_binding(object_manifest)
    if evaluated_binding != registered_binding:
        mismatches = [
            filename
            for filename in _PEFT_BINDING_FILES
            if evaluated_binding[filename] != registered_binding[filename]
        ]
        raise RegistryError(
            "evaluation candidate PEFT artifacts do not match the registered adapter; "
            f"mismatched={mismatches}"
        )
    staging = Path(tempfile.mkdtemp(prefix="evaluation-", dir=root / ".tmp"))
    try:
        copied = staging / "results.json"
        _copy_file(results_path, copied)
        artifacts = [_artifact(copied, staging, "evaluation_results")]
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": "adapter_evaluation",
            "adapter_object_id": object_id,
            "name": name,
            "candidate": selected_name,
            "adapter_binding": {
                "format": "peft-lora",
                "artifacts": evaluated_binding,
            },
            "summary": summary,
            "artifacts": artifacts,
        }
        digest = _content_id(manifest, "evaluation_id")
        evaluation_id = f"sha256:{digest}"
        manifest["evaluation_id"] = evaluation_id
        _atomic_json(staging / "manifest.json", manifest)
        destination = _evaluation_path(root, evaluation_id)
        if destination.exists():
            _verify_record_path(destination, id_field="evaluation_id")
            _discard_staging(staging)
        else:
            _freeze_tree(staging)
            try:
                os.rename(staging, destination)
            except OSError as exc:
                if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise
                _discard_staging(staging)
                _verify_record_path(destination, id_field="evaluation_id")

        ref_path = _evaluation_ref_path(root, object_id, name)
        existing = _read_json(ref_path) if ref_path.exists() else None
        if existing is not None and existing.get("evaluation_id") != evaluation_id:
            if not replace:
                raise RegistryError(
                    f"evaluation name {name!r} is already attached; use --replace "
                    "with --reason to move it"
                )
            if not reason:
                raise RegistryError("replacing an evaluation reference requires --reason")
        ref: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "adapter_object_id": object_id,
            "name": name,
            "evaluation_id": evaluation_id,
            "updated_at": _utc_now(),
        }
        if existing is not None and existing.get("evaluation_id") != evaluation_id:
            ref["previous_evaluation_id"] = existing.get("evaluation_id")
            ref["reason"] = reason
        _atomic_json(ref_path, ref)
        return manifest
    except BaseException:
        _discard_staging(staging)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default=os.environ.get("TINYLORA_REGISTRY", DEFAULT_REGISTRY),
        help="local registry root (default: %(default)s)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    add = commands.add_parser("add", help="register a run or adapter directory")
    add.add_argument("source")
    add.add_argument("--label", required=True)
    peft_group = add.add_mutually_exclusive_group()
    peft_group.add_argument("--peft", help="existing PEFT adapter directory to bundle")
    peft_group.add_argument("--export-peft", action="store_true")
    add.add_argument("--base-model")
    add.add_argument("--base-revision")
    add.add_argument("--replace-label", action="store_true")
    add.add_argument("--reason")

    commands.add_parser("list", help="list labelled adapters").add_argument(
        "--json", action="store_true"
    )
    show = commands.add_parser("show", help="show an adapter manifest and evaluations")
    show.add_argument("identifier", help="label, full digest, or sha256:<digest>")
    verify = commands.add_parser("verify", help="verify bundle and evaluation hashes")
    verify.add_argument("identifier")
    path_command = commands.add_parser("path", help="print a verified adapter directory")
    path_command.add_argument("identifier")
    path_command.add_argument("--format", choices=("native", "peft"), required=True)

    evaluation = commands.add_parser("add-eval", help="attach an evaluation JSON file")
    evaluation.add_argument("identifier")
    evaluation.add_argument("results")
    evaluation.add_argument("--name", required=True)
    evaluation.add_argument("--candidate")
    evaluation.add_argument("--replace", action="store_true")
    evaluation.add_argument("--reason")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "add":
            result = add_adapter(
                args.registry,
                args.source,
                label=args.label,
                peft=args.peft,
                export_peft=args.export_peft,
                base_model=args.base_model,
                base_revision=args.base_revision,
                replace_label=args.replace_label,
                reason=args.reason,
            )
        elif args.command == "list":
            entries = list_adapters(args.registry)
            if args.json:
                print(json.dumps(entries, indent=2, sort_keys=True))
            else:
                for entry in entries:
                    adapter = entry.get("adapter") or {}
                    print(
                        "\t".join(
                            (
                                str(entry["label"]),
                                str(entry["object_id"]),
                                str((entry.get("base_model") or {}).get("id", "")),
                                str(adapter.get("format", "")),
                                f"evals={entry['evaluation_count']}",
                            )
                        )
                    )
            return 0
        elif args.command == "show":
            result = show_adapter(args.registry, args.identifier)
        elif args.command == "verify":
            result = verify_adapter(args.registry, args.identifier)
        elif args.command == "path":
            print(
                adapter_path(
                    args.registry,
                    args.identifier,
                    adapter_format=args.format,
                )
            )
            return 0
        elif args.command == "add-eval":
            result = add_evaluation(
                args.registry,
                args.identifier,
                args.results,
                name=args.name,
                candidate=args.candidate,
                replace=args.replace,
                reason=args.reason,
            )
        else:  # pragma: no cover - argparse enforces this
            raise AssertionError(args.command)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (RegistryError, OSError) as exc:
        print(f"registry error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
