"""An append-only release chain for adapters.

Every accepted update is a release directory holding the native TinyLoRA
adapter, the optimizer state, the PEFT transport vLLM loads, and a manifest
that names its parent. ``HEAD`` points at the incumbent, ``BEST`` at the
release with the highest held-out accuracy, and ``history.jsonl`` records
every commit. Rollback appends a new release pointing at older content;
nothing is rewritten.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

NATIVE_FILES = ("adapter.safetensors", "adapter_config.json")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def content_id(native_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in NATIVE_FILES:
        digest.update(name.encode())
        digest.update((native_dir / name).read_bytes())
    return "content:" + digest.hexdigest()[:32]


class ReleaseChain:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.releases = self.root / "releases"
        self.releases.mkdir(parents=True, exist_ok=True)
        self.history_path = self.root / "history.jsonl"

    # ------------------------------------------------------------ pointers
    def _read_pointer(self, name: str) -> str | None:
        path = self.root / name
        if not path.exists():
            return None
        value = path.read_text().strip()
        return value or None

    def head(self) -> str | None:
        return self._read_pointer("HEAD")

    def best(self) -> str | None:
        return self._read_pointer("BEST")

    def set_head(self, release_id: str) -> None:
        atomic_write_text(self.root / "HEAD", release_id + "\n")

    def set_best(self, release_id: str) -> None:
        atomic_write_text(self.root / "BEST", release_id + "\n")

    # ------------------------------------------------------------ releases
    def release_dir(self, release_id: str) -> Path:
        return self.releases / release_id

    def manifest(self, release_id: str) -> dict[str, Any]:
        return json.loads((self.release_dir(release_id) / "manifest.json").read_text())

    def has_manifest(self, release_id: str) -> bool:
        return (self.release_dir(release_id) / "manifest.json").exists()

    def next_id(self) -> str:
        numbers = [int(path.name[1:]) for path in self.releases.iterdir() if path.name.startswith("r") and path.name[1:].isdigit()]
        return f"r{(max(numbers) + 1) if numbers else 0:06d}"

    def begin(self) -> tuple[str, Path]:
        release_id = self.next_id()
        directory = self.release_dir(release_id)
        directory.mkdir(parents=True, exist_ok=False)
        return release_id, directory

    def commit(self, release_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
        directory = self.release_dir(release_id)
        native = directory / "final_adapter"
        record = {
            "release_id": release_id,
            "content_id": content_id(native),
            "committed_at": utc_now(),
            **manifest,
        }
        atomic_write_json(directory / "manifest.json", record)
        with self.history_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.set_head(release_id)
        return record

    def history(self) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []
        return [json.loads(line) for line in self.history_path.read_text().splitlines() if line.strip()]

    def copy_content(self, source_release: str, destination: Path) -> None:
        """Copy the adapter, optimizer and PEFT transport of one release into a new directory."""
        source = self.release_dir(source_release)
        shutil.copytree(source / "final_adapter", destination / "final_adapter")
        if (source / "optimizer.pt").exists():
            shutil.copy2(source / "optimizer.pt", destination / "optimizer.pt")
        shutil.copytree(source / "peft_adapter", destination / "peft_adapter")

    def discard(self, release_id: str) -> None:
        """Remove an uncommitted release directory left by a failed step."""
        if self.has_manifest(release_id):
            raise ValueError(f"refusing to discard committed release {release_id}")
        shutil.rmtree(self.release_dir(release_id), ignore_errors=True)
