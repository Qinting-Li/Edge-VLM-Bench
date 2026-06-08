from __future__ import annotations

from typing import Any
import importlib.metadata
import json
import subprocess


class ModelAdapter:
    family = "base"

    def __init__(self, config: dict[str, Any], input_size: tuple[int, int], batch_size: int):
        self.config = config
        self.input_size = input_size
        self.batch_size = batch_size

    def load(self) -> None:
        raise NotImplementedError

    def preprocess(self, batch: list[Any]) -> Any:
        return batch

    def infer(self, prepared: Any) -> Any:
        raise NotImplementedError

    def postprocess(self, raw: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "name": self.config.get("name", self.family),
            "backend": self.config.get("backend", "unknown"),
            "version": self.config.get("version", "unversioned"),
            "weights": self.config.get("weights"),
            "backend_versions": self._backend_versions(),
        }

    def _backend_versions(self) -> dict[str, str]:
        versions = {}
        for package in ("torch", "onnxruntime", "ultralytics", "segment-anything", "sam2"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                continue
        return versions

    def run_external_json(self, batch: list[Any]) -> list[dict[str, Any]]:
        command = self.config.get("external_command")
        if not command:
            raise ValueError("backend=external requires model.external_command")
        payload = {
            "batch": batch,
            "input_size": list(self.input_size),
            "batch_size": self.batch_size,
            "model": self.config,
        }
        completed = subprocess.run(
            command,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=True,
            timeout=float(self.config.get("timeout_s", 300)),
        )
        output = json.loads(completed.stdout)
        if not isinstance(output, list):
            raise ValueError("external model command must print a JSON list of predictions")
        return output
