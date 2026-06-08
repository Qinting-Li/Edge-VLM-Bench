from __future__ import annotations

from typing import Any
import importlib.metadata


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

