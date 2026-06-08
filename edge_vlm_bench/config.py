from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import hashlib
import json

import yaml


@dataclass(frozen=True)
class BenchmarkConfig:
    device: dict[str, Any]
    model: dict[str, Any]
    dataset: dict[str, Any]
    protocol: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return int(self.protocol.get("batch_size", 1))

    @property
    def input_size(self) -> tuple[int, int]:
        value = self.protocol.get("input_size", [640, 640])
        if len(value) != 2:
            raise ValueError("protocol.input_size must contain [height, width]")
        return int(value[0]), int(value[1])

    @property
    def warmup_runs(self) -> int:
        return int(self.protocol.get("warmup_runs", 5))

    @property
    def measured_runs(self) -> int:
        return int(self.protocol.get("measured_runs", 30))

    @property
    def include_prepost(self) -> bool:
        return bool(self.protocol.get("include_prepost", True))


def load_config(path: str | Path) -> BenchmarkConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a YAML mapping")
    for key in ("device", "model", "dataset", "protocol"):
        if key not in raw:
            raise ValueError(f"missing required config section: {key}")
    return BenchmarkConfig(
        device=raw["device"],
        model=raw["model"],
        dataset=raw["dataset"],
        protocol=raw["protocol"],
        metrics=raw.get("metrics", {}),
    )


def canonical_config_hash(config: BenchmarkConfig) -> str:
    payload = json.dumps(config.__dict__, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

