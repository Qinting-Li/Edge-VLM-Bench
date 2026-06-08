from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import platform
import subprocess

import psutil


@dataclass(frozen=True)
class ResourceSample:
    ram_mb: float
    vram_mb: float | None = None


class DeviceAdapter:
    name = "generic"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.power_samples_w: list[float] = []

    def prepare(self) -> None:
        return None

    def describe(self) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "name": self.config.get("name", self.name),
            "platform": platform.platform(),
            "thermal_policy": self.config.get("thermal_policy", "record_only"),
            "power_source": self.config.get("power_source", "unknown"),
        }

    def resources(self) -> ResourceSample:
        process = psutil.Process()
        return ResourceSample(ram_mb=process.memory_info().rss / (1024 * 1024), vram_mb=self.vram_mb())

    def vram_mb(self) -> float | None:
        return None

    def energy_j(self) -> float | None:
        return None

    def mean_power_w(self) -> float | None:
        return sum(self.power_samples_w) / len(self.power_samples_w) if self.power_samples_w else None

    def thermal_snapshot(self) -> dict[str, Any]:
        return {
            "temperatures_c": self._psutil_temperatures(),
            "throttling": "unknown",
            "notes": "Device-specific adapters should record clocks, governors, and throttling flags.",
        }

    def _psutil_temperatures(self) -> dict[str, Any]:
        try:
            temps = psutil.sensors_temperatures()
        except (AttributeError, OSError):
            return {}
        return {name: [entry._asdict() for entry in entries] for name, entries in temps.items()}

    @staticmethod
    def run_text(command: list[str]) -> str | None:
        try:
            return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=2).strip()
        except (subprocess.SubprocessError, OSError):
            return None

