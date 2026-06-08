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
        data = {
            "adapter": self.name,
            "name": self.config.get("name", self.name),
            "platform": platform.platform(),
            "thermal_policy": self.config.get("thermal_policy", "record_only"),
            "power_source": self.config.get("power_source", "unknown"),
        }
        gpu = self.nvidia_gpu_snapshot()
        if gpu:
            data["nvidia_gpu"] = gpu
        return data

    def resources(self) -> ResourceSample:
        process = psutil.Process()
        return ResourceSample(ram_mb=process.memory_info().rss / (1024 * 1024), vram_mb=self.vram_mb())

    def vram_mb(self) -> float | None:
        gpu = self.nvidia_gpu_snapshot()
        if gpu and gpu.get("memory.used [MiB]") not in {None, "[N/A]"}:
            return _parse_float(gpu["memory.used [MiB]"])
        return None

    def energy_j(self) -> float | None:
        return None

    def mean_power_w(self) -> float | None:
        return sum(self.power_samples_w) / len(self.power_samples_w) if self.power_samples_w else None

    def thermal_snapshot(self) -> dict[str, Any]:
        return {
            "temperatures_c": self._psutil_temperatures(),
            "throttling": "unknown",
            "nvidia_gpu": self.nvidia_gpu_snapshot(),
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

    def nvidia_gpu_snapshot(self) -> dict[str, Any] | None:
        index = self.config.get("cuda_index")
        if index is None:
            return None
        fields = [
            "index",
            "name",
            "driver_version",
            "memory.total",
            "memory.used",
            "temperature.gpu",
            "power.draw",
            "power.limit",
            "clocks.current.graphics",
            "clocks.current.memory",
        ]
        output = self.run_text(
            [
                "nvidia-smi",
                f"--id={index}",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ]
        )
        if not output:
            return None
        values = [item.strip() for item in output.splitlines()[0].split(",")]
        keys = [
            "index",
            "name",
            "driver_version",
            "memory.total [MiB]",
            "memory.used [MiB]",
            "temperature.gpu [C]",
            "power.draw [W]",
            "power.limit [W]",
            "clocks.current.graphics [MHz]",
            "clocks.current.memory [MHz]",
        ]
        snapshot = dict(zip(keys, values))
        power_w = _parse_float(snapshot.get("power.draw [W]"))
        if power_w is not None:
            self.power_samples_w.append(power_w)
        return snapshot


def _parse_float(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None
