from __future__ import annotations

from typing import Any

from edge_vlm_bench.devices.base import DeviceAdapter


class JetsonAdapter(DeviceAdapter):
    name = "jetson"

    def describe(self) -> dict[str, Any]:
        data = super().describe()
        data.update(
            {
                "jetson_stats": self.run_text(["tegrastats", "--interval", "1000", "--count", "1"]),
                "nvpmodel": self.run_text(["nvpmodel", "-q"]),
            }
        )
        return data

    def vram_mb(self) -> float | None:
        return None

