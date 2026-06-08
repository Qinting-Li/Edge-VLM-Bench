from __future__ import annotations

from typing import Any

from edge_vlm_bench.devices.base import DeviceAdapter


class MacOSAdapter(DeviceAdapter):
    name = "macos"

    def describe(self) -> dict[str, Any]:
        data = super().describe()
        data.update(
            {
                "chip": self.run_text(["sysctl", "-n", "machdep.cpu.brand_string"]),
                "powermetrics": "record with sudo powermetrics for production power runs",
            }
        )
        return data

