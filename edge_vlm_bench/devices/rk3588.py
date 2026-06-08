from __future__ import annotations

from typing import Any

from edge_vlm_bench.devices.base import DeviceAdapter


class RK3588Adapter(DeviceAdapter):
    name = "rk3588"

    def describe(self) -> dict[str, Any]:
        data = super().describe()
        data.update(
            {
                "npu_info": self.run_text(["cat", "/sys/kernel/debug/rknpu/load"]),
                "soc": "RK3588",
            }
        )
        return data

