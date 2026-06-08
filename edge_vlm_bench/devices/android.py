from __future__ import annotations

from typing import Any

from edge_vlm_bench.devices.base import DeviceAdapter


class AndroidAdapter(DeviceAdapter):
    name = "android"

    def describe(self) -> dict[str, Any]:
        data = super().describe()
        adb = self.config.get("adb", "adb")
        data.update(
            {
                "android_serial": self.config.get("serial", "default"),
                "device_props": self.run_text([adb, "shell", "getprop", "ro.product.model"]),
                "battery": self.run_text([adb, "shell", "dumpsys", "battery"]),
            }
        )
        return data

