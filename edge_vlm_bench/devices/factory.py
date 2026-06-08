from __future__ import annotations

from edge_vlm_bench.devices.android import AndroidAdapter
from edge_vlm_bench.devices.base import DeviceAdapter
from edge_vlm_bench.devices.jetson import JetsonAdapter
from edge_vlm_bench.devices.macos import MacOSAdapter
from edge_vlm_bench.devices.rk3588 import RK3588Adapter


def create_device(config: dict) -> DeviceAdapter:
    kind = str(config.get("type", "generic")).lower()
    adapters = {
        "generic": DeviceAdapter,
        "jetson": JetsonAdapter,
        "rk3588": RK3588Adapter,
        "macos": MacOSAdapter,
        "android": AndroidAdapter,
    }
    if kind not in adapters:
        raise ValueError(f"unsupported device type: {kind}")
    return adapters[kind](config)

