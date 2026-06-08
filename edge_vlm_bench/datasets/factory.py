from __future__ import annotations

from edge_vlm_bench.datasets.coco import COCODataset
from edge_vlm_bench.datasets.custom import CustomDataset


def create_dataset(config: dict, demo: bool = False):
    kind = str(config.get("type", "custom")).lower()
    adapters = {
        "custom": CustomDataset,
        "coco": COCODataset,
    }
    if kind not in adapters:
        raise ValueError(f"unsupported dataset type: {kind}")
    return adapters[kind](config, demo=demo)

