from __future__ import annotations

from edge_vlm_bench.models.dummy import DummyAdapter
from edge_vlm_bench.models.sam import SAMAdapter
from edge_vlm_bench.models.vlm import VLMAdapter
from edge_vlm_bench.models.yolo import YOLOAdapter


def create_model(config: dict, input_size: tuple[int, int], batch_size: int):
    kind = str(config.get("type", "dummy")).lower()
    adapters = {
        "dummy": DummyAdapter,
        "yolo": YOLOAdapter,
        "sam": SAMAdapter,
        "sam2": SAMAdapter,
        "vlm": VLMAdapter,
    }
    if kind not in adapters:
        raise ValueError(f"unsupported model type: {kind}")
    return adapters[kind](config, input_size, batch_size)

