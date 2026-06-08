from __future__ import annotations

from typing import Any

from edge_vlm_bench.models.base import ModelAdapter


class SAMAdapter(ModelAdapter):
    family = "sam"

    def load(self) -> None:
        backend = self.config.get("backend", "python")
        if backend not in {"python", "sam2"}:
            raise ValueError("SAMAdapter supports backend=python or backend=sam2")
        self.backend = backend

    def infer(self, prepared: Any) -> Any:
        raise RuntimeError(
            "SAM/SAM2 adapters are placeholders for production packages. "
            "Use model.type=dummy for demo or add a concrete SAM backend implementation."
        )

    def postprocess(self, raw: Any) -> list[dict[str, Any]]:
        return []

