from __future__ import annotations

from typing import Any

from edge_vlm_bench.models.base import ModelAdapter


class SAMAdapter(ModelAdapter):
    family = "sam"

    def load(self) -> None:
        backend = self.config.get("backend", "external")
        if backend not in {"external", "python", "sam2"}:
            raise ValueError("SAMAdapter supports backend=external, backend=python, or backend=sam2")
        self.backend = backend

    def infer(self, prepared: Any) -> Any:
        if self.backend == "external":
            return self.run_external_json(prepared)
        raise RuntimeError(
            "SAM/SAM2 adapters are placeholders for production packages. "
            "Use backend=external to call a deployment script or add a concrete SAM backend."
        )

    def postprocess(self, raw: Any) -> list[dict[str, Any]]:
        return []
