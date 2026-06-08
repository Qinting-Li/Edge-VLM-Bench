from __future__ import annotations

from typing import Any

from edge_vlm_bench.models.base import ModelAdapter


class VLMAdapter(ModelAdapter):
    family = "vlm"

    def load(self) -> None:
        self.backend = self.config.get("backend", "external")
        self.prompt = self.config.get("prompt", "Describe the image.")

    def infer(self, prepared: Any) -> Any:
        if self.backend == "external":
            return self.run_external_json(prepared)
        raise RuntimeError(
            "VLMAdapter defines the benchmark contract. Add a backend such as llama.cpp, "
            "MLX, MNN, ExecuTorch, or a vendor SDK for production runs."
        )

    def postprocess(self, raw: Any) -> list[dict[str, Any]]:
        return []
