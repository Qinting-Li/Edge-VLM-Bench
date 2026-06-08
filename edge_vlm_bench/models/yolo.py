from __future__ import annotations

from typing import Any

from edge_vlm_bench.models.base import ModelAdapter


class YOLOAdapter(ModelAdapter):
    family = "yolo"

    def load(self) -> None:
        backend = self.config.get("backend", "ultralytics")
        if backend != "ultralytics":
            raise ValueError("YOLOAdapter currently supports backend=ultralytics")
        from ultralytics import YOLO

        self.model = YOLO(self.config["weights"])

    def infer(self, prepared: Any) -> Any:
        return self.model.predict(
            prepared,
            imgsz=self.input_size[0],
            batch=self.batch_size,
            verbose=False,
        )

    def postprocess(self, raw: Any) -> list[dict[str, Any]]:
        predictions: list[dict[str, Any]] = []
        for image_idx, result in enumerate(raw):
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                predictions.append(
                    {
                        "image_id": image_idx,
                        "bbox": [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]],
                        "score": float(box.conf[0]),
                        "category_id": int(box.cls[0]) + 1,
                    }
                )
        return predictions

