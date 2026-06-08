from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any
import platform
import time
import uuid

from edge_vlm_bench.config import BenchmarkConfig, canonical_config_hash
from edge_vlm_bench.datasets.factory import create_dataset
from edge_vlm_bench.devices.factory import create_device
from edge_vlm_bench.metrics.accuracy import evaluate_predictions
from edge_vlm_bench.metrics.latency import percentile
from edge_vlm_bench.models.factory import create_model


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig, demo: bool = False):
        self.config = config
        self.demo = demo

    def run(self) -> dict[str, Any]:
        device = create_device(self.config.device)
        dataset = create_dataset(self.config.dataset, demo=self.demo)
        model = create_model(self.config.model, self.config.input_size, self.config.batch_size)

        run_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc).isoformat()
        device.prepare()
        model.load()
        env = device.describe()
        backend_versions = model.describe()

        cold_sample = self._measure_one(model, dataset.sample_batch(self.config.batch_size), device)
        warmup_samples = []
        for _ in range(self.config.warmup_runs):
            warmup_samples.append(
                self._measure_one(model, dataset.sample_batch(self.config.batch_size), device)
            )

        measured_samples = []
        predictions = []
        for idx in range(self.config.measured_runs):
            batch = dataset.get_batch(idx, self.config.batch_size)
            sample = self._measure_one(model, batch, device)
            measured_samples.append(sample)
            predictions.extend(sample["predictions"])

        accuracy = evaluate_predictions(dataset, predictions, self.config.model.get("task", "detect"))
        latencies_ms = [sample["latency_ms"] for sample in measured_samples]
        energy_j = [sample.get("energy_j") for sample in measured_samples if sample.get("energy_j") is not None]

        finished_at = datetime.now(timezone.utc).isoformat()
        thermal = device.thermal_snapshot()
        return {
            "schema_version": "edge-vlm-bench/v1",
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "host": {
                "platform": platform.platform(),
                "python": platform.python_version(),
            },
            "config": asdict(self.config),
            "config_sha256": canonical_config_hash(self.config),
            "device": env,
            "model": backend_versions,
            "protocol": {
                "cold_start_included": True,
                "warmup_runs": self.config.warmup_runs,
                "measured_runs": self.config.measured_runs,
                "batch_size": self.config.batch_size,
                "input_size": list(self.config.input_size),
                "include_preprocess_in_latency": self.config.include_prepost,
                "include_postprocess_in_latency": self.config.include_prepost,
            },
            "summary": {
                "fps": self.config.batch_size * 1000.0 / max(sum(latencies_ms) / len(latencies_ms), 1e-9),
                "latency_ms_p50": percentile(latencies_ms, 50),
                "latency_ms_p95": percentile(latencies_ms, 95),
                "cold_start_latency_ms": cold_sample["latency_ms"],
                "ram_peak_mb": max(s["ram_mb"] for s in measured_samples + [cold_sample]),
                "vram_peak_mb": max(s.get("vram_mb") or 0 for s in measured_samples + [cold_sample]),
                "energy_j_mean": (sum(energy_j) / len(energy_j)) if energy_j else None,
                "power_w_mean": device.mean_power_w(),
                **accuracy,
            },
            "thermal": thermal,
            "samples": measured_samples,
        }

    def _measure_one(self, model: Any, batch: list[Any], device: Any) -> dict[str, Any]:
        energy_before = device.energy_j()
        resource_before = device.resources()
        start = time.perf_counter()
        if self.config.include_prepost:
            prepared = model.preprocess(batch)
            raw = model.infer(prepared)
            predictions = model.postprocess(raw)
        else:
            prepared = model.preprocess(batch)
            start = time.perf_counter()
            raw = model.infer(prepared)
            predictions = model.postprocess(raw)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        resource_after = device.resources()
        energy_after = device.energy_j()
        return {
            "latency_ms": elapsed_ms,
            "ram_mb": max(resource_before.ram_mb, resource_after.ram_mb),
            "vram_mb": max(resource_before.vram_mb or 0, resource_after.vram_mb or 0),
            "energy_j": (
                energy_after - energy_before
                if energy_before is not None and energy_after is not None
                else None
            ),
            "predictions": predictions,
        }

