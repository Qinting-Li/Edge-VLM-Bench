# Edge-VLM-Bench

Edge-VLM-Bench is a Python + CLI + Docker benchmark framework for answering the edge deployment question that cloud GPU papers often leave open: can YOLO, SAM/SAM2, and VLM models run on real devices, how fast, how much energy do they use, and how much accuracy is lost after deployment?

The project is organized around five layers:

- Device adapters: Jetson, RK3588, macOS, Android, and generic local runs.
- Model adapters: YOLO, SAM/SAM2, VLM, plus a deterministic dummy backend for CI and demos.
- Dataset evaluation: COCO-style annotations and custom JSON manifests.
- Metric collection: FPS, P50/P95 latency, RAM/VRAM, energy, mAP/IoU, thermal and throttling metadata.
- Reports and leaderboard inputs: JSON, CSV, and Markdown with config and result SHA256 hashes.

## Why This Exists

Many 2026 VLM, SAM, and YOLO papers report cloud GPU metrics but do not answer the engineering questions that determine deployment value:

- Does the model run on Jetson, RK3588, macOS, or Android?
- How fast is end-to-end inference when preprocessing and postprocessing are included?
- How much RAM, VRAM, power, and energy does it consume?
- How much mAP or IoU is lost after quantization, backend conversion, or edge runtime deployment?
- Which backend is the best tradeoff for a fixed device, input resolution, batch size, and thermal policy?

## Benchmark Protocol

Every run records:

- Cold start latency separately from warmup and measured runs.
- Fixed batch size and fixed input resolution.
- Whether preprocessing and postprocessing are included in measured latency.
- Device adapter, thermal policy, throttling notes, RAM/VRAM, and energy source.
- Model family, weights path, model version, backend, and installed backend package versions.
- Dataset configuration, metric method, config hash, result hash, and full per-sample measurements.

Result files are designed to be reproducible and tamper-evident. `result.json` contains a canonical `config_sha256` and `result_sha256`; `summary.csv` and `report.md` carry the same result hash for leaderboard ingestion.

## Quick Start

```bash
python -m pip install -e ".[dev]"
python bench.py run --config examples/demo_config.yaml --output runs/demo --demo
```

The demo uses the deterministic dummy adapter, so it validates the benchmark protocol without requiring model weights or device SDKs.

## CLI

```bash
edge-vlm-bench run --config examples/demo_config.yaml --output runs/demo --demo
edge-vlm-bench report --result runs/demo/result.json
```

The top-level `bench.py` provides the same entry point:

```bash
python bench.py run --config examples/demo_config.yaml --output runs/demo --demo
```

## Docker

```bash
docker build -t edge-vlm-bench .
docker run --rm -v "$PWD/runs:/app/runs" edge-vlm-bench
```

For Jetson, RK3588, and Android production runs, mount model weights, datasets, device telemetry files, and SDK runtime libraries into the container as required by the selected backend.

## Configuration

```yaml
device:
  type: jetson
  name: orin-nx-16gb
  thermal_policy: fixed_fan_record_clocks
  power_source: tegrastats

model:
  type: yolo
  name: yolov8n
  task: detect
  backend: ultralytics
  version: 8.x
  weights: weights/yolov8n.pt

dataset:
  type: coco
  annotation_file: data/coco/annotations/instances_val2017.json

protocol:
  batch_size: 1
  input_size: [640, 640]
  warmup_runs: 20
  measured_runs: 200
  include_prepost: true
```

## Device Adapters

Current adapters expose a common contract and record the available metadata:

- `generic`: local CPU process RAM and platform metadata.
- `jetson`: `tegrastats`, `nvpmodel`, RAM, and thermal snapshot hooks.
- `rk3588`: RKNPU load hook and SoC metadata.
- `macos`: Apple platform metadata and `powermetrics` guidance.
- `android`: ADB device properties and battery telemetry hook.

Production power/energy runs should use a consistent source per leaderboard track, such as on-device telemetry for exploratory runs and an external power meter for final published numbers.

## Model Adapters

Implemented:

- `dummy`: deterministic CI/demo backend.
- `yolo`: Ultralytics YOLO adapter.

Contract placeholders:

- `sam` / `sam2`: segmentation benchmark shape, with `backend: external` for deployment scripts.
- `vlm`: prompt-based VLM benchmark shape, with `backend: external` for deployment scripts.

For `backend: external`, the adapter sends a JSON payload to the configured command over stdin and expects a JSON list of predictions on stdout. This gives a stable contract for TensorRT, ONNX Runtime, RKNN, MNN, Core ML, MLX, ExecuTorch, llama.cpp, Android NNAPI, or vendor SDK scripts without changing the benchmark protocol.

```yaml
model:
  type: vlm
  name: qwen2.5-vl-edge
  task: vlm
  backend: external
  external_command: ["python", "deploy/run_vlm.py"]
  prompt: "Describe safety-relevant objects in the image."
```

## Dataset Evaluation

COCO evaluation supports JSON annotation loading and a lightweight mAP proxy for smoke tests. For publication-grade COCO metrics, install `.[coco]` and extend `edge_vlm_bench/metrics/accuracy.py` to call `pycocotools` with the same prediction JSON produced by the model adapter.

Custom datasets use a JSON manifest:

```json
[
  {
    "image_id": 1,
    "path": "images/example_001.jpg",
    "category_id": 1,
    "bbox": [10, 10, 100, 100]
  }
]
```

## Output

Each run writes:

- `result.json`: full protocol, metadata, per-sample measurements, hashes.
- `summary.csv`: one-row leaderboard-friendly summary.
- `report.md`: human-readable report.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest -q
python bench.py run --config examples/demo_config.yaml --output runs/dev --demo
```

## Roadmap

- Add production `pycocotools` COCO mAP and SAM mask IoU evaluators.
- Add TensorRT, ONNX Runtime, RKNN, MNN, Core ML, MLX, ExecuTorch, and llama.cpp adapters.
- Add external power-meter ingestion for reproducible Joules-per-inference.
- Add signed leaderboard submissions and schema validation.
- Add Android APK harness for NNAPI/GPU/NPU backend runs.
