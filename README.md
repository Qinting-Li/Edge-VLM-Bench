# Edge-VLM-Bench

Edge-VLM-Bench is a Python, CLI, and Docker-based benchmark framework for testing whether YOLO, SAM/SAM2, and VLM models are actually deployable on edge devices.

It focuses on the questions that cloud GPU benchmarks often miss:

* Can the model run on real devices such as Jetson, RK3588, macOS, or Android?
* What is the true end-to-end latency, including preprocessing and postprocessing?
* How much RAM, VRAM, power, and energy does it use?
* How much accuracy is lost after quantization, backend conversion, or edge runtime deployment?
* Which backend gives the best tradeoff for a fixed device, input size, batch size, and thermal policy?

## What It Measures

Each benchmark run records:

* Cold-start latency, warmup runs, and measured inference runs.
* FPS, P50/P95 latency, RAM, VRAM, power, and energy.
* mAP or IoU, depending on the task.
* Device metadata, thermal policy, throttling notes, and energy source.
* Model type, weights, backend, version, and backend package versions.
* Dataset configuration, metric method, per-sample results, and SHA256 hashes.

Each run produces reproducible, tamper-evident outputs. `result.json` contains canonical `config_sha256` and `result_sha256` values, and the same result hash is also written into `summary.csv` and `report.md`.

## Project Structure

Edge-VLM-Bench is organized around five layers:

* **Device adapters**: Jetson, RK3588, macOS, Android, and generic local runs.
* **Model adapters**: YOLO, SAM/SAM2, VLM, dummy, and torch_dummy.
* **Dataset evaluation**: COCO-style annotations and custom JSON manifests.
* **Metric collection**: latency, FPS, memory, energy, accuracy, thermal state, and throttling metadata.
* **Reports**: JSON, CSV, and Markdown outputs for analysis and leaderboard ingestion.

## Quick Start

```bash
python -m pip install -e ".[dev]"
python bench.py run --config examples/demo_config.yaml --output runs/demo --demo
```

The demo uses the deterministic `dummy` adapter, so it can validate the benchmark pipeline without model weights, device SDKs, or edge hardware.

You can also use the CLI directly:

```bash
edge-vlm-bench run --config examples/demo_config.yaml --output runs/demo --demo
edge-vlm-bench report --result runs/demo/result.json
```

## Docker

```bash
docker build -t edge-vlm-bench .
docker run --rm -v "$PWD/runs:/app/runs" edge-vlm-bench
```

For production runs on Jetson, RK3588, or Android, mount the required model weights, datasets, telemetry files, SDK libraries, and runtime dependencies into the container.

## Example Configuration

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

Current device adapters provide a common metadata and telemetry interface:

* `generic`: local CPU process RAM and platform metadata.
* `jetson`: `tegrastats`, `nvpmodel`, RAM, and thermal snapshots.
* `rk3588`: RKNPU load hook and SoC metadata.
* `macos`: Apple platform metadata and `powermetrics` guidance.
* `android`: ADB device properties and battery telemetry.

For leaderboard-quality energy results, use a consistent measurement source within each track. On-device telemetry is useful for development; external power meters are preferred for final published numbers.

## Model Adapters

Implemented adapters:

* `dummy`: deterministic backend for CI and demos.
* `torch_dummy`: PyTorch CUDA matmul backend for validating device selection, VRAM reporting, and benchmark plumbing without downloading model weights.
* `yolo`: Ultralytics YOLO adapter.

Planned or external adapters:

* `sam` / `sam2`: segmentation benchmark interface.
* `vlm`: prompt-based VLM benchmark interface.

For deployment-specific backends, use `backend: external`. The benchmark sends a JSON payload to the configured command through stdin and expects a JSON list of predictions from stdout.

This keeps the benchmark protocol stable while allowing integration with TensorRT, ONNX Runtime, RKNN, MNN, Core ML, MLX, ExecuTorch, llama.cpp, Android NNAPI, and vendor SDK scripts.

```yaml
model:
  type: vlm
  name: qwen2.5-vl-edge
  task: vlm
  backend: external
  external_command: ["python", "deploy/run_vlm.py"]
  prompt: "Describe safety-relevant objects in the image."
```

## RTX 6000 Smoke Test

Use the included RTX 6000 config when the target CUDA device is device 1:

```bash
python bench.py run --config examples/rtx6000_torch_config.yaml --output runs/rtx6000 --demo
```

A successful report should include:

* `torch_cuda_name: Quadro RTX 6000` in model metadata.
* `nvidia_gpu.name: Quadro RTX 6000` in device metadata.

## Dataset Evaluation

COCO-style datasets are supported through JSON annotation files. For smoke tests, Edge-VLM-Bench provides a lightweight mAP proxy.

For publication-grade COCO evaluation, install:

```bash
python -m pip install -e ".[coco]"
```

Then extend `edge_vlm_bench/metrics/accuracy.py` to call `pycocotools` using the prediction JSON produced by the model adapter.

Custom datasets can use a simple JSON manifest:

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

## Outputs

Each run writes three files:

* `result.json`: full protocol, metadata, per-sample measurements, and hashes.
* `summary.csv`: one-row summary for leaderboard ingestion.
* `report.md`: human-readable benchmark report.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest -q
python bench.py run --config examples/demo_config.yaml --output runs/dev --demo
```

## Roadmap

* Add production COCO mAP through `pycocotools`.
* Add SAM/SAM2 mask IoU evaluation.
* Add TensorRT, ONNX Runtime, RKNN, MNN, Core ML, MLX, ExecuTorch, and llama.cpp adapters.
* Add external power-meter ingestion for reproducible joules-per-inference.
* Add signed leaderboard submissions and schema validation.
* Add Android APK harness for NNAPI, GPU, and NPU runs.
