from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import hashlib
import json


def result_hash(result: dict[str, Any]) -> str:
    clean = dict(result)
    clean.pop("result_sha256", None)
    payload = json.dumps(clean, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_reports(result: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = dict(result)
    result["result_sha256"] = result_hash(result)
    json_path = out_dir / "result.json"
    csv_path = out_dir / "summary.csv"
    md_path = out_dir / "report.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(result, csv_path)
    _write_markdown(result, md_path)
    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}


def from_json(result_path: str | Path, out_dir: Path | None = None) -> dict[str, str]:
    result_path = Path(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return write_reports(result, out_dir or result_path.parent)


def _write_csv(result: dict[str, Any], path: Path) -> None:
    summary = result["summary"]
    row = {
        "run_id": result["run_id"],
        "device": result["device"].get("name"),
        "model": result["model"].get("name"),
        "backend": result["model"].get("backend"),
        "fps": summary.get("fps"),
        "latency_ms_p50": summary.get("latency_ms_p50"),
        "latency_ms_p95": summary.get("latency_ms_p95"),
        "ram_peak_mb": summary.get("ram_peak_mb"),
        "vram_peak_mb": summary.get("vram_peak_mb"),
        "energy_j_mean": summary.get("energy_j_mean"),
        "power_w_mean": summary.get("power_w_mean"),
        "mAP": summary.get("mAP"),
        "IoU": summary.get("IoU"),
        "result_sha256": result["result_sha256"],
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    summary = result["summary"]
    lines = [
        "# Edge VLM Benchmark Report",
        "",
        f"- Run ID: `{result['run_id']}`",
        f"- Config SHA256: `{result['config_sha256']}`",
        f"- Result SHA256: `{result['result_sha256']}`",
        f"- Device: `{result['device'].get('name')}`",
        f"- Model: `{result['model'].get('name')}`",
        f"- Backend: `{result['model'].get('backend')}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in (
        "fps",
        "latency_ms_p50",
        "latency_ms_p95",
        "cold_start_latency_ms",
        "ram_peak_mb",
        "vram_peak_mb",
        "energy_j_mean",
        "power_w_mean",
        "mAP",
        "IoU",
    ):
        lines.append(f"| {key} | {summary.get(key)} |")
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            "Latency includes preprocessing and postprocessing when configured. "
            "Cold start, warmup, measured runs, batch size, input size, device thermal state, "
            "model version, backend version, and result hashes are recorded for reproducibility.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
