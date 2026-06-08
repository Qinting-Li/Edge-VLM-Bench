from pathlib import Path
import json

from edge_vlm_bench.benchmark import BenchmarkRunner
from edge_vlm_bench.config import load_config
from edge_vlm_bench.reports.writer import write_reports


def test_demo_run_writes_reports(tmp_path: Path):
    config = load_config("examples/demo_config.yaml")
    result = BenchmarkRunner(config, demo=True).run()
    paths = write_reports(result, tmp_path)

    saved = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert saved["summary"]["fps"] > 0
    assert saved["summary"]["latency_ms_p95"] >= saved["summary"]["latency_ms_p50"]
    assert saved["config_sha256"]
    assert saved["result_sha256"]
    assert Path(paths["csv"]).exists()
    assert Path(paths["markdown"]).exists()

