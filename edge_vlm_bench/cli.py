from __future__ import annotations

from pathlib import Path
import argparse

from edge_vlm_bench.benchmark import BenchmarkRunner
from edge_vlm_bench.config import load_config
from edge_vlm_bench.reports.writer import from_json, write_reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edge-vlm-bench",
        description="Benchmark YOLO, SAM/SAM2, and VLM models on real edge devices.",
    )
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="run a benchmark from a YAML config")
    run.add_argument("--config", required=True, help="benchmark YAML file")
    run.add_argument("--output", default="runs/latest", help="output directory")
    run.add_argument("--demo", action="store_true", help="force demo-safe dummy assets")

    report = sub.add_parser("report", help="rebuild CSV/Markdown reports from result JSON")
    report.add_argument("--result", required=True, help="result JSON path")
    report.add_argument("--output", default=None, help="output directory")

    parser.add_argument("--version", action="store_true", help="print package version")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        from edge_vlm_bench import __version__

        print(__version__)
        return
    if args.command is None:
        parser.print_help()
        return
    if args.command == "run":
        config = load_config(args.config)
        result = BenchmarkRunner(config=config, demo=args.demo).run()
        out_dir = Path(args.output)
        paths = write_reports(result, out_dir)
        print(f"result_json={paths['json']}")
        print(f"result_csv={paths['csv']}")
        print(f"result_markdown={paths['markdown']}")
        return
    if args.command == "report":
        paths = from_json(args.result, Path(args.output) if args.output else None)
        print(f"result_csv={paths['csv']}")
        print(f"result_markdown={paths['markdown']}")
        return
    raise SystemExit(f"unknown command: {args.command}")
