FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY edge_vlm_bench ./edge_vlm_bench
COPY bench.py ./bench.py
COPY examples ./examples
RUN pip install --no-cache-dir .

ENTRYPOINT ["edge-vlm-bench"]
CMD ["run", "--config", "examples/demo_config.yaml", "--output", "runs/demo", "--demo"]

