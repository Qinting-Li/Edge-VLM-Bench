from edge_vlm_bench.metrics.latency import percentile


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([1, 2, 3, 4], 95) == 3.8499999999999996

