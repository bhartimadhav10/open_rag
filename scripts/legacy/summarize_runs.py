"""Aggregate multiple bench_report JSON files → mean + std dev per metric.

Usage:
  python summarize_runs.py run1.json run2.json run3.json
"""
import json
import statistics
import sys
from pathlib import Path


def main(paths: list[str]):
    if not paths:
        print("usage: python summarize_runs.py <report1.json> [report2.json ...]")
        return
    reports = [json.loads(Path(p).read_text()) for p in paths]
    print(f"Aggregating {len(reports)} runs\n")
    print(f"{'stage':<8} {'metric':<6}  {'mean':>8} {'std':>8} {'min':>8} {'max':>8}  runs")
    print("-" * 72)

    stages = ["embed", "ann"]
    if any("rerank" in r["aggregate_ms"] for r in reports):
        stages.append("rerank")
    stages.append("total")

    for stage in stages:
        for metric in ["mean", "p50", "p95", "p99"]:
            vals = []
            for r in reports:
                if stage in r["aggregate_ms"]:
                    vals.append(r["aggregate_ms"][stage][metric])
            if not vals:
                continue
            m = statistics.fmean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            runs = ", ".join(f"{v:.2f}" for v in vals)
            print(f"{stage:<8} {metric:<6}  {m:>8.2f} {sd:>8.2f} {min(vals):>8.2f} {max(vals):>8.2f}  [{runs}]")


if __name__ == "__main__":
    main(sys.argv[1:])
