"""Paired statistical tests for SMILE BIBM experiments."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "results" / "bibm_audit_fixed"
DEFAULT_LONG = DEFAULT_RESULTS / "bibm_results_long.csv"
DEFAULT_OUT = DEFAULT_RESULTS / "significance_tests.csv"


def read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def marker(p: float | None) -> str:
    if p is None or math.isnan(p):
        return ""
    if p < 0.01:
        return "‡"
    if p < 0.05:
        return "†"
    return ""


def paired_tests(a: list[float], b: list[float]) -> tuple[float | None, float | None]:
    if len(a) != len(b) or len(a) < 2 or stats is None:
        return None, None
    diffs = [x - y for x, y in zip(a, b)]
    if all(abs(d) < 1e-12 for d in diffs):
        return 1.0, 1.0
    t_p = float(stats.ttest_rel(a, b).pvalue)
    try:
        w_p = float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
    except ValueError:
        w_p = 1.0
    return t_p, w_p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--long-csv", type=Path, default=DEFAULT_LONG)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--full", default="SMILE-Full")
    parser.add_argument("--baseline", default="Backbone")
    parser.add_argument("--capacity-control", default="Capacity-Control")
    args = parser.parse_args()

    rows = [r for r in read_rows(args.long_csv) if r.get("status") == "ok"]
    data = defaultdict(dict)
    for r in rows:
        key = (r["dataset"], r["metric"], r["variant_name"])
        data[key][int(r["seed"])] = float(r["value"])

    comparisons = [(args.full, args.baseline), (args.full, args.capacity_control)]
    out = []
    for dataset, metric, variant in sorted({(r["dataset"], r["metric"], r["variant_name"]) for r in rows}):
        if variant != args.full:
            continue
        for left, right in comparisons:
            left_vals = data.get((dataset, metric, left), {})
            right_vals = data.get((dataset, metric, right), {})
            seeds = sorted(set(left_vals) & set(right_vals))
            if len(seeds) < 2:
                out.append({
                    "dataset": dataset,
                    "metric": metric,
                    "comparison": f"{left} vs {right}",
                    "n_pairs": len(seeds),
                    "mean_diff": "",
                    "paired_t_p": "",
                    "wilcoxon_p": "",
                    "marker": "",
                    "status": "insufficient_pairs",
                })
                continue
            a = [left_vals[s] for s in seeds]
            b = [right_vals[s] for s in seeds]
            t_p, w_p = paired_tests(a, b)
            mean_diff = sum(x - y for x, y in zip(a, b)) / len(a)
            out.append({
                "dataset": dataset,
                "metric": metric,
                "comparison": f"{left} vs {right}",
                "n_pairs": len(seeds),
                "mean_diff": f"{mean_diff:.6f}",
                "paired_t_p": "" if t_p is None else f"{t_p:.6g}",
                "wilcoxon_p": "" if w_p is None else f"{w_p:.6g}",
                "marker": marker(t_p),
                "status": "ok" if t_p is not None else "scipy_unavailable",
            })

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["dataset", "metric", "comparison", "n_pairs", "mean_diff",
                      "paired_t_p", "wilcoxon_p", "marker", "status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out)
    print(f"Wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
