"""Aggregate completed ATENet MIMIC runs without re-evaluating checkpoints."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "export" / "recent_baselines",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 42, 3407])
    return parser.parse_args()


def aggregate_dataset(output_root: Path, dataset: str, seeds: list[int]) -> dict:
    runs = []
    for seed in seeds:
        path = output_root / dataset / "atenet" / f"seed_{seed}" / "eval_results.json"
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        runs.append(
            {
                "seed": seed,
                "auroc": result["test"]["auroc"],
                "auprc": result["test"]["auprc"],
                "n": result["test"]["n"],
                "amp": result["config"]["amp"],
                "skipped_nonfinite_batches": sum(
                    epoch["skipped_nonfinite_batches"] for epoch in result["history"]
                ),
                "artifact": str(path.relative_to(ROOT)),
            }
        )

    if len({run["n"] for run in runs}) != 1:
        raise ValueError(f"Inconsistent test sizes for {dataset}")

    aggregate = {
        "model": "ATENet",
        "dataset": dataset,
        "seeds": seeds,
        "split_seed": 42,
        "precision": "fp32",
        "selection_metric": "validation AUPRC",
        "runs": runs,
        "test": {
            metric: {
                "mean": statistics.mean(run[metric] for run in runs),
                "sample_std": statistics.stdev(run[metric] for run in runs),
            }
            for metric in ("auroc", "auprc")
        },
    }
    destination = output_root / dataset / "atenet" / "aggregate_seeds_1_42_3407.json"
    destination.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    return aggregate


def main() -> None:
    args = parse_args()
    for dataset in ("mimic_mortality", "mimic_decompensation"):
        print(json.dumps(aggregate_dataset(args.output_root, dataset, args.seeds), indent=2))


if __name__ == "__main__":
    main()
