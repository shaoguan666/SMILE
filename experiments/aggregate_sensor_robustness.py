"""Aggregate, bootstrap, and reconcile sensor-removal evaluation artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = (
    "smart-smile-lean", "smart", "ists_plm", "wavegnn", "misstm", "atenet"
)
DEFAULT_DATASETS = ("mimic_mortality", "mimic_decompensation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root", type=Path,
        default=ROOT / "export" / "sensor_robustness_v1" / "runs"
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "export" / "sensor_robustness_v1" / "aggregate.json"
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 42, 3407])
    parser.add_argument("--ks", nargs="+", type=int, default=[0, 2, 3, 5, 7, 8])
    parser.add_argument("--replicates", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260720)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--clean-reference", type=Path,
        default=ROOT / "experiments" / "sensor_robustness_clean_reference.json"
    )
    return parser.parse_args()


def discover_runs(input_root: Path) -> list[dict]:
    runs = []
    for path in input_root.rglob("run.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["artifact"] = str(path.resolve())
        runs.append(payload)
    return runs


def expected_keys(args: argparse.Namespace) -> set[tuple]:
    keys = set()
    for dataset in args.datasets:
        for model in args.models:
            for seed in args.seeds:
                if 0 in args.ks:
                    keys.add((dataset, model, seed, 0, 0))
                for k in args.ks:
                    if k:
                        for replicate in args.replicates:
                            keys.add((dataset, model, seed, k, replicate))
    return keys


def run_key(run: dict) -> tuple:
    return (
        run["dataset"], run["model"], int(run["train_seed"]), int(run["k"]),
        int(run["corruption_replicate"]),
    )


def _prediction(run: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions_path = Path(run["predictions"])
    if not predictions_path.exists():
        # run.json may carry an absolute path produced on another host;
        # the predictions.npz always sits next to the run.json artifact.
        predictions_path = Path(run["artifact"]).parent / "predictions.npz"
    with np.load(predictions_path, allow_pickle=False) as archive:
        return (
            np.asarray(archive["patient_ids"]).astype(str),
            np.asarray(archive["labels"]).reshape(-1),
            np.asarray(archive["scores"]).reshape(-1),
        )


def _stratified_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    parts = []
    for label in np.unique(labels):
        pool = np.flatnonzero(labels == label)
        parts.append(rng.choice(pool, size=len(pool), replace=True))
    return np.concatenate(parts)


def _ci(values: list[float]) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def hierarchical_bootstrap(
    group_runs: list[dict], clean_runs: dict[int, dict], *, samples: int, seed: int
) -> dict:
    if samples <= 0:
        return {}
    by_seed = defaultdict(list)
    for run in group_runs:
        by_seed[int(run["train_seed"])].append(run)
    seeds = sorted(by_seed)
    if not seeds:
        return {}
    cache = {}
    for run in group_runs + [clean_runs[seed_value] for seed_value in seeds]:
        cache[run["artifact"]] = _prediction(run)
    rng = np.random.default_rng(seed)
    auprc_values = []
    retention_values = []
    for _ in range(samples):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        seed_auprcs = []
        seed_retentions = []
        for seed_value in sampled_seeds:
            candidates = by_seed[int(seed_value)]
            sampled_runs = rng.choice(candidates, size=len(candidates), replace=True)
            clean = clean_runs[int(seed_value)]
            clean_ids, clean_labels, clean_scores = cache[clean["artifact"]]
            patient_indices = _stratified_indices(clean_labels, rng)
            clean_ap = average_precision_score(
                clean_labels[patient_indices], clean_scores[patient_indices]
            )
            prevalence = float(clean_labels[patient_indices].mean())
            replicate_aps = []
            replicate_retention = []
            for run in sampled_runs:
                ids, labels, scores = cache[run["artifact"]]
                if not np.array_equal(ids, clean_ids) or not np.array_equal(labels, clean_labels):
                    raise ValueError("Prediction patient order/labels differ within a paired group")
                ap = average_precision_score(labels[patient_indices], scores[patient_indices])
                replicate_aps.append(float(ap))
                denominator = clean_ap - prevalence
                replicate_retention.append(
                    float((ap - prevalence) / denominator) if denominator else float("nan")
                )
            seed_auprcs.append(float(np.mean(replicate_aps)))
            seed_retentions.append(float(np.nanmean(replicate_retention)))
        auprc_values.append(float(np.mean(seed_auprcs)))
        retention_values.append(float(np.nanmean(seed_retentions)))
    return {"auprc_95ci": _ci(auprc_values), "retention_95ci": _ci(retention_values)}


def paired_difference_bootstrap(
    ours: list[dict], baseline: list[dict], *, samples: int, seed: int
) -> dict:
    if samples <= 0:
        return {}
    ours_map = {(int(run["train_seed"]), int(run["corruption_replicate"])): run for run in ours}
    base_map = {(int(run["train_seed"]), int(run["corruption_replicate"])): run for run in baseline}
    common = sorted(set(ours_map) & set(base_map))
    seeds = sorted({key[0] for key in common})
    if not common or len(common) != len(ours_map) or len(common) != len(base_map):
        return {"status": "unpaired_or_incomplete"}
    cache = {run["artifact"]: _prediction(run) for run in ours + baseline}
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(samples):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        seed_differences = []
        for seed_value in sampled_seeds:
            reps = [key[1] for key in common if key[0] == seed_value]
            sampled_reps = rng.choice(reps, size=len(reps), replace=True)
            rep_differences = []
            for replicate in sampled_reps:
                ours_run = ours_map[(int(seed_value), int(replicate))]
                base_run = base_map[(int(seed_value), int(replicate))]
                ours_ids, labels, ours_scores = cache[ours_run["artifact"]]
                base_ids, base_labels, base_scores = cache[base_run["artifact"]]
                if not np.array_equal(ours_ids, base_ids) or not np.array_equal(labels, base_labels):
                    raise ValueError("Methods do not share patient order/labels")
                indices = _stratified_indices(labels, rng)
                rep_differences.append(
                    float(
                        average_precision_score(labels[indices], ours_scores[indices])
                        - average_precision_score(labels[indices], base_scores[indices])
                    )
                )
            seed_differences.append(float(np.mean(rep_differences)))
        differences.append(float(np.mean(seed_differences)))
    return {
        "auprc_difference_mean": float(np.mean(differences)),
        "auprc_difference_95ci": _ci(differences),
    }


def aggregate_group(runs: list[dict], clean_by_seed: dict[int, dict]) -> dict:
    by_seed = defaultdict(list)
    for run in runs:
        by_seed[int(run["train_seed"])].append(run)
    per_seed = []
    replicate_stds = []
    for seed_value in sorted(by_seed):
        values = [float(run["metrics"]["auprc"]) for run in by_seed[seed_value]]
        clean = clean_by_seed[seed_value]
        prevalence = float(clean["metrics"]["positive_rate"])
        clean_auprc = float(clean["metrics"]["auprc"])
        denominator = clean_auprc - prevalence
        retentions = [
            (value - prevalence) / denominator if denominator else float("nan")
            for value in values
        ]
        per_seed.append(
            {"seed": seed_value, "auprc": statistics.mean(values),
             "retention": statistics.mean(retentions)}
        )
        if len(values) > 1:
            replicate_stds.append(statistics.stdev(values))
    auprcs = [row["auprc"] for row in per_seed]
    retentions = [row["retention"] for row in per_seed]
    return {
        "per_seed": per_seed,
        "auprc_mean": statistics.mean(auprcs),
        "auprc_sample_std": statistics.stdev(auprcs) if len(auprcs) > 1 else None,
        "retention_mean": statistics.mean(retentions),
        "retention_sample_std": statistics.stdev(retentions) if len(retentions) > 1 else None,
        "mean_within_seed_replicate_std": (
            statistics.mean(replicate_stds) if replicate_stds else 0.0
        ),
        "n_train_seeds": len(per_seed),
        "n_runs": len(runs),
    }


def reconcile(aggregates: dict, reference_path: Path) -> dict:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    checks = []
    for dataset, models in reference.items():
        if dataset == "source":
            continue
        for model, expected in models.items():
            actual = aggregates.get(dataset, {}).get(model, {}).get("0")
            if actual is None:
                checks.append({"dataset": dataset, "model": model, "status": "missing"})
                continue
            actual_mean = round(actual["auprc_mean"] * 100, 2)
            actual_std = (
                round(actual["auprc_sample_std"] * 100, 2)
                if actual["auprc_sample_std"] is not None else None
            )
            passed = (
                actual_mean == expected["auprc_mean"]
                and actual_std == expected["auprc_sample_std"]
            )
            checks.append(
                {"dataset": dataset, "model": model, "status": "pass" if passed else "mismatch",
                 "expected": expected,
                 "actual": {"auprc_mean": actual_mean, "auprc_sample_std": actual_std}}
            )
    return {"passed": bool(checks) and all(row["status"] == "pass" for row in checks),
            "checks": checks, "reference": str(reference_path.resolve())}


def main() -> None:
    args = parse_args()
    discovered = discover_runs(args.input_root)
    selected = [
        run for run in discovered
        if run.get("status") == "complete"
        and run.get("dataset") in args.datasets
        and run.get("model") in args.models
        and int(run.get("train_seed")) in args.seeds
        and int(run.get("k")) in args.ks
    ]
    keyed = {run_key(run): run for run in selected}
    expected = expected_keys(args)
    missing = sorted(expected - set(keyed))
    unexpected = sorted(set(keyed) - expected)
    complete = not missing
    if missing and not args.allow_incomplete:
        raise SystemExit(f"Incomplete sensor robustness grid: {len(missing)} missing conditions")

    aggregates = defaultdict(lambda: defaultdict(dict))
    comparisons = defaultdict(lambda: defaultdict(dict))
    for dataset in args.datasets:
        for model in args.models:
            model_runs = [
                run for run in selected if run["dataset"] == dataset and run["model"] == model
            ]
            clean_by_seed = {
                int(run["train_seed"]): run for run in model_runs if int(run["k"]) == 0
            }
            for k in args.ks:
                group = [run for run in model_runs if int(run["k"]) == k]
                if not group or any(seed not in clean_by_seed for seed in {int(r["train_seed"]) for r in group}):
                    continue
                block = aggregate_group(group, clean_by_seed)
                block.update(
                    hierarchical_bootstrap(
                        group, clean_by_seed, samples=args.bootstrap_samples,
                        seed=args.bootstrap_seed + k,
                    )
                )
                aggregates[dataset][model][str(k)] = block

        ours_model = "smart-smile-lean"
        for baseline in args.models:
            if baseline == ours_model:
                continue
            for k in args.ks:
                ours = [
                    run for run in selected if run["dataset"] == dataset
                    and run["model"] == ours_model and int(run["k"]) == k
                ]
                other = [
                    run for run in selected if run["dataset"] == dataset
                    and run["model"] == baseline and int(run["k"]) == k
                ]
                if ours and other:
                    comparisons[dataset][baseline][str(k)] = paired_difference_bootstrap(
                        ours, other, samples=args.bootstrap_samples,
                        seed=args.bootstrap_seed + 1000 + k,
                    )

    aggregate_dict = {
        dataset: {model: dict(values) for model, values in models.items()}
        for dataset, models in aggregates.items()
    }
    reconciliation = reconcile(aggregate_dict, args.clean_reference)
    payload = {
        "complete": complete,
        "planned_conditions": len(expected),
        "completed_conditions": len(expected & set(keyed)),
        "missing_conditions": [list(key) for key in missing],
        "unexpected_conditions": [list(key) for key in unexpected],
        "runs": selected,
        "aggregates": aggregate_dict,
        "paired_comparisons_vs_smile": {
            dataset: {model: dict(values) for model, values in models.items()}
            for dataset, models in comparisons.items()
        },
        "clean_reconciliation": reconciliation,
        "publication_ready": complete and reconciliation["passed"],
        "bootstrap": {"samples": args.bootstrap_samples, "seed": args.bootstrap_seed,
                      "stratified_by_label": True, "hierarchical_seed_replicate": True},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {args.output}; complete={complete}; "
        f"clean_reconciliation={reconciliation['passed']}; "
        f"publication_ready={payload['publication_ready']}"
    )


if __name__ == "__main__":
    main()

