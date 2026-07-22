"""Shared utilities for locally adapted recent baselines."""

from __future__ import annotations

import json
import random
import statistics
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[2]
MIMIC_DATASETS = ("mimic_mortality", "mimic_decompensation")
STANDARD_SEEDS = (1, 42, 3407)


def add_sensor_robustness_args(parser) -> None:
    """Add the shared frozen-checkpoint sensor-removal evaluation flags."""
    parser.add_argument("--sensor-manifest", type=Path, default=None)
    parser.add_argument("--sensor-ks", nargs="+", type=int, default=[0, 2, 3, 5, 7, 8])
    parser.add_argument("--sensor-replicates", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--sensor-resume", action="store_true")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_limit(dataset, limit: int):
    if limit <= 0 or limit >= len(dataset):
        return dataset
    subset = torch.utils.data.Subset(dataset, range(limit))
    subset.feature_names = getattr(dataset, "feature_names", None)
    patient_ids = getattr(dataset, "patient_ids", None)
    if patient_ids is not None:
        subset.patient_ids = tuple(patient_ids[:limit])
    return subset


def load_mimic_splits(dataset: str, split_seed: int, limits: tuple[int, int, int]):
    from data.challenge2012 import load_challenge_2012
    from data.challenge2019 import load_challenge_2019
    from data.mimiciii import (
        load_mimic_iii_decompensation,
        load_mimic_iii_mortality,
    )

    loader_fn = {
        "mimic_mortality": load_mimic_iii_mortality,
        "mimic_decompensation": load_mimic_iii_decompensation,
        "c12": load_challenge_2012,
        "c19": load_challenge_2019,
    }[dataset]
    splits = loader_fn(split_seed=split_seed)
    return tuple(maybe_limit(split, limit) for split, limit in zip(splits, limits))


def make_loaders(
    splits,
    batch_size: int,
    collate_fn: Callable,
    num_workers: int,
    *,
    shuffle_train: bool = True,
    train_sampler=None,
):
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": torch.cuda.is_available(),
    }
    train, val, test = splits
    return (
        DataLoader(
            train,
            shuffle=shuffle_train if train_sampler is None else False,
            sampler=train_sampler,
            **common,
        ),
        DataLoader(val, shuffle=False, **common),
        DataLoader(test, shuffle=False, **common),
    )


def balanced_sampler(dataset) -> WeightedRandomSampler:
    if isinstance(dataset, torch.utils.data.Subset):
        labels = np.asarray(
            [int(dataset.dataset.data[index]["labels"]) for index in dataset.indices]
        )
    else:
        labels = np.asarray([int(sample["labels"]) for sample in dataset.data])
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    class_weights = 1.0 / np.maximum(counts, 1.0)
    return WeightedRandomSampler(
        torch.as_tensor(class_weights[labels], dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )


def binary_metrics(labels: list[np.ndarray], scores: list[np.ndarray]) -> dict[str, float]:
    labels_np, scores_np = binary_prediction_arrays(labels, scores)
    return {
        "auroc": float(roc_auc_score(labels_np, scores_np)),
        "auprc": float(average_precision_score(labels_np, scores_np)),
        "n": int(labels_np.size),
        "positive_rate": float(labels_np.mean()),
    }


def binary_prediction_arrays(
    labels: list[np.ndarray], scores: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate(labels).reshape(-1), np.concatenate(scores).reshape(-1)


def maybe_run_sensor_grid(
    args,
    *,
    evaluator,
    test_dataset,
    base_collate,
    batch_size: int,
    num_workers: int,
    output_dir: Path,
    model_name: str,
    checkpoint_path: Path,
) -> bool:
    """Run all requested manifest conditions after a checkpoint is loaded."""
    if args.sensor_manifest is None:
        return False
    if not args.eval_only:
        raise ValueError("--sensor-manifest requires --eval-only")
    if args.eval_output_dir is None:
        raise ValueError("--sensor-manifest requires --eval-output-dir")
    from experiments.sensor_robustness import load_manifest, run_condition_grid

    manifest = load_manifest(args.sensor_manifest)
    run_condition_grid(
        evaluator=evaluator,
        test_dataset=test_dataset,
        base_collate=base_collate,
        manifest=manifest,
        ks=args.sensor_ks,
        replicates=args.sensor_replicates,
        batch_size=batch_size,
        num_workers=num_workers,
        output_root=args.eval_output_dir,
        model=model_name,
        dataset_name=args.dataset,
        train_seed=args.seed,
        split_seed=args.split_seed,
        checkpoint_path=checkpoint_path,
        resume=args.sensor_resume,
    )
    return True


def upstream_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def json_ready_config(args) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def write_result(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


def aggregate_model(output_root: Path, model_slug: str, model_name: str, seeds: list[int]):
    outputs = []
    for dataset in MIMIC_DATASETS:
        runs = []
        for seed in seeds:
            path = output_root / dataset / model_slug / f"seed_{seed}" / "eval_results.json"
            with path.open(encoding="utf-8") as handle:
                result = json.load(handle)
            runs.append(
                {
                    "seed": seed,
                    "auroc": result["test"]["auroc"],
                    "auprc": result["test"]["auprc"],
                    "n": result["test"]["n"],
                    "artifact": str(path.relative_to(ROOT)),
                }
            )
        if len({run["n"] for run in runs}) != 1:
            raise ValueError(f"Inconsistent test sizes for {dataset}")
        aggregate = {
            "model": model_name,
            "dataset": dataset,
            "seeds": seeds,
            "split_seed": 42,
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
        destination = output_root / dataset / model_slug / "aggregate_seeds_1_42_3407.json"
        destination.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
        outputs.append(aggregate)
    return outputs
