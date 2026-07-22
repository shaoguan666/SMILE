"""Auditable, manifest-based sensor-removal evaluation utilities.

This module is deliberately independent of the ordinary training collator.  A
manifest fixes one sensor permutation per (patient, corruption replicate), and
larger removal conditions take longer prefixes of that same permutation.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset


PROTOCOL_VERSION = "smile-sensor-removal-v1"
ROW_KEY = "__sensor_manifest_row__"
DEFAULT_KS = (0, 2, 3, 5, 7, 8)
DEFAULT_REPLICATES = 5
DEFAULT_MANIFEST_SEED = 20260720


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _base_dataset_and_indices(dataset: Dataset) -> tuple[Dataset, list[int]]:
    if isinstance(dataset, Subset):
        base, parent_indices = _base_dataset_and_indices(dataset.dataset)
        return base, [parent_indices[int(index)] for index in dataset.indices]
    return dataset, list(range(len(dataset)))


def dataset_patient_ids(dataset: Dataset) -> tuple[str, ...]:
    base, indices = _base_dataset_and_indices(dataset)
    patient_ids = getattr(base, "patient_ids", None)
    if patient_ids is None:
        return tuple(f"index:{index}" for index in indices)
    return tuple(str(patient_ids[index]) for index in indices)


def dataset_feature_names(dataset: Dataset) -> tuple[str, ...]:
    base, _ = _base_dataset_and_indices(dataset)
    names = getattr(base, "feature_names", None)
    if names:
        return tuple(str(name) for name in names)
    sample = dataset[0]
    return tuple(f"feature_{index}" for index in range(np.asarray(sample["x"]).shape[-1]))


def _patient_permutation(
    dataset: str,
    split_seed: int,
    patient_id: str,
    replicate: int,
    manifest_seed: int,
    num_features: int,
) -> np.ndarray:
    material = (
        f"{PROTOCOL_VERSION}\0{dataset}\0{split_seed}\0{patient_id}\0"
        f"{replicate}\0{manifest_seed}"
    ).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(material).digest()[:16], "little")
    return np.random.default_rng(seed).permutation(num_features)


@dataclass(frozen=True)
class SensorManifest:
    path: Path
    dataset: str
    split_seed: int
    manifest_seed: int
    patient_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    permutations: np.ndarray
    file_sha256: str

    @property
    def num_replicates(self) -> int:
        return int(self.permutations.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.permutations.shape[2])

    def validate_condition(self, replicate: int, k: int) -> None:
        if not 0 <= replicate < self.num_replicates:
            raise ValueError(
                f"replicate={replicate} outside [0, {self.num_replicates})"
            )
        if not 0 <= k <= self.num_features:
            raise ValueError(f"k={k} outside [0, {self.num_features}]")

    def validate_dataset(self, dataset: Dataset) -> None:
        ids = dataset_patient_ids(dataset)
        names = dataset_feature_names(dataset)
        positions = {patient_id: index for index, patient_id in enumerate(self.patient_ids)}
        missing = [patient_id for patient_id in ids if patient_id not in positions]
        if missing:
            raise ValueError(f"Dataset patients absent from manifest: {missing[:3]}")
        if names != self.feature_names:
            raise ValueError(
                "Feature registry differs from manifest: "
                f"dataset={names}, manifest={self.feature_names}"
            )

    def row_lookup(self) -> dict[str, int]:
        return {patient_id: index for index, patient_id in enumerate(self.patient_ids)}


def create_manifest(
    path: Path,
    *,
    dataset_name: str,
    split_seed: int,
    test_dataset: Dataset,
    replicates: int = DEFAULT_REPLICATES,
    manifest_seed: int = DEFAULT_MANIFEST_SEED,
    overwrite: bool = False,
) -> SensorManifest:
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if path.exists() and not overwrite:
        manifest = load_manifest(path)
        manifest.validate_dataset(test_dataset)
        return manifest

    patient_ids = dataset_patient_ids(test_dataset)
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError("Patient IDs must be unique within the test split")
    feature_names = dataset_feature_names(test_dataset)
    num_features = len(feature_names)
    dtype = np.int16 if num_features < np.iinfo(np.int16).max else np.int32
    permutations = np.empty(
        (replicates, len(patient_ids), num_features), dtype=dtype
    )
    for replicate in range(replicates):
        for row, patient_id in enumerate(patient_ids):
            permutations[replicate, row] = _patient_permutation(
                dataset_name,
                split_seed,
                patient_id,
                replicate,
                manifest_seed,
                num_features,
            )

    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": dataset_name,
        "split": "test",
        "split_seed": int(split_seed),
        "manifest_seed": int(manifest_seed),
        "replicates": int(replicates),
        "num_patients": len(patient_ids),
        "num_features": num_features,
    }
    _atomic_npz(
        path,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        patient_ids=np.asarray(patient_ids, dtype=np.str_),
        feature_names=np.asarray(feature_names, dtype=np.str_),
        permutations=permutations,
    )
    return load_manifest(path)


def load_manifest(path: Path) -> SensorManifest:
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        patient_ids = tuple(str(value) for value in archive["patient_ids"].tolist())
        feature_names = tuple(str(value) for value in archive["feature_names"].tolist())
        permutations = np.asarray(archive["permutations"]).copy()
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(
            f"Unsupported sensor manifest protocol: {metadata.get('protocol_version')!r}"
        )
    expected_shape = (
        int(metadata["replicates"]),
        int(metadata["num_patients"]),
        int(metadata["num_features"]),
    )
    if permutations.shape != expected_shape:
        raise ValueError(
            f"Manifest permutation shape {permutations.shape} != {expected_shape}"
        )
    expected = np.arange(expected_shape[-1])
    if not np.all(np.sort(permutations, axis=-1) == expected):
        raise ValueError("Manifest contains a row that is not a sensor permutation")
    return SensorManifest(
        path=path,
        dataset=str(metadata["dataset"]),
        split_seed=int(metadata["split_seed"]),
        manifest_seed=int(metadata["manifest_seed"]),
        patient_ids=patient_ids,
        feature_names=feature_names,
        permutations=permutations,
        file_sha256=sha256_file(path),
    )


class ManifestIndexedDataset(Dataset):
    """Evaluation-only view that attaches a manifest row to each sample."""

    def __init__(self, dataset: Dataset, manifest: SensorManifest):
        manifest.validate_dataset(dataset)
        lookup = manifest.row_lookup()
        self.dataset = dataset
        self.rows = tuple(lookup[patient_id] for patient_id in dataset_patient_ids(dataset))
        self.patient_ids = dataset_patient_ids(dataset)
        self.feature_names = dataset_feature_names(dataset)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = dict(self.dataset[index])
        if ROW_KEY in sample:
            raise KeyError(f"Reserved sensor robustness key already present: {ROW_KEY}")
        sample[ROW_KEY] = self.rows[index]
        return sample


class SensorDropCollator:
    """Pickle-safe collator applying one manifest condition to x and mask."""

    def __init__(
        self,
        base_collate: Callable,
        permutations: np.ndarray,
        *,
        replicate: int,
        k: int,
    ):
        self.base_collate = base_collate
        self.permutations = np.asarray(permutations)
        self.replicate = int(replicate)
        self.k = int(k)

    def __call__(self, features):
        batch = self.base_collate(features)
        rows = batch.pop(ROW_KEY).long().cpu().numpy()
        if self.k == 0:
            return batch
        for batch_row, manifest_row in enumerate(rows):
            columns = torch.as_tensor(
                self.permutations[self.replicate, manifest_row, : self.k],
                dtype=torch.long,
            )
            batch["x"][batch_row, :, columns] = 0
            batch["mask"][batch_row, :, columns] = 0
        return batch


def make_sensor_loader(
    dataset: Dataset,
    *,
    base_collate: Callable,
    manifest: SensorManifest,
    replicate: int,
    k: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool | None = None,
) -> DataLoader:
    manifest.validate_condition(replicate, k)
    view = ManifestIndexedDataset(dataset, manifest)
    collator = SensorDropCollator(
        base_collate, manifest.permutations, replicate=replicate, k=k
    )
    return DataLoader(
        view,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available() if pin_memory is None else pin_memory,
    )


def condition_audit(
    dataset: Dataset, manifest: SensorManifest, *, replicate: int, k: int
) -> dict[str, int | float]:
    manifest.validate_condition(replicate, k)
    lookup = manifest.row_lookup()
    observed_before = 0
    removed_observed = 0
    selected_empty_channels = 0
    for index, patient_id in enumerate(dataset_patient_ids(dataset)):
        mask = np.asarray(dataset[index]["mask"])
        row = lookup[patient_id]
        columns = manifest.permutations[replicate, row, :k]
        observed_before += int(mask.sum())
        if k:
            selected = mask[:, columns]
            removed_observed += int(selected.sum())
            selected_empty_channels += int(np.sum(selected.sum(axis=0) == 0))
    return {
        "observed_entries_before": observed_before,
        "removed_observed_entries": removed_observed,
        "effective_observed_removal_fraction": (
            float(removed_observed / observed_before) if observed_before else 0.0
        ),
        "selected_empty_channels": selected_empty_channels,
    }


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    if labels.size != scores.size:
        raise ValueError("labels and scores must have equal length")
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "n_patients": int(labels.size),
        "positive_rate": float(labels.mean()),
    }


def condition_directory(root: Path, k: int, replicate: int) -> Path:
    return Path(root) / f"k_{k}" / ("clean" if k == 0 else f"replicate_{replicate}")


def run_condition_grid(
    *,
    evaluator: Callable[[DataLoader], tuple[np.ndarray, np.ndarray]],
    test_dataset: Dataset,
    base_collate: Callable,
    manifest: SensorManifest,
    ks: Sequence[int],
    replicates: Sequence[int],
    batch_size: int,
    num_workers: int,
    output_root: Path,
    model: str,
    dataset_name: str,
    train_seed: int,
    split_seed: int,
    checkpoint_path: Path,
    resume: bool = False,
) -> list[dict]:
    manifest.validate_dataset(test_dataset)
    if manifest.dataset != dataset_name or manifest.split_seed != split_seed:
        raise ValueError("Manifest dataset/split metadata does not match the evaluation")
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint_sha256 = sha256_file(checkpoint_path)
    conditions = [(0, 0)] if 0 in ks else []
    conditions.extend(
        (int(k), int(replicate))
        for k in ks
        if int(k) != 0
        for replicate in replicates
    )
    results = []
    for k, replicate in conditions:
        destination = condition_directory(output_root, k, replicate)
        run_path = destination / "run.json"
        if resume and run_path.exists():
            existing = json.loads(run_path.read_text(encoding="utf-8"))
            if (
                existing.get("manifest_sha256") == manifest.file_sha256
                and existing.get("checkpoint_sha256") == checkpoint_sha256
                and existing.get("status") == "complete"
            ):
                results.append(existing)
                continue
            raise ValueError(f"Resume artifact hash mismatch: {run_path}")

        loader = make_sensor_loader(
            test_dataset,
            base_collate=base_collate,
            manifest=manifest,
            replicate=replicate,
            k=k,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        labels, scores = evaluator(loader)
        labels = np.asarray(labels).reshape(-1)
        scores = np.asarray(scores).reshape(-1)
        patient_ids = np.asarray(dataset_patient_ids(test_dataset), dtype=np.str_)
        if labels.size != len(patient_ids):
            raise ValueError(
                f"Prediction count {labels.size} != patient count {len(patient_ids)}"
            )
        metrics = binary_metrics(labels, scores)
        audit = condition_audit(test_dataset, manifest, replicate=replicate, k=k)
        predictions_path = destination / "predictions.npz"
        _atomic_npz(
            predictions_path,
            patient_ids=patient_ids,
            labels=labels.astype(np.int64),
            scores=scores.astype(np.float64),
        )
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "status": "complete",
            "model": model,
            "dataset": dataset_name,
            "train_seed": int(train_seed),
            "split_seed": int(split_seed),
            "k": int(k),
            "num_features": manifest.num_features,
            "sensor_removal_fraction": float(k / manifest.num_features),
            "corruption_replicate": int(replicate),
            "manifest": str(manifest.path),
            "manifest_sha256": manifest.file_sha256,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "predictions": str(predictions_path.resolve()),
            "metrics": metrics,
            "audit": audit,
        }
        _atomic_json(run_path, payload)
        results.append(payload)
    return results


def parse_int_list(values: Iterable[int], *, name: str) -> tuple[int, ...]:
    parsed = tuple(dict.fromkeys(int(value) for value in values))
    if not parsed:
        raise ValueError(f"{name} cannot be empty")
    return parsed

