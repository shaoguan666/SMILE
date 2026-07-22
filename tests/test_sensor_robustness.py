from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from data.dataloader import collate_fn
from experiments.aggregate_sensor_robustness import aggregate_group, reconcile
from experiments.run_sensor_robustness import (
    checkpoint_problem,
    resolved_devices,
    validate_local_devices,
)
from experiments.sensor_robustness import (
    ManifestIndexedDataset,
    SensorDropCollator,
    condition_audit,
    create_manifest,
    load_manifest,
    make_sensor_loader,
    run_condition_grid,
)


class ToyDataset(Dataset):
    def __init__(self):
        self.feature_names = ("a", "b", "c", "d")
        self.patient_ids = ("p0", "p1", "p2", "p3")
        self.data = []
        for index in range(4):
            mask = np.ones((3, 4), dtype=np.float32)
            if index == 0:
                mask[:, 3] = 0
            self.data.append(
                {
                    "x": (np.arange(12, dtype=np.float32).reshape(3, 4) + index) * mask,
                    "mask": mask,
                    "time": np.arange(1, 4, dtype=np.float32),
                    "lens": 3,
                    "labels": index % 2,
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


@pytest.fixture()
def toy_manifest(tmp_path):
    dataset = ToyDataset()
    manifest = create_manifest(
        tmp_path / "manifest.npz",
        dataset_name="toy",
        split_seed=42,
        test_dataset=dataset,
        replicates=3,
        manifest_seed=7,
    )
    return dataset, manifest


def test_manifest_is_deterministic_and_nested(tmp_path):
    dataset = ToyDataset()
    first = create_manifest(
        tmp_path / "first.npz", dataset_name="toy", split_seed=42,
        test_dataset=dataset, replicates=3, manifest_seed=7,
    )
    second = create_manifest(
        tmp_path / "second.npz", dataset_name="toy", split_seed=42,
        test_dataset=dataset, replicates=3, manifest_seed=7,
    )
    np.testing.assert_array_equal(first.permutations, second.permutations)
    for replicate in range(first.num_replicates):
        for row in range(len(dataset)):
            assert set(first.permutations[replicate, row, :2]).issubset(
                set(first.permutations[replicate, row, :3])
            )


def test_patient_permutation_is_stable_across_processes():
    code = (
        "from experiments.sensor_robustness import _patient_permutation; "
        "print(','.join(map(str,_patient_permutation(" 
        "'toy',42,'patient-x',2,7,17).tolist())))"
    )
    first = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    second = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert first == second
    assert sorted(map(int, first.split(","))) == list(range(17))


def test_manifest_rejects_feature_registry_mismatch(toy_manifest):
    dataset, manifest = toy_manifest
    dataset.feature_names = ("a", "b", "c", "changed")
    with pytest.raises(ValueError, match="Feature registry"):
        manifest.validate_dataset(dataset)


def test_clean_collator_preserves_original_schema_and_values(toy_manifest):
    dataset, manifest = toy_manifest
    original = collate_fn([dataset[0], dataset[1]])
    loader = make_sensor_loader(
        dataset,
        base_collate=collate_fn,
        manifest=manifest,
        replicate=0,
        k=0,
        batch_size=2,
        num_workers=0,
    )
    clean = next(iter(loader))
    assert clean.keys() == original.keys()
    for key in clean:
        torch.testing.assert_close(clean[key], original[key])


def test_corruption_zeros_x_and_mask_only(toy_manifest):
    dataset, manifest = toy_manifest
    loader = make_sensor_loader(
        dataset,
        base_collate=collate_fn,
        manifest=manifest,
        replicate=1,
        k=2,
        batch_size=4,
        num_workers=0,
    )
    batch = next(iter(loader))
    for row in range(4):
        columns = manifest.permutations[1, row, :2]
        assert torch.count_nonzero(batch["x"][row, :, columns]) == 0
        assert torch.count_nonzero(batch["mask"][row, :, columns]) == 0
    original = collate_fn([dataset[index] for index in range(4)])
    torch.testing.assert_close(batch["time"], original["time"])
    torch.testing.assert_close(batch["labels"], original["labels"])
    torch.testing.assert_close(batch["lens"], original["lens"])


def test_batch_order_does_not_change_patient_masks(toy_manifest):
    dataset, manifest = toy_manifest
    view = ManifestIndexedDataset(dataset, manifest)
    collator = SensorDropCollator(
        collate_fn, manifest.permutations, replicate=2, k=2
    )
    forward = collator([view[0], view[1]])
    reverse = collator([view[1], view[0]])
    torch.testing.assert_close(forward["mask"][0], reverse["mask"][1])
    torch.testing.assert_close(forward["mask"][1], reverse["mask"][0])


def test_condition_audit_reports_effective_observation_removal(toy_manifest):
    dataset, manifest = toy_manifest
    audit = condition_audit(dataset, manifest, replicate=0, k=2)
    assert audit["observed_entries_before"] > 0
    assert 0 <= audit["removed_observed_entries"] <= audit["observed_entries_before"]
    assert 0 <= audit["effective_observed_removal_fraction"] <= 1


def test_condition_grid_writes_predictions_and_resumes(toy_manifest, tmp_path):
    dataset, manifest = toy_manifest
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint")

    def evaluator(loader):
        labels, scores = [], []
        for batch in loader:
            labels.append(batch["labels"].numpy())
            scores.append(torch.sigmoid(batch["x"].sum(dim=(1, 2)) / 100).numpy())
        return np.concatenate(labels), np.concatenate(scores)

    output = tmp_path / "runs"
    runs = run_condition_grid(
        evaluator=evaluator,
        test_dataset=dataset,
        base_collate=collate_fn,
        manifest=manifest,
        ks=[0, 2],
        replicates=[0, 1],
        batch_size=2,
        num_workers=0,
        output_root=output,
        model="toy",
        dataset_name="toy",
        train_seed=42,
        split_seed=42,
        checkpoint_path=checkpoint,
    )
    assert len(runs) == 3
    assert sum(run["k"] == 0 for run in runs) == 1
    assert all(Path(run["predictions"]).exists() for run in runs)
    resumed = run_condition_grid(
        evaluator=lambda _: pytest.fail("resume should not evaluate"),
        test_dataset=dataset,
        base_collate=collate_fn,
        manifest=load_manifest(manifest.path),
        ks=[0, 2],
        replicates=[0, 1],
        batch_size=2,
        num_workers=0,
        output_root=output,
        model="toy",
        dataset_name="toy",
        train_seed=42,
        split_seed=42,
        checkpoint_path=checkpoint,
        resume=True,
    )
    assert len(resumed) == 3


def test_aggregate_uses_seed_means_then_sample_std():
    clean = {
        1: {"metrics": {"auprc": 0.6, "positive_rate": 0.1}},
        42: {"metrics": {"auprc": 0.8, "positive_rate": 0.1}},
    }
    runs = [
        {"train_seed": 1, "metrics": {"auprc": 0.4}},
        {"train_seed": 1, "metrics": {"auprc": 0.6}},
        {"train_seed": 42, "metrics": {"auprc": 0.7}},
        {"train_seed": 42, "metrics": {"auprc": 0.9}},
    ]
    block = aggregate_group(runs, clean)
    assert block["auprc_mean"] == pytest.approx(0.65)
    assert block["auprc_sample_std"] == pytest.approx(np.std([0.5, 0.8], ddof=1))


def test_clean_reconciliation_detects_mismatch(tmp_path):
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps({"source": "test", "toy": {"model": {
            "auprc_mean": 50.0, "auprc_sample_std": 1.0
        }}}), encoding="utf-8"
    )
    result = reconcile(
        {"toy": {"model": {"0": {
            "auprc_mean": 0.51, "auprc_sample_std": 0.01
        }}}}, reference,
    )
    assert not result["passed"]
    assert result["checks"][0]["status"] == "mismatch"


def test_preflight_rejects_corrupt_checkpoint(tmp_path):
    checkpoint = tmp_path / "broken.pt"
    checkpoint.write_bytes(b"not-a-zip")
    assert checkpoint_problem(checkpoint) == "not a readable PyTorch zip checkpoint"


def test_scheduler_resolves_explicit_unique_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4, 7")
    assert resolved_devices(SimpleNamespace(devices=None, gpu=None)) == ("4", "7")
    assert resolved_devices(SimpleNamespace(devices=["0", "1"], gpu=None)) == ("0", "1")
    with pytest.raises(ValueError, match="unique"):
        resolved_devices(SimpleNamespace(devices=["0", "0"], gpu=None))


def test_scheduler_rejects_unavailable_device(monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "0\n1\n")
    validate_local_devices(("0", "1"))
    with pytest.raises(ValueError, match="unavailable"):
        validate_local_devices(("0", "2"))


def test_plot_watermarks_draft_and_blocks_publication(tmp_path, monkeypatch):
    from experiments.plot_sensor_robustness import main as plot_main

    model_block = {
        "0": {"auprc_mean": 0.7, "retention_mean": 1.0},
        "8": {"auprc_mean": 0.5, "retention_mean": 0.7},
    }
    aggregate = tmp_path / "aggregate.json"
    aggregate.write_text(
        json.dumps({
            "complete": True,
            "publication_ready": False,
            "clean_reconciliation": {"passed": False},
            "aggregates": {
                "mimic_mortality": {"smart-smile-lean": model_block},
                "mimic_decompensation": {"smart-smile-lean": model_block},
            },
        }), encoding="utf-8",
    )
    output = tmp_path / "figure"
    monkeypatch.setattr(
        "sys.argv",
        ["plot_sensor_robustness.py", "--aggregate", str(aggregate),
         "--output-prefix", str(output)],
    )
    plot_main()
    assert output.with_suffix(".pdf").exists()
    assert output.with_suffix(".png").exists()

    monkeypatch.setattr(
        "sys.argv",
        ["plot_sensor_robustness.py", "--aggregate", str(aggregate),
         "--output-prefix", str(output), "--publication"],
    )
    with pytest.raises(SystemExit, match="Publication plot blocked"):
        plot_main()
