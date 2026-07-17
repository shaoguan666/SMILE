"""Leak-free ODP forecasting metrics and DDP aggregation."""

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import average_precision_score


def gather_forecast_records(records):
    """Gather records across ranks and remove sampler-padding duplicates by id."""
    gathered = [records]
    if dist.is_available() and dist.is_initialized():
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, records)
    unique = {}
    for shard in gathered:
        for record in shard:
            unique.setdefault(int(record["sample_id"]), record)
    return [unique[key] for key in sorted(unique)]


def make_forecast_records(prob, target, lens, sample_ids):
    records = []
    for index in range(prob.shape[0]):
        length = int(lens[index])
        records.append({
            "sample_id": int(sample_ids[index]),
            "prob": prob[index, :length].detach().float().cpu().numpy(),
            "target": target[index, :length].detach().float().cpu().numpy(),
        })
    return records


def _average_precision(y_true, y_prob):
    if np.unique(y_true).size < 2:
        return None
    return float(average_precision_score(y_true, y_prob))


def forecasting_metrics(records, include_t0=False, epsilon=1e-7):
    if not records:
        raise ValueError("forecasting metrics require at least one sample")
    num_vars = records[0]["target"].shape[-1]
    per_variable = []
    eligible = []
    start = 0 if include_t0 else 1
    for variable_id in range(num_vars):
        targets = np.concatenate([r["target"][start:, variable_id] for r in records])
        probs = np.concatenate([r["prob"][start:, variable_id] for r in records])
        probs = np.clip(probs.astype(np.float64), epsilon, 1.0 - epsilon)
        targets = targets.astype(np.float64)
        ap = _average_precision(targets, probs)
        brier = float(np.mean((probs - targets) ** 2)) if targets.size else float("nan")
        bce = float(np.mean(-(targets * np.log(probs) + (1 - targets) * np.log(1 - probs)))) if targets.size else float("nan")
        item = {"variable": variable_id, "auprc": ap, "brier": brier, "bce": bce}
        per_variable.append(item)
        if ap is not None:
            eligible.append(item)
    macro = {
        "auprc": float(np.mean([x["auprc"] for x in eligible])) if eligible else None,
        "brier": float(np.mean([x["brier"] for x in eligible])) if eligible else None,
        "bce": float(np.mean([x["bce"] for x in eligible])) if eligible else None,
    }
    return {
        "scope": "t>=0_secondary" if include_t0 else "t>=1_primary",
        "macro": macro,
        "eligible_variables": len(eligible),
        "total_variables": num_vars,
        "per_variable": per_variable,
        "epsilon": epsilon,
    }


def training_variable_prior(dataset, epsilon=1e-7):
    observed = None
    total = None
    for idx in range(len(dataset)):
        mask = np.asarray(dataset[idx]["mask"], dtype=np.float64)[1:]
        current = mask.sum(axis=0)
        count = np.full(mask.shape[1], mask.shape[0], dtype=np.float64)
        observed = current if observed is None else observed + current
        total = count if total is None else total + count
    return np.clip(observed / np.maximum(total, 1), epsilon, 1.0 - epsilon)


def prior_records(dataset, prior):
    records = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        target = np.asarray(sample["mask"], dtype=np.float32)
        records.append({
            "sample_id": int(sample.get("sample_id", idx)),
            "target": target,
            "prob": np.broadcast_to(prior, target.shape).copy(),
        })
    return records


def paired_bootstrap_auprc(records_a, records_b, seed=20260717, repetitions=1000):
    """Patient-level paired bootstrap delta for macro AUPRC."""
    by_a = {int(r["sample_id"]): r for r in records_a}
    by_b = {int(r["sample_id"]): r for r in records_b}
    ids = sorted(set(by_a) & set(by_b))
    if not ids:
        raise ValueError("paired bootstrap requires shared sample ids")
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repetitions):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        a = [dict(by_a[int(i)], sample_id=j) for j, i in enumerate(sampled)]
        b = [dict(by_b[int(i)], sample_id=j) for j, i in enumerate(sampled)]
        ma = forecasting_metrics(a)["macro"]["auprc"]
        mb = forecasting_metrics(b)["macro"]["auprc"]
        if ma is not None and mb is not None:
            deltas.append(ma - mb)
    if not deltas:
        return {"seed": seed, "repetitions": repetitions, "low": None, "high": None}
    low, high = np.percentile(deltas, [2.5, 97.5])
    return {"seed": seed, "repetitions": repetitions, "low": float(low), "high": float(high)}
