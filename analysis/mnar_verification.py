#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Structured-missingness audit for SMART clinical time-series tasks.

The audit is intentionally descriptive:

* T1 is a Little MCAR compatibility check on per-record aggregated values.
* T2 compares observation fractions by outcome for binary tasks only.
* T3 compares early, middle, and late observation frequencies.
* T4 measures candidate-system co-observation structure with Spearman
  correlations and selects groups only by the declared delta threshold.

Run from the ``SMART/`` directory:

    python analysis/mnar_verification.py [--split train|all] [--no-plot]

Outputs are written under ``analysis/results`` by default, without writing to
paper directories or experiment-export directories. When plotting is enabled,
``figure2_mnar_audit_summary.pdf`` and ``.png`` provide the compact main-paper
T2--T4 summary figure.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy import stats
from scipy.stats import chi2 as chi2_dist


TASKS = {
    "c12": {
        "path": "data/Challenge2012/data_normalized.pkl",
        "mask_index": 3,
        "binary": True,
    },
    "c19": {
        "path": "data/Challenge2019/data_normalized.pkl",
        "mask_index": 3,
        "binary": True,
    },
    "mimic_mortality": {
        "path": "data/MIMIC-III/mortality_normalized.pkl",
        "mask_index": 2,
        "binary": True,
    },
    "mimic_decompensation": {
        "path": "data/MIMIC-III/decompensation_normalized.pkl",
        "mask_index": 2,
        "binary": True,
    },
    "mimic_phenotyping": {
        "path": "data/MIMIC-III/phenotyping_normalized.pkl",
        "mask_index": 2,
        "binary": False,
    },
    "mimic_lengthofstay": {
        "path": "data/MIMIC-III/lengthofstay_normalized.pkl",
        "mask_index": 2,
        "binary": False,
    },
}

SELECTION_RULE = "delta > 0.5"

FIGURE_DATASETS = (
    ("c12", "C12"),
    ("c19", "C19"),
    ("mimic_mortality", "MIMIC-Mort."),
    ("mimic_decompensation", "MIMIC-Decomp."),
    ("mimic_phenotyping", "MIMIC-Pheno."),
    ("mimic_lengthofstay", "MIMIC-LOS"),
)
FIGURE2_HEATMAP_TASKS = (
    ("c12", "C12"),
    ("c19", "C19"),
    ("mimic_mortality", "MIMIC-Mort."),
)
FIGURE2_GROUP_LABELS = {
    "Vital_NonInvasive": "NI-BP",
    "Vital_Invasive": "I-BP",
    "Vital_BP": "BP",
    "Vital_Basic": "Vitals",
    "Blood_Gas": "Blood gas",
    "Resp_Gas": "Resp.",
    "Electrolyte": "Electr.",
    "Liver_Enzyme": "Liver",
    "Bilirubin": "Bili.",
    "Renal": "Renal",
    "CBC": "CBC",
    "Coagulation": "Coag.",
    "Cardiac_Biomarker": "Cardiac",
    "GCS": "GCS",
    "Anthropometric": "Anthro.",
    "Other": "Other",
}
FIGURE2_FEATURE_LABELS = {
    "Fraction inspired oxygen": "FiO2",
    "Oxygen saturation": "O2Sat",
    "Glascow coma scale eye opening": "GCS-Eye",
}
FIGURE2_BLUE = "#527FAF"
FIGURE2_LIGHT_BLUE = "#B8D2E7"
FIGURE2_PEACH = "#D98468"
FIGURE2_DARK = "#27364A"
FIGURE2_GRID = "#DCE3EA"


def load_registry_api():
    """Import the shared feature/group registry once it is available."""
    smart_root = str(Path(__file__).resolve().parents[1])
    if smart_root not in sys.path:
        sys.path.insert(0, smart_root)
    try:
        from data.feature_registry import (  # pylint: disable=import-outside-toplevel
            REGISTRY_VERSION,
            get_candidate_systems,
            get_feature_names,
            registry_fingerprint,
        )
    except ImportError as exc:
        raise RuntimeError(
            "data.feature_registry is required for the structured-missingness "
            "audit; integrate the shared registry before running it."
        ) from exc
    return (
        REGISTRY_VERSION,
        get_feature_names,
        get_candidate_systems,
        registry_fingerprint,
    )


def _has_stored_split_sizes(payload: tuple[Any, ...] | list[Any]) -> bool:
    """Return whether the payload ends in explicit train/validation/test sizes."""
    if len(payload) != 5:
        return False
    candidate = payload[4]
    if not isinstance(candidate, (list, tuple, np.ndarray)) or len(candidate) != 3:
        return False
    return all(isinstance(value, (int, np.integer)) for value in candidate)


def split_indices(
    task: str,
    payload: tuple[Any, ...] | list[Any],
    n_records: int,
    split: str,
    split_seed: int,
) -> tuple[list[int], str]:
    """Match the training split convention used by SMART data loaders."""
    if split == "all":
        return list(range(n_records)), "all_records"

    if task.startswith("mimic_") and _has_stored_split_sizes(payload):
        n_train = int(payload[4][0])
        return list(range(n_train)), "stored_split_sizes"

    indices = list(range(n_records))
    random.Random(split_seed).shuffle(indices)
    n_train = int(n_records * 0.8)
    return indices[:n_train], "seeded_80_percent"


def load_task_records(
    task: str, config: dict[str, Any], split: str, split_seed: int
) -> tuple[list[Any], list[Any], list[Any], str]:
    """Load one task and select the requested record subset."""
    with open(config["path"], "rb") as handle:
        payload = pickle.load(handle)

    x_all = payload[0]
    y_all = payload[1]
    masks_all = payload[config["mask_index"]]
    if not (len(x_all) == len(y_all) == len(masks_all)):
        raise ValueError("%s pickle has inconsistent record counts" % task)

    indices, split_method = split_indices(
        task, payload, len(x_all), split=split, split_seed=split_seed
    )
    return (
        [x_all[index] for index in indices],
        [y_all[index] for index in indices],
        [masks_all[index] for index in indices],
        split_method,
    )


def build_record_matrices(
    x_list: list[Any], mask_list: list[Any], n_features: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create per-record observed means and per-record observation fractions."""
    n_records = len(mask_list)
    observed_means = np.full((n_records, n_features), np.nan, dtype=np.float64)
    observation_fraction = np.full((n_records, n_features), np.nan, dtype=np.float64)

    for row, (values, mask) in enumerate(zip(x_list, mask_list)):
        x_arr = np.asarray(values, dtype=np.float64)
        mask_arr = np.asarray(mask, dtype=np.float64)
        if mask_arr.ndim != 2 or mask_arr.shape[1] != n_features:
            raise ValueError(
                "Record %d mask has shape %s, expected (*, %d)"
                % (row, mask_arr.shape, n_features)
            )
        if x_arr.shape != mask_arr.shape:
            raise ValueError(
                "Record %d value/mask shapes differ: %s vs %s"
                % (row, x_arr.shape, mask_arr.shape)
            )
        if mask_arr.shape[0] == 0:
            continue
        observation_fraction[row] = mask_arr.mean(axis=0)
        for feature in range(n_features):
            is_observed = mask_arr[:, feature] == 1
            if np.any(is_observed):
                observed_means[row, feature] = np.mean(x_arr[is_observed, feature])

    return observed_means, observation_fraction


def benjamini_hochberg(p_values: np.ndarray | list[float]) -> np.ndarray:
    """Apply Benjamini-Hochberg correction while retaining missing values."""
    p_array = np.asarray(p_values, dtype=np.float64)
    q_values = np.full(p_array.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(p_array)
    if not np.any(valid):
        return q_values

    valid_positions = np.flatnonzero(valid)
    valid_p = p_array[valid]
    order = np.argsort(valid_p)
    ranked = valid_p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    q_values[valid_positions[order]] = adjusted
    return q_values


def littles_mcar_compatibility_check(observed_means: np.ndarray) -> dict[str, Any]:
    """Run the legacy Little-style MCAR compatibility calculation."""
    n_records, n_features = observed_means.shape
    if n_records == 0:
        return {
            "chi2_stat": np.nan,
            "p_value": np.nan,
            "df": np.nan,
            "n_patterns": 0,
            "interpretation": "not_computable",
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        grand_mean = np.nanmean(observed_means, axis=0)

    covariance = np.zeros((n_features, n_features), dtype=np.float64)
    for left in range(n_features):
        for right in range(left, n_features):
            valid = np.isfinite(observed_means[:, left]) & np.isfinite(
                observed_means[:, right]
            )
            if np.count_nonzero(valid) > 1:
                if left == right:
                    value = float(np.var(observed_means[valid, left], ddof=1))
                else:
                    value = float(
                        np.cov(
                            observed_means[valid, left],
                            observed_means[valid, right],
                        )[0, 1]
                    )
                covariance[left, right] = covariance[right, left] = value
    covariance += np.eye(n_features) * 1e-6

    patterns: dict[tuple[bool, ...], list[int]] = {}
    for index, pattern in enumerate(np.isnan(observed_means)):
        patterns.setdefault(tuple(pattern.tolist()), []).append(index)

    chi2_sum = 0.0
    df_sum = 0
    for pattern, indices in patterns.items():
        retained = [feature for feature, missing in enumerate(pattern) if not missing]
        if not retained:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pattern_mean = np.nanmean(observed_means[np.ix_(indices, retained)], axis=0)
        difference = pattern_mean - grand_mean[retained]
        if not np.all(np.isfinite(difference)):
            continue
        sub_covariance = covariance[np.ix_(retained, retained)]
        distance = len(indices) * float(
            difference @ np.linalg.pinv(sub_covariance) @ difference
        )
        if np.isfinite(distance) and distance >= 0:
            chi2_sum += distance
            df_sum += len(retained)

    degrees_freedom = max(df_sum - n_features, 1)
    p_value = float(chi2_dist.sf(chi2_sum, degrees_freedom))
    return {
        "chi2_stat": chi2_sum,
        "p_value": p_value,
        "df": degrees_freedom,
        "n_patterns": len(patterns),
        "interpretation": (
            "reject_mcar_compatibility"
            if p_value < 0.05
            else "mcar_compatibility_not_rejected"
        ),
    }


def _binary_labels(labels: list[Any]) -> np.ndarray:
    """Convert scalar binary labels without transforming other label protocols."""
    flattened = []
    for label in labels:
        array = np.asarray(label)
        if array.ndim != 0:
            raise ValueError("Binary task contains non-scalar labels")
        flattened.append(float(array))
    result = np.asarray(flattened, dtype=np.float64)
    values = set(np.unique(result).tolist())
    if not values.issubset({0.0, 1.0}) or not values:
        raise ValueError("Binary task labels are not encoded as 0/1: %s" % sorted(values))
    return result


def outcome_observation_test(
    observation_fraction: np.ndarray,
    labels: list[Any],
    feature_names: list[str],
    is_binary: bool,
) -> pd.DataFrame:
    """T2: compare observation fraction between positive and negative records."""
    columns = [
        "feature",
        "status",
        "n_positive",
        "n_negative",
        "mean_positive",
        "mean_negative",
        "effect_size",
        "test",
        "p_value",
        "q_value",
    ]
    if not is_binary:
        return pd.DataFrame(
            [
                {
                    "feature": "N/A",
                    "status": "not_applicable_non_binary_task",
                    "n_positive": np.nan,
                    "n_negative": np.nan,
                    "mean_positive": np.nan,
                    "mean_negative": np.nan,
                    "effect_size": np.nan,
                    "test": "N/A",
                    "p_value": np.nan,
                    "q_value": np.nan,
                }
            ],
            columns=columns,
        )

    y_values = _binary_labels(labels)
    positive = y_values == 1
    negative = y_values == 0
    rows = []
    for index, feature in enumerate(feature_names):
        pos_values = observation_fraction[positive, index]
        neg_values = observation_fraction[negative, index]
        pos_values = pos_values[np.isfinite(pos_values)]
        neg_values = neg_values[np.isfinite(neg_values)]
        p_value = np.nan
        if len(pos_values) and len(neg_values):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, p_value = stats.mannwhitneyu(
                    pos_values, neg_values, alternative="two-sided"
                )
        mean_positive = float(np.mean(pos_values)) if len(pos_values) else np.nan
        mean_negative = float(np.mean(neg_values)) if len(neg_values) else np.nan
        rows.append(
            {
                "feature": feature,
                "status": "computed",
                "n_positive": len(pos_values),
                "n_negative": len(neg_values),
                "mean_positive": mean_positive,
                "mean_negative": mean_negative,
                "effect_size": mean_positive - mean_negative,
                "test": "Mann-Whitney U",
                "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
            }
        )
    frame = pd.DataFrame(rows)
    frame["q_value"] = benjamini_hochberg(frame["p_value"].to_numpy())
    return frame[columns].sort_values("q_value", na_position="last")


def temporal_observation_fractions(
    mask_list: list[Any], n_features: int
) -> np.ndarray:
    """Compute each record's early/middle/late observation fractions."""
    phases = np.full((len(mask_list), 3, n_features), np.nan, dtype=np.float64)
    for row, mask in enumerate(mask_list):
        mask_arr = np.asarray(mask, dtype=np.float64)
        if mask_arr.ndim != 2 or mask_arr.shape[1] != n_features:
            raise ValueError("Invalid mask shape in temporal calculation: %s" % (mask_arr.shape,))
        for phase, indices in enumerate(np.array_split(np.arange(mask_arr.shape[0]), 3)):
            if len(indices):
                phases[row, phase, :] = np.mean(mask_arr[indices, :], axis=0)
    return phases


def temporal_observation_test(
    temporal_fraction: np.ndarray, feature_names: list[str]
) -> pd.DataFrame:
    """T3: Kruskal-Wallis comparison of early/middle/late observation frequency."""
    rows = []
    for index, feature in enumerate(feature_names):
        groups = [
            temporal_fraction[:, phase, index][
                np.isfinite(temporal_fraction[:, phase, index])
            ]
            for phase in range(3)
        ]
        means = [float(np.mean(group)) if len(group) else np.nan for group in groups]
        p_value = np.nan
        statistic = np.nan
        if all(len(group) for group in groups):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    statistic, p_value = stats.kruskal(*groups)
                except ValueError:
                    statistic, p_value = np.nan, np.nan
        temporal_range = (
            float(np.nanmax(means) - np.nanmin(means))
            if np.any(np.isfinite(means))
            else np.nan
        )
        rows.append(
            {
                "feature": feature,
                "mean_early": means[0],
                "mean_middle": means[1],
                "mean_late": means[2],
                "temporal_range": temporal_range,
                "effect_size": temporal_range,
                "test": "Kruskal-Wallis",
                "kw_stat": float(statistic) if np.isfinite(statistic) else np.nan,
                "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
            }
        )
    frame = pd.DataFrame(rows)
    frame["q_value"] = benjamini_hochberg(frame["p_value"].to_numpy())
    return frame.sort_values("q_value", na_position="last")


def spearman_matrix(observation_fraction: np.ndarray) -> np.ndarray:
    """Compute pairwise Spearman correlations for record-level fractions."""
    n_features = observation_fraction.shape[1]
    correlations = np.full((n_features, n_features), np.nan, dtype=np.float64)
    np.fill_diagonal(correlations, 1.0)
    for left in range(n_features):
        for right in range(left + 1, n_features):
            valid = np.isfinite(observation_fraction[:, left]) & np.isfinite(
                observation_fraction[:, right]
            )
            if np.count_nonzero(valid) < 2:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho, _ = stats.spearmanr(
                    observation_fraction[valid, left],
                    observation_fraction[valid, right],
                )
            if np.isfinite(rho):
                correlations[left, right] = correlations[right, left] = float(rho)
    return correlations


def block_coobservation_test(
    observation_fraction: np.ndarray,
    candidate_systems: dict[str, list[int]],
    n_features: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """T4: summarize candidate-system within versus between correlations."""
    correlations = spearman_matrix(observation_fraction)
    rows = []
    for group, raw_indices in candidate_systems.items():
        indices = [int(index) for index in raw_indices]
        if any(index < 0 or index >= n_features for index in indices):
            raise ValueError("Candidate group %s has an out-of-range feature index" % group)
        outside = [index for index in range(n_features) if index not in set(indices)]
        within = [
            correlations[left, right]
            for position, left in enumerate(indices)
            for right in indices[position + 1 :]
            if np.isfinite(correlations[left, right])
        ]
        between = [
            correlations[left, right]
            for left in indices
            for right in outside
            if np.isfinite(correlations[left, right])
        ]
        mean_within = float(np.mean(within)) if within else np.nan
        mean_between = float(np.mean(between)) if between else np.nan
        delta = (
            mean_within - mean_between
            if np.isfinite(mean_within) and np.isfinite(mean_between)
            else np.nan
        )
        rows.append(
            {
                "group": group,
                "indices": json.dumps(indices),
                "n_variables": len(indices),
                "n_within_pairs": len(within),
                "n_between_pairs": len(between),
                "mean_within": mean_within,
                "mean_between": mean_between,
                "delta": delta,
                "selection_rule": SELECTION_RULE,
                "selected": bool(np.isfinite(delta) and delta > 0.5),
            }
        )
    return pd.DataFrame(rows).sort_values("delta", ascending=False, na_position="last"), correlations


def plot_temporal_observation(
    frame: pd.DataFrame, task: str, split: str, output_path: Path
) -> None:
    """Render a phase-by-feature observation-frequency heatmap."""
    values = frame.set_index("feature")[["mean_early", "mean_middle", "mean_late"]]
    figure_height = max(4.0, 0.22 * len(values) + 1.6)
    fig, axis = plt.subplots(figsize=(5.4, figure_height))
    image = axis.imshow(values.to_numpy(), cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_xticks(range(3), ["Early", "Middle", "Late"])
    axis.set_yticks(range(len(values)), values.index.tolist(), fontsize=8)
    axis.set_title("%s (%s split): observation frequency by phase" % (task, split))
    colorbar = fig.colorbar(image, ax=axis, shrink=0.75)
    colorbar.set_label("Observation fraction")
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_coobservation(
    correlations: np.ndarray,
    feature_names: list[str],
    task: str,
    split: str,
    output_path: Path,
) -> None:
    """Render the T4 co-observation Spearman correlation matrix."""
    figure_size = max(5.5, 0.22 * len(feature_names) + 2.5)
    fig, axis = plt.subplots(figsize=(figure_size, figure_size))
    image = axis.imshow(correlations, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ticks = range(len(feature_names))
    axis.set_xticks(ticks, feature_names, rotation=90, fontsize=7)
    axis.set_yticks(ticks, feature_names, fontsize=7)
    axis.set_title("%s (%s split): co-observation Spearman correlation" % (task, split))
    colorbar = fig.colorbar(image, ax=axis, shrink=0.75)
    colorbar.set_label("Spearman rho")
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def _style_figure2_axis(axis: plt.Axes, grid_axis: str = "x") -> None:
    """Apply consistent journal-style formatting to the top summary panels."""
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#6B7280")
    axis.spines["bottom"].set_color("#6B7280")
    axis.tick_params(colors=FIGURE2_DARK, labelsize=7.0, length=3)
    axis.grid(axis=grid_axis, color=FIGURE2_GRID, linewidth=0.65, zorder=0)


def _short_feature_name(feature: str) -> str:
    """Shorten long MIMIC labels used as sparse annotations in Figure 2."""
    return FIGURE2_FEATURE_LABELS.get(feature, feature)


def _plot_figure2_endpoint_panel(
    axis: plt.Axes, payloads: dict[str, dict[str, Any]]
) -> None:
    """Plot full T2 distributions for binary task cohorts."""
    binary_tasks = [(task, label) for task, label in FIGURE_DATASETS if TASKS[task]["binary"]]
    frames = []
    for task, label in binary_tasks:
        frame = payloads[task]["t2"].loc[
            payloads[task]["t2"]["status"].eq("computed")
        ].copy()
        frame["task"] = task
        frame["label"] = label
        frame["missing_gap_pp"] = -100.0 * frame["effect_size"]
        frame["significant"] = frame["q_value"] < 0.05
        frames.append(frame)
    points = pd.concat(frames, ignore_index=True)
    limit = max(16.0, np.ceil(float(points["missing_gap_pp"].abs().max()) * 1.55 / 2.0) * 2.0)

    for row, (task, _label) in enumerate(binary_tasks):
        task_points = points.loc[points["task"].eq(task)].sort_values("missing_gap_pp")
        jitter = np.linspace(-0.15, 0.15, len(task_points))
        axis.scatter(
            task_points["missing_gap_pp"],
            row + jitter,
            s=7,
            color="#CFD6DD",
            alpha=0.8,
            linewidths=0,
            zorder=2,
        )
        significant = task_points.loc[task_points["significant"]]
        sig_jitter = jitter[task_points["significant"].to_numpy()]
        axis.scatter(
            significant["missing_gap_pp"],
            row + sig_jitter,
            s=10,
            color=FIGURE2_BLUE,
            linewidths=0,
            zorder=3,
        )
        strongest = task_points.loc[task_points["missing_gap_pp"].abs().idxmax()]
        strongest_y = row + jitter[task_points.index.get_loc(strongest.name)]
        value = float(strongest["missing_gap_pp"])
        axis.scatter(value, strongest_y, s=29, marker="D", color=FIGURE2_PEACH, zorder=4)
        if value > limit * 0.45:
            align, offset = "right", -0.8
        else:
            align = "left" if value >= 0 else "right"
            offset = 0.7 if value >= 0 else -0.7
        axis.text(
            value + offset,
            strongest_y - 0.17,
            _short_feature_name(str(strongest["feature"])),
            ha=align,
            va="center",
            fontsize=6.0,
            color=FIGURE2_DARK,
        )
        axis.text(
            0.985,
            row,
            "%d/%d sig." % (len(significant), len(task_points)),
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=6.3,
            color=FIGURE2_DARK,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.92},
        )

    axis.axvline(0, color="#5E6B78", linewidth=0.8, zorder=1)
    axis.set_yticks(range(len(binary_tasks)), [label for _task, label in binary_tasks])
    axis.invert_yaxis()
    axis.set_xlim(-limit, limit)
    axis.set_xlabel("Missing-rate difference by outcome (pp)", labelpad=3)
    axis.set_title("A  Outcome-associated missingness", loc="left", fontweight="bold", pad=13)
    axis.text(
        0,
        1.01,
        r"Blue = $q < 0.05$; diamond = largest absolute difference",
        transform=axis.transAxes,
        fontsize=6.2,
        color="#58677A",
        ha="left",
        va="bottom",
    )
    _style_figure2_axis(axis)


def _plot_figure2_temporal_panel(
    axis: plt.Axes, payloads: dict[str, dict[str, Any]]
) -> None:
    """Plot full T3 temporal-range distributions for all task cohorts."""
    maximum = max(
        float((100.0 * payloads[task]["t3"]["temporal_range"]).max())
        for task, _label in FIGURE_DATASETS
    )
    limit = max(10.0, np.ceil(maximum * 1.47 / 2.0) * 2.0)
    for row, (task, _label) in enumerate(FIGURE_DATASETS):
        task_points = payloads[task]["t3"].copy()
        task_points["temporal_range_pp"] = 100.0 * task_points["temporal_range"]
        task_points["significant"] = task_points["q_value"] < 0.05
        task_points = task_points.sort_values("temporal_range_pp")
        jitter = np.linspace(-0.15, 0.15, len(task_points))
        significant = task_points.loc[task_points["significant"]]
        axis.plot(
            [0, float(task_points["temporal_range_pp"].max())],
            [row, row],
            color=FIGURE2_LIGHT_BLUE,
            linewidth=1.1,
            zorder=1,
        )
        axis.scatter(
            task_points["temporal_range_pp"],
            row + jitter,
            s=7,
            color="#CFD6DD",
            alpha=0.8,
            linewidths=0,
            zorder=2,
        )
        axis.scatter(
            significant["temporal_range_pp"],
            row + jitter[task_points["significant"].to_numpy()],
            s=10,
            color=FIGURE2_BLUE,
            linewidths=0,
            zorder=3,
        )
        strongest = task_points.loc[task_points["temporal_range_pp"].idxmax()]
        strongest_y = row + jitter[task_points.index.get_loc(strongest.name)]
        max_value = float(strongest["temporal_range_pp"])
        axis.scatter(max_value, strongest_y, s=29, marker="D", color=FIGURE2_PEACH, zorder=4)
        axis.text(
            min(max_value + 0.45, limit - 0.25),
            row,
            "%d/%d" % (len(significant), len(task_points)),
            va="center",
            ha="left",
            fontsize=6.3,
            color=FIGURE2_DARK,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.92},
        )

    axis.set_yticks(range(len(FIGURE_DATASETS)), [label for _task, label in FIGURE_DATASETS])
    axis.invert_yaxis()
    axis.set_xlim(0, limit)
    axis.set_xlabel("Temporal observation-rate range (pp)", labelpad=3)
    axis.set_title("B  Temporal structure is widespread", loc="left", fontweight="bold", pad=13)
    axis.text(
        0,
        1.01,
        "Dots = variables; label = significant / tested",
        transform=axis.transAxes,
        fontsize=6.2,
        color="#58677A",
        ha="left",
        va="bottom",
    )
    _style_figure2_axis(axis)


def _figure2_ordered_matrix(
    payload: dict[str, Any],
) -> tuple[np.ndarray, list[tuple[str, int, int, bool]]]:
    """Order variables by physiological group so co-observation blocks are visible."""
    t4 = payload["t4"].sort_values(["selected", "delta"], ascending=[False, False])
    grouped_indices: list[int] = []
    spans: list[tuple[str, int, int, bool]] = []
    for _, result in t4.iterrows():
        group = str(result["group"])
        indices = [int(index) for index in payload["candidate_systems"][group]]
        start = len(grouped_indices)
        grouped_indices.extend(indices)
        spans.append((group, start, len(grouped_indices), bool(result["selected"])))
    remainder = [
        index
        for index in range(len(payload["feature_names"]))
        if index not in set(grouped_indices)
    ]
    if remainder:
        start = len(grouped_indices)
        grouped_indices.extend(remainder)
        spans.append(("Other", start, len(grouped_indices), False))
    matrix = payload["correlations"][np.ix_(grouped_indices, grouped_indices)]
    return matrix, spans


def _plot_figure2_heatmap(
    axis: plt.Axes,
    payload: dict[str, Any],
    selection: dict[str, Any],
    title: str,
) -> matplotlib.image.AxesImage:
    """Render one compact, system-ordered T4 correlation matrix."""
    matrix, spans = _figure2_ordered_matrix(payload)
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0, interpolation="nearest")
    centers = []
    labels = []
    for group, start, stop, selected in spans:
        centers.append((start + stop - 1) / 2.0)
        labels.append(FIGURE2_GROUP_LABELS.get(group, group))
        if start:
            axis.axvline(start - 0.5, color="white", linewidth=0.55)
            axis.axhline(start - 0.5, color="white", linewidth=0.55)
        if selected:
            axis.add_patch(
                Rectangle(
                    (start - 0.48, start - 0.48),
                    stop - start - 0.04,
                    stop - start - 0.04,
                    fill=False,
                    edgecolor=FIGURE2_PEACH,
                    linewidth=1.15,
                )
            )
    delta_values = [
        value for value in selection["delta_by_group"].values() if value is not None
    ]
    axis.set_title(
        "%s\n%d retained | max $\\delta$ = %.2f"
        % (title, len(selection["selected_groups"]), max(delta_values)),
        fontsize=7.0,
        fontweight="bold",
        pad=4,
    )
    axis.set_xticks(centers, labels, rotation=90, fontsize=5.0)
    axis.set_yticks([])
    axis.tick_params(axis="x", length=0, pad=2, colors=FIGURE2_DARK)
    for spine in axis.spines.values():
        spine.set_visible(False)
    return image


def plot_main_paper_figure2(
    payloads: dict[str, dict[str, Any]],
    selections: dict[str, Any],
    output_root: Path,
    split: str,
    split_seed: int,
) -> tuple[Path, Path]:
    """Render the compact T2--T4 main-paper Figure 2."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.1,
            "axes.titlesize": 9.0,
            "axes.labelsize": 7.1,
            "savefig.dpi": 360,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(7.25, 5.35), facecolor="white")
    grid = figure.add_gridspec(
        3,
        2,
        height_ratios=[1.0, 0.16, 1.02],
        left=0.102,
        right=0.974,
        bottom=0.115,
        top=0.925,
        hspace=0.40,
        wspace=0.43,
    )
    _plot_figure2_endpoint_panel(figure.add_subplot(grid[0, 0]), payloads)
    _plot_figure2_temporal_panel(figure.add_subplot(grid[0, 1]), payloads)
    title_axis = figure.add_subplot(grid[1, :])
    title_axis.axis("off")
    title_axis.text(
        0,
        0.68,
        "C  Representative co-observation structure (Spearman correlation)",
        fontsize=9.0,
        fontweight="bold",
        color="#111827",
    )
    title_axis.text(
        0,
        0.08,
        "Variables reordered by physiological system; peach outline = retained block",
        fontsize=6.2,
        color="#58677A",
    )
    lower = grid[2, :].subgridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.045], wspace=0.27)
    image = None
    for column, (task, label) in enumerate(FIGURE2_HEATMAP_TASKS):
        image = _plot_figure2_heatmap(
            figure.add_subplot(lower[0, column]),
            payloads[task],
            selections[task],
            label,
        )
    colorbar_axis = figure.add_subplot(lower[0, 3])
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Spearman correlation", fontsize=6.5, color=FIGURE2_DARK)
    colorbar.ax.tick_params(labelsize=6.2, colors=FIGURE2_DARK, length=2)
    figure.text(
        0.102,
        0.035,
        r"%s split only (seed %d). T2/T3: Benjamini-Hochberg $q < 0.05$. T4: retained when $\delta > 0.50$."
        % (split.capitalize(), split_seed),
        fontsize=6.2,
        color="#536274",
    )
    png_path = output_root / "figure2_mnar_audit_summary.png"
    pdf_path = output_root / "figure2_mnar_audit_summary.pdf"
    figure.savefig(png_path, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return pdf_path, png_path


def _finite_float(value: Any) -> float | None:
    """Convert finite numeric results to JSON-safe floats."""
    return float(value) if value is not None and np.isfinite(value) else None


def run_task(
    task: str,
    config: dict[str, Any],
    output_root: Path,
    split: str,
    split_seed: int,
    make_plots: bool,
    get_feature_names,
    get_candidate_systems,
    registry_fingerprint,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run T1-T4 and return summary, selection metadata, and figure data."""
    print("\n[%s] loading %s split" % (task, split))
    x_list, labels, masks, split_method = load_task_records(
        task, config, split=split, split_seed=split_seed
    )
    feature_names = list(get_feature_names(task))
    candidate_systems = dict(get_candidate_systems(task))
    n_features = len(feature_names)
    output_dir = output_root / task
    output_dir.mkdir(parents=True, exist_ok=True)

    observed_means, observation_fraction = build_record_matrices(
        x_list, masks, n_features=n_features
    )
    temporal_fraction = temporal_observation_fractions(masks, n_features=n_features)

    t1_result = littles_mcar_compatibility_check(observed_means)
    t1_frame = pd.DataFrame(
        [
            {
                "test": "Little MCAR compatibility check",
                **t1_result,
            }
        ]
    )
    t1_frame.to_csv(output_dir / "t1_little.csv", index=False)

    t2_frame = outcome_observation_test(
        observation_fraction,
        labels,
        feature_names,
        is_binary=bool(config["binary"]),
    )
    t2_frame.to_csv(output_dir / "t2_outcome.csv", index=False)

    t3_frame = temporal_observation_test(temporal_fraction, feature_names)
    t3_frame.to_csv(output_dir / "t3_temporal.csv", index=False)

    t4_frame, correlations = block_coobservation_test(
        observation_fraction, candidate_systems, n_features=n_features
    )
    t4_frame.to_csv(output_dir / "t4_block.csv", index=False)

    if make_plots:
        plot_temporal_observation(
            t3_frame,
            task,
            split,
            output_dir / "t3_temporal_observation_heatmap.png",
        )
        plot_coobservation(
            correlations,
            feature_names,
            task,
            split,
            output_dir / "t4_coobservation_spearman_heatmap.png",
        )

    computed_t2 = t2_frame[t2_frame["status"] == "computed"]
    t2_q_count = int((computed_t2["q_value"] < 0.05).sum()) if len(computed_t2) else 0
    t3_q_count = int((t3_frame["q_value"] < 0.05).sum())
    selected_rows = t4_frame[t4_frame["selected"]]
    selected_groups = {
        group: [int(index) for index in candidate_systems[group]]
        for group in selected_rows["group"].tolist()
    }
    delta_by_group = {
        row["group"]: _finite_float(row["delta"])
        for _, row in t4_frame.iterrows()
    }
    summary = {
        "task": task,
        "split": split,
        "split_seed": split_seed,
        "split_method": split_method,
        "n_records": len(masks),
        "n_features": n_features,
        "mean_observation_fraction": float(np.nanmean(observation_fraction)),
        "t1_p_value": t1_result["p_value"],
        "t1_interpretation": t1_result["interpretation"],
        "t2_status": "computed" if config["binary"] else "not_applicable_non_binary_task",
        "t2_q_below_0_05_count": t2_q_count if config["binary"] else np.nan,
        "t3_q_below_0_05_count": t3_q_count,
        "t4_selected_group_count": len(selected_groups),
    }
    selection = {
        "registry_fingerprint": registry_fingerprint(task),
        "selected_groups": selected_groups,
        "delta_by_group": delta_by_group,
    }
    figure_payload = {
        "t2": t2_frame,
        "t3": t3_frame,
        "t4": t4_frame,
        "correlations": correlations,
        "feature_names": feature_names,
        "candidate_systems": candidate_systems,
    }
    print(
        "[%s] records=%d; T2=%s; T3 q<0.05=%d; selected groups=%d"
        % (
            task,
            len(masks),
            summary["t2_status"],
            t3_q_count,
            len(selected_groups),
        )
    )
    return summary, selection, figure_payload


def build_parser() -> argparse.ArgumentParser:
    """Create the command line interface for the audit."""
    parser = argparse.ArgumentParser(description="Structured-missingness audit")
    parser.add_argument(
        "--output-dir",
        default="analysis/results",
        help="Output root directory (default: analysis/results)",
    )
    parser.add_argument(
        "--split",
        choices=("train", "all"),
        default="train",
        help="Record subset to audit; use train for paper analyses (default: train)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for loader-compatible shuffled splits without stored indices (default: 42)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=tuple(TASKS),
        default=None,
        help="Optional task subset. Use '--tasks c19' to build a seed-specific "
             "C19 mask-group contract without auditing unrelated datasets.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not generate per-task or main-paper summary figures",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the structured-missingness audit for every supported SMART task."""
    args = build_parser().parse_args(argv)
    (
        registry_version,
        get_feature_names,
        get_candidate_systems,
        registry_fingerprint,
    ) = load_registry_api()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    selections: dict[str, Any] = {}
    figure_payloads: dict[str, dict[str, Any]] = {}
    requested_tasks = args.tasks or list(TASKS)
    for task in requested_tasks:
        config = TASKS[task]
        if not os.path.exists(config["path"]):
            raise FileNotFoundError("Dataset pickle not found for %s: %s" % (task, config["path"]))
        summary, selection, figure_payload = run_task(
            task,
            config,
            output_root,
            split=args.split,
            split_seed=args.split_seed,
            make_plots=not args.no_plot,
            get_feature_names=get_feature_names,
            get_candidate_systems=get_candidate_systems,
            registry_fingerprint=registry_fingerprint,
        )
        summaries.append(summary)
        selections[task] = selection
        figure_payloads[task] = figure_payload

    summary_path = output_root / "structured_missingness_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    selection_contract = {
        "registry_version": registry_version,
        "selection_rule": SELECTION_RULE,
        "split": args.split,
        "split_seed": args.split_seed,
        "tasks": selections,
    }
    selection_path = output_root / "selected_mask_groups.json"
    with open(selection_path, "w", encoding="utf-8") as handle:
        json.dump(selection_contract, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    figure_paths = None
    # The combined paper figure has a fixed multi-task panel contract. A task
    # subset may still emit its per-task plots, but cannot build that summary.
    if not args.no_plot and args.tasks is None:
        figure_paths = plot_main_paper_figure2(
            figure_payloads,
            selections,
            output_root=output_root,
            split=args.split,
            split_seed=args.split_seed,
        )

    print("\nStructured-missingness audit complete.")
    print("Summary: %s" % summary_path)
    print("Selected mask groups: %s" % selection_path)
    if figure_paths is not None:
        print("Main-paper Figure 2: %s" % figure_paths[0])
        print("Main-paper Figure 2 preview: %s" % figure_paths[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
