#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Patient-level structured-missingness case-study figures for the SMILE paper.

The script selects label-stratified patient exemplars and plots, for each
patient, a co-missingness residual heatmap above the temporal observation mask.
It is intended as a candidate replacement/supplement for Figure 2 when the
paper text needs concrete training-split structured-missingness examples.

Run from SMART/:
    python analysis/patient_mnar_case_studies.py
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

SMART_ROOT = Path(__file__).resolve().parents[1]
if str(SMART_ROOT) not in sys.path:
    sys.path.insert(0, str(SMART_ROOT))

from data.challenge2012 import load_challenge_2012  # noqa: E402
from data.challenge2019 import load_challenge_2019  # noqa: E402
from data.feature_registry import get_candidate_systems, get_feature_names, validate_registry  # noqa: E402
from data.mimiciii import load_mimic_iii_mortality  # noqa: E402


DATASETS = {
    "C12_Mortality": {
        "registry": "c12",
        "loader": load_challenge_2012,
        "max_t": 48,
        "short": "C12",
    },
    "C19_Sepsis": {
        "registry": "c19",
        "loader": load_challenge_2019,
        "max_t": 60,
        "short": "C19",
    },
    "MIMIC3_Mortality": {
        "registry": "mimic_mortality",
        "loader": load_mimic_iii_mortality,
        "max_t": 48,
        "short": "MIMIC",
    },
}

SYSTEM_COLORS = [
    "#92B1D9",
    "#C1D8E9",
    "#DBDDEF",
    "#F6C8B6",
    "#B9D8C2",
    "#D4D4D4",
    "#E9D7B8",
    "#B7CCD4",
    "#D6C6E1",
    "#C8D6A8",
]


def load_training_examples(cfg):
    train_dataset, _val_dataset, _test_dataset = cfg["loader"]()
    y = [sample["labels"] for sample in train_dataset.data]
    masks = [sample["mask"] for sample in train_dataset.data]
    names = list(getattr(train_dataset, "patient_ids", ()))
    return y, masks, names


def scalar_label(label) -> int:
    arr = np.asarray(label).ravel()
    if arr.size:
        return int(float(arr[0]))
    return int(float(label))


def compute_cb(mask: np.ndarray) -> np.ndarray:
    observed = np.asarray(mask, dtype=np.float64)
    missing = 1.0 - observed
    q = missing.mean(axis=0)
    cb = (missing.T @ missing) / missing.shape[0] - np.outer(q, q)
    np.fill_diagonal(cb, 0.0)
    return cb


def system_pairs(systems: dict[str, list[int]], n_vars: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for var_ids in systems.values():
        valid = [idx for idx in var_ids if idx < n_vars]
        for pos, i in enumerate(valid):
            for j in valid[pos + 1 :]:
                pairs.append((i, j))
    return pairs


def panel_order(feature_names: list[str], systems: dict[str, list[int]]):
    used = set()
    ordered = []
    boundaries = []
    n_vars = len(feature_names)
    for system, var_ids in systems.items():
        valid = [idx for idx in var_ids if idx < n_vars and idx not in used]
        if not valid:
            continue
        start = len(ordered)
        ordered.extend(valid)
        used.update(valid)
        boundaries.append((start, len(ordered), system))
    remaining = [idx for idx in range(n_vars) if idx not in used]
    if remaining:
        start = len(ordered)
        ordered.extend(remaining)
        boundaries.append((start, len(ordered), "Other"))
    return ordered, boundaries


def compute_patient_stats(y, masks, names, systems, max_t: int) -> pd.DataFrame:
    n_vars = np.asarray(masks[0]).shape[1]
    pairs = system_pairs(systems, n_vars)
    rows = []
    for idx, mask in enumerate(masks):
        m_full = np.asarray(mask, dtype=np.float64)
        if m_full.ndim != 2 or m_full.shape[0] < min(24, max_t // 2):
            continue
        m = m_full[: min(max_t, m_full.shape[0])]
        observed_fraction = m.mean(axis=1)
        cb = compute_cb(m)
        pair_vals = np.asarray([cb[i, j] for i, j in pairs], dtype=np.float64)
        block_pos = float(np.mean(np.maximum(pair_vals, 0.0))) if len(pair_vals) else 0.0
        block_mean = float(np.mean(pair_vals)) if len(pair_vals) else 0.0
        rows.append(
            {
                "idx": idx,
                "patient_file": str(names[idx]) if isinstance(names, list) and idx < len(names) else "",
                "label": scalar_label(y[idx]),
                "length": int(m_full.shape[0]),
                "overall_observed": float(m.mean()),
                "missing_rate": float(1.0 - m.mean()),
                "temporal_std": float(np.std(observed_fraction)),
                "temporal_range": float(np.max(observed_fraction) - np.min(observed_fraction)),
                "early_observed": float(np.mean(observed_fraction[: min(6, len(observed_fraction))])),
                "late_observed": float(np.mean(observed_fraction[-min(6, len(observed_fraction)) :])),
                "early_late_shift": float(
                    abs(
                        np.mean(observed_fraction[: min(6, len(observed_fraction))])
                        - np.mean(observed_fraction[-min(6, len(observed_fraction)) :])
                    )
                ),
                "block_cb_positive_mean": block_pos,
                "block_cb_mean": block_mean,
                "dense_time_fraction": float(np.mean(observed_fraction >= 0.65)),
                "sparse_time_fraction": float(np.mean(observed_fraction <= 0.15)),
            }
        )
    return pd.DataFrame(rows)


def _readability_score(overall_observed: pd.Series) -> pd.Series:
    return (1.0 - (overall_observed - 0.35).abs() / 0.35).clip(0.0, 1.0)


def select_contrast_cases(stats: pd.DataFrame, per_label: int = 3) -> pd.DataFrame:
    """Choose volatile positives and comparatively stable negatives."""
    selected = []
    usable = stats[
        (stats["overall_observed"].between(0.05, 0.85))
        & (stats["temporal_range"] >= 0.15)
    ].copy()
    if usable.empty:
        usable = stats.copy()

    for label in (1, 0):
        pool = usable[usable["label"] == label].copy()
        if pool.empty:
            continue
        block = pool["block_cb_positive_mean"]
        if label == 1:
            pool["selection_score"] = (
                1.7 * pool["temporal_std"]
                + 0.9 * pool["temporal_range"]
                + 1.2 * block
                + 0.2 * _readability_score(pool["overall_observed"])
            )
            pool = pool.sort_values("selection_score", ascending=False)
            reason = "high temporal fluctuation and block co-missingness"
        else:
            median_std = float(pool["temporal_std"].median())
            stable_pool = pool[pool["temporal_std"] <= median_std].copy()
            if len(stable_pool) < per_label:
                stable_pool = pool.copy()
            stable_pool["selection_score"] = (
                1.5 * stable_pool["block_cb_positive_mean"]
                + 0.5 * _readability_score(stable_pool["overall_observed"])
                - 0.7 * stable_pool["temporal_std"]
            )
            pool = stable_pool.sort_values("selection_score", ascending=False)
            reason = "stable survivor contrast with visible block structure"
        head = pool.head(per_label).copy()
        head["selection_note"] = reason
        selected.append(head)

    out = pd.concat(selected, ignore_index=True)
    return out.sort_values(["label", "selection_score"], ascending=[False, False]).reset_index(drop=True)


def draw_heatmap(ax, mask, feature_names, order, boundaries, title, vlim):
    cb = compute_cb(mask)
    cb_ordered = cb[np.ix_(order, order)]
    im = ax.imshow(cb_ordered, cmap="RdBu_r", vmin=-vlim, vmax=vlim, interpolation="nearest")
    ax.set_title(title, loc="left", fontsize=8.2, fontweight="bold", pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for start, end, system in boundaries:
        rect = plt.Rectangle(
            (start - 0.5, start - 0.5),
            end - start,
            end - start,
            linewidth=0.8,
            edgecolor="#111827",
            facecolor="none",
        )
        ax.add_patch(rect)
        if end - start >= 3:
            ax.text(
                start + 0.15,
                end - 0.18,
                system.replace("_", " "),
                fontsize=4.6,
                ha="left",
                va="bottom",
                color="#111827",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.68, "pad": 0.4},
            )
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color("#667085")
    return im


def draw_temporal_mask(fig, spec, mask, order, boundaries, max_t, show_ylabel=False):
    sub = spec.subgridspec(nrows=2, ncols=1, height_ratios=[0.28, 1.0], hspace=0.03)
    ax_trace = fig.add_subplot(sub[0])
    ax_mask = fig.add_subplot(sub[1], sharex=ax_trace)

    m = np.asarray(mask, dtype=np.float64)[:max_t]
    m_ordered = m[:, order].T
    observed_fraction = m.mean(axis=1)
    time = np.arange(len(observed_fraction))

    ax_trace.plot(time, observed_fraction, color="#B85B50", lw=1.05)
    ax_trace.fill_between(time, 0, observed_fraction, color="#F6C8B6", alpha=0.35, lw=0)
    ax_trace.set_ylim(0, 1)
    ax_trace.set_yticks([0, 1])
    ax_trace.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_trace.tick_params(axis="x", labelbottom=False, length=0)
    ax_trace.tick_params(axis="y", labelsize=5.4, pad=1)
    ax_trace.grid(axis="y", color="#E5E7EB", lw=0.35)
    for spine in ax_trace.spines.values():
        spine.set_visible(False)

    cmap = mcolors.ListedColormap(["#F4F7FA", "#2F5F98"])
    ax_mask.imshow(m_ordered, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    ax_mask.set_xlim(-0.5, max_t - 0.5)
    xticks = [x for x in (0, 12, 24, 36, 48, 60) if x <= max_t]
    ax_mask.set_xticks(xticks)
    ax_mask.set_xticklabels(xticks, fontsize=5.7)
    ax_mask.set_yticks([])
    ax_mask.tick_params(axis="x", length=2, pad=1)

    for hour in range(24, max_t + 1, 24):
        ax_trace.axvline(hour - 0.5, color="#9CA3AF", ls="--", lw=0.6, alpha=0.75)
        ax_mask.axvline(hour - 0.5, color="#9CA3AF", ls="--", lw=0.6, alpha=0.75)
    for start, end, _system in boundaries:
        ax_mask.axhline(start - 0.5, color="white", lw=0.55)
        ax_mask.axhline(end - 0.5, color="white", lw=0.55)

    strip_x = -2.0
    for k, (start, end, _system) in enumerate(boundaries):
        rect = plt.Rectangle(
            (strip_x, start - 0.5),
            0.5,
            end - start,
            facecolor=SYSTEM_COLORS[k % len(SYSTEM_COLORS)],
            edgecolor="none",
            clip_on=False,
        )
        ax_mask.add_patch(rect)

    if show_ylabel:
        ax_mask.set_ylabel("Grouped\nvariables", fontsize=6)

    for spine in ax_mask.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#344054")


def plot_dataset_cases(dataset: str, out_dir: Path, paper_dir: Path | None, per_label: int):
    cfg = DATASETS[dataset]
    y, masks, names = load_training_examples(cfg)
    validate_registry(cfg["registry"], np.asarray(masks[0]).shape[1])
    feature_names = get_feature_names(cfg["registry"])
    systems = get_candidate_systems(cfg["registry"])
    order, boundaries = panel_order(feature_names, systems)
    stats = compute_patient_stats(y, masks, names, systems, cfg["max_t"])
    selected = select_contrast_cases(stats, per_label=per_label)
    selected.insert(0, "dataset", dataset)

    selected_masks = [np.asarray(masks[int(row.idx)], dtype=np.float64)[: cfg["max_t"]] for _, row in selected.iterrows()]
    cbs = [compute_cb(mask) for mask in selected_masks]
    offdiag_values = np.concatenate([np.abs(cb[~np.eye(cb.shape[0], dtype=bool)]) for cb in cbs])
    vlim = float(np.nanpercentile(offdiag_values, 98))
    vlim = min(max(vlim, 0.05), 0.25)

    n = len(selected)
    fig = plt.figure(figsize=(max(11.8, 2.1 * n), 6.6), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=n,
        height_ratios=[1.0, 1.22],
        left=0.035,
        right=0.935,
        top=0.91,
        bottom=0.10,
        wspace=0.16,
        hspace=0.13,
    )

    im = None
    for col, (_, row) in enumerate(selected.iterrows()):
        panel = chr(ord("A") + col)
        mask = np.asarray(masks[int(row.idx)], dtype=np.float64)[: cfg["max_t"]]
        outcome = "y=1" if int(row.label) == 1 else "y=0"
        title = (
            f"{panel}. Record {int(row.idx)}, {outcome}\n"
            f"obs={row.overall_observed:.2f}, std={row.temporal_std:.2f}, block={row.block_cb_positive_mean:.2f}"
        )
        im = draw_heatmap(fig.add_subplot(gs[0, col]), mask, feature_names, order, boundaries, title, vlim)
        draw_temporal_mask(fig, gs[1, col], mask, order, boundaries, cfg["max_t"], show_ylabel=(col == 0))

    fig.suptitle(
        f"{cfg['short']} train-split structured missingness examples",
        fontsize=13,
        fontweight="bold",
        y=0.975,
    )
    fig.text(0.49, 0.045, "Time step (hours since admission)", ha="center", fontsize=8)

    cbar_ax = fig.add_axes([0.948, 0.60, 0.012, 0.25])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("$C_{b,ij}$", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#2F5F98", edgecolor="none", label="Observed"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#F4F7FA", edgecolor="#D0D5DD", label="Missing"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#F6C8B6", edgecolor="#B85B50", label="Observed feature fraction"),
    ]
    fig.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(0.935, 0.033),
        frameon=False,
        ncol=3,
        fontsize=7,
        handlelength=1.1,
        columnspacing=0.9,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"figure2_patient_structured_missingness_cases_{dataset.lower()}"
    paths = []
    for ext in ("png", "pdf", "svg"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=360 if ext == "png" else None, bbox_inches="tight", pad_inches=0.04)
        paths.append(path)
    plt.close(fig)

    selected_path = out_dir / f"{stem}_selected.csv"
    selected.to_csv(selected_path, index=False)
    paths.append(selected_path)

    if paper_dir is not None:
        paper_dir.mkdir(parents=True, exist_ok=True)
        for path in paths:
            if path.suffix.lower() in {".png", ".pdf", ".svg", ".csv"}:
                shutil.copy2(path, paper_dir / path.name)
    return paths, selected


def default_paper_dir() -> Path | None:
    for candidate in (Path("../smile-paper/figures"), Path("smile-paper/figures")):
        if candidate.parent.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="Draw training-split structured-missingness case-study figures.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS.keys()),
        default=["C12_Mortality", "C19_Sepsis", "MIMIC3_Mortality"],
    )
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/results/patient_case_candidates"))
    parser.add_argument("--paper-dir", type=Path, default=default_paper_dir())
    parser.add_argument("--per-label", type=int, default=3)
    args = parser.parse_args()

    all_selected = []
    for dataset in args.datasets:
        paths, selected = plot_dataset_cases(dataset, args.out_dir, args.paper_dir, args.per_label)
        all_selected.append(selected)
        print(f"{dataset}:")
        for path in paths:
            print(f"  {path}")
        print(selected[[
            "idx",
            "label",
            "overall_observed",
            "temporal_std",
            "temporal_range",
            "early_observed",
            "late_observed",
            "block_cb_positive_mean",
        ]].to_string(index=False))

    combined = pd.concat(all_selected, ignore_index=True)
    combined_path = args.out_dir / "figure2_patient_structured_missingness_cases_selected_all.csv"
    combined.to_csv(combined_path, index=False)
    if args.paper_dir is not None:
        args.paper_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(combined_path, args.paper_dir / combined_path.name)
    print(f"Combined metadata: {combined_path}")


if __name__ == "__main__":
    main()
