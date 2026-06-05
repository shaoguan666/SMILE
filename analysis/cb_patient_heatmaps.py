#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Patient-level structured-missingness C_b heatmaps for SMILE.

Run from SMART/:
    python analysis/cb_patient_heatmaps.py

The script selects examples from the fixed training split and computes
    C_b = R_b.T @ R_b / T - q_b q_b.T,  q_b = mean_t R_b[t]
where R_b = 1 - mask_b and mask_b uses 1=observed, 0=missing.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SMART_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SMART_ROOT))

from data.feature_registry import get_candidate_systems, get_feature_names, validate_registry  # noqa: E402
from data.mimiciii import load_mimic_iii_mortality  # noqa: E402


MIMIC_DATASET = "mimic_mortality"
MIMIC_SYSTEMS = get_candidate_systems(MIMIC_DATASET)


def repo_paths() -> tuple[Path, Path]:
    smart_root = Path(__file__).resolve().parents[1]
    repo_root = smart_root.parent
    return repo_root, smart_root


def load_training_examples(split_seed: int):
    train_dataset, _val_dataset, _test_dataset = load_mimic_iii_mortality(split_seed=split_seed)
    x = [sample["x"] for sample in train_dataset.data]
    y = [sample["labels"] for sample in train_dataset.data]
    masks = [sample["mask"] for sample in train_dataset.data]
    names = list(getattr(train_dataset, "patient_ids", ()))
    return x, y, masks, names


def scalar_label(label) -> int:
    if isinstance(label, (list, tuple, np.ndarray)):
        return int(float(label[0]))
    return int(float(label))


def compute_cb(mask) -> np.ndarray:
    observed = np.asarray(mask, dtype=np.float64)
    if observed.ndim != 2:
        raise ValueError(f"Expected mask shape (T,V), got {observed.shape}")
    missing = 1.0 - observed
    q = missing.mean(axis=0)
    cb = (missing.T @ missing) / missing.shape[0] - np.outer(q, q)
    np.fill_diagonal(cb, 0.0)
    return cb


def build_order(n_vars: int):
    order: list[int] = []
    boundaries = []
    used = set()
    for system, indices in MIMIC_SYSTEMS.items():
        group = [idx for idx in indices if idx < n_vars and idx not in used]
        if len(group) >= 2:
            start = len(order)
            order.extend(group)
            used.update(group)
            boundaries.append((start, len(order), system))
        else:
            for idx in group:
                if idx not in used:
                    order.append(idx)
                    used.add(idx)
    for idx in range(n_vars):
        if idx not in used:
            order.append(idx)
    return order, boundaries


def selected_groups(t4_csv: Path) -> set[str]:
    if not t4_csv.exists():
        return set(MIMIC_SYSTEMS)
    df = pd.read_csv(t4_csv)
    if "group" not in df or "selected" not in df:
        raise ValueError(f"T4 CSV does not use the structured-audit schema: {t4_csv}")
    keep = set()
    for _, row in df.iterrows():
        selected = str(row["selected"]).lower() in {"true", "1", "yes"}
        if selected:
            keep.add(str(row["group"]))
    return keep or set(MIMIC_SYSTEMS)


def block_pairs(n_vars: int, keep_systems: set[str]) -> list[tuple[int, int]]:
    pairs = []
    for system, indices in MIMIC_SYSTEMS.items():
        if system not in keep_systems:
            continue
        valid = [idx for idx in indices if idx < n_vars]
        for pos, i in enumerate(valid):
            for j in valid[pos + 1:]:
                pairs.append((i, j))
    return pairs


def summarize_sample(idx: int, y, masks, names, pairs) -> dict:
    cb = compute_cb(masks[idx])
    missing_rate = float(1.0 - np.asarray(masks[idx], dtype=np.float64).mean())
    offdiag = cb[~np.eye(cb.shape[0], dtype=bool)]
    if pairs:
        pair_vals = np.asarray([cb[i, j] for i, j in pairs], dtype=np.float64)
        block_strength = float(np.nanmean(pair_vals))
        block_positive = float(np.nanmean(np.maximum(pair_vals, 0.0)))
    else:
        block_strength = float(np.nanmean(offdiag))
        block_positive = float(np.nanmean(np.maximum(offdiag, 0.0)))
    return {
        "patient_index": idx,
        "patient_file": names[idx] if isinstance(names, list) and idx < len(names) else f"sample_{idx}",
        "label": scalar_label(y[idx]),
        "mean_missing": missing_rate,
        "block_cb_mean": block_strength,
        "block_cb_positive_mean": block_positive,
        "max_abs_cb": float(np.nanmax(np.abs(offdiag))),
    }


def select_patients(y, masks, names, pairs, per_label: int, min_missing: float, max_missing: float):
    rows = [summarize_sample(i, y, masks, names, pairs) for i in range(len(masks))]
    df = pd.DataFrame(rows)
    pool = df[(df["mean_missing"] >= min_missing) & (df["mean_missing"] <= max_missing)].copy()
    if pool.empty:
        pool = df.copy()

    selected = []
    for label in sorted(pool["label"].unique()):
        label_pool = pool[pool["label"] == label].copy()
        label_pool = label_pool.sort_values(
            ["block_cb_positive_mean", "block_cb_mean", "max_abs_cb"],
            ascending=[False, False, False],
        )
        selected.append(label_pool.head(per_label))
    out = pd.concat(selected, axis=0).reset_index(drop=True)
    return out.sort_values(["label", "block_cb_positive_mean"], ascending=[True, False])


def short_patient_id(patient_file: str) -> str:
    text = Path(str(patient_file)).stem
    match = re.search(r"(\d+)", text)
    return match.group(1) if match else text[:14]


def draw_heatmap(
    ax,
    cb,
    title: str,
    names: list[str],
    order,
    boundaries,
    vlim: float,
    show_xlabels: bool = True,
    show_ylabels: bool = True,
    annotate_groups: bool = True,
):
    cb_ordered = cb[np.ix_(order, order)]
    labels = [names[i] for i in order]
    image = ax.imshow(
        cb_ordered,
        cmap="RdBu_r",
        vmin=-vlim,
        vmax=vlim,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels if show_xlabels else [], rotation=90, ha="center", fontsize=6.2)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels if show_ylabels else [], fontsize=6.2)
    ax.tick_params(axis="both", length=0)
    ax.set_title(title, fontsize=10, pad=7)

    for spine in ax.spines.values():
        spine.set_visible(False)

    for start, end, system in boundaries:
        rect = plt.Rectangle(
            (start - 0.5, start - 0.5),
            end - start,
            end - start,
            linewidth=1.5,
            edgecolor="black",
            facecolor="none",
        )
        ax.add_patch(rect)
        if annotate_groups:
            ax.text(
                end - 0.15,
                start + 0.15,
                system.replace("_", " "),
                fontsize=6.2,
                ha="right",
                va="top",
                color="black",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.8},
            )
    return image


def make_grid(selected: pd.DataFrame, masks, out_base: Path, feature_names: list[str]):
    n_panels = len(selected)
    n_cols = min(2, n_panels)
    n_rows = int(math.ceil(n_panels / n_cols))
    order, boundaries = build_order(len(feature_names))

    cbs = [compute_cb(masks[int(row.patient_index)]) for _, row in selected.iterrows()]
    abs_values = np.concatenate([
        np.abs(cb[~np.eye(cb.shape[0], dtype=bool)]).ravel()
        for cb in cbs
    ])
    vlim = float(np.nanpercentile(abs_values, 98))
    vlim = min(max(vlim, 0.05), 0.25)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7.05, 3.05 * n_rows),
        constrained_layout=False,
    )
    axes = np.atleast_1d(axes).ravel()

    image = None
    for panel_idx, (ax, (_, row), cb) in enumerate(zip(axes, selected.iterrows(), cbs)):
        outcome = "non-survivor" if int(row.label) == 1 else "survivor"
        title = (
            f"P{short_patient_id(row.patient_file)} ({outcome})\n"
            f"miss={row.mean_missing:.2f}, block $C_b$={row.block_cb_mean:.3f}"
        )
        row_idx = panel_idx // n_cols
        col_idx = panel_idx % n_cols
        image = draw_heatmap(
            ax,
            cb,
            title,
            feature_names,
            order,
            boundaries,
            vlim,
            show_xlabels=(row_idx == n_rows - 1),
            show_ylabels=(col_idx == 0),
            annotate_groups=False,
        )

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.suptitle(
        "Train-split structured missingness: patient-level residual matrix $C_b$",
        fontsize=11,
        y=0.985,
    )
    fig.subplots_adjust(left=0.115, right=0.88, bottom=0.115, top=0.895, wspace=0.07, hspace=0.25)
    cbar_ax = fig.add_axes([0.905, 0.22, 0.018, 0.56])
    cbar = fig.colorbar(image, cax=cbar_ax)
    cbar.set_label("$C_{b,ij}$", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    for ext in ("png", "pdf", "svg"):
        path = out_base.with_suffix(f".{ext}")
        fig.savefig(path, dpi=450 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def make_individuals(selected: pd.DataFrame, masks, out_dir: Path, feature_names: list[str]):
    order, boundaries = build_order(len(feature_names))
    for _, row in selected.iterrows():
        cb = compute_cb(masks[int(row.patient_index)])
        offdiag = cb[~np.eye(cb.shape[0], dtype=bool)]
        vlim = min(max(float(np.nanpercentile(np.abs(offdiag), 98)), 0.05), 0.25)
        fig, ax = plt.subplots(figsize=(4.8, 4.5))
        outcome = "non-survivor" if int(row.label) == 1 else "survivor"
        title = f"Patient {short_patient_id(row.patient_file)} ({outcome})"
        image = draw_heatmap(ax, cb, title, feature_names, order, boundaries, vlim)
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("$C_{b,ij}$", fontsize=8)
        fig.tight_layout()
        stem = f"cb_patient_{int(row.patient_index):05d}_y{int(row.label)}"
        for ext in ("png", "pdf", "svg"):
            fig.savefig(out_dir / f"{stem}.{ext}", dpi=450 if ext == "png" else None, bbox_inches="tight")
        plt.close(fig)


def main():
    repo_root, smart_root = repo_paths()
    parser = argparse.ArgumentParser(description="Plot patient-level SMILE C_b heatmaps.")
    parser.add_argument(
        "--t4-csv",
        type=Path,
        default=smart_root / "analysis" / "results" / "mimic_mortality" / "t4_block.csv",
        help="Structured-audit T4 CSV used to identify selected systems.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "smile-paper" / "output" / "figures" / "cb_heatmaps",
        help="Output directory.",
    )
    parser.add_argument("--per-label", type=int, default=2, help="Number of selected patients per label.")
    parser.add_argument("--split-seed", type=int, default=42, help="Training-split seed used by the SMART loader.")
    parser.add_argument("--min-missing", type=float, default=0.05)
    parser.add_argument("--max-missing", type=float, default=0.95)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    x, y, masks, names = load_training_examples(args.split_seed)
    _ = x  # The heatmap uses masks only; loading x confirms the dataset payload.

    validate_registry(MIMIC_DATASET, np.asarray(masks[0]).shape[1])
    feature_names = get_feature_names(MIMIC_DATASET)
    keep_systems = selected_groups(args.t4_csv)
    pairs = block_pairs(len(feature_names), keep_systems)
    selected = select_patients(y, masks, names, pairs, args.per_label, args.min_missing, args.max_missing)

    selected["selection_note"] = (
        "Train-split label-stratified examples by positive within-retained-system C_b."
    )
    metadata_path = args.out_dir / "mimic_mortality_structured_missingness_cb_patient_metadata.csv"
    selected.to_csv(metadata_path, index=False)

    out_base = args.out_dir / "mimic_mortality_structured_missingness_cb_patient_grid"
    make_grid(selected, masks, out_base, feature_names)
    make_individuals(selected, masks, args.out_dir, feature_names)

    print(f"Selected {len(selected)} patients.")
    print(f"Metadata: {metadata_path}")
    print(f"Grid PNG: {out_base.with_suffix('.png')}")
    print(f"Grid PDF: {out_base.with_suffix('.pdf')}")
    print(f"Grid SVG: {out_base.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
