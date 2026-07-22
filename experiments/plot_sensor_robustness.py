"""Plot absolute AUPRC and prevalence-adjusted retention curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DISPLAY = {
    "smart-smile-lean": "SMILE (Ours)",
    "smart": "SMART",
    "ists_plm": "ISTS-PLM",
    "wavegnn": "WaveGNN",
    "misstm": "MissTSM",
    "atenet": "ATENet",
}
STYLES = {
    "smart-smile-lean": {"color": "#B2182B", "lw": 2.8, "ls": "-", "marker": "o"},
    "smart": {"color": "#2166AC", "lw": 2.0, "ls": "--", "marker": "s"},
    "ists_plm": {"color": "#1B9E77", "lw": 1.6, "ls": "-.", "marker": "^"},
    "wavegnn": {"color": "#7570B3", "lw": 1.6, "ls": "-", "marker": "D"},
    "misstm": {"color": "#E6AB02", "lw": 1.6, "ls": "--", "marker": "v"},
    "atenet": {"color": "#666666", "lw": 1.6, "ls": ":", "marker": "P"},
}
TITLES = {
    "mimic_mortality": "MIMIC-III Mortality",
    "mimic_decompensation": "MIMIC-III Decompensation",
    "c12": "PhysioNet-2012 Mortality",
    "c19": "PhysioNet-2019 Sepsis",
}
FEATURES = {
    "mimic_mortality": 17,
    "mimic_decompensation": 17,
    "c12": 37,
    "c19": 34,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregate", type=Path,
        default=ROOT / "export" / "sensor_robustness_v1" / "aggregate.json"
    )
    parser.add_argument(
        "--output-prefix", type=Path,
        default=ROOT / "export" / "sensor_robustness_v1" / "figure_sensor_robustness"
    )
    parser.add_argument("--publication", action="store_true")
    parser.add_argument(
        "--single-row", action="store_true",
        help="Lay the four panels out in a single 1x4 row (paper figure).",
    )
    return parser.parse_args()


def _series(model_block: dict, key: str):
    ks = sorted(int(value) for value in model_block)
    values = np.asarray([model_block[str(k)][key] for k in ks], dtype=float)
    ci_key = "auprc_95ci" if key == "auprc_mean" else "retention_95ci"
    ci = np.asarray(
        [model_block[str(k)].get(ci_key, [np.nan, np.nan]) for k in ks], dtype=float
    )
    return np.asarray(ks), values, ci


def _plot_single_row(payload: dict, datasets: list, args: argparse.Namespace) -> None:
    """Four panels in one row: (dataset x metric) = 1x4 for a paper figure."""
    panels = [
        (datasets[0], "auprc_mean"),
        (datasets[0], "retention_mean"),
        (datasets[1], "auprc_mean"),
        (datasets[1], "retention_mean"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.7))
    for column, (dataset, key) in enumerate(panels):
        axis = axes[column]
        dataset_block = payload["aggregates"][dataset]
        for model in DISPLAY:
            model_block = dataset_block.get(model)
            if not model_block:
                continue
            style = STYLES[model]
            ks, values, ci = _series(model_block, key)
            axis.plot(ks, values, label=DISPLAY[model], **style)
            if np.isfinite(ci).all():
                axis.fill_between(
                    ks, ci[:, 0], ci[:, 1], color=style["color"], alpha=0.10,
                    linewidth=0,
                )
        nfeat = FEATURES.get(dataset, 17)
        present_ks = sorted(
            {int(k) for model in DISPLAY if dataset_block.get(model)
             for k in dataset_block[model]}
        )
        tick_labels = [f"{k}\n{100.0 * k / nfeat:.1f}%" for k in present_ks]
        metric_short = "AUPRC" if key == "auprc_mean" else "Retention"
        short_dataset = TITLES[dataset].replace("MIMIC-III ", "")
        axis.set_title(f"{short_dataset}: {metric_short}", fontsize=11)
        axis.set_ylabel("AUPRC" if key == "auprc_mean" else "Adjusted AUPRC Retention")
        axis.set_xlabel(f"Sensors removed (k of {nfeat})")
        if key == "retention_mean":
            axis.axhline(1.0, color="#BBBBBB", lw=0.8, zorder=0)
        axis.grid(axis="y", color="#DDDDDD", lw=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xticks(present_ks, tick_labels)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.10),
               ncol=6, frameon=False)
    if not args.publication:
        fig.text(
            0.5, 0.5, "EXPLORATORY — NOT FOR PUBLICATION", ha="center", va="center",
            fontsize=20, color="#B2182B", alpha=0.14, rotation=12,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(
        f"Wrote {args.output_prefix.with_suffix('.pdf')} and "
        f"{args.output_prefix.with_suffix('.png')}"
    )


def main() -> None:
    args = parse_args()
    payload = json.loads(args.aggregate.read_text(encoding="utf-8"))
    if args.publication and not payload.get("publication_ready"):
        reasons = {
            "complete": payload.get("complete"),
            "clean_reconciliation": payload.get("clean_reconciliation", {}).get("passed"),
        }
        raise SystemExit(f"Publication plot blocked: {reasons}")

    datasets = [value for value in TITLES if value in payload["aggregates"]]
    if len(datasets) != 2:
        raise SystemExit("The figure requires both MIMIC binary datasets")

    if args.single_row:
        _plot_single_row(payload, datasets, args)
        return

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.8), sharex="col")
    for column, dataset in enumerate(datasets):
        dataset_block = payload["aggregates"][dataset]
        for model in DISPLAY:
            model_block = dataset_block.get(model)
            if not model_block:
                continue
            style = STYLES[model]
            for row, key in enumerate(("auprc_mean", "retention_mean")):
                ks, values, ci = _series(model_block, key)
                axis = axes[row, column]
                axis.plot(ks, values, label=DISPLAY[model], **style)
                if np.isfinite(ci).all():
                    axis.fill_between(
                        ks, ci[:, 0], ci[:, 1], color=style["color"], alpha=0.10,
                        linewidth=0,
                    )
        nfeat = FEATURES.get(dataset, 17)
        present_ks = sorted(
            {int(k) for model in DISPLAY if dataset_block.get(model)
             for k in dataset_block[model]}
        )
        tick_labels = [f"{k}\n{100.0 * k / nfeat:.1f}%" for k in present_ks]
        axes[0, column].set_title(TITLES[dataset], fontsize=11)
        axes[0, column].set_ylabel("AUPRC")
        axes[1, column].set_ylabel("Adjusted AUPRC Retention")
        axes[1, column].axhline(1.0, color="#BBBBBB", lw=0.8, zorder=0)
        axes[1, column].set_xlabel(f"Sensors removed (k of {nfeat})")
        for row in range(2):
            axes[row, column].grid(axis="y", color="#DDDDDD", lw=0.6)
            axes[row, column].spines[["top", "right"]].set_visible(False)
            axes[row, column].set_xticks(present_ks, tick_labels)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=3, frameon=False)
    if not args.publication:
        fig.text(
            0.5, 0.5, "EXPLORATORY — NOT FOR PUBLICATION", ha="center", va="center",
            fontsize=20, color="#B2182B", alpha=0.14, rotation=25,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(
        f"Wrote {args.output_prefix.with_suffix('.pdf')} and "
        f"{args.output_prefix.with_suffix('.png')}"
    )


if __name__ == "__main__":
    main()
