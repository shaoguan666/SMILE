"""Aggregate completed WaveGNN, MissTSM, or ISTS-PLM local runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baseline_utils import ROOT, STANDARD_SEEDS, aggregate_model


MODELS = {
    "wavegnn": "WaveGNN",
    "misstm": "MissTSM",
    "ists_plm": "ISTS-PLM",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(STANDARD_SEEDS))
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "export" / "recent_baselines"
    )
    args = parser.parse_args()
    outputs = aggregate_model(
        args.output_root, args.model, MODELS[args.model], args.seeds
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
