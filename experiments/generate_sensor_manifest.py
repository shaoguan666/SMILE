"""Generate deterministic test-split manifests for sensor-removal evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.challenge2012 import load_challenge_2012
from data.challenge2019 import load_challenge_2019
from data.mimiciii import (
    load_mimic_iii_decompensation,
    load_mimic_iii_mortality,
)
from experiments.sensor_robustness import (
    DEFAULT_MANIFEST_SEED,
    DEFAULT_REPLICATES,
    create_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mimic_mortality", "mimic_decompensation"],
        choices=["mimic_mortality", "mimic_decompensation", "c12", "c19"],
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--manifest-seed", type=int, default=DEFAULT_MANIFEST_SEED)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "export" / "sensor_robustness_v1" / "manifests",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaders = {
        "mimic_mortality": load_mimic_iii_mortality,
        "mimic_decompensation": load_mimic_iii_decompensation,
        "c12": load_challenge_2012,
        "c19": load_challenge_2019,
    }
    for dataset_name in args.datasets:
        _, _, test_dataset = loaders[dataset_name](split_seed=args.split_seed)
        path = args.output_root / f"{dataset_name}_test_seed{args.split_seed}.npz"
        manifest = create_manifest(
            path,
            dataset_name=dataset_name,
            split_seed=args.split_seed,
            test_dataset=test_dataset,
            replicates=args.replicates,
            manifest_seed=args.manifest_seed,
            overwrite=args.overwrite,
        )
        print(
            f"{dataset_name}: {manifest.path} sha256={manifest.file_sha256} "
            f"patients={len(manifest.patient_ids)} features={manifest.num_features} "
            f"replicates={manifest.num_replicates}"
        )


if __name__ == "__main__":
    main()
