"""Run the audited SMILE BIBM experiment grid in an isolated output root."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
SMART_DIR = HERE.parents[1]
DEFAULT_CONFIG = HERE / "configs" / "bibm_smile_experiments.json"

VARIANT_FLAGS = {
    "smart": [],
    "smart-smile-lean": ["--use-smile-lean"],
    "smart-smile-lean-v2": ["--use-smile-lean-v2"],
    "smart-smile-lean-no-density": ["--use-smile-lean", "--abl-no-density"],
    "smart-smile-lean-no-mnar-bias": ["--use-smile-lean", "--abl-no-mnar-bias"],
    "smart-smile-lean-no-film": ["--use-smile-lean", "--abl-no-film"],
    "smart-smile-lean-no-time-mnar": ["--use-smile-lean", "--abl-no-time-mnar"],
    "smart-smile-lean-no-time-pe": ["--use-smile-lean", "--abl-no-time-pe"],
    "smart-smile-lean-no-mnar-bias-no-time-mnar": [
        "--use-smile-lean", "--abl-no-mnar-bias", "--abl-no-time-mnar"
    ],
    "smart-smile-lean-samepretrain": ["--use-smile-lean-samepretrain"],
    "smart-smile-lean-v2-no-dynamic-mnar": [
        "--use-smile-lean-v2", "--abl-no-dynamic-mnar"
    ],
}


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def launch_prefix(args, run_idx: int) -> list[str]:
    if not args.use_torchrun:
        return [args.python_executable]
    return [
        args.python_executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node",
        str(args.nproc_per_node),
        "--master_port",
        str(args.master_port_base + run_idx - 1),
    ]


def launch_env(args) -> dict[str, str]:
    env = os.environ.copy()
    if args.devices:
        env["CUDA_VISIBLE_DEVICES"] = args.devices
    if args.use_torchrun and env.get("SMART_SAFE_NCCL", "1") == "1":
        for key, value in {
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "TORCH_NCCL_BLOCKING_WAIT": "1",
            "NCCL_P2P_DISABLE": "1",
            "NCCL_IB_DISABLE": "1",
        }.items():
            env.setdefault(key, value)
    return env


def run_cmd(cmd: list[str], tag: str, args, env: dict[str, str]) -> bool:
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {tag}")
    print("CMD:", " ".join(cmd))
    if args.devices:
        print(f"CUDA_VISIBLE_DEVICES={args.devices}")
    if args.dry_run:
        print("[DRY RUN] skipped")
        return True
    return subprocess.run(cmd, cwd=SMART_DIR, env=env).returncode == 0


def absolute_output_root(output_root: Path) -> Path:
    return output_root if output_root.is_absolute() else SMART_DIR / output_root


def checkpoint(output_root: Path, dataset: str, variant: str, seed: int, name: str) -> Path:
    return absolute_output_root(output_root) / dataset / variant / f"seed_{seed}" / name


def los_flags(dataset: str) -> list[str]:
    if dataset != "mimic_lengthofstay":
        return []
    return [
        "--los-task", "classification",
        "--los-label-unit", "auto",
        "--los-save-metric", "auc_micro",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Display names from config, e.g. Backbone SMILE-Full.")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--mask-group-config", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--use-torchrun", action="store_true")
    parser.add_argument("--devices", default=None)
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--master-port-base", type=int, default=29500)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--finetune-only", action="store_true")
    parser.add_argument("--pretrain-only", action="store_true")
    args = parser.parse_args()
    if args.finetune_only and args.pretrain_only:
        parser.error("--finetune-only and --pretrain-only cannot be combined")

    cfg = load_config(args.config)
    revision = cfg["revision_defaults"]
    output_root = args.output_root or Path(revision["output_root"])
    mask_group_config = args.mask_group_config or revision["mask_group_config"]
    split_seed = args.split_seed if args.split_seed is not None else revision["split_seed"]
    run_tag = args.run_tag or revision["run_tag"]
    if split_seed != 42:
        parser.error("The audited BIBM revision is fixed to --split-seed 42.")

    variant_map = cfg["implemented_variants"]
    selected_names = args.models or list(variant_map.keys())
    missing = [name for name in selected_names if name not in variant_map]
    if missing:
        parser.error(f"Unknown or not implemented variant names: {missing}")
    variants = [(name, variant_map[name]) for name in selected_names]
    unmapped = [variant for _, variant in variants if variant not in VARIANT_FLAGS]
    if unmapped:
        parser.error(f"No command flag mapping for implemented variants: {unmapped}")

    datasets = args.datasets or cfg["datasets"]
    seeds = args.seeds or cfg["seeds"]
    defaults = cfg["runner_defaults"]
    plan = [
        (name, variant, dataset, seed)
        for name, variant in variants
        for dataset in datasets
        for seed in seeds
    ]
    print(f"Revision: {run_tag}")
    print(f"Output root: {output_root}")
    print(f"Mask group config: {mask_group_config}")
    print(f"Split seed: {split_seed}")
    print(f"Total experiments: {len(plan)}")

    env = launch_env(args)
    failures = []
    skipped_pretrain = 0
    skipped_finetune = 0
    for index, (display_name, variant, dataset, seed) in enumerate(plan, 1):
        flags = VARIANT_FLAGS[variant]
        tag = f"[{run_tag} {index:>3}/{len(plan)}] {display_name} | {dataset} | seed={seed}"
        pre_ckpt = checkpoint(output_root, dataset, variant, seed, "checkpoint-mse.pth")
        ft_ckpt = checkpoint(output_root, dataset, variant, seed, "checkpoint-prc.pth")

        if not args.finetune_only:
            if not args.force and pre_ckpt.exists():
                print(f"{tag} | pretrain: SKIP (revision checkpoint exists)")
                skipped_pretrain += 1
            else:
                pretrain = launch_prefix(args, index) + [
                    "main_pretrain.py",
                    "--dataset", dataset,
                    "--seed", str(seed),
                    "--epochs", str(defaults["pretrain_epochs"]),
                    "--batch_size", str(defaults["batch_size"]),
                    "--save_dir", str(output_root),
                    "--mask-group-config", str(mask_group_config),
                    "--split-seed", str(split_seed),
                ] + flags
                if variant == "smart" and dataset not in ("mimic_decompensation", "mimic_lengthofstay"):
                    pretrain.append("--save-last")
                if not run_cmd(pretrain, f"{tag} | PRETRAIN", args, env):
                    failures.append(f"{tag} pretrain")
                    continue

        if not args.pretrain_only:
            if not args.force and ft_ckpt.exists():
                print(f"{tag} | finetune: SKIP (revision checkpoint exists)")
                skipped_finetune += 1
                continue
            if not args.dry_run and not pre_ckpt.exists():
                print(f"{tag} | finetune: missing revision pretrain checkpoint")
                failures.append(f"{tag} finetune (missing pretrain checkpoint)")
                continue
            finetune = launch_prefix(args, index) + [
                "main_finetune.py",
                "--dataset", dataset,
                "--seed", str(seed),
                "--epochs", str(defaults["finetune_epochs"]),
                "--batch_size", str(defaults["batch_size"]),
                "--save_dir", str(output_root),
                "--pretrain-dir", str(output_root / dataset / variant / f"seed_{seed}"),
                "--split-seed", str(split_seed),
            ] + los_flags(dataset) + flags
            if not run_cmd(finetune, f"{tag} | FINETUNE", args, env):
                failures.append(f"{tag} finetune")

    print(f"\nFinished. Skipped pretrain={skipped_pretrain}, skipped finetune={skipped_finetune}")
    if failures:
        print("Failed:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
