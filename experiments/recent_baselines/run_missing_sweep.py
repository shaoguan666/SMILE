"""Leave-random-sensor-out missing-ratio sweep (smoke test).

Evaluates SMILE (smart-smile-lean) and the recent baselines under increasing
test-time sensor removal, using already-trained checkpoints. No retraining.

The channel dropout itself lives in data.dataloader.collate_fn and is switched
on via the environment variables SMILE_EVAL_SENSOR_DROP (ratio) and
SMILE_EVAL_SENSOR_DROP_SEED, so every model drops the identical channels for a
given (seed, ratio). This harness only orchestrates eval-only subprocess calls
and aggregates their eval_results.json outputs.

Purpose: check whether SMILE stays clearly on top at 30% / 50% before investing
in the full Figure-A grid.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Baseline runners: script + eval batch size (conservative for an 8 GB laptop GPU).
BASELINES = {
    "ists_plm": ("run_ists_plm_mimic.py", ["--batch-size", "8"]),
    "wavegnn": ("run_wavegnn_mimic.py", ["--batch-size", "8"]),
    "atenet": ("run_atenet_mimic.py", ["--batch-size", "16"]),
    "misstm": ("run_misstm_mimic.py", ["--batch-size", "16"]),
}
SMILE_MODEL = "smart-smile-lean"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mimic_decompensation")
    p.add_argument("--ratios", nargs="+", type=float, default=[0.0, 0.3, 0.5])
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 42, 3407])
    p.add_argument(
        "--models",
        nargs="+",
        default=["smile", *BASELINES.keys()],
        help="Subset of: smile ists_plm wavegnn atenet misstm",
    )
    p.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--sweep-root", type=Path, default=ROOT / "export" / "missing_sweep")
    p.add_argument("--limit-test", type=int, default=0,
                   help="Cap test samples for a fast plumbing check (0 = full test set).")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def child_env(gpu: str, ratio: float, seed: int) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["SMILE_EVAL_SENSOR_DROP"] = repr(float(ratio))
    env["SMILE_EVAL_SENSOR_DROP_SEED"] = str(seed)
    return env


def run(cmd, env, log_path: Path, dry_run: bool) -> int:
    rendered = subprocess.list2cmdline([str(c) for c in cmd])
    if dry_run:
        print(f"    DRY: SMILE_EVAL_SENSOR_DROP={env['SMILE_EVAL_SENSOR_DROP']} {rendered}")
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [str(c) for c in cmd], cwd=ROOT, env=env,
            stdout=log, stderr=subprocess.STDOUT, text=True, check=False,
        )
    return completed.returncode


def read_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    block = data.get("test") or data.get("test_metrics") or {}
    if "auroc" not in block:
        return None
    return {"auroc": float(block["auroc"]), "auprc": float(block.get("auprc", float("nan")))}


def eval_baseline(model, ratio, seed, args, out_dir: Path) -> dict | None:
    script, extra = BASELINES[model]
    limit = ["--limit-test", str(args.limit_test)] if args.limit_test else []
    cmd = [
        args.python, ROOT / "experiments" / "recent_baselines" / script,
        "--dataset", args.dataset, "--seed", str(seed),
        "--eval-only", "--eval-output-dir", out_dir, *extra, *limit,
    ]
    log = args.sweep_root / "logs" / f"{model}_r{ratio}_s{seed}.log"
    rc = run(cmd, child_env(args.gpu, ratio, seed), log, args.dry_run)
    if rc != 0:
        print(f"    [FAIL rc={rc}] {model} r={ratio} s={seed}  (log: {log})")
        return None
    return read_metrics(out_dir / "eval_results.json")


def eval_smile(ratio, seed, args) -> dict | None:
    # run_all_experiments writes eval_results.json into the checkpoint dir; back it
    # up and restore so the canonical (clean) result is never clobbered by a drop run.
    ckpt_dir = ROOT / "export" / args.dataset / SMILE_MODEL / f"seed_{seed}"
    result_json = ckpt_dir / "eval_results.json"
    backup = ckpt_dir / "eval_results.json.sweepbak"
    if result_json.exists():
        shutil.copy2(result_json, backup)
    cmd = [
        args.python, ROOT / "run_all_experiments.py",
        "--models", SMILE_MODEL, "--datasets", args.dataset,
        "--seeds", str(seed), "--eval-only",
    ]
    log = args.sweep_root / "logs" / f"smile_r{ratio}_s{seed}.log"
    rc = run(cmd, child_env(args.gpu, ratio, seed), log, args.dry_run)
    metrics = read_metrics(result_json) if rc == 0 else None
    if rc != 0:
        print(f"    [FAIL rc={rc}] smile r={ratio} s={seed}  (log: {log})")
    # Restore tree to its prior state.
    if not args.dry_run:
        if backup.exists():
            shutil.move(backup, result_json)
        elif result_json.exists():
            result_json.unlink()
    return metrics


def aggregate(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    mean = statistics.mean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def main() -> None:
    raise SystemExit(
        "run_missing_sweep.py has been retired because its environment-variable "
        "corruption path was not publication-safe. Use "
        "experiments/run_sensor_robustness.py; the existing export/missing_sweep "
        "artifacts remain available for exploratory provenance."
    )
    args = parse_args()
    results = {}  # results[model][ratio] = {"auroc": [...], "auprc": [...]}
    for model in args.models:
        results[model] = {r: {"auroc": [], "auprc": []} for r in args.ratios}
        for ratio in args.ratios:
            for seed in args.seeds:
                print(f"[{model}] ratio={ratio} seed={seed} ...", flush=True)
                if model == "smile":
                    m = eval_smile(ratio, seed, args)
                else:
                    out_dir = args.sweep_root / model / f"ratio_{ratio}" / f"seed_{seed}"
                    m = eval_baseline(model, ratio, seed, args, out_dir)
                if m:
                    results[model][ratio]["auroc"].append(m["auroc"])
                    results[model][ratio]["auprc"].append(m["auprc"])

    summary = {"dataset": args.dataset, "seeds": args.seeds, "ratios": args.ratios, "models": {}}
    for model in args.models:
        summary["models"][model] = {}
        for ratio in args.ratios:
            au_m, au_s = aggregate(results[model][ratio]["auroc"])
            pr_m, pr_s = aggregate(results[model][ratio]["auprc"])
            summary["models"][model][str(ratio)] = {
                "auroc_mean": au_m, "auroc_std": au_s,
                "auprc_mean": pr_m, "auprc_std": pr_s,
                "n": len(results[model][ratio]["auroc"]),
            }

    if not args.dry_run:
        args.sweep_root.mkdir(parents=True, exist_ok=True)
        out = args.sweep_root / f"{args.dataset}_sweep.json"
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote {out}")

    # Console table (AUPRC is the selection metric for these imbalanced tasks).
    print("\n=== AUPRC mean +/- std by missing ratio ===")
    header = "model".ljust(12) + "".join(f"r={r:<10}" for r in args.ratios)
    print(header)
    for model in args.models:
        row = model.ljust(12)
        for ratio in args.ratios:
            cell = summary["models"][model][str(ratio)]
            row += (f"{cell['auprc_mean']:.3f}+/-{cell['auprc_std']:.3f}"
                    if cell["auprc_mean"] is not None else "  --  ").ljust(12)
        print(row)


if __name__ == "__main__":
    main()
