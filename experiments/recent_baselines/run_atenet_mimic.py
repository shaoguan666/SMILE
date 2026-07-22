"""Run the official ATENet architecture on SMILE's MIMIC-III cohorts.

The upstream repository supports P12/P19/PAM only. This adapter preserves the
ATENet model and consistency losses while reusing SMILE's patient splits,
normalization, masks, and threshold-free AUROC/AUPRC evaluation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "external" / "atenet"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataloader import collate_fn  # noqa: E402
from data.challenge2012 import load_challenge_2012  # noqa: E402
from data.challenge2019 import load_challenge_2019  # noqa: E402
from data.mimiciii import (  # noqa: E402
    load_mimic_iii_decompensation,
    load_mimic_iii_mortality,
)
from experiments.recent_baselines.baseline_utils import (  # noqa: E402
    add_sensor_robustness_args,
    binary_prediction_arrays,
    maybe_run_sensor_grid,
)


def load_upstream_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


atenet_models = load_upstream_module("smile_external_atenet_models", UPSTREAM / "models.py")
atenet_utils = load_upstream_module("smile_external_atenet_utils", UPSTREAM / "utils.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("mimic_mortality", "mimic_decompensation", "c12", "c19"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--embed-time", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "export" / "recent_baselines",
    )
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; load best_auprc.pt and evaluate only.")
    parser.add_argument("--eval-output-dir", type=Path, default=None,
                        help="Write eval result here instead of the checkpoint dir.")
    add_sensor_robustness_args(parser)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_limit(dataset, limit: int):
    if limit <= 0 or limit >= len(dataset):
        return dataset
    subset = torch.utils.data.Subset(dataset, range(limit))
    subset.feature_names = getattr(dataset, "feature_names", None)
    return subset


def labels_from_dataset(dataset) -> np.ndarray:
    if isinstance(dataset, torch.utils.data.Subset):
        return np.asarray([int(dataset.dataset.data[i]["labels"]) for i in dataset.indices])
    return np.asarray([int(sample["labels"]) for sample in dataset.data])


def make_train_sampler(dataset) -> WeightedRandomSampler:
    labels = labels_from_dataset(dataset)
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    class_weights = 1.0 / np.maximum(counts, 1.0)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )


def prepare_batch(batch: dict[str, torch.Tensor], device: torch.device):
    values = batch["x"].float().to(device, non_blocking=True)
    mask = batch["mask"].float().to(device, non_blocking=True)
    values = values * mask
    model_input = torch.cat([values, mask], dim=-1)
    lengths = batch["lens"].float().clamp_min(1).to(device)
    time = batch["time"].float().to(device, non_blocking=True) / lengths.unsqueeze(1)
    labels = batch["labels"].long().view(-1).to(device, non_blocking=True)
    return model_input, time, labels


@torch.no_grad()
def evaluate(model, loader, device: torch.device, *, return_predictions=False):
    model.eval()
    labels_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    for batch in loader:
        model_input, time, labels = prepare_batch(batch, device)
        _, _, logits, _, _, _ = model(model_input, time, None)
        scores = torch.softmax(logits, dim=-1)[:, 1]
        labels_all.append(labels.cpu().numpy())
        scores_all.append(scores.cpu().numpy())
    labels_np = np.concatenate(labels_all)
    scores_np = np.concatenate(scores_all)
    if return_predictions:
        return binary_prediction_arrays([labels_np], [scores_np])
    return {
        "auroc": float(roc_auc_score(labels_np, scores_np)),
        "auprc": float(average_precision_score(labels_np, scores_np)),
        "n": int(labels_np.size),
        "positive_rate": float(labels_np.mean()),
    }


def upstream_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The upstream ATENet implementation requires CUDA.")
    device = torch.device("cuda")
    set_seed(args.seed)

    loader_fn = {
        "mimic_mortality": load_mimic_iii_mortality,
        "mimic_decompensation": load_mimic_iii_decompensation,
        "c12": load_challenge_2012,
        "c19": load_challenge_2019,
    }[args.dataset]
    train_ds, val_ds, test_ds = loader_fn(split_seed=args.split_seed)
    train_ds = maybe_limit(train_ds, args.limit_train)
    val_ds = maybe_limit(val_ds, args.limit_val)
    test_ds = maybe_limit(test_ds, args.limit_test)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate_fn,
        "pin_memory": True,
    }
    train_loader = DataLoader(train_ds, sampler=make_train_sampler(train_ds), **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    input_dim = len(getattr(train_ds, "feature_names", []) or [])
    if input_dim == 0:
        sample = train_ds[0]
        input_dim = int(np.asarray(sample["x"]).shape[-1])
    model = atenet_models.ATE(
        ori_input_dim=input_dim,
        static_dim=0,
        nhidden=args.hidden,
        embed_time=args.embed_time,
        num_heads=args.num_heads,
        learn_emb=True,
        device=str(device),
        n_classes=2,
        static=False,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    bce_criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=2
    )
    # torch.amp.GradScaler was introduced after the target server's PyTorch.
    # The CUDA namespace works on both the older and current supported builds.
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    output_dir = args.output_root / args.dataset / "atenet" / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_auprc.pt"
    best_auprc = -float("inf")
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, (0 if args.eval_only else args.epochs) + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        skipped_nonfinite = 0
        for batch in train_loader:
            model_input, time, labels = prepare_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                out, out_masked, logits, sx, sout, _ = model(model_input, time, None)
                classification = criterion(logits, labels)
                temporal = atenet_utils.temporal_contrastive_loss(out, out_masked)
                instance = atenet_utils.instance_contrastive_loss(out, out_masked)
                consistency = temporal + instance

            # BCELoss is intentionally excluded from autocast by PyTorch because
            # its fp16 backward pass can overflow. Keep this small auxiliary
            # objective in fp32 while retaining AMP for the encoder.
            with torch.cuda.amp.autocast(enabled=False):
                intervariable = bce_criterion(sout.float(), sx.float())
            loss = classification + args.alpha * consistency + args.beta * intervariable
            if not torch.isfinite(loss):
                # AMP's GradScaler protects the parameters, but recording and
                # explicitly skipping the batch keeps the run artifact valid JSON
                # and makes numerical instability auditable.
                optimizer.zero_grad(set_to_none=True)
                skipped_nonfinite += 1
                continue
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * labels.numel()
            seen += labels.numel()

        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["auprc"])
        epoch_record = {
            "epoch": epoch,
            "train_loss": loss_sum / max(seen, 1),
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
            "skipped_nonfinite_batches": skipped_nonfinite,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record), flush=True)

        if val_metrics["auprc"] > best_auprc:
            best_auprc = val_metrics["auprc"]
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    if maybe_run_sensor_grid(
        args,
        evaluator=lambda loader: evaluate(
            model, loader, device, return_predictions=True
        ),
        test_dataset=test_ds,
        base_collate=collate_fn,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=output_dir,
        model_name="atenet",
        checkpoint_path=checkpoint_path,
    ):
        return
    test_metrics = evaluate(model, test_loader, device)
    result = {
        "model": "ATENet",
        "dataset": args.dataset,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "upstream_commit": upstream_commit(),
        "upstream_url": "https://github.com/shlee-labs/ATENet",
        "upstream_local_patches": [
            "device-aware random masks instead of hard-coded .cuda()",
            "dtype-aware attention-mask sentinel for AMP compatibility",
        ],
        "selection_metric": "validation AUPRC",
        "test": test_metrics,
        "best_validation_auprc": best_auprc,
        "history": history,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    result_dir = args.eval_output_dir or output_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / "eval_results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result["test"], indent=2), flush=True)


if __name__ == "__main__":
    main()
