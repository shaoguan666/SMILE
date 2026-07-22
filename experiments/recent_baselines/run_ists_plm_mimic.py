"""Run the official series-based ISTS-PLM on SMILE's MIMIC cohorts."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "external" / "ists-plm"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataloader import collate_fn  # noqa: E402
from experiments.recent_baselines.baseline_utils import (  # noqa: E402
    add_sensor_robustness_args,
    binary_metrics,
    binary_prediction_arrays,
    json_ready_config,
    load_mimic_splits,
    make_loaders,
    maybe_run_sensor_grid,
    set_seed,
    upstream_commit,
    write_result,
)


def load_upstream_transformers_extensions() -> None:
    """Register upstream wope modules without modifying site-packages.

    The official repository asks users to copy these files into transformers.
    Loading them under their expected fully qualified names is equivalent and
    keeps the Python installation untouched.  Python 3.13 needs a newer
    tokenizers wheel, but ISTS-PLM never invokes tokenization; report the
    upstream-compatible version only while transformers performs its import
    metadata check.
    """

    original_version = importlib.metadata.version

    def compatible_version(package: str) -> str:
        if package == "tokenizers":
            return "0.13.3"
        return original_version(package)

    importlib.metadata.version = compatible_version
    try:
        import transformers  # noqa: F401
    finally:
        importlib.metadata.version = original_version

    modules = {
        "transformers.models.gpt2.modeling_gpt2_wope": UPSTREAM
        / "model_wope"
        / "modeling_gpt2_wope.py",
        "transformers.models.bert.modeling_bert_wope": UPSTREAM
        / "model_wope"
        / "modeling_bert_wope.py",
    }
    for name, path in modules.items():
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {name} from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--n-te-plm-layers", type=int, default=6)
    parser.add_argument("--n-st-plm-layers", type=int, default=6)
    parser.add_argument("--te-model", choices=("gpt", "bert"), default="gpt")
    parser.add_argument("--st-model", choices=("gpt", "bert"), default="bert")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "export" / "recent_baselines"
    )
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; load best_auprc.pt and evaluate only.")
    parser.add_argument("--eval-output-dir", type=Path, default=None,
                        help="Write eval result here instead of the checkpoint dir.")
    add_sensor_robustness_args(parser)
    return parser.parse_args()


def prepare_batch(batch, device: torch.device):
    values = torch.nan_to_num(batch["x"].float()).to(device, non_blocking=True)
    observed = batch["mask"].float().to(device, non_blocking=True)
    values = values * observed
    timestamps = batch["time"].float().to(device, non_blocking=True)
    timestamps = timestamps.unsqueeze(-1).expand_as(values)
    labels = batch["labels"].long().view(-1).to(device, non_blocking=True)
    return timestamps, values, observed, labels


@torch.no_grad()
def evaluate(model, classifier, loader, device: torch.device, *, return_predictions=False):
    model.eval()
    classifier.eval()
    labels_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    for batch in loader:
        timestamps, values, observed, labels = prepare_batch(batch, device)
        encoded = model(timestamps, values, observed)
        logits = classifier(encoded)
        labels_all.append(labels.cpu().numpy())
        scores_all.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    if return_predictions:
        return binary_prediction_arrays(labels_all, scores_all)
    return binary_metrics(labels_all, scores_all)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("ISTS-PLM local runs require CUDA.")
    for relative in ("PLMs/gpt2/config.json", "PLMs/bert-base-uncased/config.json"):
        if not (UPSTREAM / relative).exists():
            raise FileNotFoundError(f"Missing ISTS-PLM pretrained model file: {relative}")

    load_upstream_transformers_extensions()
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from models.plm4ts import Classifier, ists_plm

    device = torch.device("cuda")
    set_seed(args.seed)
    splits = load_mimic_splits(
        args.dataset,
        args.split_seed,
        (args.limit_train, args.limit_val, args.limit_test),
    )
    train_loader, val_loader, test_loader = make_loaders(
        splits, args.batch_size, collate_fn, args.num_workers
    )
    sample = splits[0][0]
    _, input_dim = np.asarray(sample["x"]).shape
    options = SimpleNamespace(
        input_dim=int(input_dim),
        d_model=args.d_model,
        device=device,
        dropout=args.dropout,
        te_model=args.te_model,
        st_model=args.st_model,
        n_te_plmlayer=args.n_te_plm_layers,
        n_st_plmlayer=args.n_st_plm_layers,
        semi_freeze=True,
    )
    with working_directory(UPSTREAM):
        model = ists_plm(options)
    classifier = Classifier(args.d_model * input_dim, 2)
    model = model.to(device)
    classifier = classifier.to(device)
    trainable = [parameter for parameter in list(model.parameters()) + list(classifier.parameters()) if parameter.requires_grad]
    optimizer = torch.optim.Adam(
        trainable,
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-5,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    # Keep compatibility with the target host's pre-torch.amp PyTorch build.
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    output_dir = args.output_root / args.dataset / "ists_plm" / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_auprc.pt"
    best_auprc = -float("inf")
    stale_epochs = 0
    history = []

    for epoch in range(1, (0 if args.eval_only else args.epochs) + 1):
        model.train()
        classifier.train()
        loss_sum = 0.0
        seen = 0
        for batch in train_loader:
            timestamps, values, observed, labels = prepare_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                encoded = model(timestamps, values, observed)
                logits = classifier(encoded)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * labels.numel()
            seen += labels.numel()
        scheduler.step()

        val = evaluate(model, classifier, val_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / max(seen, 1),
            "val_auroc": val["auroc"],
            "val_auprc": val["auprc"],
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if val["auprc"] > best_auprc:
            best_auprc = val["auprc"]
            stale_epochs = 0
            torch.save(
                {"model": model.state_dict(), "classifier": classifier.state_dict()},
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    classifier.load_state_dict(checkpoint["classifier"])
    if maybe_run_sensor_grid(
        args,
        evaluator=lambda loader: evaluate(
            model, classifier, loader, device, return_predictions=True
        ),
        test_dataset=splits[2],
        base_collate=collate_fn,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=output_dir,
        model_name="ists_plm",
        checkpoint_path=checkpoint_path,
    ):
        return
    test = evaluate(model, classifier, test_loader, device)
    result = {
        "model": "ISTS-PLM",
        "dataset": args.dataset,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "upstream_commit": upstream_commit(UPSTREAM),
        "upstream_url": "https://github.com/usail-hkust/ISTS-PLM",
        "adapter_notes": [
            "official series-based ISTS-PLM architecture and pretrained weights",
            "upstream wope modules loaded at runtime without modifying transformers",
            "SMILE split, normalization, observation mask, and timestamps",
            "checkpoint selected by validation AUPRC",
        ],
        "selection_metric": "validation AUPRC",
        "best_validation_auprc": best_auprc,
        "test": test,
        "history": history,
        "config": json_ready_config(args),
    }
    write_result(result, args.eval_output_dir or output_dir)
    print(json.dumps(test, indent=2), flush=True)


if __name__ == "__main__":
    main()
