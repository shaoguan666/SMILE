"""Diagnose SMILE checkpoint branch usage.

The script loads a SMART/SMILE checkpoint, reports branch-specific parameter
norms, runs a small activation pass, and estimates whether density, CoMiss
bias, FiLM, and dual-head mask branches are effectively near zero.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.challenge2012 import load_challenge_2012  # noqa: E402
from data.challenge2019 import load_challenge_2019  # noqa: E402
from data.dataloader import collate_fn  # noqa: E402
from data.mimiciii import (  # noqa: E402
    load_mimic_iii_decompensation,
    load_mimic_iii_lengthofstay,
    load_mimic_iii_mortality,
    load_mimic_iii_phenotyping,
)
from models.smart import (  # noqa: E402
    Classifier,
    Encoder,
    MNAREncoder,
    SMILEEncoder,
    SMILEFiLMEncoder,
    SMILELeanEncoder,
    SMILEv2Encoder,
    SMILEv2FiLMEncoder,
    TimeFiLMEncoder,
)
from utils.variable_order import get_variable_order  # noqa: E402


DATASET_SPECS = {
    "c12": {"input_dim": 37, "demo_dim": 4, "num_class": 2, "max_len": 48},
    "c19": {"input_dim": 34, "demo_dim": 5, "num_class": 2, "max_len": 60},
    "mimic_mortality": {"input_dim": 17, "demo_dim": 0, "num_class": 2, "max_len": 48},
    "mimic_phenotyping": {"input_dim": 17, "demo_dim": 0, "num_class": 25, "max_len": 60},
    "mimic_decompensation": {"input_dim": 17, "demo_dim": 0, "num_class": 2, "max_len": 24},
    "mimic_lengthofstay": {"input_dim": 17, "demo_dim": 0, "num_class": 10, "max_len": 24},
}

ENCODER_CLASSES = {
    "smart": Encoder,
    "smart-film": TimeFiLMEncoder,
    "smart-mnar": MNAREncoder,
    "smart-smile": SMILEEncoder,
    "smart-smile-film": SMILEFiLMEncoder,
    "smart-smile-v2": SMILEv2Encoder,
    "smart-smile-v2-film": SMILEv2FiLMEncoder,
    "smart-smile-lean": SMILELeanEncoder,
    "smart-smile-lean-samepretrain": SMILELeanEncoder,
}

ABLATION_FLAGS = {
    "abl_no_density": "no-density",
    "abl_no_mnar_bias": "no-mnar-bias",
    "abl_no_film": "no-film",
    "abl_no_time_mnar": "no-time-mnar",
    "abl_no_time_pe": "no-time-pe",
    "abl_no_cross_attn": "no-cross-attn",
    "abl_no_mnar_cls": "no-mnar-cls",
}


class RunningTensorStats:
    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.abs_sum = 0.0
        self.max_abs = 0.0
        self.batches = 0
        self.shapes: set[tuple[int, ...]] = set()

    def update(self, value: torch.Tensor | None) -> None:
        if value is None or not torch.is_tensor(value):
            return
        detached = value.detach().float()
        if detached.numel() == 0:
            return
        flat = detached.reshape(-1)
        self.count += int(flat.numel())
        self.sum += float(flat.sum().item())
        self.sumsq += float((flat * flat).sum().item())
        abs_flat = flat.abs()
        self.abs_sum += float(abs_flat.sum().item())
        self.max_abs = max(self.max_abs, float(abs_flat.max().item()))
        self.batches += 1
        self.shapes.add(tuple(detached.shape))

    def as_dict(self) -> dict[str, Any]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": None,
                "var": None,
                "mean_abs": None,
                "max_abs": None,
                "batches": self.batches,
                "shapes": [],
            }
        mean = self.sum / self.count
        var = max(0.0, self.sumsq / self.count - mean * mean)
        return {
            "count": self.count,
            "mean": mean,
            "var": var,
            "mean_abs": self.abs_sum / self.count,
            "max_abs": self.max_abs,
            "batches": self.batches,
            "shapes": [list(shape) for shape in sorted(self.shapes)],
        }


def strip_module_prefix(state: dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    stripped: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state.items():
        new_key = key[len("module.") :] if key.startswith("module.") else key
        stripped[new_key] = value
    return stripped


def load_checkpoint(path: Path, map_location: str) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected dict checkpoint, got {type(checkpoint)!r}")
    return checkpoint


def select_state_dict(checkpoint: dict[str, Any], key: str) -> OrderedDict[str, torch.Tensor] | None:
    candidate = checkpoint.get(key)
    if isinstance(candidate, dict) and all(torch.is_tensor(v) for v in candidate.values()):
        return strip_module_prefix(candidate)
    if key == "encoder" and all(torch.is_tensor(v) for v in checkpoint.values()):
        return strip_module_prefix(checkpoint)  # direct state_dict
    return None


def infer_from_path(path: Path) -> dict[str, str | int | None]:
    parts = [part.lower() for part in path.parts]
    dataset = None
    for name in sorted(DATASET_SPECS, key=len, reverse=True):
        if name in parts:
            dataset = name
            break
    variant = None
    if dataset is not None:
        index = parts.index(dataset)
        if index + 1 < len(parts):
            variant = parts[index + 1]
    if variant is None:
        for part in reversed(parts):
            if part.startswith("smart"):
                variant = part
                break
    seed = None
    for part in parts:
        match = re.fullmatch(r"seed_(\d+)", part)
        if match:
            seed = int(match.group(1))
            break
    return {"dataset": dataset, "variant": variant, "seed": seed}


def base_variant(variant: str) -> str:
    for candidate in sorted(ENCODER_CLASSES, key=len, reverse=True):
        if variant == candidate or variant.startswith(candidate + "-"):
            return candidate
    return "smart-smile-lean" if "lean" in variant else "smart"


def infer_model_args(
    dataset: str,
    variant: str,
    encoder_state: dict[str, torch.Tensor],
    classifier_state: dict[str, torch.Tensor] | None,
    overrides: argparse.Namespace,
) -> SimpleNamespace:
    spec = dict(DATASET_SPECS[dataset])
    query = encoder_state.get("query")
    if torch.is_tensor(query) and query.ndim == 3:
        spec["input_dim"] = int(query.shape[0])
        spec["d_model"] = int(query.shape[2])
    else:
        spec["d_model"] = overrides.d_model

    block_ids = []
    for key in encoder_state:
        match = re.match(r"blocks\.(\d+)\.", key)
        if match:
            block_ids.append(int(match.group(1)))
    spec["e_layers"] = max(block_ids) + 1 if block_ids else overrides.e_layers

    n_heads = overrides.n_heads
    for key, value in encoder_state.items():
        if key.endswith("mnar_bias_scale") and value.ndim >= 1:
            n_heads = int(value.shape[0])
            break
    spec["n_heads"] = n_heads

    time_dim = overrides.time_dim
    time_w = encoder_state.get("time_encoder.w")
    if torch.is_tensor(time_w) and time_w.ndim == 1:
        time_dim = int(time_w.numel() * 2)
    else:
        for key, value in encoder_state.items():
            if key.endswith("film_gen.weight") and value.ndim == 2:
                time_dim = int(value.shape[1])
                break
    spec["time_dim"] = time_dim

    if classifier_state is not None:
        for key in ("out.weight", "module.out.weight"):
            value = classifier_state.get(key)
            if torch.is_tensor(value) and value.ndim == 2:
                spec["num_class"] = int(value.shape[0])
                break

    args = SimpleNamespace(
        dataset=dataset,
        model_name=variant,
        d_model=spec["d_model"],
        input_dim=spec["input_dim"],
        demo_dim=spec["demo_dim"],
        num_class=spec["num_class"],
        max_len=spec["max_len"],
        e_layers=spec["e_layers"],
        n_heads=spec["n_heads"],
        time_dim=spec["time_dim"],
        dropout=overrides.dropout,
        obs_density_window=overrides.obs_density_window,
        los_task=overrides.los_task,
        los_label_unit=overrides.los_label_unit,
    )

    variant_lower = variant.lower()
    for attr, token in ABLATION_FLAGS.items():
        setattr(args, attr, token in variant_lower)
    return args


def attach_variable_order(args: SimpleNamespace, device: torch.device) -> None:
    registry_name = args.dataset.split("_")[0] if args.dataset.startswith("mimic") else args.dataset
    var_order_idx, inv_order_idx = get_variable_order(registry_name)
    args.var_order_idx = var_order_idx.to(device)
    args.inv_order_idx = inv_order_idx.to(device)


def build_encoder(args: SimpleNamespace, variant: str) -> torch.nn.Module:
    cls = ENCODER_CLASSES.get(base_variant(variant), SMILELeanEncoder)
    return cls(args)


def build_classifier(
    args: SimpleNamespace,
    variant: str,
    classifier_state: dict[str, torch.Tensor] | None,
) -> torch.nn.Module | None:
    if classifier_state is None:
        return None
    return Classifier(args)


def load_matching_state(
    module: torch.nn.Module,
    state: dict[str, torch.Tensor],
    component: str,
) -> dict[str, Any]:
    target = module.state_dict()
    loadable = OrderedDict()
    mismatched = []
    unexpected = []
    for key, value in state.items():
        if key not in target:
            unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(target[key].shape):
            mismatched.append({"key": key, "checkpoint": list(value.shape), "model": list(target[key].shape)})
            continue
        loadable[key] = value
    incompatible = module.load_state_dict(loadable, strict=False)
    report = {
        "component": component,
        "loaded_tensors": len(loadable),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": unexpected,
        "mismatched_shapes": mismatched,
    }
    if len(loadable) == 0:
        raise RuntimeError(f"No tensors from checkpoint matched {component}; check --variant.")
    return report


def load_dataset(dataset: str, split_seed: int, los_task: str, los_label_unit: str):
    if dataset == "c12":
        return load_challenge_2012()
    if dataset == "c19":
        return load_challenge_2019()
    if dataset == "mimic_mortality":
        return load_mimic_iii_mortality(split_seed=split_seed)
    if dataset == "mimic_phenotyping":
        return load_mimic_iii_phenotyping(split_seed=split_seed)
    if dataset == "mimic_decompensation":
        return load_mimic_iii_decompensation(split_seed=split_seed)
    if dataset == "mimic_lengthofstay":
        return load_mimic_iii_lengthofstay(
            split_seed=split_seed,
            task=los_task,
            label_unit=los_label_unit,
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def tensor_group_stats(named_tensors: list[tuple[str, torch.Tensor]]) -> dict[str, Any]:
    tensors = [tensor.detach().float().reshape(-1).cpu() for _, tensor in named_tensors if tensor.numel() > 0]
    if not tensors:
        return {"present": False, "tensors": 0, "params": 0, "l2": None, "mean_abs": None, "max_abs": None}
    flat = torch.cat(tensors)
    return {
        "present": True,
        "tensors": len(tensors),
        "params": int(flat.numel()),
        "l2": float(torch.linalg.vector_norm(flat).item()),
        "mean_abs": float(flat.abs().mean().item()),
        "max_abs": float(flat.abs().max().item()),
        "keys": [name for name, _ in named_tensors],
    }


def is_parameter_like_key(name: str) -> bool:
    return not (
        name.endswith("pos_table")
        or name.endswith("var_order_idx")
        or name.endswith("inv_order_idx")
        or name.endswith("num_batches_tracked")
    )


def collect_parameter_groups(
    encoder_state: dict[str, torch.Tensor],
    classifier_state: dict[str, torch.Tensor] | None,
    args: SimpleNamespace,
) -> dict[str, dict[str, Any]]:
    enc_params = [
        (name, tensor)
        for name, tensor in encoder_state.items()
        if torch.is_tensor(tensor) and is_parameter_like_key(name)
    ]
    groups: dict[str, dict[str, Any]] = {}

    groups["encoder_total"] = tensor_group_stats(enc_params)
    groups["density_input_columns"] = tensor_group_stats(
        [
            (name + "[:,2:]", param[:, 2:])
            for name, param in enc_params
            if name == "embedder.embed.0.weight" and param.ndim == 2 and param.shape[1] > 2
        ]
    )
    groups["density_policy_or_projection"] = tensor_group_stats(
        [
            (name, param)
            for name, param in enc_params
            if name.startswith("obs_density_embedder.")
            or name.startswith("cls_density_proj.")
            or name.startswith("embedder.recency_gate.")
            or name.startswith("embedder.embed_observed")
            or name.startswith("embedder.embed_recent_missing")
            or name.startswith("embedder.embed_long_missing")
        ]
    )
    groups["comiss_bias_scale"] = tensor_group_stats(
        [(name, param) for name, param in enc_params if name.endswith("mnar_bias_scale")]
    )
    groups["comiss_time_gate"] = tensor_group_stats(
        [
            (name, param)
            for name, param in enc_params
            if "mnar_time_proj" in name or "var_time_proj" in name
        ]
    )
    groups["film_gen"] = tensor_group_stats(
        [(name, param) for name, param in enc_params if "film_gen" in name]
    )
    groups["time_pe"] = tensor_group_stats(
        [(name, param) for name, param in enc_params if "time_pe_proj" in name]
    )

    if classifier_state is None:
        groups["classifier_total"] = {"present": False, "tensors": 0, "params": 0, "l2": None, "mean_abs": None, "max_abs": None}
        groups["dual_head_mask_proj"] = dict(groups["classifier_total"])
        groups["dual_head_out_mask_slice"] = dict(groups["classifier_total"])
    else:
        clf_params = [
            (name, tensor)
            for name, tensor in classifier_state.items()
            if torch.is_tensor(tensor) and is_parameter_like_key(name)
        ]
        groups["classifier_total"] = tensor_group_stats(clf_params)
        groups["dual_head_mask_proj"] = tensor_group_stats(
            [(name, param) for name, param in clf_params if name.startswith("mask_proj.")]
        )
        mask_slice = []
        out_weight = classifier_state.get("out.weight")
        if torch.is_tensor(out_weight):
            split = args.input_dim * args.d_model
            weight = out_weight
            if weight.ndim == 2 and weight.shape[1] > split:
                mask_slice.append(("out.weight[:, mask_slice]", weight[:, split:]))
        groups["dual_head_out_mask_slice"] = tensor_group_stats(mask_slice)
        if mask_slice and torch.is_tensor(out_weight):
            split = args.input_dim * args.d_model
            weight = out_weight.detach().float()
            cls_norm = float(torch.linalg.vector_norm(weight[:, :split]).item())
            mask_norm = float(torch.linalg.vector_norm(weight[:, split:]).item())
            groups["dual_head_out_mask_slice"]["l2_ratio_vs_out"] = mask_norm / (cls_norm + mask_norm + 1e-12)
    return groups


def compute_density(original_mask: torch.Tensor, window_size: int) -> torch.Tensor:
    batch, steps, variables = original_mask.shape
    mask = original_mask.float().permute(0, 2, 1).reshape(batch * variables, 1, steps)
    density = F.avg_pool1d(mask, kernel_size=window_size, stride=1, padding=window_size // 2)
    return density.reshape(batch, variables, steps).permute(0, 2, 1)


def compute_mask_full(mask: torch.Tensor) -> torch.Tensor:
    cls_mask = torch.ones(mask.shape[0], 1, mask.shape[-1], device=mask.device, dtype=mask.dtype)
    return torch.cat((cls_mask, mask), dim=1).transpose(1, 2).float()


def compute_time_enc(encoder: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
    if not hasattr(encoder, "time_encoder") or "time" not in batch:
        return None
    time = batch["time"]
    cls_time = torch.zeros(time.shape[0], 1, device=time.device, dtype=time.dtype)
    return encoder.time_encoder(torch.cat([cls_time, time], dim=1))


def co_miss_bias_stats(
    encoder: torch.nn.Module,
    original_mask: torch.Tensor,
    mask_full: torch.Tensor,
    time_enc: torch.Tensor | None,
    stats: defaultdict[str, RunningTensorStats],
) -> None:
    if not hasattr(encoder, "mnar_cooccur_encoder"):
        return
    try:
        cooccur = encoder.mnar_cooccur_encoder(original_mask)
    except Exception:
        return
    stats["comiss.cooccur_input"].update(cooccur)
    for index, block in enumerate(getattr(encoder, "blocks", [])):
        var_block = getattr(block, "var_att_block", None)
        attn_var = getattr(var_block, "attn_var", None)
        scale = getattr(attn_var, "mnar_bias_scale", None)
        if scale is None:
            continue
        if cooccur.ndim == 3:
            if scale.ndim == 1:
                bias = cooccur.unsqueeze(1) * scale.view(1, -1, 1, 1)
            else:
                bias = cooccur.unsqueeze(1) * scale.unsqueeze(0)
            if hasattr(var_block, "mnar_time_proj") and time_enc is not None:
                decay = torch.sigmoid(var_block.mnar_time_proj(time_enc.mean(dim=1)))
                bias = bias * decay.view(decay.shape[0], decay.shape[1], 1, 1)
        elif cooccur.ndim == 4:
            bias = cooccur.unsqueeze(1) * scale.view(1, -1, 1, 1, 1)
            if hasattr(var_block, "compute_var_time_decay") and time_enc is not None:
                decay = var_block.compute_var_time_decay(time_enc, mask_full)
                bias = bias * decay.unsqueeze(2)
        else:
            continue
        stats[f"comiss.bias_logits.block_{index}"].update(bias)


def dual_head_mask_stats(
    classifier: torch.nn.Module | None,
    hidden: torch.Tensor,
    original_mask: torch.Tensor,
    args: SimpleNamespace,
    stats: defaultdict[str, RunningTensorStats],
) -> None:
    if classifier is None or not hasattr(classifier, "mask_proj") or not hasattr(classifier, "out"):
        return
    batch, variables, _, hidden_dim = hidden.shape
    steps = original_mask.shape[1]
    t1 = steps // 3
    t2 = 2 * steps // 3
    mask_summary = torch.cat(
        [
            original_mask[:, :t1].float().mean(dim=1),
            original_mask[:, t1:t2].float().mean(dim=1),
            original_mask[:, t2:].float().mean(dim=1),
        ],
        dim=-1,
    )
    mask_emb = classifier.mask_proj(mask_summary)
    stats["dual_head.mask_emb"].update(mask_emb)
    split = args.input_dim * args.d_model
    weight = classifier.out.weight
    if weight.shape[1] <= split:
        return
    mask_logits = mask_emb @ weight[:, split:].T
    cls_token = classifier.cls_mlp(hidden[:, :, 0]).reshape(batch, variables * hidden_dim)
    cls_logits = cls_token @ weight[:, :split].T
    stats["dual_head.mask_logits"].update(mask_logits)
    stats["dual_head.cls_logits"].update(cls_logits)
    stats["dual_head.mask_logit_fraction_abs"].update(
        mask_logits.abs() / (mask_logits.abs() + cls_logits.abs() + 1e-12)
    )


def register_activation_hooks(
    encoder: torch.nn.Module,
    classifier: torch.nn.Module | None,
    stats: defaultdict[str, RunningTensorStats],
) -> list[torch.utils.hooks.RemovableHandle]:
    handles = []

    def simple_hook(label: str) -> Callable:
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, (tuple, list)) else output
            if torch.is_tensor(value):
                stats[label].update(value)
        return hook

    def film_hook(label: str) -> Callable:
        def hook(_module, _inputs, output):
            if not torch.is_tensor(output):
                return
            gamma, beta = output.chunk(2, dim=-1)
            stats[label + ".gamma"].update(gamma)
            stats[label + ".beta"].update(beta)
        return hook

    for name, module in encoder.named_modules():
        if name == "time_encoder":
            handles.append(module.register_forward_hook(simple_hook("encoder.time_encoder.output")))
        elif name.endswith("mnar_cooccur_encoder"):
            handles.append(module.register_forward_hook(simple_hook("comiss.cooccur_encoder.output")))
        elif name.endswith("film_gen"):
            handles.append(module.register_forward_hook(film_hook("film." + name)))
        elif name.endswith("mnar_time_proj") or name.endswith("var_time_proj"):
            handles.append(module.register_forward_hook(simple_hook("comiss." + name + ".output")))
        elif name.endswith("time_pe_proj"):
            handles.append(module.register_forward_hook(simple_hook("time_pe.output")))

    if classifier is not None:
        for name, module in classifier.named_modules():
            if name == "mask_proj":
                handles.append(module.register_forward_hook(simple_hook("dual_head.mask_proj.output")))
            elif name == "cls_mlp":
                handles.append(module.register_forward_hook(simple_hook("dual_head.cls_mlp.output")))

    return handles


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def run_activation_pass(
    encoder: torch.nn.Module,
    classifier: torch.nn.Module | None,
    args: SimpleNamespace,
    split_dataset,
    device: torch.device,
    batch_size: int,
    num_batches: int,
) -> tuple[dict[str, dict[str, Any]], str | None]:
    stats: defaultdict[str, RunningTensorStats] = defaultdict(RunningTensorStats)
    loader = DataLoader(
        split_dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(split_dataset),
        collate_fn=collate_fn,
        num_workers=0,
    )
    handles = register_activation_hooks(encoder, classifier, stats)
    encoder.eval()
    if classifier is not None:
        classifier.eval()
    error = None
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if batch_index >= num_batches:
                    break
                batch = move_batch(batch, device)
                original_mask = batch["mask"].clone()
                if hasattr(encoder, "embedder") and not getattr(args, "abl_no_density", False):
                    try:
                        density = compute_density(original_mask, args.obs_density_window)
                        stats["density.input"].update(density)
                        if hasattr(encoder.embedder, "embed"):
                            zero = torch.zeros_like(density)
                            if "original_mask" in encoder.embedder.forward.__code__.co_varnames:
                                full = encoder.embedder(batch["x"], batch["mask"], density, original_mask)
                                baseline = encoder.embedder(batch["x"], batch["mask"], zero, original_mask)
                            else:
                                full = encoder.embedder(batch["x"], batch["mask"], density)
                                baseline = encoder.embedder(batch["x"], batch["mask"], zero)
                            stats["density.embed_delta"].update(full - baseline)
                    except Exception as exc:
                        error = f"density manual stats failed: {exc}"
                mask_full = compute_mask_full(batch["mask"])
                time_enc = compute_time_enc(encoder, batch)
                co_miss_bias_stats(encoder, original_mask, mask_full, time_enc, stats)
                hidden = encoder(**batch, original_mask=original_mask)
                stats["encoder.output"].update(hidden)
                if classifier is not None:
                    logits = classifier(hidden, original_mask=original_mask, **batch)
                    stats["classifier.logits"].update(logits)
                    dual_head_mask_stats(classifier, hidden, original_mask, args, stats)
    except Exception as exc:  # keep parameter diagnostics useful even if data pass fails
        error = f"activation pass failed: {type(exc).__name__}: {exc}"
    finally:
        for handle in handles:
            handle.remove()
    return {name: stat.as_dict() for name, stat in sorted(stats.items())}, error


def first_metric(stats: dict[str, dict[str, Any]], prefixes: tuple[str, ...], field: str = "mean_abs") -> float | None:
    values = []
    for name, payload in stats.items():
        if name.startswith(prefixes):
            value = payload.get(field)
            if value is not None:
                values.append(float(value))
    return max(values) if values else None


def group_metric(groups: dict[str, dict[str, Any]], names: tuple[str, ...], field: str = "mean_abs") -> float | None:
    values = []
    for name in names:
        value = groups.get(name, {}).get(field)
        if value is not None:
            values.append(float(value))
    return max(values) if values else None


def classify_value(value: float | None, present: bool, near_zero: float, small: float) -> str:
    if not present:
        return "absent"
    if value is None:
        return "not_measured"
    if value <= near_zero:
        return "near_zero"
    if value <= small:
        return "small_nonzero"
    return "active"


def build_verdicts(
    groups: dict[str, dict[str, Any]],
    activations: dict[str, dict[str, Any]],
    near_zero: float,
    small: float,
) -> list[dict[str, Any]]:
    density_present = groups["density_input_columns"]["present"] or groups["density_policy_or_projection"]["present"]
    density_metric = max(
        group_metric(groups, ("density_input_columns", "density_policy_or_projection")) or 0.0,
        first_metric(activations, ("density.embed_delta",)) or 0.0,
    )
    comiss_present = groups["comiss_bias_scale"]["present"]
    comiss_metric = max(
        group_metric(groups, ("comiss_bias_scale", "comiss_time_gate")) or 0.0,
        first_metric(activations, ("comiss.bias_logits",)) or 0.0,
    )
    film_present = groups["film_gen"]["present"]
    film_metric = max(
        group_metric(groups, ("film_gen",)) or 0.0,
        first_metric(activations, ("film.",)) or 0.0,
    )
    dual_present = groups["dual_head_mask_proj"]["present"] or groups["dual_head_out_mask_slice"]["present"]
    dual_metric = max(
        group_metric(groups, ("dual_head_mask_proj", "dual_head_out_mask_slice")) or 0.0,
        float(groups.get("dual_head_out_mask_slice", {}).get("l2_ratio_vs_out") or 0.0),
        first_metric(activations, ("dual_head.mask_logits", "dual_head.mask_logit_fraction_abs")) or 0.0,
    )
    return [
        {
            "component": "density",
            "present": density_present,
            "metric": density_metric if density_present else None,
            "verdict": classify_value(density_metric, density_present, near_zero, small),
            "evidence": "density input-column params + density.embed_delta activation",
        },
        {
            "component": "CoMiss bias",
            "present": comiss_present,
            "metric": comiss_metric if comiss_present else None,
            "verdict": classify_value(comiss_metric, comiss_present, near_zero, small),
            "evidence": "mnar_bias_scale/time gate params + CoMiss bias logits",
        },
        {
            "component": "FiLM",
            "present": film_present,
            "metric": film_metric if film_present else None,
            "verdict": classify_value(film_metric, film_present, near_zero, small),
            "evidence": "film_gen params + gamma/beta activations",
        },
        {
            "component": "dual-head mask branch",
            "present": dual_present,
            "metric": dual_metric if dual_present else None,
            "verdict": classify_value(dual_metric, dual_present, near_zero, small),
            "evidence": "mask_proj/out mask slice params + mask-logit contribution",
        },
    ]


def format_float(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, str)):
        return str(value)
    try:
        return f"{float(value):.4e}"
    except (TypeError, ValueError):
        return str(value)


def to_jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, set):
        return [to_jsonable(item) for item in sorted(value)]
    return value


def print_table(title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    print(f"\n{title}")
    if not rows:
        print("(empty)")
        return
    widths = {
        column: max(len(column), *(len(format_float(row.get(column))) for row in rows))
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(format_float(row.get(column)).ljust(widths[column]) for column in columns))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), default=None)
    parser.add_argument("--variant", default=None, help="Default: infer from checkpoint path.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-batches", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--skip-activations", action="store_true")
    parser.add_argument(
        "--allow-partial-activations",
        action="store_true",
        help="Run activation pass even when checkpoint tensors do not fully match the current source model.",
    )
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--near-zero", type=float, default=1e-6)
    parser.add_argument("--small", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--time-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--obs-density-window", type=int, default=5)
    parser.add_argument("--los-task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--los-label-unit", choices=["auto", "hours", "days"], default="auto")
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    inferred = infer_from_path(cli.checkpoint)
    dataset = cli.dataset or inferred["dataset"]
    variant = (cli.variant or inferred["variant"] or "smart-smile-lean").lower()
    split_seed = cli.split_seed if cli.split_seed is not None else (inferred["seed"] or 42)
    if dataset is None:
        raise ValueError("Unable to infer dataset from checkpoint path; pass --dataset.")
    if dataset not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset {dataset!r}; pass one of {sorted(DATASET_SPECS)}.")

    if cli.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cli.device)

    checkpoint = load_checkpoint(cli.checkpoint, "cpu")
    encoder_state = select_state_dict(checkpoint, "encoder")
    if encoder_state is None:
        raise KeyError("Checkpoint does not contain an encoder state_dict.")
    classifier_state = select_state_dict(checkpoint, "classifier")

    model_args = infer_model_args(dataset, variant, encoder_state, classifier_state, cli)
    attach_variable_order(model_args, device)
    encoder = build_encoder(model_args, variant).to(device)
    classifier = build_classifier(model_args, variant, classifier_state)
    if classifier is not None:
        classifier = classifier.to(device)

    load_reports = [load_matching_state(encoder, encoder_state, "encoder")]
    if classifier is not None and classifier_state is not None:
        load_reports.append(load_matching_state(classifier, classifier_state, "classifier"))

    parameter_groups = collect_parameter_groups(encoder_state, classifier_state, model_args)
    activation_stats = {}
    activation_error = None
    model_state_matches = not any(
        report["missing_keys"] or report["mismatched_shapes"]
        for report in load_reports
    )
    activation_ran = False
    if cli.skip_activations:
        activation_error = "activation pass skipped by --skip-activations"
    elif not model_state_matches and not cli.allow_partial_activations:
        activation_error = (
            "activation pass skipped: checkpoint tensors do not fully match the current "
            "source model; use --allow-partial-activations only for exploratory, non-final checks"
        )
    else:
        try:
            datasets = load_dataset(dataset, split_seed, cli.los_task, cli.los_label_unit)
            split_index = {"train": 0, "val": 1, "test": 2}[cli.split]
            activation_ran = True
            activation_stats, activation_error = run_activation_pass(
                encoder,
                classifier,
                model_args,
                datasets[split_index],
                device,
                cli.batch_size,
                cli.num_batches,
            )
        except Exception as exc:
            activation_error = f"dataset/activation setup failed: {type(exc).__name__}: {exc}"

    verdicts = build_verdicts(parameter_groups, activation_stats, cli.near_zero, cli.small)

    payload = {
        "checkpoint": str(cli.checkpoint),
        "epoch": checkpoint.get("epoch"),
        "dataset": dataset,
        "variant": variant,
        "split_seed": split_seed,
        "device": str(device),
        "model_args": to_jsonable(vars(model_args)),
        "load_reports": load_reports,
        "parameter_groups": parameter_groups,
        "activation_stats": activation_stats,
        "model_state_matches": model_state_matches,
        "activation_ran": activation_ran,
        "activation_trusted": (
            activation_ran
            and model_state_matches
            and not cli.allow_partial_activations
            and activation_error is None
        ),
        "activation_error": activation_error,
        "thresholds": {"near_zero": cli.near_zero, "small": cli.small},
        "verdicts": verdicts,
    }

    print(f"Checkpoint: {cli.checkpoint}")
    print(f"Dataset/variant/seed: {dataset} / {variant} / {split_seed}")
    print(f"Epoch: {checkpoint.get('epoch', '-')}; device: {device}")
    for report in load_reports:
        print(
            f"Loaded {report['component']}: {report['loaded_tensors']} tensors, "
            f"missing={len(report['missing_keys'])}, unexpected={len(report['unexpected_keys'])}, "
            f"mismatched={len(report['mismatched_shapes'])}"
        )
    if activation_error:
        print(f"Activation note: {activation_error}")
    else:
        print(f"Activation trusted: {payload['activation_trusted']}")

    param_rows = [
        {
            "group": name,
            "present": stats.get("present"),
            "tensors": stats.get("tensors"),
            "params": stats.get("params"),
            "l2": stats.get("l2"),
            "mean_abs": stats.get("mean_abs"),
            "max_abs": stats.get("max_abs"),
            "l2_ratio_vs_out": stats.get("l2_ratio_vs_out"),
        }
        for name, stats in parameter_groups.items()
    ]
    print_table(
        "Parameter Groups",
        param_rows,
        ["group", "present", "tensors", "params", "l2", "mean_abs", "max_abs", "l2_ratio_vs_out"],
    )

    activation_rows = [
        {
            "activation": name,
            "batches": stats.get("batches"),
            "count": stats.get("count"),
            "mean": stats.get("mean"),
            "var": stats.get("var"),
            "mean_abs": stats.get("mean_abs"),
            "max_abs": stats.get("max_abs"),
        }
        for name, stats in activation_stats.items()
    ]
    print_table(
        "Activation Stats",
        activation_rows,
        ["activation", "batches", "count", "mean", "var", "mean_abs", "max_abs"],
    )

    print_table(
        "Near-Zero Verdicts",
        verdicts,
        ["component", "present", "metric", "verdict", "evidence"],
    )

    if cli.out_json is not None:
        cli.out_json.parent.mkdir(parents=True, exist_ok=True)
        with cli.out_json.open("w", encoding="utf-8") as handle:
            json.dump(to_jsonable(payload), handle, indent=2)
        print(f"\nWrote JSON: {cli.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
