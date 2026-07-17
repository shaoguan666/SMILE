"""Bounded ODP late-fusion redesign.

This module intentionally contains no clinical values in the policy path and
no token-wise conditioning.  Observation histories are shifted right before
the causal convolution, and only per-variable CLS tokens are modified by the
late residual branch.
"""

import copy

import torch
from torch import nn
import torch.nn.functional as F

from .smart import SMILELeanEncoder, TimeEncoder, length_to_mask


POLICY_HIDDEN_DIM = 64
POLICY_KERNEL_SIZE = 7
POLICY_LOSS_WEIGHT = 0.1


def policy_valid_mask(lens, max_len, num_vars, include_t0=False, device=None):
    valid = length_to_mask(lens, max_len=max_len, device=device)
    if not include_t0 and max_len:
        valid = valid.clone()
        valid[:, 0] = False
    return valid.unsqueeze(-1).expand(-1, -1, num_vars)


def causal_policy_bce(logits, target_mask, lens, include_t0=False):
    """BCE on clean, unpadded targets; the primary loss excludes ``t=0``."""
    if logits.shape != target_mask.shape:
        raise ValueError(
            f"policy target shape {tuple(target_mask.shape)} does not match "
            f"logits shape {tuple(logits.shape)}"
        )
    _, time_steps, num_vars = logits.shape
    valid = policy_valid_mask(
        lens, time_steps, num_vars, include_t0=include_t0, device=logits.device
    )
    if not valid.any():
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[valid], target_mask.float()[valid])


def masked_policy_mean(hidden, lens):
    """Per-variable masked mean over unpadded ``t>=1`` positions.

    ``hidden`` is ``(B,T,V,H)`` and the result is ``(B,V,H)``.  Samples with no
    eligible position return exact zeros without NaNs.
    """
    batch, time_steps, num_vars, _ = hidden.shape
    valid = policy_valid_mask(
        lens, time_steps, num_vars, include_t0=False, device=hidden.device
    ).unsqueeze(-1)
    numerator = (hidden * valid).sum(dim=1)
    denominator = valid.sum(dim=1).clamp(min=1)
    pooled = numerator / denominator
    has_value = valid.any(dim=1)
    return torch.where(has_value, pooled, torch.zeros_like(pooled))


class CausalObservationPolicyEncoder(nn.Module):
    """Predict ``M_t`` using only ``M_<t``, time and variable identity."""

    MODES = {"time", "own", "full", "shuffled"}

    def __init__(self, num_vars, hidden_dim=POLICY_HIDDEN_DIM, time_dim=16,
                 kernel_size=POLICY_KERNEL_SIZE, mode="full"):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown forecasting mode: {mode}")
        if kernel_size < 1:
            raise ValueError("policy_kernel_size must be at least 1")
        self.num_vars = int(num_vars)
        self.hidden_dim = int(hidden_dim)
        self.kernel_size = int(kernel_size)
        self.mode = mode
        self.time_encoder = TimeEncoder(time_dim)
        self.var_emb = nn.Embedding(num_vars, hidden_dim)
        self.causal_conv = nn.Conv1d(num_vars, hidden_dim, kernel_size, padding=0)
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        self.density_proj = nn.Linear(1, hidden_dim)
        self.head = nn.Sequential(
            nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        nn.init.normal_(self.var_emb.weight, std=0.02)

    @staticmethod
    def shift_right(mask):
        return F.pad(mask, (0, 0, 1, 0))[:, :-1]

    @staticmethod
    def running_density(shifted_own):
        time_steps = shifted_own.shape[1]
        denom = torch.arange(
            time_steps, device=shifted_own.device, dtype=shifted_own.dtype
        ).clamp(min=1).view(1, time_steps, 1)
        return torch.cumsum(shifted_own, dim=1) / denom

    def _full_convolution(self, shifted):
        seq = shifted.permute(0, 2, 1)
        conv = self.causal_conv(F.pad(seq, (self.kernel_size - 1, 0)))
        return conv.transpose(1, 2).unsqueeze(2).expand(-1, -1, self.num_vars, -1)

    def _targetwise_convolution(self, shifted):
        # shifted: (B,T,target,source).  Flatten target into the batch while
        # keeping the same full causal-convolution parameter container.
        batch, time_steps, targets, sources = shifted.shape
        seq = shifted.permute(0, 2, 3, 1).reshape(batch * targets, sources, time_steps)
        conv = self.causal_conv(F.pad(seq, (self.kernel_size - 1, 0)))
        return conv.reshape(batch, targets, self.hidden_dim, time_steps).permute(0, 3, 1, 2)

    def _own_convolution(self, shifted):
        seq = shifted.permute(0, 2, 1)
        padded = F.pad(seq, (self.kernel_size - 1, 0))
        outputs = []
        for variable_id in range(self.num_vars):
            outputs.append(F.conv1d(
                padded[:, variable_id:variable_id + 1],
                self.causal_conv.weight[:, variable_id:variable_id + 1],
                bias=self.causal_conv.bias,
            ))
        return torch.stack(outputs, dim=2).permute(0, 3, 2, 1)

    def forward(self, original_mask, time=None, lens=None, policy_history_mask=None):
        if original_mask.ndim != 3:
            raise ValueError("original_mask must have shape (B,T,V)")
        batch, time_steps, num_vars = original_mask.shape
        if num_vars != self.num_vars:
            raise ValueError(f"expected V={self.num_vars}, received V={num_vars}")
        if lens is None:
            lens = torch.full((batch,), time_steps, device=original_mask.device, dtype=torch.long)
        valid_bt = length_to_mask(lens, max_len=time_steps, device=original_mask.device)
        clean = original_mask.float() * valid_bt.unsqueeze(-1)
        shifted_own = self.shift_right(clean)

        if self.mode == "time":
            hidden = clean.new_zeros(batch, time_steps, num_vars, self.hidden_dim)
        elif self.mode == "own":
            hidden = self._own_convolution(shifted_own)
        elif self.mode == "full":
            hidden = self._full_convolution(shifted_own)
        else:
            if policy_history_mask is None or policy_history_mask.ndim != 4:
                raise ValueError("shuffled mode requires policy_history_mask with shape (B,T,V,V)")
            if policy_history_mask.shape != (batch, time_steps, num_vars, num_vars):
                raise ValueError("policy_history_mask shape mismatch")
            history = policy_history_mask.float() * valid_bt[:, :, None, None]
            shifted = F.pad(history, (0, 0, 0, 0, 1, 0))[:, :-1]
            hidden = self._targetwise_convolution(shifted)

        if time is None:
            time_condition = hidden.new_zeros(batch, time_steps, self.hidden_dim)
        else:
            if time.shape != (batch, time_steps):
                raise ValueError(f"time must have shape {(batch, time_steps)}")
            safe_time = time.float() * valid_bt
            time_condition = self.time_proj(self.time_encoder(safe_time))

        hidden = hidden + self.var_emb.weight.view(1, 1, num_vars, -1)
        hidden = hidden + time_condition.unsqueeze(2)
        density = self.running_density(shifted_own)
        if self.mode != "time":
            hidden = hidden + self.density_proj(density.unsqueeze(-1))
        hidden = hidden * valid_bt[:, :, None, None]
        logits = self.head(hidden).squeeze(-1) * valid_bt.unsqueeze(-1)
        return {
            "hidden": hidden,
            "logits": logits,
            "prob": torch.sigmoid(logits),
            "valid_mask": valid_bt,
            "pooled": masked_policy_mean(hidden, lens),
        }


def _clinical_args(args, use_density):
    clinical_args = copy.copy(args)
    clinical_args.abl_no_mnar_bias = True
    clinical_args.abl_random_bias = False
    clinical_args.abl_global_comiss = False
    clinical_args.abl_no_density = not use_density
    return clinical_args


class NoCoMissClinicalEncoder(SMILELeanEncoder):
    """Matched Lean clinical backbone with CoMiss disabled."""

    def __init__(self, args, use_density):
        super().__init__(_clinical_args(args, use_density=use_density))


class ODPLateFusionEncoder(nn.Module):
    """Density clinical control plus per-variable zero-gated ODP late fusion."""

    def __init__(self, args, shuffled=False, forecast_mode="full"):
        super().__init__()
        self.clinical = NoCoMissClinicalEncoder(args, use_density=True)
        mode = "shuffled" if shuffled else forecast_mode
        self.policy_encoder = CausalObservationPolicyEncoder(
            num_vars=args.input_dim,
            hidden_dim=getattr(args, "policy_hidden_dim", POLICY_HIDDEN_DIM),
            time_dim=getattr(args, "time_dim", 16),
            kernel_size=getattr(args, "policy_kernel_size", POLICY_KERNEL_SIZE),
            mode=mode,
        )
        self.policy_projection = nn.Linear(self.policy_encoder.hidden_dim, args.d_model)
        nn.init.xavier_uniform_(self.policy_projection.weight)
        nn.init.zeros_(self.policy_projection.bias)
        self.alpha = nn.Parameter(torch.zeros(()))
        self.register_buffer("shuffle_seed", torch.tensor(int(getattr(args, "seed", 0))))

    def forward(self, x, lens, mask, time=None, original_mask=None,
                policy_history_mask=None, return_policy=False, **kwargs):
        clinical = self.clinical(
            x=x, lens=lens, mask=mask, time=time, original_mask=original_mask, **kwargs
        )
        if original_mask is None:
            raise ValueError("ODP requires the clean natural original_mask")
        policy = self.policy_encoder(
            original_mask=original_mask,
            time=time,
            lens=lens,
            policy_history_mask=policy_history_mask,
        )
        residual = self.alpha * self.policy_projection(policy["pooled"])
        # Avoid in-place mutation and modify only h[:,:,0,:].
        fused = torch.cat((clinical[:, :, :1] + residual.unsqueeze(2), clinical[:, :, 1:]), dim=2)
        if return_policy:
            return fused, policy
        return fused


def odp_incremental_parameters(args):
    probe = ODPLateFusionEncoder(args)
    return sum(p.numel() for name, p in probe.named_parameters() if not name.startswith("clinical."))


class ParameterMatchedAdapterEncoder(nn.Module):
    """Functional non-ODP residual adapter matched to ODP branch capacity."""

    def __init__(self, args):
        super().__init__()
        self.clinical = NoCoMissClinicalEncoder(args, use_density=True)
        target = odp_incremental_parameters(args)
        d_model = int(args.d_model)
        per_width = 2 * d_model + 1
        width = max(1, round((target - d_model - 1) / per_width))
        self.adapter_in = nn.Linear(d_model, width)
        self.adapter_out = nn.Linear(width, d_model)
        nn.init.xavier_uniform_(self.adapter_in.weight)
        nn.init.zeros_(self.adapter_in.bias)
        nn.init.xavier_uniform_(self.adapter_out.weight)
        nn.init.zeros_(self.adapter_out.bias)
        self.beta = nn.Parameter(torch.zeros(()))
        self.match_target = target
        self.match_actual = sum(
            p.numel() for name, p in self.named_parameters() if not name.startswith("clinical.")
        )
        self.match_delta = self.match_actual - self.match_target
        self.match_percent = 100.0 * abs(self.match_delta) / self.match_target
        if self.match_percent > 2.0:
            raise ValueError(
                f"parameter match error {self.match_percent:.3f}% exceeds 2% "
                f"(target={self.match_target}, actual={self.match_actual})"
            )

    def forward(self, x, lens, mask, time=None, original_mask=None, **kwargs):
        clinical = self.clinical(
            x=x, lens=lens, mask=mask, time=time, original_mask=original_mask, **kwargs
        )
        cls = clinical[:, :, 0]
        adapted = self.adapter_out(F.gelu(self.adapter_in(cls)))
        fused_cls = cls + self.beta * adapted
        return torch.cat((fused_cls.unsqueeze(2), clinical[:, :, 1:]), dim=2)
