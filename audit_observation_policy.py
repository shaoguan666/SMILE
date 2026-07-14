"""Observation Policy Audit v4 (standalone diagnostic, no downstream task).

Strictly causal predictors of the natural observation mask m_{t,v} (1=observed).
Every predictor at position t uses ONLY information strictly before t.

Pre-registered primary hypothesis (fixed BEFORE looking at results):
    D2 (cross-variable causal mask history) predicts observation better than a
    capacity-and-time-matched shallow baseline S4+, beyond per-variable history
    (D1-wide) and beyond a cross-variable-destroying control (D2-shuffled).
D3 (adds clinical values) is EXPLORATORY only, never the champion.

Predictor set:
    S1  static per-variable        P(m_v)
    S2  static per-(time,var)      P(m_{t,v})
    S3  persistence                m_{t-1,v}
    S4  shallow logistic reg       recent-k / run-length / TSLO   (no time, linear)
    S4+ shallow + time + MLP head  same capacity/time as D1        (fair shallow)
    D1  per-variable causal conv   m_{<t,v}
    D1w per-variable, params ~= D2 (capacity control)
    D2  cross-variable causal conv m_{<t,:}                        (PRIMARY)
    D2s cross-variable, non-target channels patient-shuffled       (structure control)
    D3  cross-variable + values    (m, m*x)_{<t,:}                 (exploratory)

Two prediction tasks are scored, both strictly causal:
  (A) mask nowcast:   predict m_{t,v};  metrics maAUPRC/micro/Brier/NLL/ECE.
  (B) transition event: predict z_{t,v}=1[m_t!=m_{t-1}] over all valid t>=1 using
      q^flip = m_{t-1} + pi - 2*m_{t-1}*pi  (P(flip) implied by the mask model),
      metrics maAUPRC/Brier/NLL. This asks "can you anticipate a change", NOT
      "given a change, name the new state".

Go/no-go uses VALIDATION only; test sealed unless --evaluate-test. Multi-seed
aggregation, the frozen paired decision, and the patient bootstrap live in
aggregate_audit.py, which consumes the per-record score dumps written here.

Run:  python audit_observation_policy.py --dataset c12 --seed 0 --epochs 20
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from sklearn.metrics import average_precision_score

from data.challenge2012 import load_challenge_2012
from data.challenge2019 import load_challenge_2019
from data.mimiciii import (
    load_mimic_iii_mortality,
    load_mimic_iii_phenotyping,
    load_mimic_iii_decompensation,
    load_mimic_iii_lengthofstay,
)
from data.dataloader import collate_fn
from models.smart import TimeEncoder


_DATASET_CFG = {
    'c12':                  dict(input_dim=37, max_len=48),
    'c19':                  dict(input_dim=34, max_len=60),
    'mimic_mortality':      dict(input_dim=17, max_len=48),
    'mimic_phenotyping':    dict(input_dim=17, max_len=60),
    'mimic_decompensation': dict(input_dim=17, max_len=24),
    'mimic_lengthofstay':   dict(input_dim=17, max_len=24),
}

# Models whose per-record validation scores are dumped for the patient bootstrap.
PRIMARY_MODELS = ['S4plus', 'D1wide', 'D2', 'D2shuf']


def load_dataset(name, split_seed):
    if name == 'c12':
        return load_challenge_2012()
    if name == 'c19':
        return load_challenge_2019()
    if name == 'mimic_mortality':
        return load_mimic_iii_mortality(split_seed=split_seed)
    if name == 'mimic_phenotyping':
        return load_mimic_iii_phenotyping(split_seed=split_seed)
    if name == 'mimic_decompensation':
        return load_mimic_iii_decompensation(split_seed=split_seed)
    if name == 'mimic_lengthofstay':
        return load_mimic_iii_lengthofstay(
            task='regression', label_unit='auto', split_seed=split_seed)
    raise ValueError('Unknown dataset: %s' % name)


# ---------------------------------------------------------------------------
# Causal feature helpers (operate on m_shift = mask shifted right by 1)
# ---------------------------------------------------------------------------

def shift_right(z):
    return F.pad(z, (0, 0, 1, 0))[:, :-1, :]


def running_density(m_shift):
    B, T, V = m_shift.shape
    cum = torch.cumsum(m_shift, dim=1)
    denom = torch.arange(T, device=m_shift.device).clamp(min=1).view(1, T, 1).float()
    return cum / denom


def recent_k(m_shift, k):
    B, T, V = m_shift.shape
    x = m_shift.permute(0, 2, 1).reshape(B * V, 1, T)
    x = F.avg_pool1d(F.pad(x, (k - 1, 0)), kernel_size=k, stride=1)
    return x.reshape(B, V, T).permute(0, 2, 1)


def time_since_last_obs(m_shift, max_len):
    B, T, V = m_shift.shape
    idx = torch.arange(T, device=m_shift.device).view(1, T, 1).float().expand(B, T, V)
    obs_idx = torch.where(m_shift > 0.5, idx, torch.full_like(idx, -1.0))
    last_obs = torch.cummax(obs_idx, dim=1).values
    tslo = idx - last_obs
    tslo = torch.where(last_obs < 0, torch.full_like(tslo, float(max_len)), tslo)
    return tslo / float(max_len)


def observed_run_length(m_shift, max_len):
    # consecutive observed count ending at each position (vectorized).
    B, T, V = m_shift.shape
    idx = torch.arange(T, device=m_shift.device).view(1, T, 1).float().expand(B, T, V)
    zero_pos = torch.where(m_shift < 0.5, idx, torch.full_like(idx, -1.0))
    last_zero = torch.cummax(zero_pos, dim=1).values
    return (idx - last_zero) / float(max_len)


def shallow_features(m, max_len):
    ms = shift_right(m)
    return torch.stack([
        recent_k(ms, 1), recent_k(ms, 3), recent_k(ms, 5),
        running_density(ms),
        time_since_last_obs(ms, max_len),
        observed_run_length(ms, max_len),
    ], dim=-1)                                              # (B, T, V, 6)


# ---------------------------------------------------------------------------
# Predictor models (all strictly causal; return pi in (0,1))
# ---------------------------------------------------------------------------

class ShallowLogReg(nn.Module):
    """S4: logistic regression on shallow causal features. No time, linear."""

    def __init__(self, num_vars, max_len, var_emb_dim=8):
        super().__init__()
        self.max_len = max_len
        self.var_emb = nn.Embedding(num_vars, var_emb_dim)
        nn.init.normal_(self.var_emb.weight, std=0.02)
        self.linear = nn.Linear(6 + var_emb_dim, 1)

    def forward(self, m, x, time):
        B, T, V = m.shape
        feats = shallow_features(m, self.max_len)
        ve = self.var_emb.weight.view(1, 1, V, -1).expand(B, T, V, -1)
        return torch.sigmoid(self.linear(torch.cat([feats, ve], dim=-1)).squeeze(-1))


class ShallowPlus(nn.Module):
    """S4+: shallow features + TimeEncoder + the SAME nonlinear MLP head as D1.

    Isolates D1's unique ingredient (temporal-convolution history) from absolute
    time and nonlinear capacity, which S4 lacked.
    """

    def __init__(self, num_vars, max_len, time_dim=16, d_hidden=64):
        super().__init__()
        self.max_len = max_len
        self.time_encoder = TimeEncoder(time_dim)
        self.var_emb = nn.Embedding(num_vars, d_hidden)
        self.feat_proj = nn.Linear(6, d_hidden)
        self.time_proj = nn.Linear(time_dim, d_hidden)
        self.head = nn.Sequential(
            nn.GELU(), nn.Linear(d_hidden, d_hidden),
            nn.GELU(), nn.Linear(d_hidden, 1))

    def forward(self, m, x, time):
        B, T, V = m.shape
        h = self.feat_proj(shallow_features(m, self.max_len))    # (B,T,V,H)
        h = h + self.var_emb.weight.view(1, 1, V, -1)
        h = h + self.time_proj(self.time_encoder(time)).unsqueeze(2)
        return torch.sigmoid(self.head(h).squeeze(-1))


class PerVarConv(nn.Module):
    """D1 / D1-wide: per-variable causal temporal conv over m_{<t,v} only."""

    def __init__(self, num_vars, max_len, time_dim=16, d_hidden=64, kernel=7):
        super().__init__()
        self.kernel = kernel
        self.time_encoder = TimeEncoder(time_dim)
        self.var_emb = nn.Embedding(num_vars, d_hidden)
        self.causal_conv = nn.Conv1d(1, d_hidden, kernel, padding=0)
        self.time_proj = nn.Linear(time_dim, d_hidden)
        self.dens_proj = nn.Linear(1, d_hidden)
        self.head = nn.Sequential(
            nn.GELU(), nn.Linear(d_hidden, d_hidden),
            nn.GELU(), nn.Linear(d_hidden, 1))

    def forward(self, m, x, time):
        B, T, V = m.shape
        ms = shift_right(m)
        mv = ms.permute(0, 2, 1).reshape(B * V, 1, T)
        h = self.causal_conv(F.pad(mv, (self.kernel - 1, 0)))        # (B*V, H, T)
        h = h.transpose(1, 2).reshape(B, V, T, -1).permute(0, 2, 1, 3)   # (B,T,V,H)
        h = h + self.var_emb.weight.view(1, 1, V, -1)
        h = h + self.time_proj(self.time_encoder(time)).unsqueeze(2)
        h = h + self.dens_proj(running_density(ms).unsqueeze(-1))
        return torch.sigmoid(self.head(h).squeeze(-1))


class CrossVarConv(nn.Module):
    """D2 / D3: cross-variable causal conv over m_{<t,:} (+ values if use_values)."""

    def __init__(self, num_vars, max_len, time_dim=16, d_hidden=64, kernel=7,
                 use_values=False):
        super().__init__()
        self.num_vars = num_vars
        self.kernel = kernel
        self.use_values = use_values
        self.time_encoder = TimeEncoder(time_dim)
        self.var_emb = nn.Embedding(num_vars, d_hidden)
        in_ch = num_vars * (2 if use_values else 1)
        self.causal_conv = nn.Conv1d(in_ch, d_hidden, kernel, padding=0)
        self.time_proj = nn.Linear(time_dim, d_hidden)
        self.dens_proj = nn.Linear(1, d_hidden)
        self.head = nn.Sequential(
            nn.GELU(), nn.Linear(d_hidden, d_hidden),
            nn.GELU(), nn.Linear(d_hidden, 1))

    def _shared(self, h_seq, ms, time, V):
        # h_seq: (B, T, H) shared temporal embedding; broadcast to variables.
        B, T, _ = h_seq.shape
        h = h_seq.unsqueeze(2) + self.var_emb.weight.view(1, 1, V, -1)
        h = h + self.time_proj(self.time_encoder(time)).unsqueeze(2)
        h = h + self.dens_proj(running_density(ms).unsqueeze(-1))
        return torch.sigmoid(self.head(h).squeeze(-1))

    def forward(self, m, x, time):
        B, T, V = m.shape
        ms = shift_right(m)
        if self.use_values:
            inp = torch.cat([ms, shift_right(m * x)], dim=-1)   # (B, T, 2V)
        else:
            inp = ms
        seq = inp.permute(0, 2, 1)                              # (B, C, T)
        h = self.causal_conv(F.pad(seq, (self.kernel - 1, 0))).transpose(1, 2)  # (B,T,H)
        return self._shared(h, ms, time, V)


class CrossVarConvShuffled(CrossVarConv):
    """D2-shuffled: exact D2 architecture and parameters, but for target j the
    non-j channels of the mask history are replaced by an independent random
    patient (per channel). Target j keeps its true own history and own density.

    Uses conv linearity: for input with channel j true and others shuffled,
    conv = conv(S) + W[:, j] * (true_j - S_j).  So D2 > D2-shuffled isolates
    genuine cross-variable co-observation from capacity.
    """

    def __init__(self, num_vars, max_len, time_dim=16, d_hidden=64, kernel=7):
        super().__init__(num_vars, max_len, time_dim, d_hidden, kernel,
                         use_values=False)

    def forward(self, m, x, time):
        B, T, V = m.shape
        k = self.kernel
        ms = shift_right(m)                                     # (B,T,V) true
        seqT = ms.permute(0, 2, 1)                              # (B,V,T)
        # Independent per-channel patient permutation.
        S = torch.empty_like(seqT)
        for v in range(V):
            perm = torch.randperm(B, device=m.device)
            S[:, v, :] = seqT[perm, v, :]
        convS = self.causal_conv(F.pad(S, (k - 1, 0)))          # (B,H,T) incl bias
        delta = seqT - S                                        # (B,V,T)
        dpad = F.pad(delta, (k - 1, 0))                         # (B,V,T+k-1)
        W = self.causal_conv.weight                            # (H, V, k)
        corr = []
        for j in range(V):
            corr.append(F.conv1d(dpad[:, j:j + 1, :], W[:, j:j + 1, :]))  # (B,H,T)
        corr = torch.stack(corr, dim=1)                         # (B,V,H,T)
        h = convS.unsqueeze(1) + corr                           # (B,V,H,T)
        h = h.permute(0, 3, 1, 2)                               # (B,T,V,H)
        h = h + self.var_emb.weight.view(1, 1, V, -1)
        h = h + self.time_proj(self.time_encoder(time)).unsqueeze(2)
        h = h + self.dens_proj(running_density(ms).unsqueeze(-1))   # true own density
        return torch.sigmoid(self.head(h).squeeze(-1))


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def match_pervar_width(input_dim, max_len, time_dim, kernel, target_params):
    """Pick D1 hidden width whose param count is closest to target (D2)."""
    best_h, best_gap = 64, float('inf')
    for h in range(64, 512, 2):
        p = count_params(PerVarConv(input_dim, max_len, time_dim, h, kernel))
        gap = abs(p - target_params)
        if gap < best_gap:
            best_gap, best_h = gap, h
    return best_h


# ---------------------------------------------------------------------------
# Data / contract helpers
# ---------------------------------------------------------------------------

def build_loader(dataset, batch_size, shuffle, workers=0, seed=42):
    generator = torch.Generator().manual_seed(seed)
    sampler = (RandomSampler(dataset, generator=generator)
               if shuffle else SequentialSampler(dataset))
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      collate_fn=collate_fn, num_workers=workers, drop_last=False)


def valid_mask_from_lens(lens, T):
    ar = torch.arange(T, device=lens.device).view(1, T)
    return ar < lens.view(-1, 1)


def validate_dataset_contract(dataset, dataset_name, input_dim, max_len):
    if len(dataset) == 0:
        raise ValueError('%s split is empty' % dataset_name)
    required = {'x', 'mask', 'time', 'lens'}
    lengths = []
    for index, sample in enumerate(dataset.data):
        missing = required.difference(sample)
        if missing:
            raise ValueError('%s sample %d missing keys %s'
                             % (dataset_name, index, sorted(missing)))
        lens = int(sample['lens'])
        if not (len(sample['x']) == len(sample['mask']) == len(sample['time']) == lens):
            raise ValueError('%s sample %d inconsistent lengths' % (dataset_name, index))
        if lens < 1 or lens > max_len:
            raise ValueError('%s sample %d lens=%d outside [1,%d]'
                             % (dataset_name, index, lens, max_len))
        mask = np.asarray(sample['mask'])
        time = np.asarray(sample['time'], dtype=np.float64)
        if mask.ndim != 2 or mask.shape[1] != input_dim:
            raise ValueError('%s sample %d mask shape %s expected (*,%d)'
                             % (dataset_name, index, mask.shape, input_dim))
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError('%s sample %d mask not binary' % (dataset_name, index))
        if not np.all(np.isfinite(time)) or np.any(np.diff(time) < 0):
            raise ValueError('%s sample %d bad time' % (dataset_name, index))
        lengths.append(lens)
    return dict(records=len(lengths), min_len=min(lengths),
                max_len=max(lengths), mean_len=float(np.mean(lengths)))


def compute_static_priors(loader, input_dim, max_len, device):
    obs_v = torch.zeros(input_dim, device=device)
    tot_v = torch.zeros(input_dim, device=device)
    obs_v_prod = torch.zeros(input_dim, device=device)
    total_steps_prod = 0
    obs_tv = torch.zeros(max_len, input_dim, device=device)
    tot_tv = torch.zeros(max_len, input_dim, device=device)
    for batch in loader:
        m = batch['mask'].float().to(device)
        lens = batch['lens'].to(device)
        B, T, V = m.shape
        vmask = valid_mask_from_lens(lens, T).unsqueeze(-1).float()
        mv = m * vmask
        obs_v += mv.sum(dim=(0, 1))
        tot_v += vmask.expand(-1, -1, V).sum(dim=(0, 1))
        obs_v_prod += m.sum(dim=(0, 1))
        total_steps_prod += B * T
        Tc = min(T, max_len)
        obs_tv[:Tc] += mv[:, :Tc].sum(dim=0)
        tot_tv[:Tc] += vmask[:, :Tc].expand(-1, -1, V).sum(dim=0)
    p_var_valid = obs_v / tot_v.clamp(min=1.0)
    p_var_prod = obs_v_prod / max(total_steps_prod, 1)
    p_time_var = torch.where(
        tot_tv > 0, obs_tv / tot_tv.clamp(min=1.0),
        p_var_valid.view(1, -1).expand_as(obs_tv))
    return p_var_valid, p_var_prod, p_time_var


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_model(model, train_loader, device, epochs, lr, name):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for batch in train_loader:
            m = batch['mask'].float().to(device)
            x = batch['x'].float().to(device)
            time = batch['time'].float().to(device)
            lens = batch['lens'].to(device)
            B, T, V = m.shape
            vmask = valid_mask_from_lens(lens, T).unsqueeze(-1).expand(-1, -1, V)
            pi = model(m, x, time).clamp(1e-4, 1 - 1e-4)
            loss = F.binary_cross_entropy(pi[vmask], m[vmask])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * B
            n += B
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            print('[Audit]   %-8s epoch %3d  train BCE %.4f' % (name, ep, tot / max(n, 1)))
    return model


def collect_arrays(loader, models, p_var, p_var_prod, p_time_var, max_len, device):
    """One eval pass. Returns flat valid-position arrays and a name->pi dict."""
    for mdl in models.values():
        mdl.eval()
    static = ['S1_static_var', 'S2_static_tv', 'S3_persistence']
    names = static + list(models.keys())
    acc = {n: [] for n in names}
    y_all, vidx_all, mprev_all, hasprev_all, rec_all = [], [], [], [], []
    rec_base = 0
    with torch.no_grad():
        for batch in loader:
            m = batch['mask'].float().to(device)
            x = batch['x'].float().to(device)
            time = batch['time'].float().to(device)
            lens = batch['lens'].to(device)
            B, T, V = m.shape
            vmask = valid_mask_from_lens(lens, T)
            vm = vmask.unsqueeze(-1).expand(-1, -1, V)
            sel = vm.reshape(-1)
            pi = {}
            pi['S1_static_var'] = p_var.view(1, 1, V).expand(B, T, V)
            t_idx = torch.arange(T, device=device).clamp(max=max_len - 1)
            pi['S2_static_tv'] = p_time_var[t_idx].unsqueeze(0).expand(B, -1, -1)
            mprev = shift_right(m)
            pers = mprev.clone()
            pers[:, 0, :] = p_var.view(1, V)
            pi['S3_persistence'] = pers
            for nm, mdl in models.items():
                pi[nm] = mdl(m, x, time)
            has_prev = vm.clone()
            has_prev[:, 0, :] = False
            for nm in names:
                acc[nm].append(pi[nm].reshape(-1)[sel].cpu())
            y_all.append(m.reshape(-1)[sel].cpu())
            vi = torch.arange(V, device=device).view(1, 1, V).expand(B, T, V)
            vidx_all.append(vi.reshape(-1)[sel].cpu())
            mprev_all.append(mprev.reshape(-1)[sel].cpu())
            hasprev_all.append(has_prev.reshape(-1)[sel].cpu())
            rec = (rec_base + torch.arange(B, device=device)).view(B, 1, 1).expand(B, T, V)
            rec_all.append(rec.reshape(-1)[sel].cpu())
            rec_base += B
    out = {n: torch.cat(acc[n]).numpy() for n in names}
    return dict(
        y=torch.cat(y_all).numpy(),
        var_idx=torch.cat(vidx_all).numpy(),
        m_prev=torch.cat(mprev_all).numpy(),
        has_prev=torch.cat(hasprev_all).numpy().astype(bool),
        record=torch.cat(rec_all).numpy().astype(np.int32),
        pi=out,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def brier(y, p):
    return float(np.mean((p - y) ** 2))


def nll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y, p, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    total, e = len(y), 0.0
    for b in range(n_bins):
        mb = idx == b
        if np.any(mb):
            e += (np.sum(mb) / total) * abs(np.mean(y[mb]) - np.mean(p[mb]))
    return float(e)


def macro_auprc(y, p, var_idx, num_vars, subset=None):
    scores = []
    for v in range(num_vars):
        mv = var_idx == v
        if subset is not None:
            mv = mv & subset
        if not np.any(mv):
            continue
        yv = y[mv]
        if yv.sum() == 0 or yv.sum() == len(yv):
            continue
        scores.append(average_precision_score(yv, p[mv]))
    return (float(np.mean(scores)) if scores else float('nan')), len(scores)


def micro_auprc(y, p, subset=None):
    if subset is not None:
        y, p = y[subset], p[subset]
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return float('nan')
    return float(average_precision_score(y, p))


def flip_prob(pi, m_prev):
    # P(m_t != m_{t-1}) implied by a mask model with P(m_t=1)=pi.
    return m_prev + pi - 2.0 * m_prev * pi


def summarize(name, arrs, pi, num_vars):
    y, vidx, m_prev, has_prev = arrs['y'], arrs['var_idx'], arrs['m_prev'], arrs['has_prev']
    ma, used = macro_auprc(y, p=pi, var_idx=vidx, num_vars=num_vars)
    # Transition-event task: label z=flip, score q^flip, over valid t>=1.
    z = (y != m_prev).astype(np.float64)
    q = flip_prob(pi, m_prev)
    ma_tr, used_tr = macro_auprc(z, p=q, var_idx=vidx, num_vars=num_vars, subset=has_prev)
    zt, qt = z[has_prev], q[has_prev]
    return dict(
        predictor=name,
        macro_auprc=ma, macro_vars_used=used,
        micro_auprc=micro_auprc(y, pi),
        brier=brier(y, pi), nll=nll(y, pi), ece=ece(y, pi),
        transition_macro_auprc=ma_tr, transition_vars_used=used_tr,
        transition_micro_auprc=micro_auprc(zt, qt),
        transition_brier=brier(zt, qt), transition_nll=nll(zt, qt),
        transition_prevalence=float(zt.mean()) if len(zt) else float('nan'),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='mimic_decompensation',
                    choices=list(_DATASET_CFG.keys()))
    ap.add_argument('--split-seed', '--split_seed', dest='split_seed', type=int, default=42)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--batch-size', '--batch_size', dest='batch_size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--d-hidden', '--d_hidden', dest='d_hidden', type=int, default=64)
    ap.add_argument('--kernel', type=int, default=7)
    ap.add_argument('--time-dim', '--time_dim', dest='time_dim', type=int, default=16)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--evaluate-test', action='store_true', default=False)
    ap.add_argument('--no-dump-scores', dest='dump_scores', action='store_false', default=True,
                    help='Skip writing per-record validation scores for the bootstrap.')
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    if args.dataset in ('c12', 'c19') and args.split_seed != 42:
        raise ValueError('%s loader exposes only the fixed split seed 42.' % args.dataset)
    if args.kernel < 1:
        raise ValueError('--kernel must be positive')

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    cfg = _DATASET_CFG[args.dataset]
    input_dim, max_len = cfg['input_dim'], cfg['max_len']
    print('[Audit] dataset=%s input_dim=%d max_len=%d device=%s seed=%d'
          % (args.dataset, input_dim, max_len, device, args.seed))

    train_ds, val_ds, test_ds = load_dataset(args.dataset, args.split_seed)
    print('[Audit] sizes: train=%d val=%d test=%d'
          % (len(train_ds), len(val_ds), len(test_ds)))
    split_contracts = {
        s: validate_dataset_contract(d, '%s/%s' % (args.dataset, s), input_dim, max_len)
        for s, d in (('train', train_ds), ('val', val_ds), ('test', test_ds))}

    train_loader = build_loader(train_ds, args.batch_size, True, args.workers, seed=args.seed)
    train_eval_loader = build_loader(train_ds, args.batch_size, False, args.workers, seed=args.seed)
    val_loader = build_loader(val_ds, args.batch_size, False, args.workers, seed=args.seed)
    test_loader = build_loader(test_ds, args.batch_size, False, args.workers, seed=args.seed)

    p_var, p_var_prod, p_time_var = compute_static_priors(
        train_eval_loader, input_dim, max_len, device)
    prior_abs_diff = (p_var - p_var_prod).abs()

    # Capacity-matched D1 width from D2 param count.
    d2_ref = CrossVarConv(input_dim, max_len, args.time_dim, args.d_hidden,
                          args.kernel, use_values=False)
    d2_params = count_params(d2_ref)
    d1wide_h = match_pervar_width(input_dim, max_len, args.time_dim, args.kernel, d2_params)

    models = {
        'S4':     ShallowLogReg(input_dim, max_len).to(device),
        'S4plus': ShallowPlus(input_dim, max_len, args.time_dim, args.d_hidden).to(device),
        'D1':     PerVarConv(input_dim, max_len, args.time_dim, args.d_hidden, args.kernel).to(device),
        'D1wide': PerVarConv(input_dim, max_len, args.time_dim, d1wide_h, args.kernel).to(device),
        'D2':     CrossVarConv(input_dim, max_len, args.time_dim, args.d_hidden, args.kernel, use_values=False).to(device),
        'D2shuf': CrossVarConvShuffled(input_dim, max_len, args.time_dim, args.d_hidden, args.kernel).to(device),
        'D3':     CrossVarConv(input_dim, max_len, args.time_dim, args.d_hidden, args.kernel, use_values=True).to(device),
    }
    param_counts = {k: count_params(v) for k, v in models.items()}
    print('[Audit] params: ' + '  '.join('%s=%d' % (k, param_counts[k]) for k in models))
    print('[Audit] D2=%d  D1wide(h=%d)=%d  (capacity-matched)'
          % (d2_params, d1wide_h, param_counts['D1wide']))
    for nm, mdl in models.items():
        train_model(mdl, train_loader, device, args.epochs, args.lr, nm)

    results = {}
    eval_splits = [('val', val_loader)]
    if args.evaluate_test:
        eval_splits.append(('test', test_loader))
    order = ['S1_static_var', 'S2_static_tv', 'S3_persistence',
             'S4', 'S4plus', 'D1', 'D1wide', 'D2', 'D2shuf', 'D3']
    label = {
        'S1_static_var': 'S1 static_var', 'S2_static_tv': 'S2 static_tv',
        'S3_persistence': 'S3 persistence', 'S4': 'S4 shallow-lin',
        'S4plus': 'S4+ shallow+t', 'D1': 'D1 per-var', 'D1wide': 'D1-wide',
        'D2': 'D2 cross-var', 'D2shuf': 'D2-shuffled', 'D3': 'D3 cross+val'}
    for split_name, loader in eval_splits:
        arrs = collect_arrays(loader, models, p_var, p_var_prod, p_time_var, max_len, device)
        prevalence = float(arrs['y'].mean())
        rows = [summarize(label[k], arrs, arrs['pi'][k], input_dim) for k in order]
        results[split_name] = dict(prevalence=prevalence, predictors=rows)
        print('\n===== %s (prevalence=%.4f) =====' % (split_name.upper(), prevalence))
        print('%-16s %8s %10s %8s %8s' % ('predictor', 'maAUPRC', 'trAUPRC', 'Brier', 'trBrier'))
        for r in rows:
            print('%-16s %8.4f %10.4f %8.4f %8.4f'
                  % (r['predictor'], r['macro_auprc'], r['transition_macro_auprc'],
                     r['brier'], r['transition_brier']))
        # Dump per-record validation scores for the patient bootstrap.
        if split_name == 'val' and args.dump_scores:
            dump = dict(
                y=arrs['y'].astype(np.int8), var_idx=arrs['var_idx'].astype(np.int16),
                m_prev=arrs['m_prev'].astype(np.int8),
                has_prev=arrs['has_prev'], record=arrs['record'])
            for k in PRIMARY_MODELS:
                dump['pi_' + k] = arrs['pi'][k].astype(np.float16)
            npz = os.path.join('export', 'audit',
                               'scores_%s_seed%d.npz' % (args.dataset, args.seed))
            os.makedirs(os.path.dirname(npz), exist_ok=True)
            np.savez_compressed(npz, **dump)
            print('[Audit] per-record val scores -> %s' % npz)

    # Per-run screening (frozen decision is in aggregate_audit.py).
    def by(pfx):
        return next(r for r in results['val']['predictors'] if r['predictor'].startswith(pfx))
    s4p, d1w, d2m, d2s = by('S4+'), by('D1-wide'), by('D2 cross'), by('D2-shuffled')
    ladder = dict(
        D2_minus_S4plus=d2m['macro_auprc'] - s4p['macro_auprc'],
        D2_minus_D1wide=d2m['macro_auprc'] - d1w['macro_auprc'],
        D2_minus_D2shuffled=d2m['macro_auprc'] - d2s['macro_auprc'],
        D2_minus_S4plus_transition=d2m['transition_macro_auprc'] - s4p['transition_macro_auprc'],
        D2_brier_minus_S4plus=d2m['brier'] - s4p['brier'],
        D2_transition_brier_minus_S4plus=d2m['transition_brier'] - s4p['transition_brier'],
    )
    print('\n===== D2 contrasts (validation) =====')
    for k, v in ladder.items():
        print('  %-34s %+.4f' % (k, v))

    results['protocol'] = dict(
        audit_version=4, dataset=args.dataset, split_seed=args.split_seed,
        model_seed=args.seed, batch_size=args.batch_size, epochs=args.epochs,
        lr=args.lr, d_hidden=args.d_hidden, kernel=args.kernel, time_dim=args.time_dim,
        input_dim=input_dim, max_len=max_len,
        train_size=len(train_ds), val_size=len(val_ds), test_size=len(test_ds),
        split_contracts=split_contracts, test_evaluated=args.evaluate_test,
        decision_split='val', strictly_causal=True, primary_model='D2',
        param_counts=param_counts, d1wide_hidden=d1wide_h,
        transition_definition='label z=1[m_t!=m_{t-1}] scored by q_flip=m_prev+pi-2*m_prev*pi',
        static_prior_padding_delta_mean=float(prior_abs_diff.mean()),
        static_prior_padding_delta_max=float(prior_abs_diff.max()))
    results['verdict'] = dict(decision_split='val', d2_contrasts=ladder,
                              note='Frozen paired decision + patient bootstrap in aggregate_audit.py.')

    out = args.output or os.path.join('export', 'audit', '%s_seed%d.json' % (args.dataset, args.seed))
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print('\n[Audit] results written to %s' % out)


if __name__ == '__main__':
    main()
