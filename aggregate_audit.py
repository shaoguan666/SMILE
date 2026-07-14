"""Aggregate Observation Policy Audit v4 and apply the FROZEN paired decision.

Primary hypothesis is fixed to D2 (never chosen post hoc). For each dataset we
form per-seed PAIRED differences of validation macro-AUPRC and use a Student-t
interval with n-1 df (t_4=2.776 for 5 seeds), NOT a normal approximation.

A dataset SUPPORTS the claim iff ALL hold on validation:
  (1) D2 - S4+   mean >= --gain AND paired-t 95% CI lower bound > 0   (mask nowcast)
  (2) D2 - D1wide  paired-t 95% CI lower bound > 0                    (capacity control)
  (3) D2 - D2shuf  paired-t 95% CI lower bound > 0                    (structure control)
  (4) D2 - S4+   transition-event macro-AUPRC mean > 0 AND D2 transition-Brier
      <= S4+ transition-Brier (mean)                                  (correct transition task)
  (5) patient-level bootstrap of (D2 - S4+) macro-AUPRC on the seed-averaged
      predictions has a 95% percentile CI whose lower bound > 0

Overall = GO iff >= --min-datasets of exactly 4 datasets support. Config is
enforced: exactly 4 datasets, exactly --seeds unique model seeds each, identical
(epochs,batch_size,lr,d_hidden,kernel,time_dim), audit_version==4,
test_evaluated==False. Any mismatch aborts (no partial GO).

Usage:
    python aggregate_audit.py --datasets c12 c19 mimic_mortality mimic_decompensation
"""

import argparse
import glob
import json
import math
import os

import numpy as np
from sklearn.metrics import average_precision_score

_T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
           6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

_LABEL = {'S4+': 'S4+', 'D1-wide': 'D1-wide', 'D2 cross': 'D2 cross',
          'D2-shuffled': 'D2-shuffled'}


def _by(rows, pfx):
    return next(r for r in rows if r['predictor'].startswith(pfx))


def paired_t_ci(diffs):
    d = np.asarray(diffs, dtype=np.float64)
    n = len(d)
    mean = float(d.mean())
    if n < 2:
        return mean, 0.0, (mean, mean)
    sd = float(d.std(ddof=1))
    tcrit = _T_CRIT.get(n - 1, 1.96)
    half = tcrit * sd / math.sqrt(n)
    return mean, sd, (mean - half, mean + half)


def macro_auprc(y, p, var_idx, num_vars):
    scores = []
    for v in range(num_vars):
        mv = var_idx == v
        if not np.any(mv):
            continue
        yv = y[mv]
        if yv.sum() == 0 or yv.sum() == len(yv):
            continue
        scores.append(average_precision_score(yv, p[mv]))
    return float(np.mean(scores)) if scores else float('nan')


def patient_bootstrap(dataset, num_vars, n_boot=300, seed=12345):
    """Percentile CI of (D2 - S4+) macro-AUPRC over the seed-averaged predictions,
    resampling whole records (patients) with replacement."""
    files = sorted(glob.glob(os.path.join('export', 'audit',
                                           'scores_%s_seed*.npz' % dataset)))
    if not files:
        return None
    ref = np.load(files[0])
    y = ref['y'].astype(np.float64)
    var_idx = ref['var_idx'].astype(np.int64)
    record = ref['record'].astype(np.int64)
    # Average pi across seeds (identical position ordering: fixed split + sequential loader).
    pi_d2 = np.zeros_like(y)
    pi_s4 = np.zeros_like(y)
    for f in files:
        d = np.load(f)
        if not (np.array_equal(d['record'], ref['record'])
                and np.array_equal(d['var_idx'], ref['var_idx'])
                and np.array_equal(d['y'], ref['y'])):
            raise ValueError('score dump position mismatch across seeds for %s' % dataset)
        pi_d2 += d['pi_D2'].astype(np.float64)
        pi_s4 += d['pi_S4plus'].astype(np.float64)
    pi_d2 /= len(files)
    pi_s4 /= len(files)

    # Group positions by record for O(1) resampling.
    uniq = np.unique(record)
    order = np.argsort(record, kind='stable')
    rec_sorted = record[order]
    bounds = np.searchsorted(rec_sorted, uniq, side='left')
    bounds = np.append(bounds, len(rec_sorted))
    pos_by_rec = [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]

    rng = np.random.default_rng(seed)
    n_rec = len(uniq)
    diffs = np.empty(n_boot)
    import time as _time
    t0 = _time.time()
    print('    [boot] %s: %d records, %d positions, %d replicates ...'
          % (dataset, n_rec, len(y), n_boot), flush=True)
    for b in range(n_boot):
        pick = rng.integers(0, n_rec, size=n_rec)
        idx = np.concatenate([pos_by_rec[i] for i in pick])
        diffs[b] = (macro_auprc(y[idx], pi_d2[idx], var_idx[idx], num_vars)
                    - macro_auprc(y[idx], pi_s4[idx], var_idx[idx], num_vars))
        if (b + 1) % 50 == 0:
            print('    [boot] %s: %d/%d  (%.1fs)'
                  % (dataset, b + 1, n_boot, _time.time() - t0), flush=True)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return dict(mean=float(diffs.mean()), ci=(float(lo), float(hi)), n_boot=n_boot,
                n_records=int(n_rec), n_seeds=len(files))


def load_runs(dataset, seeds):
    runs = []
    for s in range(seeds):
        p = os.path.join('export', 'audit', '%s_seed%d.json' % (dataset, s))
        if not os.path.exists(p):
            raise ValueError('missing run: %s' % p)
        with open(p, 'r', encoding='utf-8') as f:
            runs.append(json.load(f))
    return runs


def enforce_config(dataset, runs, expected_seeds):
    protos = [r['protocol'] for r in runs]
    seeds = sorted(p['model_seed'] for p in protos)
    if seeds != list(range(expected_seeds)):
        raise ValueError('%s: model seeds %s != 0..%d' % (dataset, seeds, expected_seeds - 1))
    keys = ('epochs', 'batch_size', 'lr', 'd_hidden', 'kernel', 'time_dim')
    base = {k: protos[0][k] for k in keys}
    for p in protos:
        if any(p[k] != base[k] for k in keys):
            raise ValueError('%s: inconsistent config across seeds' % dataset)
        if p.get('audit_version') != 4:
            raise ValueError('%s: audit_version != 4' % dataset)
        if p.get('test_evaluated', False):
            raise ValueError('%s: a run evaluated test (sealed split violated)' % dataset)
        if p.get('primary_model') != 'D2':
            raise ValueError('%s: primary_model != D2' % dataset)
    return protos[0]['input_dim']


def summarize_dataset(dataset, gain_thr, seeds, do_bootstrap):
    runs = load_runs(dataset, seeds)
    input_dim = enforce_config(dataset, runs, seeds)
    d = {k: [] for k in ['D2_S4p', 'D2_D1w', 'D2_D2s', 'D2_S4p_tr',
                         'D2_brier_tr', 'S4p_brier_tr']}
    for r in runs:
        vr = r['val']['predictors']
        s4p, d1w, d2m, d2s = _by(vr, 'S4+'), _by(vr, 'D1-wide'), _by(vr, 'D2 cross'), _by(vr, 'D2-shuffled')
        d['D2_S4p'].append(d2m['macro_auprc'] - s4p['macro_auprc'])
        d['D2_D1w'].append(d2m['macro_auprc'] - d1w['macro_auprc'])
        d['D2_D2s'].append(d2m['macro_auprc'] - d2s['macro_auprc'])
        d['D2_S4p_tr'].append(d2m['transition_macro_auprc'] - s4p['transition_macro_auprc'])
        d['D2_brier_tr'].append(d2m['transition_brier'])
        d['S4p_brier_tr'].append(s4p['transition_brier'])

    g_mean, g_sd, g_ci = paired_t_ci(d['D2_S4p'])
    d1_mean, _, d1_ci = paired_t_ci(d['D2_D1w'])
    ds_mean, _, ds_ci = paired_t_ci(d['D2_D2s'])
    tr_mean, _, tr_ci = paired_t_ci(d['D2_S4p_tr'])
    tr_brier_ok = float(np.mean(d['D2_brier_tr'])) <= float(np.mean(d['S4p_brier_tr']))

    boot = patient_bootstrap(dataset, input_dim) if do_bootstrap else None

    cond1 = (g_mean >= gain_thr) and (g_ci[0] > 0.0)
    cond2 = d1_ci[0] > 0.0
    cond3 = ds_ci[0] > 0.0
    cond4 = (tr_mean > 0.0) and tr_brier_ok
    cond5 = (boot is not None) and (boot['ci'][0] > 0.0)
    supports = bool(cond1 and cond2 and cond3 and cond4 and cond5)

    return dict(
        dataset=dataset, n_seeds=seeds,
        D2_S4plus=dict(mean=g_mean, sd=g_sd, t_ci=g_ci),
        D2_D1wide=dict(mean=d1_mean, t_ci=d1_ci),
        D2_D2shuffled=dict(mean=ds_mean, t_ci=ds_ci),
        transition_D2_S4plus=dict(mean=tr_mean, t_ci=tr_ci, brier_ok=tr_brier_ok),
        patient_bootstrap=boot,
        conds=dict(gain=cond1, d1wide=cond2, d2shuf=cond3, transition=cond4, bootstrap=cond5),
        supports=supports)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+',
                    default=['c12', 'c19', 'mimic_mortality', 'mimic_decompensation'])
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--gain', type=float, default=0.02)
    ap.add_argument('--min-datasets', type=int, default=3)
    ap.add_argument('--no-bootstrap', dest='bootstrap', action='store_false', default=True)
    ap.add_argument('--output', default=os.path.join('export', 'audit', '_aggregate_v4.json'))
    args = ap.parse_args()

    if len(args.datasets) != 4:
        raise ValueError('Frozen decision requires exactly 4 datasets; got %d.'
                         % len(args.datasets))

    per = [summarize_dataset(ds, args.gain, args.seeds, args.bootstrap) for ds in args.datasets]

    print('%-22s %-24s %-20s %-20s %-16s %-18s %s'
          % ('dataset', 'D2-S4+ (t95 CI)', 'D2-D1wide CI', 'D2-D2shuf CI',
             'trans mean/Brier', 'patient-boot CI', 'SUPP'))
    for s in per:
        g, d1, ds, tr = (s['D2_S4plus'], s['D2_D1wide'], s['D2_D2shuffled'],
                         s['transition_D2_S4plus'])
        boot = s['patient_bootstrap']
        bstr = ('[%+.4f,%+.4f]' % boot['ci']) if boot else 'n/a'
        print('%-22s %+.4f[%+.3f,%+.3f]  [%+.3f,%+.3f]  [%+.3f,%+.3f]  %+.3f/%s  %-18s %s'
              % (s['dataset'], g['mean'], g['t_ci'][0], g['t_ci'][1],
                 d1['t_ci'][0], d1['t_ci'][1], ds['t_ci'][0], ds['t_ci'][1],
                 tr['mean'], 'Y' if tr['brier_ok'] else 'N', bstr,
                 'SUPPORT' if s['supports'] else 'no'))

    n_support = sum(1 for s in per if s['supports'])
    overall_go = n_support >= args.min_datasets
    print('\n===== FROZEN GO/NO-GO (D2 primary, paired t, patient bootstrap) =====')
    print('  datasets supporting: %d / %d  (need >= %d)'
          % (n_support, len(per), args.min_datasets))
    for s in per:
        c = s['conds']
        print('    %-22s support=%-3s [gain=%s d1wide=%s d2shuf=%s trans=%s boot=%s]'
              % (s['dataset'], 'yes' if s['supports'] else 'no',
                 *('Y' if c[k] else 'N' for k in ('gain', 'd1wide', 'd2shuf', 'transition', 'bootstrap'))))
    if overall_go:
        print('  DECISION: GO -- cross-variable co-observation carries predictive information '
              'beyond time, recency, run-length, and per-variable history. Proceed to D2 '
              'hidden-state FiLM injection (bounded residual as ablation; D3 not primary).')
    else:
        print('  DECISION: NO-GO. Do not write the cross-variable claim as established; '
              'reframe/inspect failing conditions before injection.')

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(dict(gain=args.gain, min_datasets=args.min_datasets,
                       n_support=n_support, overall_go=overall_go, per_dataset=per), f, indent=2)
    print('\n[Aggregate] written to %s' % args.output)


if __name__ == '__main__':
    main()
