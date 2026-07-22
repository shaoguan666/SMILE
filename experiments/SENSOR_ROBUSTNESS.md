# Manifest-based sensor-removal evaluation

This is the publication-oriented replacement for the exploratory
`recent_baselines/run_missing_sweep.py` harness. It evaluates frozen,
clean-validation-selected checkpoints and corrupts the test split only.

## Protocol

- MIMIC mortality and decompensation use 17 variables.
- Conditions remove `k = 0, 2, 3, 5, 7, 8` sensors per patient, corresponding
  to 0%, 11.8%, 17.6%, 29.4%, 41.2%, and 47.1%.
- A manifest fixes one sensor permutation for every patient and corruption
  replicate. Larger `k` values take longer prefixes of that permutation.
- Removed sensors have both values and observation-mask entries set to zero.
- Training and validation datasets are never corrupted.
- The standard training seeds are `1`, `42`, and `3407`; corruption replicates
  are independent and default to `0..4`.

## Commands

Generate or validate the two deterministic manifests:

```powershell
python experiments/generate_sensor_manifest.py
```

Inspect the formal job grid without running it:

```powershell
python experiments/run_sensor_robustness.py --dry-run
```

On the Linux experiment host, `--devices 0 1` starts one serial worker per
physical GPU. Jobs are pulled from a shared queue, so the faster GPU worker
immediately takes the next dataset/model/seed job. Each child process sees
exactly one device as CUDA device 0; this evaluation scheduler does not use
DDP and does not change the effective evaluation batch size.

```bash
nohup python -u experiments/run_sensor_robustness.py \
  --profile paper --devices 0 1 --resume \
  > export/sensor_robustness_v1/nohup-paper.log 2>&1 &
echo $!
```

The parent writes per-job logs below `export/sensor_robustness_v1/logs/`, an
atomic latest `execution_summary.json`, and immutable timestamped summaries
below `execution_summaries/`. Do not launch two independent orchestrators at
the same output root.

Run a bounded plumbing smoke test:

```powershell
python experiments/run_sensor_robustness.py `
  --datasets mimic_decompensation --seeds 42 `
  --ks 0 8 --replicates 0 --limit-test 128 `
  --allow-incomplete-smoke
```

Aggregate a complete formal grid and create the draft figure:

```powershell
python experiments/aggregate_sensor_robustness.py
python experiments/plot_sensor_robustness.py
```

The default plot is watermarked `EXPLORATORY`. Publication output requires:

```powershell
python experiments/plot_sensor_robustness.py --publication
```

Publication mode is blocked unless every planned run is complete and the clean
endpoints reconcile, after manuscript rounding, with
`sensor_robustness_clean_reference.json`. Do not bypass this gate for a paper
figure. `--allow-incomplete-smoke` and aggregator `--allow-incomplete` exist
only for plumbing checks.

## Artifacts

All new outputs live under `export/sensor_robustness_v1/`:

- `manifests/*.npz`: patient IDs, feature names, and fixed permutations;
- `runs/<dataset>/<model>/seed_<seed>/k_<k>/.../run.json`;
- paired `predictions.npz` files with patient ID, label, and score;
- `aggregate.json`, reconciliation status, and paired bootstrap intervals;
- PDF/PNG plots and per-model logs.

Checkpoint directories are read-only. A damaged or missing checkpoint fails
formal preflight; the known ISTS-PLM seed-1 corruption is therefore visible
before a full run begins.
