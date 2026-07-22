# Recent baseline adapters

This directory connects source-released 2025--2026 baselines to SMILE's data
pipeline without treating source-paper numbers as local reproductions. Source
availability and exact upstream revisions are tracked in
`external/BASELINE_STATUS.md`.

## ATENet on MIMIC-III

The upstream ATENet repository supports P12, P19, and PAM. The local adapter
retains its model and three auxiliary objectives, but supplies SMILE's MIMIC
splits, feature normalization, value/mask inputs, and AUROC/AUPRC evaluator.
The base revision is `dc3149c`; two local changes in `models.py` make random
masks device-aware and attention masking safe under AMP. Both are execution
compatibility patches, not objective changes.

```powershell
python experiments\recent_baselines\run_atenet_mimic.py `
  --dataset mimic_mortality --epochs 10 --patience 3 `
  --batch-size 256 --seed 42

python experiments\recent_baselines\run_atenet_mimic.py `
  --dataset mimic_decompensation --epochs 10 --patience 3 `
  --batch-size 256 --seed 42
```

Checkpoints and JSON histories are written below
`export/recent_baselines/<dataset>/atenet/seed_<seed>/`. The adapter selects the
checkpoint with the highest validation AUPRC. Non-finite AMP batches are
explicitly skipped and counted in each epoch record.

The reported seeds 1, 42, and 3407 use full-precision training and contain zero
skipped batches. `--amp` remains available for faster exploratory runs.

After completing seeds 1, 42, and 3407, regenerate the mean/std summaries with:

```powershell
python experiments\recent_baselines\aggregate_atenet_results.py
```

Run additional seeds before presenting these results as matched-protocol
mean/std estimates.

## WaveGNN, MissTSM, and ISTS-PLM on missing MIMIC tasks

The source papers already report C12 and C19 classification, so those values
are kept in the manuscript's daggered source-reported block. WaveGNN also
reports MIMIC mortality AUPRC, but not AUROC. The adapters below cover the
classification tasks/metrics absent from the source papers using SMILE's fixed
MIMIC splits and validation-AUPRC checkpoint selection.

```powershell
python experiments\recent_baselines\run_wavegnn_mimic.py `
  --dataset mimic_decompensation --seed 42

python experiments\recent_baselines\run_misstm_mimic.py `
  --dataset mimic_mortality --seed 42

python experiments\recent_baselines\run_ists_plm_mimic.py `
  --dataset mimic_mortality --seed 42 --batch-size 1
```

Each adapter supports both `mimic_mortality` and `mimic_decompensation`, plus
`--limit-train`, `--limit-val`, and `--limit-test` for smoke tests. Outputs are
written to `export/recent_baselines/<dataset>/<model>/seed_<seed>/`.

On the target two-GPU host, first inspect the complete 18-job queue:

```powershell
python experiments\recent_baselines\run_recent_mimic_baselines.py --dry-run
```

Then run it. The upstream baselines are single-process implementations, so the
queue assigns independent model/seed jobs one per GPU rather than altering the
models with an unvalidated DDP port:

```powershell
python experiments\recent_baselines\run_recent_mimic_baselines.py --gpus 0 1
```

To run only the tasks absent from every source paper, use three calls:

```powershell
python experiments\recent_baselines\run_recent_mimic_baselines.py `
  --models wavegnn --datasets mimic_decompensation --gpus 0 1
python experiments\recent_baselines\run_recent_mimic_baselines.py `
  --models misstm ists_plm --datasets mimic_mortality mimic_decompensation `
  --gpus 0 1
```

The equivalent single-queue shorthand is `--source-missing-only`.

After all three seeds of both MIMIC tasks finish for a model, aggregate them:

```powershell
python experiments\recent_baselines\aggregate_recent_results.py --model wavegnn
python experiments\recent_baselines\aggregate_recent_results.py --model misstm
python experiments\recent_baselines\aggregate_recent_results.py --model ists_plm
```

The current laptop has one 8 GB RTX 4060. Use `--local-8gb --gpus 0` only for
diagnostics; a complete queue on that hardware is not a practical paper run.
