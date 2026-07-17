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
