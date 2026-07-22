# SMILE Agent Guide

## Project scope

This repository contains the SMILE clinical time-series research code and the
AAAI manuscript workspace. The public model line is `smart-smile-lean`.

## Sources of truth

- Model and training code: `models/smart.py`, `main_pretrain.py`,
  `main_finetune.py`, and `run_all_experiments.py`.
- Paper source: `smile-paper/main1.tex`.
- Canonical paper artifact metadata: `smile-paper/CANONICAL_PAPER.md`.
- Active ODP redesign handoff: `docs/handoff-odp-redesign-2026-07-17.md`.

## Research and evaluation rules

- Do not claim that the audit identifies a causal MNAR mechanism. Use
  "structured/informative observation patterns" unless stronger evidence is
  provided.
- Never invent or interpolate experiment values. Keep validation model
  selection separate from held-out test evaluation.
- Standard experiment seeds are `1`, `42`, and `3407`.
- Preserve unrelated worktree changes and inspect `git status` before editing.
- The local `main` line and the ODP prototype history have diverged. Do not
  assume commit `9d05325d` is already present; inspect and port selectively.

## Verification

- Run focused unit tests for edited model/training paths.
- Run `python run_all_experiments.py --dry-run` for runner changes.
- The target experiment host has two visible RTX 4090 GPUs. Use single-node
  two-process DDP for one experiment, and preserve the matched global batch
  size across single- and dual-GPU comparisons.
- During architecture selection, use validation metrics and keep held-out test
  evaluation disabled. Run the held-out test only after the design is locked.
