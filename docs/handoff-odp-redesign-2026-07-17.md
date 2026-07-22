# ODP Model Redesign Handoff (2026-07-17)

## Objective

Give Observation Dynamics Prediction (ODP) one bounded redesign iteration for
the AAAI submission. Do not keep adding modules if the redesigned ODP remains
dominated by simple density or shuffled-history controls.

## Repository state

- Workspace: `D:\实验室\mamba-cde\SMILE`
- Local checked-out branch at handoff: `main`
- Local HEAD at handoff: `f389c3e`
- ODP prototype commit: `9d05325d340504469e13b2057a58fd178611ccdf`
- The ODP commit is not an ancestor of local `main`.
- It is reachable from `codex/backup-policy-20260715` and `origin/main`.
- Treat the commit as reference code. Do not cherry-pick it wholesale without
  checking conflicts and changes that local `main` already supersedes.

The prototype implements:

- `CausalD2PolicyEncoder`: predicts the natural observation mask at time `t`
  from strictly earlier mask history `M_{<t}`.
- `causal_policy_bce`: valid-position binary mask prediction loss.
- `PolicyFiLM`: token-level conditioning of the clinical encoder.
- causality, padding, checkpoint, gradient-isolation, and runner tests in
  `tests/test_smile_lean_policy.py`.

The prototype is a discrete-grid, one-step observation-mask predictor. It does
not predict continuous time to the next measurement or a multi-step horizon.
It also retains the original CoMiss pathway, so it is not yet a clean CoMiss
replacement.

## Target execution host

The implementation and experiments will be transferred to this Linux host:

- host prompt: `jy@jy-x11dpi-n-2`;
- working area shown by the user: `/data/sgc` (discover the actual repository
  root with `pwd` and `git rev-parse --show-toplevel`; do not assume whether the
  checkout is `/data/sgc` or `/data/sgc/SMILE`);
- GPUs: two NVIDIA GeForce RTX 4090 cards, 24 GB each, device IDs `0` and `1`;
- NVIDIA driver: `535.309.01`;
- driver-reported CUDA compatibility: `12.2`.

Run one experiment across both GPUs with single-node, two-process DDP via
`python -m torch.distributed.run`. Do not interpret "dual GPU" as launching two
uncoordinated single-GPU copies. Verify the installed PyTorch build and NCCL
before training; the CUDA version bundled with PyTorch need not exactly equal
the version displayed by `nvidia-smi`, but it must be supported by the driver.

The current runner already exposes `--use-torchrun`, `--nproc-per-node`,
`--master-port-base`, and `--devices`. It also defaults to conservative NCCL
settings through `SMART_SAFE_NCCL=1`, including disabling P2P and InfiniBand.
Do not remove those defaults during the first smoke test. If performance
tuning later enables P2P, verify the host topology and stability first.

### DDP correctness requirement

The current runner describes `--batch-size` as per-GPU and also hard-codes
Lean models to `64`. With two DDP processes this changes the effective global
batch from the historical single-GPU value `64` to `128`, which invalidates a
strict ablation comparison. Fix and test this before experiments:

- make batch-size semantics explicit and consistent;
- for the matched experiments, preserve global batch `64`, hence use batch
  `32` per rank with two GPUs, unless gradient accumulation is explicitly
  documented and produces the same effective batch;
- print per-rank batch, world size, and effective global batch in every dry
  run and real run;
- ensure only rank 0 writes checkpoints, logs, and result JSON files;
- retain `DistributedSampler` plus `set_epoch` for training;
- make ODP forecasting metrics globally correct under DDP rather than
  reporting one rank's shard;
- verify checkpoint save/load across DDP and non-DDP `module.` prefixes;
- port or add a runner-level `--skip-test` path, because the checked-out local
  `main` does not currently expose that flag even though the prototype history
  and prior screening used validation-only runs;
- use a new export root so existing results are never overwritten.

## Existing screening evidence

The following values are best validation AUPRC from the existing logs, not
held-out test results:

| Dataset | SMILE-Lean | ODP hidden | Density control | Shuffled control |
|---|---:|---:|---:|---:|
| C12 | 56.53 +/- 1.46 | 56.65 +/- 0.62 (n=2 complete) | 56.95 +/- 0.30 | 57.49 +/- 2.39 |
| C19 | 78.68 +/- 0.21 | 79.56 +/- 0.11 (n=2 complete) | 80.41 +/- 0.18 | 79.90 +/- 0.88 |
| MIMIC mortality | 51.40 +/- 1.54 | 50.32 +/- 0.62 | 51.51 +/- 0.45 | 50.92 +/- 0.83 |
| MIMIC decompensation | 75.87 +/- 1.10 | 74.13 +/- 1.29 | 74.68 +/- 0.35 | 75.00 +/- 1.68 |

Policy BCE is lower than its shuffled counterpart on all four tasks:

| Dataset | ODP BCE | Shuffled BCE |
|---|---:|---:|
| C12 | 0.198 | 0.221 |
| C19 | 0.173 | 0.187 |
| MIMIC mortality | 0.226 | 0.245 |
| MIMIC decompensation | 0.256 | 0.275 |

Interpretation: the causal branch learns some cross-variable observation
predictability, but the learned hidden representation does not transfer
reliably to the clinical tasks. Density and shuffled controls dominate the ODP
hidden representation in the current downstream screening. This is a signal
alignment or fusion problem, not evidence that more architectural complexity
is automatically justified.

Most policy runs used `--skip-test`. One C12 full-policy seed was incomplete,
and one C19 full-policy seed produced no usable downstream metric. Preserve the
held-out test during redesign; do not repeatedly inspect it.

## Required redesign

Implement only this bounded redesign before deciding whether to abandon ODP:

1. Start from local `main` and create a dedicated `codex/odp-late-fusion`
   branch after inspecting `git status`.
2. Port only the causal policy encoder, policy loss, necessary CLI/runner
   wiring, and focused tests from `9d05325d`.
3. Make ODP a clean CoMiss replacement in the new variants: disable the
   CoMiss bias path while preserving the ordinary SMART/SMILE backbone pieces
   needed for a fair comparison.
4. Remove token-wise `PolicyFiLM` from the experimental ODP variant. Pool the
   causal policy hidden state and use zero-initialized late residual fusion:

   `z_fused = z_clinical + alpha * W(pool(h_policy))`, with `alpha = 0` at
   initialization.

   Fuse at the patient/CLS representation rather than perturbing every
   variable-time token.
5. During fine-tuning, train with

   `L = L_task + lambda_policy * L_policy`

   so the policy encoder can adapt to the task without discarding observation
   prediction. Do not use the old default combination of a frozen policy
   encoder and token-level FiLM. Start with one documented lambda rather than a
   large search; use a small validation-only sensitivity check only if needed.
6. Log ODP quality separately with macro per-variable AUPRC and Brier score.
   Raw BCE alone is insufficient under mask imbalance.
7. Add prediction baselines that separate dynamics from marginal frequency:
   per-variable prior, time-only, density plus own-variable history, and
   shuffled cross-variable history.

Do not add multi-horizon prediction, continuous-time hazard heads, dynamic
graphs, or new policy tokens in this iteration. Those are allowed only if the
clean late-fusion design first passes the go/no-go criteria.

## Experiment sequence

1. Run focused unit tests for strict causality, padding, no future leakage,
   zero-initialized equivalence, fusion gradients, checkpoint round-trip, and
   runner flag propagation. Add tests for two-process launch construction and
   matched effective global batch size.
2. On the target Linux host, verify that PyTorch sees both 4090s, then run a
   short two-process DDP smoke test. Run runner dry-runs and report the exact
   generated commands before starting full training.
3. Screen seed `42` on C19 (previously favorable) and MIMIC decompensation
   (previously unfavorable), using validation only and `--skip-test`.
4. If both pilots are credible, run seeds `1`, `42`, and `3407` on all four
   binary tasks.
5. Lock the design before running held-out test evaluation.

Required clean downstream ladder:

- matched backbone without CoMiss or ODP;
- density control;
- ODP prediction loss without downstream fusion, if it actually shares useful
  parameters with the clinical encoder;
- ODP plus late gated fusion;
- shuffled-history ODP plus the same late fusion;
- parameter-matched non-ODP control.

## Go/no-go criteria

Continue ODP as a core paper contribution only if:

- ODP forecasting macro-AUPRC clearly exceeds the strongest simple
  time/density/own-history baseline on most datasets;
- late-fusion ODP beats both density and shuffled controls on at least three of
  four binary validation tasks;
- paired seed directions are reasonably consistent;
- the two MIMIC tasks no longer show the existing one-to-two AUPRC degradation;
- all headline comparisons have three complete seeds before held-out testing.

If these conditions fail, stop model surgery. Do not add multi-horizon or graph
complexity to chase a positive result. Remove ODP from the main contribution,
retain it only as an honest negative diagnostic if useful, and report the
decision explicitly.

## Constraints

- Do not edit `smile-paper/main1.tex` during this implementation task.
- Do not fabricate, infer, or copy metrics from a different split/protocol.
- Do not evaluate held-out test data during architecture selection.
- Do not claim causal clinician policy or identified MNAR mechanisms.
- Preserve unrelated user changes and avoid destructive Git operations.
- Do not silently change the public `smart-smile-lean` model. Add explicit
  experimental model IDs and flags.

## Deliverables

- code changes and focused tests;
- exact model IDs and copy-paste Linux CLI commands for the target two-4090
  host, including environment check, dry run, DDP smoke test, both seed-42
  pilots, gated full validation run, resume/evaluate commands, and monitoring;
- every training command must visibly use devices `0,1`, two DDP processes,
  global batch `64`, a collision-safe master port, `--skip-test` during model
  selection, and a fresh export root;
- parameter counts for every clean control;
- validation-only pilot table with run completeness noted;
- policy forecasting metrics and downstream metrics kept separate;
- a written go/no-go decision based on the criteria above;
- no manuscript edits until the model design is locked.
