# InnerLoop research guide for PhysioAtlas v0.4

## Complete software gate

Linux, macOS, or WSL:

```bash
bash scripts/innerloop-bootstrap.sh
bash scripts/household-research-bootstrap.sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/innerloop-bootstrap.ps1
powershell -ExecutionPolicy Bypass -File scripts/household-research-bootstrap.ps1
```

For an installed environment:

```bash
make physioatlas-ci
```

## Household Research Hub task

Give Gemini or another InnerLoop agent the task in
[`INNERLOOP_GEMINI_RESEARCH_HUB_TASK.md`](INNERLOOP_GEMINI_RESEARCH_HUB_TASK.md).
The agent may iterate on UI and versioned APIs, but it must not weaken consent,
open-set identity, unknown-person handling, observability, or research claim
boundaries.

## Agent protocol

1. Run the full gate before changing code or configs.
2. State one falsifiable hypothesis and one claim boundary.
3. Use a committed config and change one factor at a time.
4. Validate consent, timestamps, representations, geometry, and calibration.
5. Keep enrollment sessions disjoint from identity evaluation sessions.
6. Create anonymous tracks before attempting enrolled identity.
7. Select a split that holds out the deployment domain being claimed.
8. Compare learned models with simple baselines and required nulls.
9. Inspect observability, interval coverage, abstention, and unknown-person errors.
10. Verify checkpoints and preserve complete run and study directories.
11. Treat regressions and negative results as evidence.
12. Never infer anatomical visibility, diagnosis, or clinical utility from
    synthetic data or an unvalidated household pilot.

## Useful commands

```bash
python -m physioatlas doctor --repo .
python -m physioatlas list-adapters
python -m physioatlas list-priority-studies
python -m physioatlas init-household-research --help
python -m physioatlas household-enroll --help
python -m physioatlas household-evaluate --help
python -m physioatlas household-replay-jsonl --help
python -m physioatlas household-listen-udp --help
python -m physioatlas serve-research-hub --help
python -m physioatlas research-hub-worker --help
python -m physioatlas run-priority-studies --help
```

The operational runbook is
[`docs/physioatlas/HOUSEHOLD_PILOT_RUNBOOK.md`](docs/physioatlas/HOUSEHOLD_PILOT_RUNBOOK.md).
