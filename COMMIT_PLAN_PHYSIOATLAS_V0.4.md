# PhysioAtlas v0.4 suggested commit sequence

Apply these commits on a branch such as `physioatlas/v0.4-household-research`.
The sequence keeps privacy contracts ahead of identity, and scientific methods
ahead of UI presentation.

## 1. `feat(privacy): add explicit household identity and live-tracking consent gates`

Add identity-enrollment, live-tracking, household-dashboard, and participant-
acknowledgement fields to consent policies. Reject household uses unless each
permission is explicitly granted. Keep synthetic consent fixtures explicit and
local-only.

```bash
git add physioatlas/privacy.py physioatlas/synthetic.py
git commit -m "feat(privacy): add explicit household identity and live-tracking consent gates"
```

## 2. `feat(household): add consent-gated open-set enrollment and held-out evaluation`

Add per-modality identity templates, biological baseline summaries, open-set
matching, unknown-person rejection, calibration-session provenance, disjoint
held-out evaluation, coverage reporting, and biological-distinctiveness reports
that cannot be used as sole identity evidence.

```bash
git add physioatlas/household.py tests/physioatlas/test_household.py
git commit -m "feat(household): add consent-gated open-set enrollment and held-out evaluation"
```

## 3. `feat(tracking): add anonymous-first multiperson tracking and live bridges`

Add position/embedding association, track lifecycle management, identity only
after repeated evidence, rolling physiology, JSONL replay, local UDP input,
open-set anonymous handling, and deterministic household simulation.

```bash
git add physioatlas/tracking.py physioatlas/household_live.py \
  configs/physioatlas/household/observation.example.jsonl \
  tests/physioatlas/test_tracking_household.py \
  tests/physioatlas/test_household_operational.py
git commit -m "feat(tracking): add anonymous-first multiperson tracking and live bridges"
```

## 4. `feat(studies): implement five priority physiological research protocols`

Add respiration/sleep mechanics, cardiac mechanics, event-level pulse
propagation, mobility/gait, and ultrasound-supervised regional-motion analyzers.
Include reference requirements, held-out prediction, exploratory scientific
gates, shift/jitter nulls, machine-readable reports, and all-study execution.

```bash
git add physioatlas/studies.py tests/physioatlas/test_priority_studies.py \
  configs/physioatlas/household/study-*.yaml
git commit -m "feat(studies): implement five priority physiological research protocols"
```

## 5. `feat(hub): add local household research hub and safe action worker`

Add file-backed local APIs and a no-CDN UI for live tracks, rolling physiology,
member baselines, modality calibration, study results, experiments, consent, and
provenance. Queue only allow-listed actions and process them without arbitrary
shell execution.

```bash
git add physioatlas/research_hub.py physioatlas/research_hub_worker.py \
  physioatlas/static/research_hub/index.html \
  tests/physioatlas/test_research_hub.py
git commit -m "feat(hub): add local household research hub and safe action worker"
```

## 6. `feat(protocol): add consent-first household workspace and home pilot configs`

Add draft-consent workspace creation, household aliases/configuration, hardware
boundaries, negative controls, local bootstrap scripts, and InnerLoop experiment
configuration.

```bash
git add physioatlas/household_protocol.py \
  configs/physioatlas/household/home-network.example.yaml \
  scripts/household-research-bootstrap.sh \
  scripts/household-research-bootstrap.ps1 \
  innerloop/physioatlas_household_research.yaml
git commit -m "feat(protocol): add consent-first household workspace and home pilot configs"
```

## 7. `feat(cli): expose household enrollment, tracking, studies, and hub commands`

Expose household initialization, enrollment, held-out evaluation, JSONL replay,
UDP listening, simulation, study execution, hub serving, action processing,
smoke testing, and privacy fault testing through the public CLI. Package the hub
UI with editable and wheel installs.

```bash
git add physioatlas/cli.py physioatlas/__init__.py physioatlas/acquisition.py \
  pyproject.toml
git commit -m "feat(cli): expose household enrollment, tracking, studies, and hub commands"
```

## 8. `test(household): add deterministic integration and fail-closed privacy gates`

Add complete household smoke testing, revoked-consent rejection, unknown-person
rejection, unsafe-action rejection, disjoint-split regression, UDP loopback, and
integration-module doctor checks. Add all gates to Make, scripts, and CI.

```bash
git add physioatlas/household_verification.py physioatlas/verification.py \
  tests/physioatlas/test_household*.py tests/physioatlas/test_tracking_household.py \
  Makefile scripts/innerloop-bootstrap.sh scripts/innerloop-validate.sh \
  scripts/innerloop-validate.ps1 .github/workflows/physioatlas-ci.yml
git commit -m "test(household): add deterministic integration and fail-closed privacy gates"
```

## 9. `docs(physioatlas): add household pilot, study, hub, and Gemini runbooks`

Document hardware requirements, calibration blocks, untouched validation,
observation contracts, each priority study, go/no-go gates, Research Hub APIs,
and the exact InnerLoop/Gemini UI task. Update project goals and changelogs.

```bash
git add README.md INNERLOOP.md INNERLOOP_GEMINI_RESEARCH_HUB_TASK.md \
  PROJECT_GOALS_PHYSIOATLAS.md PHYSIOATLAS_CHANGELOG.md CHANGELOG.md \
  docs/physioatlas/README.md docs/physioatlas/HOUSEHOLD_RESEARCH.md \
  docs/physioatlas/HOUSEHOLD_PILOT_RUNBOOK.md \
  docs/physioatlas/PRIORITY_STUDIES.md docs/physioatlas/RESEARCH_HUB.md
git commit -m "docs(physioatlas): add household pilot, study, hub, and Gemini runbooks"
```

## 10. `chore(release): add v0.4 verification evidence`

Commit the release verification reports after reproducing the commands on your
machine. Do not edit measured results to make them more favorable.

```bash
git add VERIFICATION_REPORT_V0.4.0.md VERIFICATION_REPORT_V0.4.0.json \
  COMMIT_PLAN_PHYSIOATLAS_V0.4.md
git commit -m "chore(release): add PhysioAtlas v0.4 verification evidence"
```
