# PhysioAtlas v0.3 suggested commit series

The complete ZIP can be committed as one release, but the following sequence
keeps review boundaries crisp.

## 1. `feat(physioatlas-schema): expand multimodal, clock, consent, and RF contracts`

Stage `physioatlas/schema.py`, `physioatlas/config.py`, and schema tests.

## 2. `feat(physioatlas-acquisition): add concurrent adapters and live UDP bridge`

Stage acquisition, adapters, I/O, synchronization, calibration, calibration
memory, acquisition configs, and corresponding tests.

## 3. `feat(physioatlas-data): add gap-aware complex multi-link window datasets`

Stage dataset construction, domain metadata, split strategies, and tests.

## 4. `feat(physioatlas-model): add complex-link prediction and masked RF pretraining`

Stage model, training, pretraining, uncertainty outputs, and model tests.

## 5. `feat(physioatlas-physiology): add propagation graphs and ultrasound supervision`

Stage physiology, ultrasound, privileged distillation, and tests.

## 6. `feat(physioatlas-rf-field): add CIR, delay-Doppler, and active sensing baselines`

Stage RF-field tools, public CLI commands, and tests.

## 7. `feat(physioatlas-evaluation): add domain holdouts, conformal abstention, and nulls`

Stage metrics, uncertainty calibration, experiment execution, verification, and
fault injection.

## 8. `feat(physioatlas-governance): add consent, privacy export, audit, and federated scaffolds`

Stage privacy, federated, dashboard, and governance tests.

## 9. `ci(physioatlas): expand InnerLoop gates and deterministic smoke validation`

Stage Makefile targets, scripts, workflow, synthetic cohorts, and all tests.

## 10. `docs(physioatlas): document v0.3 operation, limits, and hardware roadmap`

Stage `docs/physioatlas`, root guides, configs, changelog, and verification
reports.
