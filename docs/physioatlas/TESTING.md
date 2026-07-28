# Testing and InnerLoop gates

## Complete local gate

```bash
make physioatlas-ci
```

It runs:

1. Python compilation and CLI discovery
2. environment and repository doctor
3. all PhysioAtlas unit, contract, UDP loopback, model, and integration tests
4. deterministic synthetic cohort creation
5. consent and stream validation
6. concurrent acquisition and calibration checks
7. masked RF pretraining
8. RF field, active sensing, propagation, ultrasound, privacy, federated,
   calibration-memory, distillation, and uncertainty checks
9. waveform training with subject-disjoint evaluation
10. mean/ridge baselines and time-shift/label-permutation nulls
11. checkpoint hash and actual forward-pass replay
12. fault injection

## Faults that must be detected

- corrupted checkpoint
- duplicate/non-monotonic timestamps
- expired consent
- excessive clock drift
- invalid complex RF feature width
- audit-log tampering
- calibration-profile tampering
- non-finite active-sensing candidates

## Evidence files

```text
outputs/physioatlas/innerloop-smoke/smoke-report.json
outputs/physioatlas/innerloop-smoke/run/experiment.json
outputs/physioatlas/innerloop-smoke/run/neural/run.json
outputs/physioatlas/innerloop-smoke/integrations/rf-pretraining/run.json
outputs/physioatlas/fault-test/fault-report.json
```

## Interpretation

A passing smoke test proves software integration, determinism, and fail-closed
behavior on generated data. It does not prove RF measurement validity,
physiological accuracy, organ localization, or safety.
