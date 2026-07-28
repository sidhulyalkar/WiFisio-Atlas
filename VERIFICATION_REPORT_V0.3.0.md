# PhysioAtlas v0.3.0 verification report

## Result

- PhysioAtlas tests: **38 passed, 0 failed**
- Inherited Cargo invariant tests: **34 passed, 0 failed**
- Environment doctor: **ready**, 0 required failures
- Deterministic end-to-end smoke: **passed**
- Integration matrix: **passed**
- Checkpoint replay and SHA-256 verification: **passed**
- Fault injection: **8/8 detected**
- Editable installation with disabled build isolation: **passed**

## Reproducibility evidence

```text
Dataset SHA-256:              87f0b284ef2c09b91a9360712267498647d97d470c51f5a31be6846e27b57bd2
Run ID:                       2e1ce286e0757a49
Supervised checkpoint SHA-256:545d5856e8bacf88a7474f9f1df73c172ad63628e94dc434abd22e3d021f5d90
RF pretrain checkpoint SHA-256:86bb9dbd2143218ec3bc357fe107c6a3ebb1f8aa0901e94a03984e04188c001e
```

Two smoke runs with identical seed and configuration in different directories
produced identical dataset hashes, run IDs, supervised checkpoints, RF-pretrain
checkpoints, and final metrics.

## Faults detected

- `checkpoint_corruption`
- `non_monotonic_timestamps`
- `expired_consent`
- `excessive_clock_drift`
- `invalid_rf_representation`
- `audit_log_tamper`
- `calibration_profile_tamper`
- `invalid_active_sensing_candidates`

## Verified integration areas

Concurrent adapter acquisition, live UDP loopback, clock and rigid geometry
calibration, calibration memory, consent and privacy export, audit-chain
integrity, synthetic federated aggregation, dashboard state publication,
physiological propagation analysis, ultrasound displacement, masked RF
pretraining, complex-link waveform prediction, conformal uncertainty,
privileged distillation, delay-domain RF features, delay-Doppler maps, active
measurement selection, held-out-domain splits, null controls, and checkpoint
replay.

## Boundaries

No physical device or human-participant data was available in the execution
environment. The release therefore validates software contracts and generated
signal behavior only. It does not validate physiological accuracy, direct organ
visibility, blood pressure, diagnosis, safety, or clinical utility.
