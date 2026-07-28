# PhysioAtlas v0.4.0 verification report

## Result

- PhysioAtlas tests: **49 passed, 0 failed**
- Environment doctor: **ready**, 0 required failures
- Deterministic end-to-end waveform smoke: **passed**
- Checkpoint replay and SHA-256 verification: **passed**
- Household enrollment/tracking/Research Hub smoke: **passed**
- Enrollment/evaluation session overlap gate: **passed**
- Unknown synthetic visitor remained anonymous: **passed**
- Standard fault injection: **8/8 detected**
- Household/privacy fault injection: **3/3 detected**
- Editable installation: **passed**
- YAML, Bash, and Research Hub JavaScript validation: **passed**

## Reproducibility evidence

```text
Dataset SHA-256:    d725e195f93c7dc62ea001c8ea9e2f1ea93f18bbc8354e78b52c6a1339dd6da6
Run ID:             40a8315870421c14
Checkpoint SHA-256: 4e6f414b3dcbd020f8632ff2e7fc2f3baf793d9a8f9ae46c5dca63e3668f2867
```

Identical seeds in separate output directories reproduced the dataset hash, run
ID, checkpoint, metrics, and stable household scientific payload.

## Synthetic household gate

The deterministic software pilot enrolled four synthetic members using
calibration sessions and evaluated one disjoint held-out session per member.
Accepted accuracy and coverage were both 1.0, and an injected unknown track
remained anonymous. These values are **not evidence that real family members are
separable from home RF**. Synthetic identity textures were added solely to test
the software and open-set decision path.

## Priority studies

- `respiration_sleep`: infrastructure **passed**; exploratory scientific gate **passed**
- `cardiac_mechanics`: infrastructure **passed**; exploratory scientific gate **passed**
- `pulse_propagation`: infrastructure **passed**; exploratory scientific gate **passed**
- `mobility_gait`: infrastructure **passed**; exploratory scientific gate **not_evaluable**
- `organ_correlated_motion`: infrastructure **passed**; exploratory scientific gate **passed**

Mobility/gait intentionally remains `not_evaluable` until an independent pose,
motion-capture, instrumented-walkway, or equivalent reference is supplied. The
pulse-propagation null was strengthened from waveform reversal to event-level
independent timing jitter because reversal is not a valid hard null for periodic
signals.

## Operational surfaces verified

- Draft household initialization leaves identity, live tracking, dashboard, and
  participant acknowledgement disabled.
- JSONL replay accepted localized observations and preserved anonymous tracks.
- Local UDP loopback accepted validated observations in the test suite.
- Research Hub health, state, calibration, tracks, members, and static UI were
  served over localhost.
- The action worker accepts only versioned allow-listed research actions and
  rejects unsafe requests.
- All five study reports are emitted as machine-readable JSON.

## Faults detected

- `checkpoint_corruption`
- `non_monotonic_timestamps`
- `expired_consent`
- `excessive_clock_drift`
- `invalid_rf_representation`
- `audit_log_tamper`
- `calibration_profile_tamper`
- `invalid_active_sensing_candidates`
- `revoked_identity_consent`
- `out_of_distribution_identity`
- `unsafe_dashboard_action`

## Rust verification boundary

`cargo` was not installed in the v0.4 verification container, so inherited Rust
tests were **not rerun** for this release. The v0.4 implementation changes are
confined to Python, configuration, UI, scripts, tests, and documentation. The
prior v0.3 report recorded 34 inherited Cargo invariant tests passing, but that
historical result is not counted as fresh v0.4 execution evidence.

## Scientific and deployment boundaries

No physical home CSI, mmWave, ECG, PPG, respiratory belt, depth, ultrasound, or
human-participant hardware was available. The release validates code paths,
contracts, generated signals, open-set privacy behavior, and local UI operation
only.

A normal home router may not expose CSI. Reliable multi-person household studies
require compatible sensing hardware and an upstream localization/separation
source. The release does not validate real household identity, unique biological
signatures, diagnosis, emergency use, blood pressure, direct organ visibility,
medical safety, or clinical utility.
