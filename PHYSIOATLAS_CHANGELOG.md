# PhysioAtlas implementation changelog

## 0.4.0

### Consent-first household research

- Added draft-consent household workspace initialization and explicit identity,
  live-tracking, dashboard, and participant-acknowledgement gates.
- Added anonymous-first multi-person tracking and open-set enrolled identity.
- Added disjoint enrollment/evaluation enforcement and unknown-person controls.
- Added derived identity-template disclosure without storing raw biometric streams
  in the household registry.

### Priority studies and live operations

- Added executable respiration/sleep, cardiac-mechanics, pulse-propagation,
  mobility/gait, and ultrasound-supervised regional-motion analyzers.
- Added localized JSONL replay and local UDP observation bridges.
- Added rolling per-track physiology, motion, quality, and observability history.
- Added per-member and per-modality calibration matrices and biological baseline
  comparison with explicit non-identity warnings.

### Research Hub and InnerLoop

- Added a local household Research Hub with live, people, calibration, studies,
  experiments, and privacy views.
- Added allow-listed action queuing and a safe local action worker.
- Added a Gemini/InnerLoop UI task, household bootstrap scripts, CI gates, a full
  pilot runbook, household smoke testing, and privacy fault injection.

## 0.3.0

### Acquisition and calibration

- Added concurrent declarative acquisition and adapter capability registry.
- Added canonical NPZ, RuView JSONL/RVCSI, synthetic, and live UDP adapters.
- Added affine clock drift/offset and rigid 3-D geometry calibration.
- Added hashed calibration profiles and environment/hardware profile matching.

### RF representations and learning

- Added explicit feature, amplitude/phase, IQ, and range-Doppler contracts.
- Added complex multi-link geometry-conditioned waveform architecture.
- Added masked label-free RF pretraining with 128-D embeddings.
- Added privileged teacher distillation utilities.
- Added bounded CIR, delay-spread, delay-Doppler, and active-sensing tools.

### Physiology and uncertainty

- Added cardiac/respiratory propagation graphs and timing analysis.
- Added ultrasound regional-displacement extraction.
- Added observability and heteroscedastic uncertainty heads.
- Added split-conformal intervals and fail-closed abstention.
- Added subject, session, environment, posture, condition, and hardware splits.

### Operations and governance

- Added consent audits, pseudonymized exports, hash-chain audit logs, and local
  clipped federated aggregation experiments.
- Added a local live-state dashboard and expanded InnerLoop CI.
- Expanded fault injection to eight integrity and scientific-contract failures.
- Expanded the PhysioAtlas test suite while preserving inherited RuView code.

## 0.2.0

- Added deterministic smoke replay, run verification, fail-closed fault tests,
  CI, and an InnerLoop bootstrap workflow.

## 0.1.0

- Added strict multimodal session schemas, RuView/CSV adapters, synchronized
  windows, a geometry-aware waveform baseline, subject holdouts, baselines,
  null controls, hashed checkpoints, synthetic data, and initial documentation.
