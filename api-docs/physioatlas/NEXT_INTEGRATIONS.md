# Integration status and remaining research work

## Implemented in v0.3

- synchronized multi-adapter orchestration and live UDP bridge
- clock and rigid geometry calibration
- versioned calibration memory
- complex multi-link encoding and label-free masked pretraining
- observability, heteroscedastic uncertainty, conformal intervals, abstention
- physiological propagation graphs
- ultrasound displacement supervision
- domain-held-out splits
- privileged teacher distillation utilities
- bounded CIR and delay-Doppler features
- active-measurement scheduling
- live dashboard state API
- consent, pseudonymized exports, audit logs, and local federated simulation

## Largest remaining integrations

### 1. Hardware-backed acquisition drivers

Build and test concrete bridges for the exact RuView firmware, a chosen mmWave
board, ECG/PPG hardware, respiratory belts, and a synchronization controller.
The v0.3 adapter interface is ready, but vendor SDKs and physical timing tests
cannot be validated without those devices.

### 2. A real repeated-measures dataset

Collect at least four participants, two days, two rooms, multiple postures, and
shared synchronization events. This is the first point where model results can
say anything about real physiological recoverability.

### 3. Stronger RF inverse methods

Compare the IFFT baseline against calibrated sparse CIR recovery, beamforming,
and multistatic field reconstruction. Resolution claims must be tied to actual
bandwidth and array aperture.

### 4. Deployment-grade uncertainty

Add environment-aware conformal calibration, ensembles, drift detectors, and a
formal OOD benchmark. Abstention should be evaluated as a primary endpoint.

### 5. Full teacher-student training studies

Train with pose, mmWave, ECG/PPG, belts, and ultrasound teachers, then remove
each teacher at inference and quantify what RF retained.

### 6. Secure distributed learning

Replace local aggregation with authenticated transport, secure aggregation,
participant deletion, attack tests, and an explicit privacy accounting method.

### 7. Clinical and regulatory validation

Only after robust real-data performance should the project define a clinical
endpoint, preregister evaluation, obtain ethics approval, and compare against
medical reference devices. No current package output is clinical evidence.
