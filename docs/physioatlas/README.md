# PhysioAtlas v0.4

PhysioAtlas is a research-only extension inside the RuView fork for testing a
specific question:

> Which physiological waveforms or regional motions are recoverable from RF
> measurements, under which sensor geometry, subject, room, hardware, and
> movement conditions, and with what uncertainty?

It does not claim that commodity WiFi directly images internal organs. Every
session records the measured modality, privileged supervision, geometry,
clock domain, consent scope, quality mask, and claim boundary.

## Implemented system

```text
RuView CSI / UDP RF / replay adapters       ECG / PPG / belts / pose / ultrasound
                    │                                      │
                    └──── concurrent acquisition + clock calibration ────┘
                                           │
                         session manifest + geometry + consent
                                           │
                 gap-aware alignment + quality/observability masks
                                           │
      ┌───────────────────────┬─────────────┴──────────────┬──────────────────┐
      │                       │                            │                  │
masked RF pretraining   complex multi-link model   propagation graph   RF field tools
      │                       │                            │                  │
128-D embedding      waveform + uncertainty       delay constraints    CIR / delay-Doppler
      └───────────────────────┴─────────────┬──────────────┴──────────────────┘
                                           │
       held-out domains + simple baselines + nulls + conformal abstention
                                           │
        hashed checkpoints + audit log + dashboard + InnerLoop verifier
```

### Implemented integrations

- Concurrent acquisition registry with canonical NPZ replay, RuView JSONL
  replay, deterministic synthetic sources, and live timestamped UDP JSON.
- Affine clock offset/drift calibration and rigid 3-D geometry calibration.
- Versioned, hashed environment calibration profiles and profile selection.
- Explicit amplitude/phase, IQ, feature, and range-Doppler RF representations.
- Complex multi-link temporal modeling with geometry-conditioned link weights.
- Label-free masked RF reconstruction with 128-dimensional embeddings.
- Waveform, observability, and heteroscedastic uncertainty outputs.
- Split-conformal intervals and fail-closed abstention rules.
- Cardiac and respiratory propagation graphs with delay estimation and losses.
- Ultrasound frame-to-displacement extraction for regional-motion supervision.
- Privileged teacher-to-RF embedding distillation utilities.
- Subject, session, environment, posture, condition, and hardware holdouts.
- IFFT-based CIR baseline, delay-spread features, delay-Doppler maps, and an
  active-measurement scheduler.
- Consent validation, pseudonymized exports, hash-chain audit logs, and local
  clipped federated aggregation experiments.
- Atomic live state publication and a local research dashboard.
- Consent-first household workspace initialization, anonymous-first live tracks,
  and open-set identity for explicitly enrolled members.
- Five executable priority studies, rolling live track histories, per-modality
  calibration, a local household Research Hub, and safe queued actions.

## One-command verification

```bash
python -m pip install -r requirements-physioatlas.txt
python -m pip install -e . --no-deps --no-build-isolation
make physioatlas-ci
```

This runs environment diagnostics, compilation, all PhysioAtlas tests, a
complete deterministic smoke experiment, checkpoint replay, and fault
injection. See [TESTING.md](TESTING.md).

## First experiment

```bash
python -m physioatlas synthesize \
  --output data/physioatlas/synthetic \
  --subjects 4 --sessions-per-subject 2

python -m physioatlas validate \
  --data data/physioatlas/synthetic \
  --require-consent

python -m physioatlas pretrain-rf \
  --data data/physioatlas/synthetic \
  --output outputs/physioatlas/rf-pretraining \
  --epochs 2 --device cpu

python -m physioatlas run-config \
  configs/physioatlas/synthetic_ecg.yaml
```

## Live acquisition contract

List adapters:

```bash
python -m physioatlas list-adapters
```

Run a declarative acquisition:

```bash
python -m physioatlas acquire \
  configs/physioatlas/acquisition_synthetic.yaml
```

The generic `udp_json` adapter allows vendor or microcontroller bridges to send
frames without coupling PhysioAtlas to a proprietary SDK. See
[ACQUISITION.md](ACQUISITION.md) and [HARDWARE_SUPPORT.md](HARDWARE_SUPPORT.md).

## Household pilot

```bash
bash scripts/household-research-bootstrap.sh
python -m physioatlas init-household-research --help
```

Follow [HOUSEHOLD_PILOT_RUNBOOK.md](HOUSEHOLD_PILOT_RUNBOOK.md). Real household
identity and physiology must be evaluated on untouched sessions. Synthetic
identity texture tests only the software and open-set gates.

## Important boundaries

- CIR here is an auditable IFFT baseline. Zero padding does not create physical
  range resolution beyond the captured RF bandwidth.
- Ultrasound support extracts supervised displacement from supplied frame
  arrays. It does not segment organs or ingest proprietary scanner formats.
- Federated aggregation is a local research simulator, not cryptographic secure
  aggregation.
- No physical device, human participant, diagnostic endpoint, or clinical
  efficacy was validated in the packaged software environment.
