# Priority research study implementations

## Respiration and sleep mechanics

Inputs: WiFi/mmWave plus respiratory belt reference. Outputs include respiratory
rate, waveform correlation, and pause fraction. A future sleep study must compare
against polysomnography and report false alerts per night.

## Cardiac mechanics

Inputs: preferably mmWave plus ECG or PPG. Outputs include rate, waveform
correlation, and electrical-to-mechanical timing. This does not diagnose rhythm
conditions.

## Pulse propagation

Inputs: synchronized ECG and one or more PPG sites. Outputs include pulse-arrival
delay and correlation. This is not a blood-pressure estimate unless separately
validated under a defined population and calibration procedure.

## Mobility and gait

Inputs: pose, mmWave, or CSI motion features. Outputs include motion intensity,
periodic cadence candidates, active fraction, and transition count. Instrumented
gait or motion capture should be used as reference.

## Organ-correlated motion

Inputs: RF plus synchronized ultrasound displacement. A ridge baseline predicts a
held-out displacement trace and is compared with a shifted-label null. The output
is a regional-motion correlation, never a generated organ image.

Run all studies:

```bash
python -m physioatlas run-priority-studies --data DATASET --output OUTPUT
```

Run one declarative protocol:

```bash
python -m physioatlas run-priority-study \
  configs/physioatlas/household/study-respiration_sleep.yaml
```
