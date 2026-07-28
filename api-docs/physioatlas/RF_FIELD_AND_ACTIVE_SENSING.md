# RF field features and active sensing

## Delay-domain baseline

`reconstruct_cir` applies an optional Hann taper and IFFT across complex CSI
subcarriers. It supports delay-spread, dominant-tap-ratio, and coarse tap-count
features. It is explicitly not compressed-sensing super-resolution.

`delay_doppler_map` applies a temporal FFT to the delay response and returns a
magnitude map suitable for inspectable baselines.

```bash
python -m physioatlas analyze-rf-field \
  --input session/wifi_csi.npz \
  --representation amplitude_phase \
  --bandwidth-hz 20000000 \
  --output outputs/rf_field.npz
```

## Active measurement selection

The scheduler ranks candidate links, channels, or beams from expected
information gain, observability gain, novelty, energy cost, and switching cost.
It is an experiment planner. It does not directly command RuView firmware or a
radar vendor API.

```bash
python -m physioatlas select-active-sensing \
  --candidates configs/physioatlas/active_sensing_candidates.json
```
