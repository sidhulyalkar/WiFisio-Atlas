# Acquisition and synchronization

## Adapter contract

Each adapter declares modality, sample rate, RF representation, clock domain,
quality support, live/replay status, and adapter-specific parameters. All
adapters return strictly increasing timestamps, a fixed-width value matrix, and
an optional quality vector in `[0, 1]`.

Built-ins:

| Adapter | Purpose | Hardware status |
|---|---|---|
| `npz_replay` | Replay canonical streams | Tested |
| `ruview_jsonl` | Convert and replay RuView CSI JSONL/RVCSI | Tested on fixtures |
| `synthetic_wave` | Deterministic integration testing | Tested |
| `udp_json` | Live vendor-neutral datagram bridge | Loopback tested |

## UDP frame format

```json
{
  "timestamp_s": 0.120,
  "amplitude": [1.0, 1.1, 0.9],
  "phase": [0.1, 0.2, 0.3],
  "quality": 0.94
}
```

`timestamp_ns` may replace `timestamp_s`. A flat `values` array may replace
amplitude and phase. Frames with non-finite values are discarded. A changing
feature width is a hard error.

## Concurrent start

`run_acquisition` starts adapters through a thread pool and records launch skew.
It does not imply hardware-level synchronization. Use shared trigger events or
PTP-capable timestamps where available, then fit an affine clock calibration:

```bash
python -m physioatlas calibrate-clock \
  --source-events sensor_events.csv \
  --reference-events reference_events.csv \
  --output calibration.json
```

The fitted mapping estimates offset, scale, drift in ppm, residual RMSE, and
reference/source clock identities. Excessive drift fails dataset validation.

## Real protocol checklist

1. Record empty-room RF before the subject enters.
2. Measure every sensor position in one coordinate frame.
3. Generate at least three shared synchronization events over the session.
4. Store raw device timestamps and packet sequence numbers.
5. Record subject posture, condition, clothing, and environment ID.
6. Preserve dropouts in quality masks rather than interpolating through them.
7. Run consent and data-contract validation before model training.
