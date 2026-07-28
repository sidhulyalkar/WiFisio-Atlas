# Session format

Each session is a directory containing `manifest.json`, `geometry.json`, and one
or more compressed NumPy streams.

## Stream NPZ

Required arrays:

- `timestamps_s`: strictly increasing float64 seconds, shape `[T]`
- `values`: float array, shape `[T, ...]`

Optional:

- `quality`: quality score or mask, first dimension `[T]`

## Geometry

`geometry.json` declares sensors, RF links, and optional anatomical anchors.
Anatomical anchors are operator-supplied coordinates in the same frame as the
sensors. They are priors, not detected anatomy.

```json
{
  "coordinate_frame": "room_meters",
  "sensors": [
    {
      "sensor_id": "tx-1",
      "modality": "wifi_csi",
      "position_m": {"x": -1.5, "y": 0.0, "z": 1.1},
      "sample_rate_hz": 100.0,
      "clock_domain": "ptp-host"
    }
  ],
  "rf_links": [
    {
      "link_id": "tx1-rx1",
      "tx_sensor_id": "tx-1",
      "rx_sensor_id": "rx-1",
      "center_frequency_hz": 5180000000.0,
      "bandwidth_hz": 40000000.0,
      "n_subcarriers": 114
    }
  ],
  "anatomy_anchors_m": {
    "thorax": {"x": 0.0, "y": 0.0, "z": 1.25}
  }
}
```

## Manifest

The manifest ties streams to the protocol and defines what each target means.
Every target names its ground-truth modality and anatomical region. Clinical
claims default to false.

```json
{
  "schema_version": "physioatlas.session.v1",
  "session_id": "participant-001-rest-01",
  "subject_id": "participant-001",
  "started_at_utc": "2026-07-27T20:00:00Z",
  "geometry_path": "geometry.json",
  "streams": [
    {
      "modality": "wifi_csi",
      "sensor_id": "mesh",
      "path": "wifi_csi.npz"
    },
    {
      "modality": "ecg",
      "sensor_id": "ecg-1",
      "path": "ecg.npz"
    }
  ],
  "targets": [
    {
      "name": "ecg_waveform",
      "source_modality": "ecg",
      "anatomy_region": "cardiac_apex",
      "task": "waveform",
      "units": "millivolts",
      "supervision_required": true,
      "clinical_claim_allowed": false,
      "description": "Reference ECG waveform; RF predicts association only."
    }
  ],
  "protocol": "paced_breathing_v1",
  "environment_id": "lab-a",
  "posture": "seated",
  "condition": "rest"
}
```
