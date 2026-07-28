# Hardware support matrix

| System | Representation | Integration level | Validation in this release |
|---|---|---|---|
| RuView CSI JSONL/RVCSI | amplitude + phase | Native converter/replay | Fixture and integration tests |
| ESP32/RuView live stream | amplitude + phase | Generic UDP bridge | Local UDP loopback only |
| Canonical NPZ sensors | arbitrary fixed-width | Native replay | Tested |
| TI/Infineon mmWave | range-Doppler or features | Adapter interface / UDP bridge | No vendor hardware test |
| UWB | IQ or features | Schema and adapter interface | No vendor hardware test |
| ECG/PPG/belts | timestamped values | CSV/NPZ ingestion | Synthetic and fixture tests |
| Depth/pose | keypoints/features | CSV/NPZ ingestion | Synthetic tests |
| Ultrasound | frame arrays/displacement | NPZ displacement extraction | Synthetic frame-shift tests |

A vendor device becomes “supported” only after its bridge records raw timing,
fixed feature semantics, quality, packet loss, device configuration, and a
hardware-backed integration test. The generic UDP adapter is the seam for that
work, not evidence that every listed device already works.
