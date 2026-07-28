# CSI/BFI fusion and reflector runbook

This runbook exercises the implemented research baseline. It does not validate
a physical receiver, reflector, person, position, identity, or vital sign.

## 1. Decode and normalize BFI

PhysioAtlas begins with decoded numeric BFI. Packet capture, legal scope
filtering, and 802.11ac/ax compressed-angle decoding remain external acquisition
steps. Preserve the pcap and decoder version.

Each raw JSONL frame declares:

- timestamp, fixed link and client IDs;
- band, channel, bandwidth and packet sequence;
- required receiver quality;
- exactly one numeric feature vector or rectangular decoded matrix.

Build an empty-room calibration:

```powershell
python -m physioatlas household-bfi-calibrate `
  --empty captures/bfi-empty.jsonl `
  --environment-id living-room-v1 `
  --output calibrations/bfi-empty-v1.json
```

Normalize a capture:

```powershell
python -m physioatlas household-bfi-process `
  --input captures/bfi-zone-desk.jsonl `
  --calibration calibrations/bfi-empty-v1.json `
  --output outputs/bfi-zone-desk.features.jsonl
```

BFI features remain explicitly non-identifying.

## 2. Fit BFI zone prototypes

Create a manifest whose paths point to normalized, single-person zone captures:

```json
{
  "desk": "../../outputs/bfi-zone-desk.features.jsonl",
  "sofa": "../../outputs/bfi-zone-sofa.features.jsonl"
}
```

Fit the prototype model:

```powershell
python -m physioatlas household-bfi-zone-calibrate `
  --zones captures/bfi-zones.json `
  --output calibrations/bfi-zone-prototypes-v1.json
```

Use held-out days and people to determine whether these prototypes transfer.
The training captures cannot be reported as localization evaluation.

## 3. Fuse calibrated evidence

The fusion engine accepts JSONL windows. Each row contains `evidence`, a list of
strict `physioatlas.rf-zone-evidence.v1` objects. One window can contain
multiple source tracks. CSI household observations can be converted with
`csi_observation_to_zone_evidence`; normalized BFI features use
`bfi_feature_to_zone_evidence`.

Example configuration:

```json
{
  "zones": [
    {"zone_id": "desk", "center_m": [1.0, 1.0], "position_uncertainty_m": 0.75},
    {"zone_id": "sofa", "center_m": [4.0, 2.0], "position_uncertainty_m": 0.75}
  ],
  "source_weights": {"wifi_csi": 1.0, "wifi_bfi": 1.0},
  "maximum_time_skew_s": 0.1,
  "maximum_sync_uncertainty_s": 0.025,
  "occupancy_threshold": 0.65,
  "maximum_cross_modal_conflict": 0.35,
  "minimum_observability": 0.55,
  "require_both_sources": true
}
```

Run:

```powershell
python -m physioatlas household-csi-bfi-fuse `
  --input outputs/fusion-windows.jsonl `
  --config configs/fusion-living-room.json `
  --output outputs/fusion-decisions.jsonl
```

The current baseline uses quality-weighted log-odds and abstains for missing
sources, clock uncertainty, time skew, low observability, modal conflict, or no
zone above threshold. Do not tune thresholds on the locked test set.

## 4. Verify the reflector software safely

Run the explicit simulator:

```powershell
python -m physioatlas reflector-simulate-calibration `
  --config configs/physioatlas/household/motorized-reflector.example.yaml `
  --telemetry outputs/reflector/telemetry.jsonl `
  --health outputs/reflector/health.json
```

The simulator does not actuate hardware. A physical driver must implement the
`ReflectorDriver` protocol and independently enforce hard limits, current
cutoff, limit switches, obstacle detection, local child/pet lockout, and a
physical emergency stop.

The minimum study prototype is a passive 25-35 cm conductive plate or corner
reflector plus an interchangeable nonconductive sham. Use a guarded low-speed
rail or turntable with encoder and independent fiducial reference. Motor power
must be reported off during RF dwells.

## 5. Build and compare transfer atlases

Transform CSI/BFI RF timestamps into the reflector telemetry clock using an
immutable alignment receipt. Each `ReflectorRfSample` carries that receipt ID,
source, link, band, energy, features, and quality.

Build:

```powershell
python -m physioatlas reflector-build-rf-atlas `
  --telemetry outputs/reflector/telemetry.jsonl `
  --health outputs/reflector/health.json `
  --rf-samples outputs/reflector/aligned-rf.jsonl `
  --atlas-id living-room-baseline `
  --environment-id living-room-v1 `
  --output calibrations/reflector-atlas-baseline.json
```

Only stable, aligned, motor-off dwell samples are accepted.

Compare a later atlas:

```powershell
python -m physioatlas reflector-compare-atlas `
  --baseline calibrations/reflector-atlas-baseline.json `
  --candidate calibrations/reflector-atlas-today.json `
  --maximum-drift-score 0.35 `
  --output outputs/reflector/drift-report.json
```

The drift report describes reflector transfer drift. Whether it predicts
occupancy or physiology degradation is a study hypothesis, not an implemented
fact.

## 6. Evaluation and frontend

- Use `run_fusion_ablation` for matched CSI-only, BFI-only and fused evaluation.
- Follow [CSI_BFI_FUSION_STUDY.md](CSI_BFI_FUSION_STUDY.md) for locked splits,
  null controls and preregistered hypotheses.
- Follow
  [REFLECTOR_FUSION_FRONTEND_SPEC.md](REFLECTOR_FUSION_FRONTEND_SPEC.md) for
  rendering, replay, safety, abstention and accessibility.
