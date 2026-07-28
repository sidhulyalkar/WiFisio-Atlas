# Dual-band household CSI implementation and study plan

## Direct answer

Yes. The BE400's 2.4 GHz and 5 GHz transmissions can provide complementary
views of the same motion. They should be treated as separate calibrated links
and fused at the evidence level.

- 2.4 GHz has a wavelength near 12.5 cm. It usually reaches farther and
  penetrates obstructions better, but offers coarser spatial phase sensitivity
  and often more interference.
- 5 GHz has a wavelength near 5.8 cm around 5.2 GHz. The same path-length
  change produces roughly twice the phase rotation, but attenuation and
  sensitivity to blockage are greater.
- Different channels and bandwidths illuminate different multipath responses.
  That diversity can make two zones or people less correlated and therefore
  easier to separate.
- Raw subcarrier indexes, amplitudes, or phases from different bands are not
  directly interchangeable. Each fixed transmitter/receiver/band/channel link
  needs its own empty-room normalization and zone response.

This is diversity, not magical resolution. Two bands cannot uniquely solve an
underdetermined household mixture. The number and geometry of independent
spatial links, time synchronization, traffic stability, calibration coverage,
and independent references still dominate the result.

The Archer BE400 is a useful stable dual-band transmitter and traffic source.
Its documented interface does not expose research-grade CSI. Keep its stock
firmware and add receivers that do expose CSI. TP-Link documents simultaneous
2.4/5 GHz MLO for compatible clients; this is useful for traffic generation,
but does not by itself synchronize external sensing receivers.

## What is now implemented

PhysioAtlas now has a conservative raw-CSI producer:

- `household-csi-calibrate` learns per-link empty-room statistics and
  single-person zone signatures.
- Every link records a band, center frequency, channel, bandwidth, and optional
  receiver clock calibration.
- A link whose band/channel metadata changes inside a capture is rejected.
- Optional shared transmitter packet sequence IDs estimate clock offset and
  jitter between receivers.
- `household-csi-process` synchronizes and regularizes fixed links, then fuses
  both bands in one calibrated spatial response matrix.
- Non-negative zone activation estimates which calibrated zones are occupied.
- Ridge demixing produces a separate temporal component for each active zone.
- Each component produces a zone position, covariance, observability,
  respiration estimate, experimental identity embedding, and measurement
  status.
- Multi-person heart rate is disabled by default. Heart-rate estimation is an
  explicit experimental option and rejects peaks that coincide with
  respiration harmonics.
- Identity embedding versions are checked against enrollment versions. A new
  CSI embedding cannot accidentally match an incompatible old profile.

This baseline separates occupied calibrated zones. It does not yet separate two
people standing in the same zone, reconstruct skeletal pose, or establish
clinical-grade vital signs.

## Recommended physical topology

Start with one room, not the whole house:

1. Keep the Archer BE400 in its normal central position.
2. Use at least six fixed CSI receive links:
   three on a fixed 2.4 GHz channel and three on a fixed 5 GHz channel.
3. Prefer ESP32-C5 boards for new nodes because Espressif documents CSI and
   operation on either band. One C5 does not receive both bands simultaneously;
   use dedicated fixed-band nodes.
4. Place receivers around the perimeter at varied heights and angles. Avoid a
   line of nodes on one wall.
5. Use a dedicated traffic client or controlled UDP stream so the packet rate
   is measurable and stable. Log transmitter sequence, receiver ID, MAC/link
   ID, band, channel, bandwidth, RSSI, noise floor, and CSI together.
6. Keep sensing channels fixed during a session. Recalibrate after DFS channel
   changes, router/node movement, or major furniture changes.
7. Timestamp locally at capture and align receivers using the same transmitter
   packet sequence. NTP alone is not evidence of packet-level alignment.

Move to eight or more links, additional rooms, or a mesh only after ablations
show that they improve held-out results.

## Input contract

Each JSONL row is one CSI frame from one fixed link:

```json
{
  "timestamp_s": 42.150,
  "sequence": 18421,
  "link_id": "be400-to-c5-west-5g",
  "frequency_band": "5ghz",
  "center_frequency_hz": 5180000000,
  "channel": 36,
  "bandwidth_mhz": 20,
  "rssi_dbm": -51,
  "noise_floor_dbm": -92,
  "amplitude": [12.1, 11.8, 12.4, 12.0]
}
```

`phase` may accompany `amplitude`, or `iq_hex` may replace both. Stable
hardware-derived quality is preferred. When it is absent, the processor uses a
deliberately conservative derived quality rather than assuming perfect data.

## Calibration protocol

### 1. Hardware and channel record

Record router firmware, receiver firmware and MAC, antenna orientation,
coordinates, band, channel, bandwidth, packet rate, and room geometry. Assign a
new environment ID when the RF geometry materially changes.

### 2. Shared-packet synchronization

Capture at least 30 seconds in which every receiver observes packets from the
same controlled transmitter. Store the transmitter sequence number. Inspect
per-link packet matches, median clock offset, median absolute jitter, packet
loss, and inter-arrival distributions.

### 3. Empty-room baseline

Capture five to ten minutes with no people or pets. Repeat at several times of
day. Fans, HVAC, doors, curtains, robot vacuums, and moving foliage are separate
null-control sessions, not part of the empty baseline.

### 4. Spatial dictionary

Mark a coarse grid of 0.75-1.0 m zones. One consented participant occupies one
zone at a time for 60-90 seconds while facing four orientations. Repeat with
multiple body sizes. The current baseline uses each zone's normalized
multi-link response as a spatial dictionary column.

### 5. Path and count references

For development sessions only, use an independent position reference such as a
depth camera, AprilTag camera, or UWB tag. Include zero through the intended
maximum number of people, crossing paths, adjacent zones, visitors, pets, and
two people in one zone. Camera/reference files should be separately consented,
encrypted, access-controlled, and removable after derived ground truth is
approved.

### 6. Physiological references

Use a respiratory inductance belt for respiration and ECG or a validated
chest-strap/PPG reference for cardiac rate. Align them with a visible/electrical
sync event. Include stillness, speaking, posture changes, exercise recovery,
sleeping orientations, blankets, fans, and deliberate motion corruption.

### 7. Identity reference

Collect identity only under explicit opt-in. Split enrollment and evaluation by
day and session. Test unenrolled visitors, clothing changes, carried objects,
weight/posture changes, and leave-one-room-out transfer. Report false accepts
per unknown-person hour, not accuracy alone.

## Running the baseline

Create a zone manifest:

```json
{
  "desk": {"center_m": [1.2, 0.8], "capture": "zone-desk.jsonl"},
  "sofa": {"center_m": [4.1, 2.2], "capture": "zone-sofa.jsonl"}
}
```

Build calibration:

```powershell
python -m physioatlas household-csi-calibrate `
  --empty captures/empty.jsonl `
  --sync captures/shared-packet-sync.jsonl `
  --zones captures/zones.json `
  --environment-id living-room-v1 `
  --sample-rate-hz 20 `
  --minimum-links 6 `
  --output calibrations/living-room-v1.json
```

Process a mixed capture and require both bands:

```powershell
python -m physioatlas household-csi-process `
  --input captures/family-mixed.jsonl `
  --calibration calibrations/living-room-v1.json `
  --require-band 2.4ghz `
  --require-band 5ghz `
  --output outputs/family-observations.jsonl
```

The resulting JSONL can be replayed through `household-replay-jsonl`. Heart
rate remains disabled unless an explicit processing configuration enables the
experimental path.

## The most useful novel project

Build **OpenHome RF Atlas**: a consent-first, self-calibrating benchmark and
local observatory for household RF sensing.

Its novel contribution should not be another impressive body animation. It
should answer a question the field badly needs answered:

> When do multi-band CSI and standard Wi-Fi beamforming feedback provide
> enough independent evidence to separate, localize, and monitor multiple
> people across real home changes—and when should the system abstain?

The project combines:

- fixed 2.4/5 GHz CSI links for micro-motion and respiration;
- monitor-mode beamforming-feedback information as an optional additional
  spatial/identity view;
- a reusable active calibration object, such as a small motorized reflector,
  moved through known locations to measure the room transfer response without
  storing human biometrics;
- test-time calibration/adaptation using empty-room and calibration-object
  captures, never evaluation labels;
- explicit open-set identity and an anonymous mode;
- a public de-identified benchmark of raw RF, calibration state, null controls,
  references, observability, and abstentions.

The motorized calibration object is particularly promising. A repeatable
reflector with known trajectory/frequency can measure link sensitivity,
receiver timing, drift, dead zones, and cross-band transfer every day. That
turns room change from an invisible failure into a measurable calibration
event.

## First publishable studies

1. **Dual-band value:** 2.4 only versus 5 only versus early fusion versus
   calibrated late fusion, evaluated by held-out day, room, device, and person.
2. **Link sufficiency:** progressively add spatial links and measure count,
   position, respiration attribution, coverage, and compute cost.
3. **Self-calibration:** static empty-room calibration versus motorized
   reflector calibration after furniture and channel changes.
4. **CSI plus BFI:** compare CSI, beamforming feedback, and fusion for anonymous
   tracking and consented open-set identity.
5. **Abstention:** test whether observability and condition-number gates reduce
   false attribution during same-zone occupancy, path crossings, pets, fans,
   and packet loss.

Pre-register primary metrics. Position should use median and 95th-percentile
error; counting should use per-window precision/recall and count MAE;
respiration/cardiac results should include MAE, coverage, Bland-Altman limits,
and wrong-person attribution; identity should include unknown false accepts
and equal error rate.

## Evidence anchors

- [TP-Link Archer BE400 specifications](https://www.tp-link.com/us/home-networking/wifi-router/archer-be400/)
- [Espressif ESP32-C5 dual-band behavior](https://docs.espressif.com/projects/esp-idf/en/stable/esp32c5/api-guides/wifi-driver/overview.html)
- [Espressif CSI API](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-guides/wifi-driver/wifi-vendor-features.html)
- [Person-in-WiFi 3D multi-person pose](https://openaccess.thecvf.com/content/CVPR2024/html/Yan_Person-in-WiFi_3D_End-to-End_Multi-Person_3D_Pose_Estimation_with_Wi-Fi_CVPR_2024_paper.html)
- [WiMUSE multi-person respiration](https://www.sciopen.com/article/10.1007/s11390-023-2722-z)
- [WiMANS multi-user benchmark](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05826.pdf)
- [WiTTA-Bench cross-domain adaptation](https://openaccess.thecvf.com/content/CVPR2026/html/Li_WiTTA-Bench_Benchmarking_Test-Time_Adaptation_for_WiFi_Sensing_CVPR_2026_paper.html)
- [Wi-BFI extraction tool](https://arxiv.org/abs/2309.04408)
- [BFId identity publication record](https://publikationen.bibliothek.kit.edu/1000185756)
- [BFId dataset](https://radar.kit.edu/radar/en/dataset/hdcds6a8fukdennd)
- [RuView evidence disclosures](https://github.com/ruvnet/RuView/blob/main/README.md)
- [RuView ESP32 firmware caveats](https://github.com/ruvnet/RuView/blob/main/firmware/esp32-csi-node/README.md)
