# Consent-first household research

PhysioAtlas v0.4 supports a local household research workflow, but it does not
assume that ordinary network traffic reveals a reliable identity or medical
measurement. Full WiFi Channel State Information normally requires compatible
ESP32 nodes, a research NIC, or another supported RuView capture source.
Router RSSI and device MAC addresses are not substitutes for physiological CSI.

## Safety and privacy boundary

The system creates anonymous tracks first. It may attach an enrolled household
alias only when all of the following are true:

1. The participant explicitly acknowledged identity enrollment and live tracking.
2. At least the minimum number of calibration sessions exists.
3. The observation lies inside that member's enrolled distribution.
4. The best profile is sufficiently separated from the second-best profile.
5. Observability and signal quality remain adequate.

Visitors, unconsented people, and ambiguous observations remain
`unknown-track-*`. Do not use the system for covert surveillance, access control,
employment, insurance, diagnosis, or emergency decision-making.

## 1. Initialize the workspace

```bash
python -m physioatlas init-household-research \
  --output data/physioatlas/household \
  --household-id home-pilot \
  --member "member-01=Sid" \
  --member "member-02=Family Member 2"
```

This writes draft consent files. It does not grant consent automatically. Review
each file with the participant. Only after they agree, set the requested
permissions and `participant_acknowledged` to `true`, then save it as
`consent.json`.

## 2. Collect calibration sessions

Record at least three sessions per member, preferably on different days. Use the
same synchronized acquisition stack for every participant:

- RuView CSI mesh with at least four useful links
- mmWave when cardiac mechanics are a target
- ECG or PPG for cardiac reference
- chest and abdominal respiratory belts for respiration
- depth/pose for mobility and multiperson separation
- ultrasound only for narrow regional-motion supervision
- shared synchronization events and measured sensor geometry

Recommended calibration blocks:

1. Quiet seated rest
2. Quiet supine rest
3. Paced slow and faster breathing
4. Speaking and small voluntary motion
5. Leave and re-enter the room

Also record empty-room, fan/curtain, wrong-geometry, and consented unknown-person
controls. Never create an identity profile for an unconsented visitor.

## 3. Validate and enroll

```bash
python -m physioatlas validate \
  --data data/physioatlas/household/sessions \
  --require-consent

python -m physioatlas household-enroll \
  --data data/physioatlas/household/sessions \
  --registry data/physioatlas/household/household-registry.json \
  --household-id home-pilot \
  --aliases data/physioatlas/household/aliases.json \
  --minimum-sessions 3

python -m physioatlas household-evaluate \
  --data data/physioatlas/household/sessions \
  --registry data/physioatlas/household/household-registry.json
```

Enrollment-set accuracy is only an infrastructure check. The meaningful test is
an unseen-day, unseen-position, and preferably unseen-room evaluation.

## 4. Start the research hub

Terminal 1:

```bash
python -m physioatlas serve-research-hub \
  --state outputs/physioatlas/research-hub-state.json \
  --port 8770
```

Terminal 2:

```bash
python -m physioatlas research-hub-worker \
  --state outputs/physioatlas/research-hub-state.json \
  --data data/physioatlas/household/sessions \
  --registry data/physioatlas/household/household-registry.json
```

Open `http://127.0.0.1:8770`.

## 5. Feed localized observations

The household tracker consumes already-separated person observations. This is
important: raw single-link CSI does not tell the software which person produced
each component. Use multi-link localization, mmWave tracks, depth pose, or a
validated separation model upstream.

JSONL replay:

```bash
python -m physioatlas household-replay-jsonl \
  --input configs/physioatlas/household/observation.example.jsonl \
  --registry data/physioatlas/household/household-registry.json \
  --state outputs/physioatlas/research-hub-state.json
```

Live local UDP:

```bash
python -m physioatlas household-listen-udp \
  --registry data/physioatlas/household/household-registry.json \
  --state outputs/physioatlas/research-hub-state.json \
  --host 127.0.0.1 --port 8790 --duration-s 300
```

Each JSON object uses this contract:

```json
{
  "timestamp_s": 12.5,
  "position_m": [1.2, 0.8],
  "embeddings": {
    "wifi_csi": [0.1, 0.2, 0.3],
    "mmwave": [0.4, 0.5, 0.6]
  },
  "signal_quality": 0.91,
  "observability": 0.87,
  "respiratory_rate_bpm": 14.3,
  "heart_rate_bpm": 69.8,
  "motion_index": 0.05
}
```

Do not send names in the live packet. Identity is resolved locally against the
consented registry.

## 6. Run the priority studies

```bash
python -m physioatlas run-priority-studies \
  --data data/physioatlas/household/sessions \
  --output outputs/physioatlas/studies
```

Reports preserve reference availability, null comparisons, calibration status,
claim boundaries, and failures. A valid software run is not evidence of clinical
accuracy.
