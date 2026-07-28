# Household Pilot Runbook

This runbook converts the PhysioAtlas household infrastructure into a staged,
consent-first home experiment. It is a research protocol, not a medical-device
procedure. Ordinary WiFi RSSI is not sufficient. Use a compatible RuView CSI
source, preferably multiple spatially separated links, and add independent
reference sensors for every physiological claim being tested.

## Stage A: verify the software before involving people

```bash
bash scripts/household-research-bootstrap.sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/household-research-bootstrap.ps1
```

Pass criteria:

- all PhysioAtlas tests pass
- household smoke report is `passed`
- all five study reports are generated
- the synthetic unknown visitor stays anonymous
- household privacy fault injection is `passed`

Synthetic separability verifies plumbing only. It does not establish that real
family members are separable from RF.

## Stage B: choose hardware and map the room

Minimum useful home pilot:

- two or more compatible RuView CSI receivers or links
- fixed transmitter and receiver locations
- one independent localization source during development, such as depth pose or
  mmWave tracks
- respiratory belt for respiration studies
- ECG or PPG for cardiac studies
- two PPG sites for pulse-arrival studies
- synchronized ultrasound only for regional-motion studies

Record sensor coordinates in meters in one room coordinate frame. Photograph or
sketch placements, record firmware revisions and radio channels, and do not move
nodes between calibration and validation without creating a new calibration
profile.

## Stage C: initialize the household workspace

```bash
python -m physioatlas init-household-research \
  --output data/physioatlas/household \
  --household-id home-pilot \
  --member "member-01=Member One" \
  --member "member-02=Member Two"
```

The generated consent files are drafts with identity and live-tracking
permissions disabled. Each adult participant must review and explicitly enable
the intended uses. A parent or guardian must determine whether participation by
a minor is appropriate under the applicable research and privacy rules. Do not
enroll visitors.

## Stage D: collect calibration data

Collect at least three sessions per member on separate occasions. Four to six is
better for estimating within-person variation. Keep a fourth or later session
completely untouched for evaluation.

Recommended blocks per session:

1. empty-room baseline, 60 seconds
2. seated quiet breathing, 3 minutes
3. supine quiet breathing, 3 minutes
4. paced slow and faster breathing, 2 minutes each
5. speaking and small arm movements, 2 minutes
6. sit-to-stand and short walking path, 3 minutes
7. leave and re-enter the room, 3 repetitions
8. fan, curtain, pet, or other household distractor controls

Create a new session manifest for every block or preserve block annotations in
one manifest. Record synchronization events, packet loss, sensor geometry,
posture, clothing condition, room state, and reference-sensor quality.

## Stage E: validate, enroll, and evaluate without leakage

Put enrollment sessions and held-out evaluation sessions in separate roots:

```text
data/physioatlas/household/
  enrollment-sessions/
  validation-sessions/
```

Validate consent and data contracts:

```bash
python -m physioatlas validate \
  --data data/physioatlas/household/enrollment-sessions \
  --require-consent

python -m physioatlas validate \
  --data data/physioatlas/household/validation-sessions \
  --require-consent
```

Enroll only from the calibration root:

```bash
python -m physioatlas household-enroll \
  --data data/physioatlas/household/enrollment-sessions \
  --registry data/physioatlas/household/household-registry.json \
  --household-id home-pilot \
  --aliases data/physioatlas/household/aliases.json \
  --minimum-sessions 3
```

Evaluate only on untouched sessions:

```bash
python -m physioatlas household-evaluate \
  --data data/physioatlas/household/validation-sessions \
  --registry data/physioatlas/household/household-registry.json
```

The evaluator rejects any session ID already used for enrollment. Report both
accepted accuracy and coverage. A classifier that rejects everyone can have no
false identities but is not useful. A classifier that labels every visitor is
unsafe.

## Stage F: connect the live observation stream

PhysioAtlas expects an upstream localization or separation layer to produce one
observation per currently tracked person. Raw single-link CSI cannot reliably
assign a mixed signal to individual people.

Replay the example contract:

```bash
python -m physioatlas household-replay-jsonl \
  --input configs/physioatlas/household/observation.example.jsonl \
  --registry data/physioatlas/household/household-registry.json \
  --state outputs/physioatlas/research-hub-state.json
```

Or receive local UDP observations:

```bash
python -m physioatlas household-listen-udp \
  --registry data/physioatlas/household/household-registry.json \
  --state outputs/physioatlas/research-hub-state.json \
  --host 127.0.0.1 --port 8790 --duration-s 300
```

Do not include names in packets. The local open-set matcher attaches an alias
only after repeated, consented, sufficiently separated evidence.

## Stage G: operate the Research Hub

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
  --data data/physioatlas/household/validation-sessions \
  --registry data/physioatlas/household/household-registry.json
```

Open `http://127.0.0.1:8770`.

The hub displays:

- current anonymous and enrolled tracks
- rolling respiratory, cardiac, motion, quality, and observability traces
- per-member RF identity calibration
- physiological-baseline calibration
- modality availability and reference support
- all five priority-study reports
- experiment records, events, and privacy state

## Stage H: run each priority study

Run the complete suite:

```bash
python -m physioatlas run-priority-studies \
  --data data/physioatlas/household/validation-sessions \
  --output outputs/physioatlas/studies \
  --minimum-sessions 4
```

Or update and run one committed protocol:

```bash
python -m physioatlas run-priority-study \
  configs/physioatlas/household/study-respiration_sleep.yaml
```

Interpretation boundaries:

- respiration: compare RF waveform and rate with belts or polysomnography
- cardiac mechanics: compare with ECG/PPG; do not diagnose arrhythmia
- pulse propagation: report arrival delay; do not infer blood pressure without
  a separate validated calibration study
- mobility/gait: use pose, motion capture, or instrumented walkway reference
- organ-correlated motion: predict synchronized ultrasound displacement; never
  render a fictional organ image

## Stage I: go/no-go gates for a real household result

Do not claim household distinguishability or unique physiology until all gates
pass on real held-out sessions:

- no overlap between enrollment and validation session IDs
- every participant has explicit, current consent
- unknown-person controls remain unknown
- accepted identity accuracy and coverage are both reported
- performance survives unseen-day and changed-position tests
- physiological estimates beat constant and simple signal baselines
- time-shift, permutation, and wrong-geometry controls fail as expected
- observability falls during motion, dropout, poor geometry, or absent references
- uncertainty intervals are calibrated on held-out data
- results are reproduced after restarting the complete hardware stack

Store every negative run. Failed separability is valuable evidence about the
limits of the current radio geometry and should guide additional links, mmWave,
or improved reference sensing rather than threshold shopping.
