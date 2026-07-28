# WiFisio Atlas product and measurement blueprint

## Executive verdict

The implemented dual-band CSI bridge and physical study protocol are documented
in [DUAL_BAND_HOUSEHOLD_CSI_PLAN.md](DUAL_BAND_HOUSEHOLD_CSI_PLAN.md).

The project should become a household RF research observatory, not a dashboard
that merely renders convincing-looking anatomy.

PhysioAtlas v0.4 already contains a strong software foundation:

- consent-first household enrollment
- anonymous-first multi-person tracks
- open-set identity with an unknown-person path
- synchronized session schemas and calibration records
- uncertainty, observability, abstention, provenance, and null controls
- a local Research Hub API, action queue, and deterministic synthetic pilot

The central missing capability is upstream of the frontend. The current live
household bridge accepts positions, per-person embeddings, and per-person vital
estimates that another system has already separated. It does not yet transform
mixed raw household CSI into reliable localized family tracks and attributed
physiology.

The right product sequence is therefore:

1. Build an honest, contract-driven frontend against replay and synthetic state.
2. Establish synchronized real measurements and independent references.
3. Produce anonymous multi-person localization with quantified uncertainty.
4. Extract per-track respiratory signals, then cardiac signals, under explicit
   observability gates.
5. Add consented household identity only after held-out unknown-person testing.
6. Add pose and higher-order physiological research only when their independent
   validation gates pass.

The application may visualize RF rays, bodies, rooms, and furniture as a design
language, but it must never imply that a decorative reconstruction was directly
measured.

## Evaluation evidence

Evaluation date: 2026-07-28.

- `python -m physioatlas doctor --repo .`: ready, zero required failures,
  zero warnings.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -o addopts=''
  tests/physioatlas -q`: 49 passed, one pytest configuration warning.
- `python -m physioatlas household-smoke-test --output
  outputs/physioatlas/household-smoke-evaluation`: passed all checks, including
  a disjoint identity split and an unknown synthetic visitor remaining
  anonymous.
- The unmodified pytest command stalled because globally installed plugins were
  auto-loaded. The deterministic developer command should disable plugin
  auto-loading or use a project-owned environment.
- This workspace does not contain a `.git` directory. It should be placed under
  source control before a large AI-generated frontend change is accepted.

These results validate software paths and contracts. They do not validate a
person, home, physical sensor, vital sign, identity, or clinical endpoint.

## What is implemented today

### Strong foundation

The following areas are unusually thoughtful for an early research system:

- Pydantic schemas reject undeclared fields and invalid ranges.
- Enrollment requires explicit identity permission.
- Enrollment and evaluation session overlap is detected.
- Visitors can remain unknown instead of being forced into the nearest member.
- Research runs preserve hashes, split information, claim boundaries, and
  negative controls.
- The Research Hub is local-first and the worker accepts only allow-listed
  actions.
- The dashboard distinguishes research-only outputs and already uses
  "not observable" language in several places.
- The repository documents the important difference between synthetic
  verification and biological evidence.

### Current live household contract

The live bridge expects one object per already-separated person:

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

This is a useful anti-corruption boundary. It is not a raw CSI inference
pipeline. Position, separation, embeddings, and physiology are upstream
responsibilities that still need real implementations.

## Evidence boundary by capability

| Capability | Current state | Safe UI representation | Required next evidence |
|---|---|---|---|
| Software schemas, consent, reports | Verified in tests | Operational | Keep contract tests passing |
| Synthetic household tracking | Verified in deterministic simulation | Demo or replay only | Real localized observations |
| Presence | RuView has measured published models and heuristic paths | Show source and model version | Reproduce on the home's hardware |
| Person count | ESP32 firmware count is a subcarrier-slot heuristic | Experimental, never ground truth | Labeled count study with visitors, pets, fans |
| Position | Consumed from upstream by PhysioAtlas | Show source, covariance, and age | Real multi-link/mmWave/depth comparison |
| Family identity | Open-set software path exists | Anonymous by default | Unseen-day, unseen-position, unknown-person FAR |
| Respiration | Spectral baselines and study analyzers exist | Estimated only when observable | Belt-referenced multi-person validation |
| Heart rate | Spectral baseline exists | Experimental and stationary-only first | ECG/PPG-referenced held-out validation |
| 17-joint live pose | Published MM-Fi benchmark exists; live single-ESP32 path is not equivalent | Do not present as measured live pose | Wire a compatible model and validate the home domain |
| Furniture/material inversion | Design concept only in this app | Clearly labeled room model or hypothesis | Independent geometry/material reference |
| Internal organ view, diagnosis, blood pressure | Not validated and intentionally outside current claims | Do not render or claim | Separate approved research program |

## Screenshot and UX evaluation

### What to preserve

- The dark research-instrument aesthetic is a good fit.
- The top-level split between live sensing, sessions, evidence, experiments,
  rhythm, and privacy is conceptually strong.
- The 3D/2D/dual-view switch makes sense for expert inspection.
- Session confidence and "derived" labels point in the right direction.
- The large spatial canvas creates a memorable product center.

### What must change for a household product

1. The visual center is a single body at a desk. A household view needs the home
   and its people to be the center: rooms, zones, tracks, transitions, and
   current observability.
2. "3D holographic reconstruction," exact posture, RF ray tracing, furniture
   permittivity, and body mesh imagery currently look more certain than the
   underlying evidence. Decorative geometry must be visually distinct from
   measured, derived, predicted, and hypothesized layers.
3. A generic confidence percentage is insufficient. Every value needs source,
   age, observability, uncertainty or interval, calibration state, and algorithm
   version.
4. Demo mode is too subtle. The app needs a persistent environment banner and
   watermark for `Demo`, `Replay`, `Calibration`, and `Live measured`.
5. The layout is dense and uses small text. The main live view should answer
   three questions before exposing research detail:
   - Who or what is currently observable?
   - Where is each anonymous track?
   - Which estimates are trustworthy enough to show?
6. A family member requires a dedicated drill-down with history, calibration,
   references, and consent state. It should not require reading cards layered
   over a 3D scene.
7. The UI needs explicit degraded states: stale sensor, clock drift, packet
   loss, crossed tracks, ambiguous identity, motion contamination, absent
   reference, and recalibration required.
8. Accessibility needs to be designed in. Color cannot be the only distinction,
   charts need textual summaries, motion needs a reduced-motion mode, and the
   3D scene needs a usable 2D equivalent.

## Software gaps that matter before live family use

### Tracking

The current tracker is a good deterministic baseline: constant-velocity
prediction, Hungarian assignment, a position gate, and an embedding gate. It is
not yet a deployment tracker.

Required upgrades:

- reject or explicitly reorder non-monotonic live timestamps
- attach covariance to positions and velocities
- use time-based expiration instead of only update counts
- add identity and assignment hysteresis
- enforce one household identity per simultaneous active track
- detect and surface track merges, splits, and crossings
- preserve stable anonymous IDs across short service restarts
- evaluate HOTA/IDF1, ID switches per hour, fragmentation, and track coverage
- make gates dependent on localization uncertainty and elapsed time

### Measurement freshness

When a new observation omits a vital value, the current smoother retains the
old value. Without timestamps, the UI can show a stale heart or respiration rate
as if it were current.

Every metric must carry:

- `status`: `measured`, `estimated`, `not_observable`, `stale`, or `error`
- `captured_at`
- `window_start` and `window_end`
- `stale_after_ms`
- `source_kind` and `source_id`
- `algorithm_version`
- `calibration_id`
- `observability`
- `signal_quality`
- an uncertainty interval or explicit reason it is unavailable

The frontend should never infer freshness from the page refresh time.

### Identity consent

Consent is validated during enrollment, but the live tracker consumes the saved
registry without revalidating the current consent document on every activation
or consent change. Before real use:

- retain a resolvable consent reference and hash
- revalidate live-tracking and dashboard permission at startup and periodically
- immediately detach an alias when consent is revoked or expires
- record the transition in the audit chain
- avoid storing identifying embeddings in browser persistence

### API and local security

The current file-backed server is appropriate for a prototype. The next API
version should add:

- typed versioned response models rather than unconstrained dictionaries
- incremental SSE or WebSocket updates instead of polling the complete state
  every second
- a snapshot sequence number and event cursor
- server timestamps and per-source clock quality
- origin validation and a local session token for state-changing actions
- concurrency-safe store transactions
- bounded event payloads and explicit retention
- health endpoints for each adapter, not just the web process
- a read-only frontend mode that cannot enqueue actions

## Recommended product architecture

### Name

Recommended working name: **WiFisio Atlas**.

Tagline: **A private spatial physiology research observatory for the home.**

Why:

- it matches the workspace name
- it retains the useful "atlas" metaphor
- it distinguishes the project from unrelated products already using
  "PhysioAtlas"
- it does not imply diagnosis or regulated care

Use **InnerLoop** as the single-person workspace inside WiFisio Atlas, not as
the household application name.

Alternative working names:

| Name | Character | Caution |
|---|---|---|
| Signal Kin | Warm, household-first | Requires trademark/domain screening |
| Hearth Field | Home plus RF field | Less obviously physiological |
| Kin Atlas | Simple family map | "Kin" brands are crowded |
| Ambient Atlas | Broad and expandable | Less specific to WiFi |
| Home Signal Lab | Honest and descriptive | More research tool than product |

The final name still needs legal, app-store, package, and domain clearance.

### Frontend location and stack

Create the new application independently at `apps/wifisio-atlas/`. Keep the
existing dependency-light Research Hub as a fallback and contract fixture.

Recommended stack:

- React, TypeScript, and Vite
- React Router
- TanStack Query for server state
- Zod for runtime contract validation
- Zustand or React context only for UI state
- accessible headless primitives with local styling
- ECharts or uPlot for long time-series data
- Three.js through React Three Fiber for optional 3D inspection
- Vitest and Testing Library
- Playwright with axe for end-to-end and accessibility checks

No runtime CDN, remote font, remote analytics, or cloud dependency should be
required. The 2D interface must remain fully functional when WebGL is absent.

### Information architecture

1. **Live Atlas**
   - floor plan first
   - current anonymous and enrolled tracks
   - room/zone occupancy and transitions
   - observability field and sensor health
   - selection opens a person/track inspector
2. **People**
   - enrolled aliases and anonymous recent tracks
   - current status and last-seen location
   - consent, calibration, and identity evidence
3. **Person detail**
   - respiration, cardiac, movement, posture, and sleep-research timelines
   - source and provenance for every metric
   - accepted, abstained, stale, and not-observable windows
4. **Rooms and sensors**
   - floor-plan editor
   - nodes, links, coverage, clocks, packet loss, and calibration age
5. **Calibration Lab**
   - per-person and per-modality readiness
   - missing sessions and negative controls
   - guided capture workflow
6. **Studies**
   - the five existing protocols
   - references, baselines, nulls, held-out splits, and failure modes
7. **Sessions and evidence**
   - replay, annotations, export, checkpoint hashes, and experiment ledger
8. **Privacy**
   - current consent
   - unknown-person policy
   - retention and local storage
   - audit history and delete/revoke controls

### Visual hierarchy

- Layer 1: household status, urgent data-quality problems, and live map.
- Layer 2: selected person or track and its trustworthy current estimates.
- Layer 3: research diagnostics, RF field, model details, and raw evidence.

Use calm graphite and deep navy surfaces, cool cyan for measured RF data, green
for healthy infrastructure, amber for abstention/degraded quality, and violet
only for predicted or model-derived content. Reserve red for faults, not normal
uncertainty.

## Versioned frontend measurement contract

The next API should return metric objects rather than unqualified scalars:

```json
{
  "schema_version": "wifisio.metric.v1",
  "metric": "respiratory_rate",
  "status": "estimated",
  "value": 14.3,
  "unit": "breaths/min",
  "captured_at": "2026-07-28T20:15:13.211Z",
  "window": {
    "start": "2026-07-28T20:14:43.211Z",
    "end": "2026-07-28T20:15:13.211Z"
  },
  "stale_after_ms": 5000,
  "source": {
    "kind": "wifi_csi_prediction",
    "source_ids": ["link-01", "link-03", "link-04"],
    "algorithm_version": "respiration-baseline.v2",
    "calibration_id": "room-a-2026-07-20"
  },
  "quality": {
    "observability": 0.87,
    "signal_quality": 0.91,
    "motion_contamination": 0.08
  },
  "uncertainty": {
    "kind": "conformal_interval",
    "lower": 13.5,
    "upper": 15.1,
    "nominal_coverage": 0.9
  },
  "reference": {
    "available": true,
    "kind": "respiratory_belt",
    "display_policy": "research_operator_only"
  },
  "claim_class": "predicted",
  "research_only": true
}
```

A track should additionally include:

- position covariance or confidence ellipse
- location source and room coordinate frame
- first seen, last observed, and last updated timestamps
- lifecycle state: tentative, confirmed, occluded, lost
- identity state: anonymous, candidate, accepted, revoked, or ambiguous
- best and second-best identity distance
- assignment/merge/split warnings
- which measurements were spatially attributed to that track

## Measurement program

### Phase 0: define truth and observability

Deliverables:

- freeze `ObservationV2`, `TrackV2`, `MetricV1`, `SensorHealthV1`, and
  `RoomModelV1`
- define clock, packet-loss, geometry, motion, and source-quality gates
- remove defaults that silently turn absent observability or quality into `1.0`
- implement stale and not-observable transitions
- publish a claim registry mapping each UI feature to its evidence level

Exit gate: a replay can intentionally remove a sensor or metric and every
consumer shows the same explicit degraded state.

### Phase 1: synchronized physical acquisition

Use dedicated CSI-capable devices; an ordinary home router's RSSI is not a
substitute. Start with multiple fixed spatial links in one room. Add an
independent localization teacher during development and a reference sensor for
each physiological target.

Recommended references:

- depth or motion capture for position and pose during calibration only
- mmWave people tracks as an additional localization modality
- respiratory belt for respiration
- ECG or research-grade PPG for cardiac timing
- two synchronized PPG sites only for pulse-arrival research

Measure node coordinates in one room frame. Record firmware, channel, bandwidth,
packet type, antenna, packet rate, packet loss, RSSI, noise floor, and clock
quality. Do not move a sensor without creating a new calibration profile.

Exit gate: repeated empty-room and single-person captures have verified timing,
geometry, provenance, and reference alignment.

### Phase 2: anonymous multi-person localization

Start with occupancy likelihood on a 2D grid and an independent teacher. Build
tracklets from spatial evidence before attempting identity or per-person
physiology.

Compare:

- calibrated tomography or likelihood-grid baseline
- learned multistatic detector
- teacher-distilled detector
- WiFi-only inference after the teacher is removed

Track with uncertainty-aware association and explicit merge/split events.

Metrics:

- count precision, recall, F1, and mean absolute error
- position median and 90th-percentile error
- HOTA, IDF1, track coverage, fragmentation, and ID switches/hour
- latency and stale-track duration
- results separately for one, two, and three people

Exit gate: two-person crossing and leave/re-enter tests pass on an untouched day,
and a fan, pet, and empty room do not create persistent people.

### Phase 3: per-person respiration

Do not partition top subcarriers and call each group a person. Attribute the
signal through spatial separation: multistatic link weights, angle/range cells,
tomographic voxels, or a validated source-separation model.

Begin with still seated and supine windows. Add motion states only after the
stationary result is stable.

Metrics:

- rate MAE and 90th-percentile absolute error
- waveform correlation and band coherence
- coverage/selective error as observability threshold changes
- cross-person leakage
- wrong-person and time-shift null performance

Exit gate: belt-referenced respiration beats a simple spectral baseline on
unseen-day and changed-position sessions for each enrolled participant.

### Phase 4: cardiac motion

Cardiac CSI is weaker and more motion-sensitive than respiration. Gate it more
strictly, start with one stationary person, and only then attempt multiple
people.

Metrics:

- heart-rate MAE
- beat timing precision/recall when beat-level output is claimed
- Bland-Altman bias and limits of agreement
- waveform coherence
- coverage and error under posture, range, and small motion

Exit gate: an independent ECG/PPG comparison passes on held-out sessions.
Otherwise the UI says `not observable`, not a smoothed stale number.

### Phase 5: consented household identity

Identity must remain downstream of anonymous tracking. Combine evidence across
time, room transitions, and validated RF embeddings; do not identify a person
solely from heart rate, breathing rate, or motion.

Metrics:

- accepted identity accuracy and coverage
- false accept rate for unknown people
- false identity assignments per hour
- time to accepted identity
- calibration of the acceptance score
- unseen day, clothing, posture, location, and room

Exit gate: consented unknown visitors remain unknown and concurrent tracks
cannot receive the same alias.

### Phase 6: pose and body visualization

First ship coarse state: standing, seated, lying, walking, and fall-candidate
with explicit evidence. Treat the published MM-Fi pose score as a benchmark, not
as proof of the live home pipeline. Resolve the model-format/runtime gap and
evaluate the exact deployed model.

Metrics:

- MPJPE and torso-normalized PCK@20
- per-joint error, especially distal joints
- performance by person count and occlusion
- subject-, day-, and room-held-out performance
- calibration and abstention

Exit gate: the exact deployed artifact meets a committed threshold on local
held-out captures. Until then, the scene uses an abstract track marker or a
clearly labeled illustrative avatar.

### Phase 7: longitudinal and innovative research

Only after Phases 0-5 are stable:

- self-supervised link-masked pretraining on unlabeled home CSI
- teacher-student training with depth/mmWave/reference sensors removed at
  inference
- environment-aware conformal calibration
- active sensing that selects links/channels by expected information gain
- RF digital-twin simulation for augmentation, always separated from measured
  evidence
- federated household adaptation with revocation and deletion
- change-point detection over each participant's own baseline

The purpose is to improve recoverability and abstention, not to manufacture
anatomical interpretations from weak correlations.

## Concrete delivery sequence

| Milestone | Product result | Scientific gate |
|---|---|---|
| M0 Contract hardening | Versioned data and honest degraded states | Missing/stale values cannot appear current |
| M1 Frontend foundation | New responsive app with replay and mock adapters | Every visual value exposes provenance |
| M2 One-person live room | Real sensor health, position, respiration | Reference-aligned repeated capture |
| M3 Multi-person atlas | Anonymous 2-3 person tracks and zones | Crossing/visitor/pet/fan held-out tests |
| M4 Per-track physiology | Attributed respiration; cardiac only when observable | Cross-person leakage and reference gates |
| M5 Household enrollment | Consented aliases with revoke and abstain | Unknown FAR and coverage targets |
| M6 Validated body view | Coarse pose, then optional joints | Exact deployed model meets local threshold |
| M7 Longitudinal lab | Baselines, experiments, drift, active sensing | Reproducible subject/day/room holdouts |

## Primary external references

- [RuView repository and current model/evidence disclosures](https://github.com/ruvnet/RuView)
- [RuView ESP32 firmware capability and heuristic caveats](https://github.com/ruvnet/RuView/blob/main/firmware/esp32-csi-node/README.md)
- [Espressif CSI configuration documentation](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-guides/wifi-driver/wifi-vendor-features.html)
- [Person-in-WiFi 3D, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Yan_Person-in-WiFi_3D_End-to-End_Multi-Person_3D_Pose_Estimation_with_Wi-Fi_CVPR_2024_paper.html)
- [MM-Fi, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/3baf7a39d07e9f4f1e258a412df94521-Paper-Datasets_and_Benchmarks.pdf)
- [Multi-person respiration with WiMUSE](https://www.sciopen.com/article/10.1007/s11390-023-2722-z)
- [MultiSense multi-person respiration](https://dl.acm.org/doi/10.1145/3411816)
- [Pulse-Fi heart-rate study](https://arxiv.org/abs/2510.24744)
