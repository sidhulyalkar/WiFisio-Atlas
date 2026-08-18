# PhysioPersona: evidence-first physiological visualisation

PhysioPersona is a proposed presentation and experimentation layer on top of
PhysioAtlas. Its goal is to make RF sensing understandable, reproducible, and fun
to explore without turning uncertain research estimates into medical-looking
claims.

The core rule is simple:

> The visual layer may become whimsical. The evidence layer may not.

A breathing torso, sleepy forest, energetic gait, or animated avatar is a visual
encoding of a versioned signal object. Every visible state retains its source,
method, confidence, observability, age, reference status, and claim boundary.
When the evidence is weak, the signal disappears or becomes explicitly unknown.

## Why this belongs in PhysioAtlas

PhysioAtlas already has the pieces that matter scientifically:

- synchronized RF and reference acquisition;
- subject, room, session, hardware, and condition holdouts;
- required negative controls and null tests;
- uncertainty and abstention;
- consent-first household tracking;
- local-only research infrastructure; and
- a machine-readable experiment record.

The missing layer is a stable boundary between those research objects and
external visual clients. `physioatlas.persona.v1` is that boundary.

## Contract

`physioatlas/persona.py` defines a presentation-safe snapshot with:

- scalar/categorical signals such as respiration rate or movement intensity;
- optional four-class sleep probabilities;
- confidence and observability on every estimate;
- measured/derived/predicted/self-report/demo evidence labels;
- explicit reference-validation status;
- per-signal claim boundaries;
- abstention below configured confidence/observability thresholds; and
- privacy metadata.

The contract intentionally cannot contain raw RF, raw camera frames, or biometric
templates. A subject alias can only be attached when the caller supplies the
`presentation_identity` consent scope.

This makes the JSON useful for several clients without exposing the underlying
research data:

```text
compatible CSI / mmWave / reference devices
                  |
                  v
      PhysioAtlas acquisition + QC
                  |
                  v
    per-modality estimators / studies
                  |
                  v
 confidence + observability + provenance
                  |
                  v
       physioatlas.persona.v1
          /          |          \
         /           |           \
 public replay   local dashboard   secure live bridge
         |                           |
         +------ web PhysioPersona --+
                      |
             stylized 3-D avatar
```

## Sleep: what is implemented now

The existing `respiration_sleep` priority study is a respiratory-mechanics
baseline. It estimates respiratory rate, waveform correlation, and pause
fraction. It is **not** a validated sleep-stage model.

`physioatlas/sleep_benchmark.py` adds the evaluation half needed to change that.
It standardizes a four-class target:

- wake;
- light sleep (N1 + N2);
- deep sleep (N3); and
- REM.

It reports accuracy for comparison with prior WiFi literature, but does not let
accuracy hide model weaknesses. First-class outputs also include:

- balanced accuracy;
- macro F1;
- Cohen's kappa;
- per-stage precision/recall/F1;
- confusion matrix;
- top-label calibration error;
- probability log loss and multiclass Brier score when probabilities exist;
- coverage after confidence-based abstention;
- sleep-stage transition rates; and
- per-subject metrics.

`leave_one_subject_out_folds()` supplies deterministic subject-disjoint folds.
Room, device, and deployment holdouts should be added around the same evaluator
as data volume grows.

## Best first sleep dataset

The SPAN Lab / University of Utah RF Respiration Monitoring Sleep-Wake dataset is
a strong first external benchmark because it contains approximately eight-hour
sleep studies from 20 patients with simultaneous WiFi CSI and polysomnography.
The associated PSG includes sleep-state and breathing-rate ground truth.

Dataset page:

- https://patwarilab.com/products.html
- Follow the "RF Respiration Monitoring Sleep-Wake Dataset" link.

Recommended sequence:

1. Preserve the original patient/session boundaries before feature extraction.
2. Write a read-only adapter that produces canonical PhysioAtlas sessions.
3. Reproduce respiration-rate performance first.
4. Build a deliberately simple sleep/wake baseline from movement + respiration.
5. Train a four-stage temporal baseline only after the leakage audit passes.
6. Evaluate leave-one-patient-out and report the full selective benchmark.
7. Keep the best model frozen and test room/device transfer on a second dataset or
   newly collected synchronized nights.

Do not tune on the reported test folds. Do not use random epoch splits.

## Model ladder

The first useful model does not need to be exotic. A progression that produces
interpretable evidence is preferable:

### S0: respiratory mechanics

Current filtered RF component plus respiratory-belt/PSG respiratory reference.
Goal: validate signal recoverability and failure modes.

### S1: sleep/wake

Input windows:

- respiratory waveform/envelope;
- respiratory-rate variability;
- RF motion energy;
- movement event density;
- signal quality/observability; and
- time-of-night context as an explicitly ablated feature.

Compare logistic/ridge, gradient boosting, and a small temporal model.

### S2: four-stage sequence model

Use 30-second epochs and several minutes of context. Start with a compact
TCN/GRU/Transformer encoder over RF-derived cardiorespiratory and motion features.
Report both raw and temporally smoothed predictions so smoothing cannot conceal
weak epoch-level discrimination.

### S3: representation transfer

Pretrain on large public PSG datasets using the subset of cardiorespiratory and
movement-like channels that WiFi can plausibly approximate. Fine-tune the RF
encoder with modality alignment or teacher-student distillation. Never train on
EEG and then describe the resulting RF model as if RF measured EEG.

### S4: multimodal low-burden fusion

Evaluate WiFi alone, camera-derived non-identifying motion/expression features,
PPG/wearable signals when voluntarily connected, and their combinations. The
ablation table is part of the product, not an appendix.

## Web deployment modes

### 1. Public demo

Safe for a portfolio site. Uses synthetic or pre-recorded de-identified
`physioatlas.persona.v1` snapshots. No home sensor connection and no camera
required.

### 2. Local interactive mode

The browser can run a local camera model and local avatar customisation. Raw
camera frames should remain in the browser. Only derived, user-visible features
may enter the persona state.

### 3. Live home-sensor mode

Do **not** expose the current Research Hub directly to the public internet. It is
intentionally local and has no internet-facing authentication model.

A future bridge should use:

- an explicit pairing gesture;
- short-lived scoped tokens;
- an origin allow-list;
- TLS;
- a narrow schema that accepts only `physioatlas.persona.v1` snapshots;
- replay protection / monotonic sequence numbers;
- rate limits;
- no raw RF or camera transport by default; and
- a one-click disconnect/revoke path.

The preferred deployment is a local companion that establishes an outbound
secure session. The public website should never scan the LAN or assume access to
`localhost` services.

## Facial expression and "mood"

A face model can provide useful animation signals, but facial expression should
not be presented as ground-truth emotion or mental state.

Recommended layers:

1. camera -> on-device landmarks/blendshapes;
2. landmarks -> expression features such as smile intensity, blink rate, head
   pose, and speaking/mouth motion;
3. optional user self-report for actual mood labels;
4. visual fusion with physiological signals; and
5. explicit uncertainty and source badges.

Raw frames should be discarded unless the user deliberately saves them for a
separate consented study.

## Persona learning

Do not infer personality from a face or physiological state. The playful persona
can still become deeply personal by learning from explicit interactions:

- which worlds the visitor chooses to explore;
- preferred avatar accessories/colors;
- whether they interact quickly or linger;
- explicitly liked activities;
- optional text/self-description; and
- local returning-session history.

These can update transparent traits such as `curiosity`, `calm_vs_bouncy`,
`social_vs_solo`, and `collector_vs_explorer`. They are **game preferences**, not
psychological diagnoses. The user should be able to inspect, edit, reset, or keep
them local-only.

## Visual semantics

A useful avatar encoding should be stable enough to learn:

| Evidence | Avatar behavior | Failure/unknown behavior |
| --- | --- | --- |
| respiration waveform/rate | subtle torso breathing | breathing animation returns to neutral |
| movement intensity | stride/bounce/idle energy | neutral idle |
| cardiac estimate | very subtle ambient pulse/glow | glow disabled |
| sleep-stage history | environment/time-of-day narrative | "not enough evidence" world state |
| facial expression features | eyes/mouth/head animation | neutral face |
| self-reported mood | optional palette/activity preference | no inferred replacement |

The avatar should never visibly "look sick" because a noisy estimator crossed a
threshold.

## Experiment products valuable to other researchers

The long-term package should expose three things:

1. **PhysioBench**: adapter -> split -> baseline -> null -> calibration -> report.
2. **Evidence Replay**: a browser viewer that can replay any experiment result and
   show where the model knew, abstained, or failed.
3. **PhysioPersona**: an optional 3-D consumer of the exact same evidence objects.

That creates a useful separation: researchers can use the benchmark without the
avatar, educators can use the replay without hardware, and the portfolio can use
all three.

## Next implementation milestones

### P1: external sleep benchmark

- SPAN dataset adapter.
- dataset manifest and hash.
- patient-disjoint split audit.
- respiratory reproduction report.
- sleep/wake baseline.
- four-stage baseline.
- generated benchmark JSON and replay fixture.

### P2: secure web bridge

- local companion service;
- pairing and token scopes;
- outbound encrypted session;
- persona snapshot stream;
- origin and sequence validation;
- disconnect/revoke UI.

### P3: browser expression adapter

- local face landmarks/blendshapes;
- no upload by default;
- derived-feature schema;
- explicit self-report channel;
- ablation against WiFi-only visual state.

### P4: longitudinal persona

- user-editable preference state;
- cute task/world policy driven by explicit interaction history;
- local persistence by default;
- export/reset controls;
- no clinical or psychological interpretation.
