# Gemini AI Studio build prompt: WiFisio Atlas

You are implementing a new production-quality frontend application inside the
existing RuView/PhysioAtlas repository.

Read these files before changing code:

- `CLAUDE.md`
- `PROJECT_GOALS_PHYSIOATLAS.md`
- `docs/physioatlas/WIFISIO_ATLAS_PRODUCT_BLUEPRINT.md`
- `docs/physioatlas/DUAL_BAND_HOUSEHOLD_CSI_PLAN.md`
- `docs/physioatlas/CSI_BFI_FUSION_STUDY.md`
- `docs/physioatlas/REFLECTOR_FUSION_FRONTEND_SPEC.md`
- `docs/physioatlas/CLAIM_BOUNDARIES.md`
- `docs/physioatlas/RESEARCH_HUB.md`
- `docs/physioatlas/HOUSEHOLD_RESEARCH.md`
- `physioatlas/research_hub.py`
- `physioatlas/household_live.py`
- `physioatlas/household_csi.py`
- `physioatlas/household_csi_schema.py`
- `physioatlas/household_bfi_schema.py`
- `physioatlas/rf_fusion_schema.py`
- `physioatlas/calibration_reflector_schema.py`
- `physioatlas/tracking.py`
- `physioatlas/static/research_hub/index.html`
- `tests/physioatlas/test_research_hub.py`
- `tests/physioatlas/test_tracking_household.py`

## Product

Build **WiFisio Atlas**, a private, local-first household spatial physiology
research observatory.

Tagline:

> A private spatial physiology research observatory for the home.

This is a research application, not a medical product. It visualizes what the
sensors and models actually support, including uncertainty, abstention,
calibration deficits, stale data, and unknown people.

Use **InnerLoop** as the name of the single-person workspace within the larger
household application.

## Outcome

Create a new application at:

```text
apps/wifisio-atlas/
```

Do not replace or break the dependency-light existing Research Hub. The new
application consumes its current API through an adapter and includes a
deterministic mock/replay adapter for design and testing.

The result must feel polished and distinctive, but the visual design must never
make a prediction look like a direct measurement.

## Technology

Use:

- React
- TypeScript with strict mode
- Vite
- React Router
- TanStack Query for server state
- Zod for runtime API validation
- lightweight local UI state only where necessary
- Lucide icons
- ECharts, uPlot, or another performant local chart library
- React Three Fiber only for the optional 3D inspection view
- Vitest and Testing Library
- Playwright and axe for end-to-end accessibility checks

Requirements:

- no runtime CDN
- no remote fonts
- no analytics or telemetry
- no cloud dependency
- no secrets in browser code
- no giant single component or single state file
- no `any` for domain data
- no silent fallback from malformed live data to plausible mock values
- the complete product remains useful when WebGL is unavailable

Use CSS variables for design tokens. A utility CSS framework is allowed only if
it is installed locally and does not turn the UI into a generic template.

## Existing API

The current local server exposes:

- `GET /api/health`
- `GET /api/hub`
- `GET /api/tracks`
- `GET /api/members`
- `GET /api/modalities`
- `GET /api/calibration`
- `GET /api/reflector`
- `GET /api/rf-fusion`
- `GET /api/studies`
- `GET /api/experiments`
- `GET /api/events`
- `POST /api/actions`

The current full-state schema is
`physioatlas.research-hub-state.v1`. Preserve compatibility. Create a typed
adapter layer so a future `wifisio.*.v1` API can be introduced without
rewriting components.

Create:

- a transport interface
- a current REST polling implementation
- a deterministic mock/replay implementation
- an event-stream interface ready for SSE or WebSocket, even if the current
  server still polls
- Zod schemas for every current endpoint used
- explicit parse-error, disconnected, empty, stale, and unsupported-version
  states

The UI must not infer data freshness from its own render time.

The repository now also contains a research baseline that produces
`physioatlas.household-observation.v2` JSONL from synchronized, calibrated,
fixed-channel multi-link CSI. Display its `measurement_status`, `confidence`,
`observability`, `frequency_bands`, `synchronization_verified`,
`separation_condition_number`, covariance, and provenance. A calibrated zone
position is not continuous pose. Never hide an unverified clock, single-band
capture, ill-conditioned separation, or experimentally disabled heart rate.

Implement `/calibration/reflector` exactly according to
`REFLECTOR_FUSION_FRONTEND_SPEC.md`. The current API exposes reflector and
fusion sections separately; keep that adapter and be ready to consume the
optional additive `reflector_fusion` projection when introduced. Never animate
unreported reflector motion, never render an occupant for an abstained fusion
decision, and never present a queued calibration action as physical movement.

## Claim and display rules

Every metric or visual layer belongs to exactly one class:

1. `measured`: directly recorded by a named reference sensor
2. `derived`: deterministic processing of a measured stream
3. `predicted`: model output evaluated against a named target
4. `hypothesized`: proposed interpretation without adequate validation

Implement a reusable `EvidenceBadge` and `MetricProvenance` disclosure. Use
consistent colors, icons, and text for these four classes. Color alone is not
enough.

Never add:

- diagnostic labels or disease probabilities
- blood-pressure values
- arrhythmia claims
- direct organ images or organ visibility claims
- fictional furniture permittivity
- a person name for an unknown or ambiguous track
- a zero where the correct value is `not observable`
- a live pose skeleton that is actually a decorative animation

If the current API does not provide provenance or uncertainty for a scalar,
label it `legacy estimate`, show which fields are unavailable, and do not invent
them.

Demo and replay data must have a persistent header banner and scene watermark.
A small mode chip is not sufficient.

## Primary user questions

The default screen must answer these in under five seconds:

1. How many tracks are observable now?
2. Where are they in the home?
3. Which tracks are anonymous, ambiguous, or consented enrolled matches?
4. Which estimates are current and sufficiently observable?
5. Is any sensor, clock, calibration, or data stream degraded?

Research detail is available through drill-down, not displayed at equal visual
weight on the default view.

## Routes

Implement:

```text
/                         Live Atlas
/people                   People
/people/:memberOrTrackId  Person or anonymous-track detail
/rooms                    Rooms and sensors
/calibration              Calibration Lab
/studies                  Studies
/sessions                 Sessions and evidence
/privacy                  Privacy and provenance
/settings                 Local application settings
```

Unknown tracks are valid route targets. Never require a household identity to
inspect a track.

## App shell

Desktop:

- compact left navigation rail
- top status bar with mode, connection, last snapshot age, and privacy state
- main canvas/content area
- optional contextual inspector on the right

Tablet and mobile:

- bottom navigation for the four primary destinations
- inspector becomes a full-height sheet
- live floor plan remains pan/zoom capable
- all primary actions remain keyboard and touch accessible

The shell must support light reduction, high contrast, reduced motion, and
WebGL-disabled modes. The default visual theme is dark.

## Live Atlas

This is the product center.

Create a 2D floor-plan view first. It must show:

- room boundaries and named zones from a local room model
- fixed sensing nodes and active RF links
- anonymous track markers and enrolled aliases
- position uncertainty as an ellipse or halo
- track lifecycle: tentative, confirmed, occluded, or lost
- last-update age
- room transitions
- signal coverage/observability overlay
- sensor or link degradation

Track selection opens an inspector with:

- alias or anonymous track ID
- identity state and reason
- best/second-best margin when available
- position and velocity
- respiration, heart, motion, signal quality, and observability
- freshness and source for every value
- a short rolling timeline
- buttons to open full detail or start a permitted calibration workflow

Add an optional 3D view for expert inspection. It may show a room, nodes, links,
track points, and uncertainty volumes. Use an abstract body marker unless the
API supplies a validated pose with provenance. Label illustrative geometry.

Do not make the 3D scene the only way to understand the live state.

## Person and track detail

Create a focused detail page rather than a wall of equal cards.

Header:

- display alias or anonymous ID
- current room and last-seen time
- consent and identity state
- calibration readiness
- current observability

Content:

- respiratory timeline
- cardiac timeline
- motion and posture timeline
- location history
- accepted, abstained, stale, and not-observable windows
- reference-versus-RF comparison when a reference exists
- provenance and algorithm/calibration version
- annotations and session links

The chart legend must distinguish:

- measured reference
- RF estimate
- uncertainty interval
- excluded or low-observability regions

Never connect across missing data as if it were continuous.

## Rooms and sensors

Create:

- room cards with occupancy, quality, and calibration age
- a sensor/link table with state, packet rate, loss, RSSI, noise/quality, clock
  offset/drift, firmware, and last seen where available
- a floor-plan editor shell that can load and save a local room model later
- clear "API does not provide this yet" placeholders rather than invented data

## Calibration Lab

Make deficits actionable.

For each participant and modality show:

- required sessions
- completed sessions
- conditions covered
- held-out session availability
- score and status
- missing reference
- room/day/posture gaps
- low observability
- ambiguous identity margin
- expired or revoked consent

Include guided workflow cards for:

- empty-room baseline
- quiet seated
- quiet supine
- paced breathing
- speaking and small movement
- walking/crossing
- leave and re-enter
- fan, curtain, pet, and consented unknown-person controls

Buttons may enqueue only the existing allow-listed actions. Unsupported actions
are disabled with an explanation.

## Studies and evidence

Represent all five existing studies:

- respiration and sleep research
- cardiac mechanics
- pulse propagation
- mobility and gait
- organ-correlated motion

Each study card/page shows:

- target
- reference sensors
- sessions and held-out dimensions
- baselines
- required nulls
- observability/coverage
- primary metrics
- infrastructure result
- scientific result
- failure mode
- claim boundary
- artifact/checkpoint identifiers

`not_evaluable` is a first-class result, not a visual error.

## Privacy

Show:

- local-only state
- consent requirement
- enrolled-only identity mode
- unknown-person behavior
- retention settings
- current consent status per enrolled alias
- identity template storage status
- recent audit events

Do not persist raw state, aliases, embeddings, or physiological history in
`localStorage`. Only non-sensitive display preferences may be persisted.

## Design language

Aim for a calm scientific instrument, not a cyberpunk game dashboard.

Tokens:

- background: near-black graphite/deep navy
- panels: subtly elevated blue-gray
- measured data: cyan
- healthy infrastructure: green
- degraded/abstained: amber
- predicted/model output: violet
- faults: red
- primary text: high-contrast cool white
- secondary text: accessible blue-gray

Use:

- generous spacing
- a restrained 8 px grid
- 12-16 px radii
- subtle borders instead of excessive glow
- tabular numerals for live measurements
- smooth transitions under 200 ms
- no nonessential looping animation
- skeleton loading only when data is genuinely loading

Charts should be calm, legible, and update without reflow. Use a fixed time
window, stable axes where appropriate, and accessible textual summaries.

## Component boundaries

At minimum, create reusable components for:

- `ApplicationModeBanner`
- `ConnectionStatus`
- `SnapshotAge`
- `EvidenceBadge`
- `MetricValue`
- `MetricProvenance`
- `ObservabilityIndicator`
- `UncertaintyInterval`
- `TrackMarker`
- `TrackInspector`
- `FloorPlan`
- `SensorNode`
- `RfLink`
- `TimeSeriesChart`
- `CalibrationReadiness`
- `StudyResult`
- `EmptyState`
- `ErrorState`
- `UnsupportedField`

Keep data adaptation out of rendering components.

## Seed scenarios

The deterministic mock adapter must include:

1. empty room
2. one consented enrolled member
3. two family members crossing paths
4. one enrolled member plus an unknown visitor
5. ambiguous household identity
6. lost sensor and stale track
7. respiration observable but cardiac not observable
8. replay mode with a measured reference
9. all five study result states, including `not_evaluable`
10. revoked consent causing an alias to detach

Add a developer scenario picker that is absent from production builds.

## Accessibility

Meet WCAG 2.2 AA for the implemented screens.

- full keyboard navigation
- visible focus
- semantic landmarks and headings
- accessible names for icon-only controls
- contrast-compliant text and controls
- no color-only status
- reduced-motion support
- chart summaries and data-table alternatives
- a 2D alternative to every 3D-only interaction
- touch targets of at least 44 by 44 CSS pixels on mobile

## Tests

Add unit tests for:

- every Zod adapter and legacy-field conversion
- not-observable and stale transitions
- measured/derived/predicted/hypothesized display
- unknown and ambiguous identity
- demo/replay watermark
- revoked consent
- malformed and unsupported API payloads
- no false zero for absent metrics

Add end-to-end tests for:

- navigation and responsive shell
- selecting an anonymous track from the floor plan
- disconnect/reconnect behavior
- two tracks crossing without the UI swapping their displayed identities
- keyboard-only calibration navigation
- WebGL-disabled fallback
- accessibility scans of all primary routes

## Verification

Before changing code, run:

```powershell
python -m physioatlas doctor --repo .
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -o addopts='' tests/physioatlas -q
```

After implementation, run:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -o addopts='' tests/physioatlas -q
python -m physioatlas household-smoke-test --output outputs/physioatlas/household-smoke
npm --prefix apps/wifisio-atlas run typecheck
npm --prefix apps/wifisio-atlas run lint
npm --prefix apps/wifisio-atlas run test
npm --prefix apps/wifisio-atlas run build
npm --prefix apps/wifisio-atlas run test:e2e
```

If an inherited command fails before your changes, record the exact failure. Do
not weaken a validator, remove an assertion, or relabel an unavailable metric to
make the UI appear complete.

## Implementation sequence

Work in this order:

1. scaffold and tooling
2. typed schemas and transport adapters
3. deterministic scenarios
4. app shell and routing
5. Live Atlas 2D view
6. track inspector and person detail
7. rooms/sensors and calibration
8. studies/sessions/privacy
9. optional 3D inspection
10. accessibility, performance, tests, and documentation

At the end, report:

- files created and changed
- API assumptions
- fields the current backend does not supply
- mock-only visuals
- tests and build results
- remaining blockers to real household sensing

The task is complete only when the app is usable with the current local API,
works fully with deterministic replay, visibly preserves unknown people and
not-observable measurements, and passes its frontend checks without weakening
the existing PhysioAtlas gates.
