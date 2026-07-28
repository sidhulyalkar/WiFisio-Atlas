# Reflector and CSI–BFI fusion frontend specification

Status: implementation-ready UI specification; research-only
Surface: WiFisio Atlas `/calibration/reflector`, with a compact read-only card on Live Atlas
Source contracts: `reflector-trajectory.v1`, `reflector-telemetry.v1`, `reflector-health.v1`, `household-bfi-feature.v1`, `rf-zone-evidence.v1`, and `csi-bfi-fusion-decision.v1`

## 1. Non-negotiable display boundary

- The commanded position is intent; only `actual_position` is actuator telemetry.
- Update the reflector icon only when a telemetry sample is received or selected in replay. Never tween, extrapolate, dead-reckon, or procedurally animate it.
- A line joining samples is not observed motion. Use discrete points and zero-order-held values labelled “last reported,” with gaps rendered as gaps.
- `synthetic: true` produces a persistent banner and scene watermark: **SIMULATION — SOFTWARE VERIFICATION ONLY**.
- `synthetic: false` means physical telemetry, not validated RF or biological accuracy. Show **PHYSICAL RESEARCH HARDWARE — UNVALIDATED**.
- `research_only` must be `true`. Reject the section if it is absent or false.
- Reflector health establishes actuator trajectory/synchronization health only; it is not RF, occupancy, localization, or physiology accuracy.
- BFI features are non-identifying. Do not derive an alias, body, trajectory, respiration, or identity from a BFI layer.
- An abstained decision never draws an occupant marker. Preserve probabilities as muted diagnostic evidence, overlaid with the reason.

## 2. Information architecture

Desktop uses a 12-column, 8 px grid:

- Rows 1–2: persistent mode/claim banner, connection and backend-reported freshness.
- Columns 1–8: `ReflectorRoomView`, a 2D SVG/canvas room instrument.
- Columns 9–12: `ReflectorInspector` with Safety, Health, Alignment, and Evidence tabs.
- Full width below: synchronized `CalibrationTimeline` and replay controls.
- Bottom: accessible data tables for telemetry, link layers, evidence, and receipts.

Tablet places the inspector below the room. Mobile uses the room first, then a full-width inspector sheet and timeline. The map remains useful at 320 CSS px. The Live Atlas card shows only safety state, calibration validity, last sample age, current position, fusion status, and a link to this route.

## 3. Two-dimensional room instrument

### Room, rail, and waypoints

- Draw room bounds and named zones from the supplied room model.
- For `motion_unit: "millimeters"`, transform installation `origin_m + axis_unit_vector × position/1000` into room coordinates. Draw a solid rail, end stops, signed ticks, home, and labelled waypoints.
- For `motion_unit: "degrees"`, draw an anchored turntable, orientation arc, angular ticks, home bearing, and waypoint bearings. Do not draw a linear rail.
- Waypoints are plan geometry, not reached locations. Use hollow diamonds; mark “reached” only from matching actual telemetry within a backend-provided tolerance.
- A solid reflector icon marks actual telemetry; a dashed outline marks command. An error bracket joins values from the same sample.
- Do not clamp out-of-bounds actual position: show its direction at the edge and a numeric fault disclosure.
- Pan, zoom, reset, fit-room, and layer visibility are view-only. Default is north-up, stable scale, and no automatic camera movement.

### Dwell and motor state

`motor_power_enabled: false` is the only evidence for “motor off.” A trajectory segment named `dwell` alone is insufficient.

- Motor-off dwell: teal square icon and label **Motor off · RF dwell**.
- Dwell commanded but motor powered: amber **Dwell requested · motor powered**.
- No current telemetry: gray **Motor state unknown**.
- Nonzero reported velocity while motor power is off: red contradiction badge;
  preserve both raw values and do not “correct” either.

### Safety rendering

| State | Token and shape | Required label |
|---|---|---|
| `safe_hold` | slate, pause icon | Safe hold |
| `armed` | amber, shield outline | Armed — motion permitted |
| `running` | green, directional chevron | Running |
| `estopped` | red, octagon, hatched scene | E-stop active |
| `faulted` | red, warning triangle, hatched scene | Faulted |

`estop_active`, any `fault_code`, or active limit switches remain separately
visible even if `safety_state` is inconsistent. Fault and E-stop banners override
healthy styling. No decorative motion, glow pulse, or looping rail animation.

## 4. Calibration health and alignment

The Health tab presents a verdict first:

- `valid_for_rf_calibration: true`: **Actuator capture eligible**, not “accurate.”
- `false`: **Calibration rejected**, followed by every invalidation reason.
- Metrics: estimated lag (s), trajectory RMSE in installation units, maximum
  absolute error, repeatability RMSE, sample count, and fault count.
- `null` is **Not computed**, never zero. Show threshold only if supplied by API.
- Synthetic and physical health may never be visually merged in one trend.

The Alignment tab shows receipt ID, reference clock, validity, source clocks,
offset, drift, jitter, residual RMSE, matched anchors, valid time span, and
invalidation reasons. A missing receipt is **Alignment receipt unavailable**.
An invalid receipt forces a degraded banner; the UI does not independently
change a backend `FusionDecision`.

## 5. CSI, band, BFI, and fusion layers

Modality has a color plus a line pattern:

- CSI 2.4 GHz: `#2DD4BF`, solid;
- CSI 5 GHz: `#60A5FA`, long dash;
- CSI 6 GHz, if supplied later: `#F59E0B`, dash-dot;
- BFI: `#C084FC`, dotted with square endpoints;
- fused probability: neutral violet fill plus labeled percentage.

These are modality tokens, not evidence-class colors. Continue using the global
EvidenceBadge tokens: measured cyan, derived teal, predicted violet,
hypothesized amber, fault red. Every legend entry contains text and pattern.

Each response link is an AP/client/sensor connection, not a ray-traced path.
Width represents the named `response_metric`; opacity represents observability.
Show quality numerically. Do not spatially interpolate a heat map unless the API
supplies a gridded field and its method/claim class. Zone probabilities fill
only the declared zone polygons.

The Evidence tab supports `2.4`, `5`, `BFI`, and `Fusion` toggles, plus:

- per-source probability bars in identical zone order;
- source quality, observability, synchronization uncertainty, timestamp, and ID;
- fused probability, active threshold, sources present, conflict, and decision reason;
- conflict above threshold as amber/red split arrows between source bars;
- `abstained` as diagonal hatching and a prominent reason, with no person icon;
- missing modality as **Not present**, not probability zero.

Respiration, heart rate, and motion appear only inside source evidence that
actually provides them. They retain provenance and `not observable` behavior.

## 6. Timeline and deterministic replay

Use aligned lanes with a shared cursor:

1. plan segment and repeat;
2. commanded position points;
3. actual position points and error;
4. motor power, velocity, current, limits, safety/fault;
5. CSI 2.4, CSI 5, and BFI evidence availability/observability;
6. alignment anchors and validity;
7. fused/abstained decisions, active zones, and conflict.

Controls are pause, live edge, previous/next reported sample, scrub, 0.25×–4×
replay, and time-window selection. Replay advances only to recorded timestamps.
When the cursor lies between reports, retain the last point only with a visible
“last reported at …” label. Crossing a data gap does not connect charts or move
the reflector. Export copies the selected receipt/data IDs, not raw sensitive
arrays, unless a separately authorized backend export exists.

## 7. Additive Research Hub state contract

`physioatlas.research-hub-state.v1` remains unchanged. The backend may add one
optional top-level key, `reflector_fusion`. Its absence means unsupported, not
empty or disconnected. All objects reject unknown fields in the frontend Zod
adapter; nullable fields below accept JSON `null`, not omission.

```text
reflector_fusion: {
  schema_version: "wifisio.reflector-fusion-ui.v1",
  updated_at_utc: ISO-8601 string,
  mode: "live" | "replay",
  research_only: true,
  synthetic: boolean,
  freshness: { age_s: number>=0, stale_after_s: number>0, is_stale: boolean },
  room: {
    environment_id: string,
    bounds_m: { x_min:number, x_max:number, y_min:number, y_max:number },
    zones: [{ zone_id:string, label:string, polygon_m:[[number,number],...] }],
    installation: ReflectorInstallation projection,
    limits: ReflectorSafetyLimits projection,
    waypoints: [{ waypoint_id:string, label:string, position:number, kind:string }]
  },
  reflector: {
    device: ReflectorDeviceIdentity projection,
    plan: ReflectorTrajectoryPlan | null,
    latest_telemetry: ReflectorTelemetrySample | null,
    telemetry_window: ReflectorTelemetrySample[],
    health: ReflectorCalibrationHealth | null
  },
  alignment_receipt: AlignmentReceiptView | null,
  response_layers: ResponseLayerView[],
  zone_evidence: ZoneEvidence[],
  latest_fusion: FusionDecision | null,
  fusion_window: FusionDecision[],
  available_actions: AvailableAction[]
}
```

“Projection” means every field from the named strict Pydantic model, unchanged
in name and unit. `AlignmentReceiptView` is exactly:

```text
{ schema_version:"wifisio.alignment-receipt-view.v1", receipt_id:string,
  reference_clock_id:string, valid:boolean, valid_from_s:number,
  valid_to_s:number, maximum_time_skew_s:number>=0,
  sources:[{ source:"wifi_csi"|"wifi_bfi"|"reflector", clock_id:string,
    offset_s:number, drift_ppm:number, jitter_s:number>=0,
    residual_rmse_s:number>=0, matched_anchors:integer>=0 }],
  invalidation_reasons:string[], provenance_sha256:64-lowercase-hex,
  research_only:true }
```

`ResponseLayerView` is exactly:

```text
{ layer_id:string, source:"wifi_csi"|"wifi_bfi",
  frequency_band:"2.4ghz"|"5ghz"|"6ghz", link_id:string,
  client_id:string|null, channel:integer>=1, from_point_m:[number,number],
  to_point_m:[number,number], quality:number[0..1],
  observability:number[0..1], status:"observable"|"not_observable"|"stale",
  response_metric:{ name:string, value:number, unit:string,
    normalization:string, claim_class:"measured"|"derived"|"predicted"|"hypothesized" },
  timestamp_s:number>=0, evidence_id:string|null }
```

`AvailableAction` is exactly:

```text
{ action:"run_calibration", enabled:boolean, hardware_authorized:boolean,
  requires_confirmation:boolean, reason:string }
```

Example additive payload:

```json
{
  "reflector_fusion": {
    "schema_version": "wifisio.reflector-fusion-ui.v1",
    "updated_at_utc": "2026-07-28T21:04:12Z",
    "mode": "replay",
    "research_only": true,
    "synthetic": true,
    "freshness": {"age_s": 0.2, "stale_after_s": 2.0, "is_stale": false},
    "room": {
      "environment_id": "living-room-v1",
      "bounds_m": {"x_min": 0, "x_max": 6, "y_min": 0, "y_max": 4},
      "zones": [{"zone_id": "sofa", "label": "Sofa", "polygon_m": [[3,1],[5,1],[5,2],[3,2]]}],
      "installation": {"environment_id":"living-room-v1","origin_m":[1,1,0.4],"axis_unit_vector":[1,0,0],"motion_unit":"millimeters","home_position":0,"reflector_diameter_m":0.2,"reflector_material":"synthetic-reference"},
      "limits": {"minimum_position":0,"maximum_position":1000,"maximum_velocity_per_s":100,"maximum_acceleration_per_s2":200,"maximum_motor_current_a":1.5,"fail_closed":true,"require_estop_input":true},
      "waypoints": [{"waypoint_id":"dwell-500","label":"Center dwell","position":500,"kind":"dwell"}]
    },
    "reflector": {
      "device": {"device_id":"sim-01","hardware_revision":"sim-v1","firmware_version":"1.0","driver_name":"deterministic-simulated-reflector","telemetry_clock_id":"sim-clock","serial_number":null},
      "plan": null,
      "latest_telemetry": {"schema_version":"physioatlas.reflector-telemetry.v1","sequence":42,"monotonic_timestamp_s":4.2,"clock_id":"sim-clock","plan_id":"sweep-01","segment_index":1,"repeat_index":0,"phase":0.4,"commanded_position":500,"actual_position":493,"actual_velocity_per_s":0,"motor_current_a":0.08,"motor_power_enabled":false,"minimum_limit_active":false,"maximum_limit_active":false,"estop_active":false,"fault_code":null,"safety_state":"running","plan_complete":false,"synthetic":true,"research_only":true},
      "telemetry_window": [],
      "health": null
    },
    "alignment_receipt": null,
    "response_layers": [{"layer_id":"csi-west-5","source":"wifi_csi","frequency_band":"5ghz","link_id":"west-5","client_id":null,"channel":36,"from_point_m":[0.2,2],"to_point_m":[5.8,2],"quality":0.82,"observability":0.76,"status":"observable","response_metric":{"name":"perturbation_energy","value":1.8,"unit":"robust_z","normalization":"empty-room-v1","claim_class":"derived"},"timestamp_s":4.2,"evidence_id":"csi-42"}],
    "zone_evidence": [],
    "latest_fusion": {"schema_version":"physioatlas.csi-bfi-fusion-decision.v1","timestamp_s":4.2,"status":"abstained","reason":"missing_required_source","active_zones":[],"fused_probabilities":{},"cross_modal_conflict":null,"observability":0.42,"sources_present":["wifi_csi"],"evidence_ids":["csi-42"],"household_observations":[],"research_only":true},
    "fusion_window": [],
    "available_actions": [{"action":"run_calibration","enabled":false,"hardware_authorized":false,"requires_confirmation":true,"reason":"Replay is view-only"}]
  }
}
```

## 8. Interaction authority

Always view-only: pan/zoom, layer toggles, inspector selection, timeline/replay,
table expansion, copy IDs, and local display preferences.

The current Research Hub supports only queued `run_calibration`; it does not
expose reflector arm, move, clear-fault, or E-stop APIs. Therefore:

- show **Queue calibration** only when the exact available action is enabled,
  hardware-authorized, and confirmed in a modal naming environment, plan, and
  synthetic/physical mode;
- send only `POST /api/actions` with `action: "run_calibration"` and displayed
  parameters; a `202 queued` response is not motion confirmation;
- never expose Arm, Jog, Home, Clear fault, or remote E-stop controls;
- tell users to use the physical E-stop. A web button must never look like a
  safety-rated stop;
- replay, stale, disconnected, faulted, or estopped state disables all writes;
- absent capability data denies writes by default.

## 9. WebGL, performance, and accessibility

The primary room uses SVG or Canvas 2D. Optional WebGL may add height but cannot add facts or controls. On failure, preserve every zone, rail, sample, link, probability, safety state, and interaction in 2D and tables. Cap visible history and decimate only chart rendering; never discard source samples from inspector access.

Meet WCAG 2.2 AA: semantic landmarks, keyboard layer controls, 44 px touch
targets, visible focus, text/pattern redundant status, high-contrast mode,
screen-reader summaries, table alternatives, and `prefers-reduced-motion`.
Reduced motion removes panel transitions; reflector telemetry never animates in
either mode.

## 10. Acceptance tests

1. Optional-section absence renders “Reflector fusion not supported” without
   failing the rest of `research-hub-state.v1`.
2. Zod rejects false/missing `research_only`, extra fields, unsupported versions,
   invalid probability ranges, and malformed hashes.
3. Synthetic banner and scene watermark remain visible on every tab and viewport.
4. Physical mode says unvalidated and never inherits synthetic health/results.
5. Millimeter geometry maps origin/axis correctly; degree geometry uses an arc.
6. The actual icon changes only on a reported/selected sample; no CSS or WebGL
   interpolation occurs, including at replay speeds and across gaps.
7. Commanded and actual positions are visually and accessibly distinguishable.
8. Motor-off dwell requires `motor_power_enabled:false`; contradictions surface.
9. E-stop, fault, limits, stale data, and invalid health override healthy styling.
10. Null health metrics read “Not computed”; missing modalities read “Not present.”
11. 2.4, 5, and BFI layers have distinct text, color, and patterns; hidden layers
    remain available in the data table.
12. Invalid/missing alignment is explicit and never rewritten by the frontend.
13. Abstention draws no occupant, states the reason, and retains diagnostic bars.
14. Replay emits no POST; queued calibration requires capability plus confirmation.
15. A `202 queued` action displays “Queued,” never “Moving” or “Calibration valid.”
16. Disconnect/reconnect and backend-reported stale transitions preserve the last
    sample as stale without fabricated updates.
17. WebGL-disabled and Canvas-failed modes retain the complete SVG/table workflow.
18. Keyboard-only, reduced-motion, 200% zoom, mobile, and axe scans pass WCAG 2.2 AA.
