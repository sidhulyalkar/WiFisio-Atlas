# OpenHome RF Atlas: CSI–BFI fusion study protocol

Status: proposed, research-only protocol; scope: household occupancy, zone localization, respiration, opt-in identity,
calibration drift, and abstention
Claim boundary: this document specifies experiments. It does not establish
biological, clinical, localization, occupancy, or identity accuracy.

## 1. Research question

Can fixed-link, dual-band CSI and passively captured beamforming-feedback
information (BFI) provide complementary evidence for multi-person household
sensing across real environmental changes, and can a passive motorized
reflector determine when a calibration remains usable?

The intended contribution is not merely “CSI plus BFI.” CSI and BFI have
already been recorded together for comparison, and BFI sensing has been
demonstrated for localization, activity, and identity. The proposed novelty is:

1. paired CSI–BFI fusion evaluated under locked, cross-domain household splits;
2. an active but non-transmitting reflector as a repeatable transfer standard;
3. calibrated selective prediction that reports coverage and abstains under
   drift, weak observability, modality disagreement, or unknown identity;
4. an auditable, de-identified benchmark containing null controls and failures,
   not only successful sessions.

## 2. Established evidence versus proposed work

### Established

- In 802.11ac/ax explicit beamforming, a beamformee estimates the channel from a sounding exchange and returns a compressed representation of the feedback matrix as quantized angles. The matrix is related to the right singular vectors of CSI; it is not the original complex CSI matrix. See [Wi-BFI](https://arxiv.org/abs/2309.04408) and the [bi-directional BFM study](https://arxiv.org/abs/2112.06695).
- Wi-BFI reconstructs BFI from captured beamforming angles for 802.11ac/ax, 20/40/80/160 MHz, and SU/MU-MIMO. It groups reports by standard, bandwidth, and device before decoding.
- BFI report frequency is AP- and traffic-dependent and is much lower than CSI packet frequency in the BFId study. BFId recorded simultaneous CSI and BFI, four perspectives, and 197 participants, and showed that BFI can disclose identity. Its authors identify environment dynamics, long-term robustness, and interference as open questions. See the [BFId paper](https://publikationen.bibliothek.kit.edu/1000185756/168100988) and [dataset record](https://radar.kit.edu/radar/en/dataset/hdcds6a8fukdennd).
- Direction matters: the bi-directional BFM study found unequal sensing ability in opposite report directions and improved localization by combining them.
- Si-FI demonstrates standard-compliant BFI for simultaneous multi-subject activity recognition in three environments with three subjects. This is feasibility evidence, not evidence that BFI separates household respiration or same-zone occupants. See [Si-FI](https://www.sciencedirect.com/science/article/pii/S1389128625008734).
- WiFi sensing degrades across environment, subject, and device shifts. [WiTTA-Bench](https://openaccess.thecvf.com/content/CVPR2026/html/Li_WiTTA-Bench_Benchmarking_Test-Time_Adaptation_for_WiFi_Sensing_CVPR_2026_paper.html) benchmarks those shifts and test-time adaptation.
- The Archer BE400 is documented as dual-band Wi-Fi 7 with beamforming and 2.4/5 GHz MLO, but TP-Link does not document a research CSI export interface. See the [official BE400 page](https://www.tp-link.com/us/home-networking/wifi-router/archer-be400/).

### Proposed and unvalidated

- That CSI and BFI improve one another for household person separation or vital
  estimation.
- That a movable reflector predicts human-sensing accuracy or recalibration
  need.
- That cross-band fusion improves over added same-band spatial links.
- That BFI contains reliable respiration evidence at its observed report rate.
- That any identity representation transfers across homes, devices, clothing,
  time, or carried objects.

Every result must retain this distinction. Synthetic and replay results verify
software only.

## 3. What BFI provides—and what capture cannot assume

An explicit sounding exchange includes an announcement, a null-data sounding packet, and one or more compressed beamforming reports. A report describes an AP–client propagation perspective, even when a separate monitor records it. That differs from CSI measured on the AP–CSI-receiver path.

Capture requirements:

- First run a capability survey for every AP, client, band, channel, bandwidth, MIMO mode, and firmware version. Record actual report frames; never infer BFI availability from a “beamforming” product label.
- Use one fixed-channel monitor radio per simultaneously observed channel. A single monitor cannot continuously capture 2.4 and 5 GHz at once by hopping.
- Preserve raw pcap, radiotap time, transmitter/receiver MACs or irreversible study pseudonyms, sequence/control fields, standard, channel, bandwidth, direction, antenna/stream dimensions, tone grouping, quantization, SNR if present, and decoder version.
- Preserve variable inter-report time. Do not manufacture a constant BFI rate.
- Treat report scheduling, missing reports, retransmission, client sleep, roaming, MLO link selection, and firmware changes as observed covariates.
- Quantized BFI is compressed channel orientation, not full CSI. It may omit scale and channel information needed for physiology. Do not label BFI-derived rates as measured unless independently validated.
- A report can be passively observable in studied 802.11 deployments. That is a privacy risk, not blanket permission to capture neighbors. Filter to enrolled study equipment at ingestion and delete out-of-scope frames.

## 4. Synchronization and alignment

CSI clocks, monitor-pcap clocks, physiology references, reflector control, and position references are separate clock domains. Fusion is invalid without an alignment receipt.

1. Timestamp every sample at acquisition; preserve device-monotonic and host time when available.
2. Feed collectors a shared wired time source when practical. PTP/NTP is a prior, not proof of packet-level alignment. Wi-Fi software PTP behavior has been measured in the [USENIX ATC study](https://www.usenix.org/conference/atc21/presentation/chen).
3. Create radio anchors visible to both streams: retain identifiers and times for sounding exchanges whose NDP/data packet is observed by CSI receivers and whose report is observed by BFI monitors. Use robust affine clock fits, not one offset.
4. Add a hardware sync pulse to the reflector controller, reference logger, and acquisition host when supported.
5. Estimate offset, drift, jitter, matched-anchor count, loss, and residuals per session. Store them in an immutable alignment receipt.
6. Permit complex/coherent early fusion only when hardware demonstrates phase/time coherence. Default to noncoherent, window-level feature fusion.
7. Fail closed when residual timing uncertainty exceeds a preregistered task limit; respiration can tolerate wider windows than crossing-event tracking.

As a negative control, inject held-out offsets and drift after alignment. A
fusion method that does not degrade is likely ignoring one modality or leaking
labels.

## 5. Three fusion families

All families receive identical time windows, data availability masks, packet
budgets, and train/validation/test partitions.

### Feature-level (“early”) fusion

Use separate CSI and BFI encoders because dimensions, sampling rates, and physics differ. Resample only timestamped latent sequences onto a common window, concatenate or cross-attend with missing-modality masks, then estimate the task and uncertainty.

Test advantage: can learn cross-modal interactions.
Risk: easiest route to domain overfit, timing leakage, and sparse-BFI failure; require modality dropout during training.

### Decision-level (“late”) fusion

Each modality independently emits a calibrated posterior or regression distribution, quality score, and abstention. A learned gate combines only available outputs.

Test advantage: debuggable and tolerant of missing streams.
Risk: discards complementary subdecision structure and can become a disguised winner-take-all selector.

### Evidence-level selective fusion

Each modality emits task evidence plus provenance and observability. A fixed preregistered rule combines compatible evidence, discounts correlated links, and returns `accepted`, `degraded`, `disagreement`, `not_observable`, or `recalibration_required`. Do not multiply posteriors under an untested independence assumption.

This is the recommended first production path because it preserves the PhysioAtlas evidence boundary. Learned early fusion remains experimental until it beats this baseline on locked cross-domain tests and selective risk.

## 6. Motorized reflector specification

Use a passive corner reflector or plate on a position-controlled carriage. It
does not transmit, impersonate a client, or carry a person identifier.

Minimum prototype:

- interchangeable matte nonconductive control and conductive reflector;
- approximately 25–35 cm reflector aperture as a starting prototype, explicitly
  tuned in pilot measurements rather than treated as a known optimum;
- repeatable 2D waypoints or a floor rail plus discrete orientations;
- encoder-derived position, independent fiducial reference, hard end stops,
  bumper/obstacle detection, emergency stop, low speed, and child/pet lockout;
- controller-monotonic timestamps, trajectory ID, waypoint, orientation, motor
  state, position uncertainty, and firmware/configuration hash;
- motor power off during each RF dwell to prevent motor EMI and vibration from
  becoming the learned signature.

Core experiments:

1. **Empty response map:** reflector absent, nonconductive sham, conductive
   reflector stationary at each waypoint, and repeated closed-loop trajectories.
2. **Timing impulse:** pseudorandom dwell/move schedule; correlate CSI/BFI
   changes with controller events to estimate end-to-end lag and clock drift.
3. **Sensitivity/dead zones:** per link, band, and BFI perspective, estimate
   response magnitude, repeatability, SNR, missingness, and spatial gradients.
4. **Cross-band transfer:** run the identical physical path with simultaneous
   fixed 2.4/5 GHz collectors; compare maps without equating subcarriers.
5. **Day-to-day stability:** repeat before household activity, after activity,
   and on later days without changing geometry.
6. **Controlled drift:** move one furniture item, open a door, change client or
   receiver, change AP channel, and move one sensor separately. Test whether
   reflector-map change predicts task-performance loss.
7. **Human coexistence:** compare empty, one stationary person outside the
   path, and normal household motion. Never adapt using held-out task labels.

Primary reflector output is a calibrated response tensor indexed by waypoint,
orientation, modality, band, link/perspective, and time. Derived outputs are
coverage maps, repeatability, a drift score with uncertainty, and a
recalibration decision. Do not call it RF ground truth: its scattering differs
from a human body.

## 7. Factorial evaluation

Use a preregistered balanced or blocked factorial design. When the complete
factorial is too large, publish the deterministic subset selection.

| Factor | Required levels |
|---|---|
| Modality/fusion | CSI only; BFI only; early; late; evidence |
| Band | 2.4 GHz; 5 GHz; both |
| Spatial evidence | 1, 3, 6, all CSI links; 1, 2, all BFI perspectives |
| Calibration | none; empty-room; reflector map; daily reflector update |
| Synchronization | aligned; timestamp-only; injected offset/drift |
| People | 0, 1, 2, 3+; enrolled and unknown |
| Geometry | separate zones; adjacent zones; same zone; crossing; occlusion |
| Drift | unchanged; furniture; door/HVAC; channel; client; sensor relocation |

Control information volume: compare at equal time duration and additionally at
equal packet/report budgets. Report whether improvement comes from modality
diversity, more perspectives, or simply more samples.

Locked partitions must be disjoint by session and include:

- next-day and multi-week holdout;
- leave-one-room and leave-one-home out;
- leave-one-CSI-device and leave-one-client/AP firmware family out;
- leave-one-person out for non-identity tasks;
- enrollment-day versus later-day and known versus unknown people for identity.

No window from one continuous capture may cross partitions. Fit calibration,
normalization, thresholds, adaptation, and identity enrollment on the allowed
training/calibration partition only.

## 8. Required null controls and hard cases

- truly empty room and one forgotten observer check;
- traffic generator active with no motion, and no traffic;
- fan/HVAC, curtains, door motion, foliage, appliance vibration;
- pet, robot vacuum, rolling chair, carried reflector-like metal object;
- reflector absent, sham object, stationary reflector, motor moving but RF
  sampling only after motor-off dwell;
- dropped link, BFI starvation, channel change, corrupted timestamps;
- swapped device identifiers and shuffled modality pairing;
- human reference present but RF path deliberately blocked;
- two people in one calibrated zone, identity swap at a crossing, people
  entering together, and one person stopping while another moves.

Same-zone separation and crossing identity continuity are separate primary hard
sets, not examples averaged into easy room-wide scores.

## 9. Metrics and acceptance reporting

Report point estimates, participant/day-clustered bootstrap confidence
intervals, per-domain results, and selective performance versus coverage.

- **Occupancy:** exact-count accuracy, count MAE, per-count confusion matrix,
  presence sensitivity/specificity, false alarms per empty-room hour.
- **Localization/tracking:** median and 95th-percentile position error, GOSPA or
  OSPA for unknown cardinality, HOTA, identity switches, fragmentation, and
  track latency. Score same-zone and crossing sets separately.
- **Respiration:** paired valid-window MAE/RMSE in breaths/minute, correlation,
  Bland–Altman bias/limits, gross-error rate, time-to-estimate, and coverage.
  Motion-corrupted and multi-person results remain separate.
- **Identity:** verification ROC, EER, false-match and false-nonmatch rates,
  open-set identification rate, OSCR, and false accepts per unknown-person-hour.
  Closed-set accuracy alone is insufficient.
- **Calibration/drift:** waypoint repeatability, map correlation and normalized
  error, changed/unchanged drift AUROC, detection delay, false alarms/day, and
  correlation between drift score and downstream performance loss.
- **Uncertainty/abstention:** ECE, Brier or NLL where applicable, risk–coverage
  curve, area under risk–coverage, error rate at fixed coverage, and reasons for
  abstention.
- **Systems:** observed CSI/BFI rates, packet loss, BFI starvation duration,
  alignment residual, CPU/memory, end-to-end latency, storage, and network load.

Any accuracy headline must name its split, reference sensor, number of people,
number of homes, coverage, and confidence interval.

## 10. Preregistered hypotheses

Register direction, primary metric, alpha, exclusions, stopping rule, model
selection budget, and analysis code hash before opening the locked test set.
Estimate sample size from a pilot and use family-wise or false-discovery
correction for secondary comparisons.

- **H1:** evidence-level CSI–BFI fusion lowers GOSPA versus the best unimodal
  system on cross-day, separate-zone multi-person sessions at matched coverage.
- **H2:** fusion lowers occupancy count MAE versus the best unimodal system
  under BFI starvation and one dropped CSI link.
- **H3:** adding BFI does not materially improve respiration MAE at matched
  coverage. This deliberate null hypothesis prevents unsupported physiology
  claims.
- **H4:** reflector-map drift predicts downstream localization degradation on
  unseen controlled room changes better than elapsed time or RSSI drift alone.
- **H5:** reflector-gated recalibration lowers selective risk without using
  task labels, compared with fixed periodic recalibration.
- **H6:** open-set identity fusion lowers false accepts per unknown-person-hour
  at matched known-person true-accept rate, compared with either modality.
- **H7:** same-zone and crossing performance remains significantly worse than
  separate-zone performance; no success claim is permitted unless those sets
  meet a separately preregistered threshold.

Publish null and negative results. H3 and H7 are scientifically useful even if
the proposed system does not overcome them.

## 11. Privacy, consent, and open-set rules

- Default output is anonymous occupancy and tracks. Identity is separately
  opt-in, purpose-bound, revocable, and disabled for visitors.
- Treat raw CSI, BFI, MAC addresses, gait/identity embeddings, and synchronized
  references as sensitive. Pseudonymize device addresses at acquisition with a
  study-specific keyed transform; do not release the key.
- Retain raw frames only by explicit policy. Publish de-identified derived
  features only after evaluating re-identification risk; BFId establishes that
  BFI itself is identifying data.
- Enrollment and evaluation captures are disjoint. Embedding versions and
  modality versions must match; otherwise classification abstains.
- Unknown-person evidence can produce only `unknown`, never the nearest family
  member. Conflicting modalities produce `disagreement`.
- Consent withdrawal invalidates future use and triggers deletion/tombstoning
  according to the registry and artifact-retention policy.

## 12. Staged implementation architecture

### Stage A — capture proof

Add typed, strict, versioned contracts:

- `BfiFrameV1`: native report metadata, quantized angles and/or reconstructed
  matrix, pcap provenance, decoder version, timing, missingness;
- `AlignmentReceiptV1`: clock model, anchors, residuals, hashes, validity span;
- `ReflectorRunV1`: controller/trajectory hashes, waypoints, reference
  uncertainty, motor-off dwell status;
- `FusionEvidenceV1`: modality evidence, covariance/confidence, observability,
  agreement, gate decision, and lineage.

Adapters preserve native values and raw timing. Validation rejects changing
band/channel/dimensions inside a declared fixed link unless a new segment is
created. Artifact manifests contain input, configuration, code, model, and
decoder hashes.

Exit gate: real pcap BFI from the actual BE400/client matrix decodes
reproducibly; paired CSI frames and alignment receipts replay bit-for-bit.

### Stage B — reflector atlas

Implement a controller event log, safety state machine, trajectory manifest,
paired capture orchestrator, per-modality response-map builder, null/sham runs,
and drift report. Keep reflector calibration separate from human enrollment.

Exit gate: repeatability and timing tests pass on physical hardware; synthetic
tests are not substituted.

### Stage C — unimodal baselines and evidence fusion

Freeze CSI-only and BFI-only baselines first. Add alignment-window joins,
quality masks, calibrated unimodal outputs, then fixed evidence-level fusion.
Emit the existing `physioatlas.household-observation.v2` fields for position,
covariance, embeddings, quality, observability, respiration/heart-rate status,
and provenance. Additive fields may reference alignment, BFI, reflector, and
fusion receipts; schema migration must remain explicit.

Exit gate: null controls, missing-modality tests, version mismatch tests,
open-set identity tests, and abstention tests all fail closed.

### Stage D — learned fusion and locked study

Add modality-specific encoders, early and late fusion, modality dropout, and
calibration without evaluation labels. Freeze checkpoints and split manifests,
record hashes, run the factorial ablations, and produce a claim table linked to
artifacts.

Exit gate: no model is promoted because of development-set visualization.
Only preregistered locked-test results can advance an evidence level.

### Stage E — public benchmark

Release contracts, capture/decoder versions, reflector CAD/controller design,
trajectory manifests, calibration/null subsets, split manifests, evaluation
code, and provenance receipts. Release human-derived data only at the consented
privacy tier; otherwise publish replay-safe features or aggregate results.

The frontend consumes reports and receipts, never silently recomputes evidence.
It must show modality availability, synchronization quality, reflector
trajectory/state, drift and recalibration status, per-task uncertainty,
abstention reason, and the explicit research-only claim boundary.

## 13. Go/no-go sequence

1. Confirm real BFI emission and decodability for the BE400 and selected clients
   on each desired band.
2. Demonstrate paired clocks and stable alignment residuals.
3. Demonstrate reflector/sham discrimination and repeatability.
4. Freeze unimodal baselines before tuning fusion.
5. Pass empty-room and nuisance null controls.
6. Pass separate-zone two-person localization before attempting same-zone
   physiological attribution.
7. Keep multi-person heart rate and body-pose claims disabled until independently
   referenced studies meet preregistered acceptance gates.

If any step fails, preserve the dataset and report the boundary. A reliable
“not observable” result is a first-class OpenHome RF Atlas contribution.
