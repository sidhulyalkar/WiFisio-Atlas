# PhysioAtlas project goals

## North star

Build an auditable, consent-first RF-to-physiology research platform that
identifies the boundary of what can and cannot be recovered from ambient and
dedicated RF under explicit geometry, synchronization, supervision,
observability, uncertainty, and privacy controls.

## Current release objective

PhysioAtlas v0.4 establishes a household research substrate around the v0.3
measurement stack:

- anonymous multi-person tracks before identity
- explicitly enrolled, open-set household matching
- disjoint calibration and identity-evaluation sessions
- per-person and per-modality calibration reports
- rolling live physiology, motion, quality, and observability histories
- five executable priority-study analyzers
- JSONL and local UDP observation bridges
- local Research Hub APIs, UI, and safe action worker
- InnerLoop/Gemini task contracts and fail-closed privacy testing

## Scientific success criteria

A result matters only when it:

- uses synchronized measured ground truth
- uses explicit, current participant consent
- keeps enrollment and evaluation sessions disjoint
- preserves unknown people rather than forcing identity
- beats simple RF and auxiliary baselines
- survives time-shift, label-permutation, and wrong-geometry controls
- generalizes across subjects and at least one deployment domain
- reports coverage as well as accepted identity accuracy
- reports observability, uncertainty, and abstentions
- preserves negative findings and failed runs
- stays within its declared measurement and claim boundary

## Immediate empirical milestone

Run a small repeated-measures household pilot with compatible CSI hardware,
independent localization, respiratory and cardiac references, at least three
enrollment sessions and one untouched validation session per consenting member.
Synthetic household separability is only a software test. The next decisive
question is whether real RF measurements remain distinguishable across days,
positions, rooms, clothing, movement, and unknown-person controls.
