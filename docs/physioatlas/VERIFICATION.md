# Verification contract

A run is valid only when:

- the dataset contract passes
- the claim boundary and `research_only` flag are present
- declared training and validation groups are disjoint
- metrics and serialized numbers are finite
- required baselines and null controls exist
- the checkpoint exists, matches its SHA-256, loads strictly, and executes
- the exact config snapshot and environment record are retained
- privacy and propagation analyses are recorded when applicable

`physioatlas verify-run <run-directory>` implements these checks. Verification
cannot certify data collection that never occurred or a sensor capability not
represented in the evidence.
