# Clock, geometry, and calibration memory

PhysioAtlas separates three calibration layers.

## Clock calibration

An affine mapping converts source timestamps into a reference clock and reports
offset, drift, and residual error. This is appropriate for clocks with stable
linear drift over a session. Piecewise drift or timestamp resets must be split
into separate segments.

## Geometry calibration

A Kabsch rigid transform maps measured fiducials from a sensor coordinate frame
into the declared room frame. Reflections are rejected, and the fit reports
RMSE. At least three non-collinear correspondences are recommended.

## Calibration memory

`CalibrationProfile` stores:

- environment and hardware identities
- geometry fingerprint
- empty-room features
- clock and rigid transforms
- uncertainty profile
- metadata and a content hash

Profiles are selected by environment match, hardware overlap, and geometry
distance. Hash tampering is a fail-closed error. A selected profile is a prior,
not permission to skip a fresh calibration check.
