# Observability and uncertainty

PhysioAtlas distinguishes three questions:

1. Was the synchronized target sample observed?
2. Does the RF geometry support a prediction?
3. How wide should the prediction interval be?

The model emits observability and log variance alongside the waveform.
`fit_split_conformal` calibrates an absolute-residual interval on a held-out
calibration set. `should_abstain` returns true when observability is low, the
interval is too wide, the sample is out of distribution, or any required value
is non-finite.

Conformal coverage is only meaningful under exchangeability assumptions. It
must be recalibrated across rooms, hardware, subjects, and protocol shifts.
