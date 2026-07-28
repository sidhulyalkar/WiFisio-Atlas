# Scientific and safety boundaries

PhysioAtlas output must use one of these labels:

1. **Measured:** directly recorded by a named sensor.
2. **Derived:** deterministic processing of a measured signal.
3. **Predicted:** model output evaluated against a named ground truth.
4. **Hypothesized:** proposed physiological interpretation not yet validated.

Never relabel a predicted waveform as a measured organ signal.

The following claims require dedicated validation and are disabled by default:

- Medical diagnosis or disease screening
- Blood-pressure measurement
- Arrhythmia detection
- Internal-organ localization or segmentation
- Blood-flow velocity measurement
- Replacement of ECG, ultrasound, PPG, or regulated monitoring equipment

Every experiment record includes `research_only: true` and
`clinical_claim_allowed: false`. Changing those values in a data manifest does
not itself establish validity.
