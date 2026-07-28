# Models, pretraining, and privileged supervision

## Canonical RF tensor

The dataset produces `[time, link, channel]` RF tensors, link-observation masks,
quality masks, and twelve geometry features per link. Amplitude/phase and IQ
streams are converted without discarding complex structure.

## Label-free RF pretraining

`RFMaskedFoundationEncoder` independently masks channels in two views,
reconstructs the hidden RF values, aligns the view embeddings, and applies a
variance floor. It exports a 128-dimensional embedding without using the
physiological target value.

This is a software baseline, not a pretrained biological foundation model.
Meaningful transfer requires diverse real subjects, rooms, devices, and
protocols.

## Supervised waveform model

The `complex_link` architecture includes:

- per-link value encoding
- geometry encoding
- link and observation masks
- temporal recurrent modeling
- waveform prediction
- observability probability
- log variance for heteroscedastic uncertainty

A compact flattened baseline remains available for ablation.

## Privileged teacher distillation

Camera pose, mmWave maps, ECG, PPG, respiratory belts, or ultrasound features
can be projected into the RF embedding space during training. The loss is
masked when a teacher is unavailable and reports teacher coverage. Every study
should measure the performance gap after the privileged teacher is removed.
