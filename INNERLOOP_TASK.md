# InnerLoop task: validate PhysioAtlas v0.3

Run the repository's complete research-infrastructure gate without optimizing
model performance or weakening any validator.

## Command

```bash
make physioatlas-ci
```

## Required evidence

- doctor reports ready
- every PhysioAtlas test passes
- smoke report status is passed
- acquisition, calibration, pretraining, RF field, active sensing,
  uncertainty, distillation, propagation, ultrasound, privacy, federated, and
  dashboard integration checks are valid
- train and validation groups are disjoint
- required baselines and nulls are present
- checkpoint hash and replay pass
- every injected fault is detected

Return exact commands, exit codes, test count, dataset and checkpoint hashes,
held-out groups, final metrics, null metrics, integration failures or warnings,
and the paths to all machine-readable reports. Synthetic results must be labeled
software verification only.
