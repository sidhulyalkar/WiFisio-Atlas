# Research Hub architecture

The local Research Hub is a file-backed, dependency-light observatory designed
for InnerLoop and agent-driven UI iteration.

## Data surfaces

- `GET /api/hub`: complete state
- `GET /api/tracks`: anonymous and enrolled live tracks
- `GET /api/members`: enrolled baselines
- `GET /api/modalities`: acquisition health
- `GET /api/calibration`: per-person/per-modality calibration
- `GET /api/studies`: priority-study summaries
- `GET /api/experiments`: experiment ledger
- `GET /api/events`: local event history
- `POST /api/actions`: allow-listed research actions only

The UI never executes shell commands. Buttons append allow-listed requests to a
local action queue. `research-hub-worker` processes that queue with explicit
dataset and registry paths.

## InnerLoop integration

InnerLoop or Gemini may improve the UI, but must retain:

- anonymous-first tracks
- explicit consent state
- observability and uncertainty
- modality calibration quality
- reference-versus-RF distinction
- research-only claim boundaries
- no remote analytics or third-party CDN dependency by default

Acceptance tests are in `tests/physioatlas/test_research_hub.py` and the complete
household smoke test.
