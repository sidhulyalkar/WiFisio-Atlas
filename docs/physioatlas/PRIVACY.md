# Privacy, consent, and distributed experiments

Physiological RF data can reveal occupancy, activity, identity, and health
correlates even without a camera. PhysioAtlas therefore stores consent scopes
inside the dataset contract.

Implemented controls:

- consent presence, expiration, and modality-scope validation
- HMAC-based pseudonymous subject IDs for exports
- optional modality removal during export
- append-only hash-chain audit logs
- deterministic clipped and optionally noised local federated aggregation

The current federated implementation is not secure aggregation and does not
provide transport security, malicious-client robustness, or a formal privacy
budget. Those require a dedicated networking and cryptographic deployment.
