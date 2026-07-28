import numpy as np

from physioatlas.household import (
    BiologicalSummary,
    HouseholdMemberProfile,
    HouseholdRegistry,
    ModalityIdentityProfile,
)
from physioatlas.schema import Modality
from physioatlas.tracking import MultiPersonTracker, TrackObservation


def _registry() -> HouseholdRegistry:
    vectors = [np.eye(4)[0], np.eye(4)[1]]
    members = []
    for index, vector in enumerate(vectors):
        members.append(
            HouseholdMemberProfile(
                member_id=f"member-{index}",
                display_alias=f"Member {index}",
                consent_id=f"consent-{index}",
                enrolled_at_utc="2026-01-01T00:00:00+00:00",
                profiles=[
                    ModalityIdentityProfile(
                        modality=Modality.WIFI_CSI,
                        centroid=vector.tolist(),
                        within_distance_q95=0.01,
                        acceptance_threshold=0.15,
                        samples=4,
                    )
                ],
                biological_summary=BiologicalSummary(sessions=4),
            )
        )
    return HouseholdRegistry(
        household_id="test",
        created_at_utc="2026-01-01T00:00:00+00:00",
        members=members,
        open_set_margin=0.15,
    )


def test_tracker_preserves_tracks_and_enrolled_identity():
    tracker = MultiPersonTracker(_registry(), identity_after_observations=2, position_gate_m=2.0)
    for step in range(4):
        tracks = tracker.update(
            [
                TrackObservation(float(step), np.array([0.1 * step, 0.0]), {Modality.WIFI_CSI: np.array([1.0, 0.01, 0, 0])}, heart_rate_bpm=65),
                TrackObservation(float(step), np.array([3.0 - 0.1 * step, 0.0]), {Modality.WIFI_CSI: np.array([0.01, 1.0, 0, 0])}, heart_rate_bpm=78),
            ]
        )
    assert len(tracks) == 2
    assert {track.member_id for track in tracks} == {"member-0", "member-1"}
    assert all(track.observations >= 4 for track in tracks)


def test_tracker_keeps_unenrolled_person_anonymous():
    tracker = MultiPersonTracker(_registry(), identity_after_observations=2)
    for step in range(3):
        tracks = tracker.update(
            [TrackObservation(float(step), np.array([1.0, 1.0]), {Modality.WIFI_CSI: np.array([0, 0, 1.0, 0])})]
        )
    assert len(tracks) == 1
    assert tracks[0].member_id is None
    assert tracks[0].track_id.startswith("unknown-track-")


def test_tracker_rejects_incompatible_embedding_version():
    tracker = MultiPersonTracker(_registry(), identity_after_observations=2)
    for step in range(3):
        tracks = tracker.update(
            [
                TrackObservation(
                    float(step),
                    np.array([0.0, 0.0]),
                    {Modality.WIFI_CSI: np.array([1.0, 0.01, 0, 0])},
                    embedding_versions={
                        Modality.WIFI_CSI: "physioatlas.household-csi-identity.v1"
                    },
                )
            ]
        )
    assert tracks[0].member_id is None
    assert tracks[0].identity_reason == "no_compatible_profiles"
