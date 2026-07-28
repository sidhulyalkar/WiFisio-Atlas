import numpy as np
import pytest

from physioatlas.geometry import anatomy_link_prior, geometry_summary, link_feature_matrix
from physioatlas.schema import (
    AnatomyRegion,
    Modality,
    RFLink,
    Sensor,
    SensorGeometry,
    Vec3,
)


def make_geometry() -> SensorGeometry:
    return SensorGeometry(
        sensors=[
            Sensor(
                sensor_id="tx",
                modality=Modality.WIFI_CSI,
                position_m=Vec3(x=-1, y=0, z=1),
                sample_rate_hz=100,
            ),
            Sensor(
                sensor_id="rx",
                modality=Modality.WIFI_CSI,
                position_m=Vec3(x=1, y=0, z=1),
                sample_rate_hz=100,
            ),
        ],
        rf_links=[
            RFLink(
                link_id="link",
                tx_sensor_id="tx",
                rx_sensor_id="rx",
                center_frequency_hz=5.18e9,
                bandwidth_hz=40e6,
                n_subcarriers=114,
            )
        ],
        anatomy_anchors_m={
            AnatomyRegion.THORAX: Vec3(x=0, y=0, z=1.2),
        },
    )


def test_geometry_features_are_fixed_and_finite():
    geometry = make_geometry()
    ids, matrix = link_feature_matrix(geometry)
    assert ids == ["link"]
    assert matrix.shape == (1, 12)
    assert np.isfinite(matrix).all()
    summary = geometry_summary(geometry)
    assert summary.shape == (24 + len(AnatomyRegion),)
    assert np.isfinite(summary).all()


def test_anatomy_prior_is_normalized():
    regions, prior = anatomy_link_prior(make_geometry())
    assert regions == [AnatomyRegion.THORAX]
    assert prior.shape == (1, 1)
    assert prior[0, 0] == pytest.approx(1.0)


def test_unknown_link_sensor_is_rejected():
    with pytest.raises(ValueError, match="unknown sensor"):
        SensorGeometry(
            sensors=[
                Sensor(
                    sensor_id="tx",
                    modality=Modality.WIFI_CSI,
                    position_m=Vec3(x=0, y=0, z=0),
                    sample_rate_hz=10,
                )
            ],
            rf_links=[
                RFLink(
                    link_id="bad",
                    tx_sensor_id="tx",
                    rx_sensor_id="missing",
                    center_frequency_hz=2.4e9,
                    bandwidth_hz=20e6,
                    n_subcarriers=56,
                )
            ],
        )
