import pytest
from iot_detector.demo import make_synthetic
from iot_detector.data import split_by_file


@pytest.fixture(scope="session")
def synth():
    return make_synthetic(n=3000, seed=0)


@pytest.fixture(scope="session")
def split(synth):
    return split_by_file(synth)
