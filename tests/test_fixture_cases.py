import math

from tests.fixtures import load_all


def test_fixture_cases():
    for name, (inputs, expected) in load_all():
        result = inputs["x"] + inputs["y"]
        assert math.isclose(result, expected["sum"], abs_tol=1e-6), name
