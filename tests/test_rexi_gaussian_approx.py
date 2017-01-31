from rexi_coefficient_python import GaussianApproximation
import pytest

def test_rexi_gaussian_approx():
    ga = GaussianApproximation()
    h = 1
    for x in range(10):
        exact = ga.evalGaussian(x, h)
        approx = ga.approxGaussian(x, h)
        assert abs(exact - approx) < 1.e-12
