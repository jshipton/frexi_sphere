from frexi_sphere import GaussianApproximation
from math import exp, sqrt, pi
import pytest

def test_rexi_gaussian_approx():
    ga = GaussianApproximation()
    h = 1
    for x in range(10):
        exact = exp(-(x*x)/(4.0*h*h))/sqrt(4.0*pi)
        approx = ga.approxGaussian(x, h)
        assert abs(exact - approx) < 1.e-12
