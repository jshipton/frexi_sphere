from frexi_sphere import ExponentialApproximation
from cmath import exp
import pytest

def test_exponential_approx():

    h = 0.2
    M = 32
    e = ExponentialApproximation(h, M)
    max = 0.
    for x in range(-int(h*M)+1, int(h*M)):
        exact = exp(1j*x)
        approx = e.approx_e_ix(x)
        assert abs(exact - approx) < 5.e-8
