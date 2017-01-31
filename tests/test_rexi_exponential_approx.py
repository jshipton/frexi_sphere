from frexi_sphere import ExponentialApproximation
import pytest

def test_exponential_approx():

    h = 0.2
    M = 32
    e = ExponentialApproximation(h, M)
    max = 0.
    for x in range(-int(h*M)+1, int(h*M)):
        exact = e.eval_e_ix(x)
        approx = e.approx_e_ix(x)
        assert abs(exact - approx) < 5.e-8
