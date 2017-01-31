from frexi_sphere import REXI
from cmath import exp
import pytest

def test_rexi():

    h = 0.2
    M = 64
    rexi = REXI(h, M)
    for x in range(-int(h*M)+1, int(h*M)):
        exact = exp(1j*x)
        approx = rexi.approx_e_ix(x)
        assert abs(exact - approx) < 1.5
