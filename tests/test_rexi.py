from rexi_coefficient_python import REXI
import pytest

def test_rexi():

    h = 0.2
    M = 64
    rexi = REXI(h, M)
    for x in range(-int(h*M)+1, int(h*M)):
        exact = rexi.eval_e_ix(x)
        approx = rexi.approx_e_ix(x)
        assert abs(exact - approx) < 1.5
