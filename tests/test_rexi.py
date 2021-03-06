from frexi_sphere.rexi_coefficients import REXIParameters, RexiCoefficients, b_coefficients
from cmath import exp, sqrt, pi
import pytest

params = REXIParameters()
mu = params.mu
L = params.L
a = params.a

def approx_e_ix(x, h, M, use_Gaussian_approx):
    b = b_coefficients(h, M)

    sum = 0
    if use_Gaussian_approx:
        for m in range(-M, M+1):
            sum += b[m+M] * approxGaussian(x+m*h, h)
    else:
        alpha, beta = RexiCoefficients(h, M, False)
        for n in range(len(alpha)):
            denom = (1j*x + alpha[n]);
            sum += beta[n] / denom

    return sum


def approx_phi(x, h, M, n):
    b = b_coefficients(h, M, n)

    sum = 0
    for m in range(-M, M+1):
        sum += b[m+M] * exp(-((x+m*h)*(x+m*h))/(4.0*h*h))/sqrt(4.0*pi)

    return sum

def approxGaussian(x, h):
    """
    evaluate approximation of Gaussian basis function
    with sum of complex rational functions
    """
    x /= h

    sum = 0

    for l in range(0, len(a)):
        j = l-L

        # WORKS with max error 7.15344e-13
        sum += (a[l]/(1j*x + mu + 1j*j)).real

    return sum


def test_exponential_approx():
    h = 0.2
    M = 64
    for x in range(-int(h*M)+1, int(h*M)):
        exact = exp(1j*x)
        approx = approx_e_ix(x, h, M, True)
        assert abs(exact - approx) < 2.e-11


def test_rexi_gaussian_approx():
    h = 0.2
    for x in range(10):
        exact = exp(-(x*x)/(4.0*h*h))/sqrt(4.0*pi)
        approx = approxGaussian(x, h)
        assert abs(exact - approx) < 7.15344e-13


def test_rexi_exponential_approx():
    h = 0.2
    M = 64
    for x in range(-int(h*M)+1, int(h*M)):
        exact = exp(1j*x)
        approx = approx_e_ix(x, h, M, False)
        assert abs(exact - approx) < 2.e-11
