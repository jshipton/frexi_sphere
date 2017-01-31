from frexi_sphere import REXIParameters, REXI, b_coefficients
from cmath import exp, sqrt, pi
import pytest

h = 0.2
M = 64
b = b_coefficients(h, M)
alpha, beta_re, beta_im = REXI(h, M)
params = REXIParameters()
mu = params.mu
L = params.L
a = params.a

def approx_e_ix(x):
    sum_re = 0;
    sum_im = 0;

    S = len(alpha)

    # Split computation into real part of \f$ cos(x) \f$ and imaginary part \f$ sin(x) \f$
    for n in range(0, S):
	denom = (1j*x + alpha[n]);
	sum_re += (beta_re[n] / denom).real
	sum_im += (beta_im[n] / denom).real

    return sum_re + 1j*sum_im;

def approx_e_ix2(x):
    sum = 0
    for m in range(-M, M+1):
	sum += b[m+M] * approxGaussian(x+float(m)*h, h)

    return sum

def approxGaussian(x, h):
    """
    evaluate approximation of Gaussian basis function
    with sum of complex rational functions
    """
    # scale x, since it depends linearly on h:
    # x^2 ~ h^2
    x /= h

    sum = 0

    for l in range(0, len(a)):
	j = l-L

	# WORKS with max error 7.15344e-13
	sum += (a[l]/(1j*x + mu + 1j*j)).real

    return sum


def test_exponential_approx():

    for x in range(-int(h*M)+1, int(h*M)):
        exact = exp(1j*x)
        approx = approx_e_ix2(x)
        assert abs(exact - approx) < 5.e-8


def test_rexi_gaussian_approx():
    h = 1
    for x in range(10):
        exact = exp(-(x*x)/(4.0*h*h))/sqrt(4.0*pi)
        approx = approxGaussian(x, h)
        assert abs(exact - approx) < 1.e-12


def test_rexi():

    for x in range(-int(h*M)+1, int(h*M)):
        exact = exp(1j*x)
        approx = approx_e_ix(x)
        assert abs(exact - approx) < 1.5
