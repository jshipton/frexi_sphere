#
#  Created on: 2 Aug 2015
#      Author: Martin Schreiber <schreiberx@gmail.com>
#
# Changelog:
#     2016-05-14: Converted to Python
#                   Source: https://github.com/schreiberx/sweet
#

import math
import cmath
import numpy

#
# This class provides the weights and coefficients for the
# approximation of a Gaussian
#
# \f$
#       exp(-(x*x)/(4*h*h))/sqrt(4 \pi)
# \f$
#
# with a sum over complex rational functions.
#
# See e.g. Near optimal rational approximations of large data sets, Damle et. al.
#

class REXIParameters(object):

    """    
    mu and a coefficients from
    "A high-order time-parallel scheme for solving wave propagation problems via the direct construction of an approximate time-evolution operator", Haut et.al.
    """

    mu = -4.315321510875024 + 1j*0
    L = 11
    a = [
        -1.0845749544592896e-7 + 1j*2.77075431662228e-8,
        1.858753344202957e-8 + 1j*-9.105375434750162e-7,
        3.6743713227243024e-6 + 1j*7.073284346322969e-7,
        -2.7990058083347696e-6 + 1j*0.0000112564827639346,
        0.000014918577548849352 + 1j*-0.0000316278486761932,
        -0.0010751767283285608 + 1j*-0.00047282220513073084,
        0.003816465653840016 + 1j*0.017839810396560574,
        0.12124105653274578 + 1j*-0.12327042473830248,
        -0.9774980792734348 + 1j*-0.1877130220537587,
        1.3432866123333178 + 1j*3.2034715228495942,
        4.072408546157305 + 1j*-6.123755543580666,
        -9.442699917778205 + 1j*0.,
        4.072408620272648 + 1j*6.123755841848161,
        1.3432860877712938 + 1j*-3.2034712658530275,
        -0.9774985292598916 + 1j*0.18771238018072134,
        0.1212417070363373 + 1j*0.12326987628935386,
        0.0038169724770333343 + 1j*-0.017839242222443888,
        -0.0010756025812659208 + 1j*0.0004731874917343858,
        0.000014713754789095218 + 1j*0.000031358475831136815,
        -2.659323898804944e-6 + 1j*-0.000011341571201752273,
        3.6970377676364553e-6 + 1j*-6.517457477594937e-7,
        3.883933649142257e-9 + 1j*9.128496023863376e-7,
        -1.0816457995911385e-7 + 1j*-2.954309729192276e-8
    ]

def b_coefficients(h, M):
    m = numpy.arange(-M,M+1, dtype=numpy.complex128)
    return numpy.exp(h*h, dtype=numpy.complex128)*numpy.exp(-1j*m*h, dtype=numpy.complex128)


def RexiCoefficients(h, M, reduce_to_half=False):
    print("Computing Rexi coefficients.")
    params = REXIParameters()
    L = params.L
    mu = params.mu
    a = params.a
    b = b_coefficients(h, M)
    N = M + L

    import numpy
    alpha = numpy.zeros((2*N+1,), dtype=numpy.complex128)
    beta_re = numpy.zeros((2*N+1,), dtype=numpy.complex128)
    beta_im = numpy.zeros((2*N+1,), dtype=numpy.complex128)

    for l in range(-L, L+1):
        for m in range(-M, M+1):
            n = l+m
            alpha[n+N] = h*(mu + 1j*n);
            beta_re[n+N] += b[m+M].real*h*a[l+L];
            beta_im[n+N] += b[m+M].imag*h*a[l+L];

    if reduce_to_half:
        # reduce the computational amount to its half,
        # see understanding REXI in the documentation folder
        alpha = alpha[:N+1]
        beta_re = beta_re[:N+1]
        beta_im = beta_im[:N+1]

        # don't rescale beta_re[N]
        for i in range(N):
            beta_re[i] *= 2.0
            beta_im[i] *= 2.0

    beta = numpy.concatenate((beta_re + 1j*beta_im,
                              -beta_re + 1j*beta_im))/2
    alpha = numpy.concatenate((alpha, -alpha))

    return alpha, beta
