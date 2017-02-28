#
#  Created on: 2 Aug 2015
#      Author: Martin Schreiber <schreiberx@gmail.com>
#
# Changelog:
# 	2016-05-14: Converted to Python
#                   Source: https://github.com/schreiberx/sweet
#

import math
import cmath

#
# This class provides the weights and coefficients for the
# approximation of a Gaussian
#
# \f$
# 	  exp(-(x*x)/(4*h*h))/sqrt(4 \pi)
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

def b_coefficients(h, M, n=0):
    if n == 0:
        return [math.exp(h*h)*cmath.exp(-1j*(float(m)*h)) for m in range(-M, M+1)]
    elif n == 1:
        from scipy import pi, cos, sin, exp, integrate
        def expr_real(xi):
            return cos(2*pi*m*h*xi)*exp(4*pi**2*xi**2*h**2)
        def expr_imag(xi):
            return sin(2*pi*m*h*xi)*exp(4*pi**2*xi**2*h**2)
        b = []
        for m in range(-M, M+1):
            re, err= integrate.quadrature(expr_real, max(-1./(2*pi), -1./(2*h)), 0, tol=1.e-10, miniter=300)
            im, err = integrate.quadrature(expr_imag, max(-1./(2*pi), -1./(2*h)), 0, tol=1.e-10, miniter=300)
            b.append(2*pi*(re + 1j*im))
        return b
    elif n == 2:
        from scipy import pi, cos, sin, exp, integrate
        def expr_real(xi):
            return cos(2*pi*m*h*xi)*exp(4*pi**2*xi**2*h**2)*(xi+1/(2*pi))
        def expr_imag(xi):
            return sin(2*pi*m*h*xi)*exp(4*pi**2*xi**2*h**2)*(xi+1/(2*pi))
        b = []
        for m in range(-M, M+1):
            re, err= integrate.quadrature(expr_real, max(-1./(2*pi), -1./(2*h)), 0, tol=1.e-10, miniter=300)
            im, err = integrate.quadrature(expr_imag, max(-1./(2*pi), -1./(2*h)), 0, tol=1.e-10, miniter=300)
            b.append(4*pi**2*(re + 1j*im))
        return b        
    else:
        print "n must be 0, 1 or 2"

def RexiCoefficients(h, M, n=0, reduce_to_half=False):
    params = REXIParameters()
    L = params.L
    mu = params.mu
    a = params.a
    b = b_coefficients(h, M, n)
    N = M + L

    alpha = [0 for i in range(0, 2*N+1)]
    beta_re = [0 for i in range(0, 2*N+1)]
    beta_im = [0 for i in range(0, 2*N+1)]

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

    return alpha, beta_re, beta_im


# This class computes an approximation of an exponential \f$ e^{ix} \f$ by
# a combination of Gaussians.
#
# The approximation is given by
#
# \f$
# 		e^{ix} - \sum_{m=-M}^{M}{b_m \psi_h(x+m*h)}
# \f$
#
# see eq. (3.1) in
# "A high-order time-parallel scheme for solving wave propagation problems via the direct construction of an approximate time-evolution operator", Haut et.al.
#
# Here, the coefficents b_m are given by the coefficents \f$ c_m \f$ (Yes, it's the \f$ c_m \f$ here!)
# in the equation below equation (3.4):
#
# \f$
#    c_m := \int_{-1/(2h)}^{1/2h}  exp(-2 \pi i m h \xi) F(\xi)/\psi_h(\xi) d \xi
# \f$
#
# with F and \f$ \psi \f$ the functions f and \f$ \psi \f$ in Fourier space
#

	#
	# See Section 3.1 for approx. of general function
	#
	# Note, that in the following we use the Fourier transformation
	#  \f$ F(f(x)) := (xi) -> \int_{-\inf}^\inf {  exp(-i*2*\pi*x*\xi)  } dx \f$
	# hence, with 2*Pi in the exponent
	#
	# STEP 1: specialize on \f$ f(x) = e^{i x} \f$
	#
	# In Fourier space, the function f(x) is given by
	#
	# \f$
	# 		F(\xi) = \delta( \xi-1.0/(2 \pi) )
	# \f$
	#
	# with \delta the Kronecker delta
	#
	# This simplifies the equation
	#
	# \f$
	#    c_m := h * int_{-1/(2 h)}^{1/(2 h)}  exp(-2 \pi i m h \xi) F(\xi)/Psi_h(\xi)   d \xi
	# \f$
	#
	# to
	#
	# \f$
	#    c_m := h * exp(-i m h) / Psi_h( 1/(2*\pi) )
	# \f$
	#
	#
	# STEP 2: Compute \f$ Psi_h( 1/(2*\pi) ) \f$
	#
	# Furthermore, \psi_h(\xi) is given by
	#
	# \f$
	#    \psi_h(xi) := 1/sqrt(2) * e^{-(2*\pi*h*\xi)^2}
	# \f$
	#
	# and restricting it to xi=1/(2*pi) (see Kronecker delta above), yields
	#
	# \f$
	#    \psi_h(1/2 \pi) := 1/\sqrt{2} * e^{-h^2}
	# \f$
	#
	# URL:
	# http://www.wolframalpha.com/input/?i=FourierTransform[1%2Fsqrt%284*pi%29*exp%28-x^2%2F%284*h^2%29%29%2C+x%2C+\[Omega]%2C+FourierParameters+-%3E+{0%2C+2+Pi}]
	# RESULT: \f$ exp(-4 h^2 \pi^2 \omega^2)/\sqrt(1/h^2)
	#         = h * exp(-(2 h \pi \omega)^2) \f$
	# and with \f$ \xi = 1/(2 \pi) \f$:
	# 			\f$ h * exp(-h^2) \f$
	#
	# STEP 3: combine (1) and (2):
	#
	# This simplifies c_m to
	#
	# \f$
	#    c_m := e^{h^2} * e^{-i*m*h}
	# \f$
	#
	# Let's hope, that these equations are right.
	#
