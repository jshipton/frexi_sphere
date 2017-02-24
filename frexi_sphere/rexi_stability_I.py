from rexi_coefficients import *
import cmath
import numpy

h = 0.1
M = 128
alpha, beta_re, beta_im = RexiCoefficients(h, M)

xmax = h*M

xs = numpy.arange(-xmax,xmax,xmax/100)
Gs = 0.*xs
Os = Gs*0.

for m, x in enumerate(xs):
    sum = 0.
    for n in range(len(alpha)):
        denom = (1j*x + alpha[n]);
        sum += (beta_re[n] / denom).real + 1j*(beta_im[n] / denom).real
    Gs[m] = abs(sum)
        
import pylab
pylab.plot(xs,Gs-1.0,'-r',xs,Os,'-b')




