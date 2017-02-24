from rexi_coefficients import *
import cmath
import numpy

h = 0.2
M = 64
alpha, beta_re, beta_im = RexiCoefficients(h, M)

xmax = 2.0
ymax = 150.0

xs = numpy.arange(-xmax,xmax,xmax/100)
ys = numpy.arange(0.,ymax,ymax/100)

xx, yy = numpy.meshgrid(xs,ys)
xx = xx.flatten()
yy = yy.flatten()
zz = (xx + 1j*yy)
G = 0.0*zz

xp = []
yp = []
Gp = []

for m, x in enumerate(zz):
    sum = 0.
    for n in range(len(alpha)):
        denom = (x + alpha[n]);
        sum += (beta_re[n] / denom).real + 1j*(beta_im[n] / denom).real
    if abs(sum) <= 1.0:
        xp.append(xx[m])
        yp.append(yy[m])
        Gp.append(abs(sum))
        
import pylab
pylab.plot(xp,yp,'.')




