from rexi_coefficients import *
import cmath
import numpy

h = 0.2
M = 64

# load coefficients *WITHOUT* reduction to the half
alpha, beta = RexiCoefficients(h, M, False)

xmax = 2.0
#ymax = 150.0
ymax = 25.0

# resolution of stability plots
#res=100
res=20

xs = numpy.arange(-xmax,xmax,xmax/res)
#ys = numpy.arange(0.,ymax,ymax/res)
ys = numpy.arange(-ymax,ymax,ymax/res)

xx, yy = numpy.meshgrid(xs,ys)
xx = xx.flatten()
yy = yy.flatten()
zz = (xx + 1j*yy)
G = 0.0*zz

xp = []
yp = []
Gp = []
error_re = []
error_im = []

for m, x in enumerate(zz):
    sum = 0.

    # compute REXI SUM
    for n in range(len(alpha)):
        denom = (x + alpha[n]);
        sum += beta[n] / denom

    # Check for stability and add if stable
    if abs(sum) <= 1.0:
        xp.append(xx[m])
        yp.append(yy[m])
        Gp.append(abs(sum))

    # analytical solution
    val = numpy.exp(1j*x)
    error_re.append(abs((val-sum).real))
    error_im.append(abs((val-sum).imag))

# Comment from MaS:
# Didn't work on my system, replaced it directly with matplotlib
#import pylab
#pylab.plot(xp,yp,'.')


if True:
	import matplotlib.pyplot as plt
	plt.scatter(xp,yp,color='blue',s=10,edgecolor='none')
	plt.xlabel("Imag")
	plt.ylabel("Real")
	plt.savefig("output_rexi_stability__stability_region.png")
	plt.clf()
