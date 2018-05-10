#! /usr/bin/env python3

#
# Author: Martin Schreiber
# Email: M.Schreiber@exeter.ac.uk
# Date: 2017-06-17
#


import sys
import math
import cmath
from FAFCoefficients import *
from FAFFunctions import *
from REXIGaussianPhi0 import *
import EFloat as ef


class REXIGaussianPhi0:

	def __init__(
			self,
                        gaussphi0_N,	# required argument
                        gaussphi0_basis_function_spacing,	# required argument

			floatmode = None
	):
		self.efloat = ef.EFloat(floatmode)

		self.h = self.efloat.to(gaussphi0_basis_function_spacing)
		self.M = int(gaussphi0_N)

		self.b = [self.efloat.exp(self.h*self.h)*self.efloat.exp(-1j*(float(m)*self.h)) for m in range(-self.M, self.M+1)]

		# Generate dummy Gaussian function
		fafcoeffs = FAFCoefficients()
		fafcoeffs.function_name = 'gaussianorig'
		fafcoeffs.function_scaling = self.efloat.to(1.0)

		self.gaussian_fun = FAFFunctions(fafcoeffs)


	def output(self):
		for i in self.b:
			print(i)


	def fun(
		self,
		i_x
	):
		return self.efloat.exp(self.efloat.i*i_x)


	def approx_fun(
		self,
		i_x
	):
		rsum = 0

		i_x = self.efloat.to(i_x)

		# \f$ \sum_{m=-M}^{M}{b_m \psi_h(x+m*h)} \f$
		for m in range(-self.M, self.M+1):
			x = i_x + self.efloat.to(m)*self.h
			x = x/self.h
			rsum += self.b[m+self.M] * self.gaussian_fun.fun_run(x)

		return rsum



	def runTests(self):
		maxerror = 0

		h = self.h
		M = self.M
		print("REXIGaussianPhi0:")
		print("h: "+str(h))
		print("M: "+str(M))
		print("error")
		for x in range(-int(h*M-self.efloat.pi*0.5), int(h*M-self.efloat.pi*0.5)):
			a = self.efloat.re(self.fun_re(x))
			b = self.efloat.re(self.approx_fun(x))

			e = abs(a-b)
			#print("Error at "+str(float(x))+": "+str(float(e)))
			maxerror = max(e, maxerror)

		print("Max error: "+str(maxerror))
		print("Max error (float): "+str(float(maxerror)))





if __name__ == "__main__":

	ef.default_floatmode = 'mpfloat'

	# load float handling library
	efloat = ef.EFloat()


	##################################################
	##################################################
	##################################################

	print("")
	h = 0.1
	M = 64

	print("*"*80)
	print("Approx of Phi0 (Original REXI coefficients)")
	print("*"*80)
	phi0 = REXIGaussianPhi0(
		gaussphi0_N = M,
		gaussphi0_basis_function_spacing = h
	)

	phi0.runTests()



	print("*"*80)
	print("Approx of Phi0 (Original REXI mu, computed coefficients)")
	print("*"*80)
	phi0 = REXIGaussianPhi0(
		gaussphi0_N = M,
		gaussphi0_basis_function_spacing = h,
	)

	phi0.runTests()



	print("*"*80)
	print("Approx of Phi0 (With possibly optimized mu)")
	print("*"*80)
	phi0 = REXIGaussianPhi0(
		gaussphi0_N = M,
		gaussphi0_basis_function_spacing = h,
	)

	phi0.runTests()
