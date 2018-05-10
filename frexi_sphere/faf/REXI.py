#! /usr/bin/env python3

#
# Author: Martin Schreiber
# Email: M.Schreiber@exeter.ac.uk
# Date: 2017-06-16
#

import sys
import math
import cmath
from FAFCoefficients import *
from REXIGaussianPhi0 import *
import EFloat as ef
import traceback
import numpy



#
# This class offers the coefficients for the
#
# [R ] ational approximation of an
# [Ex] ponential
# [I ] ntegrator
#
class REXI:
	
	def __init__(
		self,
		floatmode = None,
		**kwargs
	):
		self.efloat = ef.EFloat(floatmode)
		self.basis_function_name = "gaussianorig"

		self.setup(floatmode = floatmode, **kwargs)


	def setup(
		self,
		# gaussian approximation of phi function

		N = None,		# Number of basis functions
		h = None,		# spacing of basisfunctions

		test_min = None,	# Test range
		test_max = None,

		basis_function_name = "gaussianorig",	# gaussianorig uses original approximation of approximation of phi0 with an approximation of a gaussian
		function_name = "phi0",	# function to approximate

		# rational functions representing gauss function
		max_error = None,
		max_error_double_precision = None,

		# parameters which are specific to basis_function_name = "gaussianorig"
		ratgauss_N = None,
		ratgauss_basis_function_spacing = 1.0,
		ratgauss_basis_function_rat_shift = None,
		ratgauss_load_original_ratgaussian_poles = True,

		reduce_to_half = True,

		floatmode = "float",

		swap_real = False,
		normalize = False,

		merge = True
	):
		self.N = N
		self.h = h

		self.test_min = test_min
		self.test_max = test_max

		self.basis_function_name = basis_function_name
		self.function_name = function_name

		self.max_error = max_error
		self.max_error_double_precision = max_error_double_precision

		self.ratgauss_N = ratgauss_N
		self.ratgauss_basis_function_spacing = ratgauss_basis_function_spacing
		self.ratgauss_basis_function_rat_shift = ratgauss_basis_function_rat_shift

		self.reduce_to_half = reduce_to_half
		self.ratgauss_load_original_ratgaussian_poles = ratgauss_load_original_ratgaussian_poles

		self.merge = merge

		if self.function_name == "phi0" and self.basis_function_name == "gaussianorig":
			self.setup_rexi_gaussianorig_phi0(floatmode = floatmode)

			# setup range of approximation interval
			self.test_max = self.h*self.N-self.efloat.pi
			self.test_min = -self.test_max

			#	
			# \return \f$ cos(x) + i*sin(x) \f$
			#
			def eval_fun(i_x):
				return self.efloat.re(self.efloat.exp(self.efloat.i*i_x))
			self.eval_fun = eval_fun

			def eval_exp(i_x, i_u0):
				return self.efloat.re(self.efloat.exp(self.efloat.i*i_x)*i_u0)
			self.eval_exp = eval_exp


			#	
			# \return \f$ cos(x) + i*sin(x) \f$
			#
			def eval_fun_cplx(i_x):
				return self.efloat.exp(self.efloat.i*i_x)
			self.eval_fun_cplx = eval_fun_cplx

			def eval_exp_cplx(i_x, i_u0):
				return self.efloat.exp(self.efloat.i*i_x)*i_u0
			self.eval_exp_cplx = eval_exp_cplx

		else:
			raise Exception("TODO")
			if self.basis_function_name != "rationalcplx":
				raise Exception("Not supported")

			self.fafcoeffs = FAFCoefficients(floatmode = floatmode)

			target_fafcoeffs = FAFCoefficients(floatmode = floatmode)
			target_fafcoeffs.max_error = self.max_error
			target_fafcoeffs.max_error_double_precision = self.max_error_double_precision
			target_fafcoeffs.N = self.N
			target_fafcoeffs.basis_function_spacing = self.h

			target_fafcoeffs.test_min = self.test_min
			target_fafcoeffs.test_max = self.test_max

			target_fafcoeffs.function_name = self.function_name
			target_fafcoeffs.basis_function_name = self.basis_function_name

			if not self.fafcoeffs.load_auto(target_fafcoeffs, floatmode = floatmode):
				raise Exception("No valid coefficients found")

			# Valid interval
			self.test_min = self.fafcoeffs.test_min
			self.test_max = self.fafcoeffs.test_max

			# Construct shifted poles
			K = self.fafcoeffs.N//2
			h = self.fafcoeffs.basis_function_spacing
			self.alpha = [self.fafcoeffs.efloat.cplx(self.fafcoeffs.basis_function_rat_shift, -k*h) for k in range(-K, K+1)]
			self.beta_re = [self.fafcoeffs.efloat.cplx(w[0], w[1]) for w in self.fafcoeffs.weights]
			self.beta_im = None	# not working

			#
			# Formulation from FAFBasisFunction:
			#
			# 1/(I*(x-x0*h) + mu)
			#    <=>
			# 1/(I*x - I*(x0*h)  + mu )
			#

			self.rfun_instantiation = FAFFunctions(self.fafcoeffs)
			self.eval_fun = self.rfun_instantiation.fun_re

		if normalize:
			val = 0.0
			for i in range(len(self.alpha)):
				val = val + self.beta[i]/self.alpha[i]

			norm = 1.0/val
			for i in range(len(self.alpha)):
				self.beta[i] *= norm




	#
	# Setup phi0(x) = Re(exp(ix))
	#
	# The coefficients are then made available in
	# self.alpha_reim
	# self.beta_re
	# self.beta_im
	#
	def setup_rexi_gaussianorig_phi0(self, floatmode = None):

		self.ratgauss_coeffs = FAFCoefficients(floatmode = floatmode)

		#
		# Step 1) Load rational approximation of gaussian basis functions
		#
		if self.ratgauss_load_original_ratgaussian_poles:
			self.ratgauss_coeffs.load_orig_ratgaussian_poles()
		else:
			target_fafcoeffs = FAFCoefficients()
			target_fafcoeffs.max_error = self.max_error
			target_fafcoeffs.max_error_double_precision = self.max_error_double_precision
			target_fafcoeffs.N = self.ratgauss_N
			target_fafcoeffs.basis_function_spacing = self.ratgauss_basis_function_spacing
			target_fafcoeffs.basis_function_rat_shift = self.ratgauss_basis_function_rat_shift

			self.ratgauss_coeffs.load_auto(target_fafcoeffs)

		# load parameters from faf coefficients
		self.ratgauss_N = self.ratgauss_coeffs.N
		self.ratgauss_basis_function_spacing = self.ratgauss_coeffs.basis_function_spacing
		self.ratgauss_basis_function_rat_shift = self.ratgauss_coeffs.basis_function_rat_shift


		#
		# Step 2) Load gaussian approximation of phi0
		#
		self.phi0 = REXIGaussianPhi0(
			gaussphi0_N = self.N,
			gaussphi0_basis_function_spacing = self.h
		)

		#
		# Step 3) Merge
		#

		# Use variable naming from original REXI paper
		L = self.ratgauss_coeffs.N//2
		h = self.h
		M = self.N
		N = M+L

		self.alpha_reim = [0 for i in range(2*N+1)]
		self.beta_re = [0 for i in range(2*N+1)]
		self.beta_im = [0 for i in range(2*N+1)]

		for l in range(-L, L+1):
			for m in range(-M, M+1):
				n = l+m
				w = self.ratgauss_coeffs.weights_cplx[l+L]
				# NOTE: We use the conjugate here!
				w = self.efloat.conj(w)

				self.alpha_reim[n+N] = h*(self.ratgauss_coeffs.basis_function_rat_shift + self.efloat.i*n)

				self.beta_re[n+N] += self.efloat.re(self.phi0.b[m+M])*h*w
				self.beta_im[n+N] += self.efloat.im(self.phi0.b[m+M])*h*w


		if self.merge:
			"""
			for i in range(len(self.alpha_reim)):
				print("alpha_reim["+str(i)+"]: "+"{0.real:.8f} {0.imag:.8f}j".format(self.alpha_reim[i])+"\t\t"+"beta_re["+str(i)+"]: "+"{0.real:.8f} {0.imag:.8f}j".format(self.beta_re[i])+"\t\t"+"beta_im["+str(i)+"]: "+"{0.real:.8f} {0.imag:.8f}j".format(self.beta_im[i]))
			print("")
			"""

			#
			# Merge real and imaginary approximation together
			#
			alpha_new = []
			beta_new = []
			M = len(self.alpha_reim)
			for i in range(M):
				beta_new.append(0.5*(self.beta_re[i] + 1.j*self.beta_im[i]))
				alpha_new.append(self.alpha_reim[i])

			for i in range(M):
				beta_new.append(-0.5*(self.beta_re[i] - 1.j*self.beta_im[i]))
				alpha_new.append(-self.alpha_reim[i])

			self.alpha = alpha_new
			self.beta = beta_new

			if self.reduce_to_half:
				# reduce the computational amount to its half,
				# see understanding REXI in the documentation folder

				"""
				for i in range(len(self.alpha)):
					print("alpha["+str(i)+"]: "+"{0.real:.8f} {0.imag:.8f}j".format(self.alpha[i])+"\t\t"+"beta["+str(i)+"]: "+"{0.real:.8f} {0.imag:.8f}j".format(self.beta[i]))
				print("")
				"""

				alpha_new = []
				beta_new = []

				N = len(self.alpha)//2
				alpha_new = self.alpha[0:N//2+1] + self.alpha[N:N+N//2+1]
				beta_new = self.beta[0:N//2+1] + self.beta[N:N+N//2+1]

				for i in range(N//2):
					beta_new[i] *= 2.0
					beta_new[N//2+1+i] *= 2.0
				print(N//2+1)

				self.alpha = alpha_new
				self.beta = beta_new

				"""
				for i in range(len(self.alpha)):
					print("alpha["+str(i)+"]: "+"{0.real:.8f} {0.imag:.8f}j".format(self.alpha[i])+"\t\t"+"beta["+str(i)+"]: "+"{0.real:.8f} {0.imag:.8f}j".format(self.beta[i]))
				print("")
				"""

				"""
				#for i in range(len(self.alpha)):
				#	print("alpha["+str(i)+"]: "+"{0.real:.8f} {0.imag:.8f}j".format(self.alpha[i]))
				#print("")

				for i in range(len(self.beta)):
					print("beta["+str(i)+"]: "+"{0.real:.8f} {0.imag:.8f}j".format(self.beta[i]))
				print("")
				"""


		else:
			self.alpha = self.alpha_reim
			self.beta = self.beta_re

			if self.reduce_to_half:
				N = len(self.alpha)//2
				self.alpha = self.alpha[:N+1]
				self.beta = self.beta[:N+1]

				for i in range(N//2):
					self.beta[i] *= 2.0


	#
	# approx with linear operator, input: complex value, output: complex value
	#
	def approx_fun_cplx_linop(self, i_x, i_u0):
		# Use linear system to avoid complex value (halving doesn't work with this)
		retval = numpy.array([0, 0], dtype=complex)
		u0 = numpy.array([i_u0.real, i_u0.imag], dtype=complex)

		if i_x.imag != 0.0:
			raise Exception("Imaginary value for i_x")

		# Convert to linear solver
		L = numpy.array([[0, -i_x], [i_x, 0]], dtype=complex)

		N = len(self.alpha)
		for n in range(N):
			retval += self.beta[n]*numpy.linalg.solve(L + numpy.eye(2, dtype=complex)*self.alpha[n], u0)

			#a = self.beta[n]*numpy.conj(numpy.linalg.solve(L + numpy.eye(2, dtype=complex)*self.alpha[n], u0))
			#b = self.beta[n]*numpy.linalg.solve(L + numpy.eye(2, dtype=complex)*numpy.conj(self.alpha[n]), u0)
			#print(numpy.abs(a-b))

		return retval[0].real + retval[1].real*1.j


	#
	# approx
	#
	def approx_fun_cplx(self, i_x):

		retval = 0
		# Split computation into real part of \f$ cos(x) \f$ and imaginary part \f$ sin(x) \f$
		for n in range(len(self.alpha)):
			denom = (self.efloat.i*i_x + self.alpha[n])
			retval += (self.beta[n] / denom)

		return retval

	#
	# approx
	#
	def approx_fun_cplx(self, i_x, i_u0):

		retval = 0
		# Split computation into real part of \f$ cos(x) \f$ and imaginary part \f$ sin(x) \f$
		for n in range(len(self.alpha)):
			denom = (self.efloat.i*i_x + self.alpha[n])
			retval += (self.beta[n] / denom * i_u0)

		return retval


	#
	# approx
	#
	def approx_fun(self, i_x):

		retval = 0

		# Split computation into real part of \f$ cos(x) \f$ and imaginary part \f$ sin(x) \f$
		for n in range(len(self.alpha)):
			denom = (self.efloat.i*i_x + self.alpha[n])
			retval += (self.beta[n] / denom).real

		return retval

	#
	# approx
	#
	def approx_fun(self, i_x, i_u0):

		retval = 0

		# Split computation into real part of \f$ cos(x) \f$ and imaginary part \f$ sin(x) \f$
		for n in range(len(self.alpha)):
			denom = (self.efloat.i*i_x + self.alpha[n])
			retval += (self.beta[n] / denom * i_u0).real

		return retval



	def runTests(self):

		h = self.h
		N = self.N

		maxerror = 0
		d = int((self.test_max-self.test_min)*22)

		for u0 in [1.0, (1.0 + 2.0j)/numpy.sqrt(5.0)]:

			print("+"*80)
			print("+"*80)
			print("N: "+str(len(self.alpha)))
			print("u0: "+str(u0))
			print("+"*80)
			print("+"*80)

			if True:
				print("*"*40)
				print(">>> ODE TEST (real only) <<<")

				maxerror = 0
				for x in self.efloat.linspace(self.test_min, self.test_max, d):
					#a = self.eval_fun(x)
					#b = self.approx_fun(x)

					a = self.eval_exp(x, u0)
					b = self.approx_fun(x, u0)

					a = a.real
					b = b.real

					e = abs(a-b)
					maxerror = max(maxerror, e)

				print("max error: "+str(maxerror))

				if maxerror > 1e-9:
					if self.reduce_to_half:
						print("\t\tERROR ignored (reduce_to_half), real-only violated")
					elif u0.imag != 0:
						print("\t\tERROR ignored (Im(u0) != 0), imaginary value")
					else:
						raise Exception("Error threshold exceeded")
				else:
					print("\t\tOK")


			if True:
				print("*"*40)
				print(">>> ODE TEST (complex) <<<")

				maxerror = 0
				for x in self.efloat.linspace(self.test_min, self.test_max, d):
					#a = self.eval_fun_cplx(x)
					#b = self.approx_fun_cplx(x)

					a = self.eval_exp_cplx(x, u0)
					b = self.approx_fun_cplx(x, u0)

					e = abs(a-b)
					maxerror = max(maxerror, e)

				print("max error: "+str(maxerror))

				if maxerror > 1e-9:
					if not self.merge:
						print("\t\tERROR ignored (not merged)")
					elif self.reduce_to_half:
						print("\t\tERROR ignored (reduce_to_half), real-only violated")
					else:
						raise Exception("Error threshold exceeded")
				else:
					print("\t\tOK")


			if True:
				print("*"*40)
				print(">>> PDE TEST (real-valued PDE) <<<")

				maxerror = 0
				for x in self.efloat.linspace(self.test_min, self.test_max, d):
					a = self.eval_exp_cplx(x, u0)
					b = self.approx_fun_cplx_linop(x, u0)

					e = abs(a-b)
					maxerror = max(maxerror, e)

				print("max cplx error: "+str(maxerror))

				if maxerror > 1e-9:
					if not self.merge:
						print("\t\tERROR ignored (not merged)")
					elif self.reduce_to_half and not self.merge:
						print("\t\tERROR ignored (reduce to half)")
					else:
						raise Exception("Error threshold exceeded")
				else:
					print("\t\tOK")

def main():
	# double precision
	floatmode = 'float'

	# multiprecision floating point numbers
	#floatmode = 'mpfloat'


	# load float handling library
	efloat = ef.EFloat(floatmode)

	max_error = efloat.to(1e-10)
	#max_error = None
	max_error_double_precision = max_error
	#max_error_double_precision = None

	h = efloat.to(0.2)
	M = 64
	print("M: "+str(M))
	print("h: "+str(h))
	print("")

	for merge in [False, True]:
		for half in [False, True]:

			print("")
			print("*"*80)
			print("* Original REXI coefficients")

			rexi = REXI(
				N = M,
				h = h,

				ratgauss_load_original_ratgaussian_poles = True,
				function_name = "phi0",

				reduce_to_half = half,
				floatmode = floatmode,
				merge = merge
			)

			if False:
				print("alphas:")
				for i in range(len(rexi.alpha)):
					print(str(i)+": "+str(rexi.alpha[i]))
				print("")

				print("betas:")
				for i in range(len(rexi.beta)):
					print(str(i)+": "+str(rexi.beta[i]))
				print("")

			print("")
			print("*"*80)
			print("* N: "+str(len(rexi.alpha)))
			print("* h: "+str(h))
			print("* half: "+str(half))
			print("* merge: "+str(merge))
			print("*"*80)

			print("Running tests...")
			rexi.runTests()


			""" CONTINUE """
			continue
			""" CONTINUE """


			print("*"*80)
			print("* Original REXI coefficients")

			rexi = REXI(
				ratgauss_load_original_ratgaussian_poles = True,

				N = M,
				h = h,

				function_name = "phi0",

				reduce_to_half = half,
				floatmode = floatmode,
			)

			print("")
			print("*"*80)
			print("Running tests...")
			rexi.runTests()


			print("*"*80)
			print("* REXI coefficients based on original REXI mu")
			print("*"*80)

			rexi = REXI(
				N = M,
				h = h,

				max_error = max_error,
				max_error_double_precision = max_error_double_precision,
				ratgauss_basis_function_spacing = 1.0,
				ratgauss_basis_function_rat_shift = efloat.to('-4.31532151087502402475593044073320925235748291015625'),

				reduce_to_half = half,

				floatmode = floatmode,
			)
			print("*"*80)
			print("Rational approx. of Gaussian data:")
			#rexi.ratgauss_coeffs.print_coefficients()

			print("")
			print("*"*80)
			print("Running tests...")
			rexi.runTests()


			print("*"*80)
			print("* REXI coefficients with optimized mu (if found)")
			print("*"*80)

			rexi = REXI(
				N = M,
				h = h,

				max_error = max_error,
				max_error_double_precision = max_error_double_precision,
				ratgauss_basis_function_spacing = 1.0,
				reduce_to_half = half,

				floatmode = floatmode,
			)
			print("*"*80)
			print("Rational approx. of Gaussian data:")
			#rexi.ratgauss_coeffs.print_coefficients()


			print("")
			print("*"*80)
			print("Running tests...")
			rexi.runTests()

		#for function_name in ["phi0", "phi1", "phi2"]:
		for function_name in []:

			print("*"*80)
			print("* Testing REXI for "+function_name)
			print("*"*80)

			test_max = M*h-efloat.pi
			test_min = -test_max

			rexi = REXI(
				test_min = test_min,
				test_max = test_max,

				max_error = max_error,
				max_error_double_precision = max_error_double_precision,
				ratgauss_basis_function_spacing = 1.0,
				reduce_to_half = False,

				basis_function_name = "rationalcplx",
				function_name = function_name,

				floatmode = floatmode
			)

			print("*"*80)
			print("* Coefficient file")
			print("*"*80)
			print("Filename: "+rexi.fafcoeffs.filename)
			print("")
			print("*"*80)
			print("Running tests...")
			rexi.runTests()
			print("")



if __name__ == "__main__":

	try:
		main()

	except SystemExit as e:
		sys.exit(e)

	except Exception as e:
		print("* ERROR:")
		print(str(e))
		traceback.print_exc()
		sys.exit(1)

