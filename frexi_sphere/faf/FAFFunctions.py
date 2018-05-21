#! /usr/bin/env python3

#
# Author: Martin Schreiber
# Email: M.Schreiber@exeter.ac.uk
# Date: 2017-06-18
#

import sys
from ..faf import EFloat as ef
from ..faf.FAFCoefficients import *



#
# Setup
#   self.fun(x)
#
class FAFFunctions:

	def __init__(
		self,
		fafcoeffs
	):
#		if not isinstance(fafcoeffs, FAFCoefficients):
#			raise TypeError("Invalid type for fafcoeffs")

		self.efloat = ef.EFloat()

		self.function_name = fafcoeffs.function_name
		self.function_scaling = fafcoeffs.function_scaling

		if self.efloat.floatmode == 'mpfloat':
			import mpmath as mp
			# Set numerical threshold to half of precision
			#self.epsthreshold = self.efloat.pow(10, -mp.mp.dps/2)
			#self.epsthreshold = self.efloat.pow(10, -mp.mp.dps/2)
			self.epsthreshold = 1e-10
		else:
			self.epsthreshold = 1e-10

		# Gaussian basis function, original formulation from Terry's paper 2015
		if self.function_name == 'gaussianorig':
			def fun(x):
				x = x*self.function_scaling
				return self.efloat.exp(-(x*x)/(4.0*self.function_scaling*self.function_scaling))/self.efloat.sqrt(4.0*self.efloat.pi)
			self.fun = fun

			self.is_real_symmetric = True
			self.is_complex_conjugate_symmetric = True	# 0 imag


		# Gaussian basis function, simplified exp(-x*x) formulation
		elif self.function_name == 'gaussianopti':
			def fun(x):
				x = x*self.function_scaling
				return self.efloat.exp(-x*x)
			self.fun = fun

			self.is_real_symmetric = True
			self.is_complex_conjugate_symmetric = True


		# Diffusion
		elif self.function_name == 'expx':
			def fun(x):
				return self.efloat.exp(x)
			self.fun = fun

			self.is_real_symmetric = False
			self.is_complex_conjugate_symmetric = False


		# Exponential integrator: phi0
		elif self.function_name == 'phi0':
			def fun(x):
				return self.efloat.exp(self.efloat.i*x)
			self.fun = fun

			if fafcoeffs.function_complex:
				self.is_real_symmetric = True
				self.is_complex_conjugate_symmetric = True
			else:
				self.is_real_symmetric = True
				self.is_complex_conjugate_symmetric = False


		#
		# Exponential integrator: phi1
		#
		# \phi_1(z) = \frac{e^z - 1}{z}
		#
		# http://www.wolframalpha.com/input/?i=(exp(i*x)-1)%2F(i*x)
		#
		elif self.function_name == 'phi1':
			#
			# Cope with singularity at x=0
			# Note that this is only important for the approximation
			#
			def fun(x):
				K = self.efloat.i * x

				if abs(x) < self.epsthreshold:
					# L'hopital (or written like that)
					# Use LH: 1 times on (exp(x)-1.0)/(x)
					return 1.0 
				else:
					return (self.efloat.exp(K)-1.0)/K
			self.fun = fun

			if fafcoeffs.function_complex:
				self.is_real_symmetric = True
				self.is_complex_conjugate_symmetric = True
			else:
				self.is_real_symmetric = True
				self.is_complex_conjugate_symmetric = False

		#
		# Exponential integrator: phi2
		#
		# Recurrence formula (See Hochbruck, Ostermann, Exponential Integrators, Acta Numerica, Equation (2.11))
		#
		# \phi_{n+1}(z) = \frac{ \phi_n(z) - \phi_n(0) }{z}
		#
		# \phi_2(z) = \frac{e^z - 1 - z}{z^2}
		#
		# http://www.wolframalpha.com/input/?i=(exp(i*x)-1-i*x)%2F(i*x*i*x)
		#
		elif self.function_name == 'phi2':
			#
			# Cope with singularity at x=0
			# Note that this is only important for the approximation
			#
			def fun(x):
				K = self.efloat.i * x
				if abs(x) < self.epsthreshold:
					# Use LH: 2 times on (exp(x)-1.0-x)/(x*x) => 1.0/2.0
					return self.efloat.to(0.5)
				else:
					return (self.efloat.exp(K) - self.efloat.to(1.0) - K)/(K*K)

			self.fun = fun

			if fafcoeffs.function_complex:
				self.is_real_symmetric = True
				self.is_complex_conjugate_symmetric = True
			else:
				self.is_real_symmetric = True
				self.is_complex_conjugate_symmetric = False

		#
		# Exponential integrator: phi3
		#
		elif self.function_name == 'phi3':
			# Cope with singularity at x=0
			def fun(x):
				K = self.efloat.i * x
				if abs(x) < self.epsthreshold:
					# Use LH: 3 times on exp(x)/(x*x*x) - 1.0/(x*x*x) - 1.0/(x*x) - 1.0/(2.0*x)
					return self.efloat.to(1.0/(2.0*3.0))
				else:
					return (self.efloat.exp(K) - self.efloat.to(1.0) - K - K*K)/(K*K*K)

			self.fun = fun

			if fafcoeffs.function_complex:
				self.is_real_symmetric = True
				self.is_complex_conjugate_symmetric = True
			else:
				self.is_real_symmetric = True
				self.is_complex_conjugate_symmetric = False


		elif self.function_name == 'ups1':
			#
			# Setup \upsilon_1 for EDTRK4
			# See document notes_on_time_splitting_methods.lyx
			#
			def fun(x):
				K = self.efloat.i * x
				if abs(x) < self.epsthreshold:
					return self.efloat.to(1.0)/self.efloat.to(2.0*3.0)
				else:
					return (-self.efloat.to(4.0)-K+self.efloat.exp(K)*(self.efloat.to(4.0)-self.efloat.to(3.0)*K+K*K))/(K*K*K)
			self.fun = fun

			if fafcoeffs.function_complex:
				self.is_real_symmetric = True
				self.is_complex_conjugate_symmetric = True
			else:
				self.is_real_symmetric = True
				self.is_complex_conjugate_symmetric = False


		elif self.function_name == 'ups2':
			#
			# Setup \upsilon_2 for EDTRK4
			# See document notes_on_time_splitting_methods.lyx
			#
			def fun(x):
				K = self.efloat.i * x
				if abs(x) < self.epsthreshold:
					return self.efloat.to(1.0)/self.efloat.to(2.0*3.0)
				else:
					return (self.efloat.to(2.0)+1.0*K+self.efloat.exp(K)*(self.efloat.to(-2.0)+K))/(K*K*K)
			self.fun = fun

			if fafcoeffs.function_complex:
				self.is_real_symmetric = True
				self.is_complex_conjugate_symmetric = True
			else:
				self.is_real_symmetric = True
				self.is_complex_conjugate_symmetric = False


		elif self.function_name == 'ups3':
			#
			# Setup \upsilon_3 for EDTRK4
			# See document notes_on_time_splitting_methods.lyx
			#
			def fun(x):
				K = self.efloat.i * x
				if abs(x) < self.epsthreshold:
					return self.efloat.to(1.0)/self.efloat.to(2.0*3.0)
				else:
					return (-self.efloat.to(4.0) - 3.0*K - K*K + self.efloat.exp(K)*(self.efloat.to(4.0)-K))/(K*K*K)
			self.fun = fun

			if fafcoeffs.function_complex:
				self.is_real_symmetric = True
				self.is_complex_conjugate_symmetric = True
			else:
				self.is_real_symmetric = True
				self.is_complex_conjugate_symmetric = False


		else:
			print("Unknown basis function "+str(self.function_name))
			sys.exit(1)


	def fun_re(self, x):
		return self.efloat.re(self.fun(x))


