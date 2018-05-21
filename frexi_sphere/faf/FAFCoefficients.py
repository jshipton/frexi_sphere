#! /usr/bin/env python3

#
# Author: Martin Schreiber
# Email: M.Schreiber@exeter.ac.uk
# Date: 2017-06-16
#

import sys
import os
import copy
from ..faf import EFloat as ef
import inspect



class FAFCoefficients:

	def __init__(self, floatmode = None):
		self.reset(floatmode = floatmode)

	def printHelp(self):
		print("	floatmode")
		print("	max_error")
		print("	max_error_double_precision")
		print("")
		print("	function_name")
		print("	function_scaling")
		print("	function_complex")
		print("")
		print("	N")
		print("	basis_function_name")
		print("	basis_function_spacing")
		print("	basis_function_scaling")
		print("	basis_function_rat_shift")
		print("")
		print("	mpfloat_digits")
		print("	mpfloat_accuracy")
		print("")
		print("...and many more...")

	def setupFloatmode(self):
		self.efloat = ef.EFloat(self.floatmode, self.mpfloat_digits)
		self.floatmode = self.efloat.floatmode


	def reset(self, floatmode = None):
		self.mpfloat_digits = None

		self.floatmode = floatmode

		# Reset floatmode in case that this was requested via a program parameter
		self.setupFloatmode()

		# Maximum error for the given nodes
		self.max_error = None

		# Maximum error for the given nodes if using only double precision
		self.max_error_double_precision = None

		# Number of support points / Number of basis functions
		self.N = None

		# Basis function name
		self.basis_function_name = None

		# Basis function distance
		self.basis_function_spacing = None

		# x scaling of basis function
		self.basis_function_scaling = None

		# Shift for rational approximated basis function
		self.basis_function_rat_shift = None


		self.function_name = None
		self.function_scaling = None
		self.function_complex = None

		self.weights = []
		self.weights_str = []
		self.weights_cplx = []

		# support points
		self.x0s = []


		# Optimizer method
		# ['BFGS', 'CG', 'POWELL', 'TNC', 'SLSQP', 'nelder-mead']:
		self.optimizer_method = None
		self.optimizer_vars = []
		self.optimizer_epsilon = None
		self.optimizer_minimizeeps = None
		self.optimizer_opti_double_precision = None

		# Generation
		self.gen_mainsetup = None

		# Oversampling stuff
		self.gen_os_samples_per_unit = None
		self.gen_os_samples_per_unit_central = None
		self.gen_os_min = None
		self.gen_os_max = None

		self.gen_solver_method = None

		# oversampling points
		self.x0os = []

		# Testing information
		self.test_samples_per_unit = None
		self.test_min = None
		self.test_max = None
		# testing points
		self.xtest = []

		# Data filename (if loaded from file)
		self.filename = None




	def copy(self):
		# deepcopy requires special treatment of efloats

		tmp = self.efloat
		self.efloat = None

		c = copy.deepcopy(self)

		self.efloat = tmp
		c.efloat = tmp
		return c

	def get_faf_data_directory(self, fafcoeffs):

		basis_function_name = fafcoeffs.basis_function_name
		function_name = fafcoeffs.function_name

		if basis_function_name == None:
			basis_function_name = "rationalcplx"

		if function_name == None:
			function_name = "gaussianorig"

		prefix = os.path.dirname(inspect.getfile(FAFCoefficients))
		return os.path.join(prefix, "faf_data_"+basis_function_name+"_"+function_name)





	#
	# Automatically load FAF coefficients with given constraints
	#
	def load_auto(
			self,
			target_fafcoeffs,
			faf_coeff_directory = None,
			enforce_double_precision_mode = False,
			floatmode = None
	):
		if floatmode == "float":
			self.floatmode = "float"
			self.setupFloatmode()
			return self.load_auto_real(target_fafcoeffs, faf_coeff_directory, True)
		else:
			self.floatmode = floatmode
			self.setupFloatmode()
			return self.load_auto_real(target_fafcoeffs, faf_coeff_directory, True)
		

	#
	# Automatically load FAF coefficients with given constraints
	#
	def load_auto_real(
			self,
			target_fafcoeffs,
			faf_coeff_directory = None,
			enforce_double_precision_mode = False
	):
		self.reset()

		if enforce_double_precision_mode:
			self.floatmode = 'float'
			self.setupFloatmode()

		if faf_coeff_directory == None:
			faf_coeff_directory = self.get_faf_data_directory(target_fafcoeffs)

		import glob
		fafdir = faf_coeff_directory+"/*csv"
		files = glob.glob(fafdir)
		if len(files) == 0:
			raise Exception("No files with coefficients found in directory "+fafdir)

		best = None

		for f in files:
			self.load_coefficients_file(f, enforce_double_precision_mode)

			eps = 1e-10

			#
			# Process hard constraints first
			#
			if target_fafcoeffs.N != None:
				if self.N != target_fafcoeffs.N:
					continue

			if target_fafcoeffs.max_error != None:
				if self.max_error > target_fafcoeffs.max_error:
					continue

			if target_fafcoeffs.max_error_double_precision != None:
				if self.max_error_double_precision > target_fafcoeffs.max_error_double_precision:
					continue

			if target_fafcoeffs.test_min != None:
				if self.test_min > target_fafcoeffs.test_min:
					continue

			if target_fafcoeffs.test_max != None:
				if self.test_max < target_fafcoeffs.test_max:
					continue

			if target_fafcoeffs.function_name != None:
				if self.function_name != target_fafcoeffs.function_name:
					continue

			if target_fafcoeffs.function_scaling != None:
				if abs(self.function_scaling-target_fafcoeffs.function_scaling) > eps:
					continue

			if target_fafcoeffs.function_complex != None:
				if self.function_complex != target_fafcoeffs.function_complex:
					continue

			if target_fafcoeffs.basis_function_name != None:
				if self.basis_function_name != target_fafcoeffs.basis_function_name:
					continue

			if target_fafcoeffs.basis_function_spacing != None:
				if abs(self.basis_function_spacing-target_fafcoeffs.basis_function_spacing) > eps:
					continue

			if target_fafcoeffs.basis_function_scaling != None:
				if abs(self.basis_function_scaling-target_fafcoeffs.basis_function_scaling) > eps:
					continue

			if target_fafcoeffs.basis_function_rat_shift != None:
				if abs(self.basis_function_rat_shift-target_fafcoeffs.basis_function_rat_shift) > eps:
					continue


			#
			# Soft constraints next
			#

			if target_fafcoeffs.max_error == None and target_fafcoeffs.max_error_double_precision == None and target_fafcoeffs.N == None:
				if best == None:
					best = self.copy()
					continue

				# If nothing is specified, search for highest accuracy!
				if best.max_error > self.max_error:
					best = self.copy()
					continue

				continue

			if target_fafcoeffs.N == None:
				if best == None:
					best = self.copy()
					continue

				# Minimize number of poles
				if best.N < self.N:
					continue


				# If poles are the same, try to minimize error
				if best.N == self.N:
					if best.max_error < self.max_error:
						if best.max_error_double_precision != None and self.max_error_double_precision != None:
							if best.max_error_double_precision < self.max_error_double_precision:
								continue
						else:
							continue

				best = self.copy()
				continue


			if True:	# max_error == None
				# Hard constraint of self.N == N was already triggered before
				if best == None:
					best = self.copy()
					continue

				# Minimize error
				if best.max_error > self.max_error:
					continue

				# Minimize error
				if best.max_error_double_precision != None and self.max_error_double_precision != None:
					if best.max_error_double_precision > self.max_error_double_precision:
						continue

				best = self.copy()
				continue
			continue

		if best == None:
			raise Exception("No suitable coefficients found!")

		#print("Using coefficients from file "+best.filename)
		self.load_coefficients_file(best.filename, enforce_double_precision_mode)
		return True




	def load_coefficients_file(self, filename, avoid_efloat_update = False):

		self.reset()

		def toString(value):
			if value == '-' or value == 'None':
				return None

			return value

		def toStringArray(value):
			if value == '-' or value == 'None':
				return []

			return value.split(',')

		def toFloat(value):
			if value == '-' or value == 'None':
				return None

			return self.efloat.to(value)

		def toInt(value):
			if value == '-' or value == 'None':
				return None

			return int(value)

		def toBool(value):
			if value == '-' or value == 'None':
				return None

			if value == '1':
				return True
			elif value == '0':
				return False

			raise Exception("Unknown boolean value "+value)



		self.filename = filename

		content = open(filename).readlines()

		efloat_updated = False

		max_weight_len = 0
		for c in content:
			c = c.replace('\n', '')

			stra = '# mpfloat_digits '
			if c[0:len(stra)] == stra:
				if c[len(stra):] == 'None':
					self.mpfloat_digits = None
				else:
					self.mpfloat_digits = toInt(c[len(stra):])

				if not avoid_efloat_update:
					efloat_updated = True

				continue

			stra = '# floatmode '
			if c[0:len(stra)] == stra:
				self.floatmode = c[len(stra):]

				if not avoid_efloat_update:
					efloat_updated = True

			stra = '# max_error '
			if c[0:len(stra)] == stra:
				self.max_error = toFloat(c[len(stra):])
				continue

			stra = '# max_error_double_precision '
			if c[0:len(stra)] == stra:
				self.max_error_double_precision = toFloat(c[len(stra):])
				continue

			stra = '# function_name '
			if c[0:len(stra)] == stra:
				self.function_name = c[len(stra):]
				continue

			stra = '# function_scaling '
			if c[0:len(stra)] == stra:
				self.function_scaling = toFloat(c[len(stra):])
				continue

			stra = '# function_complex '
			if c[0:len(stra)] == stra:
				self.function_complex = toBool(c[len(stra):])
				continue

			stra = '# basis_function_name '
			if c[0:len(stra)] == stra:
				self.basis_function_name = c[len(stra):]
				continue

			stra = '# basis_function_spacing '
			if c[0:len(stra)] == stra:
				self.basis_function_spacing = toFloat(c[len(stra):])
				continue

			stra = '# basis_function_scaling '
			if c[0:len(stra)] == stra:
				self.basis_function_scaling = toFloat(c[len(stra):])
				continue

			stra = '# basis_function_rat_shift '
			if c[0:len(stra)] == stra:
				self.basis_function_rat_shift = toFloat(c[len(stra):])
				continue

			stra = '# gen_os_samples_per_unit '
			if c[0:len(stra)] == stra:
				self.gen_os_samples_per_unit = toInt(c[len(stra):])
				continue

			stra = '# gen_mainsetup '
			if c[0:len(stra)] == stra:
				self.gen_mainsetup = c[len(stra):]
				continue

			stra = '# gen_os_min '
			if c[0:len(stra)] == stra:
				self.gen_os_min = toFloat(c[len(stra):])
				continue

			stra = '# gen_os_max '
			if c[0:len(stra)] == stra:
				self.gen_os_max = toFloat(c[len(stra):])
				continue

			stra = '# gen_solver_method '
			if c[0:len(stra)] == stra:
				self.gen_solver_method = c[len(stra):]
				continue

			stra = '# optimizer_method '
			if c[0:len(stra)] == stra:
				self.optimizer_method = c[len(stra):]
				continue

			stra = '# optimizer_vars '
			if c[0:len(stra)] == stra:
				self.optimizer_vars = c[len(stra):].split(',')
				continue

			stra = '# optimizer_opti_double_precision '
			if c[0:len(stra)] == stra:
				self.optimizer_opti_double_precision = toBool(c[len(stra):])
				continue

			stra = '# test_samples_per_unit '
			if c[0:len(stra)] == stra:
				self.test_samples_per_unit = int(c[len(stra):])
				continue

			stra = '# test_min '
			if c[0:len(stra)] == stra:
				self.test_min = toFloat(c[len(stra):])
				continue

			stra = '# test_max '
			if c[0:len(stra)] == stra:
				self.test_max = toFloat(c[len(stra):])
				continue

			if c[0] == '#':
				# ignore the rest
				continue

			nums = c.split('\t')
			self.weights_str.append(nums)

			if len(nums) != 2:
				print("ERROR in file "+filename+" with line:")
				print(c)
				sys.exit(1)

			max_weight_len = max(max_weight_len, len(nums[0]), len(nums[1]))

		self.N = len(self.weights_str)

#		if floatmode == 'mpfloat':
#			if mp.mp.dps < max_weight_len:
#				print("WARNING: mp.dps accuracy is smaller than input values")

		for w in self.weights_str:
			self.weights.append([self.efloat.to(w[0]), self.efloat.to(w[1])])
			self.weights_cplx.append(self.efloat.to(w[0]) + self.efloat.i*self.efloat.to(w[1]))

		self.x0s = self.compute_basis_support_points()

		if efloat_updated:
			self.efloat = ef.EFloat(self.floatmode, self.mpfloat_digits)
			# reload coefficients
			self.load_coefficients_file(filename, True)

		return True


	def load_orig_ratgaussian_poles(self):
		self.reset()

		self.basis_function_rat_shift = self.efloat.to('-4.31532151087502402475593044073320925235748291015625')
		self.basis_function_spacing = self.efloat.to(1.0)
		self.basis_function_scaling = self.efloat.to(1.0)
		self.function_name = 'gaussianorig'
		self.floatmode = 'float'
		self.basis_function_name = 'rationalcplx'
		self.function_scaling = self.efloat.to(1.0)
		self.function_complex = True
		self.test_samples_per_unit = 111
		self.test_min = self.efloat.to(-100.0)
		self.test_max = self.efloat.to(100.0)
		self.N = 23

		weights_tmp = [
			['-1.0845749544592896e-7',	'2.77075431662228e-8'],
			['1.858753344202957e-8',	'-9.105375434750162e-7'],
			['3.6743713227243024e-6',	'7.073284346322969e-7'],
			['-2.7990058083347696e-6',	'0.0000112564827639346'],
			['0.000014918577548849352',	'-0.0000316278486761932'],
			['-0.0010751767283285608',	'-0.00047282220513073084'],
			['0.003816465653840016',	'0.017839810396560574'],
			['0.12124105653274578',		'-0.12327042473830248'],
			['-0.9774980792734348',		'-0.1877130220537587'],
			['1.3432866123333178',		'3.2034715228495942'],
			['4.072408546157305',		'-6.123755543580666'],
			['-9.442699917778205',		'0.'],
			['4.072408620272648',		'6.123755841848161'],
			['1.3432860877712938',		'-3.2034712658530275'],
			['-0.9774985292598916',		'0.18771238018072134'],
			['0.1212417070363373',		'0.12326987628935386'],
			['0.0038169724770333343',	'-0.017839242222443888'],
			['-0.0010756025812659208',	'0.0004731874917343858'],
			['0.000014713754789095218',	'0.000031358475831136815 '],
			['-2.659323898804944e-6',	'-0.000011341571201752273'],
			['3.6970377676364553e-6',	'-6.517457477594937e-7'],
			['3.883933649142257e-9',	'9.128496023863376e-7'],
			['-1.0816457995911385e-7',	'-2.954309729192276e-8'],
		]
		self.weights_cplx= []
		for w in weights_tmp:
			self.weights_cplx.append(self.efloat.cplx(self.efloat.to(w[0]), self.efloat.to(w[1])))

		# NOTE! We use the complex conjugates here which reflects the formulation
		# of the support points of the rational functions in this implementation
		self.weights_cplx = list(map(self.efloat.conj, self.weights_cplx))

		self.x0s = self.compute_basis_support_points()


	def toShortFloatStr(self, value, digits):
		if value == None:
			return 'None'

		return ("%0.0"+str(digits)+"e") % value

	def toStr(self, value):
		if value == None:
			return 'None'

		return str(value)


	def toBool(self, value):
		if value == None:
			return '-'

		if value == True:
			return '1'
		elif value == False:
			return '0'

		raise("Unknown boolean value")


	def argsString(self, option_name, argv):

		if argv[self.args_i] != option_name:
			return False

		self.args_i = self.args_i+1
		if len(argv) <= self.args_i:
			print("Argument needed for option "+option_name)
			sys.exit(1)

		self.args_value = argv[self.args_i]
		if self.args_value == '-':
			self.args_value = None

		self.args_i = self.args_i+1

		return True





	def argsStringArray(self, option_name, argv):

		if argv[self.args_i] != option_name:
			return False

		self.args_i = self.args_i+1
		if len(argv) <= self.args_i:
			print("Argument needed for option "+option_name)
			sys.exit(1)

		self.args_value = argv[self.args_i]
		if self.args_value == '-':
			self.args_value = []
		else:
			self.args_value = self.args_value.split(',')

		self.args_i = self.args_i+1

		return True



	def argsFloat(self, option_name, argv):

		if argv[self.args_i] != option_name:
			return False

		self.args_i = self.args_i+1
		if len(argv) <= self.args_i:
			print("Argument needed for option "+option_name)
			sys.exit(1)

		self.args_value = self.efloat.to(argv[self.args_i])
		self.args_i = self.args_i+1

		return True


	def argsInt(self, option_name, argv):

		if argv[self.args_i] != option_name:
			return False

		self.args_i = self.args_i+1
		if len(argv) <= self.args_i:
			print("Argument needed for option "+option_name)
			sys.exit(1)

		self.args_value = int(argv[self.args_i])
		self.args_i = self.args_i+1

		return True



	def argsBool(self, option_name, argv):

		if argv[self.args_i] != option_name:
			return False

		self.args_i = self.args_i+1
		if len(argv) <= self.args_i:
			print("Argument needed for option "+option_name)
			sys.exit(1)

		if len(argv[self.args_i]) > 1:
			print("Argument for option "+option_name+" is wrong (0 or 1)")
			sys.exit(1)

		if argv[self.args_i] == '0':
			self.args_value = False
		elif argv[self.args_i] == '1':
			self.args_value = True
		else:
			raise Exception("Only 0/1 as boolean avlue supported, given "+str(argv[self.args_i]))

		self.args_i = self.args_i+1

		return True


	def process_args(self, argv, avoid_efloat_update = False):
		def toString(value):
			if value == '-':
				return None

			return value

		def toStringArray(value):
			if value == '-':
				return []

			return value.split(',')

		def toFloat(value):
			if value == '-':
				return None

			return self.efloat.to(value)

		def toInt(value):
			if value == '-':
				return None

			return int(value)


		self.args_i = 1

		efloat_updated = False

		while self.args_i < len(argv):
			self.args_value = None

			if self.argsInt("mpfloat_digits", argv):
				self.mpfloat_digits = self.args_value

				if not avoid_efloat_update:
					efloat_updated = True
				continue

			if self.argsInt("N", argv):
				self.N = toInt(self.args_value)
				continue

			if self.argsFloat("max_error", argv):
				self.max_error = self.args_value
				continue

			if self.argsFloat("max_error_double_precision", argv):
				self.max_error_double_precision = self.args_value
				continue

			if self.argsString("function_name", argv):
				self.function_name = self.args_value
				continue

			if self.argsFloat("function_scaling", argv):
				self.function_scaling = self.args_value
				continue

			if self.argsBool("function_complex", argv):
				self.function_complex = self.args_value
				continue

			if self.argsString("basis_function_name", argv):
				self.basis_function_name = self.args_value
				continue

			if self.argsFloat("basis_function_spacing", argv):
				self.basis_function_spacing = self.args_value
				continue

			if self.argsFloat("basis_function_scaling", argv):
				self.basis_function_scaling = self.args_value
				continue

			if self.argsFloat("basis_function_rat_shift", argv):
				self.basis_function_rat_shift = self.args_value
				continue

			if self.argsString("optimizer_method", argv):
				self.optimizer_method = self.args_value
				continue

			if self.argsStringArray("optimizer_vars", argv):
				self.optimizer_vars = self.args_value
				continue

			if self.argsBool("optimizer_opti_double_precision", argv):
				self.optimizer_opti_double_precision = self.args_value
				continue

			if self.argsString("floatmode", argv):
				self.floatmode = self.args_value

				if not avoid_efloat_update:
					efloat_updated = True
				continue

			if self.argsInt("gen_os_samples_per_unit", argv):
				self.gen_os_samples_per_unit = self.args_value
				continue

			if self.argsString("gen_mainsetup", argv):
				self.gen_mainsetup = self.args_value
				continue

			if self.argsString("gen_solver_method", argv):
				self.gen_solver_method = self.args_value
				continue


			#
			# testing
			#
			if self.argsInt("test_samples_per_unit", argv):
				self.test_samples_per_unit = self.args_value
				continue

			if self.argsFloat("test_min", argv):
				self.test_min = self.args_value
				continue

			if self.argsFloat("test_max", argv):
				self.test_max = self.args_value
				continue

			print("Invalid option "+argv[self.args_i])
			sys.exit(1)

		if efloat_updated:
			self.efloat = ef.EFloat(self.floatmode, self.mpfloat_digits)
			self.process_args(argv, True)
			

	#
	# return a string which can be used as an argument string to a program
	#
	def get_program_args(self):

		def fromString(value):
			if value == None:
				return '-'
			return value

		def fromInt(value):
			if value == None:
				return '-'
			return str(value)

		def fromBool(value):
			if value == None:
				return '-'

			if value:
				return '1'
			else:
				return '0'

		def fromFloat(value):
			if value == None:
				return '-'
			return str(value)

		def fromString(value):
			if value == '':
				return '-'
			return value

		def fromStringArray(value):
			if len(value) == 0:
				return '-'
			return ','.join(value)

		output = []


		if self.floatmode != None:
			output.append("floatmode "+fromString(self.floatmode))

		if self.mpfloat_digits != None:
			output.append("mpfloat_digits "+fromInt(self.mpfloat_digits))


		if self.function_name != None:
			output.append("function_name "+fromString(self.function_name))

		if self.function_scaling != None:
			output.append("function_scaling "+fromFloat(self.function_scaling))

		if self.function_complex != None:
			output.append("function_complex "+fromBool(self.function_complex))

		if self.basis_function_name != None:
			output.append("basis_function_name "+fromString(self.basis_function_name))

		if self.basis_function_spacing != None:
			output.append("basis_function_spacing "+fromFloat(self.basis_function_spacing))

		if self.basis_function_scaling != None:
			output.append("basis_function_scaling "+fromFloat(self.basis_function_scaling))

		if self.basis_function_rat_shift != None:
			output.append("basis_function_rat_shift "+fromFloat(self.basis_function_rat_shift))


		if self.optimizer_method != None:
			output.append("optimizer_method "+fromString(self.optimizer_method))

		if self.optimizer_vars != None:
			output.append("optimizer_vars "+fromStringArray(self.optimizer_vars))

		if self.optimizer_opti_double_precision != None:
			output.append("optimizer_opti_double_precision "+fromBool(self.optimizer_opti_double_precision))


		if self.gen_mainsetup != None:
			output.append("gen_mainsetup "+fromString(self.gen_mainsetup))

		if self.gen_solver_method != None:
			output.append("gen_solver_method "+fromString(self.gen_solver_method))

		if self.gen_os_samples_per_unit != None:
			output.append("gen_os_samples_per_unit "+fromInt(self.gen_os_samples_per_unit))

		if self.gen_os_samples_per_unit_central != None:
			output.append("gen_os_samples_per_unit_central "+fromFloat(self.gen_os_samples_per_unit_central))

		if self.gen_os_min != None:
			output.append("gen_os_min "+fromFloat(self.gen_os_min))

		if self.gen_os_max != None:
			output.append("gen_os_max "+fromFloat(self.gen_os_max))


		if self.N != None:
			output.append("N "+fromInt(self.N))

		return " ".join(output)


	def approx_fun_re(self, x):
		efloat.re(self.approx_fun_cplx(x))
		return 




	def print_coefficients(self):

		print("N: "+self.toStr(self.N))
		print("max_error: "+self.toStr(self.max_error))
		print("max_error (float): "+(self.toStr(float(self.max_error)) if self.max_error != None else "None"))
		print("Errors using double precision arithmetics:")
		print("max_error_double_precision: "+self.toStr(self.max_error_double_precision))
		print("max_error_double_precision (float): "+(self.toStr(float(self.max_error_double_precision)) if self.max_error_double_precision != None else "None"))
		print("")
		print("function_name: "+self.toStr(self.function_name))
		print("function_scaling: "+self.toStr(self.function_scaling))
		print("function_complex: "+self.toBool(self.function_complex))
		print("")
		print("basis_function_name: "+self.toStr(self.basis_function_name))
		print("basis_function_spacing: "+self.toStr(self.basis_function_spacing))
		print("basis_function_scaling: "+self.toStr(self.basis_function_scaling))
		print("basis_function_rat_shift: "+self.toStr(self.basis_function_rat_shift))

		if len(self.x0s) == 0:
			print("x0 (nodal points for basis): (#): "+str(len(self.x0s)))
		else:
			print("x0 (nodal points for basis): (# / min / max): "+str(len(self.x0s))+" / "+str(min(self.x0s))+" / "+str(max(self.x0s)))

		print("")
		print("test_samples_per_unit: "+self.toStr(self.test_samples_per_unit))
		print("test_min: "+self.toStr(self.test_min))
		print("test_max: "+self.toStr(self.test_max))
		if len(self.xtest) == 0:
			print("xtest (testing points) (#): "+str(len(self.xtest)))
		else:
			print("xtest (testing points) (# / min / max): "+str(len(self.xtest))+" / "+str(min(self.xtest))+" / "+str(max(self.xtest)))
		print("")
		print("gen_os_min: "+str(self.gen_os_min))
		print("gen_os_max: "+str(self.gen_os_max))
		print("gen_os_samples_per_unit: "+str(self.gen_os_samples_per_unit))
		print("gen_os_samples_per_unit_central: "+str(self.gen_os_samples_per_unit_central))
		if len(self.x0os) == 0:
			print("x0os (oversampling points) (#): "+str(len(self.x0os)))
		else:
			print("x0os (oversampling points) (# / min / max): "+str(len(self.x0os))+" / "+str(min(self.x0os))+" / "+str(max(self.x0os)))
		print("")
		print("floatmode: "+self.toStr(self.floatmode))
		print("mpfloat_digits: "+self.toStr(self.mpfloat_digits))
		print("")
		print("optimizer_method: "+str(self.optimizer_method))
		print("optimizer_vars: "+str(",".join(self.optimizer_vars)))
		print("optimizer_opti_double_precision: "+str(self.optimizer_opti_double_precision))
		print("")

		if self.x0s != []:
			print("nodal_points: (# / min / max): "+self.toStr(len(self.x0s))+" / "+self.toStr(min(self.x0s))+" / "+self.toStr(max(self.x0s)))

		#for w in self.weights:
		#	print(w)


	def write_file(self, filename):

		f = open(filename, 'w')
		f.write("# max_error "+self.toStr(self.max_error)+"\n")
		f.write("# max_error_in_float "+self.toStr(float(self.max_error))+"\n")
		f.write("# max_error_double_precision "+self.toStr(self.max_error_double_precision)+"\n")
		f.write("# max_error_double_precision_in_float "+(self.toStr(float(self.max_error_double_precision)) if self.max_error_double_precision != None else "None")+"\n")
		f.write("# N "+self.toStr(self.N)+"\n")

		f.write("# function_name "+self.toStr(self.function_name)+"\n")
		f.write("# function_scaling "+self.toStr(self.function_scaling)+"\n")
		f.write("# function_complex "+self.toBool(self.function_complex)+"\n")

		f.write("# basis_function_name "+self.toStr(self.basis_function_name)+"\n")
		f.write("# basis_function_spacing "+self.toStr(self.basis_function_spacing)+"\n")
		f.write("# basis_function_scaling "+self.toStr(self.basis_function_scaling)+"\n")
		f.write("# basis_function_rat_shift "+self.toStr(self.basis_function_rat_shift)+"\n")


		f.write("# optimizer_method "+self.toStr(self.optimizer_method)+"\n")
		f.write("# optimizer_vars "+self.toStr(','.join(self.optimizer_vars))+"\n")
		f.write("# optimizer_epsilon "+self.toStr(self.optimizer_epsilon)+"\n")
		f.write("# optimizer_minimizeeps "+self.toStr(self.optimizer_minimizeeps)+"\n")
		f.write("# optimizer_opti_double_precision "+self.toBool(self.optimizer_opti_double_precision)+"\n")


		f.write("# gen_mainsetup "+self.toStr(self.gen_mainsetup)+"\n")
		f.write("# gen_os_samples_per_unit "+self.toStr(self.gen_os_samples_per_unit)+"\n")
		f.write("# gen_os_samples_per_unit_central "+self.toStr(self.gen_os_samples_per_unit_central)+"\n")
		f.write("# gen_os_min "+self.toStr(self.gen_os_min)+"\n")
		f.write("# gen_os_max "+self.toStr(self.gen_os_max)+"\n")
		f.write("# gen_solver_method "+self.toStr(self.gen_solver_method)+"\n")

		f.write("# test_samples_per_unit "+self.toStr(self.test_samples_per_unit)+"\n")
		f.write("# test_min "+self.toStr(self.test_min)+"\n")
		f.write("# test_max "+self.toStr(self.test_max)+"\n")


		f.write("# floatmode "+self.toStr(self.floatmode)+"\n")
		f.write("# mpfloat_digits "+self.toStr(self.mpfloat_digits)+"\n")

		for w in self.weights_cplx:
			f.write(self.efloat.floatToStr(self.efloat.re(w))+"\t"+self.efloat.floatToStr(self.efloat.im(w))+"\n")



	#
	# return the support points of the basis functions
	#
	def compute_basis_support_points(self):
		if self.N & 1 == 0:
			return [(self.efloat.to(-int(self.N/2)) + (self.efloat.to(i)+self.efloat.to(0.5)))*self.basis_function_spacing for i in range(self.N)]
		else:
			return [(self.efloat.to(-int(self.N/2)) + self.efloat.to(i))*self.basis_function_spacing for i in range(self.N)]



	def get_basis_function_support_points(self):
		Nn = self.N
		hp = self.basis_function_spacing

		if (self.N & 1) == 0:
			# even
			return [(self.efloat.to(-int(self.N//2)) + (self.efloat.to(i)+self.efloat.to(0.5)))*hp for i in range(self.N)]
		else:
			return [(self.efloat.to(-int(self.N//2)) + self.efloat.to(i))*hp for i in range(self.N)]


	def get_oversampling_points(self):

		if self.gen_os_samples_per_unit_central == None:
			# Ignore more samples in the central
			return self.efloat.linspace(self.gen_os_min, self.gen_os_max, self.efloat.ceil((self.gen_os_max-self.gen_os_min)*float(self.gen_os_samples_per_unit)))

		else:
			# e^{-10*10} \approx 1e-44
			d = self.efloat.to(10.0)

			a = self.efloat.linspace(self.gen_os_min, -d, self.efloat.ceil((-self.gen_os_min-d)*float(self.gen_os_samples_per_unit)))
			c = self.efloat.linspace(d, self.gen_os_max, self.efloat.ceil((self.gen_os_max-d)*float(self.gen_os_samples_per_unit)))

			# use higher oversampling in gaussian area
			b = self.efloat.linspace(-d, d, self.efloat.ceil(self.efloat.to(d*2.0)*float(self.gen_os_samples_per_unit_central)))[1:-1]

			return a+b+c



	def get_test_points(self):
		#
		# Fallback test intervals
		#

		test_min = self.test_min
		test_max = self.test_max
		test_samples_per_unit = self.test_samples_per_unit

		if test_max <= test_min:
			raise Exception("test_max <= test_min: "+str(test_max)+" <= "+str(test_min))
		return self.efloat.linspace(
				self.efloat.to(test_min),
				self.efloat.to(test_max),
				int((test_max-test_min)*self.efloat.to(test_samples_per_unit))
			)



	def getID(self):
		def fromStringArray(value):
			if len(value) == 0:
				return 'None'
			return ','.join(value)

		def fromString(value):
			if value == None:
				return 'None'
			return str(value)

		def fromBool(value):
			if value == None:
				return 'None'
		
			if value:
				return "1"
			else:
				return "0"

		output = ""
		output += "N"+self.toStr(self.N).zfill(5)
		#output += "_FUNCTION"
		output += "_FUN"
		output += "_NAM"+fromString(self.function_name)
		output += "_SCA"+self.toShortFloatStr(self.function_scaling, 5)
		output += "_CPX"+fromBool(self.function_complex)

		#output += "_BASISFUNCTION"
		output += "_BAFUN"
		output += "_NAM"+fromString(self.basis_function_name)
		output += "_SPA"+self.toShortFloatStr(self.basis_function_spacing, 2)
		output += "_SCA"+self.toShortFloatStr(self.basis_function_scaling, 5)
		output += "_SFT"+self.toShortFloatStr(self.basis_function_rat_shift, 5)

		if self.floatmode == "mpfloat":
			if self.mpfloat_digits != None:
				output += "_MPDPS"+str(self.mpfloat_digits)

		#output += "_OPTIMIZER"
		output += "_OPTI"
		output += "_METH"+fromString(self.optimizer_method)
		t = fromStringArray(self.optimizer_vars).replace('_','')
		t = t.replace('basisfunctionratshift', 'basratsh')
		t = t.replace('basisfunctionscaling', 'basscal')
		output += "_VARS"+t
#		output += "_EPSILON"+self.toShortFloatStr(self.optimizer_epsilon, 5)
#		output += "_MINIMIZEEPS"+self.toShortFloatStr(self.optimizer_minimizeeps, 5)

#		output += "_MAINSETUP"+fromString(self.gen_mainsetup)
		output += "_SOLVM"+fromString(self.gen_solver_method)

		#output = output.replace('-','').replace('.','_')
		output = output.replace('.','_')
		return output



if __name__ == "__main__":
	from FAFTestCoefficients import *

	ef.default_floatmode = 'float'
	#ef.default_floatmode = 'mpfloat'


	efloat = ef.EFloat()

	

	################################################################
	################################################################
	# test with original FAF coefficients
	################################################################
	################################################################

	print("*"*80)
	print("Original FAF coefficients")
	print("*"*80)
	faf = FAFCoefficients()
	faf.load_orig_ratgaussian_poles()
	print("File: "+str(faf.filename))
	faf.process_args(sys.argv)
	faf.print_coefficients()

	test = FAFTestCoefficients()
	test.load_fafcoeffs(faf)
	test.runTests()


	################################################################
	################################################################
	# test with new coefficients for Gaussian basis function
	################################################################
	################################################################

	print("*"*80)
	print("New FAF coefficients")
	print("*"*80)

	faf = FAFCoefficients()
	faf.process_args(sys.argv)

	faf_test = FAFCoefficients()

	faf_test.N = 23
	faf_test.basis_function_spacing = efloat.to(1.0)
	faf_test.function_name = 'gaussianorig'
	faf_test.basis_function_rat_shift = efloat.to("-4.315321510875024024755930440733209252357")
	faf_test.basis_function_name = 'rationalcplx'

	if not faf.load_auto(faf_test):
		raise Exception("No valid FAF coefficients found")

	print("File: "+str(faf.filename))
	faf.print_coefficients()

	test = FAFTestCoefficients()
	test.load_fafcoeffs(faf)
	test.runTests()

	################################################################
	################################################################
	# test with new coefficients for Gaussian basis function and new 
	################################################################
	################################################################

	print("*"*80)
	print("New FAF coefficients with flexible shift/spacing")
	print("*"*80)

	faf = FAFCoefficients()
	faf.process_args(sys.argv)

	faf_test = FAFCoefficients()
	faf_test.N = 23
	faf_test.basis_function_spacing = efloat.to(1.0)
	faf_test.function_name = 'gaussianorig'
	faf_test.basis_function_name = 'rationalcplx'

	if not faf.load_auto(faf_test):
		raise Exception("No valid FAF coefficients found")

	print("File: "+str(faf.filename))
	faf.print_coefficients()

	test = FAFTestCoefficients()
	test.load_fafcoeffs(faf)
	test.runTests()

