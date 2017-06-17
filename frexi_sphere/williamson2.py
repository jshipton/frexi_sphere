from firedrake import *
from frexi_sphere.exponential_integrators import SSPRK2V
from frexi_sphere.sw_setup import SetupShallowWater
import frexi_sphere.diagnostics
import frexi_sphere.timestepping
import numpy
from os import path
import json
parameters["pyop2_options"]["lazy_evaluation"] = False
# set up mesh and initial conditions for Williamson 2 testcase
R = 6371220.
ref_level = 3
mesh = IcosahedralSphereMesh(radius=R, refinement_level=ref_level, degree=3)
global_normal = Expression(("x[0]", "x[1]", "x[2]"))
mesh.init_cell_orientations(global_normal)

# setup parameters for timestepping
tmax = 5*24.*60.*60.
dt = 1500.

degree = 1
setup = SetupShallowWater(mesh, family="BDM", degree=degree, problem_name="w2")
setup.ics()

# get spaces and initialise
V1 = setup.spaces['u']
V2 = setup.spaces['h']
u0 = Function(V1,name="u").assign(setup.u0)
h0 = Function(V2,name="h").assign(setup.h0)

# setup parameters for REXI
h = 0.2
M = 64

# make timestepper
direct = False
ncells = 20*4**ref_level
dx0 = numpy.pi*4*R**2/ncells
timestepper = SSPRK2V(setup, dt, direct, h, M, False, IPcoeff=10*ncells)

# output file and output fields
dirname = 'SSPRK2V_w2_deg%s_dt%s_h%s_M%s' % (degree, dt, h, M)
fields = [u0, h0]

timestepping = frexi_sphere.timestepping.Timestepping(dirname, fields, setup.params, timestepper)
timestepping.run(dt, tmax, steady_state=True)
