from firedrake import *
from frexi_sphere.exponential_integrators import CoarsePropagator
from frexi_sphere.sw_setup import SetupShallowWater
import frexi_sphere.diagnostics
import frexi_sphere.timestepping
from os import path
import json

# set up mesh and initial conditions for Williamson 2 testcase
R = 6371220.
mesh = IcosahedralSphereMesh(radius=R, refinement_level=3, degree=3)
global_normal = Expression(("x[0]", "x[1]", "x[2]"))
mesh.init_cell_orientations(global_normal)

# setup parameters for timestepping
tmax = 5*24.*60.*60.
dt = 3000.

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
timestepper = CoarsePropagator(setup, dt, False, h, M, True, 1./dt, 100.)

# output file and output fields
dirname = 'CP_w2_deg%s_dt%s_h%s_M%s' % (degree, dt, h, M)
fields = [u0, h0]

timestepping = frexi_sphere.timestepping.Timestepping(dirname, fields, setup.params, timestepper)
timestepping.run(dt, tmax, steady_state=True)
